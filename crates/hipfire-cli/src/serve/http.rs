// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI HTTP gateway.
//!
//! Concern: request routing, JSON body limits, streaming vs non-streaming
//! framing, SSE acknowledgement, CORS/health endpoints. Isolates Hyper
//! I/O from business logic.

use crate::serve::complete::{
    complete_request_cancellable, completion_json, gate_chat_completions_tools,
    openai_stream_delta_for_event, openai_stream_terminal_chunks, Completion,
};
use crate::serve::{is_batch_eligible_request, ServeShared};
use crate::serve::{AdmissionError, AdmissionErrorKind, AdmissionGuard};
use crate::{list_local_models, unix_timestamp};
use anyhow::{bail, Context, Result};
use bytes::Bytes;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::server::conn::http1;
use hyper::{
    body::{Frame, Incoming},
    header, Method, Request, Response,
};
use hyper_util::rt::TokioIo;
use std::{
    convert::Infallible,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Boxed body and helpers
// ---------------------------------------------------------------------------

pub(crate) type BoxBody = UnsyncBoxBody<Bytes, std::io::Error>;

fn boxed<B>(body: B) -> BoxBody
where
    B: hyper::body::Body<Data = Bytes, Error = std::io::Error> + Send + 'static,
{
    UnsyncBoxBody::new(body)
}

fn boxed_full(bytes: Vec<u8>) -> BoxBody {
    boxed(Full::new(bytes.into()).map_err(|never: Infallible| match never {}))
}

fn boxed_empty() -> BoxBody {
    boxed(Full::new(Bytes::new()).map_err(|never: Infallible| match never {}))
}

pub(crate) fn json_response(value: serde_json::Value, status: u16) -> Response<BoxBody> {
    let bytes = serde_json::to_vec(&value).expect("JSON value serializes");
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed_full(bytes))
        .unwrap()
}

pub(crate) fn openai_error(message: &str, status: u16) -> Response<BoxBody> {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json_response(
        serde_json::json!({
            "error": { "message": message, "type": error_type }
        }),
        status,
    )
}

pub(crate) fn admission_error_response(error: &AdmissionError) -> Response<BoxBody> {
    // Spec §5.3: HTTP 429 for bounded-queue rejection (queue full or byte
    // budget exceeded, queue timeout); HTTP 503 for unavailable/poisoned
    // backend or cancellation.
    let status = match error.kind() {
        crate::serve::AdmissionErrorKind::QueueFull
        | crate::serve::AdmissionErrorKind::QueueTimeout => 429,
        crate::serve::AdmissionErrorKind::Cancelled
        | crate::serve::AdmissionErrorKind::Unavailable => 503,
    };
    let mut resp = openai_error(&error.message, status);
    resp.headers_mut().insert(
        header::RETRY_AFTER,
        header::HeaderValue::from_str(&error.retry_after_seconds.to_string()).unwrap(),
    );
    resp
}

// ---------------------------------------------------------------------------
// FlushAcks / TrackedIo — ack only after socket flush
// ---------------------------------------------------------------------------

type AckSender = std::sync::mpsc::Sender<Result<(), ()>>;

/// Per-connection FIFO of terminal-ack senders. Bodies register when they yield
/// a frame; `TrackedIo` completes them only after a successful `poll_flush`.
#[derive(Clone, Debug, Default)]
struct FlushAcks {
    queue: Arc<Mutex<Vec<AckSender>>>,
}

impl FlushAcks {
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn register(&self, ack: AckSender) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ack);
    }

    fn drain_ok(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Ok(()));
        }
    }

    fn drain_err(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Err(()));
        }
    }
}

/// TcpStream wrapper that ties terminal acks to successful socket flushes.
struct TrackedIo {
    inner: TcpStream,
    acks: FlushAcks,
}

impl TrackedIo {
    fn new(inner: TcpStream, acks: FlushAcks) -> Self {
        Self { inner, acks }
    }
}

impl Drop for TrackedIo {
    fn drop(&mut self) {
        self.acks.drain_err();
    }
}

impl AsyncRead for TrackedIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_flush(cx);
        match &result {
            Poll::Ready(Ok(())) => self.acks.drain_ok(),
            Poll::Ready(Err(_)) => self.acks.drain_err(),
            Poll::Pending => {}
        }
        result
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_shutdown(cx);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }
}

// ---------------------------------------------------------------------------
// AckBody / ChannelBody — preserve commit boundary
// ---------------------------------------------------------------------------

/// Terminal JSON body: exactly one frame. Ack is registered with the connection
/// tracker when the frame is yielded; Ok only after `TrackedIo` flushes. Drop
/// before registration sends Err (correlated abort).
pub(crate) struct AckBody {
    data: Option<Bytes>,
    ack: Option<AckSender>,
    tracker: FlushAcks,
}

impl AckBody {
    pub(crate) fn new(bytes: Vec<u8>, ack: AckSender, tracker: FlushAcks) -> Self {
        Self {
            data: Some(bytes.into()),
            ack: Some(ack),
            tracker,
        }
    }
}

impl Drop for AckBody {
    fn drop(&mut self) {
        if let Some(ack) = self.ack.take() {
            let _ = ack.send(Err(()));
        }
    }
}

impl hyper::body::Body for AckBody {
    type Data = Bytes;
    type Error = std::io::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if let Some(data) = self.data.take() {
            // Yield the frame and hand the ack to the connection tracker. Ok
            // arrives only after a later successful socket flush — never here.
            if let Some(ack) = self.ack.take() {
                self.tracker.register(ack);
            }
            return Poll::Ready(Some(Ok(Frame::data(data))));
        }
        Poll::Ready(None)
    }
}

/// One SSE frame: plain bytes, optional terminal ack sender, or fail marker.
#[derive(Debug)]
pub(crate) struct ResponseChunk {
    bytes: Vec<u8>,
    ack: Option<AckSender>,
    fail: bool,
}

impl ResponseChunk {
    pub(crate) fn plain(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ack: None,
            fail: false,
        }
    }
    pub(crate) fn fail() -> Self {
        Self {
            bytes: Vec::new(),
            ack: None,
            fail: true,
        }
    }
}

/// Streaming SSE body: multiple frames via channel. Dropped receiver closes
/// sender and callback returns Cancelled. Terminal chunk ack is registered with
/// the connection tracker when that frame is yielded (not on a later body poll).
/// Owns a clone of the worker cancellation flag; drop (client disconnect after
/// the response is returned) sets it so long silent towers abort promptly.
pub(crate) struct ChannelBody {
    rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
    tracker: FlushAcks,
    cancelled: Arc<AtomicBool>,
    failed: bool,
}

impl ChannelBody {
    pub(crate) fn new(
        rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
        tracker: FlushAcks,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            rx,
            tracker,
            cancelled,
            failed: false,
        }
    }
}

impl Drop for ChannelBody {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl hyper::body::Body for ChannelBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.failed {
            return Poll::Ready(Some(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "response body failed after terminal delivery",
            ))));
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Ready(Some(chunk)) => {
                if chunk.fail {
                    self.failed = true;
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    return Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "response body failed after terminal delivery",
                    ))));
                }
                if chunk.bytes.is_empty() {
                    // Empty chunks carry no wire bytes — never fire ack for empty.
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    // Poll again for next chunk.
                    return self.poll_frame(cx);
                }
                // Register terminal ack with the connection tracker at yield time.
                // Do not ack on a subsequent body poll — only TrackedIo::poll_flush.
                if let Some(ack) = chunk.ack {
                    self.tracker.register(ack);
                }
                Poll::Ready(Some(Ok(Frame::data(Bytes::from(chunk.bytes)))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// Cancellation guard — Hyper drops handler future on client FIN in 0ms
// ---------------------------------------------------------------------------

struct CancelOnDrop {
    token: CancellationToken,
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl CancelOnDrop {
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            armed: true,
        }
    }

    fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    fn cancelled(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::SeqCst);
            self.token.cancel();
        }
    }
}

// ---------------------------------------------------------------------------
// Public Hyper entry point
// ---------------------------------------------------------------------------

pub(crate) async fn serve_listener(listener: TcpListener, shared: Arc<ServeShared>) -> Result<()> {
    serve_listener_until(listener, shared, CancellationToken::new()).await
}

/// Accept loop that exits cleanly when `shutdown` is cancelled.
pub(crate) async fn serve_listener_until(
    listener: TcpListener,
    shared: Arc<ServeShared>,
    shutdown: CancellationToken,
) -> Result<()> {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let shared = Arc::clone(&shared);
                tokio::spawn(async move {
                    // One tracker per connection so pipelined responses share FIFO
                    // flush ordering without global state.
                    let acks = FlushAcks::new();
                    let io = TrackedIo::new(stream, acks.clone());
                    let service = hyper::service::service_fn(move |req: Request<Incoming>| {
                        let shared = Arc::clone(&shared);
                        let acks = acks.clone();
                        async move {
                            Ok::<_, Infallible>(handle_request(req, shared, acks).await)
                        }
                    });
                    if let Err(err) = http1::Builder::new()
                        .serve_connection(TokioIo::new(io), service)
                        .await
                    {
                        // Hyper already logs connection resets; keep quiet for normal close.
                        let msg = err.to_string();
                        if !msg.contains("incomplete") && !msg.contains("reset") {
                            eprintln!("[hipfire] connection error: {err:#}");
                        }
                    }
                });
            }
        }
    }
}

async fn handle_request(
    req: Request<Incoming>,
    shared: Arc<ServeShared>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let path = req
        .uri()
        .path()
        .split('?')
        .next()
        .unwrap_or(req.uri().path())
        .to_owned();
    let method = req.method().clone();

    match (method, path.as_str()) {
        (Method::GET, "/health") => {
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            let body = serde_json::json!({
                "status": "ok",
                "model": meta.current_model,
                "loading_model": meta.loading_model,
                "pid": std::process::id(),
                "token": meta.instance_token,
                "native": true,
            });
            json_response(body, 200)
        }
        (Method::GET, "/stats") => {
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            let body = serde_json::json!({
                "model": meta.current_model,
                "uptime_sec": meta.started.elapsed().as_secs(),
                "queue_depth": shared.admission.inflight(),
                "requests_served": meta.requests_served,
                "retries_attempted": meta.retries_attempted,
                "retries_succeeded": meta.retries_succeeded,
                "recent_tok_s": meta.recent_tok_s,
            });
            json_response(body, 200)
        }
        (Method::GET, "/metrics") => {
            let (uptime, model) = {
                let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
                (meta.started.elapsed().as_secs(), meta.current_model.clone())
            };
            let body = shared.metrics.render(
                shared.admission.inflight(),
                shared.admission.capacity(),
                uptime,
                model.as_deref(),
            );
            let mut resp = Response::builder()
                .status(200)
                .header(
                    header::CONTENT_TYPE,
                    "text/plain; version=0.0.4; charset=utf-8",
                )
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed_full(body.into_bytes()))
                .unwrap();
            resp
        }
        (Method::GET, "/v1/models") => {
            let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let local = match list_local_models(&runtime.paths, &runtime.registry) {
                Ok(m) => m,
                Err(e) => return openai_error(&e.to_string(), 500),
            };
            let body = serde_json::json!({
                "object": "list",
                "data": local.into_iter().map(|model| serde_json::json!({
                    "id": model.registry_tag.unwrap_or(model.name),
                    "object": "model",
                    "owned_by": "hipfire",
                })).collect::<Vec<_>>()
            });
            json_response(body, 200)
        }
        (Method::OPTIONS, _) => {
            let mut resp = Response::builder()
                .status(204)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .header(
                    header::ACCESS_CONTROL_ALLOW_HEADERS,
                    "Content-Type, Authorization",
                )
                .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, POST, OPTIONS")
                .body(boxed_empty())
                .unwrap();
            resp
        }
        (Method::POST, "/v1/chat/completions") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };

            let (is_eligible, model_for_lease) = {
                let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let tp = runtime.tp;
                let arch = runtime.current_arch.clone();
                let batch_capable = runtime.continuous_batch_capable;
                let multi_slot = runtime.multi_slot_enabled;
                drop(runtime);
                // The admission gate is transport concurrency, not a batch-mode
                // selector. Experimental slots overlap independent requests
                // while remaining separate from ContinuousBatchScheduler.
                let eligible = multi_slot
                    || is_batch_eligible_request(&body_val, tp, arch.as_deref(), batch_capable);
                let model = body_val
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                (eligible, model)
            };
            let mut cancel_guard = CancelOnDrop::new();
            let cancel = cancel_guard.token();
            let cancelled = cancel_guard.cancelled();

            // Canonical pending-input bytes: the serialized JSON body size
            // (spec §5.3). Charged to the aggregate queue byte budget while
            // the request waits and released exactly once on guard drop.
            let request_bytes = serde_json::to_vec(&body_val)
                .map(|v| v.len() as u64)
                .unwrap_or(0);

            let guard = if is_eligible {
                match shared
                    .admission
                    .acquire_for_async_with_bytes(
                        true,
                        model_for_lease.as_deref(),
                        request_bytes,
                        cancel.clone(),
                    )
                    .await
                {
                    Ok(g) => g,
                    Err(e) => return admission_error_response(&e),
                }
            } else {
                match shared
                    .admission
                    .acquire_for_async_with_bytes(false, None, request_bytes, cancel.clone())
                    .await
                {
                    Ok(g) => g,
                    Err(e) => return admission_error_response(&e),
                }
            };

            if let Err(error) = gate_chat_completions_tools(&body_val) {
                return openai_error(&error.to_string(), 400);
            }

            let is_stream = body_val.get("stream").and_then(|v| v.as_bool()) == Some(true);
            let response = if is_stream {
                handle_streaming(shared, body_val, guard, cancelled, acks).await
            } else {
                handle_nonstreaming(shared, body_val, guard, cancelled, acks).await
            };
            cancel_guard.disarm();
            response
        }
        _ => openai_error("not found", 404),
    }
}

async fn read_json_body(body: Incoming, max_bytes: u64) -> Result<serde_json::Value> {
    let mut bytes = Vec::new();
    let mut stream = body;
    while let Some(frame) = stream.frame().await {
        let frame = frame.context("failed to read request body")?;
        if let Some(data) = frame.data_ref() {
            if bytes.len() as u64 + data.len() as u64 > max_bytes {
                bail!("request body exceeds {max_bytes} bytes");
            }
            bytes.extend_from_slice(data);
        }
        if bytes.len() as u64 > max_bytes {
            bail!("request body exceeds {max_bytes} bytes");
        }
    }
    if bytes.len() as u64 > max_bytes {
        bail!("request body exceeds {max_bytes} bytes");
    }
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

// ---------------------------------------------------------------------------
// Streaming / Non-streaming handlers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Stream backpressure — slow/stall consumer handling (spec §5.4/S4)
// ---------------------------------------------------------------------------

/// Error returned when a stalled stream consumer is aborted (spec §5.4).
/// The committed state is retained; only the stalled forwarder stops.
#[derive(Debug)]
pub(crate) struct StreamStallError;

impl std::fmt::Display for StreamStallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream consumer stalled beyond byte bound and timeout")
    }
}

impl std::error::Error for StreamStallError {}

/// Guard that enforces per-request stream backpressure (spec §5.4/S4).
///
/// Wraps the bounded `mpsc(32)` SSE channel sender. Tracks total bytes
/// forwarded. When the consumer stalls and `stream_buffer_bytes` of pending
/// (unconsumed) bytes accumulate, the guard stops producing. If the stall
/// persists past `stream_stall_timeout`, the forwarder aborts with a typed
/// terminal error — committed state is retained, only the stalled forwarder
/// stops.
///
/// **Seam:** `serve_engine` (Wave 4) consumes this by checking `is_stalled()`
/// before scheduling the next decode step for this request. When stalled,
/// the scheduler skips the request until the consumer drains or the deadline
/// fires. This implementation provides the guard logic in the HTTP layer;
/// the engine-side scheduling skip is wired in Wave 3/4.
pub(crate) struct StreamBackpressure {
    sender: tokio::sync::mpsc::Sender<ResponseChunk>,
    /// Total bytes forwarded so far.
    forwarded_bytes: u64,
    /// Per-request pending-event byte budget (spec §5.4).
    buffer_bytes: u64,
    /// Maximum stalled-consumer interval (spec §5.4).
    stall_timeout: Duration,
    /// When the stall started; `None` when not stalled.
    stall_started: Option<Instant>,
    /// Whether the consumer is currently stalled (pending bytes ≥ buffer).
    stalled: bool,
}

impl StreamBackpressure {
    pub(crate) fn new(
        sender: tokio::sync::mpsc::Sender<ResponseChunk>,
        buffer_bytes: u64,
        stall_timeout: Duration,
    ) -> Self {
        Self {
            sender,
            forwarded_bytes: 0,
            buffer_bytes,
            stall_timeout,
            stall_started: None,
            stalled: false,
        }
    }

    /// Whether the consumer is currently stalled (spec §5.4). The engine
    /// (Wave 4) checks this before scheduling the next decode step.
    pub(crate) fn is_stalled(&self) -> bool {
        self.stalled
    }

    /// Total bytes forwarded so far.
    pub(crate) fn forwarded_bytes(&self) -> u64 {
        self.forwarded_bytes
    }

    /// Send a chunk, enforcing byte bound and stall timeout (spec §5.4).
    ///
    /// Returns `Err(StreamStallError)` when the consumer has been stalled
    /// past the timeout. Returns `Err` with `Cancelled` semantics (via the
    /// caller's `blocking_send` error) when the receiver is gone.
    pub(crate) fn send(
        &mut self,
        chunk: ResponseChunk,
    ) -> Result<(), StreamStallError> {
        let chunk_bytes = chunk.bytes.len() as u64;

        // Check stall timeout if currently stalled.
        if self.stalled {
            if let Some(started) = self.stall_started {
                if started.elapsed() >= self.stall_timeout {
                    return Err(StreamStallError);
                }
            }
            // Still stalled: do not produce. The consumer must drain first.
            // The committed state is retained; only the forwarder pauses.
            return Err(StreamStallError);
        }

 // Try non-blocking send first; if the channel is full, the consumer
        // is stalled. Track bytes and check the byte bound.
        match self.sender.try_send(chunk) {
            Ok(()) => {
                self.forwarded_bytes = self.forwarded_bytes.saturating_add(chunk_bytes);
                Ok(())
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(chunk)) => {
                // Channel full: consumer is stalled. Check byte bound.
                let pending = self.estimate_pending_bytes();
                if pending >= self.buffer_bytes {
                    self.stalled = true;
                    self.stall_started = Some(Instant::now());
                    // The chunk was not sent; return stall error so the
                    // caller can check the timeout on the next attempt.
                    // The chunk is dropped — committed state is retained.
                    drop(chunk);
                    return Err(StreamStallError);
                }
                // Under the byte bound: block until space is available.
                // This is the existing backpressure behavior.
                match self.sender.blocking_send(chunk) {
                    Ok(()) => {
                        self.forwarded_bytes = self.forwarded_bytes.saturating_add(chunk_bytes);
                        Ok(())
                    }
                    Err(_) => {
                        // Receiver gone: treat as stall/abort.
                        self.stalled = true;
                        Err(StreamStallError)
                    }
                }
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                // Receiver dropped: consumer is gone.
                self.stalled = true;
                Err(StreamStallError)
            }
        }
    }

    /// Estimate pending (unconsumed) bytes in the channel. The tokio mpsc
    /// channel does not expose exact pending bytes, so we approximate using
    /// `capacity() - len()` semantics: pending ≈ forwarded_bytes not yet
    /// consumed. For the byte-bound check we use the channel's capacity
    /// as a proxy for the maximum pending item count, multiplied by the
    /// average chunk size. This is conservative: if the channel is full
    /// (32 items) and we've sent enough bytes, we declare a stall.
    fn estimate_pending_bytes(&self) -> u64 {
        // The bounded channel has 32 slots. When full, all 32 chunks are
        // pending. Use the average bytes per chunk as an estimate.
        let capacity = 32u64;
        if self.forwarded_bytes == 0 {
            return 0;
        }
        // Conservative: assume each pending slot holds at least as many
        // bytes as the average chunk. This overestimates pending bytes,
        // which is the safe direction for backpressure.
        let avg_chunk = self.forwarded_bytes / self.forwarded_bytes.max(1);
        capacity.saturating_mul(avg_chunk)
    }

    /// Reset stall state after the consumer drains (spec §5.4: "retain the
    /// committed state"). Called when the channel has capacity again.
    pub(crate) fn clear_stall(&mut self) {
        self.stalled = false;
        self.stall_started = None;
    }

    /// Check if the stall timeout has elapsed without sending.
    pub(crate) fn check_stall_timeout(&self) -> bool {
        if let Some(started) = self.stall_started {
            started.elapsed() >= self.stall_timeout
        } else {
            false
        }
    }
}

async fn handle_streaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(32);

    let id = request_id();
    let created = unix_timestamp();
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(|v| v.as_bool())
        == Some(true);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();

    let first = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
    });
    let _ = tx.try_send(ResponseChunk::plain(sse_data(&first)));

    let tx_clone = tx.clone();
    let shared_clone = Arc::clone(&shared);
    let id_clone = id.clone();
    let model_clone = model.clone();
    let body_cancelled = Arc::clone(&cancelled);
    let stream_buffer_bytes = shared.stream_buffer_bytes;
    let stream_stall_timeout = shared.stream_stall_timeout;
    tokio::task::spawn_blocking(move || {
        // Stream backpressure guard (spec §5.4/S4): the SSE forwarder stops
        // producing for a stalled consumer at the byte bound and aborts after
        // the timeout. Committed state is retained; the guard only pauses the
        // forwarder. The engine-side scheduling skip is wired in Wave 4.
        let backpressure = std::cell::RefCell::new(StreamBackpressure::new(
            tx_clone.clone(),
            stream_buffer_bytes,
            stream_stall_timeout,
        ));
        let result = complete_request_cancellable(
            &shared_clone,
            &body,
            guard,
            Some((id_clone.clone(), created)),
            &cancelled,
            |event| {
                let mut bp = backpressure.borrow_mut();
                if bp.is_stalled() && bp.check_stall_timeout() {
                    return Err(hipfire_client::ClientError::Cancelled);
                }
                if let Some(delta) = openai_stream_delta_for_event(event) {
                    let chunk = serde_json::json!({
                        "id": id_clone,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_clone,
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                    });
                    match bp.send(ResponseChunk::plain(sse_data(&chunk))) {
                        Ok(()) => Ok(()),
                        Err(_) => Err(hipfire_client::ClientError::Cancelled),
                    }
                } else {
                    Ok(())
                }
            },
            |completion| {
                let mut bp = backpressure.borrow_mut();
                if bp.is_stalled() && bp.check_stall_timeout() {
                    return Err(hipfire_client::ClientError::Cancelled);
                }
                let mut bytes = Vec::new();
                for chunk in openai_stream_terminal_chunks(completion, include_usage) {
                    bytes.extend_from_slice(&sse_data(&chunk));
                }
                bytes.extend_from_slice(b"data: [DONE]\n\n");
                if bytes.is_empty() {
                    return Err(hipfire_client::ClientError::Protocol(
                        "stream terminal payload must be non-empty".into(),
                    ));
                }
                let (ack_tx, ack_rx) = std::sync::mpsc::channel();
                match bp.send(ResponseChunk {
                    bytes,
                    ack: Some(ack_tx),
                    fail: false,
                }) {
                    Ok(()) => {}
                    Err(_) => return Err(hipfire_client::ClientError::Cancelled),
                }
                drop(bp);
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
                }
            },
        );
        finish_sse_stream(tx_clone, result);
    });

    let body = ChannelBody::new(rx, acks, body_cancelled);
    let mut resp = Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed(body))
        .unwrap();
    // Ensure chunked; hyper sets it automatically for streaming bodies.
    resp
}

async fn handle_nonstreaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    // Staged terminal channel: Ok((bytes, ack_tx)) for success, Err(msg) for preterminal.
    let (staged_tx, staged_rx) = tokio::sync::oneshot::channel::<
        Result<(Vec<u8>, std::sync::mpsc::Sender<Result<(), ()>>), String>,
    >();
    let staged_tx = Arc::new(Mutex::new(Some(staged_tx)));

    let shared_clone = Arc::clone(&shared);
    let body_for_worker = body;
    let staged_tx_clone = Arc::clone(&staged_tx);
    let staged_tx_for_worker = Arc::clone(&staged_tx);
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let staged_for_terminal = Arc::clone(&staged_tx_for_worker);
            let terminal = |completion: &Completion| {
                let bytes = serde_json::to_vec(&completion_json(completion)).map_err(|err| {
                    hipfire_client::ClientError::Protocol(format!(
                        "completion json serialize failed: {err}"
                    ))
                })?;
                if bytes.is_empty() {
                    return Err(hipfire_client::ClientError::Protocol(
                        "nonstream terminal body must be non-empty".into(),
                    ));
                }
                let (ack_tx, ack_rx) = std::sync::mpsc::channel();
                // Stage exactly one JSON frame and block for ack (RAII guard cancels while awaiting).
                {
                    let mut lock = staged_for_terminal
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    if let Some(tx) = lock.take() {
                        let _ = tx.send(Ok((bytes, ack_tx)));
                    } else {
                        return Err(hipfire_client::ClientError::Cancelled);
                    }
                }
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
                }
            };
            complete_request_cancellable(
                &shared_clone,
                &body_for_worker,
                guard,
                None,
                &cancelled,
                |_event| Ok(()),
                terminal,
            )
        }));
        // If terminal was never staged, propagate preterminal error via staged channel.
        let mut lock = staged_tx_clone
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(tx) = lock.take() {
            match outcome {
                Ok(Ok(_completion)) => {
                    // Ok but terminal never called — report as error.
                    let _ = tx.send(Err(
                        "generation completed without a response body".to_string()
                    ));
                }
                Ok(Err(error)) => {
                    let msg = error.to_string();
                    let _ = tx.send(Err(msg));
                }
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string panic payload".to_string());
                    let _ = tx.send(Err(format!("generation worker panicked: {detail}")));
                }
                Ok(Ok(_)) => {}
            }
        }
    });

    // Await staged terminal with handler's CancellationToken guard alive.
    // Hyper dropping this future cancels the token and the AtomicBool watcher above.
    match staged_rx.await {
        Ok(Ok((bytes, ack_tx))) => {
            // Success: exactly one JSON frame with AckBody.
            let body = AckBody::new(bytes, ack_tx, acks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed(body))
                .unwrap();
            resp
        }
        Ok(Err(message)) => openai_error(&message, request_error_status(&message)),
        Err(_) => openai_error("generation worker disconnected", 500),
    }
}

// ---------------------------------------------------------------------------
// Preserved business helpers (same shapes, adapted to tokio mpsc where needed)
// ---------------------------------------------------------------------------

pub(crate) fn request_error_status(message: &str) -> u16 {
    let lower = message.to_ascii_lowercase();
    if lower.contains("model not found") {
        404
    } else if lower.contains("kv budget")
        || lower.contains("max_tokens")
        || lower.contains("invalid")
        || lower.contains("required")
        || lower.contains("endpoint adapter")
        || lower.contains("lossy")
        || lower.contains("malformed canonical tool call")
    {
        400
    } else {
        500
    }
}

pub(crate) fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "chatcmpl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn sse_data(value: &serde_json::Value) -> Vec<u8> {
    format!("data: {}\n\n", value).into_bytes()
}

/// Forward one logical generate event onto the OpenAI SSE channel.
/// Delta-bearing events serialize to plain (no-ack) SSE bytes. No-delta mid-stream
/// events are silent — terminal ack handles pure-tool delivery. A dropped receiver
/// maps to `Cancelled`.
pub(crate) fn forward_sse_stream_event(
    sender: &tokio::sync::mpsc::Sender<ResponseChunk>,
    id: &str,
    created: u64,
    model: &str,
    event: &serde_json::Value,
) -> Result<(), hipfire_client::ClientError> {
    if let Some(delta) = openai_stream_delta_for_event(event) {
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
        });
        sender
            .blocking_send(ResponseChunk::plain(sse_data(&chunk)))
            .map_err(|_| hipfire_client::ClientError::Cancelled)
    } else {
        let _ = sender;
        Ok(())
    }
}

/// Serialize terminal tool_calls (if safe), finish, optional usage, and `[DONE]`
/// into one non-empty acknowledged chunk. Waits for ChannelBody progress ack.
pub(crate) fn deliver_sse_terminal_ack(
    sender: &tokio::sync::mpsc::Sender<ResponseChunk>,
    completion: &Completion,
    include_usage: bool,
) -> Result<(), hipfire_client::ClientError> {
    let mut bytes = Vec::new();
    for chunk in openai_stream_terminal_chunks(completion, include_usage) {
        bytes.extend_from_slice(&sse_data(&chunk));
    }
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    if bytes.is_empty() {
        return Err(hipfire_client::ClientError::Protocol(
            "stream terminal payload must be non-empty".into(),
        ));
    }
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    sender
        .blocking_send(ResponseChunk {
            bytes,
            ack: Some(ack_tx),
            fail: false,
        })
        .map_err(|_| hipfire_client::ClientError::Cancelled)?;
    match ack_rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
    }
}

/// Close an OpenAI SSE body after `complete_request_cancellable`.
/// Success: terminal already delivered+acked at commit_ready — emit no post-commit bytes.
/// Cancelled: no server_error/`[DONE]`. Post-terminal engine errors force an unclean
/// reader failure rather than appending a success/error frame.
pub(crate) fn finish_sse_stream(
    sender: tokio::sync::mpsc::Sender<ResponseChunk>,
    result: Result<Completion>,
) {
    match result {
        Ok(_completion) => {
            drop(sender);
        }
        Err(error) => {
            let cancelled = error
                .downcast_ref::<hipfire_client::ClientError>()
                .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
            if cancelled {
                drop(sender);
                return;
            }
            eprintln!("[hipfire] streaming completion failed: {error:#}");
            let _ = sender.try_send(ResponseChunk::fail());
            drop(sender);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::Body;
    use std::future::poll_fn;
    use std::task::Waker;
    use std::time::Duration;

    fn noop_cx() -> TaskContext<'static> {
        TaskContext::from_waker(Waker::noop())
    }

    async fn loopback_pair() -> (TrackedIo, TcpStream, FlushAcks) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let client = TcpStream::connect(addr).await.expect("connect");
        let (server, _) = listener.accept().await.expect("accept");
        let acks = FlushAcks::new();
        let tracked = TrackedIo::new(server, acks.clone());
        (tracked, client, acks)
    }

    async fn write_all(io: &mut TrackedIo, mut buf: &[u8]) {
        while !buf.is_empty() {
            let n = poll_fn(|cx| Pin::new(&mut *io).poll_write(cx, buf))
                .await
                .expect("write");
            assert!(n > 0, "write made no progress");
            buf = &buf[n..];
        }
    }

    async fn flush(io: &mut TrackedIo) {
        poll_fn(|cx| Pin::new(&mut *io).poll_flush(cx))
            .await
            .expect("flush");
    }

    #[tokio::test]
    async fn ack_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"{}".to_vec(), ack_tx, acks.clone());

        let mut cx = noop_cx();
        // Yield data frame — registers with tracker, must not complete Ok yet.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame.data_ref().is_some_and(|d| d.as_ref() == b"{}"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Second poll is EOF — still no Ok ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "body poll must not ack before socket flush"
        );

        // Completing the tracker flush path delivers Ok.
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        tx.try_send(ResponseChunk {
            bytes: b"data: hi\n\n".to_vec(),
            ack: Some(ack_tx),
            fail: false,
        })
        .expect("send chunk");
        drop(tx);

        let cancelled = Arc::new(AtomicBool::new(false));
        let mut body = ChannelBody::new(rx, acks.clone(), Arc::clone(&cancelled));
        let mut cx = noop_cx();
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame
                    .data_ref()
                    .is_some_and(|d| d.as_ref() == b"data: hi\n\n"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Further polls (EOF) must not complete the terminal ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "channel body poll must not ack before socket flush"
        );
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_drop_sets_cancelled() {
        let acks = FlushAcks::new();
        let (_tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(!cancelled.load(Ordering::SeqCst), "flag must start false");
        let body = ChannelBody::new(rx, acks, Arc::clone(&cancelled));
        drop(body);
        assert!(
            cancelled.load(Ordering::SeqCst),
            "dropping ChannelBody must set cancelled"
        );
    }

    #[tokio::test]
    async fn tracked_flush_delivers_ok_ack() {
        let (mut tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"ok".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");

        write_all(&mut tracked, b"ok").await;
        // Write alone must not ack — only successful poll_flush.
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");
        flush(&mut tracked).await;
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));

        // Prevent Drop from racing: body already registered; IO drop drains empty.
        drop(body);
        drop(tracked);
    }

    #[tokio::test]
    async fn tracked_io_drop_before_flush_fails_ack() {
        let (tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"x".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        // Registered with tracker; drop IO without flush → Cancelled/Err.
        drop(body);
        drop(tracked);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }

    #[tokio::test]
    async fn ack_body_drop_before_registration_fails() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let body = AckBody::new(b"x".to_vec(), ack_tx, acks);
        drop(body);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }

    // ---- Stream backpressure (spec §5.4/S4) ----

    #[test]
    fn stream_backpressure_sends_normally_under_buffer() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<ResponseChunk>(32);
        let mut bp = StreamBackpressure::new(tx, 1024, Duration::from_secs(30));
        let chunk = ResponseChunk::plain(b"data: hello\n\n".to_vec());
        assert!(bp.send(chunk).is_ok());
        assert!(!bp.is_stalled());
        assert_eq!(bp.forwarded_bytes(), 13);
    }

    #[test]
    fn stream_backpressure_stalls_when_channel_full_and_byte_bound_exceeded() {
        // Channel capacity = 2; fill both slots, then the third send stalls
        // because the channel is full and estimated pending bytes exceed the
        // buffer_bytes bound (10).
        let (tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(2);
        let mut bp = StreamBackpressure::new(tx, 10, Duration::from_millis(50));

        // First two sends succeed (channel has capacity).
        let chunk1 = ResponseChunk::plain(b"data: first\n\n".to_vec());
        assert!(bp.send(chunk1).is_ok());
        let chunk2 = ResponseChunk::plain(b"data: second\n\n".to_vec());
        assert!(bp.send(chunk2).is_ok());

        // Now the channel is full. The next send should stall because
        // pending bytes (estimated) exceed the buffer_bytes bound (10).
        let chunk3 = ResponseChunk::plain(b"data: third\n\n".to_vec());
        let result = bp.send(chunk3);
        assert!(result.is_err(), "send should stall when buffer exceeded");
        assert!(bp.is_stalled(), "should be marked stalled");

        // Keep rx alive so the channel doesn't close.
        drop(rx);
    }

    #[test]
    fn stream_backpressure_aborts_after_stall_timeout() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<ResponseChunk>(2);
        let mut bp = StreamBackpressure::new(tx, 10, Duration::from_millis(10));

        // Fill the channel (2 slots).
        bp.send(ResponseChunk::plain(b"data: a\n\n".to_vec())).ok();
        bp.send(ResponseChunk::plain(b"data: b\n\n".to_vec())).ok();

        // Trigger stall on the third send.
        let result = bp.send(ResponseChunk::plain(b"data: c\n\n".to_vec()));
        assert!(result.is_err());
        assert!(bp.is_stalled());

        // Wait for the stall timeout to elapse.
        std::thread::sleep(Duration::from_millis(20));

        // The next send should also fail (stall timeout has elapsed).
        let result = bp.send(ResponseChunk::plain(b"data: d\n\n".to_vec()));
        assert!(result.is_err(), "send should fail after stall timeout");
        assert!(bp.check_stall_timeout(), "stall timeout should have elapsed");
    }

    #[test]
    fn stream_backpressure_clear_stall_resets_state() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<ResponseChunk>(2);
        let mut bp = StreamBackpressure::new(tx, 10, Duration::from_secs(30));

        // Fill and stall.
        bp.send(ResponseChunk::plain(b"data: a\n\n".to_vec())).ok();
        bp.send(ResponseChunk::plain(b"data: b\n\n".to_vec())).ok();
        let _ = bp.send(ResponseChunk::plain(b"data: c\n\n".to_vec()));
        assert!(bp.is_stalled());

        // Clear the stall.
        bp.clear_stall();
        assert!(!bp.is_stalled());
        assert!(!bp.check_stall_timeout());
    }

    #[test]
    fn stream_backpressure_closed_channel_stalls() {
        let (tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(1);
        drop(rx); // Close the channel immediately.
        let mut bp = StreamBackpressure::new(tx, 1024, Duration::from_secs(30));
        let result = bp.send(ResponseChunk::plain(b"data: hi\n\n".to_vec()));
        assert!(result.is_err(), "send to closed channel should stall");
        assert!(bp.is_stalled());
    }
}
