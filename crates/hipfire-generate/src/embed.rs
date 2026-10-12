// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Embeddings — the `embed` wire body (arch 17, the qwen3.5 multi-vector
//! encoder). The non-chat-verb sibling of [`crate::img`]: the daemon
//! dispatches (id parse, no-model and non-embed refusals) and this module
//! owns everything that names `hipfire_arch_qwen35_embed`.
//!
//! Wire request (JSONL over the daemon stdin):
//!
//! ```json
//! {"id":"…", "kind":"query"|"document",
//!  "input":"text",                       // or
//!  "input":{"text":"…", "image_base64":"…"}}
//! ```
//!
//! Wire reply — exactly one of:
//!
//! ```json
//! {"type":"embedded","id":"…","kind":"query","n_tokens":K,"dim":128,
//!  "n_input_tokens":N,"embeddings":[…K×128 floats…]}
//! {"type":"error","id":"…","message":"…"}
//! ```
//!
//! `embeddings` is the post-Dense, post-skiplist, L2-normalized token
//! matrix, row-major `[K, dim]` (late interaction: score query vs document
//! with Σ max dot — `hipfire_arch_qwen35_embed::maxsim`).

use hipfire_arch_qwen35_embed::{
    assemble_input, encode_embeddings, forward::EncodeInput, maxsim, InputKind,
};
use hipfire_engine::emit::emit_uncorrelated_error;
use hipfire_loader::LoadedModel;
use std::io::Write;

fn parse_kind(req: &serde_json::Value) -> Result<InputKind, String> {
    match req
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("document")
    {
        "query" => Ok(InputKind::Query),
        "document" | "doc" => Ok(InputKind::Document),
        other => Err(format!(
            "embed: kind {other:?} is not one of \"query\" | \"document\" \
             (queries and documents use different markers and truncation)"
        )),
    }
}

/// Extract (text, image bytes) from `input`: a bare string, or an object
/// with `text` and/or `image_base64` (a `data:…;base64,` prefix is
/// tolerated, same as the vision chat path).
fn parse_input(req: &serde_json::Value) -> Result<(String, Option<Vec<u8>>), String> {
    use base64::Engine;
    let input = req
        .get("input")
        .ok_or("embed: request has no \"input\" (string, or {text, image_base64})")?;
    if let Some(text) = input.as_str() {
        return Ok((text.to_string(), None));
    }
    let obj = input
        .as_object()
        .ok_or("embed: \"input\" must be a string or an object with text/image_base64")?;
    let text = obj
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let image = match obj.get("image_base64").and_then(|v| v.as_str()) {
        None => None,
        Some(b64) => {
            let b64 = b64.strip_prefix("base64,").unwrap_or(b64);
            let b64 = b64
                .split_once(",")
                .filter(|(head, _)| head.starts_with("data:") && head.ends_with(";base64"))
                .map(|(_, tail)| tail)
                .unwrap_or(b64);
            Some(
                base64::engine::general_purpose::STANDARD
                    .decode(b64.trim())
                    .map_err(|e| format!("embed: image_base64 is not valid base64: {e}"))?,
            )
        }
    };
    if text.is_empty() && image.is_none() {
        return Err("embed: input has neither text nor image_base64".into());
    }
    Ok((text, image))
}

/// Serve one `embed` request. Emits exactly one `embedded` or one error
/// envelope; never panics across GPU calls (every arch-crate error maps to
/// an `error` event).
pub fn generate_embed(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl Write,
    id: &str,
    req: &serde_json::Value,
) {
    fn fail(stdout: &mut impl Write, id: &str, msg: String) {
        emit_uncorrelated_error(stdout, Some(id), &msg, "validation", false, false);
        let _ = stdout.flush();
    }

    let kind = match parse_kind(req) {
        Ok(k) => k,
        Err(e) => return fail(stdout, id, e),
    };
    let (text, image) = match parse_input(req) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };
    let image_bytes = image.as_deref();

    // Typed handle: the daemon already refused non-embed arches.
    let Some(bundle) = m.qwen35_embed_mut() else {
        return fail(
            stdout,
            id,
            "embed: loaded model is not a qwen3.5 embedding checkpoint (arch 17)".into(),
        );
    };

    // Phase timing under the same knob the encoder's own trace uses. For a
    // long document the GPU kernel sum is a small fraction of wall-clock, and
    // this is what shows where the rest goes (gather, JSON of a wide output,
    // the download).
    // Its own knob, NOT HIPFIRE_EMBED_DEBUG: that one also arms the encoder's
    // per-layer finite `trace()`, which syncs the GPU every layer and inflates
    // every number here (measured 678 ms vs 334 ms on the same request).
    let timed = std::env::var("HIPFIRE_EMBED_PHASE").ok().as_deref() == Some("1");
    let t_phase = std::time::Instant::now();
    let assembled = match assemble_input(bundle, gpu, kind, &text, image_bytes) {
        Ok(a) => a,
        Err(e) => return fail(stdout, id, e),
    };
    let t_assemble = t_phase.elapsed();
    let n_input = assembled.ids.len();
    let input = EncodeInput {
        ids: &assembled.ids,
        image_embeds: &assembled.image_embeds,
        spans: &assembled.spans,
        pos3: &assembled.pos3,
    };
    let t_encode = std::time::Instant::now();
    let raw = match encode_embeddings(
        gpu,
        &bundle.weights,
        &mut bundle.state,
        &bundle.config,
        &input,
    ) {
        Ok(v) => v,
        Err(e) => return fail(stdout, id, e),
    };
    let d_encode = t_encode.elapsed();
    // MultiVectorMask + Normalize — via the shared `finalize_embeddings` so
    // the sequential and slot routes cannot drift (the helper is exactly
    // this mask/protect/normalize sequence; duplicating it here would make
    // a future skiplist change silently diverge the two paths).
    let dim = bundle.config.dense_out_features;
    let (embeddings, n_tokens) = match bundle.finalize_embeddings(kind, &assembled.ids, &raw) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };
    let d_post = t_encode.elapsed() - d_encode;

    let payload = serde_json::json!({
        "type": "embedded",
        "id": id,
        // See `generate_embed_slot`: the reply is correlated by `(id, 0)`, the
        // key a synchronous verb's caller registers.
        "attempt_id": 0,
        "kind": if kind == InputKind::Query { "query" } else { "document" },
        "n_tokens": n_tokens,
        "n_input_tokens": n_input,
        "dim": dim,
        "embeddings": embeddings,
    });
    let t_write = std::time::Instant::now();
    let _ = writeln!(stdout, "{payload}");
    let _ = stdout.flush();
    if timed {
        eprintln!(
            "  [embed-phase] n={n_input} kept={n_tokens}: assemble {:?} | encode {:?} | \
             mask+normalize {:?} | serialize+write {:?}",
            t_assemble,
            d_encode,
            d_post,
            t_write.elapsed()
        );
    }
}

/// Late-interaction score between two already-encoded inputs (the CLI's
/// `--score` path; the values are the `embeddings` arrays of an `embedded`
/// event each).
pub fn score_maxsim(
    query: &[f32],
    q_tokens: usize,
    doc: &[f32],
    d_tokens: usize,
    dim: usize,
) -> f32 {
    maxsim(query, q_tokens, doc, d_tokens, dim)
}

/// Serve one `embed` request through a multi-slot engine instead of the
/// ordinary resident model (slot mode owns the encoder and leaves the daemon's
/// `model` slot empty, so the sequential body cannot run).
///
/// Same wire body in, same `embedded` envelope out: the request is parsed
/// here, the engine tokenizes and encodes on its own thread, and the reply
/// carries the fields the sequential route writes. A per-request failure — a
/// bad kind, an undecodable image, an input the encoder refuses — becomes the
/// same `error` envelope [`generate_embed`] emits.
pub fn generate_embed_slot(
    engine: &dyn hipfire_runtime::serve::SlotEngineHandle,
    stdout: &mut impl Write,
    id: &str,
    req: &serde_json::Value,
) {
    use hipfire_runtime::serve::{EmbedKind, EmbedRequest};

    fn fail(stdout: &mut impl Write, id: &str, msg: String) {
        emit_uncorrelated_error(stdout, Some(id), &msg, "validation", false, false);
        let _ = stdout.flush();
    }

    let kind = match parse_kind(req) {
        Ok(k) => k,
        Err(e) => return fail(stdout, id, e),
    };
    let (text, image) = match parse_input(req) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };
    let wire_kind = match kind {
        InputKind::Query => EmbedKind::Query,
        InputKind::Document => EmbedKind::Document,
    };
    let mut out = match engine.encode(vec![EmbedRequest {
        kind: wire_kind,
        text,
        image,
        request_tag: id.to_string(),
    }]) {
        Ok(out) => out,
        Err(e) => return fail(stdout, id, e),
    };
    let response = match out.pop() {
        Some(Ok(r)) => r,
        Some(Err(e)) => return fail(stdout, id, e),
        None => {
            return fail(
                stdout,
                id,
                "embed: engine returned no response for the request".into(),
            )
        }
    };
    let payload = serde_json::json!({
        "type": "embedded",
        "id": id,
        // Correlates the reply with the transaction that sent the request:
        // the client registers `(id, 0)` for a synchronous verb, and an
        // attempt-less reply would be routed to the shared control slot
        // instead — which holds one transaction, so concurrent embeds would
        // drop each other's replies.
        "attempt_id": 0,
        "kind": if response.kind == EmbedKind::Query {
            "query"
        } else {
            "document"
        },
        "n_tokens": response.n_tokens,
        "n_input_tokens": response.n_input_tokens,
        "dim": response.dim,
        "embeddings": response.embeddings,
    });
    let _ = writeln!(stdout, "{payload}");
    let _ = stdout.flush();
}
