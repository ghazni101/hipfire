// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next expert-parallel (EP) row all-gather transport.
//!
//! Every rank runs the unchanged single-device program. Routed experts are
//! sharded `e -> rank e % ranks`; the down projection of a decode token (or a
//! few-row verify forward) writes `slots` expanded rows, `rows * k` in slot
//! order (rows of non-owned experts come from a zero dummy expert and are
//! garbage-by-construction). `slots` is passed per call (`1..=` the per-buffer
//! capacity of [`EpRowGatherSpec::slots`]). This transport then fills every
//! non-owned row from its owner before the unchanged canonical combine:
//!
//! 1. [`EpRowGather::publish`] records the rank's producer-complete event for a
//!    buffer index on the stream the producer kernels ran on.
//! 2. [`EpRowGather::gather`] host-waits every peer's event for that buffer
//!    index (device waits require explicit builder opt-in), then launches ONE copy-only kernel that, for
//!    slot `s`, copies row `s` of the owner's peer-mapped buffer into the
//!    rank's own buffer when the owner is not this rank. The owner comes from
//!    an [`EpSlotOwners`]: `ExpertModulo` (`slot_experts[s] % ranks`, kernel
//!    `ep_gather_rows_by_owner`, the static placement) or `Explicit` (an i32
//!    owner per slot written by the same call's owner assignment, kernel
//!    `ep_gather_rows_by_slot_owner`, balanced ownership). Owned rows and
//!    unowned (negative) slots are never written. The copy is uint4
//!    load/store, bit-exact for every pattern (±0, NaN payloads, denormals).
//! 3. [`EpRowGather::exchange`] = publish → host barrier over all ranks → gather,
//!    one host thread per rank.
//!
//! Every rank registers `K` buffers (2..=4, the same `K` on all ranks), selected
//! by a buffer index. Hazard argument: the caller never uses the same buffer
//! index on two consecutive invocations, and all ranks execute the same
//! invocation sequence. Rank `p` overwrites `buffer[b]` only after its gather of
//! the previous invocation, which waited on every peer's producer event of that
//! invocation, which in each peer's stream follows that peer's gather of
//! invocation−2 (the last possible use of `b`) ⇒ no write-after-read. The host
//! barrier guarantees an event is recorded before any peer waits on it, and no
//! rank re-records an event of the same buffer index before every peer has issued
//! its wait. (Two buffers alternating, or a third dedicated one such as an MTP
//! head, both satisfy the rule.)
//!
//! Eager only: events are created, recorded and waited on the host timeline; this
//! transport is not usable inside a captured/retained-replay body.
//!
//! # Band exchange (output-axis split, #862)
//!
//! [`EpRowGather::exchange_bands`] serves an output-axis split instead: every
//! rank writes its column band of one or more `[rows, m]` outputs into its own
//! buffer (band-compact, `rows * band_bytes` at an offset per output), and the
//! gather copies EVERY rank's band (its own included) into columns
//! `[q * band_bytes, (q + 1) * band_bytes)` of the full output with one pitched
//! copy per rank. Same publish → barrier → wait protocol and the same buffer
//! alternation rule. Each call also carries a geometry tag that must be equal on
//! all ranks, checked after the barrier, so ranks that disagree on the layout
//! fail instead of copying mismatched bands.
//!
//! # Lifetime
//!
//! `hip_bridge::Event` has no `Drop`. The producer events are created on each
//! rank's device in [`EpRowGatherBuilder::register`] and are freed by the
//! explicit [`EpRowGather::release`] (one call per rank, on that rank's thread,
//! after every peer is done waiting). Dropping an `EpRowGather` with live events
//! only logs a warning: freeing needs a `HipRuntime` + the right device, which a
//! `Drop` impl does not own.

use hip_bridge::{
    Event, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM,
};
use rdna_compute::{Gpu, GpuTensor};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Default host-barrier timeout.
pub const DEFAULT_BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

/// Largest supported rank count (the kernel takes eight source pointers).
pub const MAX_RANKS: usize = 8;

/// Supported buffers per rank: `MIN_BUFFERS..=MAX_BUFFERS`.
pub const MIN_BUFFERS: usize = 2;
pub const MAX_BUFFERS: usize = 4;

/// Where one gather call takes the owner rank of each slot from. Both are
/// device tensors of `slots` i32 values in slot order, valid on the calling
/// rank's stream before the gather runs.
#[derive(Clone, Copy)]
pub enum EpSlotOwners<'a> {
    /// Static placement: slot `s` is owned by rank `slot_experts[s] % ranks`
    /// (an id outside `[0, n_experts)` owns nothing). The historical kernel
    /// `ep_gather_rows_by_owner`; the second field is `n_experts`.
    ExpertModulo(&'a GpuTensor, usize),
    /// Per-call assigned owners (balanced ownership): slot `s` is owned by rank
    /// `owners[s]` (a negative value owns nothing), written on this rank's
    /// stream by the call's owner assignment. Kernel
    /// `ep_gather_rows_by_slot_owner`.
    Explicit(&'a GpuTensor),
}

/// Shape of one gather: `ranks` participants, a per-buffer capacity of `slots`
/// rows of `row_bytes` each (a call gathers `1..=slots` of them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpRowGatherSpec {
    pub ranks: usize,
    pub slots: usize,
    pub row_bytes: usize,
}

impl EpRowGatherSpec {
    /// Bytes one buffer must hold.
    pub fn buffer_bytes(&self) -> usize {
        self.slots * self.row_bytes
    }

    /// Reject specs the kernel cannot serve. Called at build time, never on the
    /// hot path.
    pub fn validate(&self) -> HipResult<()> {
        if !(2..=MAX_RANKS).contains(&self.ranks) {
            return Err(err(format!(
                "ep_row_gather: ranks must be 2..={MAX_RANKS}, got {}",
                self.ranks
            )));
        }
        if self.slots == 0 {
            return Err(err("ep_row_gather: slots must be >= 1".into()));
        }
        if self.row_bytes == 0 || self.row_bytes % 16 != 0 {
            return Err(err(format!(
                "ep_row_gather: row_bytes must be a non-zero multiple of 16, got {}",
                self.row_bytes
            )));
        }
        let total = self
            .slots
            .checked_mul(self.row_bytes)
            .ok_or_else(|| err("ep_row_gather: slots * row_bytes overflows".into()))?;
        if i32::try_from(self.slots).is_err() || i32::try_from(self.row_bytes / 16).is_err() {
            return Err(err(format!(
                "ep_row_gather: slots/row_bytes exceed i32 kernel arguments ({total} B)"
            )));
        }
        Ok(())
    }

    /// Buffer `index` of `len` bytes must hold every row.
    fn check_buffer_len(&self, index: usize, len: usize) -> HipResult<()> {
        if len < self.buffer_bytes() {
            return Err(err(format!(
                "ep_row_gather: buffer {index} holds {len} B, needs {} B ({} slots x {} B)",
                self.buffer_bytes(),
                self.slots,
                self.row_bytes
            )));
        }
        Ok(())
    }

    /// A call's slot count must be `1..=self.slots` (the per-buffer capacity).
    fn check_call_slots(&self, slots: usize) -> HipResult<()> {
        if slots == 0 || slots > self.slots {
            return Err(err(format!(
                "ep_row_gather: call slots {slots} outside 1..={} (buffer capacity)",
                self.slots
            )));
        }
        Ok(())
    }
}

fn err(msg: String) -> HipError {
    HipError::new(0, &msg)
}

// ───────────────────────────── host barrier ─────────────────────────────

/// Generation-counting spin(+yield) barrier with an abort flag and a timeout.
pub(crate) struct SpinBarrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    aborted: AtomicBool,
    timeout: Duration,
}

impl SpinBarrier {
    pub(crate) fn new(parties: usize, timeout: Duration) -> Self {
        Self {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            aborted: AtomicBool::new(false),
            timeout,
        }
    }

    pub(crate) fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    /// Block until all `parties` arrive. Err when aborted (before or while
    /// waiting) or on timeout; a timeout also aborts the barrier so every other
    /// waiter fails promptly instead of timing out on its own.
    pub(crate) fn wait(&self) -> Result<(), &'static str> {
        if self.is_aborted() {
            return Err("ep_row_gather barrier aborted");
        }
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return Ok(());
        }
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if self.generation.load(Ordering::Acquire) != generation {
                return Ok(());
            }
            if self.is_aborted() {
                return Err("ep_row_gather barrier aborted");
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            if start.elapsed() >= self.timeout {
                self.abort();
                return Err("ep_row_gather barrier timed out");
            }
            std::thread::yield_now();
        }
    }
}

// ───────────────────────────── builder ─────────────────────────────

struct RankReg {
    device_id: i32,
    ptrs: Vec<usize>,
    sizes: Vec<usize>,
    events: Vec<Event>,
}

/// Collects one registration per rank, then builds the shared transport.
pub struct EpRowGatherBuilder {
    spec: EpRowGatherSpec,
    regs: Vec<Option<RankReg>>,
    barrier_timeout: Duration,
    device_waits: bool,
}

impl EpRowGatherBuilder {
    /// Opt into stream waits when a launch stream exists. The default is
    /// host waits, independent of streams installed by unrelated features.
    pub fn device_waits(mut self, enabled: bool) -> Self {
        self.device_waits = enabled;
        self
    }

    /// Override the host-barrier timeout (default [`DEFAULT_BARRIER_TIMEOUT`]).
    pub fn barrier_timeout(mut self, timeout: Duration) -> Self {
        self.barrier_timeout = timeout;
        self
    }

    /// One call per rank, on any thread, with that rank's `Gpu`. `buffers[b]` is
    /// the rank's expanded-row buffer for buffer index `b` (>= `slots *
    /// row_bytes` bytes; `buffers.len()` = K in 2..=4, identical on every rank).
    /// Creates the rank's producer events (disable-timing, release to system
    /// scope so peer reads observe the producer's writes), one per buffer, on its
    /// device.
    pub fn register(
        &mut self,
        rank: usize,
        gpu: &mut Gpu,
        buffers: &[&GpuTensor],
    ) -> HipResult<()> {
        self.spec.validate()?;
        if rank >= self.spec.ranks {
            return Err(err(format!(
                "ep_row_gather: register rank {rank} out of range (ranks={})",
                self.spec.ranks
            )));
        }
        if self.regs[rank].is_some() {
            return Err(err(format!("ep_row_gather: rank {rank} registered twice")));
        }
        let k = buffers.len();
        if !(MIN_BUFFERS..=MAX_BUFFERS).contains(&k) {
            return Err(err(format!(
                "ep_row_gather: rank {rank} registered {k} buffers, need {MIN_BUFFERS}..={MAX_BUFFERS}"
            )));
        }
        if let Some(expected) = self.regs.iter().flatten().map(|r| r.ptrs.len()).next() {
            if expected != k {
                return Err(err(format!(
                    "ep_row_gather: rank {rank} registered {k} buffers but earlier ranks registered {expected}"
                )));
            }
        }
        let mut ptrs = Vec::with_capacity(k);
        let mut sizes = Vec::with_capacity(k);
        for (index, buf) in buffers.iter().enumerate() {
            self.spec.check_buffer_len(index, buf.buf.size())?;
            let ptr = buf.buf.as_ptr() as usize;
            if ptr == 0 || ptrs.contains(&ptr) {
                return Err(err(
                    "ep_row_gather: buffers must be distinct non-null allocations".into(),
                ));
            }
            ptrs.push(ptr);
            sizes.push(buf.buf.size());
        }
        gpu.bind_thread()?;
        let flags = HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM;
        let mut events = Vec::with_capacity(k);
        for _ in 0..k {
            match gpu.hip.event_create_with_flags(flags) {
                Ok(e) => events.push(e),
                Err(e) => {
                    for made in events {
                        let _ = gpu.hip.event_destroy(made);
                    }
                    return Err(e);
                }
            }
        }
        self.regs[rank] = Some(RankReg {
            device_id: gpu.device_id,
            ptrs,
            sizes,
            events,
        });
        Ok(())
    }

    /// Finish: every rank registered, distinct devices.
    pub fn build(self) -> HipResult<Arc<EpRowGather>> {
        self.spec.validate()?;
        let mut ranks = Vec::with_capacity(self.spec.ranks);
        let mut missing = None;
        for (rank, reg) in self.regs.into_iter().enumerate() {
            match reg {
                Some(r) => ranks.push(r),
                None => missing = missing.or(Some(rank)),
            }
        }
        if let Some(rank) = missing {
            return Err(err(format!(
                "ep_row_gather: rank {rank} never registered (builder dropped; registered events leak)"
            )));
        }
        let k = ranks[0].ptrs.len();
        if ranks.iter().any(|r| r.ptrs.len() != k) {
            return Err(err(
                "ep_row_gather: ranks registered different buffer counts".into(),
            ));
        }
        // Logical emulation (`HIPFIRE_EMULATE_GPUS`) deliberately maps several
        // ranks onto one physical device; physical mode needs distinct devices.
        let emulated = crate::config::get().emulate_gpus.is_some();
        for (i, a) in ranks.iter().enumerate() {
            if !emulated && ranks[..i].iter().any(|b| b.device_id == a.device_id) {
                return Err(err(format!(
                    "ep_row_gather: two ranks share device {}",
                    a.device_id
                )));
            }
        }
        let states = ranks
            .into_iter()
            .map(|r| RankState {
                device_id: r.device_id,
                tags: r.ptrs.iter().map(|_| AtomicU64::new(0)).collect(),
                ptrs: r.ptrs,
                sizes: r.sizes,
                events: r.events.into_iter().map(|e| Mutex::new(Some(e))).collect(),
                peers_enabled: AtomicBool::new(false),
            })
            .collect();
        Ok(Arc::new(EpRowGather {
            spec: self.spec,
            buffers: k,
            barrier: SpinBarrier::new(self.spec.ranks, self.barrier_timeout),
            ranks: states,
            device_waits: self.device_waits,
        }))
    }
}

// ───────────────────────────── transport ─────────────────────────────

struct RankState {
    device_id: i32,
    /// Device addresses of this rank's buffers (valid on every peer after
    /// `enable_peers`; ROCm uses one unified address space).
    ptrs: Vec<usize>,
    /// Byte sizes of the same buffers (the band gather's copy-source bounds).
    sizes: Vec<usize>,
    /// Band-exchange geometry tag per buffer index. A rank rewrites the tag of
    /// buffer `b` only at the next invocation on `b`, after the barrier every
    /// peer passes only once it has checked this one.
    tags: Vec<AtomicU64>,
    /// Producer-complete events, one per buffer index. `None` after `release`.
    /// The `Mutex` makes the type `Sync` (`Event` is `Send` only); it is
    /// uncontended by protocol (record and wait of one index never overlap).
    events: Vec<Mutex<Option<Event>>>,
    peers_enabled: AtomicBool,
}

/// One output of [`EpRowGather::exchange_bands`]: every rank's band of `rows`
/// rows of `band_bytes` sits band-compact at `offset` in its buffer; the
/// gather writes rank `q`'s band to bytes `[q * band_bytes, (q + 1) *
/// band_bytes)` of each `ranks * band_bytes` row of `dst`.
pub struct EpBand<'a> {
    pub offset: usize,
    pub band_bytes: usize,
    pub rows: usize,
    pub dst: &'a GpuTensor,
}

/// Shared EP row-gather transport (`Arc<EpRowGather>`, one clone per rank thread).
pub struct EpRowGather {
    spec: EpRowGatherSpec,
    buffers: usize,
    ranks: Vec<RankState>,
    barrier: SpinBarrier,
    device_waits: bool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl EpRowGather {
    pub fn builder(spec: EpRowGatherSpec) -> EpRowGatherBuilder {
        EpRowGatherBuilder {
            spec,
            regs: (0..spec.ranks.min(MAX_RANKS)).map(|_| None).collect(),
            barrier_timeout: DEFAULT_BARRIER_TIMEOUT,
            device_waits: false,
        }
    }

    pub fn ranks(&self) -> usize {
        self.spec.ranks
    }

    pub fn spec(&self) -> EpRowGatherSpec {
        self.spec
    }

    /// Buffers registered per rank (K).
    pub fn buffers(&self) -> usize {
        self.buffers
    }

    fn check(&self, gpu: &Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        if rank >= self.spec.ranks || buffer >= self.buffers {
            return Err(err(format!(
                "ep_row_gather: rank {rank} / buffer {buffer} out of range (ranks={}, buffers={})",
                self.spec.ranks, self.buffers
            )));
        }
        if gpu.device_id != self.ranks[rank].device_id {
            return Err(err(format!(
                "ep_row_gather: rank {rank} is device {} but was given a Gpu on device {}",
                self.ranks[rank].device_id, gpu.device_id
            )));
        }
        Ok(())
    }

    /// Enable P2P from this rank's device to every other registered device
    /// (idempotent; call on the rank's thread AFTER every peer-read allocation
    /// exists — ROCm does not map later allocations).
    pub fn enable_peers(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank, 0)?;
        gpu.bind_thread()?;
        for (peer, state) in self.ranks.iter().enumerate() {
            // Same physical device (logical emulation): no peer mapping needed.
            if peer == rank || state.device_id == gpu.device_id {
                continue;
            }
            if !gpu.hip.can_access_peer(gpu.device_id, state.device_id)? {
                return Err(err(format!(
                    "ep_row_gather: device {} cannot access peer device {} (rank {rank} -> {peer})",
                    gpu.device_id, state.device_id
                )));
            }
            match gpu.hip.enable_peer_access(state.device_id) {
                Ok(()) => {}
                Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                Err(e) => return Err(e),
            }
        }
        self.ranks[rank]
            .peers_enabled
            .store(true, Ordering::Release);
        Ok(())
    }

    /// Record the rank's producer-complete event for `buffer` on `gpu`'s launch
    /// stream (`gpu.active_stream`, or the null stream when none — it MUST be the
    /// stream the producer kernels used).
    pub fn publish(&self, gpu: &mut Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        self.check(gpu, rank, buffer)?;
        gpu.bind_thread()?;
        let guard = lock(&self.ranks[rank].events[buffer]);
        let event = guard
            .as_ref()
            .ok_or_else(|| err("ep_row_gather: publish after release".into()))?;
        gpu.hip.event_record(event, gpu.active_stream.as_ref())
    }

    /// Wait every peer's `buffer` event, then ONE copy-only kernel: for
    /// slot `s`, `owner` is `slot_owners`' owner of `s` (see [`EpSlotOwners`]);
    /// if `owner != rank` copy row `s` of the owner's `buffer[buffer]` into this
    /// rank's `buffer[buffer]`. Owned slots (and slots with a negative id or
    /// owner) are never written. Bit-exact.
    ///
    /// Host waits are the default even with an active stream. A caller that
    /// opts into `EpRowGatherBuilder::device_waits` uses stream waits when a
    /// stream exists. Events must already be recorded (publish-all first).
    pub fn gather(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
    ) -> HipResult<()> {
        self.check(gpu, rank, buffer)?;
        self.spec.check_call_slots(slots)?;
        self.wait_peers(gpu, rank, buffer)?;
        let mut src = [0usize; MAX_RANKS];
        for (peer, state) in self.ranks.iter().enumerate() {
            src[peer] = state.ptrs[buffer];
        }
        match slot_owners {
            EpSlotOwners::ExpertModulo(slot_experts, n_experts) => gpu.ep_gather_rows_by_owner(
                slot_experts,
                src,
                self.ranks[rank].ptrs[buffer],
                slots,
                self.spec.ranks,
                rank,
                self.spec.row_bytes / 16,
                n_experts,
            ),
            EpSlotOwners::Explicit(owners) => gpu.ep_gather_rows_by_slot_owner(
                owners,
                src,
                self.ranks[rank].ptrs[buffer],
                slots,
                self.spec.ranks,
                rank,
                self.spec.row_bytes / 16,
            ),
        }
    }

    /// `publish` → host barrier over all ranks (same round) → `gather`. One host
    /// thread per rank. A failure on this rank poisons the barrier so peers
    /// return Err instead of waiting out the timeout.
    pub fn exchange(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        slot_owners: EpSlotOwners<'_>,
        slots: usize,
    ) -> HipResult<()> {
        self.spec.check_call_slots(slots)?;
        if let Err(e) = self.publish(gpu, rank, buffer) {
            self.abort();
            return Err(e);
        }
        self.barrier.wait().map_err(|m| err(m.into()))?;
        if let Err(e) = self.gather(gpu, rank, buffer, slot_owners, slots) {
            self.abort();
            return Err(e);
        }
        Ok(())
    }

    /// Wait every peer's producer event of `buffer` (host waits unless the
    /// builder opted into device waits).
    fn wait_peers(&self, gpu: &mut Gpu, rank: usize, buffer: usize) -> HipResult<()> {
        gpu.bind_thread()?;
        if !self.ranks[rank].peers_enabled.load(Ordering::Acquire) {
            return Err(err(format!(
                "ep_row_gather: gather on rank {rank} before enable_peers"
            )));
        }
        for (peer, state) in self.ranks.iter().enumerate() {
            if peer == rank {
                continue;
            }
            let guard = lock(&state.events[buffer]);
            let event = guard
                .as_ref()
                .ok_or_else(|| err("ep_row_gather: gather after release".into()))?;
            match (self.device_waits, gpu.active_stream.as_ref()) {
                (true, Some(stream)) => gpu.hip.stream_wait_event(stream, event)?,
                _ => gpu.hip.event_synchronize(event)?,
            }
        }
        Ok(())
    }

    /// Band exchange (see the module docs): `publish` → barrier → geometry
    /// check → one pitched copy per rank and band. `tag` identifies the call's
    /// geometry and must be equal on every rank; a mismatch aborts the
    /// transport. Every band must lie inside the rank's buffer and `dst` must
    /// hold `rows` rows of `ranks * band_bytes`.
    pub fn exchange_bands(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        tag: u64,
        bands: &[EpBand<'_>],
    ) -> HipResult<()> {
        let result = self.check(gpu, rank, buffer).and_then(|()| {
            self.check_bands(bands)?;
            self.ranks[rank].tags[buffer].store(tag, Ordering::Release);
            self.publish(gpu, rank, buffer)
        });
        if let Err(e) = result {
            self.abort();
            return Err(e);
        }
        self.barrier.wait().map_err(|m| err(m.into()))?;
        if let Err(e) = self.gather_bands(gpu, rank, buffer, tag, bands) {
            self.abort();
            return Err(e);
        }
        Ok(())
    }

    fn check_bands(&self, bands: &[EpBand<'_>]) -> HipResult<()> {
        let capacity = self.spec.buffer_bytes();
        for band in bands {
            let end = band
                .rows
                .checked_mul(band.band_bytes)
                .and_then(|bytes| bytes.checked_add(band.offset));
            let dst = band
                .rows
                .checked_mul(band.band_bytes)
                .and_then(|bytes| bytes.checked_mul(self.spec.ranks));
            match (end, dst) {
                (Some(end), Some(dst)) if end <= capacity && dst <= band.dst.buf.size() => {}
                _ => {
                    return Err(err(format!(
                    "ep_row_gather: band of {} rows x {} B at {} exceeds the {capacity} B buffer \
                         or its {} B destination",
                    band.rows,
                    band.band_bytes,
                    band.offset,
                    band.dst.buf.size()
                )))
                }
            }
        }
        Ok(())
    }

    fn gather_bands(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        buffer: usize,
        tag: u64,
        bands: &[EpBand<'_>],
    ) -> HipResult<()> {
        if let Some(peer) = self
            .ranks
            .iter()
            .position(|state| state.tags[buffer].load(Ordering::Acquire) != tag)
        {
            return Err(err(format!(
                "ep_row_gather: band layout of rank {peer} disagrees with rank {rank} (tag {tag:#x})"
            )));
        }
        self.wait_peers(gpu, rank, buffer)?;
        let ranks = self.spec.ranks;
        for band in bands {
            for (owner, state) in self.ranks.iter().enumerate() {
                // SAFETY: non-owning view of the owner's registered buffer, live
                // until that rank's `release`; the copy is bounds-checked against it.
                let src = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(
                        state.ptrs[buffer] as *mut std::ffi::c_void,
                        state.sizes[buffer],
                    )
                };
                gpu.memcpy_dtod_2d_at_ordered_async(
                    &band.dst.buf,
                    owner * band.band_bytes,
                    ranks * band.band_bytes,
                    &src,
                    band.offset,
                    band.band_bytes,
                    band.band_bytes,
                    band.rows,
                )?;
            }
        }
        Ok(())
    }

    /// Poison the barrier: every current/future waiter returns Err (a rank
    /// failed). Barrier waits time out (default 30 s) with Err rather than
    /// hanging.
    pub fn abort(&self) {
        self.barrier.abort();
    }

    /// Whether `abort` was called (or a barrier wait timed out).
    pub fn is_aborted(&self) -> bool {
        self.barrier.is_aborted()
    }

    /// Destroy this rank's producer events. Call once per rank on that rank's
    /// thread, after the rank's stream and every peer's stream that waits on
    /// these events have drained. Later `publish`/`gather` return Err.
    pub fn release(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank, 0)?;
        gpu.bind_thread()?;
        let mut first = Ok(());
        for slot in &self.ranks[rank].events {
            if let Some(event) = lock(slot).take() {
                let r = gpu.hip.event_destroy(event);
                if first.is_ok() {
                    first = r;
                }
            }
        }
        first
    }
}

impl Drop for EpRowGather {
    fn drop(&mut self) {
        let live = self
            .ranks
            .iter()
            .filter(|r| r.events.iter().any(|e| lock(e).is_some()))
            .count();
        if live > 0 {
            eprintln!(
                "WARN: EpRowGather dropped with producer events live on {live} rank(s) — \
                 call release(rank, gpu) on each rank first; HIP event handles leaked."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ranks: usize, slots: usize, row_bytes: usize) -> EpRowGatherSpec {
        EpRowGatherSpec {
            ranks,
            slots,
            row_bytes,
        }
    }

    #[test]
    fn spec_validation() {
        assert!(spec(2, 10, 10240).validate().is_ok());
        assert!(spec(3, 10, 10240).validate().is_ok());
        assert!(spec(4, 1, 16).validate().is_ok());
        assert!(spec(1, 10, 10240).validate().is_err());
        assert!(spec(0, 10, 10240).validate().is_err());
        assert!(spec(5, 10, 10240).validate().is_ok());
        assert!(spec(6, 10, 10240).validate().is_ok());
        assert!(spec(8, 10, 10240).validate().is_ok());
        assert!(spec(9, 10, 10240).validate().is_err());
        assert!(spec(usize::MAX, 10, 10240).validate().is_err());
        assert!(spec(4, 0, 10240).validate().is_err());
        assert!(spec(4, 10, 0).validate().is_err());
        assert!(spec(4, 10, 10248).validate().is_err());
        assert!(spec(4, 10, 15).validate().is_err());
        assert!(spec(4, usize::MAX, 16).validate().is_err());
        assert!(spec(4, i32::MAX as usize + 1, 16).validate().is_err());
    }

    #[test]
    fn buffer_len_validation() {
        let s = spec(4, 10, 10240);
        assert_eq!(s.buffer_bytes(), 102_400);
        assert!(s.check_buffer_len(0, 102_400).is_ok());
        assert!(s.check_buffer_len(1, 1 << 20).is_ok());
        assert!(s.check_buffer_len(0, 102_399).is_err());
    }

    #[test]
    fn call_slots_within_capacity() {
        let s = spec(4, 80, 10240);
        assert!(s.check_call_slots(1).is_ok());
        assert!(s.check_call_slots(20).is_ok());
        assert!(s.check_call_slots(80).is_ok());
        assert!(s.check_call_slots(0).is_err());
        assert!(s.check_call_slots(81).is_err());
    }

    #[test]
    fn device_waits_require_explicit_opt_in() {
        assert!(!EpRowGather::builder(spec(2, 10, 10240)).device_waits);
        assert!(
            EpRowGather::builder(spec(2, 10, 10240))
                .device_waits(true)
                .device_waits
        );
    }

    #[test]
    fn builder_requires_valid_spec_and_all_ranks() {
        let b = EpRowGather::builder(spec(4, 10, 10248));
        assert!(b.build().is_err());
        let b = EpRowGather::builder(spec(4, 10, 10240));
        assert!(b.build().is_err(), "no rank registered");
    }

    #[test]
    fn send_sync_arc() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EpRowGather>();
        assert_send_sync::<Arc<EpRowGather>>();
    }

    #[test]
    fn barrier_passes_many_generations() {
        let barrier = Arc::new(SpinBarrier::new(4, Duration::from_secs(10)));
        let counter = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let barrier = barrier.clone();
                let counter = counter.clone();
                std::thread::spawn(move || {
                    for round in 0..500usize {
                        counter.fetch_add(1, Ordering::SeqCst);
                        barrier.wait().expect("barrier");
                        // Everyone of this round has incremented before anyone passes.
                        assert!(counter.load(Ordering::SeqCst) >= (round + 1) * 4);
                        barrier.wait().expect("barrier");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2000);
    }

    #[test]
    fn barrier_abort_releases_waiters_and_poisons() {
        let barrier = Arc::new(SpinBarrier::new(3, Duration::from_secs(30)));
        let started = Instant::now();
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || barrier.wait())
            })
            .collect();
        std::thread::sleep(Duration::from_millis(50));
        barrier.abort();
        for w in waiters {
            assert!(w.join().expect("thread").is_err());
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(barrier.wait().is_err(), "future waiters fail too");
    }

    #[test]
    fn barrier_times_out_with_err_and_aborts() {
        let barrier = SpinBarrier::new(2, Duration::from_millis(100));
        let started = Instant::now();
        let r = barrier.wait();
        assert_eq!(r, Err("ep_row_gather barrier timed out"));
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(barrier.is_aborted());
        assert!(barrier.wait().is_err());
    }
}
