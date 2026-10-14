// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next expert-parallel (EP, 2 or 4 ranks) owner-routed chunk MoE
//! exchange transport (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`).
//!
//! Every rank runs the same router, top-k and stable grouping of a prefill chunk
//! layer, so the grouped row layout (`expert_tile_ids`, `sorted_slot_index`,
//! `inverse_perm`) is byte-identical on all. Each rank runs the grouped
//! symmetric IU4 GEMMs on the tiles of the experts it owns only (the dispatch
//! layer's masked tile ids) and writes the BF16 grouped rows of those tiles into
//! its `y_down_grouped`. This transport then gives each rank the peer-owned
//! tiles' rows before the unchanged BF16-row combine reads them. The owner
//! table is the generic engine's (`ExpertRankOwnership`) per-expert owner rank,
//! uploaded by the caller; this module only reads it on the device.
//!
//! 1. [`EpTileExchange::exchange`] records the rank's producer-complete event
//!    `ready[k % 2]` after the routed down
//!    (`HIP_EVENT_RELEASE_TO_SYSTEM`, as [`crate::ep_row_gather::EpRowGather`]
//!    does), passes a host barrier, waits EVERY other rank's `ready[k % 2]`
//!    (ascending rank; a stream wait when `gpu.active_stream` is installed,
//!    else a host `hipEventSynchronize` — exactly the row gather's two
//!    branches), launches ONE pull kernel
//!    (`ep_gather_grouped_rows_by_tile_owner`: for every live tile of a
//!    nonlocal-owned expert, copy the tile's sixteen rows of the OWNER's
//!    `y_down_grouped`, selected from a four-slot source table, onto the same
//!    rows of this rank's), records `done[k % 2]` and passes a second host
//!    barrier. `k` is the rank's exchange counter; all ranks execute the same
//!    invocation sequence (replicated routing, one admission verdict per layer).
//! 2. [`EpTileExchange::before_down`] is called right before the NEXT
//!    writer of `y_down_grouped` (the next invocation's routed down, or the
//!    full-table route after an owner-routed layer): it enqueues the wait for
//!    EVERY other rank's `done[(k - 1) % 2]`, even when all of this rank's
//!    tiles were local (a peer may still be reading its owned rows). The wait
//!    sits one layer of trunk work after the pull it covers, so it is never
//!    exposed.
//!
//! # Hazard argument (N ranks, N in {2, 4})
//!
//! Fix a row `R` of rank `o`'s `y_down_grouped`, owned by `o` in exchange `k`.
//! Its only writer in an exchange is `o`'s routed down (stream order on `o`):
//! every rank's pull writes only ITS OWN `y_down_grouped`, onto rows whose
//! owner is another rank, so a pull never writes a row any peer reads (peers
//! read only rows their OWNER owns). Each row of a rank's buffer therefore has
//! one writer stream, and rows of `o`'s buffer are read by a rank `r != o` only
//! between `o`'s `ready(k)` and `r`'s `done(k)`.
//!
//! * Producer → reader, pairwise for each `(o, r)`: `o`'s down of `R` precedes
//!   `ready_o(k)` in `o`'s stream; barrier 1 of `k` orders `ready_o(k)`'s
//!   RECORD before `r`'s wait enqueue; `r`'s stream waits `ready_o(k)` before the
//!   pull launch, so the pull's read of `R` follows `o`'s write. `r` waits all
//!   `N-1` peers, so this holds for every pair simultaneously.
//! * Reader → next writer, pairwise for each `(o, r)`: the pull's read of `R`
//!   precedes `done_r(k)` in `r`'s stream. `o`'s next write of `R` (layer `k+1`'s
//!   down, or a full-table-route writer) is preceded by `before_down`, which
//!   enqueues a wait on `done_r(k)` for EVERY `r != o`, so the write follows
//!   every reader's read. `waited` advances only after all `N-1` waits are
//!   enqueued, so a partial wait set is never taken for a complete one (a
//!   failure aborts and no later write is issued).
//! * Event re-record. Two alternating event pairs suffice because every
//!   invocation passes two host barriers. `ready_o` of parity `p` is
//!   re-recorded at invocation `k+2` only after every rank `r != o` enqueued its
//!   wait on the record of invocation `k` (those waits precede `r`'s `done(k)`
//!   record and barrier 2 of `k`, which `o` passes before starting `k+1`). The
//!   `done_r` of parity `p` is re-recorded at invocation `k+2`, after barrier 1
//!   of `k+2`, hence after barrier 1 of `k+1`; every `o != r` enqueued its wait
//!   on `done_r(k)` in `before_down` BEFORE recording its `ready(k+1)` (the
//!   barrier-1 arrival), so all `N-1` waiters have submitted. Neither barrier is
//!   N-specific: a host barrier over `ranks` parties orders each rank's earlier
//!   enqueues before every rank's later ones.
//! * Record before wait: barrier 1 follows every rank's `ready(k)` record and
//!   precedes every wait on it. `done(k)` is recorded before barrier 2 of `k`
//!   and waited only in a later `before_down`, after barrier 2.
//! * Last pending done across route transitions: `before_down`'s obligation is
//!   derived from the exchange counter alone ([`pending_done_parity`]), so the
//!   last exchange's `done` stays owed through any number of layers that take
//!   the full-table route (which do not exchange) until the first writer of
//!   `y_down_grouped` after them calls `before_down`. Caller contract: every
//!   writer of `y_down_grouped` after an exchange is preceded by `before_down`.
//!
//! Eager only (events, host barriers and peer reads are not usable inside a
//! captured or retained body); no signal kernels; failures poison the barriers
//! so the other ranks return an error instead of waiting out the timeout.
//!
//! # Lifetime
//!
//! `hip_bridge::Event` has no `Drop`: the events are created on each rank's
//! device in [`EpTileExchangeBuilder::register`] and freed by the explicit
//! [`EpTileExchange::release`], one call per rank after every wait on them has
//! drained.

use crate::ep_row_gather::{SpinBarrier, DEFAULT_BARRIER_TIMEOUT};
use hip_bridge::{
    Event, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM,
};
use rdna_compute::{Gpu, GpuTensor};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// Largest supported rank count: the pull kernel takes a four-slot source
/// table. Supported counts are exactly 2 and this.
pub const MAX_RANKS: usize = 4;

/// Shape of the exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpTileExchangeSpec {
    /// Participants; 2 or [`MAX_RANKS`].
    pub ranks: usize,
    /// Experts the owner table covers.
    pub experts: usize,
    /// uint4 per grouped row (`hidden * 2 / 16` for BF16 rows).
    pub row_u4: usize,
    /// Padded tile capacity of one call (the registered buffer holds at least
    /// this many tiles of 16 rows).
    pub max_tiles: usize,
}

impl EpTileExchangeSpec {
    /// Bytes of one tile (16 grouped rows).
    pub fn tile_bytes(&self) -> usize {
        16 * self.row_u4 * 16
    }

    /// Bytes the registered buffer must hold.
    pub fn buffer_bytes(&self) -> usize {
        self.max_tiles * self.tile_bytes()
    }

    pub fn validate(&self) -> HipResult<()> {
        if !matches!(self.ranks, 2 | MAX_RANKS) {
            return Err(err(format!(
                "ep_tile_exchange: ranks must be 2 or {MAX_RANKS}, got {}",
                self.ranks
            )));
        }
        if self.experts == 0 || i32::try_from(self.experts).is_err() {
            return Err(err("ep_tile_exchange: experts must be 1..=i32::MAX".into()));
        }
        // The pull kernel moves 1024 uint4 per workgroup trip.
        let tile_u4 = self.row_u4.checked_mul(16).unwrap_or(0);
        if tile_u4 == 0 || tile_u4 % 1024 != 0 {
            return Err(err(format!(
                "ep_tile_exchange: row_u4 {} gives a tile that is not a non-zero multiple of 1024 uint4",
                self.row_u4
            )));
        }
        let total = self
            .max_tiles
            .checked_mul(self.tile_bytes())
            .filter(|&bytes| bytes <= u32::MAX as usize);
        if self.max_tiles == 0 || total.is_none() {
            return Err(err(format!(
                "ep_tile_exchange: max_tiles {} x {} B per tile must be non-zero and fit 32 bits",
                self.max_tiles,
                self.tile_bytes()
            )));
        }
        Ok(())
    }

    fn check_call(&self, tiles: usize) -> HipResult<()> {
        if tiles == 0 || tiles > self.max_tiles {
            return Err(err(format!(
                "ep_tile_exchange: call tiles {tiles} outside 1..={} (registered capacity)",
                self.max_tiles
            )));
        }
        Ok(())
    }
}

fn err(msg: String) -> HipError {
    HipError::new(0, &msg)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Parity of the event pair exchange invocation `k` uses.
pub const fn parity(k: u64) -> usize {
    (k % 2) as usize
}

/// Event parity the next down must wait on after `done` exchanges, `None`
/// before the first one: the peer's completion of the last exchange.
pub const fn pending_done_parity(done: u64) -> Option<usize> {
    if done == 0 {
        None
    } else {
        Some(parity(done - 1))
    }
}

/// Every rank except `rank`, ascending: the order each rank waits the others'
/// `ready`/`done` events (at N=2 the single peer).
fn peers(ranks: usize, rank: usize) -> impl Iterator<Item = usize> {
    (0..ranks).filter(move |&r| r != rank)
}

/// The pull kernel's four-slot source table: slot `r` is rank `r`'s
/// `y_down_grouped` base (this rank's own slot is its destination, a valid
/// local base); slots past the rank count stay zero. A fixed stack array so
/// an exchange allocates nothing.
fn pull_sources(bases: impl Iterator<Item = usize>) -> [usize; MAX_RANKS] {
    let mut sources = [0usize; MAX_RANKS];
    for (slot, base) in sources.iter_mut().zip(bases) {
        *slot = base;
    }
    sources
}

/// First device id that two ranks share, if any.
fn shared_device(devices: &[i32]) -> Option<i32> {
    devices
        .iter()
        .enumerate()
        .find_map(|(i, d)| devices[..i].contains(d).then_some(*d))
}

struct RankReg {
    device_id: i32,
    base: usize,
    len: usize,
    ready: [Event; 2],
    done: [Event; 2],
}

/// Collects one registration per rank, then builds the shared transport.
pub struct EpTileExchangeBuilder {
    spec: EpTileExchangeSpec,
    regs: Vec<Option<RankReg>>,
    barrier_timeout: Duration,
}

impl EpTileExchangeBuilder {
    /// Override the host-barrier timeout (default
    /// [`DEFAULT_BARRIER_TIMEOUT`]).
    pub fn barrier_timeout(mut self, timeout: Duration) -> Self {
        self.barrier_timeout = timeout;
        self
    }

    /// One call per rank, on any thread, with that rank's `Gpu`.
    /// `y_down_grouped` is the rank's grouped-row buffer (>= the spec's
    /// [`EpTileExchangeSpec::buffer_bytes`]); its base is what the peer reads.
    /// Creates the rank's four events (two producer, two completion;
    /// disable-timing, release to system scope so peer reads observe the
    /// producer's writes) on its device.
    pub fn register(
        &mut self,
        rank: usize,
        gpu: &mut Gpu,
        y_down_grouped: &GpuTensor,
    ) -> HipResult<()> {
        self.spec.validate()?;
        if rank >= self.spec.ranks {
            return Err(err(format!(
                "ep_tile_exchange: register rank {rank} out of range (ranks={})",
                self.spec.ranks
            )));
        }
        if self.regs[rank].is_some() {
            return Err(err(format!("ep_tile_exchange: rank {rank} registered twice")));
        }
        let len = y_down_grouped.buf.size();
        let base = y_down_grouped.buf.as_ptr() as usize;
        if base == 0 || len < self.spec.buffer_bytes() {
            return Err(err(format!(
                "ep_tile_exchange: rank {rank} y_down_grouped holds {len} B, needs {} B ({} tiles x {} B)",
                self.spec.buffer_bytes(),
                self.spec.max_tiles,
                self.spec.tile_bytes()
            )));
        }
        gpu.bind_thread()?;
        let flags = HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM;
        let mut made: Vec<Event> = Vec::with_capacity(4);
        for _ in 0..4 {
            match gpu.hip.event_create_with_flags(flags) {
                Ok(event) => made.push(event),
                Err(error) => {
                    for event in made {
                        let _ = gpu.hip.event_destroy(event);
                    }
                    return Err(error);
                }
            }
        }
        let mut made = made.into_iter();
        let mut next = || made.next().expect("four events were created");
        self.regs[rank] = Some(RankReg {
            device_id: gpu.device_id,
            base,
            len,
            ready: [next(), next()],
            done: [next(), next()],
        });
        Ok(())
    }

    /// Finish: every rank registered, distinct devices.
    pub fn build(self) -> HipResult<Arc<EpTileExchange>> {
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
                "ep_tile_exchange: rank {rank} never registered (builder dropped; registered events leak)"
            )));
        }
        let devices: Vec<i32> = ranks.iter().map(|r| r.device_id).collect();
        if let Some(device) = shared_device(&devices) {
            return Err(err(format!(
                "ep_tile_exchange: two ranks share device {device}"
            )));
        }
        let states = ranks
            .into_iter()
            .map(|r| RankState {
                device_id: r.device_id,
                base: r.base,
                len: r.len,
                ready: r.ready.map(|e| Mutex::new(Some(e))),
                done: r.done.map(|e| Mutex::new(Some(e))),
                exchanged: AtomicU64::new(0),
                waited: AtomicU64::new(0),
                peers_enabled: AtomicBool::new(false),
            })
            .collect();
        Ok(Arc::new(EpTileExchange {
            spec: self.spec,
            barrier: SpinBarrier::new(self.spec.ranks, self.barrier_timeout),
            ranks: states,
        }))
    }
}

struct RankState {
    device_id: i32,
    /// Device address of this rank's `y_down_grouped` (valid on the peer after
    /// `enable_peers`; ROCm uses one unified address space).
    base: usize,
    len: usize,
    /// Producer-complete events by exchange parity. `None` after `release`. The
    /// `Mutex` makes the type `Sync` (`Event` is `Send` only); it is
    /// uncontended by protocol.
    ready: [Mutex<Option<Event>>; 2],
    /// Pull-complete events by exchange parity.
    done: [Mutex<Option<Event>>; 2],
    /// Exchanges this rank has run (its invocation counter `k`).
    exchanged: AtomicU64,
    /// Exchanges whose peer completions this rank already waited for (all peers).
    waited: AtomicU64,
    peers_enabled: AtomicBool,
}

/// Shared EP tile-exchange transport (`Arc<EpTileExchange>`, one clone per rank
/// thread).
pub struct EpTileExchange {
    spec: EpTileExchangeSpec,
    ranks: Vec<RankState>,
    barrier: SpinBarrier,
}

impl EpTileExchange {
    pub fn builder(spec: EpTileExchangeSpec) -> EpTileExchangeBuilder {
        EpTileExchangeBuilder {
            spec,
            regs: (0..spec.ranks.min(MAX_RANKS)).map(|_| None).collect(),
            barrier_timeout: DEFAULT_BARRIER_TIMEOUT,
        }
    }

    pub fn spec(&self) -> EpTileExchangeSpec {
        self.spec
    }

    /// Exchanges `rank` has run.
    pub fn exchanged(&self, rank: usize) -> u64 {
        self.ranks[rank].exchanged.load(Ordering::Acquire)
    }

    fn check(&self, gpu: &Gpu, rank: usize) -> HipResult<()> {
        if rank >= self.spec.ranks {
            return Err(err(format!(
                "ep_tile_exchange: rank {rank} out of range (ranks={})",
                self.spec.ranks
            )));
        }
        if gpu.device_id != self.ranks[rank].device_id {
            return Err(err(format!(
                "ep_tile_exchange: rank {rank} is device {} but was given a Gpu on device {}",
                self.ranks[rank].device_id, gpu.device_id
            )));
        }
        Ok(())
    }

    /// Enable P2P from this rank's device to EVERY other rank's (idempotent; the
    /// row gather's `enable_peers` already did it on the production path). Call on
    /// the rank's thread AFTER every peer-read allocation exists — ROCm does
    /// not map later allocations.
    pub fn enable_peers(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank)?;
        gpu.bind_thread()?;
        for q in peers(self.spec.ranks, rank) {
            let peer = &self.ranks[q];
            if !gpu.hip.can_access_peer(gpu.device_id, peer.device_id)? {
                return Err(err(format!(
                    "ep_tile_exchange: device {} cannot access peer device {} (rank {rank} -> {q})",
                    gpu.device_id, peer.device_id
                )));
            }
            match gpu.hip.enable_peer_access(peer.device_id) {
                Ok(()) => {}
                Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                Err(e) => return Err(e),
            }
        }
        self.ranks[rank].peers_enabled.store(true, Ordering::Release);
        Ok(())
    }

    /// Stream-wait (an installed `gpu.active_stream`) or host-wait (none: the
    /// row gather's stream-less branch) the peer's event.
    fn wait_peer(&self, gpu: &mut Gpu, slot: &Mutex<Option<Event>>, what: &str) -> HipResult<()> {
        let guard = lock(slot);
        let event = guard
            .as_ref()
            .ok_or_else(|| err(format!("ep_tile_exchange: {what} after release")))?;
        match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.stream_wait_event(stream, event),
            None => gpu.hip.event_synchronize(event),
        }
    }

    fn record(&self, gpu: &mut Gpu, slot: &Mutex<Option<Event>>, what: &str) -> HipResult<()> {
        let guard = lock(slot);
        let event = guard
            .as_ref()
            .ok_or_else(|| err(format!("ep_tile_exchange: {what} after release")))?;
        gpu.hip.event_record(event, gpu.active_stream.as_ref())
    }

    /// Right before this rank's routed down (or any other writer of
    /// `y_down_grouped`, including the full-table route after an owner-routed
    /// layer) overwrites `y_down_grouped`: wait for EVERY peer to have finished
    /// pulling this rank's rows of the last exchange (each peer's `done` of that
    /// exchange), ascending rank, whether or not this rank's own tiles were all
    /// local — a peer may still be reading any owned row. A no-op before the
    /// first exchange and when those waits were already enqueued (once per
    /// exchange). `waited` advances only after every peer wait was enqueued; any
    /// failure aborts the transport.
    pub fn before_down(&self, gpu: &mut Gpu, rank: usize) -> HipResult<()> {
        self.check(gpu, rank)?;
        if self.is_aborted() {
            return Err(err("ep_tile_exchange: aborted".into()));
        }
        let me = &self.ranks[rank];
        let exchanged = me.exchanged.load(Ordering::Acquire);
        let Some(p) = pending_done_parity(exchanged) else {
            return Ok(());
        };
        if me.waited.load(Ordering::Acquire) == exchanged {
            return Ok(());
        }
        gpu.bind_thread()?;
        let result = peers(self.spec.ranks, rank)
            .try_for_each(|q| self.wait_peer(gpu, &self.ranks[q].done[p], "done wait"));
        match result {
            Ok(()) => me.waited.store(exchanged, Ordering::Release),
            Err(_) => self.abort(),
        }
        result
    }

    /// One exchange: publish `y_down_grouped` → host barrier → wait EVERY
    /// peer's producer event (ascending rank) → pull the live tiles of every
    /// peer-owned expert in one launch → record the completion event → host
    /// barrier. `tile_ids` is the UNMASKED scatter output, `owner_table` the
    /// device table of the owner rank per expert (`ExpertRankOwnership`
    /// semantics: values in `0..ranks`), `tiles` the padded tile capacity of
    /// the layout, `y_down_grouped` this rank's registered buffer. A failure
    /// poisons the barriers so the other ranks return an error instead of
    /// waiting out the timeout.
    pub fn exchange(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        owner_table: &GpuTensor,
        tile_ids: &GpuTensor,
        y_down_grouped: &GpuTensor,
        tiles: usize,
    ) -> HipResult<()> {
        let result = self.exchange_inner(gpu, rank, owner_table, tile_ids, y_down_grouped, tiles);
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn exchange_inner(
        &self,
        gpu: &mut Gpu,
        rank: usize,
        owner_table: &GpuTensor,
        tile_ids: &GpuTensor,
        y_down_grouped: &GpuTensor,
        tiles: usize,
    ) -> HipResult<()> {
        self.check(gpu, rank)?;
        self.spec.check_call(tiles)?;
        gpu.bind_thread()?;
        let ranks = self.spec.ranks;
        let me = &self.ranks[rank];
        if !me.peers_enabled.load(Ordering::Acquire) {
            return Err(err(format!(
                "ep_tile_exchange: exchange on rank {rank} before enable_peers"
            )));
        }
        if y_down_grouped.buf.as_ptr() as usize != me.base || y_down_grouped.buf.size() != me.len {
            return Err(err(format!(
                "ep_tile_exchange: rank {rank} was handed a y_down_grouped other than the registered buffer"
            )));
        }
        let k = me.exchanged.load(Ordering::Acquire);
        let p = parity(k);
        self.record(gpu, &me.ready[p], "publish")?;
        self.barrier.wait().map_err(|m| err(m.into()))?;
        for q in peers(ranks, rank) {
            self.wait_peer(gpu, &self.ranks[q].ready[p], "ready wait")?;
        }
        // Local base allowed (slot `rank` is this rank's own `y_down_grouped`);
        // slots past `ranks` stay zero. Fixed array: no per-exchange allocation.
        let sources = pull_sources(self.ranks.iter().map(|r| r.base));
        gpu.ep_gather_grouped_rows_by_tile_owner(
            tile_ids,
            owner_table,
            sources,
            y_down_grouped,
            tiles,
            self.spec.experts,
            ranks,
            rank,
            self.spec.row_u4,
        )?;
        self.record(gpu, &me.done[p], "completion")?;
        self.barrier.wait().map_err(|m| err(m.into()))?;
        me.exchanged.store(k + 1, Ordering::Release);
        Ok(())
    }

    /// One cross-rank host barrier (every rank calls it at the same point). A timeout or an
    /// abort returns Err and poisons the transport.
    pub fn host_barrier(&self) -> HipResult<()> {
        self.barrier.wait().map_err(|m| err(m.into()))
    }

    /// Poison the barriers: every current/future waiter returns Err (a rank
    /// failed). Barrier waits time out (default 30 s) with Err rather than
    /// hanging.
    pub fn abort(&self) {
        self.barrier.abort();
    }

    /// Whether `abort` was called (or a barrier wait timed out).
    pub fn is_aborted(&self) -> bool {
        self.barrier.is_aborted()
    }

    /// Destroy this rank's events. Call once per rank on that rank's thread,
    /// after the rank's stream and every peer stream that waits on them have
    /// drained. Later calls return Err.
    pub fn release(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check(gpu, rank)?;
        gpu.bind_thread()?;
        let mut first = Ok(());
        let state = &self.ranks[rank];
        for slot in state.ready.iter().chain(state.done.iter()) {
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

impl Drop for EpTileExchange {
    fn drop(&mut self) {
        let live = self
            .ranks
            .iter()
            .filter(|r| r.ready.iter().chain(r.done.iter()).any(|e| lock(e).is_some()))
            .count();
        if live > 0 {
            eprintln!(
                "WARN: EpTileExchange dropped with events live on {live} rank(s) — \
                 call release(rank, gpu) on each rank first; HIP event handles leaked."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(max_tiles: usize) -> EpTileExchangeSpec {
        EpTileExchangeSpec {
            ranks: 2,
            experts: 512,
            row_u4: 320,
            max_tiles,
        }
    }

    #[test]
    fn production_shape_validates() {
        let s = spec(3_040);
        assert!(s.validate().is_ok());
        assert_eq!(s.tile_bytes(), 81_920);
        assert_eq!(s.buffer_bytes(), 3_040 * 81_920);
    }

    #[test]
    fn supported_rank_counts_validate() {
        assert!(EpTileExchangeSpec { ranks: 2, ..spec(8) }.validate().is_ok());
        assert!(EpTileExchangeSpec { ranks: 4, ..spec(8) }.validate().is_ok());
    }

    #[test]
    fn spec_validation_refuses_everything_else() {
        for ranks in [0, 1, 3, 5, 8, usize::MAX] {
            assert!(EpTileExchangeSpec { ranks, ..spec(8) }.validate().is_err(), "ranks={ranks}");
        }
        assert!(EpTileExchangeSpec { experts: 0, ..spec(8) }.validate().is_err());
        assert!(EpTileExchangeSpec { experts: usize::MAX, ..spec(8) }.validate().is_err());
        assert!(EpTileExchangeSpec { row_u4: 0, ..spec(8) }.validate().is_err());
        // 16 rows x 72 uint4 = 1152 uint4: not a multiple of 1024.
        assert!(EpTileExchangeSpec { row_u4: 72, ..spec(8) }.validate().is_err());
        assert!(spec(0).validate().is_err());
        // 60,000 tiles x 80 KiB exceeds 32 bits.
        assert!(spec(60_000).validate().is_err());
    }

    #[test]
    fn call_tiles_within_capacity() {
        let s = spec(3_040);
        assert!(s.check_call(1).is_ok());
        assert!(s.check_call(3_040).is_ok());
        assert!(s.check_call(0).is_err());
        assert!(s.check_call(3_041).is_err());
    }

    /// The next down waits the peer's completion of the LAST exchange: parity
    /// alternates and nothing is owed before the first one.
    #[test]
    fn done_wait_parity_follows_the_last_exchange() {
        assert_eq!(pending_done_parity(0), None);
        assert_eq!(pending_done_parity(1), Some(0));
        assert_eq!(pending_done_parity(2), Some(1));
        assert_eq!(pending_done_parity(3), Some(0));
        assert_eq!(pending_done_parity(48), Some(1));
        assert_eq!(pending_done_parity(49), Some(0));
        assert_eq!((parity(0), parity(1), parity(2), parity(47), parity(48)), (0, 1, 0, 1, 0));
    }

    #[test]
    fn builder_requires_valid_spec_and_all_ranks() {
        // Unsupported rank counts fail at build on the spec, before any registration.
        for ranks in [0, 1, 3, 5, 8] {
            let b = EpTileExchange::builder(EpTileExchangeSpec { ranks, ..spec(8) });
            assert!(b.build().is_err(), "ranks={ranks}");
        }
        // Supported counts with nothing registered: the missing registration is the refusal.
        for ranks in [2, 4] {
            let b = EpTileExchange::builder(EpTileExchangeSpec { ranks, ..spec(8) });
            assert!(b.build().is_err(), "ranks={ranks}: no rank registered");
        }
    }

    #[test]
    fn builder_holds_one_slot_per_supported_rank() {
        for ranks in [2, 4] {
            let b = EpTileExchange::builder(EpTileExchangeSpec { ranks, ..spec(8) });
            assert_eq!(b.regs.len(), ranks);
            assert!(b.regs.iter().all(Option::is_none));
        }
        // An unsupported spec never sizes a registration table past MAX_RANKS.
        let b = EpTileExchange::builder(EpTileExchangeSpec { ranks: usize::MAX, ..spec(8) });
        assert!(b.regs.len() <= MAX_RANKS);
    }

    /// Every rank waits every OTHER rank's event, in ascending rank order; at N=2
    /// that is the single peer (the original EP2 wait order).
    #[test]
    fn peer_wait_order_covers_every_other_rank_ascending() {
        assert_eq!(peers(2, 0).collect::<Vec<_>>(), [1]);
        assert_eq!(peers(2, 1).collect::<Vec<_>>(), [0]);
        assert_eq!(peers(4, 0).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(peers(4, 1).collect::<Vec<_>>(), [0, 2, 3]);
        assert_eq!(peers(4, 2).collect::<Vec<_>>(), [0, 1, 3]);
        assert_eq!(peers(4, 3).collect::<Vec<_>>(), [0, 1, 2]);
        // Pairwise coverage: every ordered pair (waiter, writer) appears exactly once.
        for ranks in [2, 4] {
            for waiter in 0..ranks {
                for writer in 0..ranks {
                    let hits = peers(ranks, waiter).filter(|&q| q == writer).count();
                    assert_eq!(hits, usize::from(waiter != writer));
                }
            }
        }
    }

    /// The pull kernel's source table: rank r's base at index r (the local base
    /// stays valid), unused slots zero, fixed-size (no per-exchange allocation).
    #[test]
    fn pull_sources_layout() {
        assert_eq!(pull_sources([0x1000, 0x2000].into_iter()), [0x1000, 0x2000, 0, 0]);
        assert_eq!(
            pull_sources([0x1000, 0x2000, 0x3000, 0x4000].into_iter()),
            [0x1000, 0x2000, 0x3000, 0x4000]
        );
    }

    #[test]
    fn distinct_devices_required() {
        assert_eq!(shared_device(&[0, 1]), None);
        assert_eq!(shared_device(&[3, 1, 0, 2]), None);
        assert_eq!(shared_device(&[0, 0]), Some(0));
        assert_eq!(shared_device(&[0, 1, 2, 1]), Some(1));
    }

    /// N-rank schedule: after the exchange counter reaches `done`, every rank's
    /// `before_down` waits the same parity of EVERY peer's `done`, once, and the
    /// last pending done persists across any number of non-exchange (full-route)
    /// layers because it is derived from the counter alone.
    #[test]
    fn last_pending_done_persists_across_full_route_layers() {
        for done in 1..8u64 {
            let p = pending_done_parity(done).unwrap();
            assert_eq!(p, parity(done - 1));
            // Layers on the full-table route do not advance the counter.
            for _full_route_layer in 0..5 {
                assert_eq!(pending_done_parity(done), Some(p));
            }
        }
    }

    #[test]
    fn send_sync_arc() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EpTileExchange>();
        assert_send_sync::<Arc<EpTileExchange>>();
    }
}
