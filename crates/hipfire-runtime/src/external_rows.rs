// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded storage for SSD-resident embedding rows that are too large to keep
//! resident (e.g. Qwen4's hashed n-gram per-layer embeddings).
//!
//! `RowStore` deliberately owns the reader worker and its bounded resources.  It
//! does not know how row ids are derived, schedule generation, mutate model
//! state, or perform GPU work.  A caller computes the global row ids it needs,
//! enqueues them with [`RowStore::prefetch`] as early as they are known, waits
//! for the completed [`RowLease`] where the rows are consumed, uploads the
//! contiguous staging bytes, and drops the lease after the device has consumed
//! them.  [`RowFetch`] owns that request-local ticket/lease lifecycle.
//!
//! The production source is a vector of sealed [`SourceRangeDescriptor`]s,
//! one for each physical shard of one `[rows, width]` table.  Reads are
//! positional and descriptor checked; no mmap, source-wide `Vec`, or
//! file-backed GPU pointer is used.  Leases always publish rows as
//! little-endian BF16, whatever tier the shards store.

use crate::model_source::{SourceError, SourceRangeDescriptor};
use crate::ple_hot_store::HotRowStore;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Userspace cache budget.  This is deliberately fixed rather than a public
/// runtime knob: the model's SSD path must remain bounded on every request.
pub const PAGE_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Physical rows per userspace page.
///
/// A page is both the cache unit and the read unit, so the row *count* decides
/// how much of an OS page one requested row drags in. Requested rows are drawn
/// at random over the whole 100 GB physical table, so a page read almost never
/// serves any other requested row and the cost of one row is the whole page.
/// Measured on gfx1151/ext4 with a 2 MiB page: a 291-token chunk requested 4656
/// rows in 4483 distinct windows and pulled ~9.4 GB for ~1.5 MB of useful bytes
/// — 492 ms of page-cache copy (1810 ms cold) spent with the GPU idle.
///
/// The device cost of a window is `1 + (window - 1) / 4096` pages, so the row
/// count is a real lever and it pulls two ways: fewer rows per page means fewer
/// pages fetched per row (~1.04 at one row, ~1.17 at four, ~1.5 at twelve,
/// ~2.0 at twenty-four) but more cache entries, and every entry carries index
/// overhead outside the byte budget. Four rows keeps the page inside a fifth of
/// an OS page and the index overhead near a quarter of the payload.
const ROWS_PER_PAGE: usize = 4;
/// Concurrent source reads of one request's scattered page groups: a small
/// request (decode, MTP verify, short prompts) keeps this many readers.
const PARALLEL_READERS: usize = 16;
/// Reader ceiling of a large request (a prefill chunk's ~110K scattered
/// pages). Random 1280-byte buffered reads on the Halo (gfx1151) host's NVMe
/// measured ~37K IOPS at 16 readers, ~90K at 128 and ~101K at 256. The first
/// prefill chunk's fetch has no earlier chunk to hide behind, so the GPU
/// idles for its whole read and that read is bound by queue depth.
const MAX_PARALLEL_READERS: usize = 256;
/// Page groups each extra reader must have before another is spawned, so a
/// thread's spawn cost stays small against its reads.
const READS_PER_READER: usize = 64;

/// Readers for `groups` scattered page-group reads, at most `max_readers`.
fn reader_count(groups: usize, max_readers: usize) -> usize {
    (groups / READS_PER_READER).clamp(PARALLEL_READERS, max_readers)
}

/// Pages inserted into the cache per state-lock hold when a multi-group
/// request's reads are published.
const PUBLISH_BATCH: usize = 512;
/// Longest run of consecutive hot-row pack rows read with one `pread`.
const HOT_RUN_MAX_BYTES: u64 = 16 * 1024;

/// One requested row served from the hot-row pack.
#[derive(Debug, Clone, Copy)]
struct HotItem {
    /// Absolute offset of the row in the pack file.
    pack_offset: u64,
    /// Byte offset of the decoded row in the lease.
    output_offset: usize,
}

/// One `pread` of consecutive pack rows.
#[derive(Debug, Clone)]
struct HotRun {
    offset: u64,
    len: usize,
    /// The `HotPlan::items` this read serves.
    items: std::ops::Range<usize>,
}

/// The pack reads of one fetch: the rows of every page whose requested rows
/// the pack all holds, sorted by pack offset.
#[derive(Debug, Default)]
struct HotPlan {
    items: Vec<HotItem>,
    runs: Vec<HotRun>,
    /// Pages served from the pack.
    pages: u64,
    /// Requested rows served from the pack.
    rows: u64,
}

/// The shared work of one multi-read fetch.
struct ReadWork<'a> {
    /// Source page groups, ascending by file offset.
    groups: &'a [Vec<PageKey>],
    group_cursor: AtomicUsize,
    hot: &'a HotPlan,
    hot_cursor: AtomicUsize,
    store: Option<&'a HotRowStore>,
}

/// One reader's result of a multi-read fetch.
#[derive(Default)]
struct ReaderShare {
    /// Source pages read (for [`RowStoreInner::publish_pending`]).
    pages: Vec<(PageKey, Vec<u8>)>,
    /// When this reader found the source groups exhausted.
    groups_done: Duration,
}

/// Per-fetch timing of a large PLE fetch, logged under
/// `HIPFIRE_PLE_HOT_DEBUG=1` (one line per fetch of >= `FetchTrace::MIN_ROWS`
/// rows, i.e. prefill chunks, not decode steps).
struct FetchTrace {
    started: Instant,
    rows: usize,
    pages: usize,
    hot_pages: u64,
    hot_rows: u64,
    hot_reads: usize,
    classified_ms: f64,
    cache_hits: usize,
    misses: usize,
    groups: usize,
    scanned_ms: f64,
    readers: usize,
    groups_done_ms: f64,
    parallel_ms: f64,
    inline_read_ms: f64,
}

impl FetchTrace {
    const MIN_ROWS: usize = 4096;

    fn start(rows: usize, pages: usize) -> Option<FetchTrace> {
        (rows >= Self::MIN_ROWS && hipfire_config::memory::ple_hot_debug()).then(|| FetchTrace {
            started: Instant::now(),
            rows,
            pages,
            hot_pages: 0,
            hot_rows: 0,
            hot_reads: 0,
            classified_ms: 0.0,
            cache_hits: 0,
            misses: 0,
            groups: 0,
            scanned_ms: 0.0,
            readers: 0,
            groups_done_ms: 0.0,
            parallel_ms: 0.0,
            inline_read_ms: 0.0,
        })
    }

    fn ms_since(at: Instant) -> f64 {
        at.elapsed().as_secs_f64() * 1e3
    }

    fn mark_classified(&mut self, hot: &HotPlan) {
        self.hot_pages = hot.pages;
        self.hot_rows = hot.rows;
        self.hot_reads = hot.runs.len();
        self.classified_ms = Self::ms_since(self.started);
    }

    fn mark_cache_scanned(&mut self, hits: usize, misses: usize, groups: usize) {
        self.cache_hits = hits;
        self.misses = misses;
        self.groups = groups;
        self.scanned_ms = Self::ms_since(self.started);
    }

    fn parallel(&mut self, started: Instant, readers: usize, shares: &[ReaderShare]) {
        self.parallel_ms = Self::ms_since(started);
        self.readers = readers;
        self.groups_done_ms = shares
            .iter()
            .map(|share| share.groups_done.as_secs_f64() * 1e3)
            .fold(0.0, f64::max);
    }

    fn inline_reads(&mut self, started: Instant) {
        self.inline_read_ms = Self::ms_since(started);
    }

    fn finish(self) {
        eprintln!(
            "  ple fetch: {} rows {} pages | pack {} pages {} rows {} reads | classified at {:.1} ms | \
             cache {} hits {} misses {} groups, scanned at {:.1} ms | parallel {:.1} ms with {} \
             readers, source groups done at {:.1} ms | inline reads {:.1} ms | total {:.1} ms",
            self.rows,
            self.pages,
            self.hot_pages,
            self.hot_rows,
            self.hot_reads,
            self.classified_ms,
            self.cache_hits,
            self.misses,
            self.groups,
            self.scanned_ms,
            self.parallel_ms,
            self.readers,
            self.groups_done_ms,
            self.inline_read_ms,
            Self::ms_since(self.started),
        );
    }
}

/// Default byte budget of one coalesced read or one staging buffer.  The
/// effective size is rounded down to a whole number of decoded rows; a
/// caller that requests more rows at once sizes it with
/// [`RowStore::with_staging_rows`].
const STAGING_BUDGET_BYTES: usize = 8 * 1024 * 1024;
/// At most two staging buffers exist: one completed output and one read buffer.
pub const MAX_STAGING_BUFFERS: usize = 2;
/// Bounded queue of request ids.  A completed request also occupies one ticket
/// until its lease is consumed or the request is cancelled.
pub const READER_QUEUE_CAPACITY: usize = 64;

/// A PLE fetch is idle once none ran for this long: longer than the compute
/// between two prefill chunks' fetches (~4 s for an 8192-row chunk on Halo), so
/// background readers only run while no request is being served.
pub const FETCH_QUIET: Duration = Duration::from_secs(5);

/// Whether a [`RowStore`] is serving fetches.  Every fetch (a prefill chunk or
/// a decode step) holds a [`FetchGuard`]; background readers of the same table
/// (the hot-row prewarm, the PLE rows prewarm) wait until the store has been
/// idle for their quiet period.
pub struct FetchActivity {
    /// Time base of `last_ms`.
    created: Instant,
    /// Milliseconds after `created` (plus one; 0 = never) at which the last
    /// fetch finished.
    last_ms: AtomicU64,
    /// Fetches in flight.
    active: AtomicUsize,
}

/// A fetch in flight (from [`FetchActivity::busy`]); the store counts as busy
/// until [`FETCH_QUIET`] (or the waiter's own quiet period) after the last
/// guard drops.
pub struct FetchGuard<'a>(&'a FetchActivity);

impl Drop for FetchGuard<'_> {
    fn drop(&mut self) {
        let activity = self.0;
        let at = activity.created.elapsed().as_millis() as u64 + 1;
        activity.last_ms.fetch_max(at, Ordering::AcqRel);
        activity.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl FetchActivity {
    pub fn new() -> Self {
        Self {
            created: Instant::now(),
            last_ms: AtomicU64::new(0),
            active: AtomicUsize::new(0),
        }
    }

    /// A fetch starts; the returned guard marks it finished when dropped.
    pub fn busy(&self) -> FetchGuard<'_> {
        self.active.fetch_add(1, Ordering::AcqRel);
        FetchGuard(self)
    }

    /// Fetches in flight.
    pub fn in_flight(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    /// How long the store has been quiet: `None` while a fetch is in flight,
    /// `Some(MAX)` when it never served one.
    fn quiet_for(&self) -> Option<Duration> {
        if self.in_flight() > 0 {
            return None;
        }
        let last = self.last_ms.load(Ordering::Acquire);
        if last == 0 {
            return Some(Duration::MAX);
        }
        Some(Duration::from_millis(
            (self.created.elapsed().as_millis() as u64 + 1).saturating_sub(last),
        ))
    }

    /// No fetch is in flight and the last one finished at least `quiet` ago
    /// (or none ever ran).
    pub fn idle_for(&self, quiet: Duration) -> bool {
        self.quiet_for().is_some_and(|since| since >= quiet)
    }

    /// Sleep until [`FetchActivity::idle_for`]`(quiet)`; `false` when `stop`
    /// was raised meanwhile.
    pub fn wait_idle(&self, quiet: Duration, stop: &AtomicBool) -> bool {
        const POLL: Duration = Duration::from_millis(20);
        loop {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            match self.quiet_for() {
                None => std::thread::sleep(POLL),
                Some(since) if since >= quiet => return true,
                Some(since) => std::thread::sleep((quiet - since).min(POLL)),
            }
        }
    }
}

impl Default for FetchActivity {
    fn default() -> Self {
        Self::new()
    }
}

/// Stored representation of one physical row.
///
/// The artifact declares this through the shard dtype, so both tiers can be
/// read by one reader: an artifact written before the Q8F16 tier is BF16 rows,
/// and a Q8F16 artifact is five 34-byte blocks per row (`qf16` scale + 32 i8,
/// [`Q8_BLOCK_BYTES`]).  Decoding happens on the reader
/// thread while the requested rows are copied out of a cached page, so the
/// cache keeps the compact form and the device path is untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowEncoding {
    /// Raw little-endian BF16: `2 * width` bytes per row.
    Bf16,
    /// Q8F16: `width / 32` blocks of `[f16 scale][32 x i8]`.
    Q8F16,
    /// Q4LM: `width / 32` blocks of `[f16 scale][16 bytes of 4-bit Lloyd-Max
    /// labels]` (18 bytes), the derived tier of [`crate::ple_q4`].
    Q4Lm,
}

/// Bytes of one Q8F16 block: an f16 scale followed by 32 signed bytes.
const Q8_BLOCK_BYTES: usize = 34;
/// Weights per Q8F16 block.
const Q8_BLOCK_WEIGHTS: usize = 32;

impl RowEncoding {
    /// Encoded bytes of one physical row of `width` values, or `None` for a
    /// width the encoding cannot hold (zero, or for the block encodings not a
    /// whole number of 32-value blocks).  A partial block is refused rather
    /// than padded: the encoders only ever write whole blocks.
    pub const fn checked_encoded_row_bytes(self, width: usize) -> Option<usize> {
        if width == 0 {
            return None;
        }
        match self {
            Self::Bf16 => Some(width * 2),
            Self::Q8F16 if width % Q8_BLOCK_WEIGHTS == 0 => {
                Some(width / Q8_BLOCK_WEIGHTS * Q8_BLOCK_BYTES)
            }
            Self::Q4Lm if width % Q8_BLOCK_WEIGHTS == 0 => {
                Some(width / Q8_BLOCK_WEIGHTS * crate::ple_q4::Q4LM_BLOCK_BYTES)
            }
            _ => None,
        }
    }

    /// Encoded bytes of one physical row of `width` values.
    ///
    /// # Panics
    /// If `width` is zero or, for the block encodings, not a whole number of
    /// 32-value blocks.
    pub fn encoded_row_bytes(self, width: usize) -> usize {
        self.checked_encoded_row_bytes(width)
            .expect("row width is a whole number of blocks")
    }

    /// Bytes of one userspace page of `width`-value rows.
    pub fn page_bytes(self, width: usize) -> usize {
        ROWS_PER_PAGE * self.encoded_row_bytes(width)
    }

    /// The artifact dtype this encoding is declared as.
    pub const fn dtype_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Q8F16 => "Q8_0",
            Self::Q4Lm => "Q4LM",
        }
    }

    /// The encoding a shard dtype declares, or `None` when the reader has no
    /// decoder for it.
    ///
    /// The declared dtype is the contract. Nothing infers the tier from the
    /// extent, so a mislabelled shard is refused instead of decoded at the
    /// wrong width.
    pub fn from_dtype(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_uppercase().as_str() {
            "BF16" | "BFLOAT16" => Some(Self::Bf16),
            "Q8_0" | "Q8F16" | "Q8" | "Q8_F16" => Some(Self::Q8F16),
            "Q4LM" => Some(Self::Q4Lm),
            _ => None,
        }
    }

    /// Decode one encoded row into `out`, the row's little-endian BF16 form.
    ///
    /// `out` fixes the row width; `encoded` must be exactly
    /// [`Self::encoded_row_bytes`] of that width, so a short row can never be
    /// silently padded.
    fn decode_row(self, encoded: &[u8], out: &mut [u8]) -> Result<(), RowStoreError> {
        match self {
            Self::Bf16 => {
                if encoded.len() != out.len() {
                    return Err(RowStoreError::InvalidLease {
                        reason: "BF16 row length mismatch".to_string(),
                    });
                }
                out.copy_from_slice(encoded);
            }
            Self::Q8F16 => {
                if encoded.len() != self.encoded_row_bytes(out.len() / 2) {
                    return Err(RowStoreError::InvalidLease {
                        reason: "Q8F16 row length mismatch".to_string(),
                    });
                }
                for (block, chunk) in encoded.chunks_exact(Q8_BLOCK_BYTES).enumerate() {
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    for index in 0..Q8_BLOCK_WEIGHTS {
                        let value = scale * (chunk[2 + index] as i8) as f32;
                        let cell = (block * Q8_BLOCK_WEIGHTS + index) * 2;
                        out[cell..cell + 2].copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
                    }
                }
            }
            Self::Q4Lm => {
                if Some(encoded.len()) != self.checked_encoded_row_bytes(out.len() / 2) {
                    return Err(RowStoreError::InvalidLease {
                        reason: "Q4LM row length mismatch".to_string(),
                    });
                }
                crate::ple_q4::decode_q4lm_row_bf16(encoded, out)
                    .map_err(|reason| RowStoreError::InvalidLease { reason })?;
            }
        }
        Ok(())
    }
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    // The decoder the runtime uses for f16 payloads (half-backed), not a local
    // hand-rolled version: f16 scale decode must be exact, including subnormals.
    crate::llama::f16_to_f32(bits)
}

/// Round-to-nearest-even f32 -> BF16 bits, the narrowing the parity fixture
/// uses (`hipfire-arch-qwen4/examples/qwen4_parity.rs`), so a decoded row is
/// bit-identical to the
/// narrowing the reference does.
#[inline]
pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// A page in the physical shard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PageKey {
    shard: usize,
    page: usize,
}

/// Checked location of one global row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLocation {
    /// Global row across all shards.
    pub global_row: u64,
    /// Physical shard containing the row.
    pub shard: usize,
    /// Row number within that shard.
    pub local_row: usize,
    /// Page number within that shard.
    pub page: usize,
    /// Byte offset within the page.
    pub page_byte_offset: usize,
}

impl RowLocation {
    fn page_key(self) -> PageKey {
        PageKey {
            shard: self.shard,
            page: self.page,
        }
    }
}

/// Counters and bounded-resource measurements for diagnostics and unload
/// proofs.  These are snapshots; they do not grant access to mutable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCacheStats {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub resident_pages: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub reads: u64,
    pub coalesced_reads: u64,
    pub read_bytes: u64,
    pub evictions: u64,
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub staging_high_water: usize,
    /// Rows served from the attached PLE hot-row pack (no page read, no
    /// `cache_hits`/`cache_misses` accounting).
    pub hot_rows: u64,
    /// Pages whose every requested row the hot-row pack served.
    pub hot_pages: u64,
    /// Reads issued against the hot-row pack (one per run of consecutive
    /// pack rows).
    pub hot_reads: u64,
}

/// Proof returned by [`RowStore::quiesce`] and [`RowStore::unload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowQuiesceReport {
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub cache_bytes: usize,
    pub cache_pages: usize,
}

impl RowQuiesceReport {
    pub const fn is_clean(self) -> bool {
        self.queue_depth == 0
            && self.outstanding_readers == 0
            && self.outstanding_leases == 0
            && self.staging_in_use == 0
    }
}

/// Failure modes of the bounded row reader.  In particular, source read
/// failures are preserved as [`SourceError`] and never become zero rows.
#[derive(Debug)]
pub enum RowStoreError {
    Descriptor {
        index: usize,
        reason: String,
    },
    Source(SourceError),
    GlobalRowOutOfRange {
        row: u64,
        padded_rows: u64,
    },
    PaddingRow {
        row: u64,
        valid_rows: u64,
    },
    RequestTooLarge {
        rows: usize,
        max_rows: usize,
    },
    QueueFull,
    StagingUnavailable,
    Quiescing,
    EpochMismatch {
        requested: u64,
        current: u64,
    },
    EpochNotMonotonic {
        requested: u64,
        current: u64,
    },
    Canceled,
    UnknownTicket(u64),
    AlreadyConsumed(u64),
    WorkerStopped,
    InvalidLease {
        reason: String,
    },
    LeaseStale {
        lease_epoch: u64,
        current_epoch: u64,
    },
    QuiesceTimeout(RowQuiesceReport),
    WorkerJoin,
}

impl fmt::Display for RowStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor { index, reason } => {
                write!(f, "row shard descriptor {index} is invalid: {reason}")
            }
            Self::Source(error) => write!(f, "row source read failed: {error}"),
            Self::GlobalRowOutOfRange { row, padded_rows } => {
                write!(f, "row {row} is outside {padded_rows} physical rows")
            }
            Self::PaddingRow { row, valid_rows } => {
                write!(f, "row {row} is padding (valid rows end at {valid_rows})")
            }
            Self::RequestTooLarge { rows, max_rows } => {
                write!(f, "row prefetch has {rows} rows; maximum is {max_rows}")
            }
            Self::QueueFull => f.write_str("row reader queue is full"),
            Self::StagingUnavailable => f.write_str("row staging buffers are all in use"),
            Self::Quiescing => f.write_str("row store is quiescing or unloaded"),
            Self::EpochMismatch { requested, current } => {
                write!(
                    f,
                    "row request epoch {requested} does not match current epoch {current}"
                )
            }
            Self::EpochNotMonotonic { requested, current } => {
                write!(
                    f,
                    "row store epoch {requested} is not newer than current epoch {current}"
                )
            }
            Self::Canceled => f.write_str("row prefetch was canceled"),
            Self::UnknownTicket(id) => write!(f, "unknown row prefetch ticket {id}"),
            Self::AlreadyConsumed(id) => write!(f, "row prefetch ticket {id} was already consumed"),
            Self::WorkerStopped => f.write_str("row reader worker has stopped"),
            Self::InvalidLease { reason } => write!(f, "invalid row lease: {reason}"),
            Self::LeaseStale {
                lease_epoch,
                current_epoch,
            } => write!(
                f,
                "row lease epoch {lease_epoch} is stale at current epoch {current_epoch}"
            ),
            Self::QuiesceTimeout(report) => write!(f, "row store quiesce timed out: {report:?}"),
            Self::WorkerJoin => f.write_str("row reader worker did not join cleanly"),
        }
    }
}

impl std::error::Error for RowStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SourceError> for RowStoreError {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}

/// A completed, epoch-bound set of rows in request order.
///
/// The lease owns exactly one of the two fixed staging buffers until dropped.
/// It intentionally exposes a borrowed byte slice rather than moving the
/// buffer out: callers cannot accidentally retain an unbounded third staging
/// allocation past model unload.
pub struct RowLease {
    inner: Arc<RowStoreInner>,
    epoch: u64,
    ids: Vec<u64>,
    bytes: Vec<u8>,
}

impl fmt::Debug for RowLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowLease")
            .field("epoch", &self.epoch)
            .field("rows", &self.ids.len())
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl RowLease {
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn row_count(&self) -> usize {
        self.ids.len()
    }

    /// Global row ids in the order they were requested.
    pub fn row_ids(&self) -> &[u64] {
        &self.ids
    }

    /// Borrow the contiguous `[rows, width]` BF16 staging bytes only
    /// while this lease's epoch is current.  A fallible view is intentional:
    /// returning an unconditional slice would let a reset publish stale rows.
    pub fn as_bytes(&self) -> Result<&[u8], RowStoreError> {
        self.try_bytes()
    }

    pub fn validate(&self) -> Result<(), RowStoreError> {
        self.inner.validate_lease_epoch(self.epoch)
    }

    /// Re-check the reader epoch after the caller's device upload has
    /// completed.  The upload itself is intentionally owned by the caller;
    /// this method closes the second half of the publication boundary so a
    /// reset cannot silently publish stale rows.
    pub fn validate_after_upload(&self) -> Result<(), RowStoreError> {
        self.validate()
    }

    pub fn try_bytes(&self) -> Result<&[u8], RowStoreError> {
        self.validate()?;
        Ok(&self.bytes)
    }

    /// The decoded BF16 bytes of the `index`-th requested row.
    pub fn row_bytes(&self, index: usize) -> Result<&[u8], RowStoreError> {
        self.validate()?;
        if index >= self.ids.len() {
            return Err(RowStoreError::InvalidLease {
                reason: format!("row index {index} of {}", self.ids.len()),
            });
        }
        let row_bytes = self.inner.decoded_row_bytes;
        let begin = index * row_bytes;
        Ok(&self.bytes[begin..begin + row_bytes])
    }

    /// Copy all rows in one contiguous operation into caller-owned upload
    /// storage.  This is the CPU staging boundary; callers must not copy one
    /// row at a time into separate device allocations.  The epoch is checked
    /// both before and after the copy so a reset racing this operation is
    /// reported instead of publishing stale bytes.
    pub fn stage_into(&self, destination: &mut [u8]) -> Result<(), RowStoreError> {
        self.validate()?;
        if destination.len() != self.bytes.len() {
            return Err(RowStoreError::InvalidLease {
                reason: format!(
                    "staging destination has {} bytes, expected {}",
                    destination.len(),
                    self.bytes.len()
                ),
            });
        }
        destination.copy_from_slice(&self.bytes);
        self.validate_after_upload()
    }
}

impl Drop for RowLease {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        self.inner.return_staging_locked(&mut state, bytes);
        state.active_leases = state.active_leases.saturating_sub(1);
        self.inner.cv.notify_all();
    }
}

/// Handle for a queued row read.  Dropping it invalidates an unfinished or
/// completed result and returns its staging buffer once the reader is safe.
pub struct RowTicket {
    inner: Weak<RowStoreInner>,
    id: u64,
    epoch: u64,
    canceled: Arc<AtomicBool>,
    consumed: AtomicBool,
}

impl fmt::Debug for RowTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowTicket")
            .field("id", &self.id)
            .field("epoch", &self.epoch)
            .field("canceled", &self.canceled.load(Ordering::Acquire))
            .field("consumed", &self.consumed.load(Ordering::Acquire))
            .finish()
    }
}

impl RowTicket {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn row_ids(&self) -> Result<Vec<u64>, RowStoreError> {
        let inner = self.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        let state = inner.state.lock().expect("row store state mutex poisoned");
        let ticket = state.tickets.get(&self.id).ok_or_else(|| {
            if self.canceled.load(Ordering::Acquire) {
                RowStoreError::Canceled
            } else {
                RowStoreError::UnknownTicket(self.id)
            }
        })?;
        Ok(ticket.ids.clone())
    }

    pub fn cancel(&self) -> Result<(), RowStoreError> {
        if self.consumed.load(Ordering::Acquire) {
            return Err(RowStoreError::AlreadyConsumed(self.id));
        }
        self.canceled.store(true, Ordering::Release);
        let inner = self.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        inner.cancel_ticket(self.id)
    }
}

impl Drop for RowTicket {
    fn drop(&mut self) {
        if !self.consumed.load(Ordering::Acquire) {
            self.canceled.store(true, Ordering::Release);
            if let Some(inner) = self.inner.upgrade() {
                let _ = inner.cancel_ticket(self.id);
            }
        }
    }
}

/// Model-owned bounded row reader/cache.
pub struct RowStore {
    inner: Arc<RowStoreInner>,
    worker: Option<JoinHandle<()>>,
}

impl fmt::Debug for RowStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.inner.report();
        f.debug_struct("RowStore")
            .field("shards", &self.inner.shard_count)
            .field("rows_per_shard", &self.inner.rows_per_shard)
            .field("report", &report)
            .finish()
    }
}

impl RowStore {
    /// Open production row storage from sealed shard descriptors.
    ///
    /// All descriptors must refer to one immutable source identity, share one
    /// `[rows_per_shard, width]` shape and dtype, either BF16 (`2 * width`-byte
    /// rows), Q8F16 (`width / 32` 34-byte blocks) or Q4LM (`width / 32` 18-byte
    /// blocks), and be exactly
    /// `rows * encoded_row_bytes` bytes.  Global rows at or past `valid_rows`
    /// are padding: requests for them are refused, so padding can never be
    /// returned as an embedding.  `name` labels the reader thread.
    pub fn new(
        name: &str,
        descriptors: Vec<SourceRangeDescriptor>,
        valid_rows: u64,
    ) -> Result<Self, RowStoreError> {
        Self::with_staging_rows(name, descriptors, valid_rows, 0, true)
    }

    /// [`RowStore::new`] whose staging buffers (and coalesced reads) hold at
    /// least `rows` decoded rows, so a prefetch of `rows` row ids is
    /// admissible.  Never smaller than the default budget.  `wide_readers`
    /// lets a large request scale its concurrent reads up to
    /// `MAX_PARALLEL_READERS`; without it every request keeps
    /// `PARALLEL_READERS`.
    pub fn with_staging_rows(
        name: &str,
        descriptors: Vec<SourceRangeDescriptor>,
        valid_rows: u64,
        rows: usize,
        wide_readers: bool,
    ) -> Result<Self, RowStoreError> {
        let descriptors: Arc<[SourceRangeDescriptor]> = descriptors.into();
        let layout = validate_descriptors(&descriptors, valid_rows)?;
        let source: Arc<dyn PositionalRowSource> = Arc::new(DescriptorRowSource {
            descriptors: descriptors.clone(),
        });
        Self::spawn(name, source, descriptors, layout, rows, wide_readers)
    }

    /// A row store over a rank-ordered PLE table ([`HotRowStore::open_table`]):
    /// geometry and encoding come from the store, every fetch is served from
    /// its pack (the store is attached as the hot store, which also binds this
    /// store's [`FetchActivity`]), and the positional fallback source reads
    /// single rows at their pack offsets.  `valid_rows`, `staging_rows` and
    /// `wide_readers` are as for [`RowStore::with_staging_rows`].
    pub fn from_table_store(
        name: &str,
        store: Arc<HotRowStore>,
        valid_rows: u64,
        staging_rows: usize,
        wide_readers: bool,
    ) -> Result<Self, RowStoreError> {
        let invalid = |reason: String| RowStoreError::Descriptor { index: 0, reason };
        let rows_per_shard = store.rows_per_shard();
        let padded_rows = store.padded_rows();
        if rows_per_shard == 0 || padded_rows == 0 || padded_rows % rows_per_shard as u64 != 0 {
            return Err(invalid(format!(
                "table of {padded_rows} rows is not a whole number of shards of {rows_per_shard}"
            )));
        }
        let shard_count = usize::try_from(padded_rows / rows_per_shard as u64)
            .map_err(|_| invalid("table shard count overflows".to_string()))?;
        if valid_rows == 0 || valid_rows > padded_rows {
            return Err(invalid(format!(
                "valid rows {valid_rows} outside 1..={padded_rows} table rows"
            )));
        }
        let encoding = store.encoding();
        let row_width = store.row_width();
        if encoding.checked_encoded_row_bytes(row_width) != Some(store.encoded_row_bytes()) {
            return Err(invalid(format!(
                "table rows of {row_width} {} values are not {} bytes",
                encoding.dtype_name(),
                store.encoded_row_bytes()
            )));
        }
        let layout = Layout {
            shard_count,
            rows_per_shard,
            rows_per_page: ROWS_PER_PAGE,
            row_width,
            valid_rows,
            encoding,
        };
        let source: Arc<dyn PositionalRowSource> = Arc::new(TableRowSource {
            store: store.clone(),
            shard_count,
            rows_per_shard: rows_per_shard as u64,
            row_bytes: store.encoded_row_bytes(),
        });
        let rows = Self::spawn(
            name,
            source,
            Vec::new().into(),
            layout,
            staging_rows,
            wide_readers,
        )?;
        rows.attach_hot_store(store).map_err(invalid)?;
        Ok(rows)
    }

    fn spawn(
        name: &str,
        source: Arc<dyn PositionalRowSource>,
        descriptors: Arc<[SourceRangeDescriptor]>,
        layout: Layout,
        staging_rows: usize,
        wide_readers: bool,
    ) -> Result<Self, RowStoreError> {
        let decoded_row_bytes = layout.row_width * 2;
        let staging_bytes = staging_rows
            .checked_mul(decoded_row_bytes)
            .ok_or_else(|| RowStoreError::Descriptor {
                index: 0,
                reason: format!("{staging_rows} staging rows overflow"),
            })?
            .max((STAGING_BUDGET_BYTES / decoded_row_bytes) * decoded_row_bytes);
        let inner = Arc::new(RowStoreInner {
            source,
            stopped: AtomicBool::new(false),
            shard_count: layout.shard_count,
            _descriptors: descriptors,
            valid_rows: layout.valid_rows,
            padded_rows: layout.shard_count as u64 * layout.rows_per_shard as u64,
            encoding: layout.encoding,
            encoded_row_bytes: layout.encoding.encoded_row_bytes(layout.row_width),
            decoded_row_bytes,
            staging_bytes,
            max_readers: if wide_readers {
                MAX_PARALLEL_READERS
            } else {
                PARALLEL_READERS
            },
            rows_per_shard: layout.rows_per_shard,
            rows_per_page: layout.rows_per_page,
            current_epoch: AtomicU64::new(0),
            epoch_started: AtomicBool::new(false),
            state: Mutex::new(RowStoreState::new(staging_bytes)),
            cv: Condvar::new(),
            hot_store: OnceLock::new(),
            activity: Arc::new(FetchActivity::new()),
        });
        let weak = Arc::downgrade(&inner);
        let worker = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || worker_loop(weak))
            .map_err(|error| RowStoreError::Descriptor {
                index: 0,
                reason: format!("cannot start bounded reader: {error}"),
            })?;
        Ok(Self {
            inner,
            worker: Some(worker),
        })
    }

    pub fn shard_count(&self) -> usize {
        self.inner.shard_count
    }

    pub fn rows_per_shard(&self) -> usize {
        self.inner.rows_per_shard
    }

    pub const fn rows_per_page(&self) -> usize {
        ROWS_PER_PAGE
    }

    /// Decoded BF16 bytes of one row, as leases publish it.
    pub fn decoded_row_bytes(&self) -> usize {
        self.inner.decoded_row_bytes
    }

    /// Most rows one prefetch may request: one staging buffer's worth.
    pub fn max_rows_per_prefetch(&self) -> usize {
        self.inner.staging_bytes / self.inner.decoded_row_bytes
    }

    pub fn current_epoch(&self) -> u64 {
        self.inner.current_epoch.load(Ordering::Acquire)
    }

    /// Start a model/request epoch and invalidate every prior ticket.
    ///
    /// Epochs are single-use and strictly increasing after the first epoch.
    /// The first caller may choose epoch zero.  Invalidated queued tickets are
    /// removed immediately; an in-flight source read is allowed to finish
    /// under cancellation but can never publish a lease.
    pub fn begin_epoch(&self, epoch: u64) -> Result<(), RowStoreError> {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        let current = self.current_epoch();
        let started = self.inner.epoch_started.load(Ordering::Acquire);
        if started && epoch <= current {
            return Err(RowStoreError::EpochNotMonotonic {
                requested: epoch,
                current,
            });
        }
        self.inner.current_epoch.store(epoch, Ordering::Release);
        self.inner.epoch_started.store(true, Ordering::Release);
        invalidate_tickets_locked(&self.inner, &mut state);
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Invalidate the current request epoch, drain all queued/readers/leases,
    /// then resume the same model-owned cache for the next epoch.  Immutable
    /// cached pages survive; request tickets and output buffers do not.
    pub fn reset_epoch(&self, timeout: Duration) -> Result<u64, RowStoreError> {
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        let next = {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            let current = self.current_epoch();
            let next = current
                .checked_add(1)
                .ok_or(RowStoreError::EpochNotMonotonic {
                    requested: u64::MAX,
                    current: u64::MAX,
                })?;
            let started = self.inner.epoch_started.load(Ordering::Acquire);
            if started && next <= current {
                return Err(RowStoreError::EpochNotMonotonic {
                    requested: next,
                    current,
                });
            }
            self.inner.current_epoch.store(next, Ordering::Release);
            self.inner.epoch_started.store(true, Ordering::Release);
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
            next
        };
        self.quiesce(timeout)?;
        self.resume()?;
        Ok(next)
    }

    /// Enqueue a bounded prefetch of `row_ids`, in the order the lease will
    /// publish them.  Every id is checked against the valid rows before the
    /// ticket exists.
    pub fn prefetch(&self, epoch: u64, row_ids: Vec<u64>) -> Result<RowTicket, RowStoreError> {
        self.enqueue(epoch, row_ids, false)
    }

    /// Best-effort: read `row_ids`' pages into the cache ahead of a request
    /// that will need them (e.g. speculative tokens known before their
    /// forward starts). There is no lease; a later epoch drops it if it has
    /// not started.
    pub fn warm(&self, row_ids: Vec<u64>) -> Result<(), RowStoreError> {
        let ticket = self.enqueue(self.current_epoch(), row_ids, true)?;
        // The reader owns a warm ticket's lifetime.
        ticket.consumed.store(true, Ordering::Release);
        Ok(())
    }

    fn enqueue(
        &self,
        epoch: u64,
        row_ids: Vec<u64>,
        warm: bool,
    ) -> Result<RowTicket, RowStoreError> {
        let max_rows = self.max_rows_per_prefetch();
        if row_ids.len() > max_rows {
            return Err(RowStoreError::RequestTooLarge {
                rows: row_ids.len(),
                max_rows,
            });
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(RowStoreError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        let (locations, pages) = self.plan_rows(&row_ids)?;
        let ids = row_ids;
        let required_bytes = locations.len() * self.inner.decoded_row_bytes;

        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if !state.accepting || state.stop {
            return Err(RowStoreError::Quiescing);
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(RowStoreError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        if state.tickets.len() >= READER_QUEUE_CAPACITY {
            return Err(RowStoreError::QueueFull);
        }
        let id = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let canceled = Arc::new(AtomicBool::new(false));
        state.tickets.insert(
            id,
            TicketState {
                epoch,
                canceled: canceled.clone(),
                ids,
                locations,
                pages,
                required_bytes,
                output: None,
                status: TicketStatus::Pending,
                warm,
            },
        );
        state.queue.push_back(id);
        self.inner.cv.notify_one();
        Ok(RowTicket {
            inner: Arc::downgrade(&self.inner),
            id,
            epoch,
            canceled,
            consumed: AtomicBool::new(false),
        })
    }

    /// Wait for a ticket's completed rows.  This is the layer-1 consumption
    /// boundary; all source reads and cache fills are complete on return.
    pub fn wait_completed_lease(&self, ticket: &RowTicket) -> Result<RowLease, RowStoreError> {
        let inner = ticket.inner.upgrade().ok_or(RowStoreError::WorkerStopped)?;
        if !Arc::ptr_eq(&inner, &self.inner) {
            return Err(RowStoreError::InvalidLease {
                reason: "ticket belongs to a different row store".to_string(),
            });
        }
        if ticket
            .consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(RowStoreError::AlreadyConsumed(ticket.id));
        }
        let mut state = inner.state.lock().expect("row store state mutex poisoned");
        loop {
            let done = match state.tickets.get(&ticket.id) {
                Some(ticket_state) => !matches!(ticket_state.status, TicketStatus::Pending),
                None => {
                    return Err(if ticket.canceled.load(Ordering::Acquire) {
                        RowStoreError::Canceled
                    } else {
                        RowStoreError::UnknownTicket(ticket.id)
                    })
                }
            };
            if done {
                break;
            }
            if state.stop && state.active_readers == 0 {
                return Err(RowStoreError::WorkerStopped);
            }
            state = inner
                .cv
                .wait(state)
                .expect("row store state mutex poisoned");
        }
        let ticket_state = state
            .tickets
            .remove(&ticket.id)
            .ok_or(RowStoreError::UnknownTicket(ticket.id))?;
        let ticket_canceled = ticket_state.canceled.load(Ordering::Acquire)
            || ticket_state.epoch != inner.current_epoch.load(Ordering::Acquire);
        match ticket_state.status {
            TicketStatus::Pending => unreachable!("completed wait observed pending ticket"),
            TicketStatus::Completed(Ok(())) if !ticket_canceled => {
                let bytes = ticket_state
                    .output
                    .ok_or_else(|| RowStoreError::InvalidLease {
                        reason: "completed ticket has no staging buffer".to_string(),
                    })?;
                state.active_leases += 1;
                let lease_inner = Arc::clone(&inner);
                drop(state);
                Ok(RowLease {
                    inner: lease_inner,
                    epoch: ticket_state.epoch,
                    ids: ticket_state.ids,
                    bytes,
                })
            }
            TicketStatus::Completed(Ok(())) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(RowStoreError::Canceled)
            }
            TicketStatus::Completed(Err(error)) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(error)
            }
        }
    }

    /// True when `ticket` belongs to this store, is still live (not canceled,
    /// consumed, or invalidated by an epoch change/quiesce) in the current
    /// epoch, and was enqueued for exactly `row_ids` in that order.
    pub fn ticket_matches(&self, ticket: &RowTicket, row_ids: &[u64]) -> bool {
        let Some(inner) = ticket.inner.upgrade() else {
            return false;
        };
        if !Arc::ptr_eq(&inner, &self.inner)
            || ticket.consumed.load(Ordering::Acquire)
            || ticket.canceled.load(Ordering::Acquire)
        {
            return false;
        }
        let state = inner.state.lock().expect("row store state mutex poisoned");
        state.accepting
            && !state.stop
            && ticket.epoch == inner.current_epoch.load(Ordering::Acquire)
            && state.tickets.get(&ticket.id).is_some_and(|entry| {
                !entry.canceled.load(Ordering::Acquire)
                    && entry.epoch == ticket.epoch
                    && entry.ids == row_ids
            })
    }

    pub fn cancel(&self, ticket: &RowTicket) -> Result<(), RowStoreError> {
        ticket.cancel()
    }
    /// Stop accepting tickets, invalidate pending work, and wait for readers
    /// and leases to drain.  A timeout returns an unload proof showing exactly
    /// which owner is still live; it never pretends quiescence succeeded.
    pub fn quiesce(&self, timeout: Duration) -> Result<RowQuiesceReport, RowStoreError> {
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let deadline = Instant::now() + timeout;
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(RowStoreError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next, wait) = self
                .inner
                .cv
                .wait_timeout(state, remaining)
                .expect("row store state mutex poisoned");
            state = next;
            if wait.timed_out() {
                return Err(RowStoreError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
        }
    }

    fn quiesce_blocking(&self) -> Result<RowQuiesceReport, RowStoreError> {
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            state = self
                .inner
                .cv
                .wait(state)
                .expect("row store state mutex poisoned");
        }
    }

    pub fn resume(&self) -> Result<(), RowStoreError> {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        if state.stop || self.inner.stopped.load(Ordering::Acquire) {
            return Err(RowStoreError::WorkerStopped);
        }
        state.accepting = true;
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Quiesce and join the model-owned reader with explicit blocking
    /// semantics.  Regular-file positional reads cannot be safely canceled,
    /// so this API never returns a timeout while a worker can still outlive
    /// its source owner.
    pub fn unload(mut self) -> Result<RowQuiesceReport, RowStoreError> {
        let report = self.quiesce_blocking()?;
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        {
            let mut state = self
                .inner
                .state
                .lock()
                .expect("row store state mutex poisoned");
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| RowStoreError::WorkerJoin)?;
        }
        Ok(report)
    }

    pub fn cache_stats(&self) -> RowCacheStats {
        self.inner.stats()
    }

    /// The store's fetch activity: every fetch holds it busy, so background
    /// readers of the same table can wait for idleness (see
    /// [`FetchActivity::wait_idle`]).
    pub fn activity(&self) -> Arc<FetchActivity> {
        self.inner.activity.clone()
    }

    /// Attach the PLE hot-row store.  Rows its pack holds are read from the
    /// pack (the model file's own encoded rows, re-laid out by rank) and
    /// decoded instead of read from the source; everything else takes the
    /// unchanged page path.  One store per row store, attached once; the
    /// caller stops and joins the store's worker before unloading this row
    /// store.  Errs (and attaches nothing) when the store's geometry or row
    /// encoding is not this table's.
    pub fn attach_hot_store(&self, store: Arc<HotRowStore>) -> Result<(), String> {
        let inner = &self.inner;
        if store.padded_rows() != inner.padded_rows
            || store.rows_per_shard() != inner.rows_per_shard
            || store.encoded_row_bytes() != inner.encoded_row_bytes
            || store.encoding() != inner.encoding
        {
            return Err(format!(
                "hot store geometry ({} rows, {} per shard, {} B/row, {}) does not match the row \
                 store ({} rows, {} per shard, {} B/row, {})",
                store.padded_rows(),
                store.rows_per_shard(),
                store.encoded_row_bytes(),
                store.encoding().dtype_name(),
                inner.padded_rows,
                inner.rows_per_shard,
                inner.encoded_row_bytes,
                inner.encoding.dtype_name(),
            ));
        }
        store.bind_activity(inner.activity.clone())?;
        inner
            .hot_store
            .set(store)
            .map_err(|_| "a hot store is already attached".to_string())
    }

    /// Drop all immutable userspace pages.  The source reader intentionally
    /// has no mmap or fd-pinning contract; dropping these pages is therefore a
    /// portable best-effort eviction of this cache, while the counters make
    /// kernel page-cache growth observable to the outer admission harness.
    pub fn evict_cached_pages(&self) -> usize {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("row store state mutex poisoned");
        let dropped = state.cache.clear();
        self.inner.cv.notify_all();
        dropped
    }

    pub fn locate_row(&self, row: u64) -> Result<RowLocation, RowStoreError> {
        self.inner.locate_row(row)
    }

    fn plan_rows(
        &self,
        row_ids: &[u64],
    ) -> Result<(Vec<RowLocation>, Vec<PageKey>), RowStoreError> {
        let mut locations = Vec::with_capacity(row_ids.len());
        let mut pages = Vec::with_capacity(row_ids.len());
        for &row in row_ids {
            let location = self.inner.locate_row(row)?;
            pages.push(location.page_key());
            locations.push(location);
        }
        pages.sort_unstable();
        pages.dedup();
        Ok((locations, pages))
    }
}

impl Drop for RowStore {
    fn drop(&mut self) {
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut state) = self.inner.state.lock() {
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Bound on draining readers and leases when a [`RowFetch`] aborts.
const FETCH_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Owns one request's rows until the request either commits or aborts.
///
/// [`RowFetch::begin`] starts a fresh epoch, [`RowFetch::prefetch`] enqueues
/// the rows, and [`RowFetch::wait`] turns the ticket into the lease.  Immutable
/// page-cache entries belong to the [`RowStore`] and survive an abort; this
/// guard only owns the exact ticket and completed lease of its epoch.  Dropping
/// it without [`RowFetch::complete`] aborts.
pub struct RowFetch<'a> {
    store: &'a RowStore,
    epoch: u64,
    ticket: Option<RowTicket>,
    lease: Option<RowLease>,
    armed: bool,
}

impl<'a> RowFetch<'a> {
    /// Start the next request epoch, invalidating every earlier ticket.
    pub fn begin(store: &'a RowStore) -> Result<Self, RowStoreError> {
        let current = store.current_epoch();
        let epoch = current
            .checked_add(1)
            .ok_or(RowStoreError::EpochNotMonotonic {
                requested: u64::MAX,
                current,
            })?;
        store.begin_epoch(epoch)?;
        Ok(Self {
            store,
            epoch,
            ticket: None,
            lease: None,
            armed: true,
        })
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Start a fetch of `row_ids`, adopting `retained` instead when it is a
    /// live ticket of this store's current epoch for exactly those rows (a
    /// lookahead issued after the previous forward committed).  Rows are a
    /// pure function of their ids, so the adopted lease is byte-identical to
    /// a fresh read; the worker simply built it under the previous forward.
    /// Anything else drops `retained` (returning its staging buffer) and
    /// starts a fresh epoch exactly like [`Self::begin`] + [`Self::prefetch`].
    pub fn begin_or_adopt(
        store: &'a RowStore,
        retained: Option<RowTicket>,
        row_ids: Vec<u64>,
    ) -> Result<Self, RowStoreError> {
        if let Some(ticket) = retained {
            if store.ticket_matches(&ticket, &row_ids) {
                return Ok(Self {
                    store,
                    epoch: ticket.epoch,
                    ticket: Some(ticket),
                    lease: None,
                    armed: true,
                });
            }
        }
        let mut fetch = Self::begin(store)?;
        fetch.prefetch(row_ids)?;
        Ok(fetch)
    }

    /// Enqueue this request's rows.  Call once, as soon as the ids are known.
    pub fn prefetch(&mut self, row_ids: Vec<u64>) -> Result<(), RowStoreError> {
        debug_assert!(self.ticket.is_none() && self.lease.is_none());
        self.ticket = Some(self.store.prefetch(self.epoch, row_ids)?);
        Ok(())
    }

    /// The completed lease, if [`Self::wait`] has already produced it.
    pub fn lease(&self) -> Option<&RowLease> {
        self.lease.as_ref()
    }

    /// Block until the prefetched rows are complete and return the lease.
    pub fn wait(&mut self) -> Result<&RowLease, RowStoreError> {
        if self.lease.is_none() {
            let ticket = self
                .ticket
                .as_ref()
                .ok_or_else(|| RowStoreError::InvalidLease {
                    reason: "rows awaited before they were prefetched".to_string(),
                })?;
            let lease = self.store.wait_completed_lease(ticket)?;
            // `wait_completed_lease` marks the ticket consumed.  Drop that
            // handle before retaining the lease so the only live owner is
            // explicit.
            self.ticket = None;
            self.lease = Some(lease);
        }
        Ok(self.lease.as_ref().expect("lease installed above"))
    }

    /// Release the ticket and lease after the request committed.
    pub fn complete(&mut self) {
        self.armed = false;
        drop(self.ticket.take());
        drop(self.lease.take());
    }

    /// Cancel outstanding work and drain the store into a fresh epoch.  Returns
    /// that epoch, or every cleanup failure joined into one message.
    pub fn abort(&mut self) -> Result<u64, String> {
        if !self.armed {
            return Ok(self.store.current_epoch());
        }
        // Disarm first: if cleanup itself reports an error, Drop must not
        // issue a second reset against a later epoch.
        self.armed = false;
        let mut errors = Vec::new();
        if let Some(ticket) = self.ticket.take() {
            if let Err(error) = self.store.cancel(&ticket) {
                // wait_completed_lease marks a ticket consumed before waiting;
                // source-read and cancellation errors therefore legitimately
                // report AlreadyConsumed during abort.
                if !matches!(
                    error,
                    RowStoreError::AlreadyConsumed(_)
                        | RowStoreError::Canceled
                        | RowStoreError::UnknownTicket(_)
                ) {
                    errors.push(format!("cancel epoch {} ticket: {error}", self.epoch));
                }
            }
            drop(ticket);
        }
        // A lease holds one of the bounded staging buffers.  It must be
        // returned before reset_epoch waits for readers/leases to drain.
        drop(self.lease.take());
        let next_epoch = match self.store.reset_epoch(FETCH_CLEANUP_TIMEOUT) {
            Ok(next) => Some(next),
            Err(error) => {
                errors.push(format!("drain epoch {}: {error}", self.epoch));
                None
            }
        };
        if errors.is_empty() {
            Ok(next_epoch.expect("successful reset returns its next epoch"))
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for RowFetch<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.abort();
        }
    }
}

struct RowStoreInner {
    source: Arc<dyn PositionalRowSource>,
    shard_count: usize,
    _descriptors: Arc<[SourceRangeDescriptor]>,
    valid_rows: u64,
    padded_rows: u64,
    /// Stored row representation, fixed by the sealed shard dtype.
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    /// BF16 width of one published row.
    decoded_row_bytes: usize,
    /// Whole-row size of one staging buffer / coalesced read.
    staging_bytes: usize,
    /// Reader ceiling of one request's scattered reads.
    max_readers: usize,
    rows_per_shard: usize,
    rows_per_page: usize,
    epoch_started: AtomicBool,
    current_epoch: AtomicU64,
    stopped: AtomicBool,
    state: Mutex<RowStoreState>,
    cv: Condvar,
    /// PLE hot-row store consulted before the page cache (see
    /// [`RowStore::attach_hot_store`]).
    hot_store: OnceLock<Arc<HotRowStore>>,
    /// Fetch idle/busy tracking shared with background readers (see
    /// [`RowStore::activity`]).
    activity: Arc<FetchActivity>,
}

impl RowStoreInner {
    fn locate_row(&self, row: u64) -> Result<RowLocation, RowStoreError> {
        if row >= self.padded_rows {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row,
                padded_rows: self.padded_rows,
            });
        }
        if row >= self.valid_rows {
            return Err(RowStoreError::PaddingRow {
                row,
                valid_rows: self.valid_rows,
            });
        }
        let shard_rows = self.rows_per_shard as u64;
        let shard = usize::try_from(row / shard_rows).map_err(|_| RowStoreError::InvalidLease {
            reason: "shard index overflow".to_string(),
        })?;
        let local_row =
            usize::try_from(row % shard_rows).map_err(|_| RowStoreError::InvalidLease {
                reason: "local row overflow".to_string(),
            })?;
        if shard >= self.shard_count {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row,
                padded_rows: self.padded_rows,
            });
        }
        let page = local_row / self.rows_per_page;
        let row_in_page = local_row % self.rows_per_page;
        let page_byte_offset =
            row_in_page
                .checked_mul(self.encoded_row_bytes)
                .ok_or_else(|| RowStoreError::InvalidLease {
                    reason: "page byte offset overflow".to_string(),
                })?;
        Ok(RowLocation {
            global_row: row,
            shard,
            local_row,
            page,
            page_byte_offset,
        })
    }

    fn validate_lease_epoch(&self, lease_epoch: u64) -> Result<(), RowStoreError> {
        let current = self.current_epoch.load(Ordering::Acquire);
        if self.stopped.load(Ordering::Acquire) || current != lease_epoch {
            Err(RowStoreError::LeaseStale {
                lease_epoch,
                current_epoch: current,
            })
        } else {
            Ok(())
        }
    }

    fn report(&self) -> RowQuiesceReport {
        let state = self.state.lock().expect("row store state mutex poisoned");
        self.report_locked(&state)
    }

    fn report_locked(&self, state: &RowStoreState) -> RowQuiesceReport {
        RowQuiesceReport {
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            cache_bytes: state.cache.resident_bytes,
            cache_pages: state.cache.pages.len(),
        }
    }

    fn stats(&self) -> RowCacheStats {
        let state = self.state.lock().expect("row store state mutex poisoned");
        RowCacheStats {
            capacity_bytes: PAGE_CACHE_BYTES,
            resident_bytes: state.cache.resident_bytes,
            resident_pages: state.cache.pages.len(),
            cache_hits: state.cache.hits,
            cache_misses: state.cache.misses,
            reads: state.cache.reads,
            coalesced_reads: state.cache.coalesced_reads,
            read_bytes: state.cache.read_bytes,
            evictions: state.cache.evictions,
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            staging_high_water: state.staging_high_water,
            hot_rows: self.hot_store.get().map_or(0, |store| store.hot_rows()),
            hot_pages: self.hot_store.get().map_or(0, |store| store.hot_pages()),
            hot_reads: self.hot_store.get().map_or(0, |store| store.hot_reads()),
        }
    }

    fn take_staging_locked(&self, state: &mut RowStoreState) -> Option<Vec<u8>> {
        let slot = state.staging_pool.iter().position(Option::is_some)?;
        let buffer = state.staging_pool[slot].take();
        state.staging_in_use += 1;
        state.staging_high_water = state.staging_high_water.max(state.staging_in_use);
        buffer
    }

    fn return_staging_locked(&self, state: &mut RowStoreState, mut buffer: Vec<u8>) {
        buffer.clear();
        if let Some(slot) = state.staging_pool.iter().position(Option::is_none) {
            state.staging_pool[slot] = Some(buffer);
        }
        state.staging_in_use = state.staging_in_use.saturating_sub(1);
    }

    fn cancel_ticket(&self, id: u64) -> Result<(), RowStoreError> {
        let mut state = self.state.lock().expect("row store state mutex poisoned");
        let Some(ticket) = state.tickets.get(&id) else {
            return Ok(());
        };
        ticket.canceled.store(true, Ordering::Release);

        let queued = state.queue.iter().position(|queued_id| *queued_id == id);
        let completed = matches!(
            state.tickets.get(&id).map(|ticket| &ticket.status),
            Some(TicketStatus::Completed(_))
        );
        if let Some(position) = queued {
            state.queue.remove(position);
        }
        if queued.is_some() || completed {
            if let Some(ticket) = state.tickets.remove(&id) {
                if let Some(buffer) = ticket.output {
                    self.return_staging_locked(&mut state, buffer);
                }
            }
        }
        self.cv.notify_all();
        Ok(())
    }
    fn process_ticket(&self, id: u64, read_buffer: Vec<u8>) -> ProcessedTicket {
        let (epoch, canceled, pages, locations, output) = {
            let mut state = self.state.lock().expect("row store state mutex poisoned");
            let Some(ticket) = state.tickets.get_mut(&id) else {
                return ProcessedTicket {
                    output: None,
                    read_buffer,
                    result: Err(RowStoreError::Canceled),
                    pending: PendingPages::new(),
                };
            };
            (
                ticket.epoch,
                ticket.canceled.clone(),
                ticket.pages.clone(),
                ticket.locations.clone(),
                ticket.output.take(),
            )
        };
        let mut output = output.unwrap_or_default();
        let mut read_buffer = read_buffer;
        if canceled.load(Ordering::Acquire) {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(RowStoreError::Canceled),
                pending: PendingPages::new(),
            };
        }
        let current = self.current_epoch.load(Ordering::Acquire);
        if epoch != current {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(RowStoreError::EpochMismatch {
                    requested: epoch,
                    current,
                }),
                pending: PendingPages::new(),
            };
        }
        let mut pending = PendingPages::new();
        let result = self.fill_output(
            &locations,
            &pages,
            &mut output,
            &mut read_buffer,
            &canceled,
            &mut pending,
        );
        ProcessedTicket {
            output: Some(output),
            read_buffer,
            result,
            pending,
        }
    }

    /// Fill the final bounded lease directly while each page group is
    /// available.  Requested page sets may be much larger than the cache;
    /// no second pass assumes all pages remain resident.
    ///
    /// `pages` MUST be strictly ascending (sorted, deduplicated), as
    /// `RowStore::plan_rows` builds it.  A request of several page groups
    /// reads and decodes them on parallel readers and leaves the pages it read
    /// in `pending` for the caller to cache after the ticket completes (also
    /// when it returns `Canceled`); a single group is read, decoded, and
    /// cached inline.
    fn fill_output(
        &self,
        locations: &[RowLocation],
        pages: &[PageKey],
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
        pending: &mut PendingPages,
    ) -> Result<(), RowStoreError> {
        // Every PLE fetch (a prefill chunk or a decode step) marks the row
        // store busy, so background readers of the table (the hot store's
        // fill, the rows prewarm) only run while no request is being served.
        let _busy = self.activity.busy();
        if pages.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RowStoreError::InvalidLease {
                reason: "page plan is not strictly ascending".to_string(),
            });
        }
        let expected = locations
            .len()
            .checked_mul(self.decoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "staging size overflow".to_string(),
            })?;
        if output.len() != expected {
            return Err(RowStoreError::InvalidLease {
                reason: "staging size does not match row plan".to_string(),
            });
        }
        let mut requested: HashMap<PageKey, Vec<RowCopy>> = HashMap::new();
        for (index, location) in locations.iter().copied().enumerate() {
            requested
                .entry(location.page_key())
                .or_default()
                .push(RowCopy {
                    output_offset: index * self.decoded_row_bytes,
                    page_offset: location.page_byte_offset,
                });
        }

        // Pages whose every requested row the hot-row pack holds are read from
        // the pack (rank-ordered rows, through the page cache); a page with any
        // other requested row takes the unchanged page path, so a partially
        // hot page is read and decoded exactly as without the pack.
        let mut trace = FetchTrace::start(locations.len(), pages.len());
        let store: Option<&HotRowStore> = self
            .hot_store
            .get()
            .map(|store| &**store)
            .filter(|store| store.is_serving());
        let remaining_pages;
        let mut hot = HotPlan::default();
        let pages = match store {
            Some(store) => {
                (hot, remaining_pages) = self.plan_hot_reads(store, pages, &requested);
                remaining_pages.as_slice()
            }
            None => pages,
        };
        if let Some(trace) = trace.as_mut() {
            trace.mark_classified(&hot);
        }

        let mut missing = Vec::new();
        for &page in pages {
            if canceled.load(Ordering::Acquire) {
                return Err(RowStoreError::Canceled);
            }
            let mut state = self.state.lock().expect("row store state mutex poisoned");
            if let Some(cached) = state.cache.pages.get(&page) {
                copy_requested_rows(
                    self.encoding,
                    self.encoded_row_bytes,
                    self.decoded_row_bytes,
                    page,
                    &cached.bytes,
                    requested.get(&page).map(Vec::as_slice).unwrap_or(&[]),
                    output,
                )?;
                state.cache.touch(page);
                state.cache.hits += 1;
            } else {
                state.cache.misses += 1;
                missing.push(page);
            }
        }

        let groups = self.coalesce_pages(&missing)?;
        if let Some(trace) = trace.as_mut() {
            trace.mark_cache_scanned(pages.len() - missing.len(), missing.len(), groups.len());
        }
        let result = if groups.len() + hot.runs.len() > 1 {
            // Rows are drawn at random over the table, so the groups are
            // scattered single-page reads: issued serially each one waits a
            // full device round trip (~64 reads = 10-15 ms per few-token
            // request on NVMe), issued together the device overlaps them.
            // Each reader decodes its reads' requested rows into the output
            // right after the read, so assembly overlaps the reads instead of
            // following them serially.  The readers take the source groups
            // first (ascending file offset), then the pack reads.
            let readers = reader_count(groups.len() + hot.runs.len(), self.max_readers);
            let work = ReadWork {
                groups: &groups,
                group_cursor: AtomicUsize::new(0),
                hot: &hot,
                hot_cursor: AtomicUsize::new(0),
                store,
            };
            let work = &work;
            let shared = SharedOutput::new(&mut *output);
            let shared = &shared;
            let requested = &requested;
            let started = Instant::now();
            let shares = std::thread::scope(|scope| -> Result<Vec<ReaderShare>, RowStoreError> {
                let handles: Vec<_> = (0..readers)
                    .map(|_| scope.spawn(move || self.run_reader(work, requested, shared, started)))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("row reader thread panicked"))
                    .collect()
            })?;
            let read_bytes: usize = shares
                .iter()
                .flat_map(|share| share.pages.iter())
                .map(|(_, bytes)| bytes.len())
                .sum();
            {
                let mut state = self.state.lock().expect("row store state mutex poisoned");
                state.cache.reads += groups.len() as u64;
                state.cache.coalesced_reads += groups.len() as u64;
                state.cache.read_bytes += read_bytes as u64;
            }
            if let Some(store) = store {
                store.note_served(hot.rows, hot.pages, hot.runs.len() as u64);
            }
            if let Some(trace) = trace.as_mut() {
                trace.parallel(started, readers, &shares);
            }
            // Pages are cached even under cancellation: the reads are done
            // and a later request (or the warm-up's owner) wants them cached.
            // The worker inserts them once this ticket is completed.
            *pending = shares.into_iter().map(|share| share.pages).collect();
            if canceled.load(Ordering::Acquire) {
                Err(RowStoreError::Canceled)
            } else {
                Ok(())
            }
        } else {
            let started = Instant::now();
            let result = match (groups.first(), hot.runs.first(), store) {
                (Some(group), _, _) => {
                    self.read_group(group, &requested, output, read_staging, canceled)
                }
                (None, Some(run), Some(store)) => {
                    let shared = SharedOutput::new(&mut *output);
                    let read = self.read_hot_run(store, &hot, run, &shared, read_staging);
                    if read.is_ok() {
                        store.note_served(hot.rows, hot.pages, 1);
                    }
                    read
                }
                _ => Ok(()),
            };
            if let Some(trace) = trace.as_mut() {
                trace.inline_reads(started);
            }
            result
        };
        if let Some(trace) = trace {
            trace.finish();
        }
        result
    }

    /// One reader of a multi-read request: take the next source group from
    /// the shared cursor (groups ascend by file offset, so the reads in flight
    /// stay clustered: ~1.08x the IOPS of per-reader spans on the Halo host's
    /// DRAM-less NVMe), read it and decode its requested rows straight into
    /// the shared output; once the groups are exhausted, do the same for the
    /// pack reads.  Returns the source pages it read for
    /// [`Self::publish_pending`].  A group of one page hands its read buffer
    /// over without a copy.
    fn run_reader(
        &self,
        work: &ReadWork<'_>,
        requested: &HashMap<PageKey, Vec<RowCopy>>,
        output: &SharedOutput<'_>,
        started: Instant,
    ) -> Result<ReaderShare, RowStoreError> {
        let mut share = ReaderShare::default();
        loop {
            let index = work.group_cursor.fetch_add(1, Ordering::Relaxed);
            let Some(group) = work.groups.get(index) else {
                break;
            };
            let mut bytes = Vec::new();
            self.read_group_bytes(group, &mut bytes)?;
            let single = group.len() == 1;
            let mut offset = 0usize;
            for &page in group {
                let length = self.page_len(page)?;
                let page_bytes = &bytes[offset..offset + length];
                if let Some(rows) = requested.get(&page) {
                    copy_requested_rows_shared(
                        self.encoding,
                        self.encoded_row_bytes,
                        self.decoded_row_bytes,
                        page,
                        page_bytes,
                        rows,
                        output,
                    )?;
                }
                if !single {
                    share.pages.push((page, page_bytes.to_vec()));
                }
                offset += length;
            }
            if single {
                share.pages.push((group[0], bytes));
            }
        }
        share.groups_done = started.elapsed();
        if let Some(store) = work.store {
            let mut buffer = Vec::new();
            loop {
                let index = work.hot_cursor.fetch_add(1, Ordering::Relaxed);
                let Some(run) = work.hot.runs.get(index) else {
                    break;
                };
                self.read_hot_run(store, work.hot, run, output, &mut buffer)?;
            }
        }
        Ok(share)
    }

    /// Partition `pages` (ascending) into the pages whose every requested row
    /// `store` holds, planned as pack reads, and the rest (ascending).
    fn plan_hot_reads(
        &self,
        store: &HotRowStore,
        pages: &[PageKey],
        requested: &HashMap<PageKey, Vec<RowCopy>>,
    ) -> (HotPlan, Vec<PageKey>) {
        let mut plan = HotPlan::default();
        let mut rest = Vec::with_capacity(pages.len());
        for &page in pages {
            let start = plan.items.len();
            let first_row = self.first_row_of(page);
            let held = requested
                .get(&page)
                .filter(|copies| !copies.is_empty())
                .is_some_and(|copies| {
                    copies.iter().all(|copy| {
                        let row = first_row + (copy.page_offset / self.encoded_row_bytes) as u64;
                        store
                            .pack_offset(row)
                            .map(|pack_offset| {
                                plan.items.push(HotItem {
                                    pack_offset,
                                    output_offset: copy.output_offset,
                                })
                            })
                            .is_some()
                    })
                });
            if held {
                plan.pages += 1;
            } else {
                plan.items.truncate(start);
                rest.push(page);
            }
        }
        plan.rows = plan.items.len() as u64;
        plan.items
            .sort_unstable_by_key(|item| (item.pack_offset, item.output_offset));
        // One read per run of consecutive pack rows (a row requested twice
        // is read once), bounded so one read stays a few OS pages.
        let row_bytes = self.encoded_row_bytes as u64;
        let mut index = 0;
        while index < plan.items.len() {
            let start = index;
            let offset = plan.items[index].pack_offset;
            let mut end = offset + row_bytes;
            index += 1;
            while let Some(item) = plan.items.get(index) {
                if item.pack_offset + row_bytes == end {
                    index += 1;
                } else if item.pack_offset == end && end + row_bytes - offset <= HOT_RUN_MAX_BYTES {
                    end += row_bytes;
                    index += 1;
                } else {
                    break;
                }
            }
            plan.runs.push(HotRun {
                offset,
                len: (end - offset) as usize,
                items: start..index,
            });
        }
        (plan, rest)
    }

    /// Read one run of pack rows and decode its requested rows into `output`.
    fn read_hot_run(
        &self,
        store: &HotRowStore,
        plan: &HotPlan,
        run: &HotRun,
        output: &SharedOutput<'_>,
        buffer: &mut Vec<u8>,
    ) -> Result<(), RowStoreError> {
        buffer.resize(run.len, 0);
        store
            .read_pack(run.offset, buffer)
            .map_err(RowStoreError::Source)?;
        for item in &plan.items[run.items.clone()] {
            let at = (item.pack_offset - run.offset) as usize;
            let end = item
                .output_offset
                .checked_add(self.decoded_row_bytes)
                .filter(|&end| end <= output.len())
                .ok_or_else(|| RowStoreError::InvalidLease {
                    reason: "hot row output range is outside the lease".to_string(),
                })?;
            // SAFETY: as at `copy_requested_rows_shared`: each request
            // position is one `HotItem` or one `RowCopy` of a source page,
            // never both (hot and source pages partition the request's
            // pages), positions are `decoded_row_bytes` apart, and every run
            // is decoded by exactly one reader.
            let destination = unsafe { output.range_mut(item.output_offset..end) };
            self.encoding
                .decode_row(&buffer[at..at + self.encoded_row_bytes], destination)?;
        }
        Ok(())
    }

    /// Global row of the first row of `page`.
    fn first_row_of(&self, page: PageKey) -> u64 {
        page.shard as u64 * self.rows_per_shard as u64 + (page.page * self.rows_per_page) as u64
    }

    fn coalesce_pages(&self, pages: &[PageKey]) -> Result<Vec<Vec<PageKey>>, RowStoreError> {
        let mut groups: Vec<Vec<PageKey>> = Vec::new();
        let mut current_bytes = 0usize;
        for &page in pages {
            let page_bytes = self.page_len(page)?;
            let append = groups.last().is_some_and(|group| {
                let previous = group[group.len() - 1];
                previous.shard == page.shard
                    && previous.page.checked_add(1) == Some(page.page)
                    && current_bytes
                        .checked_add(page_bytes)
                        .is_some_and(|total| total <= self.staging_bytes)
            });
            if append {
                groups.last_mut().expect("group exists").push(page);
                current_bytes = current_bytes.checked_add(page_bytes).ok_or_else(|| {
                    RowStoreError::InvalidLease {
                        reason: "coalesced length overflow".to_string(),
                    }
                })?;
            } else {
                groups.push(vec![page]);
                current_bytes = page_bytes;
            }
        }
        Ok(groups)
    }

    fn page_len(&self, page: PageKey) -> Result<usize, RowStoreError> {
        if page.shard >= self.shard_count {
            return Err(RowStoreError::GlobalRowOutOfRange {
                row: self.padded_rows,
                padded_rows: self.padded_rows,
            });
        }
        let first_row = page.page.checked_mul(self.rows_per_page).ok_or_else(|| {
            RowStoreError::InvalidLease {
                reason: "page row offset overflow".to_string(),
            }
        })?;
        if first_row >= self.rows_per_shard {
            return Err(RowStoreError::InvalidLease {
                reason: "page exceeds physical shard".to_string(),
            });
        }
        let rows = (self.rows_per_shard - first_row).min(self.rows_per_page);
        rows.checked_mul(self.encoded_row_bytes)
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "page byte length overflow".to_string(),
            })
    }
    fn read_group(
        &self,
        group: &[PageKey],
        requested: &HashMap<PageKey, Vec<RowCopy>>,
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
    ) -> Result<(), RowStoreError> {
        let first = *group.first().ok_or_else(|| RowStoreError::InvalidLease {
            reason: "empty coalesced page group".to_string(),
        })?;
        let first_offset = first
            .page
            .checked_mul(self.rows_per_page)
            .and_then(|row| row.checked_mul(self.encoded_row_bytes))
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "coalesced offset overflow".to_string(),
            })? as u64;
        let lengths: Vec<usize> = group
            .iter()
            .map(|&page| self.page_len(page))
            .collect::<Result<_, _>>()?;
        let total = lengths
            .iter()
            .try_fold(0usize, |sum, len| sum.checked_add(*len))
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "coalesced length overflow".to_string(),
            })?;
        if total > self.staging_bytes {
            return Err(RowStoreError::InvalidLease {
                reason: "coalesced read exceeds staging budget".to_string(),
            });
        }

        read_staging.resize(total, 0);
        self.source
            .read_at(first.shard, first_offset, read_staging)
            .map_err(RowStoreError::Source)?;
        if canceled.load(Ordering::Acquire) {
            return Err(RowStoreError::Canceled);
        }
        self.publish_group(group, read_staging, requested, output)
    }

    /// Read one coalesced page group's bytes into `bytes`.
    fn read_group_bytes(
        &self,
        group: &[PageKey],
        bytes: &mut Vec<u8>,
    ) -> Result<(), RowStoreError> {
        let first = *group.first().ok_or_else(|| RowStoreError::InvalidLease {
            reason: "empty coalesced page group".to_string(),
        })?;
        let first_offset = first
            .page
            .checked_mul(self.rows_per_page)
            .and_then(|row| row.checked_mul(self.encoded_row_bytes))
            .ok_or_else(|| RowStoreError::InvalidLease {
                reason: "coalesced offset overflow".to_string(),
            })? as u64;
        let mut total = 0usize;
        for &page in group {
            total = total.checked_add(self.page_len(page)?).ok_or_else(|| {
                RowStoreError::InvalidLease {
                    reason: "coalesced length overflow".to_string(),
                }
            })?;
        }
        if total > self.staging_bytes {
            return Err(RowStoreError::InvalidLease {
                reason: "coalesced read exceeds staging budget".to_string(),
            });
        }
        bytes.resize(total, 0);
        self.source
            .read_at(first.shard, first_offset, bytes)
            .map_err(RowStoreError::Source)
    }

    /// Insert pages read by a multi-group request into the cache in ascending
    /// page order (the order the serial publish used to), taking the state
    /// lock once per `PUBLISH_BATCH` pages rather than across the whole
    /// request.
    fn publish_pending(&self, pending: PendingPages) {
        let mut pages: Vec<(PageKey, Vec<u8>)> = pending.into_iter().flatten().collect();
        pages.sort_unstable_by_key(|(page, _)| *page);
        let mut pages = pages.into_iter().peekable();
        while pages.peek().is_some() {
            let mut state = self.state.lock().expect("row store state mutex poisoned");
            for (page, bytes) in pages.by_ref().take(PUBLISH_BATCH) {
                state.cache.insert(page, bytes);
            }
        }
    }

    /// Copy a read group's requested rows into `output` and cache its pages.
    fn publish_group(
        &self,
        group: &[PageKey],
        read_staging: &[u8],
        requested: &HashMap<PageKey, Vec<RowCopy>>,
        output: &mut [u8],
    ) -> Result<(), RowStoreError> {
        let lengths: Vec<usize> = group
            .iter()
            .map(|&page| self.page_len(page))
            .collect::<Result<_, _>>()?;
        let total: usize = lengths.iter().sum();
        let mut state = self.state.lock().expect("row store state mutex poisoned");
        let mut cursor = 0usize;
        for (&page, &length) in group.iter().zip(&lengths) {
            let page_bytes = &read_staging[cursor..cursor + length];
            if let Some(rows) = requested.get(&page) {
                copy_requested_rows(
                    self.encoding,
                    self.encoded_row_bytes,
                    self.decoded_row_bytes,
                    page,
                    page_bytes,
                    rows,
                    output,
                )?;
            }
            state.cache.insert(page, page_bytes.to_vec());
            cursor += length;
        }
        state.cache.reads += 1;
        state.cache.coalesced_reads += 1;
        state.cache.read_bytes += total as u64;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RowCopy {
    output_offset: usize,
    page_offset: usize,
}

/// Copy the requested rows out of one cached page into the lease staging
/// buffer, decoding from the stored representation to `decoded_row_bytes` of
/// BF16.
///
/// The cache keeps the *encoded* page, so a decode is paid per access instead
/// of per distinct row; at 160 weights per row that is five Q8 blocks, and the
/// alternative (caching decoded rows) would cut the cache's row capacity by the
/// same encoding ratio.
fn copy_requested_rows(
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    decoded_row_bytes: usize,
    page: PageKey,
    page_bytes: &[u8],
    rows: &[RowCopy],
    output: &mut [u8],
) -> Result<(), RowStoreError> {
    for row in rows {
        let (page_range, output_range) = checked_row_ranges(
            encoded_row_bytes,
            decoded_row_bytes,
            page,
            page_bytes.len(),
            output.len(),
            row,
        )?;
        encoding.decode_row(&page_bytes[page_range], &mut output[output_range])?;
    }
    Ok(())
}

/// [`copy_requested_rows`] for readers that write disjoint rows of one shared
/// output at the same time.
///
/// The caller upholds [`SharedOutput::range_mut`]'s contract by handing each
/// output row to exactly one call: `rows` are the requested rows of `page`,
/// whose `output_offset`s are `index * decoded_row_bytes` of distinct request
/// positions, and every page of a request is read by exactly one thread.
fn copy_requested_rows_shared(
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    decoded_row_bytes: usize,
    page: PageKey,
    page_bytes: &[u8],
    rows: &[RowCopy],
    output: &SharedOutput<'_>,
) -> Result<(), RowStoreError> {
    for row in rows {
        let (page_range, output_range) = checked_row_ranges(
            encoded_row_bytes,
            decoded_row_bytes,
            page,
            page_bytes.len(),
            output.len(),
            row,
        )?;
        // SAFETY: `output_range` is `row.output_offset..+decoded_row_bytes`
        // (bounds checked above).  Each request position is one `RowCopy` in
        // exactly one page's list, request positions are `decoded_row_bytes`
        // apart so their ranges never overlap, and `fill_output` hands each
        // (strictly ascending, so unique) page to exactly one thread.  No
        // other slice of this row exists while this one is live.
        let destination = unsafe { output.range_mut(output_range) };
        encoding.decode_row(&page_bytes[page_range], destination)?;
    }
    Ok(())
}

/// Bounds-check one requested row against its page and the output, returning
/// the encoded byte range in the page and the decoded byte range in the output.
fn checked_row_ranges(
    encoded_row_bytes: usize,
    decoded_row_bytes: usize,
    page: PageKey,
    page_len: usize,
    output_len: usize,
    row: &RowCopy,
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>), RowStoreError> {
    let page_end = row
        .page_offset
        .checked_add(encoded_row_bytes)
        .ok_or_else(|| RowStoreError::InvalidLease {
            reason: format!("page {page:?} row offset overflow"),
        })?;
    let output_end = row
        .output_offset
        .checked_add(decoded_row_bytes)
        .ok_or_else(|| RowStoreError::InvalidLease {
            reason: "output row offset overflow".to_string(),
        })?;
    if page_end > page_len || output_end > output_len {
        return Err(RowStoreError::InvalidLease {
            reason: format!("requested row is outside page {page:?} or output"),
        });
    }
    Ok((row.page_offset..page_end, row.output_offset..output_end))
}

/// The lease staging buffer shared by the scoped reader threads of one
/// request, each of which decodes its own pages' rows straight into it.
///
/// Rust cannot split the buffer by page (a page's rows are scattered through
/// the output), so the threads share a raw pointer and the disjointness of the
/// rows they write is argued at [`copy_requested_rows_shared`].  It borrows the
/// buffer mutably for `'a`, so nothing else touches the buffer meanwhile.
struct SharedOutput<'a> {
    ptr: *mut u8,
    len: usize,
    _borrow: std::marker::PhantomData<&'a mut [u8]>,
}

// SAFETY: the handle only ever yields `range_mut` slices, whose callers
// guarantee that concurrently live ranges are disjoint; `u8` has no thread
// affinity.
unsafe impl Send for SharedOutput<'_> {}
unsafe impl Sync for SharedOutput<'_> {}

impl<'a> SharedOutput<'a> {
    fn new(output: &'a mut [u8]) -> Self {
        Self {
            ptr: output.as_mut_ptr(),
            len: output.len(),
            _borrow: std::marker::PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    /// No other slice obtained from this handle that overlaps `range` may be
    /// live (or created) while the returned slice is.
    #[allow(clippy::mut_from_ref)]
    unsafe fn range_mut(&self, range: std::ops::Range<usize>) -> &mut [u8] {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "shared output range outside the buffer"
        );
        // SAFETY: in bounds per the assert; exclusivity is the caller's
        // contract above.
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start)
        }
    }
}

/// Pages a request read, in group order, that still have to enter the cache,
/// one list per reader thread.  Empty when nothing was read.
type PendingPages = Vec<Vec<(PageKey, Vec<u8>)>>;

struct ProcessedTicket {
    output: Option<Vec<u8>>,
    read_buffer: Vec<u8>,
    result: Result<(), RowStoreError>,
    /// Pages to cache after the ticket has been handed to its waiter, whatever
    /// `result` says (see [`RowStoreInner::publish_pending`]).
    pending: PendingPages,
}

struct TicketState {
    epoch: u64,
    canceled: Arc<AtomicBool>,
    ids: Vec<u64>,
    locations: Vec<RowLocation>,
    pages: Vec<PageKey>,
    required_bytes: usize,
    output: Option<Vec<u8>>,
    status: TicketStatus,
    /// Cache warm-up only ([`RowStore::warm`]): no lease; the reader drops
    /// the ticket once its pages are cached.
    warm: bool,
}

enum TicketStatus {
    Pending,
    Completed(Result<(), RowStoreError>),
}

struct RowStoreState {
    queue: VecDeque<u64>,
    tickets: HashMap<u64, TicketState>,
    next_ticket: u64,
    accepting: bool,
    stop: bool,
    active_readers: usize,
    active_leases: usize,
    staging_pool: Vec<Option<Vec<u8>>>,
    staging_in_use: usize,
    staging_high_water: usize,
    cache: PageCache,
}

impl RowStoreState {
    fn new(staging_bytes: usize) -> Self {
        let staging_pool = (0..MAX_STAGING_BUFFERS)
            .map(|_| Some(Vec::with_capacity(staging_bytes)))
            .collect();
        Self {
            queue: VecDeque::new(),
            tickets: HashMap::new(),
            next_ticket: 1,
            accepting: true,
            stop: false,
            active_readers: 0,
            active_leases: 0,
            staging_pool,
            staging_in_use: 0,

            staging_high_water: 0,
            cache: PageCache::new(PAGE_CACHE_BYTES),
        }
    }
}
fn invalidate_tickets_locked(inner: &RowStoreInner, state: &mut RowStoreState) {
    for ticket in state.tickets.values() {
        ticket.canceled.store(true, Ordering::Release);
    }
    let queued: Vec<u64> = state.queue.drain(..).collect();
    for id in queued {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
    let completed: Vec<u64> = state
        .tickets
        .iter()
        .filter_map(|(&id, ticket)| {
            matches!(ticket.status, TicketStatus::Completed(_)).then_some(id)
        })
        .collect();
    for id in completed {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn purge_tickets_locked(inner: &RowStoreInner, state: &mut RowStoreState) {
    let ids: Vec<u64> = state.tickets.keys().copied().collect();
    for id in ids {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn worker_loop(weak: Weak<RowStoreInner>) {
    loop {
        let Some(inner) = weak.upgrade() else {
            break;
        };
        let work = {
            let mut state = inner.state.lock().expect("row store state mutex poisoned");
            loop {
                if state.stop {
                    break None;
                }
                let Some(id) = state.queue.front().copied() else {
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                let Some(mut output) = inner.take_staging_locked(&mut state) else {
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                let Some(read_buffer) = inner.take_staging_locked(&mut state) else {
                    inner.return_staging_locked(&mut state, output);
                    state = inner
                        .cv
                        .wait(state)
                        .expect("row store state mutex poisoned");
                    continue;
                };
                state.queue.pop_front();
                let Some(ticket) = state.tickets.get_mut(&id) else {
                    inner.return_staging_locked(&mut state, output);
                    inner.return_staging_locked(&mut state, read_buffer);
                    continue;
                };
                output.resize(ticket.required_bytes, 0);
                ticket.output = Some(output);
                state.active_readers += 1;
                break Some((id, read_buffer));
            }
        };
        let Some((id, read_buffer)) = work else {
            break;
        };
        let ProcessedTicket {
            output,
            read_buffer,
            result,
            pending,
        } = inner.process_ticket(id, read_buffer);
        let mut state = inner.state.lock().expect("row store state mutex poisoned");
        inner.return_staging_locked(&mut state, read_buffer);
        if pending.is_empty() {
            state.active_readers = state.active_readers.saturating_sub(1);
        }
        complete_ticket_locked(&inner, &mut state, id, output, result);
        inner.cv.notify_all();
        if pending.is_empty() {
            continue;
        }
        // The waiter can already take its lease.  The pages the reads brought
        // in are cached here, before this serial worker picks up the next
        // ticket, so a following ticket needing the same pages sees them; the
        // reader stays counted as active so quiesce waits for the inserts.
        drop(state);
        inner.publish_pending(pending);
        let mut state = inner.state.lock().expect("row store state mutex poisoned");
        state.active_readers = state.active_readers.saturating_sub(1);
        inner.cv.notify_all();
    }
}

/// Hand a processed ticket's result to its waiter (or drop its buffers when
/// the ticket is warm, canceled, stale, or gone).
fn complete_ticket_locked(
    inner: &RowStoreInner,
    state: &mut RowStoreState,
    id: u64,
    mut output: Option<Vec<u8>>,
    result: Result<(), RowStoreError>,
) {
    if state.tickets.get(&id).is_some_and(|ticket| ticket.warm) {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
        if let Some(buffer) = output.take() {
            inner.return_staging_locked(state, buffer);
        }
        return;
    }
    let Some((ticket_canceled, ticket_epoch)) = state
        .tickets
        .get(&id)
        .map(|ticket| (ticket.canceled.load(Ordering::Acquire), ticket.epoch))
    else {
        if let Some(buffer) = output.take() {
            inner.return_staging_locked(state, buffer);
        }
        return;
    };
    let canceled = ticket_canceled || ticket_epoch != inner.current_epoch.load(Ordering::Acquire);
    if canceled {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
        if let Some(buffer) = output.take() {
            inner.return_staging_locked(state, buffer);
        }
        return;
    }
    let mut return_buffer = None;
    {
        let ticket = state.tickets.get_mut(&id).expect("ticket disappeared");
        match result {
            Ok(()) => {
                ticket.output = output;
                ticket.status = TicketStatus::Completed(Ok(()));
            }
            Err(error) => {
                return_buffer = output.take();
                ticket.output = None;
                ticket.status = TicketStatus::Completed(Err(error));
            }
        }
    }
    if let Some(buffer) = return_buffer {
        inner.return_staging_locked(state, buffer);
    }
}

trait PositionalRowSource: Send + Sync {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError>;
}

struct DescriptorRowSource {
    descriptors: Arc<[SourceRangeDescriptor]>,
}

impl PositionalRowSource for DescriptorRowSource {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let descriptor = self
            .descriptors
            .get(shard)
            .ok_or_else(|| SourceError::InvalidSource {
                reason: format!("row shard {shard} is absent"),
            })?;
        let absolute =
            descriptor
                .offset
                .checked_add(local_offset)
                .ok_or(SourceError::Overflow {
                    offset: descriptor.offset,
                    length: local_offset,
                })?;
        descriptor.read_exact_at(absolute, dst)
    }
}

/// Positional source of a table store ([`RowStore::from_table_store`]): the
/// table is rank-ordered, so a shard's rows are not contiguous in the file.
/// Every read is split into rows and each row is read at its own pack offset.
/// The row store takes pages whose requested rows the store holds straight
/// from the pack, and every row is held, so this source only serves what that
/// path leaves (nothing, today) and keeps the page path total.
struct TableRowSource {
    store: Arc<HotRowStore>,
    shard_count: usize,
    rows_per_shard: u64,
    row_bytes: usize,
}

impl PositionalRowSource for TableRowSource {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let row_bytes = self.row_bytes as u64;
        let first_local = local_offset / row_bytes;
        let rows = dst.len() as u64 / row_bytes;
        if shard >= self.shard_count
            || local_offset % row_bytes != 0
            || dst.len() as u64 % row_bytes != 0
            || first_local + rows > self.rows_per_shard
        {
            return Err(SourceError::InvalidSource {
                reason: format!(
                    "table read of {} bytes at {local_offset} in shard {shard} is not whole rows \
                     of {row_bytes} bytes inside the shard",
                    dst.len()
                ),
            });
        }
        let first = shard as u64 * self.rows_per_shard + first_local;
        for (index, row) in dst.chunks_exact_mut(self.row_bytes).enumerate() {
            let global = first + index as u64;
            let at = self
                .store
                .pack_offset(global)
                .ok_or_else(|| SourceError::InvalidSource {
                    reason: format!("table row {global} has no pack offset"),
                })?;
            self.store.read_pack(at, row)?;
        }
        Ok(())
    }
}

struct CachedPage {
    bytes: Vec<u8>,
    last_used: u64,
}

struct PageCache {
    capacity_bytes: usize,
    pages: HashMap<PageKey, CachedPage>,
    /// Recency index over [`CachedPage::last_used`]: stamp -> page.  A page is
    /// small, so the cache holds tens of thousands of entries and the least
    /// recently used page must be found without scanning the map.
    by_stamp: BTreeMap<u64, PageKey>,
    resident_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    reads: u64,
    coalesced_reads: u64,
    read_bytes: u64,
    evictions: u64,
}

impl PageCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            pages: HashMap::new(),
            by_stamp: BTreeMap::new(),
            resident_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            reads: 0,
            coalesced_reads: 0,
            read_bytes: 0,
            evictions: 0,
        }
    }

    fn touch(&mut self, key: PageKey) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(page) = self.pages.get_mut(&key) {
            self.by_stamp.remove(&page.last_used);
            page.last_used = self.clock;
            self.by_stamp.insert(self.clock, key);
        }
    }

    fn insert(&mut self, key: PageKey, bytes: Vec<u8>) {
        if bytes.len() > self.capacity_bytes {
            return;
        }
        if let Some(previous) = self.pages.remove(&key) {
            self.by_stamp.remove(&previous.last_used);
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.bytes.len());
        }
        while self.resident_bytes.saturating_add(bytes.len()) > self.capacity_bytes {
            let Some((&stamp, &victim)) = self.by_stamp.iter().next() else {
                break;
            };
            self.by_stamp.remove(&stamp);
            if let Some(page) = self.pages.remove(&victim) {
                self.resident_bytes = self.resident_bytes.saturating_sub(page.bytes.len());
                self.evictions += 1;
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.resident_bytes += bytes.len();
        self.by_stamp.insert(self.clock, key);
        self.pages.insert(
            key,
            CachedPage {
                bytes,
                last_used: self.clock,
            },
        );
    }

    fn clear(&mut self) -> usize {
        let dropped = self.resident_bytes;
        self.pages.clear();
        self.by_stamp.clear();
        self.resident_bytes = 0;
        dropped
    }
}

/// Physical layout of one validated shard set.
#[derive(Debug, Clone, Copy)]
struct Layout {
    shard_count: usize,
    rows_per_shard: usize,
    rows_per_page: usize,
    row_width: usize,
    valid_rows: u64,
    encoding: RowEncoding,
}

fn validate_descriptors(
    descriptors: &[SourceRangeDescriptor],
    valid_rows: u64,
) -> Result<Layout, RowStoreError> {
    let first = descriptors
        .first()
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "at least one physical shard is required".to_string(),
        })?;
    // The declared dtype is the contract: BF16 shards are `2 * width`-byte raw
    // rows, Q8F16 shards are `width / 32` 34-byte blocks per row and Q4LM
    // shards `width / 32` 18-byte blocks per row. Nothing infers the tier from
    // the byte count, so a mislabelled artifact is refused rather than decoded
    // as the wrong width.
    let encoding =
        RowEncoding::from_dtype(first.dtype()).ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: format!("rows must be BF16, Q8_0 or Q4LM, got {}", first.dtype()),
        })?;
    let shape = first.logical_shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!("expected a non-empty [rows, width] shape, got {shape:?}"),
        });
    }
    let (rows, width) = (shape[0], shape[1]);
    if encoding.checked_encoded_row_bytes(width).is_none() {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!(
                "{} row width {width} is not a multiple of {Q8_BLOCK_WEIGHTS}",
                encoding.dtype_name()
            ),
        });
    }
    let expected_length = (rows as u64)
        .checked_mul(encoding.encoded_row_bytes(width) as u64)
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "descriptor length overflow".to_string(),
        })?;
    let identity = first.source_identity();
    for (index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.source_identity() != identity {
            return Err(RowStoreError::Descriptor {
                index,
                reason: "descriptor source identity differs from shard 0".to_string(),
            });
        }
        if descriptor.logical_shape() != [rows, width] {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!("all shards must have shape [{rows}, {width}]"),
            });
        }
        if descriptor.length != expected_length {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!(
                    "expected {expected_length} bytes for {} rows, got {}",
                    encoding.dtype_name(),
                    descriptor.length
                ),
            });
        }
        if RowEncoding::from_dtype(descriptor.dtype()) != Some(encoding) {
            return Err(RowStoreError::Descriptor {
                index,
                reason: format!(
                    "all shards must be {}, got {}",
                    encoding.dtype_name(),
                    descriptor.dtype()
                ),
            });
        }
    }
    let physical_rows = (rows as u64)
        .checked_mul(descriptors.len() as u64)
        .ok_or_else(|| RowStoreError::Descriptor {
            index: 0,
            reason: "physical row count overflow".to_string(),
        })?;
    if valid_rows == 0 || valid_rows > physical_rows {
        return Err(RowStoreError::Descriptor {
            index: 0,
            reason: format!("valid rows {valid_rows} outside 1..={physical_rows} physical rows"),
        });
    }
    Ok(Layout {
        shard_count: descriptors.len(),
        rows_per_shard: rows,
        rows_per_page: ROWS_PER_PAGE,
        row_width: width,
        valid_rows,
        encoding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::atomic::AtomicUsize;

    struct MemoryRowSource {
        shards: Vec<Vec<u8>>,
        reads: AtomicUsize,
        fail: Option<(usize, u64)>,
    }

    impl PositionalRowSource for MemoryRowSource {
        fn read_at(&self, shard: usize, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            self.reads.fetch_add(1, Ordering::AcqRel);
            if self.fail == Some((shard, offset)) {
                return Err(SourceError::Io {
                    offset,
                    source: io::Error::new(io::ErrorKind::Other, "fixture read failure"),
                });
            }
            let source = self
                .shards
                .get(shard)
                .ok_or_else(|| SourceError::InvalidSource {
                    reason: "fixture shard missing".to_string(),
                })?;
            let begin = usize::try_from(offset).map_err(|_| SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            let end = begin.checked_add(dst.len()).ok_or(SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            if end > source.len() {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: source.len().saturating_sub(begin),
                });
            }
            dst.copy_from_slice(&source[begin..end]);
            Ok(())
        }
    }

    /// The Qwen4 PLE shape these fixtures were written against: sixteen
    /// 160-value rows per token.
    const ROWS_PER_TOKEN: usize = 16;
    const ROW_WIDTH: usize = 160;
    const ROW_BYTES: usize = ROW_WIDTH * 2;

    /// Valid/padded row counts of a fixture table made of sixteen equal
    /// per-head sub-tables.
    #[derive(Clone, Copy)]
    struct Fixture {
        valid_rows: u64,
        padded_rows: u64,
    }

    fn metadata_with_head_size(head_size: u64, padded_rows: u64) -> Fixture {
        Fixture {
            valid_rows: head_size * ROWS_PER_TOKEN as u64,
            padded_rows,
        }
    }

    fn metadata(padded_rows: u64) -> Fixture {
        metadata_with_head_size(2, padded_rows)
    }

    fn rows_source(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        (0..shard_count)
            .map(|shard| {
                let mut bytes = vec![0u8; rows_per_shard * ROW_BYTES];
                for row in 0..rows_per_shard {
                    let global = shard * rows_per_shard + row;
                    let value = (global as u16).to_le_bytes();
                    for cell in bytes[row * ROW_BYTES..(row + 1) * ROW_BYTES].chunks_exact_mut(2) {
                        cell.copy_from_slice(&value);
                    }
                }
                bytes
            })
            .collect()
    }

    impl RowStore {
        /// Prefetch sixteen rows per token, one from each per-head sub-table,
        /// with ids scattered like a multiplicative hash of `(seed, token,
        /// head)`.  A repeated token repeats its ids.
        fn prefetch_tokens(
            &self,
            epoch: u64,
            seed: u32,
            tokens: &[u32],
        ) -> Result<RowTicket, RowStoreError> {
            let head_rows = self.inner.valid_rows / ROWS_PER_TOKEN as u64;
            let ids = tokens
                .iter()
                .flat_map(|&token| {
                    (0..ROWS_PER_TOKEN as u64).map(move |head| {
                        let mut x = (u64::from(token) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ u64::from(seed).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                            ^ (head + 1).wrapping_mul(0x1656_67B1_9E37_79F9);
                        x ^= x >> 29;
                        head * head_rows + x % head_rows
                    })
                })
                .collect();
            self.prefetch(epoch, ids)
        }

        fn from_test_source(
            metadata: Fixture,
            rows_per_shard: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            Self::from_test_source_with_page_rows(metadata, rows_per_shard, ROWS_PER_PAGE, source)
        }

        fn from_test_source_with_page_rows(
            metadata: Fixture,
            rows_per_shard: usize,
            rows_per_page: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            Self::from_test_source_encoded(
                metadata,
                rows_per_shard,
                rows_per_page,
                RowEncoding::Bf16,
                source,
            )
        }

        fn from_test_source_encoded(
            metadata: Fixture,
            rows_per_shard: usize,
            rows_per_page: usize,
            encoding: RowEncoding,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, RowStoreError> {
            if rows_per_shard == 0 || rows_per_page == 0 {
                return Err(RowStoreError::Descriptor {
                    index: 0,
                    reason: "invalid fixture shard geometry".to_string(),
                });
            }
            let shard_count = usize::try_from(metadata.padded_rows / rows_per_shard as u64)
                .map_err(|_| RowStoreError::Descriptor {
                    index: 0,
                    reason: "test shard count overflow".to_string(),
                })?;
            if shard_count == 0
                || (shard_count as u64) * rows_per_shard as u64 != metadata.padded_rows
            {
                return Err(RowStoreError::Descriptor {
                    index: 0,
                    reason: "fixture rows do not cover the padded table".to_string(),
                });
            }
            Self::spawn(
                "external-rows-test-reader",
                source,
                Vec::new().into(),
                Layout {
                    shard_count,
                    rows_per_shard,
                    rows_per_page,
                    row_width: ROW_WIDTH,
                    valid_rows: metadata.valid_rows,
                    encoding,
                },
                0,
                true,
            )
        }
    }

    /// Small requests keep the original 16 readers; a prefill chunk's page
    /// set saturates at the ceiling, and a store built without wide readers
    /// never leaves 16.
    #[test]
    fn reader_count_scales_only_large_requests() {
        for groups in [
            0,
            1,
            16,
            64,
            1024,
            16 * READS_PER_READER + READS_PER_READER - 1,
        ] {
            assert_eq!(
                reader_count(groups, MAX_PARALLEL_READERS),
                PARALLEL_READERS,
                "{groups} groups"
            );
        }
        assert_eq!(
            reader_count(64 * READS_PER_READER, MAX_PARALLEL_READERS),
            64
        );
        assert_eq!(
            reader_count(111_412, MAX_PARALLEL_READERS),
            MAX_PARALLEL_READERS
        );
        assert_eq!(reader_count(111_412, PARALLEL_READERS), PARALLEL_READERS);
    }

    /// The production constructor reads real sealed HFQ ranges: rows come back
    /// in request order, padding past `valid_rows` is refused, and a
    /// `valid_rows` beyond the physical table cannot open.
    #[test]
    fn production_store_reads_sealed_ranges_and_refuses_padding() {
        use crate::hfq::hfq_test_fixture::{
            write_compact_qwen4_ple_hfq, COMPACT_PLE_NAMES, COMPACT_PLE_ROW_BYTES,
        };
        use crate::hfq::HfqFile;
        use crate::model_source::{ModelSource, SourcePayload};

        let dir = tempfile::tempdir().expect("fixture directory");
        let path = dir.path().join("rows.hfq");
        write_compact_qwen4_ple_hfq(&path).expect("write compact fixture");
        let hfq = HfqFile::open(&path).expect("open compact fixture");
        let descriptors: Vec<SourceRangeDescriptor> = COMPACT_PLE_NAMES
            .iter()
            .map(
                |name| match (&hfq as &dyn ModelSource).tensor_payload(name) {
                    Ok(Some(SourcePayload::Range(descriptor))) => descriptor,
                    _ => panic!("fixture tensor {name} is not a lazy range"),
                },
            )
            .collect();
        let physical_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.logical_shape()[0] as u64)
            .sum::<u64>();
        let mut expected = Vec::new();
        for (shard, descriptor) in descriptors.iter().enumerate() {
            let mut row = vec![0u8; COMPACT_PLE_ROW_BYTES];
            descriptor
                .read_exact_at(descriptor.offset, &mut row)
                .unwrap();
            expected.push((shard as u64 * descriptor.logical_shape()[0] as u64, row));
        }

        let too_many = RowStore::new("t", descriptors.clone(), physical_rows + 1);
        assert!(matches!(too_many, Err(RowStoreError::Descriptor { .. })));

        let store = RowStore::new("t", descriptors, physical_rows - 1).unwrap();
        assert!(matches!(
            store.prefetch(0, vec![physical_rows - 1]),
            Err(RowStoreError::PaddingRow { .. })
        ));
        let ids: Vec<u64> = expected.iter().rev().map(|(row, _)| *row).collect();
        let ticket = store.prefetch(0, ids.clone()).unwrap();
        let lease = store.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), ids.as_slice());
        for (index, (_, row)) in expected.iter().rev().enumerate() {
            assert_eq!(lease.row_bytes(index).unwrap(), row.as_slice());
        }
        drop(lease);
        assert!(store.unload().unwrap().is_clean());
    }

    /// A store sized with `with_staging_rows` admits a prefetch of exactly
    /// that many rows (here past the default budget) and returns them; one
    /// more row is refused, as the default store refuses its own bound + 1.
    #[test]
    fn staging_rows_set_the_prefetch_bound() {
        use crate::hfq::hfq_test_fixture::{write_compact_qwen4_ple_hfq, COMPACT_PLE_NAMES};
        use crate::hfq::HfqFile;
        use crate::model_source::{ModelSource, SourcePayload};

        let dir = tempfile::tempdir().expect("fixture directory");
        let path = dir.path().join("rows.hfq");
        write_compact_qwen4_ple_hfq(&path).expect("write compact fixture");
        let hfq = HfqFile::open(&path).expect("open compact fixture");
        let descriptors: Vec<SourceRangeDescriptor> = COMPACT_PLE_NAMES
            .iter()
            .map(
                |name| match (&hfq as &dyn ModelSource).tensor_payload(name) {
                    Ok(Some(SourcePayload::Range(descriptor))) => descriptor,
                    _ => panic!("fixture tensor {name} is not a lazy range"),
                },
            )
            .collect();
        let valid_rows = descriptors
            .iter()
            .map(|descriptor| descriptor.logical_shape()[0] as u64)
            .sum::<u64>();

        let default = RowStore::new("t", descriptors.clone(), valid_rows).unwrap();
        let default_max = default.max_rows_per_prefetch();
        assert!(matches!(
            default.prefetch(0, vec![0; default_max + 1]),
            Err(RowStoreError::RequestTooLarge { .. })
        ));
        assert!(default.unload().unwrap().is_clean());

        let rows = default_max * 2 + 3;
        let store = RowStore::with_staging_rows("t", descriptors, valid_rows, rows, true).unwrap();
        assert_eq!(store.max_rows_per_prefetch(), rows);
        assert!(matches!(
            store.prefetch(0, vec![0; rows + 1]),
            Err(RowStoreError::RequestTooLarge { .. })
        ));
        let ids: Vec<u64> = (0..rows as u64).map(|index| index % valid_rows).collect();
        let ticket = store.prefetch(0, ids.clone()).unwrap();
        let lease = store.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), ids.as_slice());
        let first = lease.row_bytes(0).unwrap().to_vec();
        let period = valid_rows as usize;
        assert_eq!(
            lease.row_bytes(rows - 1 - (rows - 1) % period).unwrap(),
            first
        );
        drop(lease);
        assert!(store.unload().unwrap().is_clean());
    }

    // These tests exercise bounded cache and deterministic page planning.
    #[test]
    fn cache_is_bounded_and_evicts_lru() {
        let mut cache = PageCache::new(ROW_BYTES * 2);
        cache.insert(PageKey { shard: 0, page: 0 }, vec![1; ROW_BYTES]);
        cache.insert(PageKey { shard: 0, page: 1 }, vec![2; ROW_BYTES]);
        cache.touch(PageKey { shard: 0, page: 0 });
        cache.insert(PageKey { shard: 0, page: 2 }, vec![3; ROW_BYTES]);
        assert_eq!(cache.resident_bytes, ROW_BYTES * 2);
        assert_eq!(cache.pages.len(), 2);
        assert!(!cache.pages.contains_key(&PageKey { shard: 0, page: 1 }));
        assert_eq!(cache.evictions, 1);
    }

    #[test]
    fn location_rejects_padding_and_maps_shard_boundary() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        assert_eq!(
            rows.locate_row(0).unwrap(),
            RowLocation {
                global_row: 0,
                shard: 0,
                local_row: 0,
                page: 0,
                page_byte_offset: 0,
            }
        );
        let error = rows.locate_row(32).unwrap_err();
        assert!(matches!(error, RowStoreError::PaddingRow { row: 32, .. }));
        let full = RowStore::from_test_source(
            metadata_with_head_size(8, 128),
            64,
            Arc::new(MemoryRowSource {
                shards: rows_source(2, 64),
                reads: AtomicUsize::new(0),
                fail: None,
            }),
        )
        .unwrap();
        assert_eq!(full.locate_row(63).unwrap().shard, 0);
        assert_eq!(full.locate_row(64).unwrap().shard, 1);
        assert!(full.unload().unwrap().is_clean());
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn prefetch_reads_once_and_preserves_token_head_order() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source.clone()).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!((lease.row_count() / ROWS_PER_TOKEN), 2);
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..ROWS_PER_TOKEN {
                let row = lease.row_bytes(token * ROWS_PER_TOKEN + head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token * ROWS_PER_TOKEN + head]);
            }
        }
        let mut copied = vec![0; lease.as_bytes().unwrap().len()];
        lease.stage_into(&mut copied).unwrap();
        assert_eq!(copied, lease.as_bytes().unwrap());
        drop(lease);
        // Every distinct physical page is read exactly once for one ticket;
        // adjacent pages may coalesce into one call.
        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        let second = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let lease = rows.wait_completed_lease(&second).unwrap();
        drop(lease);
        assert_eq!(rows.cache_stats().read_bytes, stats.read_bytes);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn source_error_and_epoch_invalidation_are_explicit() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: Some((0, 0)),
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            RowStoreError::Source(SourceError::Io { .. })
        ));
        // This is the forward-attempt abort boundary: the consumed ticket
        // may already report a source error, but reset_epoch must still drain
        // the exact epoch before a caller returns to its request loop.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 1);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        rows.begin_epoch(1).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            RowStoreError::Canceled | RowStoreError::EpochMismatch { .. }
        ));
        // A canceled read follows the same abort path and must leave the
        // reader reusable at a later epoch rather than leaking its ticket.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 2);
        let retry = rows.prefetch_tokens(2, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&retry).unwrap();
        drop(lease);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn fetch_adopts_only_a_live_exact_lookahead_ticket() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source.clone()).unwrap();
        rows.begin_epoch(0).unwrap();
        let reference = rows.prefetch_tokens(0, 7, &[3, 4]).unwrap();
        let ids = reference.row_ids().unwrap();
        let expected = rows
            .wait_completed_lease(&reference)
            .unwrap()
            .as_bytes()
            .unwrap()
            .to_vec();

        // Exact ids in the current epoch: adopted, no new epoch, same bytes.
        let ahead = rows.prefetch_tokens(0, 7, &[3, 4]).unwrap();
        let reads = source.reads.load(Ordering::Acquire);
        let mut fetch = RowFetch::begin_or_adopt(&rows, Some(ahead), ids.clone()).unwrap();
        assert_eq!(fetch.epoch(), 0);
        assert_eq!(
            fetch.wait().unwrap().as_bytes().unwrap(),
            expected.as_slice()
        );
        fetch.complete();
        drop(fetch);
        assert_eq!(rows.current_epoch(), 0);
        assert_eq!(source.reads.load(Ordering::Acquire), reads);

        // Mismatched ids: dropped, fresh epoch, the requested rows returned.
        let ahead = rows.prefetch_tokens(0, 7, &[5]).unwrap();
        let mut fetch = RowFetch::begin_or_adopt(&rows, Some(ahead), ids.clone()).unwrap();
        assert_eq!(fetch.epoch(), 1);
        assert_eq!(
            fetch.wait().unwrap().as_bytes().unwrap(),
            expected.as_slice()
        );
        fetch.complete();
        drop(fetch);

        // An epoch change (reset/restore) or quiesce (snapshot) stales it.
        let ahead = rows.prefetch_tokens(1, 7, &[3, 4]).unwrap();
        rows.reset_epoch(Duration::from_secs(5)).unwrap();
        assert!(!rows.ticket_matches(&ahead, &ids));
        let mut fetch = RowFetch::begin_or_adopt(&rows, Some(ahead), ids.clone()).unwrap();
        assert_eq!(fetch.epoch(), 3);
        assert_eq!(
            fetch.wait().unwrap().as_bytes().unwrap(),
            expected.as_slice()
        );
        fetch.complete();
        drop(fetch);
        let ahead = rows.prefetch_tokens(3, 7, &[3, 4]).unwrap();
        rows.quiesce(Duration::from_secs(5)).unwrap();
        rows.resume().unwrap();
        assert!(!rows.ticket_matches(&ahead, &ids));
        drop(ahead);

        // An aborted adopted fetch drains into a fresh epoch.
        let ahead = rows.prefetch_tokens(3, 7, &[3, 4]).unwrap();
        let mut fetch = RowFetch::begin_or_adopt(&rows, Some(ahead), ids).unwrap();
        assert_eq!(fetch.epoch(), 3);
        assert_eq!(fetch.abort().unwrap(), 4);
        drop(fetch);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn coalesced_page_plan_is_ordered_and_deduplicated() {
        let pages = vec![
            PageKey { shard: 1, page: 2 },
            PageKey { shard: 0, page: 1 },
            PageKey { shard: 0, page: 0 },
            PageKey { shard: 0, page: 1 },
        ];
        let mut pages = pages;
        pages.sort_unstable();
        pages.dedup();
        assert_eq!(
            pages,
            vec![
                PageKey { shard: 0, page: 0 },
                PageKey { shard: 0, page: 1 },
                PageKey { shard: 1, page: 2 },
            ]
        );
    }
    /// One requested 320-byte row must not drag a multi-megabyte window out of
    /// the source.  The page unit is also the read unit, so this pins the
    /// per-row amplification that the lease wait depends on: with a 2 MiB
    /// window a 291-token chunk read ~9.4 GB for ~1.5 MB of rows, which cost
    /// 492 ms of page-cache copy with the GPU idle.
    #[test]
    fn read_window_stays_within_the_row_page() {
        // Big enough that hashed rows land in distinct windows the way the
        // production table spreads them.
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 32768),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_with_page_rows(
            metadata_with_head_size(4096, 65536),
            32768,
            ROWS_PER_PAGE,
            source.clone(),
        )
        .unwrap();
        let tokens: Vec<u32> = (0..4).collect();
        let ticket = rows.prefetch_tokens(0, 7, &tokens).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (index, &row_id) in expected_ids.iter().enumerate() {
            let row = lease.row_bytes(index).unwrap();
            assert_eq!(u16::from_le_bytes([row[0], row[1]]) as u64, row_id);
        }
        drop(lease);

        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);

        let needed = expected_ids.iter().count() * ROW_BYTES;
        assert!(
            expected_bytes <= needed * 16,
            "requested {needed} bytes and read {expected_bytes} bytes"
        );
        assert!(rows.unload().unwrap().is_clean());
    }

    /// Q8F16 rows for the same fixture layout: five `[f16 scale][32 x i8]`
    /// blocks per row, each block a distinct constant so a mis-ordered block or
    /// a mis-scaled value is visible.
    fn rows_source_q8(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        let blocks = ROW_WIDTH / Q8_BLOCK_WEIGHTS;
        (0..shard_count)
            .map(|shard| {
                let mut bytes = vec![0u8; rows_per_shard * blocks * Q8_BLOCK_BYTES];
                for row in 0..rows_per_shard {
                    let global = (shard * rows_per_shard + row) as f32;
                    for block in 0..blocks {
                        let offset = (row * blocks + block) * Q8_BLOCK_BYTES;
                        // scale = 1.0 for even blocks, 0.5 for odd ones.
                        let scale: f32 = if block % 2 == 0 { 1.0 } else { 0.5 };
                        let bits = f32_to_f16(scale);
                        bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                        for index in 0..Q8_BLOCK_WEIGHTS {
                            // Distinct, in-range quantized values per row/head.
                            let value = ((index as f32) - 16.0 + global) as i8;
                            bytes[offset + 2 + index] = value as u8;
                        }
                    }
                }
                bytes
            })
            .collect()
    }

    fn expected_q8_row(shard: usize, row: usize, rows_per_shard: usize) -> Vec<u8> {
        let blocks = ROW_WIDTH / Q8_BLOCK_WEIGHTS;
        let global = (shard * rows_per_shard + row) as f32;
        let mut out = vec![0u8; ROW_BYTES];
        for block in 0..blocks {
            let scale: f32 = if block % 2 == 0 { 1.0 } else { 0.5 };
            for index in 0..Q8_BLOCK_WEIGHTS {
                let value = scale * (((index as f32) - 16.0 + global) as i8) as f32;
                let cell = (block * Q8_BLOCK_WEIGHTS + index) * 2;
                out[cell..cell + 2].copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
            }
        }
        out
    }

    /// f16 encode for the fixture only (the reader decodes f16; this gives the
    /// fixture exact scales rather than rounded ones).
    fn f32_to_f16(value: f32) -> u16 {
        crate::llama::f32_to_f16(value)
    }

    /// Q4LM rows for the same fixture layout: each Q8F16 fixture row encoded by
    /// [`crate::ple_q4::encode_q4lm_row_from_q8`].
    fn rows_source_q4(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        let q8_row_bytes = RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH);
        let q4_row_bytes = RowEncoding::Q4Lm.encoded_row_bytes(ROW_WIDTH);
        rows_source_q8(shard_count, rows_per_shard)
            .into_iter()
            .map(|q8| {
                let mut q4 = vec![0u8; rows_per_shard * q4_row_bytes];
                for (src, dst) in q8
                    .chunks_exact(q8_row_bytes)
                    .zip(q4.chunks_exact_mut(q4_row_bytes))
                {
                    crate::ple_q4::encode_q4lm_row_from_q8(src, dst).unwrap();
                }
                q4
            })
            .collect()
    }

    #[test]
    fn q8_rows_decode_to_bf16_lease_rows() {
        let rows_per_shard = 64;
        let source = Arc::new(MemoryRowSource {
            shards: rows_source_q8(2, rows_per_shard),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_encoded(
            metadata(128),
            rows_per_shard,
            ROWS_PER_PAGE,
            RowEncoding::Q8F16,
            source.clone(),
        )
        .unwrap();
        assert_eq!(
            RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH) * ROWS_PER_PAGE,
            RowEncoding::Q8F16.page_bytes(ROW_WIDTH)
        );
        let ticket = rows.prefetch_tokens(0, 99, &[0, 1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (index, &row_id) in expected_ids.iter().enumerate() {
            let location = rows.locate_row(row_id).unwrap();
            let want = expected_q8_row(location.shard, location.local_row, rows_per_shard);
            assert_eq!(
                lease.row_bytes(index).unwrap(),
                want.as_slice(),
                "row {row_id} (request index {index}) decoded differently"
            );
        }
        drop(lease);
        // Encoded pages: the lease is still the 320-byte BF16 row shape the
        // device consumes, while the cache and the reads use the 170-byte form.
        let stats = rows.cache_stats();
        assert!(stats.read_bytes > 0);
        for row in expected_ids.iter() {
            let location = rows.locate_row(*row).unwrap();
            assert_eq!(
                rows.inner.page_len(location.page_key()).unwrap(),
                RowEncoding::Q8F16.page_bytes(ROW_WIDTH)
            );
        }
        assert!(rows.unload().unwrap().is_clean());
    }

    /// CPU cost of the Q8F16 row decode, which now sits on the reader thread.
    ///
    ///   cargo test -p hipfire-arch-qwen4 --lib -- q8_decode_cost_probe --ignored --nocapture
    ///
    /// Prints ns per row both ways: straight through the decoder, and inside a
    /// real prefetch over an in-memory shard set (which also carries the page
    /// copy the decode replaces for BF16).
    #[test]
    #[ignore = "CPU timing probe; no GPU or artifact needed"]
    fn q8_decode_cost_probe() {
        let encoded = rows_source_q8(1, 1).remove(0);
        let mut out = vec![0u8; ROW_BYTES];
        const DECODES: u32 = 200_000;
        let started = std::time::Instant::now();
        for _ in 0..DECODES {
            RowEncoding::Q8F16
                .decode_row(&encoded, &mut out)
                .expect("decode");
            std::hint::black_box(&out);
        }
        let ns_per_row = started.elapsed().as_secs_f64() * 1e9 / DECODES as f64;

        let mut arms = Vec::new();
        for encoding in [RowEncoding::Bf16, RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            let rows_per_shard = 4096;
            let shards = match encoding {
                RowEncoding::Bf16 => rows_source(2, rows_per_shard),
                RowEncoding::Q8F16 => rows_source_q8(2, rows_per_shard),
                RowEncoding::Q4Lm => rows_source_q4(2, rows_per_shard),
            };
            let tokens: Vec<u32> = (0..256).collect();
            let rows = RowStore::from_test_source_encoded(
                metadata_with_head_size(512, (2 * rows_per_shard) as u64),
                rows_per_shard,
                ROWS_PER_PAGE,
                encoding,
                Arc::new(MemoryRowSource {
                    shards,
                    reads: AtomicUsize::new(0),
                    fail: None,
                }),
            )
            .unwrap();
            // Warm the page cache so the arm measures copy + decode, not reads.
            let ticket = rows.prefetch_tokens(0, 99, &tokens).unwrap();
            drop(rows.wait_completed_lease(&ticket).unwrap());
            let started = std::time::Instant::now();
            let ticket = rows.prefetch_tokens(0, 99, &tokens).unwrap();
            drop(rows.wait_completed_lease(&ticket).unwrap());
            let elapsed = started.elapsed().as_secs_f64();
            arms.push((
                encoding,
                elapsed * 1e6 / tokens.len() as f64,
                rows.cache_stats(),
            ));
            assert!(rows.unload().unwrap().is_clean());
        }
        println!(
            "q8-decode-probe decode_row={ns_per_row:.1}ns/row ({:.1}ns per 16-row token)",
            ns_per_row * f64::from(ROWS_PER_TOKEN as u32)
        );
        for (encoding, us_per_token, stats) in &arms {
            println!(
                "q8-decode-probe {:?} prefetch={us_per_token:.2}us/token (warm pages, {} reads, \
                 {} read_bytes)",
                encoding, stats.reads, stats.read_bytes
            );
        }
    }

    #[test]
    fn row_encoding_is_declared_by_dtype_not_inferred_from_extent() {
        assert_eq!(RowEncoding::from_dtype("BF16"), Some(RowEncoding::Bf16));
        assert_eq!(RowEncoding::from_dtype("Q8_0"), Some(RowEncoding::Q8F16));
        for foreign in ["F32", "F16", "MQ4G256V2", "MQ6G256V2", "I64", ""] {
            assert_eq!(RowEncoding::from_dtype(foreign), None, "{foreign}");
        }
        // The two tiers differ exactly by the row stride the shard must cover:
        // 320 bytes for a BF16 shard, 5 x 34 for a Q8F16 one.
        assert_eq!(RowEncoding::Bf16.encoded_row_bytes(ROW_WIDTH), ROW_BYTES);
        assert_eq!(
            RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH),
            (ROW_WIDTH / Q8_BLOCK_WEIGHTS) * Q8_BLOCK_BYTES
        );
        assert_eq!(RowEncoding::Q8F16.encoded_row_bytes(ROW_WIDTH), 170);
        assert_eq!(RowEncoding::Q8F16.dtype_name(), "Q8_0");
    }

    #[test]
    fn q4lm_encoding_is_declared_sized_and_decoded() {
        assert_eq!(RowEncoding::from_dtype("Q4LM"), Some(RowEncoding::Q4Lm));
        assert_eq!(RowEncoding::from_dtype("q4lm"), Some(RowEncoding::Q4Lm));
        assert_eq!(RowEncoding::Q4Lm.dtype_name(), "Q4LM");
        assert_eq!(RowEncoding::Q4Lm.encoded_row_bytes(ROW_WIDTH), 90);
        assert_eq!(RowEncoding::Q4Lm.page_bytes(ROW_WIDTH), ROWS_PER_PAGE * 90);
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            assert_eq!(encoding.checked_encoded_row_bytes(0), None);
            assert_eq!(encoding.checked_encoded_row_bytes(33), None);
        }
        assert_eq!(RowEncoding::Bf16.checked_encoded_row_bytes(33), Some(66));
        assert_eq!(RowEncoding::Bf16.checked_encoded_row_bytes(0), None);
        assert_eq!(RowEncoding::Q8F16.checked_encoded_row_bytes(160), Some(170));

        let shards = rows_source_q4(1, 2);
        let row = &shards[0][..90];
        let mut got = vec![0u8; ROW_BYTES];
        RowEncoding::Q4Lm.decode_row(row, &mut got).unwrap();
        let mut want = vec![0u8; ROW_BYTES];
        crate::ple_q4::decode_q4lm_row_bf16(row, &mut want).unwrap();
        assert_eq!(got, want);
        assert!(RowEncoding::Q4Lm.decode_row(&row[..89], &mut got).is_err());
        assert!(RowEncoding::Q4Lm.decode_row(row, &mut got[..318]).is_err());
    }

    /// Sealed single-file reader of the descriptor fixtures below.
    struct FileRowReader {
        identity: crate::model_source::SourceIdentity,
        file: std::fs::File,
    }

    impl crate::model_source::SourceReaderImpl for FileRowReader {
        fn identity(&self) -> &crate::model_source::SourceIdentity {
            &self.identity
        }

        fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            crate::model_source::read_file_exact_at(
                &self.file,
                &self.identity.files[0],
                offset,
                dst,
            )
        }
    }

    /// Sealed descriptors over `path`: one per `(offset, length)` range, each
    /// declaring `dtype` and a `[rows_per_shard, width]` shape (the shape of a
    /// sidecar's manifest).
    fn file_descriptors(
        path: &std::path::Path,
        ranges: &[(u64, u64)],
        dtype: &str,
        rows_per_shard: usize,
        width: usize,
    ) -> Vec<SourceRangeDescriptor> {
        use crate::model_source::{
            capture_file_identity, SourceFormat, SourceIdentity, SourceReader,
        };
        let file_identity = capture_file_identity(path).unwrap();
        let identity = Arc::new(SourceIdentity {
            canonical_path: file_identity.canonical_path.clone(),
            format: SourceFormat::PleRows,
            files: vec![file_identity],
            metadata_json: "{}".to_string(),
            manifest: Vec::new(),
        });
        ranges
            .iter()
            .map(|&(offset, length)| {
                let reader = SourceReader::from_inner(FileRowReader {
                    identity: identity.as_ref().clone(),
                    file: std::fs::File::open(path).unwrap(),
                });
                SourceRangeDescriptor::from_parts(
                    identity.clone(),
                    offset,
                    length,
                    dtype.to_string(),
                    vec![rows_per_shard, width],
                    reader,
                )
                .unwrap()
            })
            .collect()
    }

    /// One file holding a 4 KiB stand-in header and then `shards` back to back,
    /// and one sealed descriptor per shard (see [`file_descriptors`]).
    fn shard_descriptors(
        shards: &[Vec<u8>],
        dtype: &str,
        rows_per_shard: usize,
        width: usize,
    ) -> (tempfile::TempDir, Vec<SourceRangeDescriptor>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.ple-q4");
        let mut file = vec![0u8; 4096];
        let mut ranges = Vec::new();
        for shard in shards {
            ranges.push((file.len() as u64, shard.len() as u64));
            file.extend_from_slice(shard);
        }
        std::fs::write(&path, &file).unwrap();
        let descriptors = file_descriptors(&path, &ranges, dtype, rows_per_shard, width);
        (dir, descriptors)
    }

    /// BF16 rows a Q4LM shard set decodes to for `ids`.
    fn expected_q4_rows(shards: &[Vec<u8>], ids: &[u64], rows_per_shard: usize) -> Vec<u8> {
        let row_bytes = RowEncoding::Q4Lm.encoded_row_bytes(ROW_WIDTH);
        ids.iter()
            .flat_map(|&id| {
                let shard = id as usize / rows_per_shard;
                let local = id as usize % rows_per_shard;
                let row = &shards[shard][local * row_bytes..(local + 1) * row_bytes];
                let mut out = vec![0u8; ROW_BYTES];
                crate::ple_q4::decode_q4lm_row_bf16(row, &mut out).unwrap();
                out
            })
            .collect()
    }

    /// A Q4LM descriptor set opens, refuses bad declarations, and serves
    /// rows through the production constructor in request order.
    #[test]
    fn q4lm_descriptors_round_trip_through_the_row_store() {
        let rows_per_shard = 64;
        let shards = rows_source_q4(2, rows_per_shard);
        let (_dir, descriptors) = shard_descriptors(&shards, "Q4LM", rows_per_shard, ROW_WIDTH);
        let rows = RowStore::new("q4lm-test-reader", descriptors, 128).unwrap();
        let ids = [0u64, 1, 5, 63, 64, 100, 127, 64, 0, 3];
        assert_eq!(
            fetch(&rows, &ids),
            expected_q4_rows(&shards, &ids, rows_per_shard)
        );
        // Cached pages keep the compact 90-byte rows.
        let location = rows.locate_row(100).unwrap();
        assert_eq!(
            rows.inner.page_len(location.page_key()).unwrap(),
            RowEncoding::Q4Lm.page_bytes(ROW_WIDTH)
        );
        assert!(rows.unload().unwrap().is_clean());

        let descriptor_reason = |shards: &[Vec<u8>], dtype: &str, width: usize| {
            let (_dir, descriptors) = shard_descriptors(shards, dtype, 4, width);
            match RowStore::new("q4lm-test-reader", descriptors, 4).err() {
                Some(RowStoreError::Descriptor { reason, .. }) => reason,
                other => panic!("expected a descriptor error, got {other:?}"),
            }
        };
        // A width that is not a whole number of blocks.
        let reason = descriptor_reason(&[vec![0u8; 4 * 18]], "Q4LM", 33);
        assert!(reason.contains("Q4LM") && reason.contains("32"), "{reason}");
        // A shard whose extent is not the declared dtype's.
        let reason = descriptor_reason(&[vec![0u8; 4 * 170]], "Q4LM", ROW_WIDTH);
        assert!(reason.contains("Q4LM"), "{reason}");
        // A dtype with no decoder names the three that exist.
        let reason = descriptor_reason(&[vec![0u8; 4 * 90]], "F32", ROW_WIDTH);
        assert!(
            reason.contains("BF16, Q8_0 or Q4LM") && reason.contains("F32"),
            "{reason}"
        );
    }

    fn q4_source() -> Arc<MemoryRowSource> {
        Arc::new(MemoryRowSource {
            shards: rows_source_q4(2, HOT_ROWS_PER_SHARD),
            reads: AtomicUsize::new(0),
            fail: None,
        })
    }

    fn q4_row_store(source: &Arc<MemoryRowSource>) -> RowStore {
        RowStore::from_test_source_encoded(
            metadata_with_head_size(512, HOT_PADDED_ROWS),
            HOT_ROWS_PER_SHARD,
            ROWS_PER_PAGE,
            RowEncoding::Q4Lm,
            source.clone(),
        )
        .unwrap()
    }

    /// Rows of a Q4LM hot pack decode identically to the row-store path.
    #[test]
    fn q4lm_hot_pack_rows_decode_like_the_row_store() {
        let source = q4_source();
        let ids = mixed_ids();
        let want = expected_q4_rows(&source.shards, &ids, HOT_ROWS_PER_SHARD);
        let plain = q4_row_store(&source);
        assert_eq!(fetch(&plain, &ids), want);

        // Half the rows hot: pages mix hot and cold requested rows.
        let pack = PackFixture::with_encoding(
            &source.shards,
            HOT_ROWS_PER_SHARD,
            128,
            4,
            77,
            RowEncoding::Q4Lm,
            ROW_WIDTH,
        );
        assert_eq!(pack.store.encoded_row_bytes(), 90);
        let rows = q4_row_store(&source);
        rows.attach_hot_store(pack.store.clone()).unwrap();
        assert_eq!(fetch(&rows, &ids), want);
        let stats = rows.cache_stats();
        assert!(stats.hot_rows > 0 && stats.hot_pages > 0 && stats.cache_misses > 0);

        // Every row hot: the source is not read at all.
        let all = PackFixture::with_encoding(
            &source.shards,
            HOT_ROWS_PER_SHARD,
            256,
            4,
            5,
            RowEncoding::Q4Lm,
            ROW_WIDTH,
        );
        let rows = q4_row_store(&source);
        rows.attach_hot_store(all.store.clone()).unwrap();
        assert_eq!(fetch(&rows, &ids), want);
        let stats = rows.cache_stats();
        assert_eq!((stats.reads, stats.cache_misses), (0, 0));
        assert_eq!(stats.hot_rows as usize, ids.len());

        // A Q8F16 table does not take a Q4LM pack.
        let q8_table = q8_row_store(&q8_source());
        let err = q8_table.attach_hot_store(all.store.clone()).unwrap_err();
        assert!(err.contains("Q4LM") && err.contains("Q8_0"), "{err}");
    }

    /// A rank-ordered Q4LM table read through `RowStore::from_table_store`
    /// returns the same decoded rows as a row-ordered table of the same rows
    /// read through shard descriptors; it serves at once and never touches the
    /// source path.
    #[test]
    fn rank_ordered_table_store_decodes_like_row_ordered_descriptors() {
        use crate::ple_hot_pack::table_support::write_test_table;
        use crate::ple_hot_pack::PackOrder;
        use crate::ple_hot_store::test_support::synthetic_buckets;
        use crate::ple_hot_store::DEFAULT_TIER_BUCKETS;

        let shards = rows_source_q4(2, HOT_ROWS_PER_SHARD);
        let buckets = synthetic_buckets(2 * HOT_ROWS_PER_SHARD, 17);
        let dir = tempfile::tempdir().unwrap();
        let rank_path = dir.path().join("rank.ple-q4");
        let row_path = dir.path().join("row.ple-q4");
        write_test_table(
            &rank_path,
            PackOrder::Rank,
            RowEncoding::Q4Lm,
            ROW_WIDTH,
            HOT_ROWS_PER_SHARD,
            &shards,
            Some(&buckets),
            DEFAULT_TIER_BUCKETS,
        );
        let row_header = write_test_table(
            &row_path,
            PackOrder::Row,
            RowEncoding::Q4Lm,
            ROW_WIDTH,
            HOT_ROWS_PER_SHARD,
            &shards,
            None,
            DEFAULT_TIER_BUCKETS,
        );
        let shard_bytes =
            (HOT_ROWS_PER_SHARD * RowEncoding::Q4Lm.encoded_row_bytes(ROW_WIDTH)) as u64;
        let ranges: Vec<(u64, u64)> = (0..2)
            .map(|shard| (row_header.data_offset + shard * shard_bytes, shard_bytes))
            .collect();
        let descriptors =
            file_descriptors(&row_path, &ranges, "Q4LM", HOT_ROWS_PER_SHARD, ROW_WIDTH);
        let by_descriptors = RowStore::new("row-order", descriptors, HOT_PADDED_ROWS).unwrap();

        let store = HotRowStore::open_table(&rank_path).unwrap();
        assert!(store.is_serving(), "a table serves at once");
        assert_eq!(store.row_width(), ROW_WIDTH);
        assert_eq!(store.kept_buckets(), 256);
        let by_table =
            RowStore::from_table_store("rank-order", store.clone(), HOT_PADDED_ROWS, 0, true)
                .unwrap();

        let ids = mixed_ids();
        let want = expected_q4_rows(&shards, &ids, HOT_ROWS_PER_SHARD);
        assert_eq!(fetch(&by_descriptors, &ids), want);
        assert_eq!(fetch(&by_table, &ids), want);
        let one = [ids[7]];
        assert_eq!(
            fetch(&by_table, &one),
            expected_q4_rows(&shards, &one, HOT_ROWS_PER_SHARD)
        );
        let twice = [ids[9], ids[9]];
        assert_eq!(
            fetch(&by_table, &twice),
            expected_q4_rows(&shards, &twice, HOT_ROWS_PER_SHARD)
        );
        let stats = by_table.cache_stats();
        assert_eq!(
            (stats.reads, stats.cache_misses),
            (0, 0),
            "every row is in the pack"
        );
        assert_eq!(stats.hot_rows as usize, ids.len() + 3);
        assert!(stats.hot_pages > 0);
        let descriptor_stats = by_descriptors.cache_stats();
        assert!(descriptor_stats.cache_misses > 0);
        assert_eq!(descriptor_stats.hot_rows, 0);

        // The row store's activity is the table's.
        assert!(!by_table.activity().idle_for(Duration::from_secs(3600)));

        // Bad valid-row counts, and a store binds one row store only.
        assert!(RowStore::from_table_store("x", store.clone(), 0, 0, true).is_err());
        assert!(
            RowStore::from_table_store("x", store.clone(), HOT_PADDED_ROWS + 1, 0, true).is_err()
        );
        match RowStore::from_table_store("x", store.clone(), HOT_PADDED_ROWS, 0, true).err() {
            Some(RowStoreError::Descriptor { reason, .. }) => {
                assert!(reason.contains("another row store"), "{reason}")
            }
            other => panic!("expected the bound-store refusal, got {other:?}"),
        }
        assert!(by_table.unload().unwrap().is_clean());
    }

    /// The positional fallback of a table store reads whole rows at their pack
    /// offsets and refuses anything else.
    #[test]
    fn table_row_source_reads_rows_from_the_pack() {
        use crate::ple_hot_pack::table_support::write_test_table;
        use crate::ple_hot_pack::PackOrder;
        use crate::ple_hot_store::test_support::synthetic_buckets;
        use crate::ple_hot_store::DEFAULT_TIER_BUCKETS;

        const RPS: usize = 256;
        let shards = rows_source_q4(3, RPS);
        let buckets = synthetic_buckets(3 * RPS, 5);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rank.ple-q4");
        write_test_table(
            &path,
            PackOrder::Rank,
            RowEncoding::Q4Lm,
            ROW_WIDTH,
            RPS,
            &shards,
            Some(&buckets),
            DEFAULT_TIER_BUCKETS,
        );
        let source = TableRowSource {
            store: HotRowStore::open_table(&path).unwrap(),
            shard_count: 3,
            rows_per_shard: RPS as u64,
            row_bytes: 90,
        };
        for (shard, first, count) in [
            (0usize, 0usize, 1usize),
            (1, 17, 5),
            (2, 250, 6),
            (0, 0, RPS),
        ] {
            let mut got = vec![0u8; count * 90];
            source
                .read_at(shard, (first * 90) as u64, &mut got)
                .unwrap();
            assert_eq!(
                got,
                shards[shard][first * 90..(first + count) * 90],
                "{shard}/{first}+{count}"
            );
        }
        let mut unaligned = vec![0u8; 90];
        assert!(source.read_at(0, 1, &mut unaligned).is_err());
        assert!(source.read_at(0, 0, &mut unaligned[..89]).is_err());
        let mut past_shard = vec![0u8; 7 * 90];
        assert!(source.read_at(2, 250 * 90, &mut past_shard).is_err());
        assert!(source.read_at(3, 0, &mut unaligned).is_err());
    }

    /// Every fetch marks the row store's shared activity busy, with or without
    /// a hot store, and the hot store pauses on that same activity.
    #[test]
    fn fetches_mark_the_shared_activity_busy() {
        let source = q8_source();
        let rows = q8_row_store(&source);
        let activity = rows.activity();
        assert!(Arc::ptr_eq(&activity, &rows.activity()));
        assert!(
            activity.idle_for(Duration::from_secs(3600)),
            "never fetched"
        );
        fetch(&rows, &[1, 2, 3]);
        assert_eq!(activity.in_flight(), 0);
        assert!(
            !activity.idle_for(Duration::from_secs(3600)),
            "a fetch just ran"
        );
        assert!(activity.idle_for(Duration::ZERO));

        let pack = hot_pack(&source, 64, 4, 3);
        rows.attach_hot_store(pack.store.clone()).unwrap();
        // Rebinding the same activity is fine, another row store's is not.
        pack.store.bind_activity(activity.clone()).unwrap();
        let other = q8_row_store(&source);
        let err = other.attach_hot_store(pack.store.clone()).unwrap_err();
        assert!(err.contains("another row store"), "{err}");
    }

    #[test]
    fn direct_fill_handles_partial_pages_duplicates_and_cache_pressure() {
        for encoding in [RowEncoding::Bf16, RowEncoding::Q8F16] {
            assert_eq!(
                encoding.page_bytes(ROW_WIDTH) % encoding.encoded_row_bytes(ROW_WIDTH),
                0
            );
        }

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_with_page_rows(
            metadata_with_head_size(8, 128),
            64,
            5,
            source.clone(),
        )
        .unwrap();
        assert_eq!(rows.inner.staging_bytes % ROW_BYTES, 0);
        {
            let mut state = rows.inner.state.lock().expect("test state mutex poisoned");
            state.cache.capacity_bytes = ROW_BYTES * 10;
        }

        let ticket = rows.prefetch_tokens(0, 999, &[0, 0]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..ROWS_PER_TOKEN {
                let row = lease.row_bytes(token * ROWS_PER_TOKEN + head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token * ROWS_PER_TOKEN + head]);
            }
        }
        let first_stats = rows.cache_stats();
        assert!(first_stats.reads > 1);
        assert_eq!(
            source.reads.load(Ordering::Acquire) as u64,
            first_stats.reads
        );
        assert!(first_stats.resident_bytes <= ROW_BYTES * 10);
        drop(lease);

        let ticket = rows.prefetch_tokens(0, 999, &[1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (head, &row_id) in lease.row_ids()[..ROWS_PER_TOKEN].iter().enumerate() {
            let row = lease.row_bytes(head).unwrap();
            let value = u16::from_le_bytes([row[0], row[1]]) as u64;
            assert_eq!(value, row_id);
        }
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        drop(lease);

        let stats = rows.cache_stats();
        assert!(stats.reads > first_stats.reads);
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        assert!(stats.cache_hits > 0);
        assert!(stats.cache_misses > 0);
        assert!(stats.evictions > 0);
        assert!(stats.resident_bytes <= ROW_BYTES * 10);
        assert!(rows.unload().unwrap().is_clean());
    }

    /// A request of thousands of scattered groups decodes on the reader
    /// threads and caches its pages after the ticket completes: the rows are
    /// the table's bytes in request order, and the next ticket for the same
    /// rows is served from the cache without a read.
    #[test]
    fn large_multi_group_fetch_is_byte_identical_and_cached_after_completion() {
        let rows_per_shard = 16384;
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, rows_per_shard),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source_with_page_rows(
            metadata_with_head_size(2048, 32768),
            rows_per_shard,
            1,
            source.clone(),
        )
        .unwrap();
        // Stride-4 rows are one-page groups; every seventh also asks for its
        // neighbour (a two-page group) and some ids repeat.
        let mut ids: Vec<u64> = Vec::new();
        for index in 0..8000u64 {
            ids.push(index * 4);
            if index % 7 == 0 {
                ids.push(index * 4 + 1);
            }
            if index % 11 == 0 {
                ids.push(index * 4);
            }
        }
        assert!(ids.len() <= rows.max_rows_per_prefetch());
        let distinct: std::collections::BTreeSet<u64> = ids.iter().copied().collect();
        let expected: Vec<u8> = ids
            .iter()
            .flat_map(|&id| {
                let shard = id as usize / rows_per_shard;
                let row = id as usize % rows_per_shard;
                source.shards[shard][row * ROW_BYTES..(row + 1) * ROW_BYTES].to_vec()
            })
            .collect();

        let ticket = rows.prefetch(0, ids.clone()).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), ids.as_slice());
        assert_eq!(lease.as_bytes().unwrap(), expected.as_slice());
        drop(lease);
        let first = rows.cache_stats();
        assert_eq!(first.cache_hits, 0);
        assert_eq!(first.cache_misses as usize, distinct.len());
        assert!(first.reads > 1 && (first.reads as usize) < distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, first.reads);
        assert_eq!(first.read_bytes as usize, distinct.len() * ROW_BYTES);

        // The worker is serial, so this ticket runs after the first one's
        // pages were inserted.
        let second = rows.prefetch(0, ids.clone()).unwrap();
        let lease = rows.wait_completed_lease(&second).unwrap();
        assert_eq!(lease.as_bytes().unwrap(), expected.as_slice());
        drop(lease);
        let stats = rows.cache_stats();
        assert_eq!(stats.cache_hits as usize, distinct.len());
        assert_eq!(stats.cache_misses, first.cache_misses);
        assert_eq!(stats.reads, first.reads);
        assert_eq!(stats.read_bytes, first.read_bytes);
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, first.reads);
        assert_eq!(stats.resident_pages, distinct.len());
        assert_eq!(stats.resident_bytes, distinct.len() * ROW_BYTES);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn epoch_reset_stales_lease_and_enforces_monotonic_epochs() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        rows.begin_epoch(0).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        std::thread::scope(|scope| {
            let resetter = scope.spawn(|| rows.reset_epoch(Duration::from_secs(1)));
            while rows.current_epoch() != 1 {
                std::thread::yield_now();
            }
            assert!(matches!(
                lease.validate(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.as_bytes(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            let mut destination =
                vec![0u8; (lease.row_count() / ROWS_PER_TOKEN) * ROWS_PER_TOKEN * ROW_BYTES];
            assert!(matches!(
                lease.stage_into(&mut destination),
                Err(RowStoreError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.validate_after_upload(),
                Err(RowStoreError::LeaseStale { .. })
            ));
            drop(lease);
            assert_eq!(resetter.join().unwrap().unwrap(), 1);
        });

        assert!(matches!(
            rows.begin_epoch(1),
            Err(RowStoreError::EpochNotMonotonic {
                requested: 1,
                current: 1
            })
        ));
        rows.begin_epoch(2).unwrap();
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn cancel_releases_queued_ticket_slot_while_lease_holds_staging() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let first = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&first).unwrap();

        let mut queued = Vec::with_capacity(READER_QUEUE_CAPACITY);
        for _ in 0..READER_QUEUE_CAPACITY {
            queued.push(rows.prefetch_tokens(0, 99, &[0]).unwrap());
        }
        assert!(matches!(
            rows.prefetch_tokens(0, 99, &[0]),
            Err(RowStoreError::QueueFull)
        ));
        rows.cancel(&queued[0]).unwrap();
        let replacement = rows.prefetch_tokens(0, 99, &[0]).unwrap();

        drop(queued);
        drop(replacement);
        drop(lease);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn unload_waits_for_a_live_lease_before_releasing_the_reader() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = RowStore::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch_tokens(0, 99, &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        let unload = std::thread::spawn(move || rows.unload());
        for _ in 0..128 {
            assert!(!unload.is_finished());
            std::thread::yield_now();
        }
        drop(lease);
        assert!(unload.join().unwrap().unwrap().is_clean());
    }

    // ---- PLE hot-row store ------------------------------------------------

    const HOT_ROWS_PER_SHARD: usize = 4096;
    const HOT_PADDED_ROWS: u64 = 2 * HOT_ROWS_PER_SHARD as u64;

    fn q8_source() -> Arc<MemoryRowSource> {
        Arc::new(MemoryRowSource {
            shards: rows_source_q8(2, HOT_ROWS_PER_SHARD),
            reads: AtomicUsize::new(0),
            fail: None,
        })
    }

    fn q8_row_store(source: &Arc<MemoryRowSource>) -> RowStore {
        RowStore::from_test_source_encoded(
            metadata_with_head_size(512, HOT_PADDED_ROWS),
            HOT_ROWS_PER_SHARD,
            ROWS_PER_PAGE,
            RowEncoding::Q8F16,
            source.clone(),
        )
        .unwrap()
    }

    use crate::ple_hot_store::test_support::PackFixture;

    /// A model file holding `source`'s shards, its hot-row pack of `kept` of
    /// the 256 buckets of a pseudo-random ranking, and the verified store.
    fn hot_pack(source: &MemoryRowSource, kept: usize, group: usize, seed: u64) -> PackFixture {
        PackFixture::new(&source.shards, HOT_ROWS_PER_SHARD, kept, group, seed)
    }

    fn fetch(rows: &RowStore, ids: &[u64]) -> Vec<u8> {
        let ticket = rows.prefetch(0, ids.to_vec()).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), ids);
        lease.as_bytes().unwrap().to_vec()
    }

    fn expected_rows(ids: &[u64]) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| {
                let shard = id as usize / HOT_ROWS_PER_SHARD;
                let row = id as usize % HOT_ROWS_PER_SHARD;
                expected_q8_row(shard, row, HOT_ROWS_PER_SHARD)
            })
            .collect()
    }

    /// Scattered ids with repeats, plus one contiguous run so many pages hold
    /// both hot and cold requested rows.
    fn mixed_ids() -> Vec<u64> {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut ids: Vec<u64> = (0..3000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % HOT_PADDED_ROWS
            })
            .collect();
        ids.extend(100..260);
        ids.extend(HOT_ROWS_PER_SHARD as u64 - 20..HOT_ROWS_PER_SHARD as u64 + 20);
        ids
    }

    /// Output bytes are identical with and without the pack for random row
    /// sets (partially hot pages included) and with an unverified pack; hot
    /// pages are counted separately from hits/misses, which keep their
    /// meaning for the remaining pages.
    #[test]
    fn hot_store_output_is_byte_identical_with_partial_pages() {
        let source = q8_source();
        let ids = mixed_ids();
        let want = expected_rows(&ids);
        let distinct_pages: std::collections::BTreeSet<u64> =
            ids.iter().map(|id| id / ROWS_PER_PAGE as u64).collect();

        let plain = q8_row_store(&source);
        assert_eq!(fetch(&plain, &ids), want);
        let plain_stats = plain.cache_stats();
        assert_eq!(plain_stats.cache_misses as usize, distinct_pages.len());
        assert_eq!(
            (
                plain_stats.hot_rows,
                plain_stats.hot_pages,
                plain_stats.hot_reads
            ),
            (0, 0, 0)
        );

        // Half the rows hot: many pages mix hot and cold requested rows.
        let pack = hot_pack(&source, 128, 4, 77);
        let rows = q8_row_store(&source);
        rows.attach_hot_store(pack.store.clone()).unwrap();
        assert_eq!(fetch(&rows, &ids), want);
        let stats = rows.cache_stats();
        assert!(stats.hot_pages > 0 && stats.hot_rows >= stats.hot_pages);
        assert!(stats.hot_reads > 0 && stats.hot_reads <= stats.hot_rows);
        assert!(
            (stats.hot_pages as usize) < distinct_pages.len(),
            "some pages stay cold"
        );
        assert_eq!(
            stats.cache_misses + stats.hot_pages,
            plain_stats.cache_misses,
            "hot pages leave hits/misses untouched for every other page"
        );
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(pack.store.hot_rows(), stats.hot_rows);

        // A pack that was not verified serves nothing.
        let rank = crate::ple_hot_store::PleHotRank::from_buckets(
            HOT_PADDED_ROWS,
            crate::ple_hot_store::test_support::FINGERPRINT,
            pack.buckets.clone(),
        );
        let model_bytes = std::fs::metadata(&pack.model_path).unwrap().len();
        let unverified = HotRowStore::open(
            rank,
            &pack.pack_path,
            HOT_ROWS_PER_SHARD,
            RowEncoding::Q8F16,
            170,
            &pack.offsets,
            model_bytes,
        )
        .unwrap();
        let rows = q8_row_store(&source);
        rows.attach_hot_store(unverified.clone()).unwrap();
        assert_eq!(fetch(&rows, &ids), want);
        let stats = rows.cache_stats();
        assert_eq!(
            (stats.hot_rows, stats.hot_pages, stats.hot_reads),
            (0, 0, 0)
        );
        assert_eq!(stats.cache_misses, plain_stats.cache_misses);
    }

    /// With every bucket in the pack nothing is read from the source, for a
    /// large fetch, a single row, and a single run of consecutive pack rows.
    #[test]
    fn an_all_hot_request_reads_only_the_pack() {
        let source = q8_source();
        let pack = hot_pack(&source, 256, 4, 5);
        let rows = q8_row_store(&source);
        rows.attach_hot_store(pack.store.clone()).unwrap();
        let ids = mixed_ids();
        assert_eq!(fetch(&rows, &ids), expected_rows(&ids));
        let one = [ids[7]];
        assert_eq!(fetch(&rows, &one), expected_rows(&one));
        let same_row_twice = [ids[9], ids[9]];
        assert_eq!(
            fetch(&rows, &same_row_twice),
            expected_rows(&same_row_twice)
        );
        let stats = rows.cache_stats();
        assert_eq!(stats.reads, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.hot_rows as usize, ids.len() + 3);
    }

    /// Concurrent requests through one row store with a pack attached all
    /// return the source's bytes.
    #[test]
    fn concurrent_requests_with_a_pack_are_byte_identical() {
        let source = q8_source();
        let pack = hot_pack(&source, 200, 3, 13);
        let rows = Arc::new(q8_row_store(&source));
        rows.attach_hot_store(pack.store.clone()).unwrap();
        let ids = mixed_ids();
        let want = expected_rows(&ids);
        std::thread::scope(|scope| {
            for round in 0..4 {
                let rows = rows.clone();
                let (ids, want) = (&ids, &want);
                scope.spawn(move || {
                    for _ in 0..4 {
                        assert_eq!(&fetch(&rows, ids), want, "round {round}");
                        rows.evict_cached_pages();
                    }
                });
            }
        });
    }

    /// A store whose geometry is not the table's is refused, as is a second
    /// store.
    #[test]
    fn attach_hot_store_validates_geometry() {
        let source = q8_source();
        let rows = q8_row_store(&source);
        // Same bytes cut into shards of half the size.
        let halves: Vec<Vec<u8>> = source
            .shards
            .iter()
            .flat_map(|shard| {
                let half = shard.len() / 2;
                [shard[..half].to_vec(), shard[half..].to_vec()]
            })
            .collect();
        let wrong_shards = PackFixture::new(&halves, HOT_ROWS_PER_SHARD / 2, 64, 4, 3);
        assert!(rows.attach_hot_store(wrong_shards.store.clone()).is_err());
        let wrong_rows = PackFixture::new(&source.shards[..1], HOT_ROWS_PER_SHARD, 64, 4, 3);
        assert!(rows.attach_hot_store(wrong_rows.store.clone()).is_err());
        let pack = hot_pack(&source, 64, 4, 3);
        rows.attach_hot_store(pack.store.clone()).unwrap();
        let second = hot_pack(&source, 32, 4, 3);
        assert!(rows.attach_hot_store(second.store.clone()).is_err());
        // Same geometry and row size, other row encoding: BF16 rows of 85
        // values are 170 B too, but are not the table's Q8F16 rows.
        let fresh = q8_row_store(&source);
        let other_encoding = PackFixture::with_encoding(
            &source.shards,
            HOT_ROWS_PER_SHARD,
            64,
            4,
            3,
            RowEncoding::Bf16,
            85,
        );
        assert_eq!(other_encoding.store.encoded_row_bytes(), 170);
        let err = fresh
            .attach_hot_store(other_encoding.store.clone())
            .unwrap_err();
        assert!(err.contains("BF16") && err.contains("Q8_0"), "{err}");
        fresh
            .attach_hot_store(hot_pack(&source, 64, 4, 3).store.clone())
            .unwrap();
    }
}
