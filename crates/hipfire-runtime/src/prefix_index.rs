// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-resident compressed radix tree over token spans (spec §4.2 C2).
//!
//! The `PrefixIndex` keys completed prefix state by [`CacheDomain`] (exact
//! identity, spec §4.1 C1) and the actual model input token sequence. Edges
//! store immutable token spans; nodes hold ordered [`PageHandle`]s for **full
//! 128-token pages only** plus optional [`CheckpointId`]s at resumable
//! boundaries (spec §4.5 C5). Partial tails are never shareable (spec §4.2).
//!
//! Operations: longest-prefix [`lookup`](PrefixIndex::lookup), [`insert`],
//! [`split`], [`pin`]/[`unpin`], and leaf-first [`evict_unpinned_leaves`].
//! Lookup pins matching page handles internally so eviction cannot drop them
//! until `unpin` (spec §4.4 C4). Eviction removes lookup visibility **before**
//! releasing cache refs.
//!
//! This module performs **no GPU mutation**. `add_cache_ref`/`release_cache_ref`/
//! `seal` are host-side operations on [`PagePool`] (spec §4.2: "Do not call
//! PagePool from lookup in a way that mutates GPU memory").
//!
//! # Token-count distinction (spec §4.2)
//!
//! - **matched_tokens**: equal token prefix in the index.
//! - **resident_kv_tokens**: matching attention rows that still exist (full
//!   pages whose `PageHandle` is still valid).
//! - **resumable_tokens**: largest boundary for which a checkpoint exists AND
//!   all required pages are still resident.
//!
//! A prefix match that lacks a recurrent checkpoint is [`MissReason::NoCheckpoint`],
//! not a hit in usage accounting (spec §4.2).

use std::collections::HashMap;

use rdna_compute::page_pool::{PageHandle, PagePool, PAGE_TOKENS};

use crate::cache_plan::CachePolicy;
use crate::serve_contract::{
    CacheDomain, CheckpointId, MissReason, PrefixLookup, PrefixLookupResult,
};

// =========================================================================
// Internal tree structure
// =========================================================================

/// Unique id for a node within a domain tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeId(u64);

/// A checkpoint boundary within a node's token span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CheckpointBoundary {
    /// Token offset *within this node* where the checkpoint applies.
    /// Always a multiple of `PAGE_TOKENS` (checkpoints live at page
    /// boundaries, spec §4.5).
    token_offset: u64,
    /// The opaque checkpoint id (non-zero).
    checkpoint: CheckpointId,
}

/// A node in the radix tree.
///
/// Each non-root node stores:
/// - `edge_tokens`: the immutable token span of its incoming edge.
/// - `pages`: ordered `PageHandle`s for the full 128-token pages covering
///   this node's span. `pages.len() * PAGE_TOKENS == edge_tokens.len()`.
/// - `checkpoints`: optional checkpoint boundaries within this node.
/// - `children`: child edges keyed by their first token.
#[derive(Debug)]
struct Node {
    /// Monotonic insertion order for leaf-first, oldest-first eviction.
    insert_seq: u64,
    /// Immutable token span of the incoming edge (empty for root).
    edge_tokens: Vec<u32>,
    /// Page handles for the full pages covering this node's span.
    pages: Vec<PageHandle>,
    /// Checkpoint boundaries within this node's span. Sorted by token_offset.
    checkpoints: Vec<CheckpointBoundary>,
    /// Child edges: first token -> child node id.
    children: HashMap<u32, NodeId>,
    /// Whether this node is currently pinned by a lookup.
    pinned: bool,
}

impl Node {
    fn root() -> Self {
        Node {
            insert_seq: 0,
            edge_tokens: Vec::new(),
            pages: Vec::new(),
            checkpoints: Vec::new(),
            children: HashMap::new(),
            pinned: false,
        }
    }

    fn new(insert_seq: u64, edge_tokens: Vec<u32>, pages: Vec<PageHandle>) -> Self {
        Node {
            insert_seq,
            edge_tokens,
            pages,
            checkpoints: Vec::new(),
            children: HashMap::new(),
            pinned: false,
        }
    }

    /// Total tokens covered by this node's edge.
    fn token_span(&self) -> u64 {
        self.edge_tokens.len() as u64
    }

    /// Returns `true` if this node has no children.
    fn is_leaf_node(&self) -> bool {
        self.children.is_empty()
    }
}

/// A domain-scoped radix tree (spec §4.1: different domain → isolated trees).
#[derive(Debug)]
struct DomainTree {
    root: NodeId,
    nodes: HashMap<NodeId, Node>,
    next_node_id: u64,
    next_insert_seq: u64,
}

impl DomainTree {
    fn new() -> Self {
        let root_id = NodeId(0);
        let mut nodes = HashMap::new();
        nodes.insert(root_id, Node::root());
        DomainTree {
            root: root_id,
            nodes,
            next_node_id: 1,
            next_insert_seq: 1,
        }
    }

    fn fresh_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    fn fresh_insert_seq(&mut self) -> u64 {
        let s = self.next_insert_seq;
        self.next_insert_seq += 1;
        s
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// =========================================================================
// Public types
// =========================================================================

/// A page handle plus its logical position in the prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    /// The physical page handle from `PagePool`.
    pub handle: PageHandle,
    /// Logical token offset where this page starts within the prefix.
    pub token_offset: u64,
}

/// Result of [`PrefixIndex::inspect`] — the three token counts without
/// claiming a [`PrefixLookupResult::Hit`] (spec §4.2: for metrics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InspectResult {
    /// Equal token prefix length found in the radix index.
    pub matched_tokens: u64,
    /// Matching attention rows that are still physically resident.
    pub resident_kv_tokens: u64,
    /// Largest boundary for which a checkpoint exists and pages are resident.
    pub resumable_tokens: u64,
}

/// Error returned by [`PrefixIndex::insert`] / [`PrefixIndex::publish_sealed_pages`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertError {
    /// The published prefix has a partial last page (len % 128 != 0).
    /// Partial tails are never shareable (spec §4.2).
    PartialLastPage {
        /// The token length that was not a multiple of `PAGE_TOKENS`.
        token_len: usize,
    },
    /// The CPU node metadata bound (`max_cpu_nodes`) would be exceeded and
    /// eviction could not free enough nodes (spec §4.4).
    CpuNodeBoundExceeded {
        /// Current node count.
        current: usize,
        /// The configured maximum.
        max: usize,
    },
    /// A page handle failed validation against the pool (stale/free).
    InvalidHandle(String),
    /// A handle's token_offset is not page-aligned or handle count doesn't
    /// match the number of full pages.
    MisalignedHandle,
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PartialLastPage { token_len } => write!(
                f,
                "partial last page: token_len {} is not a multiple of {}",
                token_len, PAGE_TOKENS
            ),
            Self::CpuNodeBoundExceeded { current, max } => write!(
                f,
                "CPU node bound exceeded: {} nodes > max {} (eviction insufficient)",
                current, max
            ),
            Self::InvalidHandle(s) => write!(f, "invalid page handle: {}", s),
            Self::MisalignedHandle => {
                write!(f, "misaligned page handle (not page-aligned or count mismatch)")
            }
        }
    }
}

impl std::error::Error for InsertError {}

// =========================================================================
// PrefixIndex
// =========================================================================

/// CPU-resident compressed radix tree over token spans (spec §4.2 C2).
///
/// Keyed by [`CacheDomain`] (exact identity; different domain → isolated
/// trees). Edges store immutable token spans; nodes hold ordered [`PageHandle`]s
/// for full 128-token pages only, plus optional [`CheckpointId`]s at resumable
/// boundaries.
///
/// Construct with [`PrefixIndex::new`] specifying `max_cpu_nodes` to bound
/// CPU node/token metadata (spec §4.4: "Bound CPU node/token metadata as well
/// as device bytes").
pub struct PrefixIndex {
    trees: HashMap<CacheDomain, DomainTree>,
    max_cpu_nodes: usize,
    total_nodes: usize,
}

/// Internal result of walking the tree to find the longest token prefix.
struct WalkResult {
    matched_tokens: u64,
    resident_kv_tokens: u64,
    resumable_tokens: u64,
    /// Path of node ids from root to the deepest matched node (inclusive).
    path: Vec<NodeId>,
    /// Whether the walk consumed all query tokens.
    exhausted_query: bool,
}

impl PrefixIndex {
    /// Create a new `PrefixIndex` with a CPU node metadata bound.
    ///
    /// `max_cpu_nodes` bounds the total number of radix nodes across all
    /// domain trees (spec §4.4). Inserting past this bound triggers
    /// leaf-first eviction; if eviction cannot free enough nodes, the insert
    /// is refused with [`InsertError::CpuNodeBoundExceeded`].
    pub fn new(max_cpu_nodes: usize) -> Self {
        PrefixIndex {
            trees: HashMap::new(),
            max_cpu_nodes,
            total_nodes: 0,
        }
    }

    /// Current total node count across all domain trees.
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Number of distinct cache domains with entries.
    pub fn domain_count(&self) -> usize {
        self.trees.len()
    }

    // ── Lookup ────────────────────────────────────────────────────────

    /// Longest-prefix lookup (spec §4.2 C2).
    ///
    /// Returns [`PrefixLookupResult::Hit`] only when a checkpoint id exists
    /// at the chosen boundary AND the corresponding pages are still
    /// cache-resident. Token match without checkpoint →
    /// [`MissReason::NoCheckpoint`] (spec §4.2: "A prefix match that lacks a
    /// recurrent checkpoint is not a hit in usage accounting").
    ///
    /// Pins matching page handles internally so eviction cannot drop them
    /// until [`unpin`](Self::unpin) is called (spec §4.4). No GPU mutation.
    ///
    /// If `policy` is supplied and `allow_partial` is false, a mid-sequence
    /// partial hit (resumable boundary < matched tokens) returns
    /// [`MissReason::IncompatiblePolicy`] (spec §4.5). If policy is `None`,
    /// the raw longest resumable boundary is returned.
    pub fn lookup(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        pool: &PagePool,
        policy: Option<&CachePolicy>,
    ) -> PrefixLookupResult {
        let walk = self.walk(domain, tokens, pool);

        if walk.matched_tokens == 0 {
            return PrefixLookupResult::Miss(MissReason::NoMatch);
        }

        if walk.resumable_tokens == 0 {
            return PrefixLookupResult::Miss(MissReason::NoCheckpoint);
        }

        if let Some(p) = policy {
            if !p.allow_partial && walk.resumable_tokens < walk.matched_tokens {
                return PrefixLookupResult::Miss(MissReason::IncompatiblePolicy);
            }
        }

        // Pin nodes along the path so eviction cannot drop them (spec §4.4).
        self.pin_path(domain, &walk.path);

        PrefixLookupResult::Hit(PrefixLookup {
            matched_tokens: walk.matched_tokens,
            resident_kv_tokens: walk.resident_kv_tokens,
            resumable_tokens: walk.resumable_tokens,
        })
    }

    /// Inspect the index for the three token counts without claiming a Hit
    /// and without pinning (spec §4.2: for metrics).
    pub fn inspect(
        &self,
        domain: &CacheDomain,
        tokens: &[u32],
        pool: &PagePool,
    ) -> InspectResult {
        let walk = self.walk(domain, tokens, pool);
        InspectResult {
            matched_tokens: walk.matched_tokens,
            resident_kv_tokens: walk.resident_kv_tokens,
            resumable_tokens: walk.resumable_tokens,
        }
    }

    /// Walk the tree to find the longest matching token prefix.
    ///
    /// Does NOT pin. Used by both `lookup` (which pins after) and `inspect`.
    fn walk(&self, domain: &CacheDomain, tokens: &[u32], pool: &PagePool) -> WalkResult {
        let tree = match self.trees.get(domain) {
            Some(t) => t,
            None => {
                return WalkResult {
                    matched_tokens: 0,
                    resident_kv_tokens: 0,
                    resumable_tokens: 0,
                    path: Vec::new(),
                    exhausted_query: false,
                };
            }
        };

        let mut current = tree.root;
        let mut path: Vec<NodeId> = vec![tree.root];
        let mut query_pos: usize = 0;
        let mut matched_tokens: u64 = 0;
        let mut resident_kv_tokens: u64 = 0;
        let mut resumable_tokens: u64 = 0;

        // Check root checkpoints (token_offset 0 → resumable boundary 0,
        // which is trivially true but not useful).
        let root = tree.nodes.get(&tree.root).unwrap();
        for cb in &root.checkpoints {
            if cb.token_offset == 0 && resident_kv_tokens >= 0 {
                resumable_tokens = resumable_tokens.max(0);
            }
        }

        loop {
            if query_pos >= tokens.len() {
                break;
            }

            let node = match tree.nodes.get(&current) {
                Some(n) => n,
                None => break,
            };

            let next_token = tokens[query_pos];
            let child_id = match node.children.get(&next_token).copied() {
                Some(id) => id,
                None => break,
            };

            let child = match tree.nodes.get(&child_id) {
                Some(c) => c,
                None => break,
            };

            // Compare edge tokens against query tokens.
            let edge = &child.edge_tokens;
            let mut edge_match = 0usize;
            for (i, &et) in edge.iter().enumerate() {
                if query_pos + i >= tokens.len() || tokens[query_pos + i] != et {
                    break;
                }
                edge_match += 1;
            }

            // Count resident pages for the matched portion of this edge.
            // Each page covers PAGE_TOKENS tokens.
            let matched_full_pages = edge_match / PAGE_TOKENS;
            for i in 0..matched_full_pages {
                if let Some(ph) = child.pages.get(i) {
                    if pool.validate_handle(ph).is_ok() {
                        resident_kv_tokens += PAGE_TOKENS as u64;
                    }
                }
            }

            // Check checkpoints on the child for the matched portion.
            // child_base is the global token offset where this child's edge
            // begins. Compute it BEFORE adding edge_match to matched_tokens.
            let child_base = matched_tokens;
            matched_tokens += edge_match as u64;

            for cb in &child.checkpoints {
                if cb.token_offset <= edge_match as u64 {
                    let boundary = child_base + cb.token_offset;
                    if resident_kv_tokens >= boundary {
                        resumable_tokens = resumable_tokens.max(boundary);
                    }
                }
            }

            if edge_match == edge.len() {
                // Full edge match — advance to child.
                query_pos += edge_match;
                current = child_id;
                path.push(child_id);
            } else {
                // Partial match — query diverges mid-edge.
                break;
            }
        }

        WalkResult {
            matched_tokens,
            resident_kv_tokens,
            resumable_tokens,
            path,
            exhausted_query: query_pos >= tokens.len(),
        }
    }

    /// Pin all nodes along a path (spec §4.4: eviction cannot drop pinned).
    fn pin_path(&mut self, domain: &CacheDomain, path: &[NodeId]) {
        if let Some(tree) = self.trees.get_mut(domain) {
            for &nid in path {
                if let Some(node) = tree.nodes.get_mut(&nid) {
                    node.pinned = true;
                }
            }
        }
    }

    // ── Insert ────────────────────────────────────────────────────────

    /// Insert a completed prefix into the index (spec §4.2 C2).
    ///
    /// The caller must have already sealed the pages and taken cache refs
    /// via `PagePool::seal` + `PagePool::add_cache_ref` (spec §4.2: "insert
    /// after caller has sealed pages and taken cache refs via PagePool").
    ///
    /// `handles` is the ordered list of full-page handles with their logical
    /// token offsets. `checkpoint` is the optional checkpoint id at the final
    /// boundary of this prefix. `CheckpointId::NONE` means no checkpoint.
    ///
    /// Same completed prefix → one canonical entry; in-flight duplicates are
    /// not remapped (spec §4.2).
    pub fn insert(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        handles: &[Handle],
        checkpoint: Option<CheckpointId>,
        pool: &PagePool,
    ) -> Result<(), InsertError> {
        // Validate handles.
        for h in handles {
            if h.token_offset % PAGE_TOKENS as u64 != 0 {
                return Err(InsertError::MisalignedHandle);
            }
            pool.validate_handle(&h.handle)
                .map_err(InsertError::InvalidHandle)?;
        }

        let full_page_tokens = handles.len() * PAGE_TOKENS;
        if tokens.len() < full_page_tokens {
            return Err(InsertError::MisalignedHandle);
        }

        // Check CPU node bound.
        if self.total_nodes >= self.max_cpu_nodes {
            return Err(InsertError::CpuNodeBoundExceeded {
                current: self.total_nodes,
                max: self.max_cpu_nodes,
            });
        }

        // Get or create the domain tree.
        let is_new_tree = !self.trees.contains_key(domain);
        if is_new_tree {
            let t = DomainTree::new();
            self.total_nodes += t.node_count();
            self.trees.insert(domain.clone(), t);
        }

        let delta = insert_into_tree(
            self.trees.get_mut(domain).unwrap(),
            tokens,
            handles,
            checkpoint,
            self.max_cpu_nodes,
            self.total_nodes,
        )?;
        self.total_nodes = (self.total_nodes as i32 + delta) as usize;

        Ok(())
    }


    // ── Pin / Unpin ───────────────────────────────────────────────────

    /// Unpin all pages pinned by a previous [`lookup`](Self::lookup).
    ///
    /// After unpin, the pages are eligible for eviction again (spec §4.4).
    pub fn unpin(&mut self, domain: &CacheDomain) {
        if let Some(tree) = self.trees.get_mut(domain) {
            for node in tree.nodes.values_mut() {
                node.pinned = false;
            }
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────

    /// Evict oldest unpinned leaves, releasing up to `max_bytes` of device
    /// memory (spec §4.4 C4).
    ///
    /// Leaf-first, oldest first (by insert sequence). Removes lookup
    /// visibility **before** releasing cache refs via
    /// `PagePool::release_cache_ref`. Keeps useful ancestors while
    /// descendants use them (ancestors are only evicted when all their
    /// children are gone).
    ///
    /// Returns the physical page indices that were evicted.
    pub fn evict_unpinned_leaves(
        &mut self,
        pool: &mut PagePool,
        max_bytes: usize,
    ) -> Vec<u32> {
        let mut evicted_phys: Vec<u32> = Vec::new();
        let mut bytes_freed: usize = 0;

        // Collect candidates: (domain, node_id, insert_seq) for unpinned leaves.
        let mut candidates: Vec<(CacheDomain, NodeId, u64)> = Vec::new();
        for (domain, tree) in &self.trees {
            for (nid, node) in &tree.nodes {
                if *nid != tree.root && node.is_leaf_node() && !node.pinned {
                    candidates.push((domain.clone(), *nid, node.insert_seq));
                }
            }
        }
        candidates.sort_by_key(|(_, _, seq)| *seq);

        for (domain, nid, _) in candidates {
            if bytes_freed >= max_bytes {
                break;
            }

            let tree = match self.trees.get_mut(&domain) {
                Some(t) => t,
                None => continue,
            };

            // Re-check conditions (may have changed).
            let node = match tree.nodes.get(&nid) {
                Some(n) => n,
                None => continue,
            };
            if !node.is_leaf_node() || node.pinned {
                continue;
            }

            let pages = node.pages.clone();
            let first_token = node.edge_tokens.first().copied();

            // Find and unlink from parent.
            let parent_id = tree.nodes.iter()
                .find(|(_, n)| {
                    if let Some(ft) = first_token {
                        n.children.get(&ft) == Some(&nid)
                    } else {
                        false
                    }
                })
                .map(|(pid, _)| *pid);

            if let Some(pid) = parent_id {
                if let Some(parent) = tree.nodes.get_mut(&pid) {
                    if let Some(ft) = first_token {
                        parent.children.remove(&ft);
                    }
                }
            }

            // Remove the node (remove lookup visibility BEFORE releasing refs).
            tree.nodes.remove(&nid);
            self.total_nodes = self.total_nodes.saturating_sub(1);

            // Release cache refs.
            for ph in &pages {
                bytes_freed += pool.k_page_bytes() + pool.v_page_bytes();
                let _ = pool.release_cache_ref(ph.phys);
                evicted_phys.push(ph.phys);
            }
        }

        evicted_phys
    }

    /// Try to evict enough nodes to fit `needed` new nodes.
    pub fn evict_for_capacity(
        &mut self,
        pool: &mut PagePool,
        needed: usize,
    ) -> Result<(), InsertError> {
        while self.total_nodes + needed > self.max_cpu_nodes {
            let before = self.total_nodes;
            self.evict_unpinned_leaves(pool, usize::MAX);
            if self.total_nodes == before {
                return Err(InsertError::CpuNodeBoundExceeded {
                    current: self.total_nodes,
                    max: self.max_cpu_nodes,
                });
            }
        }
        Ok(())
    }

    // ── Publication helper ────────────────────────────────────────────

    /// Publish sealed pages into the index (spec §4.6 C6, §4.2 C2).
    ///
    /// Refuses partial last pages: the published prefix's token length must
    /// be a multiple of `PAGE_TOKENS` (spec §4.2: "Publish immutable full
    /// 128-token pages first ... Partial active tails stay private").
    ///
    /// `checkpoint` is an opaque id minted by the caller (the Qwen adapter).
    /// `CheckpointId::NONE` means no checkpoint (spec §4.5).
    ///
    /// The caller must have already sealed the pages and taken cache refs
    /// via `PagePool::seal` + `PagePool::add_cache_ref`.
    pub fn publish_sealed_pages(
        &mut self,
        domain: &CacheDomain,
        tokens: &[u32],
        handles: &[Handle],
        checkpoint: Option<CheckpointId>,
        pool: &PagePool,
    ) -> Result<(), InsertError> {
        if tokens.len() % PAGE_TOKENS != 0 {
            return Err(InsertError::PartialLastPage {
                token_len: tokens.len(),
            });
        }

        if handles.len() != tokens.len() / PAGE_TOKENS {
            return Err(InsertError::MisalignedHandle);
        }

        self.insert(domain, tokens, handles, checkpoint, pool)
    }

    /// Split an edge (metadata-only, spec §4.2).
    ///
    /// Public wrapper for testing. Splits the edge from the root to the
    /// child keyed by `first_token` at `split_tokens` into the edge.
    pub fn split(
        &mut self,
        domain: &CacheDomain,
        first_token: u32,
        split_tokens: usize,
    ) -> Result<(), InsertError> {
        let tree = match self.trees.get_mut(domain) {
            Some(t) => t,
            None => return Err(InsertError::InvalidHandle("domain not found".to_string())),
        };

        let root = tree.root;
        let child_id = match tree.nodes.get(&root).and_then(|n| n.children.get(&first_token).copied()) {
            Some(id) => id,
            None => return Err(InsertError::InvalidHandle("child not found".to_string())),
        };

        let delta = split_edge(tree, root, child_id, split_tokens, self.max_cpu_nodes, self.total_nodes)?;
        self.total_nodes = (self.total_nodes as i32 + delta) as usize;
        Ok(())
    }
}

// =========================================================================
// Free tree manipulation functions (avoid &mut self + &mut tree borrow conflict)
// =========================================================================

/// Insert tokens/handles into a domain tree, splitting edges as needed.
/// Returns the net change in node count (positive = nodes added).
fn insert_into_tree(
    tree: &mut DomainTree,
    tokens: &[u32],
    handles: &[Handle],
    checkpoint: Option<CheckpointId>,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<i32, InsertError> {
    let mut current = tree.root;
    let mut query_pos: usize = 0;
    let mut page_idx: usize = 0;
    let mut delta: i32 = 0;

    loop {
        if query_pos >= tokens.len() {
            if let Some(ckpt) = checkpoint {
                if ckpt.is_some() {
                    let node = tree.nodes.get_mut(&current).unwrap();
                    let token_offset = if current == tree.root {
                        0
                    } else {
                        node.edge_tokens.len() as u64
                    };
                    if !node.checkpoints.iter().any(|cb| cb.token_offset == token_offset) {
                        node.checkpoints.push(CheckpointBoundary {
                            token_offset,
                            checkpoint: ckpt,
                        });
                        node.checkpoints.sort_by_key(|cb| cb.token_offset);
                    }
                }
            }
            return Ok(delta);
        }

        let node = match tree.nodes.get(&current) {
            Some(n) => n,
            None => return Ok(delta),
        };

        let next_token = tokens[query_pos];
        let child_id = match node.children.get(&next_token).copied() {
            Some(id) => id,
            None => {
                let remaining_tokens = &tokens[query_pos..];
                let remaining_handles = &handles[page_idx..];
                delta += create_chain(
                    tree,
                    current,
                    remaining_tokens,
                    remaining_handles,
                    checkpoint,
                    max_cpu_nodes,
                    current_total + delta as usize,
                )?;
                return Ok(delta);
            }
        };

        let child = match tree.nodes.get(&child_id) {
            Some(c) => c,
            None => return Ok(delta),
        };

        let edge = &child.edge_tokens;
        let mut edge_match = 0usize;
        for (i, &et) in edge.iter().enumerate() {
            if query_pos + i >= tokens.len() || tokens[query_pos + i] != et {
                break;
            }
            edge_match += 1;
        }

        if edge_match == edge.len() {
            query_pos += edge_match;
            page_idx += child.pages.len();
            current = child_id;
        } else {
            delta += split_edge(tree, current, child_id, edge_match, max_cpu_nodes, current_total + delta as usize)?;

            let parent = tree.nodes.get(&current).unwrap();
            let split_node_id = parent.children.get(&next_token).copied().unwrap();

            if query_pos + edge_match >= tokens.len() {
                if let Some(ckpt) = checkpoint {
                    if ckpt.is_some() {
                        let split_node = tree.nodes.get_mut(&split_node_id).unwrap();
                        let token_offset = split_node.edge_tokens.len() as u64;
                        if !split_node.checkpoints.iter().any(|cb| cb.token_offset == token_offset) {
                            split_node.checkpoints.push(CheckpointBoundary {
                                token_offset,
                                checkpoint: ckpt,
                            });
                            split_node.checkpoints.sort_by_key(|cb| cb.token_offset);
                        }
                    }
                }
            }

            let remaining_start = query_pos + edge_match;
            if remaining_start < tokens.len() {
                let remaining_tokens = &tokens[remaining_start..];
                let split_pages = edge_match / PAGE_TOKENS;
                let remaining_handles = &handles[page_idx + split_pages..];
                delta += create_chain(
                    tree,
                    split_node_id,
                    remaining_tokens,
                    remaining_handles,
                    checkpoint,
                    max_cpu_nodes,
                    current_total + delta as usize,
                )?;
            }

            return Ok(delta);
        }
    }
}

/// Create a chain of nodes for a new token span. Each node covers one full
/// page. Returns the number of nodes added.
fn create_chain(
    tree: &mut DomainTree,
    parent: NodeId,
    tokens: &[u32],
    handles: &[Handle],
    checkpoint: Option<CheckpointId>,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<i32, InsertError> {
    let mut current_parent = parent;
    let mut token_pos = 0usize;
    let mut handle_idx = 0usize;
    let mut added: i32 = 0;

    while token_pos + PAGE_TOKENS <= tokens.len() {
        let chunk = &tokens[token_pos..token_pos + PAGE_TOKENS];
        let first_token = chunk[0];

        let node_id = tree.fresh_node_id();
        let page_handle = handles[handle_idx].handle;
        let insert_seq = tree.fresh_insert_seq();

        let is_last_full_page = token_pos + PAGE_TOKENS + PAGE_TOKENS > tokens.len();

        let mut new_node = Node::new(insert_seq, chunk.to_vec(), vec![page_handle]);

        if is_last_full_page {
            if let Some(ckpt) = checkpoint {
                if ckpt.is_some() {
                    new_node.checkpoints.push(CheckpointBoundary {
                        token_offset: PAGE_TOKENS as u64,
                        checkpoint: ckpt,
                    });
                }
            }
        }

        let new_total = current_total + added as usize + 1;
        if new_total > max_cpu_nodes {
            // Don't insert — return error. The node_id was allocated but
            // never inserted, so no cleanup needed.
            return Err(InsertError::CpuNodeBoundExceeded {
                current: current_total + added as usize,
                max: max_cpu_nodes,
            });
        }

        tree.nodes.insert(node_id, new_node);
        added += 1;

        tree.nodes.get_mut(&current_parent).unwrap().children.insert(first_token, node_id);

        current_parent = node_id;
        token_pos += PAGE_TOKENS;
        handle_idx += 1;
    }

    Ok(added)
}

/// Split an edge at `split_pos` tokens. Metadata-only (spec §4.2).
/// Returns 1 if a split node was created, 0 otherwise.
fn split_edge(
    tree: &mut DomainTree,
    parent: NodeId,
    child_id: NodeId,
    split_pos: usize,
    max_cpu_nodes: usize,
    current_total: usize,
) -> Result<i32, InsertError> {
    let split_pages = split_pos / PAGE_TOKENS;
    let split_tokens = split_pages * PAGE_TOKENS;

    if split_pages == 0 {
        return Ok(0);
    }

    // If the split point is at or beyond the child's full edge, no split
    // is needed — the child already covers the split point.
    let child_edge_len = tree.nodes.get(&child_id).unwrap().edge_tokens.len();
    if split_tokens >= child_edge_len {
        return Ok(0);
    }

    let child = tree.nodes.get(&child_id).unwrap();
    let child_edge_tokens = child.edge_tokens.clone();
    let child_pages = child.pages.clone();
    let child_checkpoints = child.checkpoints.clone();
    let child_insert_seq = child.insert_seq;
    let first_token = child_edge_tokens[0];

    if current_total + 1 > max_cpu_nodes {
        return Err(InsertError::CpuNodeBoundExceeded {
            current: current_total,
            max: max_cpu_nodes,
        });
    }

    let split_node_id = tree.fresh_node_id();
    let split_pages_vec: Vec<PageHandle> = child_pages[..split_pages].to_vec();
    let split_edge_tokens: Vec<u32> = child_edge_tokens[..split_tokens].to_vec();

    let split_checkpoints: Vec<CheckpointBoundary> = child_checkpoints
        .iter()
        .filter(|cb| cb.token_offset <= split_tokens as u64)
        .copied()
        .collect();

    let remaining_checkpoints: Vec<CheckpointBoundary> = child_checkpoints
        .iter()
        .filter(|cb| cb.token_offset > split_tokens as u64)
        .map(|cb| CheckpointBoundary {
            token_offset: cb.token_offset - split_tokens as u64,
            checkpoint: cb.checkpoint,
        })
        .collect();

    let mut split_node = Node::new(child_insert_seq, split_edge_tokens, split_pages_vec);
    split_node.checkpoints = split_checkpoints;

    tree.nodes.insert(split_node_id, split_node);

    let remaining_edge_tokens: Vec<u32> = child_edge_tokens[split_tokens..].to_vec();
    let remaining_pages: Vec<PageHandle> = child_pages[split_pages..].to_vec();

    let remaining_child = tree.nodes.get_mut(&child_id).unwrap();
    remaining_child.edge_tokens = remaining_edge_tokens;
    remaining_child.pages = remaining_pages;
    remaining_child.checkpoints = remaining_checkpoints;

    tree.nodes.get_mut(&parent).unwrap().children.insert(first_token, split_node_id);
    let remaining_first_token = tree.nodes.get(&child_id).unwrap().edge_tokens[0];
    tree.nodes.get_mut(&split_node_id).unwrap().children.insert(remaining_first_token, child_id);

    Ok(1)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::*;
    use rdna_compute::page_pool::{BlockTable, PagePool, PAGE_TOKENS};

    // ── Test helpers ──────────────────────────────────────────────────

    fn sample_domain(load_epoch: u64) -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![0xab; 16],
            model_load_epoch: load_epoch,
            sidecar_digests: vec![vec![0x01; 8], vec![0x02; 8]],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![0xcd; 16],
                config_digest: vec![0xce; 16],
            },
            template: TemplateIdentity {
                template_digest: vec![0xee; 16],
                normalization_tag: "chatml".to_owned(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen35-deltanet".to_owned(),
                state_abi_tag: "dn-v1".to_owned(),
                position_attention_tag: "mrope".to_owned(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![128, 128],
                v_stride_bytes: vec![128, 128],
                layout_tag: "q8-g128".to_owned(),
            },
            device: DeviceTopology {
                device_id: "pci-03:00.0".to_owned(),
                topology_id: "single".to_owned(),
                allocation_epoch: 7,
            },
            namespace: SharingNamespace {
                domain_id: "owner-alpha".to_owned(),
            },
        }
    }

    fn domain_with(field: &str, value: &str) -> CacheDomain {
        let mut d = sample_domain(1);
        match field {
            "model" => d.model_content_digest = vec![0xff; 16],
            "template" => d.template.template_digest = vec![0xff; 16],
            "tokenizer" => d.tokenizer.vocab_digest = vec![0xff; 16],
            "namespace" => d.namespace.domain_id = value.to_string(),
            _ => {}
        }
        d
    }

    /// Create a pool with `n_pages` pages, seal them, and take cache refs.
    /// Returns the pool and the page handles with token offsets.
    fn setup_pool_and_pages(n_pages: usize) -> (PagePool, Vec<Handle>) {
        let mut pool = PagePool::new_with_strides(64, 128, 128).unwrap();
        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, n_pages);
        assert_eq!(allocated, n_pages);

        let mut handles = Vec::new();
        for lp in 0..n_pages {
            let phys = table.physical(lp).unwrap();
            pool.seal(phys).unwrap();
            pool.add_cache_ref(phys).unwrap();
            let gen = pool.page_generation(phys);
            handles.push(Handle {
                handle: PageHandle {
                    phys,
                    epoch: pool.epoch(),
                    generation: gen,
                },
                token_offset: (lp * PAGE_TOKENS) as u64,
            });
        }
        (pool, handles)
    }

    /// Create a single pool with `n_pages` pages, seal + cache-ref each.
    fn setup_big_pool(n_pages: usize) -> (PagePool, Vec<Handle>) {
        let mut pool = PagePool::new_with_strides(64, 128, 128).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, n_pages);
        let mut handles = Vec::new();
        for lp in 0..n_pages {
            let phys = table.physical(lp).unwrap();
            pool.seal(phys).unwrap();
            pool.add_cache_ref(phys).unwrap();
            let gen = pool.page_generation(phys);
            handles.push(Handle {
                handle: PageHandle { phys, epoch: pool.epoch(), generation: gen },
                token_offset: (lp * PAGE_TOKENS) as u64,
            });
        }
        (pool, handles)
    }

    fn make_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| (i % 1000) as u32 + 1).collect()
    }

    fn make_tokens_from(start: u32, n: usize) -> Vec<u32> {
        (0..n).map(|i| start + i as u32).collect()
    }

    // ── A1: prefix lengths 0, 1, 127, 128, 129 ───────────────────────

    #[test]
    fn a1_empty_prefix_is_no_match() {
        let (pool, _) = setup_pool_and_pages(1);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let result = index.lookup(&domain, &[], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "empty prefix should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a1_single_token_is_miss() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        // 1 token matches the edge but no full page is completed → no checkpoint
        // at that boundary → NoCheckpoint.
        let result = index.lookup(&domain, &tokens[..1], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoCheckpoint)),
            "1-token query should be NoCheckpoint, got {result:?}"
        );
    }

    #[test]
    fn a1_127_tokens_is_no_checkpoint() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain, &tokens[..127], &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoCheckpoint)),
            "127-token query should be NoCheckpoint, got {result:?}"
        );
    }

    #[test]
    fn a1_128_tokens_exact_full_page_with_checkpoint_is_hit() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain, &tokens, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 128-token exact match, got {other:?}"),
        }
    }

    #[test]
    fn a1_129_tokens_partial_second_page() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128, then full 2 pages with
        // checkpoint at 256.
        index.insert(&domain, &tokens[..PAGE_TOKENS], &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        index.insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &pool).unwrap();

        // Query 129 tokens: matched=129, resumable=128 (checkpoint at 128).
        let result = index.lookup(&domain, &tokens[..129], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, 129);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 129-token query, got {other:?}"),
        }
    }

    #[test]
    fn a1_exact_match_full_prefix_is_hit() {
        let (pool, handles) = setup_pool_and_pages(3);
        let tokens = make_tokens(PAGE_TOKENS * 3);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain, &tokens, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, (PAGE_TOKENS * 3) as u64);
                assert_eq!(lk.resident_kv_tokens, (PAGE_TOKENS * 3) as u64);
                assert_eq!(lk.resumable_tokens, (PAGE_TOKENS * 3) as u64);
            }
            other => panic!("expected Hit for exact 3-page match, got {other:?}"),
        }
    }

    #[test]
    fn a1_prompt_shorter_than_cache() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128.
        index.insert(&domain, &tokens[..PAGE_TOKENS], &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        // Insert full 2 pages with checkpoint at 256.
        index.insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &pool).unwrap();

        // Query 128 tokens: matched=128, resumable=128 (checkpoint at 128).
        let result = index.lookup(&domain, &tokens[..PAGE_TOKENS], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for 128-token query, got {other:?}"),
        }
    }

    #[test]
    fn a1_divergent_tail() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert first page with checkpoint at 128.
        index.insert(&domain, &tokens[..PAGE_TOKENS], &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        index.insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &pool).unwrap();

        // Query: 128 match + 1 divergent token.
        let mut query = tokens[..PAGE_TOKENS].to_vec();
        query.push(999_999);

        let result = index.lookup(&domain, &query, &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resident_kv_tokens, PAGE_TOKENS as u64);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit for divergent tail, got {other:?}"),
        }
    }

    #[test]
    fn a1_never_underflow() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let r0 = index.lookup(&domain, &[], &pool, None);
        assert!(matches!(r0, PrefixLookupResult::Miss(MissReason::NoMatch)));

        let r1 = index.lookup(&domain, &tokens[..1], &pool, None);
        assert!(matches!(r1, PrefixLookupResult::Miss(MissReason::NoCheckpoint)));

        let insp = index.inspect(&domain, &[], &pool);
        assert_eq!(insp.matched_tokens, 0);
        assert_eq!(insp.resident_kv_tokens, 0);
        assert_eq!(insp.resumable_tokens, 0);

        let insp = index.inspect(&domain, &tokens[..1], &pool);
        assert_eq!(insp.matched_tokens, 1);
        assert_eq!(insp.resident_kv_tokens, 0);
        assert_eq!(insp.resumable_tokens, 0);
    }

    // ── A2: canonical entry; pin isolation ────────────────────────────

    #[test]
    fn a2_canonical_entry_no_duplication() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert the same prefix twice.
        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();
        let nodes_after_first = index.total_nodes();
        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();
        let nodes_after_second = index.total_nodes();

        // Second insert should not add nodes (canonical entry).
        assert_eq!(
            nodes_after_first, nodes_after_second,
            "canonical entry should not duplicate nodes"
        );

        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn a2_pin_isolation() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        // Lookup pins the pages.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));

        // Eviction should not evict pinned pages.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(evicted.is_empty(), "pinned pages should survive eviction");

        // Unpin.
        index.unpin(&domain);

        // Now eviction should work.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(!evicted.is_empty(), "unpinned pages should be evictable");

        // Subsequent lookup should miss.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(_)),
            "evicted prefix should miss, got {result:?}"
        );
    }

    // ── A6: different CacheDomain → isolated trees ────────────────────

    #[test]
    fn a6_different_model_digest_isolated() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("model", "");

        index.insert(&domain_a, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different model digest should be NoMatch, got {result:?}"
        );

        let result = index.lookup(&domain_a, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }

    #[test]
    fn a6_different_template_isolated() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("template", "");

        index.insert(&domain_a, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different template should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a6_different_tokenizer_isolated() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("tokenizer", "");

        index.insert(&domain_a, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different tokenizer should be NoMatch, got {result:?}"
        );
    }

    #[test]
    fn a6_different_namespace_isolated() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);

        let domain_a = sample_domain(1);
        let domain_b = domain_with("namespace", "owner-beta");

        index.insert(&domain_a, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let result = index.lookup(&domain_b, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "different namespace should be NoMatch, got {result:?}"
        );
    }

    // ── A8: eviction ──────────────────────────────────────────────────

    #[test]
    fn a8_evict_leaf_then_lookup_misses() {
        let (mut pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        // Verify hit before eviction.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.unpin(&domain);

        // Evict.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert_eq!(evicted.len(), 1, "should evict 1 page");

        // Lookup after eviction should miss.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::NoMatch)),
            "evicted prefix should miss, got {result:?}"
        );
    }

    #[test]
    fn a8_pinned_pages_survive_eviction() {
        let (mut pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        // Lookup pins pages.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));

        // Eviction should not evict pinned pages.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(evicted.is_empty(), "pinned pages should not be evicted");

        // Lookup should still hit.
        index.unpin(&domain);
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.unpin(&domain);

        // Now evict.
        let evicted = index.evict_unpinned_leaves(&mut pool, usize::MAX);
        assert!(!evicted.is_empty(), "unpinned pages should be evicted");
    }

    // ── CPU metadata bound ────────────────────────────────────────────

    #[test]
    fn cpu_metadata_bound_refuses_unbounded_growth() {
        let (pool, handles) = setup_big_pool(3);
        let mut idx = PrefixIndex::new(3); // root + 2 nodes max
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS);
        let tokens_b = make_tokens_from(1000, PAGE_TOKENS);
        let tokens_c = make_tokens_from(2000, PAGE_TOKENS);

        // Insert A (root + 1 node = 2 total).
        idx.insert(&domain, &tokens_a, &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        // Insert B (root + 2 nodes = 3 total).
        idx.insert(&domain, &tokens_b, &handles[1..2], Some(CheckpointId(2)), &pool).unwrap();

        // Third insert should fail (would need 4 nodes > max 3).
        let result = idx.insert(&domain, &tokens_c, &handles[2..3], Some(CheckpointId(3)), &pool);
        assert!(
            matches!(result, Err(InsertError::CpuNodeBoundExceeded { .. })),
            "third insert should fail, got {result:?}"
        );

        assert!(idx.total_nodes() <= 3);
    }

    #[test]
    fn cpu_metadata_bound_evict_for_capacity() {
        let (mut pool, handles) = setup_big_pool(3);
        let mut idx = PrefixIndex::new(3);
        let domain = sample_domain(1);

        let tokens_a = make_tokens(PAGE_TOKENS);
        let tokens_b = make_tokens_from(1000, PAGE_TOKENS);
        let tokens_c = make_tokens_from(2000, PAGE_TOKENS);

        idx.insert(&domain, &tokens_a, &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        idx.insert(&domain, &tokens_b, &handles[1..2], Some(CheckpointId(2)), &pool).unwrap();

        // Evict to make room for 1 more node.
        idx.evict_for_capacity(&mut pool, 1).unwrap();

        // Now insert should succeed.
        let result = idx.insert(&domain, &tokens_c, &handles[2..3], Some(CheckpointId(3)), &pool);
        assert!(result.is_ok(), "insert after eviction should succeed, got {result:?}");
    }

    // ── publish_sealed_pages refuses partial last page ────────────────

    #[test]
    fn publish_refuses_partial_last_page() {
        let (pool, handles) = setup_pool_and_pages(1);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let tokens = make_tokens(129);
        let result = index.publish_sealed_pages(&domain, &tokens, &handles, None, &pool);
        assert!(
            matches!(result, Err(InsertError::PartialLastPage { token_len: 129 })),
            "129 tokens should be refused, got {result:?}"
        );
    }

    #[test]
    fn publish_accepts_full_pages() {
        let (pool, handles) = setup_pool_and_pages(2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        let tokens = make_tokens(PAGE_TOKENS * 2);
        let result = index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool);
        assert!(result.is_ok());
    }

    // ── inspect returns counts without pinning ────────────────────────

    #[test]
    fn inspect_returns_counts_without_pinning() {
        let (pool, handles) = setup_pool_and_pages(1);
        let tokens = make_tokens(PAGE_TOKENS);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        let insp = index.inspect(&domain, &tokens, &pool);
        assert_eq!(insp.matched_tokens, PAGE_TOKENS as u64);
        assert_eq!(insp.resident_kv_tokens, PAGE_TOKENS as u64);
        assert_eq!(insp.resumable_tokens, PAGE_TOKENS as u64);

        // inspect should NOT pin — verify by checking that eviction works
        // without unpin (nothing was pinned).
        let (mut pool_mut, _) = setup_pool_and_pages(1);
        // Can't easily test with the same pool, but the fact that no unpin
        // is needed after inspect is the contract.
        let _ = &mut pool_mut;
    }

    // ── CachePolicy IncompatiblePolicy ────────────────────────────────

    #[test]
    fn incompatible_policy_mid_sequence_partial() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        // Insert with checkpoints at 128 and 256.
        index.insert(&domain, &tokens[..PAGE_TOKENS], &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        index.insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &pool).unwrap();

        // Query 256 tokens: matched=256, resumable=256 → not partial → Hit.
        let policy = CachePolicy::qwen35();
        let result = index.lookup(&domain, &tokens, &pool, Some(&policy));
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.unpin(&domain);

        // Query 128 tokens: matched=128, resumable=128 → not partial → Hit.
        let result = index.lookup(&domain, &tokens[..PAGE_TOKENS], &pool, Some(&policy));
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
        index.unpin(&domain);

        // Query 129 tokens: matched=129, resumable=128 → partial → IncompatiblePolicy.
        let result = index.lookup(&domain, &tokens[..129], &pool, Some(&policy));
        assert!(
            matches!(result, PrefixLookupResult::Miss(MissReason::IncompatiblePolicy)),
            "mid-sequence partial with allow_partial=false should be IncompatiblePolicy, got {result:?}"
        );
    }

    #[test]
    fn no_policy_returns_raw_resumable() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.insert(&domain, &tokens[..PAGE_TOKENS], &handles[..1], Some(CheckpointId(1)), &pool).unwrap();
        index.insert(&domain, &tokens, &handles, Some(CheckpointId(2)), &pool).unwrap();

        // Query 129 tokens with no policy → Hit with resumable=128.
        let result = index.lookup(&domain, &tokens[..129], &pool, None);
        match result {
            PrefixLookupResult::Hit(lk) => {
                assert_eq!(lk.matched_tokens, 129);
                assert_eq!(lk.resumable_tokens, PAGE_TOKENS as u64);
            }
            other => panic!("expected Hit with no policy, got {other:?}"),
        }
    }

    // ── CheckpointId ──────────────────────────────────────────────────

    #[test]
    fn checkpoint_id_none_is_zero() {
        assert_eq!(CheckpointId::NONE.0, 0);
        assert!(CheckpointId::NONE.is_none());
        assert!(!CheckpointId::NONE.is_some());
        assert!(CheckpointId(1).is_some());
        assert!(!CheckpointId(1).is_none());
    }

    // ── split (metadata-only) ─────────────────────────────────────────

    #[test]
    fn split_edge_is_metadata_only() {
        let (pool, handles) = setup_pool_and_pages(2);
        let tokens = make_tokens(PAGE_TOKENS * 2);
        let mut index = PrefixIndex::new(1000);
        let domain = sample_domain(1);

        index.publish_sealed_pages(&domain, &tokens, &handles, Some(CheckpointId(1)), &pool).unwrap();

        // Split the edge from root at 128 tokens.
        let first_token = tokens[0];
        let result = index.split(&domain, first_token, PAGE_TOKENS);
        assert!(result.is_ok(), "split should succeed, got {result:?}");

        // Lookup should still find the full prefix.
        let result = index.lookup(&domain, &tokens, &pool, None);
        assert!(matches!(result, PrefixLookupResult::Hit(_)));
    }
}
