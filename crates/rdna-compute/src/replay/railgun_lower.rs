// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun's gfx12 lowering transport, extracted from the M1/M2 shadow
//! (`railgun_shadow::lower`) so a *directly authored* program can use it
//! without any Redline recording (design A §1 "Its emission transport can be
//! reused; its recorder/planner must not be used for this feature", slice
//! A1).
//!
//! * [`emit_segment`] is the one Pm4Op→command emission loop. The legacy M1
//!   shadow (`railgun_shadow::lower`) now calls it with the closure that does
//!   what its inline loop did, so the legacy AR/M2 path emits the same
//!   dwords in the same order.
//! * [`lower_owned`] lowers a single-segment, all-PM4 [`rk::KernelProgram`]
//!   with railgun's own kernarg images and owns everything the queue reads:
//!   loaded kernels, kernarg buffers and the retained indirect buffer. It
//!   needs no `RecordedHipLaunch`, `ReplayBindings` or Redline kernarg
//!   buffer. [`OwnedWindowLowering`] is the GPU-facing half of
//!   `railgun::window::PreparedWindow`: the integration owner implements
//!   `railgun::window::WindowQueue` over it (write images, patch grids,
//!   publish, replay).

use std::collections::BTreeMap;

use railgun::kernel as rk;
use railgun::plan::{self as rp, Pm4Op, Rung};
use redline_dispatch::aql::{load_symbols, Executable, GpuSelector, Runtime};

use super::railgun_shadow::Transport;
use super::{
    gfx10_dispatch_initiator_policy, gfx1151_cu_count_knob_refusal, gfx1151_dispatch_interleave,
    gfx1151_entry_acquire_policy, gfx1151_resource_limits_policy, retained_kernarg_pool,
    Gfx12DispatchPacing, GpuDevice, KernargBuffer, KernargPool, Kernel, LaunchGeometry,
    LegacyDependencyMode, Pm4Architecture, Pm4Commands, Pm4RegisterPolicy, ReplayQuiescence,
    SingleQueuePm4Ib,
};
use crate::code_object::{CodeObjectId, RecordedArtifact};

/// A fresh command buffer from Redline's builder, paced per `lowering`.
pub(super) fn paced_commands(transport: &Transport<'_>, lowering: rp::ArchLowering) -> Pm4Commands {
    let mut commands = (transport.new_commands)();
    if let Pm4Commands::Gfx12(gfx12) = &mut commands {
        gfx12.set_dispatch_pacing(match lowering.post_dispatch_nop {
            0 => Gfx12DispatchPacing::None,
            n => Gfx12DispatchPacing::PostDispatchNop(n),
        });
    }
    commands
}

/// Emit one segment's commands: entry acquire, the planned boundaries,
/// `dispatch` for every node, and the trailing release. `dispatch` emits the
/// node's dispatch packet (and owns its kernarg buffer).
pub(super) fn emit_segment(
    plan: &rp::Pm4Plan,
    segment: &rp::Segment,
    arch_lowering: rp::ArchLowering,
    transport: &Transport<'_>,
    mut dispatch: impl FnMut(&mut Pm4Commands, usize) -> Result<(), String>,
) -> Result<Pm4Commands, String> {
    let mut commands = paced_commands(transport, arch_lowering);
    for op in plan.segment_ops(segment) {
        match op {
            Pm4Op::Entry => {
                commands.emit_entry_sentinel_reset()?;
                commands.acquire_entry(transport.gcr_trim, transport.entry_policy);
            }
            Pm4Op::Boundary(v) => {
                if v.wait_idle {
                    commands.wait_compute_idle()?;
                }
                match v.acquire {
                    Rung::None => {}
                    Rung::InterNode => commands.acquire_inter_node(transport.gcr_trim, false),
                    Rung::System => commands.gfx12_system_acquire()?,
                }
            }
            Pm4Op::Dispatch(i) => dispatch(&mut commands, i)?,
            Pm4Op::Exit => commands.trailing_release()?,
            // Recorded tapes and DFlash windows author no conditionals, and
            // gfx12 has no COND_EXEC receipt (`rp::cp_skip_admitted`).
            Pm4Op::SkipBegin(_) | Pm4Op::SkipEnd => {
                return Err("railgun M1 lowers no CP-skip conditionals".into())
            }
        }
    }
    Ok(commands)
}

/// One kernarg buffer whose bytes differ from the bound image.
#[derive(Clone, Debug)]
pub(crate) struct KernargMismatch {
    pub(crate) node: usize,
    pub(crate) symbol: String,
    /// VA of the buffer, its length and the pool allocation extent it owns.
    pub(crate) address: u64,
    pub(crate) length: usize,
    pub(crate) allocation: usize,
    /// First differing byte and the dwords around it.
    pub(crate) offset: usize,
    pub(crate) expected: u32,
    pub(crate) actual: u32,
    pub(crate) differing_bytes: usize,
    pub(crate) expected_head: Vec<u8>,
    pub(crate) actual_head: Vec<u8>,
}

/// A failed pre-submit self-check of the lowered state.
#[derive(Clone, Debug)]
pub(crate) struct SelfCheckFailure {
    /// The one-line description of the first mismatch.
    pub(crate) first: String,
    /// Every kernarg buffer that differs from its bound image.
    pub(crate) kernarg: Vec<KernargMismatch>,
}

/// A failed replay and what is known of the queue afterwards.
#[derive(Clone, Debug)]
pub(crate) struct OwnedReplayFailure {
    pub error: String,
    pub quiescence: ReplayQuiescence,
}

/// railgun's lowering of a directly authored program, with every object the
/// queue reads owned. Field order is drop order: the indirect buffer first,
/// then the kernels it dispatches, then the kernarg buffers it points at
/// (the order `PreparedPm4Replay` uses).
pub(crate) struct OwnedWindowLowering {
    graph: SingleQueuePm4Ib,
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    blocks: Vec<[u32; 3]>,
    publish: Option<usize>,
    command_dwords: u32,
}

/// Lower `program` (one segment spanning every node, every node PM4) through
/// railgun's own plan with railgun's kernarg images. `kernels[i]` is the
/// loaded object of node `i`. Kernarg buffers come from `transport.pool`
/// (Redline's retained pool); the indirect buffer and its host-read
/// completion surfaces come from `graph_pool` (Redline's host pool). Nothing
/// is submitted.
fn lower_owned(
    program: &rk::KernelProgram,
    plan: &rp::Pm4Plan,
    arch_lowering: rp::ArchLowering,
    kernels: Vec<Kernel>,
    transport: &Transport<'_>,
    graph_pool: &KernargPool,
    device: &GpuDevice,
) -> Result<OwnedWindowLowering, String> {
    let nodes = program.nodes.len();
    if !plan.is_single_segment(nodes) {
        return Err(format!(
            "an owned window lowers exactly one segment over {nodes} nodes ({} segments)",
            plan.segments.len()
        ));
    }
    if let Some((i, node)) = program
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| !n.lowering.is_pm4())
    {
        return Err(format!(
            "node {i} {} is not PM4-lowered: {:?}",
            node.symbol, node.lowering
        ));
    }
    if kernels.len() != nodes {
        return Err(format!(
            "{} loaded kernels for {nodes} nodes",
            kernels.len()
        ));
    }
    let words = program.authored_words();
    let mut buffers: Vec<Option<KernargBuffer>> = (0..nodes).map(|_| None).collect();
    let commands = emit_segment(
        plan,
        &plan.segments[0],
        arch_lowering,
        transport,
        |commands, i| {
            let node = &program.nodes[i];
            let metadata = kernels[i].metadata();
            let image = program.kernarg_image(i, metadata.kernarg_segment_size as usize, &words)?;
            let mut buffer = transport
                .pool
                .allocate_for(metadata)
                .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
            buffer
                .write_exact(&image)
                .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
            let mut workgroup = [0u16; 3];
            for axis in 0..3 {
                workgroup[axis] = u16::try_from(node.block[axis])
                    .map_err(|_| format!("{}: block exceeds u16", node.symbol))?;
            }
            let geometry = LaunchGeometry::from_hip_workgroups(node.grid, workgroup)
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            device
                .validate_geometry(geometry)
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            commands
                .dispatch(&kernels[i], geometry, node.dynamic_lds, buffer.address())
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            buffers[i] = Some(buffer);
            Ok(())
        },
    )?;
    if commands.gfx12_dwords().is_none() {
        return Err("an owned window lowers gfx12 command buffers only".to_owned());
    }
    let kernargs: Vec<KernargBuffer> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, b)| b.ok_or_else(|| format!("node {i} was never dispatched")))
        .collect::<Result<_, _>>()?;
    let command_dwords = commands.len_dwords();
    let graph = commands.create_graph(device, graph_pool, None, false)?;
    let publish = kernargs
        .iter()
        .rposition(|kernarg| kernarg.is_device_local() && !kernarg.is_empty());
    Ok(OwnedWindowLowering {
        graph,
        _kernels: kernels,
        kernargs,
        blocks: program.nodes.iter().map(|n| n.block).collect(),
        publish,
        command_dwords,
    })
}

/// Everything a gfx12 owned lowering needs that does not depend on one window,
/// built ONCE per owner (the window builder) and shared by every skeleton it
/// lowers: one HSA [`Runtime`] and [`GpuDevice`], the host pool (indirect
/// buffers, completion surfaces), the retained kernarg pool (Redline's, VRAM by
/// default) and every loaded code object. A skeleton's kernels, kernarg buffers
/// and indirect buffer each hold an `Arc` to what they use (kernarg buffer ->
/// pool -> runtime, kernel -> executable -> runtime), so they outlive this
/// context and are released only when the skeleton is dropped.
pub(crate) struct LoweringContext {
    // Field order is drop order: loaded objects before the pools before the
    // device and runtime they belong to.
    executables: BTreeMap<CodeObjectId, Executable>,
    resolved: BTreeMap<(CodeObjectId, String), Kernel>,
    kernarg_pool: KernargPool,
    host_pool: KernargPool,
    device: GpuDevice,
    architecture: Pm4Architecture,
    register_policy: Pm4RegisterPolicy,
    dispatch_initiator_policy: super::Gfx10DispatchInitiatorPolicy,
    dispatch_interleave: Option<super::Gfx11DispatchInterleave>,
    resource_limits_policy: super::Gfx11ComputeResourceLimitsPolicy,
    entry_policy: super::Gfx11EntryAcquirePolicy,
    gcr_trim: bool,
    _runtime: Runtime,
}

impl LoweringContext {
    /// The context of device ordinal `ordinal` with exactly the transport
    /// parameters Redline's gfx12 PM4 prepare uses
    /// (`ReplayController::prepare_pm4_prefix_inner`): the same dispatch-initiator,
    /// interleave, resource-limits and entry-acquire policies, the retained
    /// kernarg pool, the `HIPFIRE_REPLAY_PM4_GCR_TRIM` setting and
    /// `LegacyDependencyMode::CsPartialFlush`. gfx12 only.
    pub(crate) fn new(ordinal: usize) -> Result<Self, String> {
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(ordinal))
            .map_err(|error| error.to_string())?;
        let architecture = Pm4Architecture::from_device(&device)?;
        if architecture != Pm4Architecture::Gfx12 {
            return Err(format!(
                "an owned window lowers gfx12 command buffers only; device {} is not gfx12",
                device.name()
            ));
        }
        let dispatch_initiator_policy =
            gfx10_dispatch_initiator_policy(architecture, device.name());
        let dispatch_interleave = gfx1151_dispatch_interleave(architecture, device.name());
        let resource_limits_policy = gfx1151_resource_limits_policy(architecture, device.name());
        gfx1151_cu_count_knob_refusal()?;
        let entry_policy = gfx1151_entry_acquire_policy(architecture, device.name());
        let host_pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let kernarg_pool = retained_kernarg_pool(&device, &host_pool, entry_policy);
        let register_policy = Pm4RegisterPolicy::from_config();
        let gcr_trim = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GCR_TRIM")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        Ok(Self {
            executables: BTreeMap::new(),
            resolved: BTreeMap::new(),
            kernarg_pool,
            host_pool,
            device,
            architecture,
            register_policy,
            dispatch_initiator_policy,
            dispatch_interleave,
            resource_limits_policy,
            entry_policy,
            gcr_trim,
            _runtime: runtime,
        })
    }

    /// Distinct code objects loaded so far (each is loaded once per context).
    pub(crate) fn loaded_objects(&self) -> usize {
        self.executables.len()
    }

    /// [`lower_owned`] over this context. `artifacts[i]` is the loaded code
    /// object that owns node `i`'s symbol; each distinct object is loaded once
    /// for the whole context and node `i` resolves `{symbol}.kd` in it.
    /// Nothing is submitted.
    pub(crate) fn lower(
        &mut self,
        program: &rk::KernelProgram,
        plan: &rp::Pm4Plan,
        arch_lowering: rp::ArchLowering,
        artifacts: &[RecordedArtifact],
    ) -> Result<(OwnedWindowLowering, LowerTiming), String> {
        let started = std::time::Instant::now();
        let loaded_before = self.executables.len();
        if artifacts.len() != program.nodes.len() {
            return Err(format!(
                "{} code objects for {} nodes",
                artifacts.len(),
                program.nodes.len()
            ));
        }
        let mut kernels = Vec::with_capacity(program.nodes.len());
        for (node, artifact) in program.nodes.iter().zip(artifacts) {
            let id = artifact.id();
            if !self.executables.contains_key(&id) {
                let executable = Executable::load(&self.device, artifact.image().clone())
                    .map_err(|error| format!("load {artifact}: {error}"))?;
                self.executables.insert(id, executable);
            }
            let key = (id, format!("{}.kd", node.symbol));
            if !self.resolved.contains_key(&key) {
                let kernel = self.executables[&id]
                    .kernel(&key.1)
                    .map_err(|error| format!("resolve {}: {error}", key.1))?;
                self.resolved.insert(key.clone(), kernel);
            }
            kernels.push(self.resolved[&key].clone());
        }
        let load_us = started.elapsed().as_micros() as u64;
        let new_objects = self.executables.len() - loaded_before;
        let (architecture, register_policy) = (self.architecture, self.register_policy);
        let (initiator, interleave, limits) = (
            self.dispatch_initiator_policy,
            self.dispatch_interleave,
            self.resource_limits_policy,
        );
        let new_commands = || {
            Pm4Commands::new_with_dependency(
                architecture,
                register_policy,
                initiator,
                interleave,
                limits,
                LegacyDependencyMode::CsPartialFlush,
            )
        };
        let transport = Transport {
            pool: &self.kernarg_pool,
            new_commands: &new_commands,
            gcr_trim: self.gcr_trim,
            entry_policy: self.entry_policy,
        };
        let emit_started = std::time::Instant::now();
        let lowering = lower_owned(
            program,
            plan,
            arch_lowering,
            kernels,
            &transport,
            &self.host_pool,
            &self.device,
        )?;
        Ok((
            lowering,
            LowerTiming {
                load_us,
                new_objects,
                emit_us: emit_started.elapsed().as_micros() as u64,
            },
        ))
    }
}

/// Where one [`LoweringContext::lower`] spent its time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerTiming {
    /// Loading and resolving code objects (zero once they are all loaded).
    pub load_us: u64,
    /// Code objects loaded by this call (not already in the context).
    pub new_objects: usize,
    /// Kernarg allocation, IB emission and graph creation.
    pub emit_us: u64,
}

impl OwnedWindowLowering {
    /// Command dwords of the retained indirect buffer.
    pub(crate) fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    /// Replace node `node`'s whole kernarg segment (one
    /// `KernelProgram::encode_kernarg_into` image). Only valid while the
    /// queue is quiescent.
    pub(crate) fn write_kernarg(&mut self, node: usize, image: &[u8]) -> Result<(), String> {
        self.kernargs
            .get_mut(node)
            .ok_or_else(|| format!("no node {node}"))?
            .write_exact(image)
            .map_err(|e| format!("node {node}: kernarg: {e}"))
    }

    /// Patch node `node`'s dispatch geometry in the indirect buffer to the
    /// HIP grid `grid` (workgroups). gfx12 dispatch packets count work-items,
    /// so the grid is scaled by the node's block, checked. Only valid while
    /// the queue is quiescent.
    pub(crate) fn patch_grid(&mut self, node: usize, grid: [u32; 3]) -> Result<(), String> {
        let block = *self
            .blocks
            .get(node)
            .ok_or_else(|| format!("no node {node}"))?;
        let mut workitems = [0u32; 3];
        for axis in 0..3 {
            workitems[axis] = grid[axis].checked_mul(block[axis]).ok_or_else(|| {
                format!(
                    "node {node}: grid overflow axis={axis} workgroups={} workgroup={}",
                    grid[axis], block[axis]
                )
            })?;
        }
        self.graph
            .patch_dispatch_dimensions(node, workitems)
            .map_err(|e| e.to_string())
    }

    /// Read back the lowered state exactly as the command processor and the
    /// kernels will see it and compare it with what the window bound: every
    /// node's kernarg buffer (explicit and hidden bytes) against its bound
    /// image, and every `DISPATCH_DIRECT` in the retained indirect buffer
    /// against the bound grid (work-items) and the node's block. Quiescent
    /// use only. Every kernarg buffer is scanned (after a publish, so a
    /// posted-but-unlanded write cannot hide); the failure names the first
    /// mismatch and carries every mismatching buffer with its VA range.
    pub(crate) fn verify_state(
        &mut self,
        symbols: &[String],
        images: &[&[u8]],
        grids: &[[u32; 3]],
    ) -> Result<(), SelfCheckFailure> {
        let nodes = self.kernargs.len();
        let plain = |first: String| SelfCheckFailure {
            first,
            kernarg: Vec::new(),
        };
        if images.len() != nodes || grids.len() != nodes || symbols.len() != nodes {
            return Err(plain(format!(
                "{} images, {} grids, {} symbols for {nodes} lowered nodes",
                images.len(),
                grids.len(),
                symbols.len()
            )));
        }
        self.publish_host_writes();
        let mut bad: Vec<KernargMismatch> = Vec::new();
        let mut shape_error: Option<String> = None;
        for (node, (buffer, image)) in self.kernargs.iter_mut().zip(images).enumerate() {
            let address = buffer.address() as usize as u64;
            let allocation = buffer.allocation_size();
            let actual: &[u8] = buffer.as_mut_bytes();
            if actual.len() != image.len() {
                shape_error.get_or_insert_with(|| {
                    format!(
                        "node {node} {}: kernarg buffer is {} bytes, the bound image {}",
                        symbols[node],
                        actual.len(),
                        image.len()
                    )
                });
                continue;
            }
            let Some(at) = actual.iter().zip(*image).position(|(a, b)| a != b) else {
                continue;
            };
            let word = at & !3;
            let get = |bytes: &[u8]| {
                let mut w = [0u8; 4];
                let end = (word + 4).min(bytes.len());
                w[..end - word].copy_from_slice(&bytes[word..end]);
                u32::from_le_bytes(w)
            };
            let head = image.len().min(16);
            bad.push(KernargMismatch {
                node,
                symbol: symbols[node].clone(),
                address,
                length: actual.len(),
                allocation,
                offset: at,
                expected: get(image),
                actual: get(actual),
                differing_bytes: actual.iter().zip(*image).filter(|(a, b)| a != b).count(),
                expected_head: image[..head].to_vec(),
                actual_head: actual[..head].to_vec(),
            });
        }
        if let Some(first) = shape_error {
            return Err(SelfCheckFailure {
                first,
                kernarg: bad,
            });
        }
        if let Some(m) = bad.first() {
            return Err(SelfCheckFailure {
                first: format!(
                    "node {} {}: kernarg byte +{} (dword +{}) expected {:#010x} actual {:#010x} \
                     ({} of {} kernarg buffers differ)",
                    m.node,
                    m.symbol,
                    m.offset,
                    m.offset & !3,
                    m.expected,
                    m.actual,
                    bad.len(),
                    nodes
                ),
                kernarg: bad,
            });
        }
        let decoded = self
            .graph
            .decode_dispatches()
            .map_err(|e| plain(e.to_string()))?;
        if decoded.len() != nodes {
            return Err(plain(format!(
                "the retained IB holds {} dispatches for {nodes} lowered nodes",
                decoded.len()
            )));
        }
        for (node, d) in decoded.iter().enumerate() {
            let block = self.blocks[node];
            let mut want = [0u32; 3];
            for axis in 0..3 {
                want[axis] = grids[node][axis].checked_mul(block[axis]).ok_or_else(|| {
                    plain(format!(
                        "node {node} {}: bound grid {:?} x block {block:?} overflows",
                        symbols[node], grids[node]
                    ))
                })?;
            }
            if d.dimensions != want {
                return Err(plain(format!(
                    "node {node} {}: IB dispatch dimensions expected {want:?} (grid {:?} x block {block:?}) actual {:?}",
                    symbols[node], grids[node], d.dimensions
                )));
            }
            if d.workgroup != Some(block) {
                return Err(plain(format!(
                    "node {node} {}: IB workgroup size expected {block:?} actual {:?}",
                    symbols[node], d.workgroup
                )));
            }
        }
        Ok(())
    }

    /// `(address, length, allocation size)` of every node's kernarg buffer.
    pub(crate) fn kernarg_ranges(&self) -> Vec<(u64, usize, usize)> {
        self.kernargs
            .iter()
            .map(|k| (k.address() as usize as u64, k.len(), k.allocation_size()))
            .collect()
    }

    /// Make every host kernarg write since the last call GPU-visible.
    pub(crate) fn publish_host_writes(&self) {
        if let Some(index) = self.publish {
            self.kernargs[index].publish_host_writes();
        }
    }

    /// Submit the retained indirect buffer once and wait for it.
    ///
    /// # Safety
    ///
    /// Every pointer in the written kernarg images must still refer to a live
    /// allocation of the owner (the window binding key proves it).
    pub(crate) unsafe fn replay_and_wait(&mut self) -> Result<(), OwnedReplayFailure> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.replay_and_wait_checked() }.map_err(|(error, quiescence)| {
            OwnedReplayFailure {
                error: error.to_string(),
                quiescence: match quiescence {
                    redline_dispatch::aql::Quiescence::Proven => ReplayQuiescence::Proven,
                    redline_dispatch::aql::Quiescence::Unknown => ReplayQuiescence::Unknown,
                },
            }
        })
    }
}
