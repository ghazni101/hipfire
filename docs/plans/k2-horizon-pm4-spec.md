# K2-Horizon PM4/Redline Retained-Replay Spec

**Status:** Draft. Branch `feat/k2-horizon-arch-spec`.

## Goal

Lower K2-Horizon's decode path onto the retained PM4 replay transport,
eliminating per-launch HIP dispatch overhead (~2.1 µs/launch × ~300
launches/token ≈ 630 µs/token) and enabling the redline fast path that
qwen35, deepseek4, and lfm2moe already use.

## Problem

The daemon already arms redline for K2-Horizon: `retained_redline_default`
returns `true` for `.mq4r` on `gfx1100` with `pp=1, tp=1`, and
`configure_model_default` sets `ReplayState::Armed` with PM4 transport.
The startup log confirms:

```
[redline] enabling fail-closed retained default on gfx1100 (model_arch=k2_horizon, drafter=off, transport=pm4)
```

But the forward path in `hipfire-arch-k2-horizon/src/forward.rs` never
calls into the replay controller — no `begin_auto_capture_if_armed`,
no `should_route_pm4`, no `replay_pm4`, no `finish_capture`. Every
decode step issues fresh HIP kernel launches through ordinary dispatch.

Additionally, the forward path has **CPU sync points** inside the per-layer
loop that are incompatible with PM4 capture/replay:

1. **MoVA routing** (45 layers): `download_f32(v_router_logits)` +
   `download_f32(v_router_bias)` → CPU sort + topk → results used to
   index `v_experts` on CPU. Two D2H syncs per layer.
2. **MoE routing** (45 layers): `download_f32(moe_router_logits)` +
   `download_f32(router_bias)` → CPU sort + topk → `memcpy_htod` to
   upload indices + weights. Two D2H + two H2D per layer.

That's **~180 D2H/H2D syncs per token** in the routing paths alone —
each forces a full device pipeline stall. PM4 replay cannot work with
these because (a) the captured kernel sequence depends on CPU-decided
expert indices that vary per token, and (b) `hipMalloc`/sync calls
inside capture are rejected.

## Design

### Phase 1: GPU-side MoVA routing (eliminate CPU topk)

Replace the CPU download + sort + topk in `forward_mova_value_routing`
with the existing GPU kernel `deepseek4_moe_topk_bias_aware_f32`.

**Current flow** (per MoVA layer):
1. `weight_gemv(v_router, normed) → v_router_logits [64]`
2. `sigmoid_f32(v_router_logits)` in-place
3. `download_f32(v_router_logits)` → CPU `[64]`
4. `download_f32(v_router_bias)` → CPU `[64]`
5. CPU: `selection = scores + bias`, topk(4), normalize, scale
6. CPU loop: per-expert `weight_gemv` + `silu_f32` + `scale_f32` + `add_inplace_f32`

**New flow**:
1. `weight_gemv(v_router, normed) → v_router_logits [64]`
2. `sigmoid_f32(v_router_logits)` in-place
3. `deepseek4_moe_topk_bias_aware_f32(v_router_logits, v_router_bias, v_topk_indices, v_topk_weights, n_exp=64, k_top=4, route_scale=cfg.router_scaling_factor)` — single GPU launch, no D2H
4. Per-expert loop stays (weight_gemv + silu + scale + add), but indices/weights come from GPU buffers

The `deepseek4_moe_topk_bias_aware_f32` kernel does exactly the
bias-for-selection semantic: `biased[i] = scores[i] + bias[i]`, pick
top-K by biased, then `weights = scores[selected] / sum * route_scale`.
This matches K2-Horizon's `calc_router_weights` exactly.

**Changes:**
- `forward_mova_value_routing`: remove `download_f32` calls, CPU sort,
  CPU topk. Replace with `deepseek4_moe_topk_bias_aware_f32`. Read
  indices from `v_topk_indices` (already on GPU) for the per-expert
  loop — but the per-expert loop uses `weight_gemv` which takes a
  `&WeightTensor`, so we still need to index `attn.v_experts` on host.
  Download only the 4 indices (16 bytes) to select experts.

  **Wait** — this is the key constraint. The per-expert `weight_gemv`
  calls use host-side `&attn.v_experts[idx]` references. The expert
  selection must be known on the CPU to pick the right weight tensor.
  Downloading 4 i32 indices (16 bytes) is a tiny sync, but it still
  breaks PM4 capture.

  **Solution**: Use the indexed MoE GEMV kernel
  `gemv_hfq4g256_moe_gate_up_k8_indexed_batched` for MoVA too. This
  kernel takes `expert_ptrs` (a GPU tensor of device pointers) +
  `topk_indices` (GPU) and dispatches all experts in one launch. But
  MoVA v_experts produce `[kv_dim]` output, not `[2*moe_inter]` — the
  gate_up kernel writes split gate/up outputs. We need a different
  indexed kernel for plain (non-gate-up) GEMV.

  **Alternative**: Keep the per-expert `weight_gemv` loop but download
  the 4 indices (16 bytes) after the topk kernel. This is a single
  16-byte D2H — much cheaper than the current 2× 256-byte downloads +
  CPU sort. But it still creates a sync point inside the layer loop.

  **For PM4**: The sync must be eliminated entirely. The solution is to
  pre-compute a `v_expert_ptrs` tensor (like `expert_gate_up_ptrs`) and
  use an indexed GEMV kernel that handles the MoVA case (single output,
  not gate_up split). The existing `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
  kernel does single-output indexed GEMV — it writes `[k_top × m]` to
  `expert_outputs`. We can use it for MoVA v_experts with `m=kv_dim,
  k=hidden`.

  **Full GPU MoVA routing**:
  1. `weight_gemv(v_router, normed) → v_router_logits [64]`
  2. `sigmoid_f32(v_router_logits)` in-place
  3. `deepseek4_moe_topk_bias_aware_f32(...)` → `v_topk_indices [4]`, `v_topk_weights [4]` on GPU
  4. `rotate_x_mq_for(v_experts[0], normed, v_x_rot, hidden)` — FWHT rotate
  5. `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(v_expert_ptrs, v_topk_indices, v_x_rot, v_expanded, m=kv_dim, k=hidden, k_top=4, batch=1)` → `v_expanded [4 × kv_dim]`
  6. `silu_f32(v_expanded)` in-place (all 4 experts at once)
  7. Scale + accumulate: for each of 4 experts, `fa_v += v_topk_weights[k] * v_expanded[k]`
     - This can be a single `scale_accum_f32` kernel or 4 `scale_f32` + `add_inplace_f32` calls

  This eliminates all D2H/H2D in MoVA routing. The per-expert loop
  becomes 4 GPU kernel launches (or 1 batched + 1 silu + 4 scale+add).

### Phase 2: GPU-side MoE routing (eliminate CPU topk)

Same pattern for `forward_sigmoid_moe_ffn`. Replace CPU download + sort
+ topk + upload with `deepseek4_moe_topk_bias_aware_f32`.

**Current flow** (per MoE layer):
1. `weight_gemv(router, normed) → moe_router_logits [100]`
2. `sigmoid_f32(moe_router_logits)` in-place
3. `download_f32(moe_router_logits)` → CPU `[100]`
4. `download_f32(router_bias)` → CPU `[100]`
5. CPU: `selection = scores + bias`, topk(8), normalize, scale
6. `memcpy_htod(moe_topk_indices)`, `memcpy_htod(moe_topk_weights)`
7. `rotate_x_mq_for` + `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
8. Per-expert: `silu_mul` + `weight_gemv(down)` + `scale` + `add`
9. Shared expert: 3× `weight_gemv` + `silu_mul`

**New flow**:
1. `weight_gemv(router, normed) → moe_router_logits [100]`
2. `sigmoid_f32(moe_router_logits)` in-place
3. `deepseek4_moe_topk_bias_aware_f32(moe_router_logits, router_bias, moe_topk_indices, moe_topk_weights, n_exp=100, k_top=8, route_scale=cfg.router_scaling_factor)` — single GPU launch
4. `rotate_x_mq_for` + `gemv_hfq4g256_moe_gate_up_k8_indexed_batched` (unchanged — already uses GPU indices)
5. Replace per-expert `weight_gemv(down)` loop with
   `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(expert_down_ptrs, moe_topk_indices, rot_batch, down_expanded, m=hidden, k=moe_inter, k_top=8, batch=1)`
6. Scale + accumulate into `h` using `moe_topk_weights`

**Changes:**
- `forward_sigmoid_moe_ffn`: remove `download_f32` + CPU sort + `memcpy_htod`.
  Replace with `deepseek4_moe_topk_bias_aware_f32`.
- Replace per-expert down `weight_gemv` loop with indexed
  `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`.
- Need `expert_down_ptrs` tensor (already exists in `MoeFfnWeights`).
- Need `rot_batch` (silu_mul output, already allocated).
- The silu_mul must be done on the full `gate_batch`/`up_batch` before
  the down GEMV. Use `fused_silu_mul_rotate_mq_batched_for` to combine
  silu_mul + FWHT rotation in one kernel, then the down indexed kernel
  reads from `rot_batch`.

### Phase 3: PM4 capture/replay in forward

Once all CPU sync points are eliminated from the per-layer loop, the
forward body becomes a pure sequence of GPU kernel launches with no
host dependencies. This is PM4-capturable.

**Pattern** (matching qwen35/deepseek4/lfm2moe):

```
forward_only(token_id, position):
    1. Set forward_eligible = true
    2. begin_auto_capture_if_armed()
    3. if should_route_pm4():
         memcpy_htod(pos_buf, position)
         replay_pm4(position)
         return
    4. memcpy_htod(pos_buf, position)
    5. embedding_lookup_q8(token_id)  ← per-token, outside capture
    6. [layer loop — all GPU, no syncs]
    7. final norm + lm_head
    8. if should_auto_finalize_capture():
         device_synchronize()
         finish_capture()
         prepare_pm4_prefix(launches)
```

**Key invariants:**
- Steps 4-5 (pos upload + embedding) are per-token inputs. They must
  run outside the captured body OR be part of the captured body with
  the pos_buf/token_id staged via a broadcast buffer.
- The lfm2moe pattern: `prepare_retained_decode_inputs` (pos + embedding)
  runs BEFORE the capture/replay check. The captured body is just the
  layer loop + final norm + lm_head.
- K2-Horizon should follow the lfm2moe pattern: extract
  `prepare_decode_inputs` (pos + embedding) and `run_decode_body`
  (layers + norm + lm_head) as separate functions.

**Restructured forward:**
```rust
fn prepare_decode_inputs(cfg, weights, state, gpu, token_id, position) {
    memcpy_htod(pos_buf, position)
    embedding_lookup_q8(token_embd, h, token_id, dim)
}

fn run_decode_body(cfg, weights, state, gpu, position) {
    for dense_layer in dense_layers { forward_dense_layer(...) }
    for moe_layer in moe_layers { forward_moe_layer(...) }
    grouped_rmsnorm_f32(final_norm)
    weight_gemv(lm_head, final_norm_buf, logits)
}

pub fn decode_step_sampled(cfg, weights, state, gpu, token_id, position, temp, top_p, rng) {
    // PM4 replay check — if ready, replay the captured body
    gpu.replay.set_forward_eligible(true)
    if gpu.replay.should_route_pm4() {
        prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)
        gpu.hip.device_synchronize()  // hand off to ROCr queue
        unsafe { gpu.replay.replay_pm4(position as usize) }?
        // sample from state.logits on GPU
        return sample(temp, top_p, rng)
    }
    // Not ready — capture or direct dispatch
    if !should_route_pm4() {
        gpu.replay.begin_auto_capture_if_armed()
    }
    prepare_decode_inputs(cfg, weights, state, gpu, token_id, position)
    run_decode_body(cfg, weights, state, gpu, position)
    if gpu.replay.should_auto_finalize_capture() {
        gpu.hip.device_synchronize()
        gpu.replay.finish_capture()
        gpu.replay.prepare_pm4_prefix(device_id, launches)
    }
    return sample(temp, top_p, rng)
}
```

**Warmup**: The first decode token runs through ordinary HIP dispatch
while `launch_maybe_blob` records the kernel sequence. After
`finish_capture` + `prepare_pm4_prefix`, subsequent tokens replay the
captured PM4 command buffer — no HIP dispatch overhead.

### Phase 4: Wire into generate_k2_horizon

No changes needed — `generate_k2_horizon` already calls
`decode_step_sampled` per token. The PM4 lifecycle is entirely inside
`decode_step_sampled`.

### Edge cases

- **KV cache growth**: If `max_seq` is bumped (layout growth), the
  captured PM4 buffer is invalid. Call `rearm_after_layout_growth()`
  to reset to `Armed` state. The next decode re-captures.
- **Prefill**: Prefill uses `decode_step` (not `decode_step_sampled`).
  PM4 capture must not trigger during prefill. Set
  `forward_eligible = false` in the prefill path.
- **MoE expert pointers**: The `v_expert_ptrs` and `expert_down_ptrs`
  tensors must be pre-built during weight loading (they already exist
  for `expert_gate_up_ptrs` and `expert_down_ptrs` in `MoeFfnWeights`).
  Need to add `v_expert_ptrs` to `MovaAttnWeights`.

## Implementation order

1. **GPU MoVA routing** — add `v_expert_ptrs` to weights, use
   `deepseek4_moe_topk_bias_aware_f32` + indexed GEMV
2. **GPU MoE routing** — use `deepseek4_moe_topk_bias_aware_f32` +
   indexed down GEMV (replace per-expert weight_gemv loop)
3. **Restructure forward** — extract `prepare_decode_inputs` +
   `run_decode_body`, add PM4 capture/replay lifecycle
4. **Build, deploy, verify** — check `[redline] retained route ready`
   log, benchmark PM4 vs non-PM4

## Expected impact

| Metric | Current (HIP dispatch) | Target (PM4 replay) |
|---|---|---|
| Launches/token | ~300 | 1 (replay) + 2 (pos+embed) |
| Dispatch overhead | ~630 µs/token | ~3 µs/token |
| D2H syncs/token | ~180 (routing) + 1 (logits) | 1 (sample result) |
| Expected tok/s | ~44 (greedy) | ~60-70 (greedy) |

The routing sync elimination (Phases 1-2) is the larger win: 180
D2H/H2D syncs × ~5-7 µs each = ~900-1260 µs/token saved. PM4 replay
(Phase 3) adds another ~630 µs/token. Combined: ~1.5-1.9 ms/token
saved, which at 22.4 ms/token (44 tok/s) is a ~7-8% to ~60-70% speedup
depending on how much of the 22.4 ms is sync overhead vs compute.
