# K2-Horizon (MoVA-36B-A4B) — Architecture Support Spec

**Status:** Implemented on `feat/k2-horizon-arch-spec`. Phases 0–6 are
done (arch crate, grouped RMSNorm, MoVA + sigmoid-MoE decode forward,
quantizer arm, carrier, AR generate path, GPU sampling, PM4
retained-replay). All §9 gaps are either fixed or confirmed out of scope
(KLD eval, multi-turn, spec-decode/MTP). Serving verified in-container on
gfx1100: q8 KV @ 32k ctx, PM4 retained replay, ~121 tok/s decode.

## Goal

End-to-end support for the K2-Horizon model family in hipfire, starting
with `K2-Horizon-MoVA-36B-A4B` (36B params, 4B active, MoVA attention +
sigmoid-routed MoE FFN). The deliverable is a new arch crate
(`hipfire-arch-k2-horizon`, `arch_id = 16`), a quantizer pipeline arm,
and a forward path — enabling an MQ4R quant via the existing
`--format mq4 --no-q8-router` recipe.

## 1. Architecture (verified from source)

All dimensions from `config.json` and `modeling_k2_horizon.py` in
`~/models/RAW/IFM/K2-Horizon-MoVA-36B-A4B/source/`.

### 1.1. Config

| field | value | notes |
|---|---|---|
| `model_type` | `"k2_horizon"` | not in hipfire's arch_mapping table |
| `architectures` | `["K2HorizonForCausalLM"]` | |
| `hidden_size` | 2560 | |
| `num_hidden_layers` | 48 | |
| `num_attention_heads` | 32 | |
| `num_key_value_heads` | 8 | GQA, 4:1 ratio |
| `head_dim` | 128 | |
| `rope_head_dim` | 128 | full rotary (== head_dim, no partial) |
| `rope_theta` | 10,000,000 | `rope_type: "default"` |
| `vocab_size` | 250,624 | |
| `max_position_embeddings` | 524,288 | 512K context |
| `rms_norm_eps` | 1e-6 | |
| `intermediate_size` | 6144 | dense FFN (layers 0–2) |
| `moe_intermediate_size` | 768 | per-expert FFN |
| `num_experts` | 100 | routed FFN experts |
| `num_experts_per_tok` | 8 | top-8 FFN routing |
| `num_shared_experts` | 1 | always-on shared FFN expert |
| `norm_topk_prob` | true | re-normalize FFN routing weights |
| `router_score_func` | `"sigmoid"` | **not softmax** — novel vs all hipfire arches |
| `router_scaling_factor` | 2.5 | post-normalization scaling |
| `moe_gate_bias` | true | router has bias; bias added to *selection* scores only |
| `mova_num_experts` | 64 | value experts in MoVA attention |
| `mova_num_experts_per_tok` | 4 | top-4 value routing |
| `attention_gate_func` | `"softplus"` | post-attention gate: `attn_out *= softplus(gate_proj(x))` |
| `attention_bias` | false | no bias on q/k/o projections |
| `query_key_norm` | false | no q/k RMSNorm |
| `layernorm_num_groups` | 2 | **grouped RMSNorm** — variance per hidden_size/2 chunk |
| `mlp_only_layers` | `[0, 1, 2]` | layers 0–2 are dense (standard attention + dense MLP) |
| `decoder_sparse_step` | 1 | every layer after the dense prefix is MoE |
| `tie_word_embeddings` | false | separate `lm_head.weight` |
| `dtype` | bfloat16 | source precision |

### 1.2. Layer topology

```
Layer 0–2:  K2HorizonDecoderLayer (dense)
  ├── K2HorizonAttention      (MHA: q/k/v/o + gate_proj — the softplus
  │                            post-attention gate applies to DENSE layers
  │                            too; gate_func is set for both attention
  │                            classes in modeling_k2_horizon.py)
  └── K2HorizonMLP            (SwiGLU: gate_proj, up_proj, down_proj)

Layer 3–47: K2HorizonDecoderLayer (sparse)
  ├── K2HorizonMoVAAttention  (MoVA: q_proj, k_proj, o_proj, v_router, v_experts[64], gate_proj)
  └── K2HorizonSparseMoeBlock (sigmoid MoE: gate[bias], experts[100], shared_experts[1])
```

### 1.3. MoVA attention (layers 3–47)

The value projection is replaced by a **mixture of 64 value experts**,
each a bias-free linear `[hidden_size, num_kv_heads * head_dim]`. A
`v_router` linear `[hidden_size, 64]` (with bias) selects top-4 per
token.

Forward (from `modeling_k2_horizon.py:351-512`):

```
# Value-expert routing (identical semantics to MoE FFN routing)
router_logits = F.linear(x, v_router.weight)           # no bias in logits
routing_weights, selected = calc_router_weights(
    router_logits, v_router.bias,
    score_func="sigmoid", top_k=4, scaling_factor=2.5,
)
# calc_router_weights:
#   routing_scores = sigmoid(router_logits)
#   selection_scores = routing_scores + v_router.bias   # bias only for selection
#   selected = topk(selection_scores, 4)
#   routing_weights = gather(routing_scores, selected)
#   routing_weights /= sum                              # normalize (top_k > 1)
#   routing_weights *= 2.5                              # scaling factor

# Value computation
mixed_value = combine_routed_experts(                   # SiLU-activated
    x, routing_weights, selected, v_experts, activation=F.silu)
#   for each hit expert:
#     expert_states = v_experts[idx](x[tokens])
#     expert_states = silu(expert_states)
#     expert_states *= routing_weights[tokens, k]
#     scatter-add into mixed_value

value_states = mixed_value.view(...).transpose(1, 2)

# Standard Q/K projections (no q_norm/k_norm — query_key_norm=false)
query_states = q_proj(x)  → RoPE
key_states   = k_proj(x)  → RoPE

# Standard attention
attn_output = attention(query, key, value_states)

# Post-attention gate (softplus)
gate = softplus(gate_proj(x), beta=log(2))              # softplus(x) = log(1 + e^x)
attn_output = attn_output * gate

# Output projection
attn_output = o_proj(attn_output)
```

Key differences from qwen35 `FullAttnLayerWeights`:
- **No `wv`** — replaced by `v_router` + `v_experts[64]`
- **`gate_proj`** — post-attention elementwise gate (softplus)
- **No `q_norm` / `k_norm`** — `query_key_norm=false`
- **No attention bias** — `attention_bias=false`

### 1.4. Sigmoid-routed MoE FFN (layers 3–47)

From `modeling_k2_horizon.py:531-609`:

```
router_logits = F.linear(x, gate.weight)                # gate has bias
# sigmoid routing (NOT softmax):
routing_weights = sigmoid(router_logits)
selection_scores = routing_weights + gate.bias          # bias for selection only
_, selected = topk(selection_scores, top_k=8)
routing_weights = gather(routing_weights, selected)
if norm_topk_prob: routing_weights /= sum               # normalize
routing_weights *= 2.5                                  # scaling factor

# Per-expert FFN (SiLU SwiGLU, same as qwen35)
for expert in hit_experts:
    y = expert(x[tokens])                                # gate_proj * silu * up_proj → down_proj
    y *= routing_weights[tokens, k]
    scatter-add

# Shared expert (always-on, no routing)
y += shared_experts(x)                                   # shared_expert_intermediate = 768 * 1
```

Differences from qwen35 `K2HorizonSparseMoeBlock`:
- `router_score_func = "sigmoid"` (qwen35 uses softmax)
- `router_scaling_factor = 2.5` (qwen35 has no scaling)
- `moe_gate_bias = true` (qwen35 gate has no bias)
- `num_shared_experts = 1` with `shared_expert_intermediate = 768` (qwen35
  uses `shared_expert_intermediate_size` from config; K2-Horizon derives
  it as `moe_intermediate_size * num_shared_experts`)

### 1.5. Grouped RMSNorm

`K2HorizonRMSNorm` (`modeling_k2_horizon.py:613-639`):

```python
# n_groups = layernorm_num_groups = 2
# variance computed per group, not over the full hidden_size
x_grouped = x.reshape(..., n_groups, hidden_size // n_groups)
var = x_grouped.pow(2).mean(-1, keepdim=True)
x_normed = x_grouped * rsqrt(var + eps)
x_normed = x_normed.reshape(..., hidden_size)
return x_normed * weight                                 # weight is [hidden_size]
```

Standard RMSNorm computes variance over the entire `hidden_size` vector.
Grouped RMSNorm splits into 2 groups of 1280 each and normalizes
independently. This is **not** supported by any existing hipfire kernel.

### 1.6. Tokenizer

| token | id | role |
|---|---|---|
| `<\|ifm\|begin_of_text\|>` | 0 | BOS |
| `<\|ifm\|endoftext\|>` | 1 | EOS (primary) |
| 250019 | 250019 | EOS (secondary, from `generation_config.json`) |

`tokenizer_class: "PreTrainedTokenizerFast"` — standard fast tokenizer,
no custom code. Chat template is a 49.8 KB Jinja file
(`chat_template.jinja`) with reasoning-effort support
(`reasoning_effort="high"` recommended).

### 1.7. Safetensors layout

Single `model.safetensors` (69.7 GB, bf16). Tensor naming follows the
Qwen3-MoE convention with MoVA additions:

```
model.embed_tokens.weight
model.layers.{N}.input_layernorm.weight
model.layers.{N}.post_attention_layernorm.weight
model.layers.{N}.self_attn.q_proj.weight
model.layers.{N}.self_attn.k_proj.weight
model.layers.{N}.self_attn.o_proj.weight
# Dense layers (0–2) only — standard MHA v_proj + the same softplus gate:
model.layers.{N}.self_attn.v_proj.weight
model.layers.{N}.self_attn.gate_proj.weight
# MoVA layers (3–47) only:
model.layers.{N}.self_attn.v_router.weight
model.layers.{N}.self_attn.v_router.bias
model.layers.{N}.self_attn.v_experts.{E}.weight       # E = 0..63
model.layers.{N}.self_attn.gate_proj.weight
# Dense FFN (layers 0–2):
model.layers.{N}.mlp.gate_proj.weight
model.layers.{N}.mlp.up_proj.weight
model.layers.{N}.mlp.down_proj.weight
# MoE FFN (layers 3–47):
model.layers.{N}.mlp.gate.weight                       # router [100, 2560]
model.layers.{N}.mlp.gate.bias
model.layers.{N}.mlp.experts.{E}.gate_proj.weight      # E = 0..99
model.layers.{N}.mlp.experts.{E}.up_proj.weight
model.layers.{N}.mlp.experts.{E}.down_proj.weight
model.layers.{N}.mlp.shared_experts.gate_proj.weight
model.layers.{N}.mlp.shared_experts.up_proj.weight
model.layers.{N}.mlp.shared_experts.down_proj.weight
model.norm.weight
lm_head.weight
```

**Per-expert tensors ship as separate 2D tensors** (not 3D stacked like
qwen3.5-MoE). This matches the DeepSeek V4 / MiniMax pattern, not the
qwen3.5 `mlp.experts.gate_up_proj` 3D pattern.

## 2. What MQ4R is

MQ4R is not a separate quant format. It is a recipe on top of MQ4G256:

```bash
hipfire quantize --format mq4 --no-q8-router --input <safetensors-dir> --output <out.hfq>
```

`--no-q8-router` drops the forced-Q8 "fixed tier" (attention, lm_head,
router, embed, conv1d) so those tensors follow `--format` (MQ4) instead
of being pinned to Q8F16. Embed_tokens stays Q8 via its own code path.
Result: ~4.25 bpw across the whole model. The "R" = reduced fixed tier.

For K2-Horizon, the MQ4R recipe applies identically once the arch is
registered — the `--no-q8-router` flag is arch-agnostic. The fixed tier
for K2-Horizon includes: q_proj, k_proj, o_proj, v_router, gate_proj
(attention), mlp.gate (FFN router), shared_experts, lm_head, embed_tokens.
All of these would drop from Q8 to MQ4 under `--no-q8-router`.

**Deviation (as implemented):** `v_experts` are excluded from the fixed
tier unconditionally — `q8_class_of` returns `None` for any name
containing `v_experts`, so they always follow `--format` (MQ4) regardless
of `--no-q8-router`. Rationale: 64 experts × [1024, 2560] at Q8 ≈ 420 MB
of VRAM for a routing-quality benefit not yet measured. If a future
variant wants Q8 value experts, `q8_class_of` needs a `Some("attn")` arm
for `v_experts` gated on an explicit flag.

## 3. Implementation plan

### Phase 0: Registration (fail-closed plumbing)

**Goal:** `hipfire quantize` recognizes `k2_horizon` and fails with a
clean "no carrier" error instead of "unknown model_type".

1. **`arch_mapping.rs`** — add `("k2_horizon", 16)` to
   `MODEL_TYPE_TO_ARCH_ID`. arch_id 16 is the next free ID (15 = maple, 14 =
   muse_glimmer, 22 = gemma4 drafter; 15–21 are free).

2. **`safetensors_source.rs`** — `derive_arch_id` already calls
   `lookup_model_type`; no change needed beyond the table entry. The
   `has_experts` flip (5→6 for qwen3.5) does not apply — K2-Horizon has
   its own ID.

3. **`carriers.rs`** — add a `K2HorizonCarrier` stub that claims
   `arch_id == 16` and returns a clean "not yet implemented" error from
   `load()`. Register it in `REGISTRY`.

4. **`reset_core.rs`** — add arch_key mapping for id 16 to the
   `arch_key_for_id` function and `ResetCoreCoverage` inventory.

**Acceptance:** `hipfire quantize --input <dir> --format mq4` fails with
"unknown carrier for arch_id 16" instead of "unknown model_type
'k2_horizon'". `cargo test carriers_are_disjoint` passes.

### Phase 1: Config parser + weight structs

**Goal:** Parse K2-Horizon config from safetensors metadata; define
weight structs matching the safetensors layout.

1. **`hipfire-arch-k2-horizon` crate** — new crate, `Cargo.toml`
   mirroring `hipfire-arch-qwen35` (deps: hipfire-runtime,
   hipfire-dispatch, hip-bridge, rdna-compute, serde, serde_json).

2. **`config.rs`** — `K2HorizonConfig` struct with all fields from §1.1.
   Parse from `config.json` via serde. Key fields beyond qwen35:
   - `mlp_only_layers: Vec<usize>` — dense-vs-MoE layer split
   - `mova_num_experts`, `mova_num_experts_per_tok`
   - `attention_gate_func: String` ("softplus")
   - `router_score_func: String` ("sigmoid")
   - `router_scaling_factor: f32` (2.5)
   - `moe_gate_bias: bool`
   - `layernorm_num_groups: usize` (2)
   - `rope_head_dim: usize` (128)

3. **`weights.rs`** — weight structs:

```rust
// Dense attention (layers 0–2)
pub struct DenseAttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,        // standard v_proj
    pub wo: WeightTensor,
}

// MoVA attention (layers 3–47)
pub struct MovaAttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wo: WeightTensor,
    pub v_router: WeightTensor,  // [64, hidden_size] + bias
    pub v_router_bias: GpuTensor, // [64] — stored as F32
    pub v_experts: Vec<WeightTensor>, // 64 × [kv_heads*head_dim, hidden_size]
    pub gate_proj: WeightTensor, // [num_heads*head_dim, hidden_size]
}

// Dense FFN (layers 0–2) — same as qwen35 K2HorizonMLP
pub struct DenseFfnWeights {
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

// MoE FFN (layers 3–47) — per-expert 2D, like DeepSeek V4
pub struct MoeFfnWeights {
    pub router: WeightTensor,         // [100, hidden_size]
    pub router_bias: GpuTensor,       // [100] — F32
    pub experts: Vec<ExpertWeights>,  // 100 × {gate_proj, up_proj, down_proj}
    pub shared_expert: SharedExpertWeights, // gate_proj + up_proj + down_proj
}

pub struct ExpertWeights {
    pub gate_proj: WeightTensor,  // [moe_intermediate, hidden]
    pub up_proj: WeightTensor,    // [moe_intermediate, hidden]
    pub down_proj: WeightTensor,  // [hidden, moe_intermediate]
}

pub struct SharedExpertWeights {
    pub gate_proj: WeightTensor,  // [moe_intermediate * num_shared, hidden]
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub enum LayerWeights {
    Dense(DenseLayerWeights),
    Moe(MoeLayerWeights),
}

pub struct K2HorizonWeights {
    pub embed_tokens: GpuTensor,
    pub layers: Vec<LayerWeights>,
    pub norm: GpuTensor,           // final grouped RMSNorm weight
    pub lm_head: WeightTensor,
}
```

4. **`Architecture` trait impl** — `arch_id() = 16`, `name() =
   "k2_horizon"`, `config_from_hfq`, `load_weights`, `new_state`.

**Acceptance:** `cargo test` in the new crate passes config parsing unit
tests against the real `config.json`.

### Phase 2: Grouped RMSNorm kernel

**Goal:** GPU kernel for `layernorm_num_groups=2` RMSNorm.

1. **New HIP kernel** `grouped_rmsnorm_f32` — variance computed per
   `hidden_size / n_groups` chunk. Input `[batch, hidden_size]`, output
   `[batch, hidden_size]`. The weight `[hidden_size]` is applied after
   reshaping back from grouped.

2. **Fused variant** `fused_grouped_rmsnorm_rotate_for_mq` — if MQ4
   weights need FWHT rotation (kmap), fuse the grouped RMSNorm with the
   rotation like qwen35's `fused_rmsnorm_rotate_for_mq`. This is only
   needed if kmap is enabled for K2-Horizon; the MQ4R recipe uses kmap
   (default for MoE), so the fused variant is required.

3. **Prefill variant** `grouped_rmsnorm_batched` — batched RMSNorm for
   prefill path.

**Acceptance:** Kernel output matches numpy reference
(`x_grouped * rsqrt(var + eps) * weight`) to within f32 rounding. Unit
test with `n_groups=2`, `hidden_size=2560` (the real shape).

### Phase 3: MoVA attention forward

**Goal:** Decode + prefill forward for MoVA attention layers.

1. **Value-expert routing kernel** — router GEMV
   (`v_router.weight @ x` → `[64]`), sigmoid, top-4 selection with bias,
   normalize + scale. This is the same `calc_router_weights` math as the
   MoE FFN router but with `score_func="sigmoid"` and
   `scaling_factor=2.5`. Can reuse the MoE router dispatch infrastructure
   with a sigmoid variant.

2. **Value-expert GEMV** — for each of the 4 selected experts, GEMV
   `v_experts[idx] @ x[token]` → `[kv_heads * head_dim]`, apply SiLU,
   multiply by routing weight, scatter-add. This mirrors the MoE FFN
   expert dispatch but with SiLU activation (FFN experts use SwiGLU:
   `silu(gate_proj(x)) * up_proj(x)`). The v_experts are single
   linears (no gate/up split), so this is a simpler GEMV.

3. **Post-attention gate kernel** — elementwise
   `attn_out *= softplus(gate_proj(x))`. `softplus(x) = log(1 + e^x)`
   with `beta = log(2)` (i.e., `log(1 + e^(x * log2)) / log2` — the HF
   code uses `F.softplus(x, beta=math.log(2))`). Can be fused with the
   o_proj GEMV epilogue or done as a standalone elementwise kernel.

4. **Attention itself** — standard causal attention with RoPE. Q/K
   projections are plain linears (no q_norm/k_norm). The attention
   computation (QK^T / sqrt(head_dim), softmax, @ value) is identical to
   qwen35's full-attention path. The only difference is the value source
   (MoVA mixture vs single v_proj). Reuse the existing KV cache +
   attention dispatch infrastructure.

5. **Dense attention (layers 0–2)** — standard q/k/v/o projections, no
   MoVA, but **with** the same softplus post-attention gate
   (`self_attn.gate_proj` exists on dense layers — verified in
   `modeling_k2_horizon.py:227` and the safetensors layout). Closest to
   qwen35's `FullAttnLayerWeights` minus q_norm/k_norm, plus the gate.

**Acceptance:** Per-layer activation cosine vs HF reference > 0.999 at
F32 (using the bf16-oracle lesson from dots-ocr: compare against numpy
F32 reference, not HF bf16 dumps). End-to-end greedy generation matches
HF token IDs on a short prompt.

### Phase 4: Sigmoid MoE FFN forward

**Goal:** Decode + prefill forward for the sigmoid-routed MoE FFN.

1. **Sigmoid router kernel** — `sigmoid(router_logits)`, add bias to
   selection scores, top-8, gather, normalize, scale by 2.5. The
   existing qwen35 MoE forward uses softmax; need a sigmoid variant.
   The router GEMV itself is the same; only the activation + scaling
   differ.

2. **Per-expert FFN** — SwiGLU: `down_proj(silu(gate_proj(x)) *
   up_proj(x))`. Per-expert 2D tensors (not 3D stacked). This matches
   the DeepSeek V4 / MiniMax pattern — reuse the indexed-MoE GEMV
   kernel family (`gemv_hfq4g256_moe_gate_up_k8_indexed` etc.) with
   per-expert weight pointers.

3. **Shared expert** — always-on, no routing. Standard SwiGLU FFN with
   `intermediate_size = moe_intermediate_size * num_shared_experts =
   768`. Same as qwen35's `SharedExpertWeights` forward.

**Acceptance:** MoE FFN output matches numpy F32 reference. End-to-end
generation continues to match HF token IDs.

### Phase 5: Quantizer pipeline

**Goal:** `hipfire quantize` produces a valid `.mq4r` (or `.mq4`) HFQ
file from the safetensors source.

1. **`pipeline.rs`** — add `is_k2_horizon = arch_id == 16` and include
   it in `is_moe_like`. This activates the MoE expert quant paths and
   the `--no-q8-router` fixed-tier logic.

2. **Per-expert 2D tensor handling** — K2-Horizon ships experts as
   separate 2D tensors (`mlp.experts.{E}.gate_proj.weight` etc.), like
   DeepSeek V4. Route through `handle_main_quant` for each 2D tensor
   (not `handle_moe_expert_3d` which expects 3D stacked). The v_experts
   (`self_attn.v_experts.{E}.weight`) are also per-expert 2D tensors —
   same path.

3. **Tensor name matching** — `should_quantize` and `is_q8_tensor` need
   to recognize K2-Horizon tensor names:
   - `self_attn.v_router.weight` → quantize (2D weight)
   - `self_attn.v_router.bias` → F32 passthrough (1D bias)
   - `self_attn.v_experts.{E}.weight` → quantize (2D weight)
   - `self_attn.gate_proj.weight` → quantize (2D weight)
   - `mlp.gate.weight` → quantize (2D weight, router)
   - `mlp.gate.bias` → F32 passthrough (1D bias)
   - `mlp.experts.{E}.{gate,up,down}_proj.weight` → quantize (2D)
   - `mlp.shared_experts.{gate,up,down}_proj.weight` → quantize (2D)
   - `input_layernorm.weight`, `post_attention_layernorm.weight` → F16
     passthrough (1D norm)
   - `embed_tokens.weight`, `lm_head.weight` → Q8 (or MQ4 with
     `--no-q8-router`)

4. **MQ4R recipe** — `--format mq4 --no-q8-router` works generically
   once `is_k2_horizon` is in `is_moe_like`. No K2-Horizon-specific
   recipe code needed.

5. **Metadata** — stamp `arch_id = 16` into the HFQ header. The
   `metadata_json` carries the full `config.json` (including
   `model_type: "k2_horizon"`) so the loader can parse it at serve time.

**Acceptance:** `hipfire quantize --input <safetensors-dir> --format mq4
--no-q8-router --output k2-horizon-36b-a4b.mq4r` completes without error.
The output `.mq4r` file has `arch_id = 16` in its header and loads via
the K2HorizonCarrier.

### Phase 6: Loader + carrier

**Goal:** `hipfire serve` loads the `.mq4r` and serves it.

1. **`K2HorizonCarrier::load`** — implement the full load path:
   - Open HFQ, parse config from metadata
   - Upload embed_tokens, lm_head
   - For each layer, dispatch Dense vs MoE based on `mlp_only_layers`
   - Upload attention weights (dense or MoVA)
   - Upload FFN weights (dense or MoE with per-expert 2D)
   - Upload norm weights (grouped RMSNorm)
   - Initialize KV cache state

2. **`ArchModel` impl** — expose the loaded model for the daemon's
   generate loop.

3. **Daemon wiring** — add K2-Horizon to the daemon's arch dispatch
   (generate, prefill, sampling). The daemon calls arch-specific forward
   functions directly (not through the trait), so a new match arm is
   needed.

4. **Tokenizer** — `PreTrainedTokenizerFast` with the custom BOS/EOS
   tokens. The daemon's tokenizer loader should handle this via the
   `tokenizer.json` + `tokenizer_config.json` files.

5. **Chat template** — the 49.8 KB Jinja template with
   `reasoning_effort` support. The daemon's template renderer needs to
   handle this (may already work if it uses the HF template engine).

**Acceptance:** `hipfire serve k2-horizon-36b-a4b.mq4r` starts, accepts
requests, and generates coherent text. Greedy generation on a short
prompt matches HF token IDs.

### Phase 7: KLD eval + coherence smoke test

**Goal:** Verify quantization quality.

1. **KLD measurement** — run the standard hipfire KLD eval
   (`hipfire eval kld`) comparing MQ4R decode logits against the F32
   oracle. Target: KLD < 0.20 (comparable to qwen3.6-35b-a3b MQ4R at
   ~0.17).

2. **Coherence smoke** — multi-turn conversation + reasoning task.
   Verify output is coherent and follows instructions.

3. **MQ4R vs MQ4** — compare KLD between `--no-q8-router` (MQ4R) and
   default (Q8 fixed tier + MQ4 FFN). MQ4R should be slightly higher KLD
   but faster decode (fewer bytes/token).

**Acceptance:** KLD < 0.25, coherent output, MQ4R decode faster than
MQ4 (Q8 fixed tier) on gfx1100.

## 4. Reusable hipfire infrastructure

### 4.1. Architecture trait (`hipfire-runtime/src/arch.rs`)

Same bring-up contract as qwen35: `arch_id()`, `name()`,
`config_from_hfq()`, `load_weights()`, `new_state()`. Forward is
static-dispatch (not on the trait).

### 4.2. Carrier (`hipfire-loader/src/carriers.rs`)

`K2HorizonCarrier` claims `arch_id == 16`. The carrier registry
auto-detects overlaps via `carriers_are_disjoint` test.

### 4.3. Quantizer pipeline (`hipfire-quantize/src/pipeline.rs`)

The per-expert 2D quant path is already implemented for DeepSeek V4
(`is_deepseek4`) and MiniMax (`is_minimax`). K2-Horizon follows the same
pattern — each expert tensor is a plain 2D weight quantized via
`handle_main_quant`. The MoVA v_experts are also per-expert 2D tensors.

### 4.4. Indexed MoE GEMV kernels

The `gemv_hfq4g256_moe_gate_up_k8_indexed` kernel family (used for
qwen3.5-MoE and DeepSeek V4) dispatches per-expert GEMVs via a device
pointer table. K2-Horizon's MoE FFN experts can reuse this
infrastructure. The MoVA v_experts need a simpler variant (single GEMV
per expert, no gate/up fusion) — a new kernel or a degenerate case of
the existing one.

### 4.5. KV cache + attention dispatch

The existing KV cache (`KvCacheExt`) and attention dispatch
(`kv_cache_attention_dispatch`) work for standard causal attention with
GQA. K2-Horizon's attention is standard once the value states are
computed — the MoVA routing happens before the attention computation,
not during it.

### 4.6. RoPE

Full rotary (`rope_head_dim == head_dim`, `rope_type == "default"`).
This is the simplest RoPE case — no partial rotary, no mrope, no
interleaving. Reuse the existing `apply_rotary_pos_emb` kernel path.

## 5. Non-goals

- **MTP / DFlash speculative decoding** — K2-Horizon has no MTP head in
  the released checkpoint. Speculative decoding support is a follow-on.
- **Expert parallelism (EP)** — single-GPU serve only for now (24 GiB
  XTX). The model is 36B params; MQ4R at ~4.25 bpw ≈ 19 GB, fits in 24
  GiB with Q8 KV cache at modest context. EP is a follow-on.
- **Vision / multimodal** — K2-Horizon is text-only.
- **Other K2-Horizon variants** — only MoVA-36B-A4B is in scope. If IFM
  releases dense or larger MoE variants, the arch crate should handle
  them via config dispatch (dense vs MoE layers), but that is not
  validated here.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Grouped RMSNorm kernel correctness | Unit test against numpy F32 reference at the real shape (2560, n_groups=2) before integration |
| Sigmoid router numerical stability | sigmoid can saturate; use the same f32-then-cast pattern as the HF reference |
| MoVA v_expert dispatch performance | 64 value experts × 4 selected = 4 GEMVs per token. Small experts (768×2560). Should be fast but needs profiling. |
| 24 GiB VRAM budget | MQ4R ≈ 19 GB weights + Q8 KV cache. At 32K context with GQA (8 kv heads, 128 dim), KV = 32K × 48 layers × 8 × 128 × 1 byte (Q8) × 2 (K+V) ≈ 3.1 GB. Total ≈ 22 GB. Tight but fits. |
| Softplus gate precision | `softplus` with `beta=log(2)` is numerically stable for large negative inputs (→ 0) but can lose precision for large positive inputs (→ x). Use the log-sum-exp trick if needed. |
| Chat template complexity | 49.8 KB Jinja template with reasoning_effort. May need template engine updates. Validate early. |

## 7. File map

```
crates/hipfire-arch-k2-horizon/           NEW crate (as built)
  Cargo.toml
  src/
    lib.rs
    arch.rs                               Architecture trait impl + packed-expert loader
    config.rs                             K2HorizonConfig parser
    weights.rs                            weight structs + free_gpu
    load.rs                               K2HorizonBundle (ArchModel)
    forward.rs                            decode forward (MoVA + MoE + dense) + PM4 replay
    lowered.rs                            LayerProgram/SuperOp dispatch (HIPFIRE_FORWARD_LOWERED, opt-in)
    prefill.rs                            batched prefill (currently unused — see gaps)
  tests/
    grouped_rmsnorm.rs                    GPU smoke test (--ignored)

kernels/src/                              shared kernel tree (no per-arch dir)
  rmsnorm.hip                             grouped_rmsnorm_f32 (appended)
  softplus_gate.hip                       fused softplus post-attn gate
  replicate_batched.hip                   [N×K] → [N×K_TOP×K] replicate
  gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip
  gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded.hip

crates/hipfire-runtime/src/arch_mapping.rs    add ("k2_horizon", 16)
crates/hipfire-loader/src/carriers.rs         add K2HorizonCarrier
crates/hipfire-quantize/src/pipeline.rs       add is_k2_horizon to is_moe_like
crates/hipfire-quantize/src/model_filter.rs   tensor name matching for K2-Horizon
crates/hipfire-runtime/src/reset_core.rs      arch_key + inventory for id 16
crates/hipfire-generate/src/dense.rs          generate_k2_horizon (AR path)
Containerfile.k2-horizon                      serving container
docs/plans/k2-horizon-arch-spec.md            this file
docs/plans/k2-horizon-pm4-spec.md             PM4/redline spec
```


## 8. MQ4R quant command (target)

```bash
hipfire quantize \
  --input ~/models/RAW/IFM/K2-Horizon-MoVA-36B-A4B/source/ \
  --format mq4 \
  --no-q8-router \
  --output ~/models/hipfire/IFM/K2-Horizon-MoVA-36B-A4B-MQ4R/k2-horizon-36b-a4b.mq4r \
  --arch-id 16
```

Per the model store layout convention (`~/models/AGENTS.md`):
- Bucket: `hipfire/` (hipfire-only quant)
- Path: `hipfire/IFM/K2-Horizon-MoVA-36B-A4B-MQ4R/`
- File: `k2-horizon-36b-a4b.mq4r`
- One quant per folder (MQ4R)

Serve registration via `models.toml` inside the serve container:
```toml
[aliases]
"k2-horizon:36b-a4b-mq4r" = "k2-horizon-36b-a4b-mq4r"

[models.k2-horizon-36b-a4b-mq4r]
path = "/home/ghazni/models/hipfire/IFM/K2-Horizon-MoVA-36B-A4B-MQ4R/k2-horizon-36b-a4b.mq4r"
```

## 9. Remaining gaps (as of 2026-09-11 second review)

Ordered by impact. Items marked [FIXED] were closed during the
2026-09-11 review passes on this branch.

### Correctness (second pass, 2026-09-11)

0a. **[FIXED] `in_think` misrouted reasoning when thinking "off"** — the
    K2-Horizon Jinja template ignores `enable_thinking`; `reasoning_effort
    | default('high')` always opens a think block. `in_think =
    max_think_tokens != 1` therefore published reasoning as visible content
    for `max_think_tokens=1`. Fixed: `in_think` always starts true,
    `reasoning_effort` is derived from `max_think_tokens` (1→low,
    ≤512→low, ≤2048→medium, else high), and the cap force-closes the think
    block by feeding the matching close tag through the marker arm.

0b. **[FIXED] Missing gfx12 packing guard** — the
    `fix/pack-mq4g256v2-experts` port packed experts unconditionally; the
    upstream fix gates on `is_rdna3()` because gfx1201 tg128 fell 171→77
    tok/s under packed views. Added `packed_experts_supported(gpu)` to both
    the v_experts and MoE expert paths.

0c. **[FIXED] Non-V2 expert dtypes silently misdecoded** — the indexed
    GEMV kernels (`gemv_mq4g256v2_moe_*_k8_indexed_batched*`) hardcode
    fp16 per-128 headers; `packable_mq4_dtype` also accepts qt=13/45 whose
    headers differ. Added `require_v2_expert_dtype` fail-fast checks on
    both v_experts and MoE experts (packed and fallback paths).

0d. **[FIXED] Double host read in `load_packed_experts`** — the packability
    probe read every expert tensor, then the concat loop read them again.
    Single-pass now.

### Correctness (first pass)

1. **[FIXED] `state.n_tokens` not advanced after sequential prefill** —
   `generate_k2_horizon` ran `decode_step` per prompt token but never set
   `n_tokens`, so decode restarted at position 0 and overwrote the KV
   cache. Fixed in `dense.rs`.

2. **[FIXED] Partial-warp hazard in `deepseek4_moe_topk_bias_aware{,_batched}`**
   — launched with `blockDim = n_exp`; for `n_exp=100` (K2-Horizon MoE
   router) the last warp has 28 inactive lanes and `__shfl_down` reads
   undefined values, which can poison the argmax with a garbage expert
   index → OOB `expert_ptrs` read → wild pointer in the indexed GEMV.
   This is the likely root cause of the "batched prefill IMA (hipMemcpy
   D2H code 700)" noted in `dense.rs`. Fixed by launching
   `round_up(n_exp, 32)` threads + defensive `n_exp`/`k_top` guards and
   padded-lane index clamps in both kernels.

3. **[FIXED] PM4 capture hole in MoVA decode** — `forward_mova_value_routing`
   used `memcpy_dtod_at_auto` to replicate `v_x_rot` into `v_rot_batch`;
   D2D memcpys are not recorded by Redline capture, so PM4 replay would
   have used stale rotated inputs. Replaced with `replicate_batched_f32`
   (a recorded kernel launch).

4. **[FIXED] VRAM leak on unload** — `K2HorizonBundle::free_gpu` dropped
   weights without freeing (`DeviceBuffer` has no `Drop`), and the packed
   expert owner tensors were dropped at load scope, making ~20 GB
   unfreeable. Added `v_experts_owner` / `experts_{gate_up,down}_owner`
   fields + `K2HorizonWeights::free_gpu`.

### Perf (third pass, 2026-09-11)

0e. **[FIXED] Rotate-once FWHT — shared `normed_rot` across all
    normed-reading GEMVs** — every MQ4G256V2 projection that reads
    `normed` in a layer applies the same weight-independent FWHT
    rotation, but the prior code rotated `normed` inside each
    `weight_gemv` call (~8 redundant `mq_rotate_x` launches per layer:
    wq, wk, wv, attn_gate, v_router, moe_router, gate, up, plus the
    MoVA/MoE expert-input rotates). Added `state.normed_rot`
    (FWHT(normed), once per norm via `norm_and_rotate`) and
    `state.proj_rot` (rotate scratch for the non-normed GEMV inputs:
    o_proj, w_down, shared down). `gemv_normed` consumes the shared
    buffer via `weight_gemv_prerotated` when the weight uses the fixed
    rotation, falling back to plain `weight_gemv` for AWQ-scaled or
    non-rotating dtypes. The MoVA `v_rot_batch` and MoE `gate_up`
    indexed GEMVs reuse `normed_rot` under the same AWQ guard.
    Verified in-container (gfx1100, K2-Horizon-MoVA-36B-A4B.mq4r):
    HIP decode 83 → 91.6 tok/s, PM4 retained replay 122 tok/s
    in-process / 121 tok/s daemon steady-state, correct output
    (17×23=391, Paris), PM4 tape 1737 launches replays clean.

### Still open

5. **[FIXED] Batched prefill re-enabled** — `forward_prefill_batch` is now
   the default in `generate_k2_horizon` (`HIPFIRE_K2_PREFILL_SEQ=1` opts
   back to sequential). Verified in-container: 400 tok/s prefill on the
   24-token smoke prompt vs ~242 tok/s sequential.

6. **[PARTIAL] End-to-end smoke validated in container** — `docker run
   k2-horizon-serve` loads the .mq4r (24.35 GB VRAM of 25.75 GB), greedy
   chat completions return correct coherent output (17×23=391, sliding-
   window code answer), think-block parsing works (reasoning_content
   separated), think-cap force-close verified. Still owed:
   `serve_harness.py battery` + `chain` for multi-prompt coverage.

7. **[FIXED] PM4 retained-replay now captures and replays** — the
   `gemv_mq4g256v2_residual` scratch-memory rejection was fixed by
   swapping the shared-expert down projection to plain `weight_gemv` +
   `add_inplace_f32` (one extra launch, PM4-safe). Verified in-container:
   `retained route ready: capture=2064 launches, packet_count=1`, decode
   83→92.4 tok/s on the smoke prompt. Still owed:
   `redline_daemon_harness.py` for the formal replay-parity report before
   any perf claim.

8. **[OUT OF SCOPE] KLD eval** — Phase 7 acceptance (KLD < 0.25 vs F32
   oracle) requires the bf16 source + eval harness, not runnable in this
   environment. Deferred; not a merge blocker for the serving path.

9. **[FIXED] Lowered path oracle-validated** — `lowered.rs` was written
   before the rotate-once FWHT optimization (0e) and called bare
   `grouped_rmsnorm_f32` + `weight_gemv`, never populating `normed_rot`.
   `forward_mova_value_routing` / `forward_sigmoid_moe_ffn` read
   `normed_rot` for every routed projection → stale buffer → divergent
   tokens. Fixed: lowered blocks now use `norm_and_rotate` + `gemv_normed`
   / `weight_gemv_prerotated`, matching the hand path. A/B in-container:
   identical greedy token sequence, 90.9 vs 91.5 tok/s. Still opt-in
   (`HIPFIRE_FORWARD_LOWERED=1`) — no perf win, default stays off.

10. **[OUT OF SCOPE] Spec-decode / MTP** — carrier returns "not yet wired"
    for spec decode; K2-Horizon has no MTP head in the released checkpoint
    (non-goal per §5). Confirmed out of scope.

11. **[OUT OF SCOPE] Multi-turn / session state** — `generate_k2_horizon`
    is single-turn AR only; no `serve_harness.py chain` coverage, no
    prefix-cache interaction tested. Confirmed out of scope for this branch.

12. **[FIXED] `flash_partials` over-allocation** — was sized with a
    `FLASH_PREFILL_SUBBATCH` (16) multiplier decode never needs. Split:
    decode `state.flash_partials` is now a single query row sized against
    `q8_flash_tile_size` (32 on gfx1100 → `n_heads*1024*(2+head_dim)` at
    max_seq=32768 ≈ 17 MB); the ×SUBBATCH buffer moved to `PrefillScratch`
    and is freed after prefill. **Trap found:** sizing decode with /128
    (attn_tile_size) under-allocates 4× vs the q8 kernel's tile=32 → OOB
    partials write → GPU page fault. The old ×16 multiplier had been
    accidentally masking it. Verified in-container: 93.8 tok/s, no fault.

13. **[VERIFIED] VRAM budget at max_seq** — `ctx.max_seq` reaches
    `new_with_max_seq` via `load_k2_horizon_bundle`; the Containerfile's
    `max_seq=32768` config is honored. Measured 24.35 GB used of 25.75 GB
    at 32k ctx — tight but stable.

14. **[VERIFIED] Concat-GEMV weight fusion** — all K2-Horizon 2D
    weights are MQ4G256V2 (qt=44, same K=2560), so same-K projections
    byte-concatenate into one GPU allocation and one GEMV launch.
    `load_fused_wts` (arch.rs) uploads `[wq‖wk‖wv‖gate]` (dense),
    `[wq‖wk‖v_router‖gate]` (MoVA), `[w_gate‖w_up]` (dense FFN), and
    `[router‖shared_gate‖shared_up]` (MoE FFN) as fused weights; the
    per-tensor `WeightTensor`s are `sub_offset` views into the owner
    blob (freed once in `free_gpu`). Forward paths slice views into
    `state.attn_fused_out` / `state.ffn_fused_out`; the MoVA router
    logits are read in place from the fused output. Falls back to
    per-tensor loads on any dtype/K mismatch.
    **Verified on gfx1100** (K2-Horizon-MoVA-36B-A4B.mq4r, q8 KV):
    PM4 tape 1737→1080 launches (−38%), `bench_k2_horizon` decode
    90.1→115.4 tok/s (+28%), prefill 241.5→734.6 tok/s (3.0×),
    `redline_daemon_harness --pm4` pass=True (exact=True,
    gdn_frame_exact=True, median 117.6 tok/s), `serve_harness battery`
    5/5 turns coherent (avg decode 111.6 tok/s). `PrefillScratch`
    logits buffer aliased to `state.logits` via borrowed view (−1 MB).
    `PREFILL_MAX_BATCH` 256→128 was tried and reverted: −32% prefill
    for ~110 MB scratch, not worth it at 1.4 GB headroom.
