# Serve (OpenAI-compatible HTTP)

`hipfire serve` starts the native Rust HTTP control plane, owns one GPU daemon
process, and exposes an OpenAI-shaped chat API. Defaults come from typed TOML
configuration ([CONFIG.md](CONFIG.md)); the HTTP surface is implemented by
`crates/hipfire-cli` and the shared transport lives in `crates/hipfire-client`.

| Field | Default (source) |
|---|---|
| Bind host | `serve.host = "127.0.0.1"` (loopback only) |
| Port | `serve.port = 11435` |
| Pre-warm model | `serve.default_model = "qwen3.5:9b"` or a positional model arg |
| Idle unload | `serve.idle_timeout_seconds = 300` (`0` = never) |
| Max request body | `serve.max_request_bytes = 67108864` (64 MiB) |
| Admission queue | `serve.max_queue = 64`, `serve.queue_timeout_ms = 600000` (10 min) |
| Prefix cache | `serve.prefix_cache = false` (opt-in; paged slots, text only) |
| Pid / log | `~/.hipfire/serve.pid`, `~/.hipfire/serve.log` |

Truth state: **shipped / ref-pinned** for the HTTP contract and lifecycle
below. Branch-only model routes (for example LFM framing details) are not
implied by this page — see [MODELS.md](MODELS.md) and [VALIDATION.md](VALIDATION.md).

## Security (no auth / no TLS)

The native handler implements **no authentication and no TLS**.
Anyone who can reach the bind address can call every endpoint, including
chat completions. Default bind is `127.0.0.1` (loopback only); set
`serve.host = "0.0.0.0"` (or pass `0.0.0.0:11435`) to listen on all interfaces.

- Prefer loopback for local use: `hipfire serve 127.0.0.1:11435`
- Expose beyond localhost only behind a trusted network or an authenticated
  TLS reverse proxy you control. Do not publish the raw port to the internet.

## Start and stop

```bash
hipfire serve                         # foreground; Ctrl-C stops (default bind 127.0.0.1)
hipfire serve 127.0.0.1:11435         # loopback-only (preferred local bind)
hipfire serve -d                      # background (setsid/nohup); polls /health up to 60s
hipfire serve qwen3.5:9b -d           # pre-warm a specific tag this run
hipfire serve 0.0.0.0:11435 -d        # all-interfaces bind — unauthenticated; see Security
hipfire serve --no-prewarm -d         # bind first; load on first request
hipfire serve --kv-mode q8 --idle-timeout 0 -d
hipfire serve --tp 2 -d               # expert-parallel load (multi-GPU arches)
hipfire serve --spec off -d           # plain AR (no MTP/DFlash) for every load, e.g. an A/B
hipfire serve --max-seq 32768 -d      # KV context for every load (replaces memory.max_seq)

hipfire ps                            # daemons / quantize / uploads + port state (Linux)
hipfire stop                          # SIGTERM tracked pid in serve.pid (safe default)
# Forceful — inspect first (hipfire ps, curl /health):
hipfire stop --force                  # also kill this install's untracked daemons (TERM, KILL after 5s)
hipfire stop --all                    # --force plus pkill orphan quantize jobs
hipfire restart -d                    # stop --force semantics, then serve again
```

`serve -d` first checks `serve.pid` and `/health`. If a tracked server already
answers on the same port it prints `hipfire serve already running (pid N, port
P, log …)` and exits 0; a tracked server on another port, or a different
process answering on the port, exits 1 with that reason. Readiness requires
the `/health` `pid` to be the child it just spawned, and if the child exits
early the error includes the last `FATAL`/`hipfire:` line from `serve.log`.

**Forceful lifecycle flags.** Plain `hipfire stop` only SIGTERMs the tracked
pid after ownership checks — use that by default. `--force`, `--all`, and
`restart` also stop this install's other daemons: PIDs from the pidfile and
the GPU lock, each verified by command line (this install's daemon binary)
and matching `HIPFIRE_HOME`, sent SIGTERM and then SIGKILL after 5 s. They no
longer kill processes by name or by port; if something else still holds the
port, the CLI prints `port N is still in use by a process that is not this
install's hipfire; not killing it`. `--all` also runs `pkill -f` against
release quantize binaries.

Flags (also accepted by `restart`):

| Flag | Effect |
|---|---|
| `-d` / `--detach` / `--background` | Fork detached child; log → `~/.hipfire/serve.log` |
| `--kv-mode <m>` | KV preset for models this service loads (sent as the `kv_mode` load param; same contract as `run --kv-mode`) |
| `--idle-timeout <s>` | Sets `HIPFIRE_IDLE_TIMEOUT` (0..86400) |
| `--no-prewarm` | Sets `HIPFIRE_NO_PREWARM=1` |
| `--tp N` | Sets `HIPFIRE_TP` (1..64) for expert-parallel load |

Positional forms: `[model] [host] [port]`, or `host:port` / `[ipv6]:port`.
Without a model arg, pre-warm uses `HIPFIRE_MODEL` if set, else
`cfg.default_model`.

Config and env owners for bind, idle, queue, and body limits:
[CONFIG.md](CONFIG.md), [env-vars.md](env-vars.md).

## Lifecycle

1. **Bind + pid record.** Serve writes its numeric pid to
   `~/.hipfire/serve.pid`. `stop` verifies that `/proc/<pid>/cmdline` still
   identifies a native `hipfire serve` process before signaling it.
2. **Daemon.** Spawns the Rust `daemon` example over stdio JSON.
3. **Pre-warm (default).** Loads the chosen model asynchronously. Failures log
   and leave the process serving; the model loads on the first real request.
   Multi-slot serve (`serve.multi_slot = true`) fails closed instead: the slot
   engine is its only backend, so a failed pre-warm (for example, slot-engine
   allocations that do not fit beside another process on the card) logs the
   load error, answers `/health` with 503 `unhealthy`, and exits 1.
4. **HTTP.** The native server accepts traffic. Serve is single-stream by
   default: one generation holds the daemon at a time and later requests wait
   in a bounded queue (`serve.max_queue`). Continuous batching covers only
   thinking-off, single-turn Qwen requests (`serve.continuous_batch_size`).
   The VMM continuous-batching route (`serve.vmm_batch`,
   `HIPFIRE_SERVE_VMM_BATCH`, **default off, experimental**) instead batches
   eligible TP1 Qwen3.5-family requests over the one resident weight set with
   per-request VMM KV/DeltaNet owners (no fixed lanes, no second model). The
   loaded ack then reports `continuous_batch_route: "vmm"` and the actual
   owner receipt (`vmm_batch_kv_backend`, `vmm_batch_kv_mode`,
   `vmm_batch_max_seq_bound`). By default it runs the **exact** route
   (per request byte-identical to the singleton route; prefill uses the
   singleton's own chunking, so a long prompt prefills in singleton-sized
   chunks), staged only where the executor admits it (gfx1201, uniform
   MQ4G256V2 dense layers); the ack reports `continuous_batch_exact: true`.
   `serve.batch_nonexact` / `HIPFIRE_SERVE_BATCH_NONEXACT=1` selects the
   non-exact shared slots body instead (`continuous_batch_nonexact: true`).
   `serve.batch_sampled` / `HIPFIRE_SERVE_BATCH_SAMPLED` (default on; `0` is
   the kill switch) additionally admits sampled (temperature>0) requests to the
   exact route, each in the lane mode its singleton route selects (AR, native
   MTP, FAST_SAMPLE selector-chain DFlash) and byte-identical to that
   singleton for the same wire seed. A sampled request the batch cannot serve
   in its own mode (stop strings or a think budget on a speculative lane, or
   a prompt + `max_tokens` beyond the lane engine's cap, 32,768 for the MTP
   head) stays on the singleton route, never an AR substitute;
   sampled speculative singletons are not promoted mid-generation. The load
   ack reports `continuous_batch_sampling: "sampled"` and
   `continuous_batch_sampled_modes`; each batched request's
   `continuous_batch` receipt carries `spec_mode`, `sampling` and
   `equivalence: "singleton_exact"`. With the flag off sampled requests stay
   on the serialized singleton route.
   Greedy requests (and sampled ones under `serve.batch_sampled`) batch with
   penalties applied as on the singleton route; `serve.batch_spec`
   (`HIPFIRE_SERVE_BATCH_SPEC=0` disables it) lets lanes decode as
   cross-request MTP/DFlash speculative verify rows. A request that is alone at
   dispatch keeps the unchanged singleton
   route (including native MTP/DFlash); requests arriving while a singleton
   generation runs wait for it to finish, so their TTFT includes that
   generation (known gap). The fixed-slot `serve.multi_slot` engine below is
   the legacy route.
   With `serve.multi_slot = true`, up to `serve.multi_slot_slots` requests
   decode at once on the experimental multi-slot engine. Concurrent requests
   can produce different greedy text than serial requests at ≥4 slots; output
   is deterministic for a fixed batch composition. 2 slots matched serial in
   earlier testing; on 35B-A3B MoE checkpoints they diverged from serial on
   1–4 of 4 prompts. A waiter that is still queued after `serve.queue_timeout_ms`
   (default 10 minutes, long enough for one full generation; `0` waits
   forever) gets
   **503** "server busy" with `Retry-After`. A failed `accept` (for example,
   out of file descriptors) is logged and retried, never fatal. At most 512
   connections are served at once; further connects wait in the listen
   backlog. A client must send a complete request head within 30 s of
   connecting or of its previous response, which also closes idle keep-alive
   connections.
5. **Idle eviction.** When `idle_timeout > 0`, an interval unloads the model
   after idle seconds **and** only when no generation is in flight and the
   serve lock is free. Next request reloads.
6. **Stop.** `hipfire stop` validates pid ownership before SIGTERM. Stale reused
   pids are **not** killed; the pidfile is removed instead. On SIGTERM or
   Ctrl-C, serve stops accepting and frees the port at once, lets requests in
   flight finish for up to 30 s, then exits; a second signal exits at once.

Detached readiness: parent polls `GET /health` for up to **60 seconds**.
`/health` means the native process is answering; inspect `model` and
`loading_model` to distinguish ready, loading, and unloaded state.

## Endpoints

Implemented paths (anything else → `404`):

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Liveness JSON: `status`, `model`, `loading_model`, `pid`, `native`, `capabilities` |
| `GET` | `/v1/models` | `{ data: [{ id }, ...] }` from local model files |
| `GET` | `/stats` | Serve telemetry: uptime, queue depth, requests served, recent decode tok/s |
| `POST` | `/v1/chat/completions` | Chat completions (stream or non-stream) |
| `POST` | `/v1/messages` | Anthropic Messages API (stream or non-stream), adapted onto chat completions |
| `POST` | `/v1/responses` | OpenAI Responses API (stream or non-stream), adapted onto chat completions |
| `GET` | `/ui`, `/ui/*` | Embedded chat UI (opt-in; see below) |

There is **no** `/v1/completions` route in the current CLI serve.

### Embedded chat UI (`--ui` / `serve.ui` / `HIPFIRE_SERVE_UI`)

Off by default. `hipfire serve --ui` (or `--open`, which implies `--ui`
and launches the browser) serves a dependency-free chat frontend at
`/ui` on the same listener; `serve.ui = true` or `HIPFIRE_SERVE_UI=1`
enable it for any serve start, including detached ones. `GET /`
redirects to `/ui` only while the UI is enabled.

The UI talks to the same `/v1/chat/completions` gateway as any other
client and keeps chat history in the browser's IndexedDB — serve stores
nothing on its behalf. Like every serve endpoint it is **unauthenticated**:
binding a non-loopback host with the UI enabled prints a warning, since
that makes the gateway reachable from any browser on the network.

For the sealed MQ4R reproduction, default-model handoff, and copyable Hermes
Agent / Pi custom-provider configuration, see
[`GOLDEN-REDLINE.md`](GOLDEN-REDLINE.md).

### `GET /health`

`200` with `status: "ok"` while the daemon is up. `503` with
`status: "restarting"` while serve respawns a daemon that exited (crash, panic,
or a sticky GPU fault, after which the daemon exits 75) and reloads the
resident model; `503` with `status: "unhealthy"` if respawning failed, or if
multi-slot serve could not load its model (pre-warm, or the reload after a
respawn), after which serve exits 1 so a service manager can restart it.
`model` is the loaded tag/path or `null` when idle/unloaded or after a failed
model switch; `loading_model` names an asynchronous pre-warm or post-restart
reload.

Model switch lifecycle: a load request is first admitted read-only; a refused
admission leaves the prior model (and its drafter) loaded. Once admitted, a
single-device or pipeline-parallel load (Flash-Next/Qwen4 included) unloads
the prior model and requires clean VMM state before constructing the new
one, so a failure after admission leaves no model loaded and the error says
`no model loaded`. Only tp>1 expert-parallel loads stage the new model before
retiring the prior one. Flash-Next with VMM QSA state (gfx1151/gfx1201 automatic) owns
VMM arenas of its own: the target and MTP context arenas and the gathered
attention workspace. All of them are released when the model unloads, so the
next load's clean-VMM check starts from zero owners in either swap direction.

```bash
# Loopback example (safe default for local smoke):
curl -s http://127.0.0.1:11435/health
```

### `POST /v1/chat/completions`

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": true
  }'
```

> **`max_tokens` includes reasoning.** Qwen tags think by default
> (`thinking = "on"`), and the `<think>` span counts against `max_tokens`. With
> a modest cap the whole budget can go to reasoning, so the response carries
> only `reasoning_content`, `content` is `""`, and `finish_reason` is `length`.
> Most OpenAI clients show only `content`, so this looks like a blank answer.
> Send `"chat_template_kwargs": {"enable_thinking": false}` (as above) for
> direct answers, or raise `max_tokens` / omit it to keep thinking.

- **`stream: true`** (typical clients): SSE `data: {chat.completion.chunk}`
  lines until `data: [DONE]`. The `200` and the role chunk go out when
  generation sends its first frame, or after 15 s of silence (cold load, long
  prefill), so a request that fails before then gets the same JSON error and
  status as a non-stream request. A stream that fails after that ends with
  `data: {"error": {"message", "type"}}` and then `data: [DONE]`. Until
  `[DONE]`, 15 s without a frame sends a `: keepalive` SSE comment, which
  SSE parsers ignore.
- **`stream: false` / omitted falsey:** single `chat.completion` JSON body.
- Oversized body → **413** before the daemon lock (Content-Length or streamed
  cap at `max_request_bytes`).
- Saturated admission queue → **503** with `Retry-After`.
- Invalid JSON body → **400**.
- Daemon errors map on their typed class: `validation`, `context_length`,
  `unsupported` → **400**; `transient` → **503** with `Retry-After: 1`; any
  other class → **500**. An unknown model → **404**.

Request fields honored by the serve layer (non-exhaustive; sampling falls
through to per-model / registry / daemon defaults when omitted):

| Field | Notes |
|---|---|
| `model` | Tag or path of an installed model; reloads if different from the resident model. A registry model that is not installed is 404 ("run `hipfire pull`") unless `serve.allow_request_pull = true`; a file outside the models directory, the catalog and the pre-warm model is 404 unless `serve.allow_request_paths = true`. |
| `max_tokens` / `max_completion_tokens` | Omitted: the configured default (Qwen tags 81920, DeepSeek V4 Flash 393216, else `generation.max_tokens`) is fitted to the context left after the prompt, minus 64. Explicit and at or above the loaded context: 400. Never triggers a reload. |
| `messages` | OpenAI chat messages (required for useful chat) |
| `messages[].content[].image_url` | One base64 PNG/JPEG data URI for VL models; remote URLs and multiple images are rejected |
| `stream`, `stream_options.include_usage` | Streaming + optional usage on stream end |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty` | Sampling; explicit request values win, otherwise per-model TOML / registry-card values are applied. On the Qwen AR samplers (GPU kernel and the CPU sampler used by Flash-Next AR, grammar-constrained Qwen turns and the multi-slot VL first token) `top_k` keeps at most k candidates from a 64-wide gather (absent = 20; `0` or > 64 = 64) and `min_p` drops candidates below `min_p` × the top candidate's probability, both before `top_p`. At `temperature: 0` all three are no-ops. Other routes do not apply every field yet: Gemma4 eager, LFM2, Cohere2, MiniMax, DeepSeek V4 AR, LLaMA AR and single-slot Qwen3.5-VL image turns ignore the request `top_k`; those plus Muse Glimmer, DeepSeek V4 speculative and n-gram speculative decoding ignore `min_p`; Qwen2, Maple and dots.ocr text decode greedily. |
| `seed` | OpenAI-compatible sampling seed: non-negative integer (≤ u64::MAX). Same seed is reproducible only when prompt, sampling parameters, and execution shape match. Cold vs prefix-resumed and solo vs co-batched shapes may differ until `serve.batch_invariant` is implemented; `null`/omitted = fresh entropy. Negative/fractional/non-integer values are rejected. |
| `presence_penalty`, `frequency_penalty` | Forwarded natively to the daemon (≥ 0); `presence_penalty` also inherits per-model / registry defaults |
| `max_tokens` | Generation cap, shared by the `<think>` span and the answer (see the note under the example above) |
| `stop` | A string or an array of up to 4 strings, each ≤ 64 characters; any other value is a 400. Matched on the answer only (never inside reasoning), across token boundaries; streaming holds back a possible partial match, the stop text and everything after it are not returned, and `finish_reason` is `stop` (or `tool_calls` when a call completed before the match). Honoured on the Qwen3.5-family routes (`qwen_ar`, `qwen_dflash`, which covers MTP) and Flash-Next (`qwen4_ar`, `qwen4_spec` = native MTP); every other route answers 400 instead of ignoring it |
| `tools` | Tool definitions (with structured `messages` when Jinja chat is on). Flash-Next accepts them with native MTP attached: greedy tool turns run on MTP and match AR byte for byte |
| `response_format` | **Not enforced on the standard route.** Any value (`json_object`, `json_schema`, `text`, even an unknown type) is accepted unvalidated with HTTP 200 and is not forwarded to the daemon; the model output is unconstrained (a model asked for JSON may still wrap it in a code fence). Only `serve.multi_slot = true` enforces `json_schema` (pre-sampling grammar mask) and refuses `json_object` / unknown types with 400 |
| `chat_template_kwargs.enable_thinking` | Mode axis. `false` forces a no-think turn (Qwen native empty closed think). Independent of effort and cap. |
| `chat_template_kwargs.preserve_thinking` | Keep `<think>` in final non-stream content |
| `thinking` / `thinking.type` | Mode axis (`on`/`off`, or DeepSeek-style `enabled`/`disabled`). Thinking disabled wins and drops effort/cap with a warning. |
| `reasoning.effort` / `reasoning_effort` | **Semantic effort only** — prompt strength. Never mapped to a think-token budget. Family ladders differ; unsupported/cross-family values are dropped with a warning (see below). |
| `reasoning.max_tokens` / `max_think_tokens` | Explicit **integer** think-span cap on Qwen Jinja contracts. `0` or omitted = uncapped. DeepSeek, Gemma, Glimmer, and unsupported contracts drop it with a warning. Independent of effort. |
| `thinking_budget` / `reasoning.budget` | Legacy **named** cap preset only on non-effort-native Qwen templates that still accept it. Dropped+warned elsewhere. |

### `POST /v1/messages` (Anthropic Messages API)

An adapter onto the chat-completions route, not a second generation path.
The request becomes a chat-completions body, which then goes through the
same admission (batching eligibility and model lease), tool gate, worker,
tool-call parser, think routing and stop handling as `/v1/chat/completions`.
Serve sends the daemon the same `generate` request as the matching chat
request, so batching and the session cache behave the same. The response is
rendered back as an Anthropic `message` object or SSE events.

```bash
curl -N http://127.0.0.1:11435/v1/messages \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen3.5:4b",
    "max_tokens": 512,
    "system": "Be brief.",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

| Messages field | Chat-completions equivalent |
|---|---|
| `model`, `max_tokens` (required) | `model`, `max_tokens` |
| `system` (string or text blocks) | A leading `system` message. Text blocks are concatenated with no separator, as chat content parts are |
| `messages[]` `text` blocks | Plain-string `content`. Text-only turns keep the plain-string shape the batch routes require |
| `image` block (base64; `url` source) | One `image_url` part (only base64 PNG/JPEG is accepted, as on chat) |
| assistant `thinking` blocks | `reasoning_content` (replayed as for chat; `signature` ignored). `redacted_thinking` is dropped |
| assistant `tool_use` blocks | `tool_calls` (input serialized as the arguments string) |
| user `tool_result` blocks | `tool` messages with `tool_call_id = tool_use_id`, placed before the turn's text. `is_error: true` prefixes the result with `Error: ` |
| `tools` (`name`, `description`, `input_schema`) | Function tools. Server tools (`type` other than `custom`) → 400 |
| `tool_choice` `auto` / `any` / `tool` / `none` | `auto` / `required` / named function / `none`. `disable_parallel_tool_use` is not enforced, like chat's `parallel_tool_calls` |
| `thinking` `enabled` (+ `budget_tokens`) / `adaptive` / `disabled` | `thinking.type` enabled/disabled; `budget_tokens` → `max_think_tokens` |
| `thinking` absent | Thinking **off** (the Messages API default). Send `thinking` to enable it |
| `stop_sequences` | `stop` (same limits: up to 4, each ≤ 64 characters; an empty list is not forwarded) |
| `temperature`, `top_p`, `top_k` | Same fields |
| `stream` | SSE (below) |
| `metadata`, `service_tier`, `cache_control` | Ignored |

Refused with 400 rather than silently changed: a final assistant message
(response prefill), `document` and other unsupported block types, images
inside `tool_result`, server tools and `mcp_servers`.

Response: `content` holds a `thinking` block (reasoning, `signature: ""`),
then a `text` block, then the released `tool_use` blocks. Each appears only
when it has content. Tool ids are the daemon's or `toolu_<request>_<n>`.
`stop_reason`: `end_turn` (stop), `max_tokens` (length), `tool_use` (released
tool calls). The daemon's terminal does not say which stop sequence matched,
so a stop-sequence stop also reports `end_turn` with `stop_sequence: null`.
`usage`: `input_tokens` is the prompt minus session-cache hits,
`cache_read_input_tokens` is the cache hits, `cache_creation_input_tokens` is
always 0, and `output_tokens` is the chat `completion_tokens`. `timings` and
`hipfire` carry the same evidence as on chat.

Streaming: `message_start`, then one `content_block_start` /
`content_block_delta` (`thinking_delta` or `text_delta`) / `content_block_stop`
run per thinking or text span, as generated. The released `tool_use` blocks
follow at the end, each with one `input_json_delta` holding the full arguments.
Then `message_delta` (`stop_reason`, `usage`) and `message_stop`. The
`message_start` usage is zero, because token counts are known only at the
terminal, and `message_delta` carries the real counts. Commit timing,
keepalives and the pre-commit JSON errors work as on chat. A failure after
commit ends the stream with `event: error`.

Errors use the Anthropic shape `{"type": "error", "error": {"type", "message"}}`
with chat's status mapping: 400 `invalid_request_error`, 404
`not_found_error`, 413 `request_too_large`, 503 `overloaded_error` (with
`Retry-After`), 500 `api_error`. `POST /v1/messages/count_tokens` is not
implemented (404).

### `POST /v1/responses` (OpenAI Responses API)

An adapter onto the chat-completions route, like `/v1/messages`: same
admission, batching eligibility, session cache, tool parsing, think routing
and stop handling. Serve stores no responses. The client resends the
conversation in `input` (`store` is always reported `false`).

```bash
curl -N http://127.0.0.1:11435/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:4b",
    "instructions": "Be brief.",
    "input": "hi",
    "max_output_tokens": 512,
    "stream": true
  }'
```

| Responses field | Chat-completions equivalent |
|---|---|
| `instructions` | A leading `system` message |
| `input` string | One `user` message |
| `input` message items (`user` / `system` / `developer` / `assistant`; `input_text` / `output_text` parts; `input_image` with `image_url`) | Chat messages with plain-string content (an image keeps the array form, as on chat) |
| `reasoning` items | `reasoning_content` of the following assistant turn (`content` text, else the `summary` text) |
| `function_call` items | `tool_calls` of the current assistant turn (`call_id` → id) |
| `function_call_output` items | `tool` messages (`call_id` → `tool_call_id`) |
| `tools` (`type: "function"`) | Function tools. Hosted tools → 400 |
| `tool_choice` `auto` / `none` / `required` / `{type: function, name}` | Same |
| `max_output_tokens` | `max_tokens` (omitted: fitted to the context as on chat) |
| `temperature`, `top_p`; extensions `top_k`, `stop` | Same fields |
| `reasoning.effort` | `reasoning_effort` (absent: the server's reasoning default, as on chat) |
| `text.format` `json_schema` / `json_object` | `response_format` (enforced only where chat enforces it) |
| `include`, `store`, `metadata`, `user`, `parallel_tool_calls`, `truncation` | Accepted; not enforced (`parallel_tool_calls` is ignored, as on chat) |

Refused with 400: `previous_response_id`, `conversation`, `prompt`,
`item_reference` items, `background: true`, hosted tools and `input_file`.

Response: `output` holds a `reasoning` item (`content[0].reasoning_text`,
empty `summary`), then a `message` item (`output_text`), then the released
`function_call` items. Each appears only when it has content. Call ids are the
daemon's or `call_<request>_<n>`. `status` is `completed`, or `incomplete`
with `incomplete_details.reason: "max_output_tokens"` on length. `usage` has
`input_tokens` (session-cache hits included and reported in
`input_tokens_details.cached_tokens`), `output_tokens` and `total_tokens`.
Reasoning tokens are not counted separately, so `output_tokens_details` is
omitted.

Streaming: `response.created`, `response.in_progress`, then per reasoning
span `response.output_item.added` / `response.reasoning_text.delta` /
`response.reasoning_text.done` / `response.output_item.done`, and per text
span `output_item.added` / `content_part.added` / `output_text.delta` /
`output_text.done` / `content_part.done` / `output_item.done`. Released
function calls follow (`output_item.added`,
`function_call_arguments.delta` with the full arguments,
`function_call_arguments.done`, `output_item.done`), then
`response.completed` (or `response.incomplete`) with the full response. Every
event carries `sequence_number`. A failure after commit sends `error`, then
`response.failed`. Errors before commit are OpenAI-shaped `{"error": {...}}`
with chat's status mapping.

### `POST /v1/images/generations`

OpenAI-shaped image generation on a loaded diffusion checkpoint (arch 40
FLUX.1, arch 45 FLUX.2 Klein). Body fields: `model` (must already be loaded on
this server), `prompt`, optional `width` /
`height` (or OpenAI `size: "WxH"`), `steps` (defaults to the architecture
default: 4 for `flux.schnell`, 28 for `flux.dev`), `seed`, `n` (must be 1),
`response_format` (must be `b64_json`). References the same denoise path as the
`img_generate` daemon message (`hipfire img`).

```json
{"model": "flux.schnell:1", "prompt": "a tiny lighthouse on a rock, sunset", "size": "512x512", "response_format": "b64_json"}
```

Response: `{ "data": [ { "b64_json": "…", "width": 512, "height": 512, "seed": 0, "steps": 4 } ], "model": "…", "hipfire": { "ms": 1234 } }`.

This body takes no image input. A request with an `images` field is refused
with a 400 that points at `/v1/images/edits`.

### `POST /v1/images/edits`

OpenAI-shaped reference edit (FLUX.2 Klein, arch 45 only): `multipart/form-data`
with one to four `image` file parts (PNG or JPEG, at most 32 MB each after
upload) plus the text fields of `/v1/images/generations` (`prompt` required;
`size`, `steps`, `seed`, `n`, `response_format`). The image bytes travel in
the request; the server never reads a file the client names. Each reference
is area-capped at 1 MP and floored to a multiple of 16, and conditions every
denoise step without being denoised. Without `size` the output takes the
first reference's size. The whole body counts against `serve.max_request_bytes`.

```bash
curl -s -X POST http://127.0.0.1:11580/v1/images/edits \
  -F image=.png -F prompt="make the bicycle blue" -F steps=4 -F seed=0 \
  | jq -r .data[0].b64_json | base64 -d > bike-blue.png
```

Response: the same shape as `/v1/images/generations`.

### Reasoning request contract

Mode, effort, and cap are three independent axes — full key table and budget
map in [`CONFIG.md`](CONFIG.md). HTTP accepts the same meanings as CLI/config.

**Normalization (warn + drop, not silent reinterpretation):**

- Recognizable unsupported or cross-family values are dropped or aliased with
  `[WARN: INVALID CONFIG]`. Warnings appear in the server log and, on OpenAI
  responses, under `hipfire.reasoning` / `hipfire.config_warnings` metadata.
- Malformed types and out-of-range integers remain hard request errors.
- Effort is **never** converted into a token cap (including on Qwen3.6 and other
  non-effort-native templates).
- Qwen3.8 and Ornith 1.5 default to an uncapped think span unless the request sets a
  positive integer cap (semantic effort stays independent of that cap). DeepSeek V4 and
  Muse Glimmer expose semantic effort but no independent parent-defined cap; cap fields
  are dropped+warned.

**Examples:**

```bash
# Qwen3.8 — disable thinking natively (empty closed think block)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'

# Qwen3.8 — semantic effort only (still uncapped unless max_think_tokens is set)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8:27b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "medium"
}'

# Ornith 1.5 — Qwen3.8-compatible semantic effort (default xhigh; still uncapped unless max_think_tokens)
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ornith-1.5:35b-a3b",
  "messages": [{"role": "user", "content": "plan a refactor"}],
  "reasoning_effort": "low"
}'

# Qwen3.6 — no native effort: effort is dropped+warned; set an explicit cap if wanted
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6:27b",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "low",
  "max_think_tokens": 2048
}'

# DeepSeek V4 — mode + effort; medium/xhigh alias to high; no cap from effort
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "hi"}],
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}'

# Gemma4 — boolean thinking (official default off); unsupported effort/cap dropped
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma4:31b",
  "messages": [{"role": "user", "content": "hi"}],
  "chat_template_kwargs": {"enable_thinking": true}
}'

# Muse Glimmer — always reasons via Onyx; strength dial; off is dropped+warned
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "muse-glimmer",
  "messages": [{"role": "user", "content": "hi"}],
  "reasoning_effort": "high"
}'
```

| Family | Mode | Effort | Cap |
|---|---|---|---|
| Qwen3.8 | `enable_thinking=false` → empty closed think | `low` \| `medium` \| `xhigh` (default `xhigh`; semantic prompt only) | Integer only; named budget dropped |
| Ornith 1.5 | same Qwen on/off | Qwen3.8-compatible `low` \| `medium` \| `xhigh` (default `xhigh`; not a budget) | Integer only; named budget dropped |
| Qwen3.6 | on/off | Dropped+warned (never → cap) | Named preset or integer if set |
| DeepSeek V4 | `thinking.type` enabled/disabled (default on) | `low` \| `high` \| `max` (`medium`/`xhigh`→`high`) | Integer only; default uncapped |
| Gemma4 | Boolean; default **off** | Dropped+warned | Dropped+warned unless explicit engine integer |
| Muse Glimmer | Always on (off dropped) | `low` \| `medium` \| `high` \| `xhigh` | Integer only; default uncapped |

When `messages` contains no `system` or `developer` role, the serve layer inserts `prompt.system` from a per-model TOML override or the registry card's `recommended_settings.system_prompt`. A client-supplied system/developer message always wins.

`finish_reason` values emitted to clients: `stop`, `length`, `tool_calls`. A
turn that runs out of `max_tokens` while still reasoning ends with `length`:
the partial `reasoning_content` is returned and `content` is empty. A turn
the model ends (end-of-turn token) while still reasoning ends the same way
with `stop`. Neither turn releases a tool call or is cached.

Streaming error contract: a failure that happens BEFORE the first byte is a
plain HTTP error status (no SSE body). A failure after the stream has started
emits an OpenAI-shaped SSE `data: {"error": {...}}` frame followed by
`data: [DONE]`, so a streaming client always sees a terminal event instead of
a silently truncated `200`. Non-streaming failures keep the typed
status mapping (`400` request/config, `503` + `Retry-After` overload or
transient, `500` internal).

Prefix-cache capable arches (daemon `cache_capable`, or arch allowlist
`deepseek4` / `qwen3_5` / `qwen3_5_moe`) skip per-request `reset` so multi-turn
LCP can hit. Other arches reset every request (stateless OpenAI shape).

### Client notes (Zed / OpenAI-compatible agents)

Live-GPU endpoint validation on gfx1100 and gfx1201 exercised Zed-shaped
streamed requests against the canonical Qwen3.6 35B-A3B MQ4R artifact. The
matrix covered automatic and required tool calls, multiple calls in one turn,
multi-turn `reasoning_content` replay, `tool_choice: "none"`, usage chunks, and
strict UTF-8 SSE/JSON decoding. This validates the OpenAI-compatible endpoint,
not the Zed application UI itself.

Inbound assistant `tool_calls[].function.arguments` strings are parsed into
JSON objects for Qwen template replay; outbound OpenAI arguments remain JSON
strings. Tool-call deltas are emitted in bulk at the terminal chunk,
`finish_reason=tool_calls` is produced, and disconnect cancels in-flight
generation.

Current caveats for agent clients:

- **Multi-turn reasoning replay:** Qwen3.5/3.6-family arches replay assistant
  `reasoning_content` into the next turn (alongside `muse_glimmer`).
- **Tool-argument streaming:** progressive per-token argument deltas are not
  emitted; arguments arrive bulk-at-terminal with the tool-call chunk.
- **`parallel_tool_calls`:** the explicit request field is ignored.
- **Zed temperature:** Zed defaults sampling temperature to **1.0** unless the
  agent profile overrides it.

For Qwen3.5/3.6 agent use, prefer temperature **0.2–0.6** in the profile, and
enable interleaved reasoning on Zed’s OpenAI-compatible capability when the
client exposes that toggle.

## Auto-routing from `hipfire run`

```bash
hipfire serve -d
hipfire run qwen3.5:9b "..."                 # uses HTTP when /health is up
HIPFIRE_LOCAL=1 hipfire run qwen3.5:9b "..." # force one-shot local daemon
```

`run` probes `http://<probe-host>:<port>/health` (500 ms). Probe host maps
`0.0.0.0` / `::` → `127.0.0.1`. If serve is up, `run` POSTs
`/v1/chat/completions` and does **not** spawn a second daemon. If serve is up
but the request fails, `run` exits rather than colliding on the GPU lock.
`--json` / `--no-stream` also force local control.

Model mismatch: serve reloads to the requested model on the chat path (cold
start cost on that first switched request).

## Experimental prefix cache (multi-slot)

`serve.prefix_cache` is **off by default**. On the multi-slot engine
(`serve.multi_slot=true`) it enables cross-session radix reuse of sealed
128-token KV pages plus Qwen hybrid checkpoints. `.mtp` sidecars are
probed as both `foo.mq4v2.mtp` and `foo.mtp` on the slot engine and the
single-slot loader (`hipfire run --spec mtp`). Verified on gfx1101 /
ROCm 10 (`test_serve_prefix_cache --mtp-k 4`): greedy MTP, sampled AR,
and JSON-Schema AR all reuse ≥256 tokens; `reset` forces a cold miss.
Vision+prefix reuse stays off. Non-empty stop sequences, logprobs, and the images+tools combination stay refused on slots; tools alone are supported.
This is not a registry admission.

### Capability advertisement

`/health` carries a `capabilities` object built once at startup from the
SAME resolved config the daemon's slot engine reads, so what is advertised
is what the engine was built with. OpenAI-compatible clients ignore the
extra field; hipfire clients (and deployment tooling) use it for discovery:

```json
{
  "status": "ok",
  "capabilities": {
    "openai_compatible": true,
    "mode": "multi-slot",
    "multi_slot": true,
    "multi_slot_slots": 2,
    "prefix_cache": true,
    "structured_output": true,
    "structured_output_subset": "json-schema-strict-v1",
    "refused_request_fields": ["stop", "logprobs", "response_format:json_object"]
  }
}
```

The standard route advertises the honest absence (`"multi_slot": false`,
`"structured_output": false`) rather than aspirational capabilities.

```bash
HIPFIRE_SERVE_MULTI_SLOT=true HIPFIRE_SERVE_PREFIX_CACHE=true \
  HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES=268435456 \
  hipfire serve 127.0.0.1:11435 <model.hfq>
```

## Production smoke (GPU)

User-facing serve behavior is a **manual GPU** route — not proven by
container builds or no-GPU CI.

| Claim | Route | Owner |
|---|---|---|
| Generic serve semantics (finish reasons, empty/runaway, cache, timing hooks) | `scripts/serve_harness.py --model <path>` | [VALIDATION.md](VALIDATION.md) |
| Optional wrapper (serve + optional Redline/perf arms) | `scripts/gates.sh --model <path>` | same |

### Branch-only: LFM framing harness

**Branch-implemented** at audited `lfm-redline@692a726dde53508cb53de1a74c720e75a7c9f33e`; **absent** from comparison base `origin/beta@202282de8759dfa6963ea5184ad2bf2b9259cef6`. Not a shipped/ref-pinned HTTP contract on this page.

| Claim | Route | Owner |
|---|---|---|
| LFM2.5 thinking / framing smoke | `scripts/serve_harness.py` with an exact `lfm2.5:*` tag | [VALIDATION.md](VALIDATION.md) |

Example (harness spawns its own serve by default on port `11520`):

```bash
python3 scripts/serve_harness.py --model ~/.hipfire/models/<file> --tag qwen3.5:9b --mode battery
```

No-GPU CI (`.github/workflows/no-gpu-ci.yml` → `scripts/no-gpu-ci.sh`) never
certifies GPU serve, model coherence, or throughput.

## Flash-Next exact lane batching (qwen4, arch 16)

Concurrent Qwen3.8 Flash-Next requests can share one trunk step on the exact
**fn-lanes** route: each admitted request owns a lane (its own request state:
QSA K/V, GDN recurrent state, PLE history) over the one resident weight set,
and every committed token and state byte per request is identical to that
request run alone on the singleton route. **Default off, experimental.**

- **Enable:** `serve.vmm_batch = true` (`HIPFIRE_SERVE_VMM_BATCH=1`) **and**
  `serve.continuous_batch_size >= 2` (`HIPFIRE_CONTINUOUS_BATCH_SIZE`); the
  row budget per step is `serve.max_batch_tokens`. The QSA context backend must
  be VMM and the lane store must fit the free device memory; otherwise the
  load logs `[daemon] qwen4 lanes refused: <reason> — fallback to sequential`
  and serves the singleton route exactly as before. With
  `continuous_batch_size >= 2` but `serve.vmm_batch` off the load logs
  `[daemon] qwen4 lanes need serve.vmm_batch — fallback to sequential` and
  advertises `continuous_batch_capable: false`.
- **Kill switch:** `HIPFIRE_SERVE_VMM_BATCH=0` (the compatibility spelling of
  `serve.vmm_batch = false`) stages no lane store and restores today's serial
  behavior.
- **Eligible requests:** greedy decoding, plain text chat only — no tools,
  images, logprobs or penalties. Ineligible requests keep the singleton route.
- **Singleton when alone:** a request that is alone at dispatch runs the
  unchanged singleton route (including native MTP).
- **MTP lanes (default when the MTP head is loaded):** each lane runs its own
  native MTP window per step: a step-synchronous batched draft for every
  drafting lane, then one `(d+1)`-row verify per lane inside the shared lane
  step (the step's row budget counts lanes, not rows). A lane that
  drafts nothing, or that the floor retires, decodes one AR row per step.
  `serve.batch_spec = false` (`HIPFIRE_SERVE_BATCH_SPEC=0`) keeps the AR lanes
  of before (one token per lane per step). There is no per-lane session
  cache, parking or promotion.
- **Per-lane AR floor:** each lane keeps the singleton's MTP floor (probe
  windows, sticky retirement, decayed per-depth costs), priced in AR-row
  units from a measured model of the shared step (an AR row costs its
  proportional share of an all-AR step; a window adds its verify rows'
  marginal cost), so a lane whose drafts do not pay falls back to AR rows.
- **Exact per lane:** every MTP lane window is byte-identical to the one
  singleton window it takes (a batched window at the lane's draft depth, or
  the singleton's AR step), on the lane's own states. Stateful stages (PLE
  conv, GDN core, QSA core) run per lane on the lane's own state; a row-local
  stage runs once over all lanes' rows only where the G0 byte-identity oracle
  admitted it, otherwise per lane.
  On gfx1151 G0 admitted every row-local stage (embedding, HC read/write, the
  GDN/QSA projections, the MoE, final HC read, head, argmax); other cards run
  every stage per lane until their own G0 run.

The loaded ack reports `continuous_batch_route: "fn-lanes"`,
`continuous_batch_history_capable: true` (the typed capability the serve
gateway uses to admit appended multi-turn histories; it never matches route
names),
`continuous_batch_slots`, `continuous_batch_exact: true`,
`continuous_batch_sampling: "greedy_only"`, the speculation state
(`continuous_batch_spec`, `continuous_batch_spec_mode` = `"mtp"` or `"off"`,
`continuous_batch_spec_k` with MTP lanes, `continuous_batch_spec_requested`)
and the lane-store receipt:
`fn_lanes_kv_backend`, `fn_lanes_qsa_format`, `fn_lanes_gdn_format`,
`fn_lanes_ring_rows`, `fn_lanes_max_lanes`, `fn_lanes_row_budget`,
`fn_lanes_mapped_bytes`, `fn_lanes_stage_policy` (the per-stage sharing
policy) and `fn_lanes_stage_evidence` (the G0 receipt id for the production
policy). Each lane's `done` event carries the singleton's MTP counters
(`mtp`, `mtp_windows`, `ar_windows`, `mtp_retired`) plus `mtp_drafts` and
`mtp_accepted`.

## Multi-process notes

- One listener per bind; second serve on the same host/port exits.
- Prefer plain `hipfire stop` when the pidfile is healthy. `hipfire stop
  --force` / `restart` also reap this install's untracked daemons, but never
  another install's or a non-hipfire port holder. Avoid ad-hoc `pkill` unless
  you know the exact process tree.
- `hipfire chat` starts a tracked detached native service if none is healthy;
  stop it explicitly with `hipfire stop` when the session is done.

## Logs and state files

| Path | Role |
|---|---|
| `~/.hipfire/serve.log` | Detached stdout/stderr |
| `~/.hipfire/serve.pid` | Ownership record for stop/ps/restart |
| `~/.hipfire/config.toml` | Sparse typed global configuration |
| `~/.hipfire/models.toml` | Local catalog, aliases, registry identities, and per-model overrides |
| `~/.hipfire/{config,models}.json`, `per_model_config.json` | Legacy migration inputs only |

```bash
tail -f ~/.hipfire/serve.log
```

## Related

- CLI inventory: [CLI.md](CLI.md)
- Chat TUI attach behavior: [CHAT.md](CHAT.md)
- Containers: [CONTAINER.md](CONTAINER.md)
- NixOS service: [NIXOS.md](NIXOS.md)
