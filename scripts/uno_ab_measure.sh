#!/usr/bin/env bash
# K2-Horizon Uno vs AR decode A/B on the current build.
#
# One fixed prompt file, greedy, Q8 KV, 3 warmups + 5 timed runs, a fresh
# process per side, and one sampling law on both sides
# (HIPFIRE_BENCH_REPEAT_PENALTY=1.0 — the benchmark's hardcoded 1.1 puts AR and
# the speculative verifier on different distributions; see
# ../K2-UNO-THROUGHPUT.md § "Token identity").
#
# Usage: scripts/uno_ab_measure.sh [OUTDIR]
set -u

OUTDIR="${1:-results/k2-uno-vs-ar}"
IMG=rocm-dev:10.0.0-rpc
MODEL=/models/hipfire/IFM/K2-Horizon-7B-MQ4/k2-horizon-7b.mq4
ADAPTER=/adapter
PROMPT_FILE=/prompt/k2_uno_qa_fixed.txt
WT=/home/ghazni/github/hipfire/k2-uno

mkdir -p "$OUTDIR"

common() {
  echo -n "--runs 5 --warmups 3 --max-tokens 128 --kv-mode q8 \
--backend noslots --workload stateless --reasoning-on \
--prompt-file $PROMPT_FILE --json"
}

side() { # $1=label  $2=extra bench args (may be empty)
  local label="$1" extra="$2"
  echo "===== $label (UNO_BLOCK=${UNO_BLOCK:-4}) =====" >&2
  printf '{"uno_block": %s, "uno_tree": "%s", "uno_tree_k": "%s", "uno_noise": "%s", "repeat_penalty": "1.0", "graph_ctx": "0"}\n' \
    "${UNO_BLOCK:-4}" "${HIPFIRE_UNO_TREE:-0}" "${HIPFIRE_UNO_TREE_K:-default}" "${HIPFIRE_UNO_NOISE:-random_uniform}" > "$OUTDIR/config.$label.json"
  # shellcheck disable=SC2086
  docker run --rm --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v "$WT/target:/t:ro" \
    -v /home/ghazni/models:/models:ro \
    -v /home/ghazni/models/RAW/IFM/K2-Horizon-7B-Uno/source:/adapter:ro \
    -v "$WT/benchmarks/prompts:/prompt:ro" \
    -e HIPFIRE_KERNEL_CACHE=/tmp/kcache \
    -e HIPFIRE_DAEMON_BIN=/t/release/daemon \
    -e HIPFIRE_BENCH_REPEAT_PENALTY=1.0 \
    -e HIPFIRE_UNO_GRAPH_CTX=0 \
    -e HIPFIRE_UNO_BLOCK="${UNO_BLOCK:-4}" \
    $IMG /t/release/hipfire bench "$MODEL" $(common) $extra
}

side "AR" "" > "$OUTDIR/ar.json" 2> "$OUTDIR/ar.log"
side "UNO" "--model-draft $ADAPTER" > "$OUTDIR/uno.json" 2> "$OUTDIR/uno.log"

python3 - "$OUTDIR" <<'PY'
import json, statistics as st, sys, hashlib, os
out = sys.argv[1]
def load(p):
    with open(os.path.join(out, p)) as f:
        return json.load(f)
def summary(j):
    runs = j.get("runs") or j.get("samples") or []
    rs = [r.get("decode_tok_s", r.get("tok_s")) for r in runs]
    return rs
for side in ("ar", "uno"):
    j = load(side + ".json")
    keys = sorted(j.keys())
    print(f"--- {side}: top-level keys {keys}")
    print(json.dumps(j, indent=1)[:1500])
PY
echo "RAW JSON -> $OUTDIR/{ar,uno}.json ; logs -> $OUTDIR/{ar,uno}.log" >&2
