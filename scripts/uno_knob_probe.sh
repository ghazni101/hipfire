#!/usr/bin/env bash
# Sweep HIPFIRE_UNO_TREE at block 16 — the last allowed-knob region not
# covered by the record (tree > 16 was only ever run at block 8).
#
# Reports the same decode tok/s and tau the A/B uses, so the ratio is
# comparable with results/k2-uno-vs-ar/ (AR baseline 67.3 tok/s on this build).
#
# Usage: scripts/uno_knob_probe.sh OUTDIR
set -u

OUTDIR="${1:-results/knob-tree}"
IMG=rocm-dev:10.0.0-rpc
WT=/home/ghazni/github/hipfire/k2-uno
MODEL=/models/hipfire/IFM/K2-Horizon-7B-MQ4/k2-horizon-7b.mq4

mkdir -p "$OUTDIR"

for TREE in 16 32 64; do
  echo "===== block 16, tree $TREE =====" >&2
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
    -e HIPFIRE_UNO_BLOCK=16 \
    -e HIPFIRE_UNO_TREE="$TREE" \
    $IMG /t/release/hipfire bench "$MODEL" \
      --runs 5 --warmups 3 --max-tokens 128 --kv-mode q8 \
      --backend noslots --workload stateless --reasoning-on \
      --prompt-file /prompt/k2_uno_qa_fixed.txt --json \
      --model-draft /adapter \
    > "$OUTDIR/tree$TREE.json" 2> "$OUTDIR/tree$TREE.log"
done

python3 - "$OUTDIR" <<'PY'
import json, os, sys
out = sys.argv[1]
print(f"{'tree':>5} {'median':>7} {'runs':>26} {'tau':>6} {'ratio':>7}")
for t in (16, 32, 64):
    p = os.path.join(out, f"tree{t}.json")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"{t:>5} {'(no json)':>7}")
        continue
    j = json.load(open(p))
    d = j["samples"]["decode"]
    tau = (j.get("tau") or {}).get("median")
    med = sorted(d)[len(d) // 2]
    print(f"{t:>5} {med:>7.1f} {str([round(x,1) for x in d]):>26} "
          f"{tau if tau is None else round(tau,2):>6} {med/67.3:>7.3f}")
PY
