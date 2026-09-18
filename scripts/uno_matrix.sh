#!/usr/bin/env bash
# UNO noise-law + tree-verify measurement matrix.
# Run inside `gpu-coord exec`. Holding the lock means no peer session is
# active, so a residual exl3 server from a FINISHED session is stopped
# (container preserved, restored at the end) to free its VRAM.
set -u
IMG=local/rocm-base:10.0.0
DOC="docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) \
  --group-add $(getent group video | cut -d: -f3) \
  -v /home/ghazni/github/hipfire/target:/t:ro \
  -v /home/ghazni/.hipfire/models:/models:ro \
  -v /home/ghazni/uno-adapter:/adapter:ro \
  -e HIPFIRE_KERNEL_CACHE=/tmp/kcache"
PROBE=/t/release/examples/uno_perf_probe
NEED_MB=8192

vram_free_mb() {
  local t u
  t=$(cat /sys/class/drm/card0/device/mem_info_vram_total)
  u=$(cat /sys/class/drm/card0/device/mem_info_vram_used)
  echo $(( (t - u) / 1048576 ))
}

FREE=$(vram_free_mb)
echo "preflight: VRAM free ${FREE} MB (need ${NEED_MB})"
if [ "$FREE" -lt "$NEED_MB" ]; then
  if docker ps --format '{{.Names}}' | grep -q '^qwen38-27b-exl3$'; then
    echo "residual exl3 server detected (its session released the lock) — stopping it"
    docker stop -t 60 qwen38-27b-exl3
  fi
  for i in $(seq 1 12); do
    FREE=$(vram_free_mb)
    [ "$FREE" -ge "$NEED_MB" ] && break
    sleep 5
  done
  echo "after teardown: VRAM free ${FREE} MB"
  if [ "$FREE" -lt "$NEED_MB" ]; then
    echo "FATAL: VRAM still held by an unknown process — not touching it. Holder:"
    rocm_smipids() { :; }
    exit 42
  fi
  RESTORE_EXL3=1
fi

run() {
  echo "===== $* ====="
  DOE=()
  for kv in "$@"; do DOE+=(-e "$kv"); done
  docker run --rm --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v /home/ghazni/github/hipfire/target:/t:ro \
    -v /home/ghazni/.hipfire/models:/models:ro \
    -v /home/ghazni/uno-adapter:/adapter:ro \
    -e HIPFIRE_KERNEL_CACHE=/tmp/kcache \
    "${DOE[@]}" $IMG $PROBE /models/k2-horizon-7b.mq4 /adapter 2>&1 \
    | grep -E "^\[AR\]|^\[UNO|^\[phase|Error|error|panicked"
}

run UNO_IDENTITY=1 HIPFIRE_UNO_BLOCK=4
run HIPFIRE_UNO_BLOCK=8
run UNO_IDENTITY=1 HIPFIRE_UNO_BLOCK=8 HIPFIRE_UNO_TREE=32 HIPFIRE_UNO_TREE_K=12
run HIPFIRE_UNO_BLOCK=8 HIPFIRE_UNO_TREE=48 HIPFIRE_UNO_TREE_K=16
run HIPFIRE_UNO_BLOCK=8 HIPFIRE_UNO_TREE=64 HIPFIRE_UNO_TREE_K=16
echo "===== MATRIX COMPLETE ====="

if [ "${RESTORE_EXL3:-0}" = "1" ]; then
  echo "restoring qwen38-27b-exl3"
  docker start qwen38-27b-exl3
fi
