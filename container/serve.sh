#!/bin/sh
set -eu

MODEL="${ORNITH_MODEL:-/models/ornith-1.5-9b.mq4}"
BIND="${ORNITH_BIND:-0.0.0.0:8420}"
KV_MODE="${ORNITH_KV_MODE:-fwht4}"

if [ ! -f "$MODEL" ]; then
    echo "ornith-serve: model not found: $MODEL" >&2
    exit 1
fi

# NOTE: /v1/models scans $HIPFIRE_HOME/models and filters by the CANONICAL
# file name's extension (.mq4/.hfq/...). For the served model to appear
# there, bind-mount this same file under a filter-passing name into
# ${HIPFIRE_HOME}/models/ at `docker run` time; symlinks/hardlinks cannot
# work (name canonicalizes away / cross-device).

exec hipfire serve "$MODEL" "$BIND" --kv-mode "$KV_MODE"
