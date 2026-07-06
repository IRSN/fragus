#!/usr/bin/env bash
# Runs reranker_server.py on the Apple Silicon GPU (MPS), throttled so it does
# not saturate the Mac, and prevents sleep while it runs.
#
#   - caffeinate -i : no idle sleep while the server is running
#                     (does NOT cover a closed lid on battery → keep it open
#                      or stay on AC power with an external display).
#   - nice -n 5 + OMP/MKL=4 : ~4 CPU threads at low priority → leaves the
#                     cores free for your own usage. The bulk of the compute is on the GPU.
#
# If the default port (RERANKER_PORT or 8001) is taken, a free port is
# automatically found and recorded in logs/server.port so that run_tunnel.sh
# can use it.
#
# All variables can be overridden from the environment.
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY="${VENV_PY:-../.venv/bin/python}"
PREFERRED_PORT="${RERANKER_PORT:-8001}"
mkdir -p logs

# Find a free port starting from the preferred port
PORT=$(python3 - "$PREFERRED_PORT" <<'EOF'
import socket, sys
start = int(sys.argv[1])
for p in range(start, start + 20):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', p)) != 0:
            print(p); break
EOF
)

if [[ "$PORT" != "$PREFERRED_PORT" ]]; then
  echo "⚠  Port $PREFERRED_PORT is taken → using port $PORT" >&2
fi
echo "$PORT" > logs/server.port
echo "Port: $PORT (see logs/server.port)" >&2

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RERANKER_BATCH="${RERANKER_BATCH:-32}"

exec caffeinate -i nice -n 5 "$VENV_PY" reranker_server.py --port "$PORT" "$@"
