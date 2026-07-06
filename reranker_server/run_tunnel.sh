#!/usr/bin/env bash
# Reverse SSH tunnel Mac -> VM with automatic reconnection (autossh).
# The reranker port on the VM side then points to the Mac's local reranker_server.
#
#   - autossh -M 0 + ServerAlive* : detects a dead connection in ~45 s and restarts.
#   - ExitOnForwardFailure=yes     : if the port is still taken on the VM side
#                                    (ghost socket after a disconnect), fails fast →
#                                    autossh retries cleanly.
#   - AUTOSSH_GATETIME=0           : retries even if the 1st connection fails early.
#
# The port is read from logs/server.port (written by run_server.sh); RERANKER_PORT
# or 8001 as fallback. The host is an alias from ~/.ssh/config.
set -euo pipefail
cd "$(dirname "$0")"

VM_HOST="${RERANKER_TUNNEL_HOST:-cleyrop-asnr-secnum-devspace}"

# Read the port from the file left by run_server.sh, otherwise fall back
if [[ -f logs/server.port ]]; then
  PORT=$(cat logs/server.port)
else
  PORT="${RERANKER_PORT:-8001}"
  echo "⚠  logs/server.port not found, using port $PORT" >&2
fi

echo "Tunnel: localhost:${PORT} → ${VM_HOST}:${PORT}" >&2
export AUTOSSH_GATETIME=0

exec autossh -M 0 -N -R "${PORT}:localhost:${PORT}" "$VM_HOST" \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o TCPKeepAlive=yes
