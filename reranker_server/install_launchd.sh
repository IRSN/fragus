#!/usr/bin/env bash
# Installs (or reloads) the reranker LaunchAgents: MPS server + SSH tunnel.
# Both start at login and restart automatically.
#
#   ./install_launchd.sh            # installs / reloads
#   ./install_launchd.sh uninstall  # unloads and removes the agents
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LABELS=(com.cleyrop.reranker.server com.cleyrop.reranker.tunnel)

if [[ "${1:-}" == "uninstall" ]]; then
  for label in "${LABELS[@]}"; do
    launchctl unload "$AGENTS_DIR/$label.plist" 2>/dev/null || true
    rm -f "$AGENTS_DIR/$label.plist"
    echo "removed $label"
  done
  exit 0
fi

mkdir -p "$AGENTS_DIR" "$SERVER_DIR/logs"
chmod +x "$SERVER_DIR/run_server.sh" "$SERVER_DIR/run_tunnel.sh"

for label in "${LABELS[@]}"; do
  sed "s|__SERVER_DIR__|$SERVER_DIR|g" \
    "$SERVER_DIR/launchd/$label.plist" > "$AGENTS_DIR/$label.plist"
  launchctl unload "$AGENTS_DIR/$label.plist" 2>/dev/null || true
  launchctl load "$AGENTS_DIR/$label.plist"
  echo "loaded $label"
done

echo
echo "OK. Checks:"
echo "  curl -s localhost:8001/health"
echo "  tail -f $SERVER_DIR/logs/server.err.log"
echo "  launchctl list | grep reranker"
