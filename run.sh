#!/usr/bin/env bash
# Start the ML-Map backend (FastAPI) and frontend (Vite) dev servers.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY=/home/primus/.pyenv/versions/machine-learning/bin/python

export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"

echo "Starting backend on http://127.0.0.1:8077 ..."
"$VPY" -m uvicorn backend.app:app --host 127.0.0.1 --port 8077 --reload &
BACK=$!

echo "Starting frontend on http://127.0.0.1:5173 ..."
( cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1 --port 5173 ) &
FRONT=$!

trap 'kill $BACK $FRONT 2>/dev/null' EXIT
wait
