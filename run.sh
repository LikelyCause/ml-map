#!/usr/bin/env bash
# Start the ML-Map backend (FastAPI) and frontend (Vite) dev servers.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter. Override with VPY=/path/to/python ./run.sh. Otherwise try
# the Linux dev box's pyenv, then a macOS pyenv path, then whatever python3 is on
# PATH (e.g. an activated venv on Apple Silicon).
if [ -z "${VPY:-}" ]; then
  for cand in \
    "/home/primus/.pyenv/versions/machine-learning/bin/python" \
    "$HOME/.pyenv/versions/machine-learning/bin/python"; do
    if [ -x "$cand" ]; then VPY="$cand"; break; fi
  done
  VPY="${VPY:-$(command -v python3)}"
fi

# On Apple Silicon (MPS), let ops not yet implemented in Metal fall back to CPU
# instead of raising NotImplementedError. Harmless on CUDA/CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Load nvm if present (skip silently on machines without it, e.g. a fresh Mac).
export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

echo "Backend Python: $VPY"

echo "Starting backend on http://127.0.0.1:8077 ..."
"$VPY" -m uvicorn backend.app:app --host 127.0.0.1 --port 8077 --reload &
BACK=$!

echo "Starting frontend on http://127.0.0.1:5173 ..."
( cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1 --port 5173 ) &
FRONT=$!

trap 'kill $BACK $FRONT 2>/dev/null' EXIT
wait
