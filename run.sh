#!/usr/bin/env bash
# Start the Swath backend (FastAPI) and frontend (Vite) dev servers.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python interpreter for the backend. Override with: VPY=/path/to/python ./run.sh
# Otherwise prefer an in-repo .venv, then an active virtualenv, then python3 on PATH.
if [ -z "${VPY:-}" ]; then
  if   [ -x "$ROOT/.venv/bin/python" ]; then VPY="$ROOT/.venv/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then VPY="$VIRTUAL_ENV/bin/python"
  else VPY="$(command -v python3)"; fi
fi

# Fail clearly if the backend deps aren't installed for the chosen interpreter.
if ! "$VPY" -c 'import fastapi, uvicorn' 2>/dev/null; then
  echo "Backend deps not found for: $VPY"
  echo "  See the README 'Setup' section (create a venv + pip install -r backend/requirements.txt),"
  echo "  then re-run, or pass VPY=/path/to/python ./run.sh"
  exit 1
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
