#!/usr/bin/env bash
# Pool Vision AI — run all 3 services with one command.
# First run installs deps automatically. Ctrl-C stops everything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_PORT="${AI_PORT:-8000}"
BACKEND_PORT="${PORT:-8090}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

pids=()
cleanup() {
  echo ""
  echo "[dev] stopping…"
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 1. AI service (Python + OpenCV) ---------------------------------------
echo "[dev] ai-service: setup…"
cd "$ROOT/ai-service"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
echo "[dev] ai-service: starting on :$AI_PORT"
.venv/bin/uvicorn app.main:app --port "$AI_PORT" --log-level warning &
pids+=($!)

# --- 2. Backend proxy (Go + Gin) -------------------------------------------
echo "[dev] backend: starting on :$BACKEND_PORT"
cd "$ROOT/backend"
( PORT="$BACKEND_PORT" AI_SERVICE_URL="http://localhost:$AI_PORT" go run . ) &
pids+=($!)

# --- 3. Frontend (Vite + React) --------------------------------------------
echo "[dev] frontend: setup…"
cd "$ROOT/frontend"
[ -d node_modules ] || npm install --silent
echo "[dev] frontend: starting on :$FRONTEND_PORT"
# --host binds 0.0.0.0 so phones on the LAN can reach it. No VITE_API_BASE:
# the client derives the backend host from the page URL (see src/api.ts).
npm run dev -- --host --port "$FRONTEND_PORT" &
pids+=($!)

echo ""
echo "[dev] all services up:"
echo "       UI       http://localhost:$FRONTEND_PORT"
echo "       backend  http://localhost:$BACKEND_PORT"
echo "       ai       http://localhost:$AI_PORT"
echo "[dev] press Ctrl-C to stop"
wait
