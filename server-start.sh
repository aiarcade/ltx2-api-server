#!/usr/bin/env bash
# Start the LTX-2.3 inference server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/uvicorn"
LOG="/tmp/ltx_server.log"
PID_FILE="/tmp/ltx_server.pid"
PORT="${LTX_PORT:-8000}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server is already running (PID $(cat "$PID_FILE"))."
    echo "  Health: http://localhost:$PORT/health"
    exit 0
fi

echo "Starting LTX-2.3 server on port $PORT …"
cd "$SCRIPT_DIR"

# Ensure vendored packages are installed (idempotent)
PYTHON="$SCRIPT_DIR/venv/bin/python"
if ! "$PYTHON" -c "import ltx_core" 2>/dev/null; then
    echo "Installing vendored ltx-core and ltx-pipelines …"
    "$SCRIPT_DIR/venv/bin/pip" install --no-deps -q -e vendor/ltx-core
    "$SCRIPT_DIR/venv/bin/pip" install --no-deps -q -e vendor/ltx-pipelines
fi

# Load .env if present (sets LTX_API_KEY etc.)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

"$VENV" server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "PID $SERVER_PID — waiting for startup …"

# Wait up to 10s for server to become healthy
for i in $(seq 1 10); do
    sleep 1
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is up."
        echo "  Health:  http://localhost:$PORT/health"
        echo "  Docs:    http://localhost:$PORT/docs"
        echo "  Log:     $LOG"
        exit 0
    fi
done

echo "ERROR: Server did not respond after 10s. Check $LOG"
exit 1
