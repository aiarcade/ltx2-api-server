#!/usr/bin/env bash
# Stop the LTX-2.3 inference server

PID_FILE="/tmp/ltx_server.pid"
PORT="${LTX_PORT:-8000}"

stop_pid() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping server (PID $pid) …"
        kill "$pid"
        for i in $(seq 1 10); do
            sleep 1
            kill -0 "$pid" 2>/dev/null || { echo "Server stopped."; return 0; }
        done
        echo "Server did not stop gracefully, force-killing …"
        kill -9 "$pid" 2>/dev/null || true
    fi
}

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    stop_pid "$PID"
    rm -f "$PID_FILE"
else
    # Fallback: find by process name
    PIDS=$(pgrep -f "uvicorn server:app" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        echo "PID file not found; killing by process name …"
        echo "$PIDS" | while read -r pid; do stop_pid "$pid"; done
    else
        echo "No running server found."
    fi
fi

# Verify port is free
if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "WARNING: Something is still listening on port $PORT."
else
    echo "Port $PORT is free."
fi
