#!/bin/bash
# 停止 PD 负载均衡代理（由 run_sharegpt_sweep 拉起的 pd_proxy）
PID_FILE="/tmp/vllm_proxy.pid"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [stop_proxy] $*"
}

if [ ! -f "$PID_FILE" ]; then
    log "No PID file at $PID_FILE, proxy may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    log "Process $PID is not running."
    rm -f "$PID_FILE"
    exit 0
fi

log "Killing proxy (PID $PID)..."
kill -9 -"$PID" 2>/dev/null
kill -9 "$PID" 2>/dev/null
i=0
while kill -0 "$PID" 2>/dev/null; do
    [ $i -ge 10 ] && break
    sleep 1
    i=$((i + 1))
done
rm -f "$PID_FILE"
log "Proxy stopped."
