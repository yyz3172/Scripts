#!/bin/bash
# 后台启动 PD 负载均衡代理，供 run_sharegpt_sweep.sh 调用
# 用法: $0 <test_name> <batch_size> [pd_dir]
# pd_dir: 含 pd_proxy.sh 的目录，默认 Qwen3-32B/1P2_2D2（相对本脚本所在目录）
TEST_NAME="${1:?Usage: $0 <test_name> <batch_size> [pd_dir]}"
BATCH_SIZE="${2:?Usage: $0 <test_name> <batch_size> [pd_dir]}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PD_DIR="${3:-$SCRIPT_DIR/Qwen3-32B/1P2_2D2}"
PROXY_SCRIPT="$PD_DIR/pd_proxy.sh"

LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}/batch_${BATCH_SIZE}"
mkdir -p "$LOG_DIR"
PID_FILE="/tmp/vllm_proxy.pid"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [proxy] $*"
}

if [ ! -f "$PROXY_SCRIPT" ]; then
    log "ERROR: PROXY_SCRIPT not found: $PROXY_SCRIPT"
    exit 1
fi

log "Starting proxy in background (TEST_NAME=$TEST_NAME, BATCH_SIZE=$BATCH_SIZE, PD_DIR=$PD_DIR)..."
source /root/autodl-tmp/py_venv/vllm2/bin/activate
set -m
nohup bash "$PROXY_SCRIPT" >> "$LOG_DIR/proxy.log" 2>&1 &
echo $! > "$PID_FILE"
log "Proxy started (PID $(cat "$PID_FILE")), log: $LOG_DIR/proxy.log"
