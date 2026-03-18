#!/bin/bash
set -euo pipefail

# PD 分离模式：每轮先起 Prefill，再起 Decode，可选起代理，压测，最后按序停止
# 用法: $0 <test_name>
TEST_NAME="${1:?Usage: $0 <test_name>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 配置项 ───────────────────────────────────────────────────────────────────
PD_MODE="Qwen3-32B/1P2_2D2"
PD_DIR="$SCRIPT_DIR/$PD_MODE"
BATCH_SIZES=(60 80 120 200)
VLLM_PORT=9010
VLLM_PROXY_PORT=8000
VLLM_PREFILL_PORT=9000
VLLM_READY_TIMEOUT=300
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}"
SWEEP_LOG="$LOG_DIR/sweep.log"
export LOG_DIR

mkdir -p "$LOG_DIR"

# 是否使用代理：pd_dir 下存在 pd_proxy.sh 则启动代理，压测连代理端口
USE_PROXY=false
if [ -f "$PD_DIR/pd_proxy.sh" ]; then
    USE_PROXY=true
fi

# ── 日志函数 ────────────────────────────────────────────────────────────────
log() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') [sweep] $*"
    echo "$msg" | tee -a "$SWEEP_LOG"
}

# ── 等待某端口健康接口就绪 ───────────────────────────────────────────────────
wait_for_port() {
    local port=$1
    local name=${2:-$port}
    log "Waiting for $name on port $port (timeout ${VLLM_READY_TIMEOUT}s)..."
    local elapsed=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if [ "$elapsed" -ge "$VLLM_READY_TIMEOUT" ]; then
            log "ERROR: $name did not become ready within ${VLLM_READY_TIMEOUT}s."
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "$name is ready (waited ${elapsed}s)."
}

# ── 停止代理（若在运行）────────────────────────────────────────────────────
stop_proxy_if_running() {
    if [ "$USE_PROXY" = true ]; then
        bash "$SCRIPT_DIR/stop_proxy.sh" || true
    fi
}

# ── 停止 PD 双进程（先 Decode 后 Prefill）───────────────────────────────────
stop_vllm() {
    log "Stopping vllm (PD: decode then prefill)..."
    bash "$SCRIPT_DIR/stop_decode.sh" || true
    bash "$SCRIPT_DIR/stop_prefill.sh" || true
}

# 异常退出时确保进程被停止
trap 'log "Interrupted, cleaning up..."; stop_proxy_if_running; stop_vllm; exit 1' INT TERM

# ── 主循环 ──────────────────────────────────────────────────────────────────
log "Sweep start (PD mode): TEST_NAME=$TEST_NAME  PD_DIR=$PD_DIR  BATCH_SIZES=${BATCH_SIZES[*]}  USE_PROXY=$USE_PROXY"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    log "========== Round: BATCH_SIZE=$BATCH_SIZE =========="

    # 1. 启动 Prefill
    log "Starting prefill (TEST_NAME=${TEST_NAME}_bs${BATCH_SIZE})..."
    bash "$SCRIPT_DIR/start_prefill.sh" "${TEST_NAME}" "$BATCH_SIZE" "$PD_DIR"

    # 2. 等待 Prefill 就绪
    if ! wait_for_port "$VLLM_PREFILL_PORT" "prefill"; then
        log "ERROR: skipping BATCH_SIZE=$BATCH_SIZE due to prefill startup failure."
        stop_vllm
        continue
    fi

    # 3. 启动 Decode
    log "Starting decode (TEST_NAME=${TEST_NAME}_bs${BATCH_SIZE})..."
    bash "$SCRIPT_DIR/start_decode.sh" "${TEST_NAME}" "$BATCH_SIZE" "$PD_DIR"

    # 4. 等待 Decode 就绪（直连端口）
    if ! wait_for_port "$VLLM_PORT" "decode"; then
        log "ERROR: skipping BATCH_SIZE=$BATCH_SIZE due to decode startup failure."
        stop_vllm
        continue
    fi

    # 5. 若启用代理：后台启动代理，sleep 10s 后压测连代理端口
    if [ "$USE_PROXY" = true ]; then
        log "Starting proxy (port $VLLM_PROXY_PORT)..."
        bash "$SCRIPT_DIR/start_proxy.sh" "${TEST_NAME}" "$BATCH_SIZE" "$PD_DIR"
        sleep 10
        BENCH_PORT=$VLLM_PROXY_PORT
    else
        BENCH_PORT=$VLLM_PORT
    fi

    # 6. 运行压测（连 BENCH_PORT：代理或直连 Decode）
    log "Running sharegpt benchmark (BATCH_SIZE=$BATCH_SIZE, port=$BENCH_PORT)..."
    if bash "$SCRIPT_DIR/sharegpt.sh" "$TEST_NAME" "$BATCH_SIZE" "$BENCH_PORT" "$PD_MODE"; then
        log "Benchmark finished: BATCH_SIZE=$BATCH_SIZE"
    else
        log "WARNING: benchmark exited with error for BATCH_SIZE=$BATCH_SIZE, continuing..."
    fi

    # 7. 先停代理，再停 PD
    stop_proxy_if_running
    stop_vllm
    log "vllm (PD) stopped. Waiting 5s before next round..."
    sleep 5
done

log "All rounds completed. Logs: $LOG_DIR"
