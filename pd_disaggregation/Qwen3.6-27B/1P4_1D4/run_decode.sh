#!/bin/sh
# Qwen3.6-27B/1P4_1D4：单脚本启动 1 个 Decode 实例。与 run_prefill.sh 配套。
# 用法：
#   bash run_decode.sh          # 后台启动 D0 并 wait（供 sweep / start_decode 调用）
#   bash run_decode.sh 0        # 仅启动第 1 个 D（卡 4,5,6,7，端口 9010）
# 日志：vllm 写入 ${LOG_DIR}/decode.log（LOG_DIR 默认 .）；非法入参的提示仍打 stderr。
# ========== 配置区（须与 run_prefill.sh 一致）==========
# NIC_NAME / LOCAL_IP 由 PdServiceCtl 注入；单独跑脚本时请 export，默认值与 pd_service_ctl 中常量一致。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/yyz/models/Qwen3.6-27B"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
dp_size=1
dp_ip="127.0.0.1"
dp_port=13495
# 单个 D：engine_port 9010，visible_devices 4,5,6,7
# ==========================================

run_one() {
    dp_rank=$1
    case $dp_rank in
        0) engine_port=9010; visible_devices="4,5,6,7" ;;
        *) echo "Usage: $0 [0]" >&2; exit 1 ;;
    esac

    export ASCEND_RT_VISIBLE_DEVICES=$visible_devices

    if [ -f /usr/lib/aarch64-linux-gnu/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
    elif [ -f /usr/lib64/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib64/libstdc++.so.6
    elif [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    fi
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export LD_LIBRARY_PATH="${python_lib_path:+$python_lib_path:}${transfer_engine_lib_path:+$transfer_engine_lib_path:}/usr/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib:${LD_LIBRARY_PATH:-}"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    # Qwen3.5/3.6 混合架构 + TP 多进程：须 OMP=1，fork+多线程会触发 Invalid thread pool
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export HCCL_BUFFSIZE=256

    export VLLM_DP_SIZE=$dp_size
    export VLLM_DP_MASTER_IP=$dp_ip
    export VLLM_DP_MASTER_PORT=$dp_port
    export VLLM_DP_RANK_LOCAL=0
    export VLLM_DP_RANK=$dp_rank
    export VLLM_DP_SIZE_LOCAL=1

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"
    exec >> "${LOG_DIR}/decode.log" 2>&1

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $engine_port \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --tensor-parallel-size 4 \
        --nnodes 1 \
        --seed 1024 \
        --served-model-name qwen3_6_27b \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser llama3_json \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --additional-config '{"recompute_scheduler_enable": true}' \
        --kv-transfer-config \
        '{
            "kv_connector": "MooncakeLayerwiseConnector",
            "kv_buffer_device": "npu",
            "kv_role": "kv_consumer",
            "kv_parallel_size": "1",
            "kv_port": "20002",
            "engine_id": "1",
            "kv_connector_extra_config": {
                "prefill": { "dp_size": 1, "tp_size": 4 },
                "decode": { "dp_size": 1, "tp_size": 4 }
            },
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector"
        }'
}

if [ $# -eq 0 ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"
    bash "$SCRIPT_DIR/run_decode.sh" 0 &
    wait
else
    run_one "$1"
fi
