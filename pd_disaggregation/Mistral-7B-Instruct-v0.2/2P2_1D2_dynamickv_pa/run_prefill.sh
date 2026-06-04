#!/bin/sh
# Mistral-7B-Instruct-v0.2/2P2_1D2_dynamickv_pa：2 个 P（各 2 卡 TP=2）+ 1 个 D（2 卡 TP=2），共 6 卡。
# 单脚本通过入参区分 Prefill 实例；与 run_decode.sh、pd_proxy.py 配套。
#
# 用法：
#   bash run_prefill.sh          # 后台启动 P0、P1 并 wait（供 sweep / start_prefill 调用）
#   bash run_prefill.sh 0        # 仅 P0（卡 0,1，端口 9000）
#   bash run_prefill.sh 1        # 仅 P1（卡 2,3，端口 9001）
# 日志：${LOG_DIR}/prefill_<rank>.log（LOG_DIR 默认 .）
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/autodl-tmp/models/Mistral-7B-Instruct-v0.2"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"

run_one() {
    prefill_rank=$1
    case $prefill_rank in
        0)
            engine_port=9000
            visible_devices="0,1"
            dp_port=13395
            kv_port="20001"
            engine_id="0"
            ;;
        1)
            engine_port=9001
            visible_devices="2,3"
            dp_port=13396
            kv_port="20003"
            engine_id="1"
            ;;
        *)
            echo "Usage: $0 [0|1]" >&2
            exit 1
            ;;
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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export HCCL_BUFFSIZE=256

    export VLLM_DP_SIZE=1
    export VLLM_DP_MASTER_IP="127.0.0.1"
    export VLLM_DP_MASTER_PORT=$dp_port
    export VLLM_DP_RANK_LOCAL=0
    export VLLM_DP_RANK=0
    export VLLM_DP_SIZE_LOCAL=1

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_WORKER_MULTIPROC_METHOD="fork"

    # DynamicKV 日志 profiling（配合 analysis/dynkv_profilers/ 解析 prefill_<rank>.log）
    export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
    export VLLM_DYNKV_PROFILE_PREPARE=0
    export VLLM_DYNKV_PROFILE_FORWARD=0
    export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=0

    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"

    # Ascend PyTorch Profiler（--profiler-config，见 service_profiling_guide）
    # 启停：curl -X POST http://${LOCAL_IP}:${engine_port}/start_profile|stop_profile
    # 分析：from torch_npu.profiler.profiler import analyse; analyse("${PROFILER_DIR}/localhost.*_ascend_pt/")
    ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-1}"
    profiler_config_json=""
    if [ "$ENABLE_TORCH_PROFILER" = "1" ]; then
      PROFILER_DIR="${PROFILER_DIR:-${LOG_DIR}/vllm_profile/prefill_${prefill_rank}}"
      mkdir -p "$PROFILER_DIR"
      export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
      # torch_profiler_record_shapes: 是否记录 tensor shape（true 增大 trace 体积）
      # torch_profiler_with_memory: 是否记录内存占用（true 便于分析显存，采集时略增开销）
      profiler_config_json='{
        "profiler": "torch",
        "torch_profiler_dir": "'"${PROFILER_DIR}"'",
        "torch_profiler_with_stack": true,
        "torch_profiler_record_shapes": false,
        "torch_profiler_with_memory": true,
        "ignore_frontend": true
      }'
    fi

    exec >> "${LOG_DIR}/prefill_${prefill_rank}.log" 2>&1

    vllm serve "$model_path" \
        ${profiler_config_json:+--profiler-config} \
        ${profiler_config_json:+"$profiler_config_json"} \
        --host 0.0.0.0 \
        --port "$engine_port" \
        --enable-prefix-caching \
        --tensor-parallel-size 2 \
        --seed 1024 \
        --served-model-name mistral_7b_instruct_v0_2 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 2048 \
        --max-num-seqs 256 \
        --long-prefill-token-threshold 1024 \
        --enable-auto-tool-choice \
        --tool-call-parser mistral \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --additional-config \
        '{
            "pa_shape_list": [1, 2, 4, 8, 16, 24, 32, 48, 64, 72, 80, 96, 128, 256],
            "dynamic_kv": {
                "enabled": true,
                "impl": "offload",
                "model_types": ["mistral"],
                "validation_mode": "none",
                "window_size": 256,
                "prompt_kv_len_budget": 2048,
                "radio_max": 10.0,
                "min_rewrite_delta": 1,
                "pooling": "avgpool",
                "kernel_size": 7,
                "uniform_kv_budget": "fixed_base"
            }
        }' \
        --kv-transfer-config \
        '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_buffer_device": "npu",
            "kv_role": "kv_producer",
            "kv_parallel_size": "1",
            "kv_port": "'"${kv_port}"'",
            "engine_id": "'"${engine_id}"'",
            "kv_connector_extra_config": {
                "prefill": { "dp_size": 1, "tp_size": 2 },
                "decode": { "dp_size": 1, "tp_size": 2 }
            },
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
        }'
}

if [ $# -eq 0 ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"
    bash "$SCRIPT_DIR/run_prefill.sh" 0 &
    bash "$SCRIPT_DIR/run_prefill.sh" 1 &
    wait
else
    run_one "$1"
fi
