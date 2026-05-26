#!/bin/sh
# Mistral-7B-Instruct-v0.2/1P1_1D1_dynamickv：单机 1 个 P（1 卡 TP=1）+ 1 个 D（1 卡 TP=1），共 2 卡。
# 与 run_decode.sh 配套；在原 1P1_1D1 基础上开启 DynamicKV（--additional-config）。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/autodl-tmp/models/Mistral-7B-Instruct-v0.2"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
dp_size=1
dp_ip="127.0.0.1"
dp_port=13395
engine_port=9000

# 1 个 Prefill 进程占用 2 张 NPU 卡（示例：0,1）
visible_devices="${PREFILL_VISIBLE_DEVICES:-0,1}"

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

export VLLM_DP_SIZE=$dp_size
export VLLM_DP_MASTER_IP=$dp_ip
export VLLM_DP_MASTER_PORT=$dp_port
export VLLM_DP_RANK_LOCAL=0
export VLLM_DP_RANK=0
export VLLM_DP_SIZE_LOCAL=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="fork"

# Profiling（配合 analysis/dynkv_profilers/ 解析 prefill.log）
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
export VLLM_DYNKV_PROFILE_PREPARE=0
export VLLM_DYNKV_PROFILE_FORWARD=0
if [ "$dp_size" -gt 1 ]; then
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
else
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=0
fi
run_prefill() {
vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $engine_port \
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
        "kv_port": "20001",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "prefill": { "dp_size": 1, "tp_size": 2 },
            "decode": { "dp_size": 1, "tp_size": 2 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
}

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
run_prefill >> "${LOG_DIR}/prefill.log" 2>&1

