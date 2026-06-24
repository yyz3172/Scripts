#!/bin/sh
# Mistral-7B-Instruct-v0.2/1P1_1D1_dynamickv：单机 1 个 Decode（1 卡 TP=1）。与 run_prefill.sh 配套。
# 在原 1P1_1D1 基础上开启 DynamicKV（--additional-config）。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/yyz/models/Mistral-7B-Instruct-v0.2"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"

dp_size=1
dp_rank=0
dp_ip="127.0.0.1"
dp_rpc_port=13495

engine_port=9010
visible_devices="${DECODE_VISIBLE_DEVICES:-1}"

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
export HCCL_BUFFSIZE=1024

export VLLM_DP_SIZE=$dp_size
export VLLM_DP_MASTER_IP=$dp_ip
export VLLM_DP_MASTER_PORT=$dp_rpc_port
export VLLM_DP_RANK_LOCAL=0
export VLLM_DP_RANK=$dp_rank
export VLLM_DP_SIZE_LOCAL=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="fork"
if [ "$dp_size" -gt 1 ]; then
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
else
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=0
fi
LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
exec >> "${LOG_DIR}/decode.log" 2>&1

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $engine_port \
    --tensor-parallel-size 1 \
    --nnodes 1 \
    --seed 1024 \
    --served-model-name mistral_7b_instruct_v0_2 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --additional-config \
    '{
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
            "kernel_size": 7
        }
    }' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": "1",
        "kv_port": "20002",
        "engine_id": "1",
        "kv_connector_extra_config": {
            "prefill": { "dp_size": 1, "tp_size": 1 },
            "decode": { "dp_size": 1, "tp_size": 1 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'

