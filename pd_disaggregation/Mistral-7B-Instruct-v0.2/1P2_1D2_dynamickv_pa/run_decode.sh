#!/bin/sh
# Mistral-7B-Instruct-v0.2/1P2_1D2_dynamickv：单机 1 个 Decode（2 卡 TP=2）。与 run_prefill.sh 配套。
# 在原 1P1_1D1 基础上开启 DynamicKV（--additional-config）。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/autodl-tmp/models/Mistral-7B-Instruct-v0.2"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"

dp_size=1
dp_rank=0
dp_ip="127.0.0.1"
dp_rpc_port=13495

engine_port=9010
visible_devices="${DECODE_VISIBLE_DEVICES:-2,3}"

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

# Profiling（配合 analysis/dynkv_profilers/ 解析 decode.log）
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
export VLLM_DYNKV_PROFILE_PREPARE=0
export VLLM_DYNKV_PROFILE_FORWARD=0

# --- Ascend PyTorch Profiler → MindStudio Insight（默认 0；设为 1 开启）---
# Graph 路径（FULL_DECODE_ONLY + PA）；与 pia（enforce-eager）分开落盘目录。
# INSIGHT_PROFILER_ENABLE=1 时：设置 VLLM_TORCH_PROFILER_DIR 并挂载 /start_profile API
INSIGHT_PROFILER_ENABLE="${INSIGHT_PROFILER_ENABLE:-0}"
PROFILER_CONFIG_ARG=""
PROFILER_CONFIG_VAL=""
if [ "$INSIGHT_PROFILER_ENABLE" = "1" ]; then
  PROFILER_DIR="${PROFILER_DIR:-/root/autodl-tmp/yyz/vllm_insight_decode_pa}"
  mkdir -p "$PROFILER_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILER_DIR"
  export VLLM_TORCH_PROFILER_WITH_STACK=0
  export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
  export VLLM_ASCEND_TORCH_PROFILER_EXPORT_TYPE=db
  export VLLM_ASCEND_TORCH_PROFILER_LEVEL=1
  # 勿在线 analyse（daemon 易失败）；落盘后再用 Insight 导入或 analyse()
  export VLLM_ASCEND_TORCH_PROFILER_CAPTURE_EXPORT=text
  export VLLM_ASCEND_TORCH_PROFILER_ANALYSE_ONLINE=0
  export VLLM_ASCEND_TORCH_PROFILER_ACTIVE_STEPS=512
  export VLLM_RPC_TIMEOUT=1800000
  PROFILER_CONFIG_ARG="--profiler-config"
  PROFILER_CONFIG_VAL="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\"}"
fi

if [ "$dp_size" -gt 1 ]; then
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
else
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=0
fi
LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
exec >> "${LOG_DIR}/decode.log" 2>&1

# shellcheck disable=SC2086
vllm serve "$model_path" \
    $PROFILER_CONFIG_ARG $PROFILER_CONFIG_VAL \
    --host 0.0.0.0 \
    --port $engine_port \
    --tensor-parallel-size 2 \
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
    --compilation-config \
    '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 72, 80, 96, 128, 256]}' \
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
            "prefill": { "dp_size": 1, "tp_size": 2 },
            "decode": { "dp_size": 1, "tp_size": 2 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'

