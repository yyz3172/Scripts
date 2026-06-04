#!/bin/sh
# Mistral-7B-Instruct-v0.2/1P2_1D2_dynamickv_pia：单机 1 个 Prefill（2 卡 TP=2），与 PIA decode 配套。
# Prefill 与 decode 使用相同 dynamic_kv 配置（含 head_aggregation=sum）。
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

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"

# DynamicKV 日志 profiling（配合 analysis/dynkv_profilers/ 解析 prefill.log）
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
export VLLM_DYNKV_PROFILE_PREPARE=0
# TTFT / offload rewrite：开 FORWARD=1 可看 [DynamicKV][offload_profile] rewrite_ms=
export VLLM_DYNKV_PROFILE_FORWARD=0

# Ascend PyTorch Profiler（--profiler-config，见 service_profiling_guide）
# 启停：curl -X POST http://${LOCAL_IP}:9000/start_profile|stop_profile
# 分析：from torch_npu.profiler.profiler import analyse; analyse("${PROFILER_DIR}/localhost.*_ascend_pt/")
ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-1}"
profiler_config_json=""
if [ "$ENABLE_TORCH_PROFILER" = "1" ]; then
  PROFILER_DIR="${PROFILER_DIR:-${LOG_DIR}/vllm_profile/prefill}"
  mkdir -p "$PROFILER_DIR"
  export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
  # profiler: 分析器类型，"torch"=PyTorch/Ascend 算子级 trace；"cuda"=CUDA/NVTX（配合 Nsight）
  # torch_profiler_dir: trace 落盘目录（需绝对路径；P/D 分离时 P/D 各设独立目录）
  # torch_profiler_with_stack: 是否采集 Python 调用栈（true 数据量大、开销高）
  # torch_profiler_record_shapes: 是否记录 tensor shape（true 增大 trace 体积）
  # torch_profiler_with_memory: 是否记录内存占用（true 便于分析显存，采集时略增开销）
  # ignore_frontend: 是否跳过 AsyncLLM 前端 CPU profiling（true 降低在线服务额外开销）
  profiler_config_json='{
    "profiler": "torch",
    "torch_profiler_dir": "'"${PROFILER_DIR}"'",
    "torch_profiler_with_stack": true,
    "torch_profiler_record_shapes": false,
    "torch_profiler_with_memory": true,
    "ignore_frontend": true
  }'
fi
if [ "$dp_size" -gt 1 ]; then
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1
else
  export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=0
fi
run_prefill() {
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
            "head_aggregation": "sum",
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

run_prefill >> "${LOG_DIR}/prefill.log" 2>&1

