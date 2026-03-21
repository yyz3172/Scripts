#!/bin/sh
# 1P2+2D2：单机 1 个 P（2 卡 TP=2）+ 2 个 D（各 2 卡 TP=2），共 6 卡，Qwen3-32B。
# 与 run_decode_1.sh / run_decode_2.sh 配套；connector 中 decode dp_size=2, tp_size=2。
# ========== 配置区 ==========
nic_name="eth0"
local_ip="172.17.0.4"
model_path="/root/autodl-tmp/models/Qwen3-32B"
# transfer_engine_lib_path="/usr/local/Ascend/ascend-toolkit/latest/lib64"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
dp_size=1
dp_ip="127.0.0.1"
dp_port=13395
engine_port=9000
visible_devices="0,1"
# ==========================================

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
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $engine_port \
    --tensor-parallel-size 2 \
    --seed 1024 \
    --served-model-name qwen3_32b \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
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
            "decode": { "dp_size": 2, "tp_size": 2 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'

