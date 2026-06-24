#!/bin/sh
# Qwen3-30B-A3B-Instruct-2507/1P2_1D2：单机 1 个 P（2 卡 TP=2）+ 1 个 D（2 卡 TP=2），共 4 卡，Qwen3 MoE。
# 参考 Qwen3-32B/1P2_2D2 的环境与 connector 写法，拓扑对齐 Qwen3-32B/1P2_1D2；connector 中 decode dp_size=1, tp_size=2。
# MoE 需 --enable-expert-parallel（与 vllm-ascend Qwen3-30B-A3B TP=2 用法一致）。
# Instruct-2507 默认 norm_topk_prob=true 会走 Ascend aclnnMoeGatingTopK 的 renorm=1，当前 CANN 仅支持 renorm=0；
# 通过 --hf-overrides 关闭 norm_topk_prob，与改模型目录下 config.json 等价，属部署侧配置规避。
# 日志：vllm 输出始终写入 ${LOG_DIR}/prefill.log；LOG_DIR 未设置时默认为当前目录（与 run_decode.sh 一致）。
# NIC_NAME / LOCAL_IP 由 PdServiceCtl 注入；单独跑脚本时请 export，默认值与 pd_service_ctl 中常量一致。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/yyz/models/Qwen3-30B-A3B-Instruct-2507"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
dp_size=1
dp_ip="127.0.0.1"
dp_port=13395
engine_port=9000
visible_devices="0,1"

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

run_prefill() {
vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $engine_port \
    --enable-prefix-caching \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --hf-overrides '{"norm_topk_prob": false}' \
    --seed 1024 \
    --served-model-name qwen3_30b_a3b_instruct_2507 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 256 \
    --long-prefill-token-threshold 1024 \
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
            "decode": { "dp_size": 1, "tp_size": 2 }
        },
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
}

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"
run_prefill >> "${LOG_DIR}/prefill.log" 2>&1
