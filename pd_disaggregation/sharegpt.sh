# 用法: $0 <test_name> <batch_size> [port] [pd_mode]
# port: 可选，vLLM API 端口（走代理时传 8000），默认 9010；会导出 VLLM_PORT 供压测配置使用
# pd_mode: 可选，PD 配置名（如 Qwen3-32B/1P2_1D2），用于 --work-dir 取 basename；未传时用 default
TEST_NAME="${1:?Usage: $0 <test_name> <batch_size> [port] [pd_mode]}"
BATCH_SIZE="${2:?Usage: $0 <test_name> <batch_size> [port] [pd_mode]}"
VLLM_PORT="${3:-9010}"
PD_MODE="${4:-}"
export VLLM_PORT
LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}/batch_${BATCH_SIZE}"
WORK_DIR="sharegpt_$PD_MODE"

source /root/autodl-tmp/py_venv/tester/bin/activate
cd /root/autodl-tmp/code/benchmark

CONFIG_PY="ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py"
sed -i "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/g" "$CONFIG_PY"
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf --num-warmups 0 --work-dir "outputs/${WORK_DIR}/" >> "$LOG_DIR/aisbench.log" 2>&1
