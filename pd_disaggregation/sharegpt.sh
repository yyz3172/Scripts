# 用法: $0 <test_name> <batch_size> [port]
# port: 可选，vLLM API 端口（走代理时传 8000），默认 9010；会导出 VLLM_PORT 供压测配置使用
TEST_NAME="${1:?Usage: $0 <test_name> <batch_size> [port]}"
BATCH_SIZE="${2:?Usage: $0 <test_name> <batch_size> [port]}"
VLLM_PORT="${3:-9010}"
export VLLM_PORT
LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}/batch_${BATCH_SIZE}"

source /root/autodl-tmp/py_venv/tester/bin/activate
cd /root/autodl-tmp/code/benchmark

CONFIG_PY="ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py"
sed -i "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/g" "$CONFIG_PY"
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf --num-warmups >> "$LOG_DIR/aisbench.log" 2>&1
