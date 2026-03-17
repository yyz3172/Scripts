#!/bin/sh
# 1P2+2D2 负载均衡代理：1 个 P（9000）+ 2 个 D（9010、9011），对外统一 8000。
# 与 run_prefill.sh / run_decode.sh 中的 local_ip、端口一致；若改过请同步修改此处。
PREFILL_HOST="${PREFILL_HOST:-172.17.0.2}"
PROXY_PORT="${PROXY_PORT:-8000}"

python /root/autodl-tmp/yyz/code/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port "$PROXY_PORT" \
  --host 0.0.0.0 \
  --prefiller-hosts "$PREFILL_HOST" \
  --prefiller-ports 9000 \
  --decoder-hosts "$PREFILL_HOST" "$PREFILL_HOST" \
  --decoder-ports 9010 9011
