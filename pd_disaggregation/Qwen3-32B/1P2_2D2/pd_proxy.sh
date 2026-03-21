#!/bin/sh
# 1P2+2D2 负载均衡代理：1 个 P（9000）+ 2 个 D（9010、9011），对外统一 8000。
# LOCAL_IP：与 run_prefill/run_decode 中 HCCL 面 IP 相同，亦作代理访问 prefill/decode 的 host。
PROXY_PORT="${PROXY_PORT:-8000}"
_host="${LOCAL_IP:-172.17.0.4}"

python /root/autodl-tmp/yyz/code/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port "$PROXY_PORT" \
  --host 0.0.0.0 \
  --prefiller-hosts "$_host" \
  --prefiller-ports 9000 \
  --decoder-hosts "$_host" "$_host" \
  --decoder-ports 9010 9011
