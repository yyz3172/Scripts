#!/bin/sh
# 在线压测时按时间窗口采集 Ascend PyTorch Profiler trace。
# 流程：start_profile → sleep N 秒 → stop_profile
#
# 用法：
#   bash profile_window.sh prefill 60    # Prefill（端口 9000），采集 60 秒
#   bash profile_window.sh decode 30     # Decode（端口 9010），采集 30 秒
#
# 环境变量：
#   LOCAL_IP  目标主机 IP，默认 127.0.0.1

set -e

role="${1:-}"
duration="${2:-}"

usage() {
  echo "用法: $0 <prefill|decode> <秒数>" >&2
  echo "  prefill -> http://\${LOCAL_IP}:9000/start_profile|stop_profile" >&2
  echo "  decode  -> http://\${LOCAL_IP}:9010/start_profile|stop_profile" >&2
  exit 1
}

case "$role" in
  prefill|p|P)
    port=9000
    role_label="Prefill"
    ;;
  decode|d|D)
    port=9010
    role_label="Decode"
    ;;
  *)
    usage
    ;;
esac

if [ -z "$duration" ] || ! echo "$duration" | grep -Eq '^[0-9]+$' || [ "$duration" -le 0 ]; then
  echo "错误: 第二个参数须为正整数（采集秒数）" >&2
  usage
fi

host="${LOCAL_IP:-127.0.0.1}"
base_url="http://${host}:${port}"

_profile_curl() {
  endpoint="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${role_label} POST ${base_url}/${endpoint}"
  if ! curl -sf -X POST "${base_url}/${endpoint}"; then
    echo "错误: ${base_url}/${endpoint} 请求失败" >&2
    exit 1
  fi
  echo
}

echo "=== ${role_label} profile 窗口: ${duration}s (${base_url}) ==="

_profile_curl start_profile
echo "采集中，等待 ${duration} 秒..."
sleep "$duration"
_profile_curl stop_profile

echo "=== ${role_label} profile 已停止 ==="
