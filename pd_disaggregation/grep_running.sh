while true; do
    echo "[$(date '+%H:%M:%S.%3N')] $(curl -s http://127.0.0.1:9000/metrics  | grep qwen3_32b | grep running)" >> logs/metrics_running.log
    sleep 1
done
