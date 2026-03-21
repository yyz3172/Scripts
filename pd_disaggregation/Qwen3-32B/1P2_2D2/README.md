# 1P2+2D2：1 个 P（2 卡）+ 2 个 D（各 2 卡），共 6 卡

- **Prefill**：1 个实例，TP=2，卡 0,1，API 端口 9000
- **Decode**：2 个实例，各 TP=2
  - rank 0：卡 2,3，API 端口 9010
  - rank 1：卡 4,5，API 端口 9011

## 手动启动

1. 先起 Prefill：`bash run_prefill.sh`
2. Decode 二选一：
   - 一键起两个 D：`bash run_decode.sh`（供 start_decode.sh 调用）
   - 只起一个：`bash run_decode.sh 0` 或 `bash run_decode.sh 1`

## 负载均衡代理（可选）

`pd_proxy.sh` 将 1 个 P（9000）与 2 个 D（9010、9011）统一到端口 8000，客户端连 8000 即可。先起 P 和 D，再执行：`bash pd_proxy.sh`。与 prefill/decode 相同：通过 `LOCAL_IP`（本机互通 IP）、`NIC_NAME` 注入；`PROXY_PORT` 可覆盖代理监听端口。

## 压测 / sweep

`start_prefill.sh` 与 `start_decode.sh` 已指向本目录的 `run_prefill.sh`、`run_decode.sh`。压测时客户端可直连 9010/9011，或先起 `pd_proxy.sh` 后连 8000。

## 配置

- `NIC_NAME`、`LOCAL_IP` 建议与 `pd_python/pd_service_ctl.py` 中常量一致，或由上层 `export` 后启动脚本。
- `run_prefill.sh` 与 `run_decode.sh` 中 `model_path` 等模型/部署项须一致。
- 若卡号/端口与默认不同，请改 `run_decode.sh` 里 `run_one()` 的 `case $dp_rank in` 对应项。
