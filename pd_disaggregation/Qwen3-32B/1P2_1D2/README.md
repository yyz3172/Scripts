# 1P2+1D2：1 个 P（2 卡）+ 1 个 D（2 卡），共 4 卡

- **Prefill**：1 个实例，TP=2，卡 0,1，API 端口 9000
- **Decode**：1 个实例，TP=2，卡 2,3，API 端口 9010

## 手动启动

1. 先起 Prefill：`bash run_prefill.sh`
2. Decode：
   - 一键起 D：`bash run_decode.sh`（供 start_decode.sh 调用）
   - 或指定 rank：`bash run_decode.sh 0`

## 负载均衡代理（可选）

`pd_proxy.sh` 将 1 个 P（9000）与 1 个 D（9010）统一到端口 8000，客户端连 8000 即可。先起 P 和 D，再执行：`bash pd_proxy.sh`。可通过环境变量 `PREFILL_HOST`、`PROXY_PORT` 覆盖本机 IP 与代理端口。

## 压测 / sweep

`start_prefill.sh` 与 `start_decode.sh` 已指向本目录的 `run_prefill.sh`、`run_decode.sh`。压测时客户端可直连 9010，或先起 `pd_proxy.sh` 后连 8000。

## 配置

- `run_prefill.sh` 与 `run_decode.sh` 顶部配置区的 `nic_name`、`local_ip`、`model_path` 须一致。
- 若卡号/端口与默认不同，请改 `run_decode.sh` 里 `run_one()` 的 `case $dp_rank in` 对应项。
