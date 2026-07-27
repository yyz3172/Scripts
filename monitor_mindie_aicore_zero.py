#!/usr/bin/env python3
"""
监控 NPU 的 AICore 使用率。

当某张 NPU 的 AICore 使用率持续为 0 达到指定时长后，打印一条日志；
当该 NPU 从“长期为 0”恢复到 AICore > 0 时，再打印一条恢复日志。
可选通过飞书自定义机器人推送告警。

依赖: npu-smi

用法示例:
  # 默认监控所有 NPU，AICore 连续 60 秒为 0 时告警
  python monitor_mindie_aicore_zero.py

  # 开启飞书通知（也可 export FEISHU_WEBHOOK_URL=...）
  python monitor_mindie_aicore_zero.py --notify --feishu-webhook 'https://open.feishu.cn/open-apis/bot/v2/hook/xxx'

  # 每 2 秒轮询，连续 30 秒为 0 时告警
  python monitor_mindie_aicore_zero.py --interval 2 --zero-seconds 30

  # 只监控指定 NPU
  python monitor_mindie_aicore_zero.py --npu 0,1,2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from notify_feishu import add_feishu_args, notify_feishu, should_notify


DEVICE_HEADER_RE = re.compile(
    r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK|WARN|WARNING|ERROR|UNKNOWN)\b.*\|$"
)
DEVICE_METRIC_RE = re.compile(
    r"^\|\s*(\d+)\s+\|\s*([0-9A-Fa-f:.]+)\s+\|\s*(\d+)\s+"
)


@dataclass(frozen=True)
class NpuMetrics:
    npu_id: int
    chip_id: int
    aicore_percent: int


@dataclass
class NpuState:
    zero_since: float | None = None
    alerted_zero: bool = False


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}", flush=True)


def emit(message: str, *, notify_enabled: bool, webhook_url: str | None) -> None:
    log(message)
    if should_notify(notify_enabled, webhook_url):
        notify_feishu(message, webhook_url)


def run_npu_smi() -> str:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 npu-smi，请确认已安装并在 PATH 中") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"npu-smi info 执行失败: {stderr or exc}") from exc
    return result.stdout


def parse_npu_arg(npu_arg: str | None) -> list[int] | None:
    if npu_arg is None:
        return None

    raw_items = [item.strip() for item in npu_arg.split(",")]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError("--npu 格式非法，应为逗号分隔整数，如 0,1,2")

    npu_ids: list[int] = []
    for item in raw_items:
        try:
            npu_id = int(item)
        except ValueError as exc:
            raise ValueError(
                f"--npu 包含非法值 {item!r}，应为逗号分隔整数，如 0,1,2"
            ) from exc
        if npu_id not in npu_ids:
            npu_ids.append(npu_id)
    return npu_ids


def parse_metrics(output: str) -> list[NpuMetrics]:
    metrics: list[NpuMetrics] = []
    pending_npu_id: int | None = None

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if "Process id" in line and "Process name" in line:
            break

        header_match = DEVICE_HEADER_RE.match(line)
        if header_match:
            pending_npu_id = int(header_match.group(1))
            continue

        if pending_npu_id is None:
            continue

        metric_match = DEVICE_METRIC_RE.match(line)
        if not metric_match:
            continue

        chip_id, _bus_id, aicore_percent = metric_match.groups()
        metrics.append(
            NpuMetrics(
                npu_id=pending_npu_id,
                chip_id=int(chip_id),
                aicore_percent=int(aicore_percent),
            )
        )
        pending_npu_id = None

    return metrics


def filter_target_npus(
    all_npus: list[int],
    requested_npus: list[int] | None,
) -> list[int]:
    return all_npus if requested_npus is None else requested_npus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="监控 NPU AICore 长时间为 0 的情况"
    )
    parser.add_argument(
        "--npu",
        type=str,
        default=None,
        help="NPU 编号，逗号分隔，如 --npu 0,1；默认监控所有 NPU",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="轮询间隔秒数，默认 1",
    )
    parser.add_argument(
        "--zero-seconds",
        type=float,
        default=60.0,
        help="AICore 连续为 0 多久后报警，默认 60 秒",
    )
    add_feishu_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0:
        print("error: --interval 必须 > 0", file=sys.stderr)
        return 2
    if args.zero_seconds <= 0:
        print("error: --zero-seconds 必须 > 0", file=sys.stderr)
        return 2

    try:
        requested_npus = parse_npu_arg(args.npu)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    webhook_enabled = should_notify(args.notify, args.feishu_webhook)
    states: dict[int, NpuState] = {}
    target_scope = "all" if requested_npus is None else ",".join(map(str, requested_npus))
    log(
        f"开始监控 NPU[{target_scope}] AICore，"
        f"interval={args.interval}s zero_seconds={args.zero_seconds}s，"
        f"feishu={'on' if webhook_enabled else 'off'}"
    )

    while True:
        try:
            output = run_npu_smi()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        metrics = parse_metrics(output)
        all_npus = sorted(metric.npu_id for metric in metrics)
        target_npus = filter_target_npus(all_npus, requested_npus)
        metric_map = {
            metric.npu_id: metric
            for metric in metrics
            if metric.npu_id in set(target_npus)
        }

        current_time = time.time()
        for npu_id in target_npus:
            state = states.setdefault(npu_id, NpuState())
            metric = metric_map.get(npu_id)

            if metric is None:
                state.zero_since = None
                state.alerted_zero = False
                continue

            if metric.aicore_percent == 0:
                if state.zero_since is None:
                    state.zero_since = current_time
                zero_duration = current_time - state.zero_since
                if not state.alerted_zero and zero_duration >= args.zero_seconds:
                    emit(
                        f"NPU{npu_id} AICore 持续为 0 已达 {zero_duration:.1f}s",
                        notify_enabled=args.notify,
                        webhook_url=args.feishu_webhook,
                    )
                    state.alerted_zero = True
            else:
                if state.alerted_zero:
                    zero_duration = current_time - (state.zero_since or current_time)
                    emit(
                        f"NPU{npu_id} AICore 已恢复为 {metric.aicore_percent}%"
                        f"（此前连续为 0 {zero_duration:.1f}s）",
                        notify_enabled=args.notify,
                        webhook_url=args.feishu_webhook,
                    )
                state.zero_since = None
                state.alerted_zero = False

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
