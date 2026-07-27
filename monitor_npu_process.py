#!/usr/bin/env python3
"""
监控 NPU 上的进程占用情况（默认：所有 NPU，所有进程）。
可选按进程名过滤，并可通过飞书自定义机器人推送变化通知。

依赖: npu-smi

用法示例:
  # 单次检查
  python monitor_npu_process.py

  # 持续监控：初始化打印全部；之后仅打印有变化的卡
  python monitor_npu_process.py --watch --interval 2

  # 只监控指定进程名（逗号分隔，子串匹配）
  python monitor_npu_process.py --watch --process mindie_llm_back,vllm

  # 开启飞书通知（也可 export FEISHU_WEBHOOK_URL=...）
  python monitor_npu_process.py --watch --notify --feishu-webhook 'https://open.feishu.cn/open-apis/bot/v2/hook/xxx'

  # 只监控指定卡
  python monitor_npu_process.py --npu 0,1 --watch

退出码（单次检查）:
  0 = 有进程占用
  1 = 全部空闲
  2 = npu-smi 执行失败或解析失败
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


PROCESS_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s+\|\s*(\d+)\s+\|\s*(\S+)\s+\|\s*(\d+)\s*\|$"
)
DEVICE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK|WARN|WARNING|ERROR|UNKNOWN)\b.*\|$"
)


@dataclass(frozen=True)
class NpuProcess:
    npu_id: int
    chip_id: int
    pid: int
    name: str
    memory_mb: int


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


def parse_processes(output: str) -> list[NpuProcess]:
    """解析 npu-smi info 底部 Process 表。"""
    processes: list[NpuProcess] = []
    in_process_table = False

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").strip()) <= {"=", "+", "-", " "}:
            continue

        match = PROCESS_ROW_RE.match(line)
        if not match:
            continue

        npu_id, chip_id, pid, name, memory_mb = match.groups()
        processes.append(
            NpuProcess(
                npu_id=int(npu_id),
                chip_id=int(chip_id),
                pid=int(pid),
                name=name,
                memory_mb=int(memory_mb),
            )
        )
    return processes


def parse_available_npus(output: str) -> list[int]:
    """解析 npu-smi info 顶部设备表中的 NPU 编号。"""
    npu_ids: list[int] = []
    in_process_table = False

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if "Process id" in line and "Process name" in line:
            in_process_table = True
        if in_process_table:
            break

        match = DEVICE_ROW_RE.match(line)
        if not match:
            continue

        npu_id = int(match.group(1))
        if npu_id not in npu_ids:
            npu_ids.append(npu_id)

    return npu_ids


def parse_npu_arg(npu_arg: str | None) -> list[int] | None:
    """解析 --npu 参数，格式如: 0,1,2。"""
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


def parse_process_arg(process_arg: str | None) -> list[str] | None:
    """解析 --process 参数，格式如: mindie_llm_back,vllm。"""
    if process_arg is None:
        return None

    raw_items = [item.strip() for item in process_arg.split(",")]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError(
            "--process 格式非法，应为逗号分隔进程名，如 mindie_llm_back,vllm"
        )

    names: list[str] = []
    for item in raw_items:
        name = item.lower()
        if name not in names:
            names.append(name)
    return names


def find_occupants(
    processes: list[NpuProcess],
    npu_ids: set[int],
    process_names: list[str] | None = None,
) -> list[NpuProcess]:
    result: list[NpuProcess] = []
    for p in processes:
        if p.npu_id not in npu_ids:
            continue
        if process_names is None:
            result.append(p)
            continue
        name_lower = p.name.lower()
        if any(target in name_lower for target in process_names):
            result.append(p)
    return result


def check_once(
    requested_npus: list[int] | None,
    process_names: list[str] | None = None,
) -> tuple[list[int], bool, list[NpuProcess]]:
    output = run_npu_smi()
    available_npus = parse_available_npus(output)
    target_npus = available_npus if requested_npus is None else requested_npus
    processes = parse_processes(output)
    occupants = find_occupants(processes, set(target_npus), process_names)
    return target_npus, bool(occupants), occupants


def group_by_npu(
    target_npus: list[int],
    occupants: list[NpuProcess],
) -> dict[int, list[NpuProcess]]:
    occ_by_npu: dict[int, list[NpuProcess]] = {npu: [] for npu in target_npus}
    for p in occupants:
        if p.npu_id in occ_by_npu:
            occ_by_npu[p.npu_id].append(p)
    return occ_by_npu


def npu_status_key(procs: list[NpuProcess]) -> tuple:
    """单卡状态键：进程身份（pid/name/chip）。"""
    return tuple((p.pid, p.name, p.chip_id) for p in procs)


def format_npu_line(npu_id: int, procs: list[NpuProcess]) -> str:
    if procs:
        detail = "  ".join(
            f"{p.name}(pid={p.pid},mem={p.memory_mb}MB)" for p in procs
        )
        return f"  NPU{npu_id}  占用中  {detail}"
    return f"  NPU{npu_id}  空闲"


def format_status_text(
    npu_ids: list[int],
    occ_by_npu: dict[int, list[NpuProcess]],
    *,
    with_timestamp: bool = True,
) -> str:
    if not npu_ids:
        return ""
    lines: list[str] = []
    if with_timestamp:
        lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    for npu_id in npu_ids:
        lines.append(format_npu_line(npu_id, occ_by_npu.get(npu_id, [])))
    return "\n".join(lines)


def print_status(
    npu_ids: list[int],
    occ_by_npu: dict[int, list[NpuProcess]],
) -> None:
    text = format_status_text(npu_ids, occ_by_npu, with_timestamp=True)
    if text:
        print(text, flush=True)


def notify_status(
    npu_ids: list[int],
    occ_by_npu: dict[int, list[NpuProcess]],
    *,
    notify_enabled: bool,
    webhook_url: str | None,
) -> None:
    if not should_notify(notify_enabled, webhook_url) or not npu_ids:
        return
    body = format_status_text(npu_ids, occ_by_npu, with_timestamp=False)
    notify_feishu(f"NPU进程变化\n{body}", webhook_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="监控 NPU 上的进程占用情况（基于 npu-smi info）"
    )
    parser.add_argument(
        "--npu",
        type=str,
        default=None,
        help="NPU 编号，逗号分隔，如 --npu 0,1；默认监控所有 NPU",
    )
    parser.add_argument(
        "--process",
        type=str,
        default=None,
        help=(
            "进程名过滤，逗号分隔，子串匹配，"
            "如 --process mindie_llm_back,vllm；默认监控所有进程"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="轮询间隔秒数，默认 1",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="持续监控：初始化打印全部；之后仅打印有变化的卡",
    )
    add_feishu_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0:
        print("error: --interval 必须 > 0", file=sys.stderr)
        return 2
    try:
        requested_npus = parse_npu_arg(args.npu)
        process_names = parse_process_arg(args.process)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    webhook_enabled = should_notify(args.notify, args.feishu_webhook)
    try:
        target_npus, occupied, occupants = check_once(
            requested_npus,
            process_names,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    occ_by_npu = group_by_npu(target_npus, occupants)
    last_keys = {
        npu_id: npu_status_key(procs) for npu_id, procs in occ_by_npu.items()
    }

    # 初始化打印全部（不推飞书，避免启动刷屏）
    print_status(target_npus, occ_by_npu)
    if args.watch:
        process_scope = (
            "all" if process_names is None else ",".join(process_names)
        )
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"process=[{process_scope}] "
            f"feishu={'on' if webhook_enabled else 'off'}",
            flush=True,
        )

    if not args.watch:
        return 0 if occupied else 1

    while True:
        time.sleep(args.interval)
        try:
            target_npus, occupied, occupants = check_once(
                requested_npus,
                process_names,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        occ_by_npu = group_by_npu(target_npus, occupants)
        changed_npus: list[int] = []
        for npu_id in target_npus:
            key = npu_status_key(occ_by_npu.get(npu_id, []))
            if key != last_keys.get(npu_id):
                changed_npus.append(npu_id)
                last_keys[npu_id] = key

        # 清理已不在监控范围的卡
        for npu_id in list(last_keys):
            if npu_id not in occ_by_npu:
                del last_keys[npu_id]

        if changed_npus:
            print_status(changed_npus, occ_by_npu)
            notify_status(
                changed_npus,
                occ_by_npu,
                notify_enabled=args.notify,
                webhook_url=args.feishu_webhook,
            )


if __name__ == "__main__":
    sys.exit(main())
