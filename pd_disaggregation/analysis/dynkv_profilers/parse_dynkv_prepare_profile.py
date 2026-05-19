#!/usr/bin/env python3
"""
从 vLLM-Ascend 日志中解析 ``[DynamicKV][prepare_profile] ...`` 行，
统计各阶段耗时：条数、均值、最小、最大、P50 / P90 / P95 / P99（毫秒）。

日志行示例::

    [DynamicKV][prepare_profile] layers=80 build_helper=1.23ms broadcast=0.00ms stacked_tensor=0.45ms layer_copy_meta=5.67ms layer_slot_remap=12.34ms layer_other=2.10ms total_loop=20.11ms

用法::

    python parse_dynkv_prepare_profile.py /path/to/decode.log
    python parse_dynkv_prepare_profile.py /path/to/decode.log --by-worker

需要设置环境变量启用日志::

    export VLLM_DYNKV_PROFILE_PREPARE=1
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

# 整行锚点：``[DynamicKV][prepare_profile]`` 之后为 key=value 对
LINE_RE = re.compile(r"\[DynamicKV\]\[prepare_profile\]\s+(.+)$")

# 各字段：key=X.XXms 或 key=N（整数如 layers）
FIELD_RE = re.compile(r"(\w+)=([\d.]+)(ms)?")

# 可选：从 Ray 风格前缀里取 worker 名，便于 ``--by-worker`` 分桶
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")


def _parse_fields(rest: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in FIELD_RE.finditer(rest):
        key = m.group(1).strip()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        # 如果有 ms 后缀，表示是毫秒；否则是纯数值（如 layers）
        out[key] = val
    return out


def _worker_bucket(line: str) -> str:
    m = WORKER_RE.search(line)
    return m.group(1) if m else "__all__"


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数，p 为 0–100（与常见 P50 表示一致）。"""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    w = k - f
    return sorted_vals[f] * (1.0 - w) + sorted_vals[c] * w


def parse_log(
    path: str,
    *,
    by_worker: bool,
) -> tuple[int, dict[tuple[str, ...], dict[str, list[float]]]]:
    """
    Returns:
        (matched_lines, stats) where stats key is () or (worker,)
        and value is field_name -> list of ms/numeric samples.
    """
    stats: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    matched = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            fields = _parse_fields(m.group(1))
            if not fields:
                continue
            matched += 1
            key: tuple[str, ...] = ()
            if by_worker:
                key = (_worker_bucket(line),)
            bucket = stats[key]
            for fname, fval in fields.items():
                bucket[fname].append(fval)
    return matched, stats


def _print_table(title: str, tag_vals: dict[str, list[float]]) -> None:
    # 定义字段输出顺序（ms 字段按预期顺序，layers 放最后）
    order = [
        "stack_init",
        "kv_list_build",
        "build_helper",
        "broadcast",
        "stacked_tensor",
        "layer_copy_meta",
        "layer_slot_remap",
        "layer_other",
        "total_loop",
        "layers",
    ]
    rows: list[
        tuple[str, int, float, float, float, float, float, float, float]
    ] = []
    
    def sort_key(item: tuple[str, list[float]]) -> tuple[int, str]:
        tag = item[0]
        if tag in order:
            return (order.index(tag), tag)
        return (len(order), tag)
    
    for tag, vals in sorted(tag_vals.items(), key=sort_key):
        n = len(vals)
        if not n:
            continue
        srt = sorted(vals)
        mean = sum(vals) / n
        vmin = srt[0]
        vmax = srt[-1]
        p50 = _percentile_linear(srt, 50.0)
        p90 = _percentile_linear(srt, 90.0)
        p95 = _percentile_linear(srt, 95.0)
        p99 = _percentile_linear(srt, 99.0)
        rows.append((tag, n, mean, vmin, vmax, p50, p90, p95, p99))
    print(title)
    hdr = (
        f"{'field':<20} {'cnt':>6} {'mean':>8} {'min':>8} {'max':>8} "
        f"{'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, n, mean, vmin, vmax, p50, p90, p95, p99 in rows:
        print(
            f"{tag:<20} {n:>6} {mean:>8.3f} {vmin:>8.3f} {vmax:>8.3f} "
            f"{p50:>8.3f} {p90:>8.3f} {p95:>8.3f} {p99:>8.3f}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse [DynamicKV][prepare_profile] lines from vllm-ascend logs; "
        "print mean/min/max and P50/P90/P95/P99 (ms) per field.",
    )
    ap.add_argument(
        "log_path",
        help="Path to a log file (plain text, e.g. decode worker redirect).",
    )
    ap.add_argument(
        "--by-worker",
        action="store_true",
        help="Split stats by Ray-style worker prefix (IntegratedWorker pid=...).",
    )
    args = ap.parse_args()

    matched, stats = parse_log(args.log_path, by_worker=args.by_worker)
    print(f"File: {args.log_path}")
    print(f"Matched [DynamicKV][prepare_profile] lines: {matched}")
    if matched == 0:
        print(
            "No matching lines. Ensure logs contain "
            "'[DynamicKV][prepare_profile]' "
            "and VLLM_DYNKV_PROFILE_PREPARE=1 was enabled.",
        )
        return

    # 稳定输出顺序
    keys = sorted(stats.keys())
    for key in keys:
        if args.by_worker and key:
            worker = key[0]
            title = f"== DynamicKV Prepare Profile | worker={worker} =="
        else:
            title = "== DynamicKV Prepare Profile =="
        _print_table(title, stats[key])


if __name__ == "__main__":
    main()
