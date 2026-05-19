#!/usr/bin/env python3
"""
从 vLLM-Ascend 日志中解析 ``Profile execute duration [Decode|Prefill]: ...`` 行，
按阶段（Decode / Prefill）与字段 tag 统计：条数、均值、最小、最大、
P50 / P90 / P95 / P99（毫秒，线性插值分位数）。

日志行示例（可能带 Ray / worker 前缀）::

    Profile execute duration [Decode]: [post process]:14.17ms [prepare input]:9.57ms [forward]:45.12ms

用法::

    python parse_profile_execute_duration.py /path/to/decode.log
    python parse_profile_execute_duration.py /path/to/decode.log --by-worker
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

# 整行锚点：``Profile execute duration [Decode]:`` 之后为一串 ``[tag]:X.XXms``
LINE_RE = re.compile(r"Profile execute duration \[([^\]]+)\]:\s*(.+)$")

# 各阶段耗时字段；tag 内可含空格（如 ``prepare input``）
FIELD_RE = re.compile(r"\[([^\]]+)\]:([\d.]+)\s*ms")

# 可选：从 Ray 风格前缀里取 worker 名，便于 ``--by-worker`` 分桶
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")


def _parse_fields(rest: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for m in FIELD_RE.finditer(rest):
        tag = m.group(1).strip()
        try:
            ms = float(m.group(2))
        except ValueError:
            continue
        out.append((tag, ms))
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
        (matched_lines, stats) where stats key is (phase,) or (phase, worker)
        and value is tag -> list of ms samples.
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
            phase = m.group(1).strip()
            fields = _parse_fields(m.group(2))
            if not fields:
                continue
            matched += 1
            key_tail: tuple[str, ...] = ()
            if by_worker:
                key_tail = (_worker_bucket(line),)
            key: tuple[str, ...] = (phase,) + key_tail
            bucket = stats[key]
            for tag, ms in fields:
                bucket[tag].append(ms)
    return matched, stats


def _print_table(title: str, tag_vals: dict[str, list[float]]) -> None:
    rows: list[
        tuple[str, int, float, float, float, float, float, float, float]
    ] = []
    for tag, vals in sorted(tag_vals.items(), key=lambda x: (-len(x[1]), x[0])):
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
        f"{'tag':<32} {'cnt':>6} {'mean':>8} {'min':>8} {'max':>8} "
        f"{'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, n, mean, vmin, vmax, p50, p90, p95, p99 in rows:
        print(
            f"{tag:<32} {n:>6} {mean:>8.3f} {vmin:>8.3f} {vmax:>8.3f} "
            f"{p50:>8.3f} {p90:>8.3f} {p95:>8.3f} {p99:>8.3f}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse Profile execute duration lines from vllm-ascend logs; "
        "print mean/min/max and P50/P90/P95/P99 (ms) per tag.",
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
    print(f"Matched Profile execute duration lines: {matched}")
    if matched == 0:
        print(
            "No matching lines. Ensure logs contain "
            "'Profile execute duration [Decode|Prefill]:' "
            "and VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1 was enabled.",
        )
        return

    # 稳定输出顺序：先 phase，再 worker
    keys = sorted(stats.keys(), key=lambda k: (k[0], k[1:]))
    for key in keys:
        phase = key[0]
        if args.by_worker and len(key) > 1:
            worker = key[1]
            title = f"== {phase} | worker={worker} =="
        else:
            title = f"== {phase} =="
        _print_table(title, stats[key])


if __name__ == "__main__":
    main()
