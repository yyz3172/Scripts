#!/usr/bin/env python3
"""
从单个 vLLM 日志文件中解析周期性打印的吞吐与并发指标行，并对各数值字段统计：
样本数、平均值、P50、P90、P99、P100（最大值）。

解析：剥离 ANSI 后，在行内搜索「Avg prompt throughput … External prefix cache hit rate」整段（不要求 INFO / 时间戳 / Engine 前缀）。

日志行示例（匹配从 Avg prompt 起的片段即可）：
  ... Avg prompt throughput: 2304.0 tokens/s, Avg generation throughput: 227.6 tokens/s, ...

用法：
  python throughput_log_percentiles.py <log_file_path>
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ANSI_STRIP = re.compile(r"\x1b\[[0-9;]*m")

THROUGHPUT_CONCURRENCY_LINE_PATTERN = re.compile(
    r"Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+"
    r"Avg generation throughput:\s+([\d.]+)\s+tokens/s,\s+"
    r"Running:\s+(\d+)\s+reqs,\s+Waiting:\s+(\d+)\s+reqs,\s+"
    r"GPU KV cache usage:\s+([\d.]+)%,\s+"
    r"Prefix cache hit rate:\s+([\d.]+)%,\s+"
    r"External prefix cache hit rate:\s+([\d.]+)%"
)

METRIC_NAMES = [
    "Avg prompt throughput",
    "Avg generation throughput",
    "Running",
    "Waiting",
    "GPU KV cache usage",
    "Prefix cache hit rate",
    "External prefix cache hit rate",
]


def strip_ansi(line: str) -> str:
    return ANSI_STRIP.sub("", line)


def parse_log_file(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            plain = strip_ansi(line)
            m = THROUGHPUT_CONCURRENCY_LINE_PATTERN.search(plain)
            if not m:
                continue
            vals = [m.group(i) for i in range(1, 8)]
            row: dict = {}
            for name, v in zip(METRIC_NAMES, vals):
                try:
                    row[name] = float(v)
                except ValueError:
                    row[name] = math.nan
            rows.append(row)
    return rows


def percentile_linear(sorted_vals: list[float], p: float) -> float:
    """p ∈ [0, 100]，线性插值分位数（与 numpy.percentile(..., method='linear') 一致）。"""
    n = len(sorted_vals)
    if n == 0:
        return math.nan
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_metric(values: list[float]) -> dict[str, float]:
    xs = [x for x in values if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "p99": math.nan,
            "p100": math.nan,
        }
    s = sorted(xs)
    return {
        "count": float(n),
        "mean": sum(s) / n,
        "p50": percentile_linear(s, 50),
        "p90": percentile_linear(s, 90),
        "p99": percentile_linear(s, 99),
        "p100": s[-1],
    }


def format_float(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.6g}"


def print_table(stats_by_metric: dict[str, dict[str, float]]) -> None:
    cols = ["metric", "count", "mean", "p50", "p90", "p99", "p100"]
    rows_out = []
    for name in METRIC_NAMES:
        st = stats_by_metric[name]
        rows_out.append(
            [
                name,
                str(int(st["count"])),
                format_float(st["mean"]),
                format_float(st["p50"]),
                format_float(st["p90"]),
                format_float(st["p99"]),
                format_float(st["p100"]),
            ]
        )
    widths = [max(len(cols[i]), max(len(r[i]) for r in rows_out)) for i in range(len(cols))]
    header = " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    print(header)
    print(sep)
    for r in rows_out:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(len(cols))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解析单日志中的吞吐/并发指标行，输出各字段均值与分位数"
    )
    parser.add_argument("log_file", type=str, help="日志文件路径")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        print(f"文件不存在: {log_path}", file=sys.stderr)
        sys.exit(1)

    rows = parse_log_file(log_path)
    if not rows:
        print(f"未匹配到任何吞吐指标行: {log_path}", file=sys.stderr)
        sys.exit(2)

    stats_by_metric: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        stats_by_metric[name] = summarize_metric([r[name] for r in rows])

    print(f"文件: {log_path}")
    print(f"匹配行数: {len(rows)}")
    print()
    print_table(stats_by_metric)


if __name__ == "__main__":
    main()
