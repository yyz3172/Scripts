#!/usr/bin/env python3
"""解析 [DynamicKV][model_acl_profile] 行，输出各字段统计（ms）。"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(r"\[DynamicKV\]\[model_acl_profile\]\s+(.+)$")
FIELD_RE = re.compile(r"(\w+)=([\d.]+)(ms)?")
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")

FIELD_ORDER = [
    "ctx_lens",
    "block_table",
    "block_table_swap",
    "graph_update",
    "gu_begin",
    "gu_pa",
    "gu_end",
    "event_record",
    "loop_other",
    "total",
    "per_layer_ctx",
    "per_layer_graph_update",
    "layers",
]


def _parse_fields(rest: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in FIELD_RE.finditer(rest):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return out


def _worker_bucket(line: str) -> str:
    m = WORKER_RE.search(line)
    return m.group(1) if m else "__all__"


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
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


def _print_table(tag_vals: dict[str, list[float]]) -> None:

    def sort_key(item: tuple[str, list[float]]) -> tuple[int, str]:
        tag = item[0]
        if tag in FIELD_ORDER:
            return (FIELD_ORDER.index(tag), tag)
        return (len(FIELD_ORDER), tag)

    rows: list[
        tuple[str, int, float, float, float, float, float, float, float]
    ] = []
    for tag, vals in sorted(tag_vals.items(), key=sort_key):
        n = len(vals)
        if not n:
            continue
        srt = sorted(vals)
        mean = sum(vals) / n
        rows.append((
            tag,
            n,
            mean,
            srt[0],
            srt[-1],
            _percentile_linear(srt, 50.0),
            _percentile_linear(srt, 90.0),
            _percentile_linear(srt, 95.0),
            _percentile_linear(srt, 99.0),
        ))
    hdr = (
        f"{'field':<24} {'cnt':>6} {'mean':>8} {'min':>8} {'max':>8} "
        f"{'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, n, mean, vmin, vmax, p50, p90, p95, p99 in rows:
        print(
            f"{tag:<24} {n:>6} {mean:>8.3f} {vmin:>8.3f} {vmax:>8.3f} "
            f"{p50:>8.3f} {p90:>8.3f} {p95:>8.3f} {p99:>8.3f}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse [DynamicKV][model_acl_profile] from vllm-ascend logs.",
    )
    ap.add_argument(
        "log_paths",
        nargs="+",
        help="One or more log files (two paths => side-by-side OFF/ON labels).",
    )
    ap.add_argument(
        "--by-worker",
        action="store_true",
        help="Split stats by Ray-style worker prefix.",
    )
    args = ap.parse_args()

    labels = ["OFF", "ON"] if len(args.log_paths) == 2 else [
        f"log{i}" for i in range(len(args.log_paths))
    ]
    any_matched = False
    sep = False
    for label, path in zip(labels, args.log_paths):
        matched, stats = parse_log(path, by_worker=args.by_worker)
        if matched == 0:
            print(
                f"no [model_acl_profile] lines in {path} "
                f"(VLLM_DYNKV_PROFILE_MODEL_ACL=1)",
                file=sys.stderr,
            )
            continue
        any_matched = True
        if len(args.log_paths) > 1:
            print(f"# {label}", file=sys.stderr)
        keys = sorted(stats.keys())
        for key in keys:
            if sep:
                print()
            sep = True
            _print_table(stats[key])
    if not any_matched:
        sys.exit(1)


if __name__ == "__main__":
    main()
