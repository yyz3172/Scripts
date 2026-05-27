#!/usr/bin/env python3
"""解析 [DynamicKV][prepare_profile] 决策字段行，输出统计（ms）。"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

LINE_PROFILE_RE = re.compile(r"\[DynamicKV\]\[prepare_profile\]\s+(.+)$")
LINE_EXT_RE = re.compile(r"\[DynamicKV\]\[prepare_profile_ext\]\s+(.+)$")
LINE_RECONCILE_RE = re.compile(
    r"\[DynamicKV\]\[prepare_profile_reconcile\]\s+(.+)$")
FIELD_RE = re.compile(r"(\w+)=([\d.]+)(ms)?")
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")

PROFILE_FIELDS = (
    "dynkv",
    "profile_prepare_est",
    "baseline",
    "core_dynkv_upload",
    "dynkv_branch",
    "slot_remap",
    "branch_setup",
    "branch_misc",
    "prepare_gap",
)


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
    line_re: re.Pattern[str],
    field_filter: str | None = None,
) -> tuple[int, dict[tuple[str, ...], dict[str, list[float]]]]:
    """field_filter: require this key in parsed fields (e.g. profile_prepare_est)."""
    stats: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    matched = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            fields = _parse_fields(m.group(1))
            if not fields:
                continue
            if field_filter is not None and field_filter not in fields:
                continue
            matched += 1
            key: tuple[str, ...] = ()
            if by_worker:
                key = (_worker_bucket(line),)
            for fname, fval in fields.items():
                stats[key][fname].append(fval)
    return matched, dict(stats)


def _row_stats(vals: list[float]) -> tuple[int, float, float, float, float, float, float, float]:
    n = len(vals)
    srt = sorted(vals)
    mean = sum(vals) / n
    return (
        n,
        mean,
        srt[0],
        srt[-1],
        _percentile_linear(srt, 50.0),
        _percentile_linear(srt, 90.0),
        _percentile_linear(srt, 95.0),
        _percentile_linear(srt, 99.0),
    )


def _print_table(tag_vals: dict[str, list[float]]) -> None:
    field_w = 28
    hdr = (
        f"{'field':<{field_w}} {'cnt':>6} {'mean':>8} {'p50':>8} "
        f"{'p90':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag in PROFILE_FIELDS:
        vals = tag_vals.get(tag)
        if not vals:
            continue
        n, mean, _vmin, _vmax, p50, p90, _p95, p99 = _row_stats(vals)
        print(
            f"{tag:<{field_w}} {n:>6} {mean:>8.3f} {p50:>8.3f} "
            f"{p90:>8.3f} {p99:>8.3f}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse [DynamicKV][prepare_profile] from vllm-ascend logs.",
    )
    ap.add_argument("log_path", help="Path to decode log.")
    ap.add_argument("--by-worker", action="store_true")
    ap.add_argument(
        "--view",
        choices=["default", "legacy"],
        default="default",
        help="default=prepare_profile decision fields; legacy=ext/reconcile/loop.",
    )
    args = ap.parse_args()

    if args.view == "default":
        matched, stats = parse_log(
            args.log_path,
            by_worker=args.by_worker,
            line_re=LINE_PROFILE_RE,
            field_filter="profile_prepare_est",
        )
        if matched == 0:
            print(
                "no [prepare_profile] decision lines "
                "(VLLM_DYNKV_PROFILE_PREPARE=1)",
                file=sys.stderr,
            )
            sys.exit(1)
        keys = sorted(stats.keys())
        for i, key in enumerate(keys):
            if i:
                print()
            _print_table(stats[key])
        return

    matched_loop, _stats_loop = parse_log(
        args.log_path,
        by_worker=args.by_worker,
        line_re=LINE_PROFILE_RE,
        field_filter="layers",
    )
    matched_ext, _ = parse_log(
        args.log_path, by_worker=args.by_worker, line_re=LINE_EXT_RE)
    matched_rec, _ = parse_log(
        args.log_path, by_worker=args.by_worker, line_re=LINE_RECONCILE_RE)
    if matched_loop + matched_ext + matched_rec == 0:
        print("no legacy prepare_profile lines", file=sys.stderr)
        sys.exit(1)
    print(
        f"legacy loop={matched_loop} ext={matched_ext} reconcile={matched_rec}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
