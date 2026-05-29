#!/usr/bin/env python3
"""解析 [DynamicKV][offload_profile] 的 rewrite_ms / finished_reqs，输出统计。"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

OFFLOAD_PROFILE_RE = re.compile(
    r"\[DynamicKV\]\[offload_profile\]\s+"
    r"rewrite_ms=([\d.]+)\s+finished_reqs=(\d+)"
)
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")


@dataclass(frozen=True)
class OffloadSample:
    rewrite_ms: float
    finished_reqs: int


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


def _stats_row(vals: list[float]) -> tuple[int, float, float, float, float, float, float]:
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
        _percentile_linear(srt, 99.0),
    )


def parse_log(path: str, *, by_worker: bool) -> tuple[int, dict[str, list[OffloadSample]]]:
    buckets: dict[str, list[OffloadSample]] = defaultdict(list)
    matched = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = OFFLOAD_PROFILE_RE.search(line)
            if not m:
                continue
            rewrite_ms = float(m.group(1))
            finished_reqs = int(m.group(2))
            matched += 1
            key = _worker_bucket(line) if by_worker else "__all__"
            buckets[key].append(
                OffloadSample(rewrite_ms=rewrite_ms, finished_reqs=finished_reqs)
            )
    return matched, dict(buckets)


def _print_bucket(title: str, samples: list[OffloadSample]) -> None:
    if not samples:
        return

    rewrite_vals = [s.rewrite_ms for s in samples]
    finished_vals = [float(s.finished_reqs) for s in samples]
    per_req_vals = [
        s.rewrite_ms / s.finished_reqs
        for s in samples
        if s.finished_reqs > 0
    ]

    n, mean, vmin, vmax, p50, p90, p99 = _stats_row(rewrite_vals)
    print(f"\n=== {title} (samples={n}) ===")
    print(
        "rewrite_ms (ms): "
        f"mean={mean:.2f} min={vmin:.2f} p50={p50:.2f} p90={p90:.2f} "
        f"p99={p99:.2f} max={vmax:.2f}"
    )

    if per_req_vals:
        _n, m, mn, mx, p50r, p90r, p99r = _stats_row(per_req_vals)
        print(
            "rewrite_ms_per_req (rewrite_ms/finished_reqs, ms): "
            f"mean={m:.2f} min={mn:.2f} p50={p50r:.2f} p90={p90r:.2f} "
            f"p99={p99r:.2f} max={mx:.2f}"
        )

    # finished_reqs distribution
    fr_counts: dict[int, int] = defaultdict(int)
    for s in samples:
        fr_counts[s.finished_reqs] += 1
    fr_parts = ", ".join(
        f"{k}={v}({100.0 * v / n:.1f}%)"
        for k, v in sorted(fr_counts.items())
    )
    print(f"finished_reqs: {fr_parts}")

    ge2 = sum(1 for s in samples if s.finished_reqs >= 2)
    ge3 = sum(1 for s in samples if s.finished_reqs >= 3)
    sum_rewrite = sum(rewrite_vals)
    sum_finished = int(sum(finished_vals))
    extra_reqs = sum(max(0, s.finished_reqs - 1) for s in samples)
    print(
        "pile-up: "
        f"steps_finished_reqs_ge2={ge2}({100.0 * ge2 / n:.1f}%) "
        f"steps_finished_reqs_ge3={ge3}({100.0 * ge3 / n:.1f}%) "
        f"sum_rewrite_ms={sum_rewrite:.2f} sum_finished_reqs={sum_finished} "
        f"sum_extra_reqs_in_pile_steps={extra_reqs}"
    )

    # Rough linearity check: rewrite_ms vs finished_reqs (only when fr>=2)
    multi = [s for s in samples if s.finished_reqs >= 2]
    if len(multi) >= 2:
        per_req_multi = [s.rewrite_ms / s.finished_reqs for s in multi]
        _n, m, *_rest, p99r = _stats_row(per_req_multi)
        print(
            f"hint: among finished_reqs>=2 steps (n={len(multi)}), "
            f"per_req mean={m:.2f}ms p99={p99r:.2f}ms "
            "(if ~constant vs single-req steps, pile-up is likely serial rewrite)"
        )


def _print_report(
    log_path: str,
    matched: int,
    buckets: dict[str, list[OffloadSample]],
    *,
    by_worker: bool,
) -> None:
    print(f"log: {log_path}")
    print(f"matched [DynamicKV][offload_profile] lines: {matched}")
    if matched == 0:
        print(
            "hint: enable prefill VLLM_DYNKV_PROFILE_FORWARD=1 and grep "
            "'[DynamicKV][offload_profile]' in prefill.log",
            file=sys.stderr,
        )
        return

    if by_worker:
        for worker in sorted(buckets):
            _print_bucket(worker, buckets[worker])
    else:
        all_samples: list[OffloadSample] = []
        for samples in buckets.values():
            all_samples.extend(samples)
        _print_bucket("__all__", all_samples)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse [DynamicKV][offload_profile] rewrite_ms / finished_reqs.",
    )
    ap.add_argument(
        "log_paths",
        nargs="+",
        help="Path(s) to prefill.log (or combined log).",
    )
    ap.add_argument(
        "--by-worker",
        action="store_true",
        help="Bucket stats by Ray worker name in log prefix.",
    )
    args = ap.parse_args()

    exit_code = 0
    for raw in args.log_paths:
        path = str(Path(raw).resolve())
        if not Path(path).is_file():
            print(f"error: log not found: {path}", file=sys.stderr)
            exit_code = 2
            continue
        matched, buckets = parse_log(path, by_worker=args.by_worker)
        if len(args.log_paths) > 1:
            print("\n" + "=" * 72)
        _print_report(path, matched, buckets, by_worker=args.by_worker)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
