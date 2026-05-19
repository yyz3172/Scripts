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
LINE_EXT_RE = re.compile(r"\[DynamicKV\]\[prepare_profile_ext\]\s+(.+)$")
LINE_RECONCILE_RE = re.compile(
    r"\[DynamicKV\]\[prepare_profile_reconcile\]\s+(.+)$")

# 各字段：key=X.XXms 或 key=N（整数如 layers）
FIELD_RE = re.compile(r"(\w+)=([\d.]+)(ms)?")

# 可选：从 Ray 风格前缀里取 worker 名，便于 ``--by-worker`` 分桶
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")

# ``total_loop`` = 以下四项之和（见 model_runner ``_dynkv_log_prepare_profile_loop``）
TOTAL_LOOP_CHILDREN: tuple[str, ...] = (
    "layer_slot_remap",
    "layer_slot_assign",
    "layer_copy_meta",
    "layer_meta_assign",
)
TOTAL_LOOP_CHILD_INDENT = "  "


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
    line_re: re.Pattern[str] = LINE_RE,
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
            m = line_re.search(line)
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


def _print_table(title: str, tag_vals: dict[str, list[float]]) -> None:
    child_set = set(TOTAL_LOOP_CHILDREN)
    # 主字段顺序；``total_loop`` 四项子字段在打印时缩进挂在 ``total_loop`` 下
    order = [
        "update_states",
        "prepare_inputs_wall",
        "prepare_core",
        "prepare_kv_setup",
        "prepare_attn_build",
        "loop_sum",
        "prepare_cos_sin",
        "prepare_tail",
        "inner_sum",
        "prepare_gap",
        "profile_prepare_est",
        "stack_init",
        "kv_list_build",
        "build_helper",
        "broadcast",
        "stacked_tensor",
        "layer_ctx_fill_batch",
        "total_loop",
        "layer_other",
        "layers",
    ]
    row_by_tag: dict[str, tuple[int, float, float, float, float, float, float, float]] = {}
    for tag, vals in tag_vals.items():
        if not vals:
            continue
        row_by_tag[tag] = _row_stats(vals)

    def sort_key(tag: str) -> tuple[int, str]:
        if tag in order:
            return (order.index(tag), tag)
        if tag in child_set:
            return (order.index("total_loop") + 1, tag)
        return (len(order) + 1, tag)

    print_order: list[str] = []
    seen: set[str] = set()
    for tag in order:
        if tag in row_by_tag and tag not in seen:
            print_order.append(tag)
            seen.add(tag)
            if tag == "total_loop":
                for child in TOTAL_LOOP_CHILDREN:
                    if child in row_by_tag and child not in seen:
                        print_order.append(child)
                        seen.add(child)
    for tag in sorted(row_by_tag.keys(), key=sort_key):
        if tag not in seen:
            print_order.append(tag)
            seen.add(tag)

    field_w = 20
    print(title)
    hdr = (
        f"{'field':<{field_w}} {'cnt':>6} {'mean':>8} {'min':>8} {'max':>8} "
        f"{'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag in print_order:
        n, mean, vmin, vmax, p50, p90, p95, p99 = row_by_tag[tag]
        label = tag
        if tag in child_set:
            label = f"{TOTAL_LOOP_CHILD_INDENT}{tag}"
        print(
            f"{label:<{field_w}} {n:>6} {mean:>8.3f} {vmin:>8.3f} {vmax:>8.3f} "
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
    matched_ext, stats_ext = parse_log(
        args.log_path, by_worker=args.by_worker, line_re=LINE_EXT_RE)
    matched_reconcile, stats_reconcile = parse_log(
        args.log_path, by_worker=args.by_worker, line_re=LINE_RECONCILE_RE)
    print(f"File: {args.log_path}")
    print(f"Matched [DynamicKV][prepare_profile] lines: {matched}")
    print(f"Matched [DynamicKV][prepare_profile_ext] lines: {matched_ext}")
    print(f"Matched [DynamicKV][prepare_profile_reconcile] lines: "
          f"{matched_reconcile}")
    if matched == 0 and matched_ext == 0 and matched_reconcile == 0:
        print(
            "No matching lines. Ensure logs contain "
            "'[DynamicKV][prepare_profile]', '[prepare_profile_ext]', or "
            "'[prepare_profile_reconcile]' and VLLM_DYNKV_PROFILE_PREPARE=1 "
            "was enabled.",
        )
        return

    keys = sorted(
        set(stats.keys()) | set(stats_ext.keys()) | set(stats_reconcile.keys()))
    for key in keys:
        if args.by_worker and key:
            worker = key[0]
            title = f"== DynamicKV Prepare Profile | worker={worker} =="
        else:
            title = "== DynamicKV Prepare Profile =="
        if key in stats_reconcile:
            _print_table(title + " (reconcile)", stats_reconcile[key])
        if key in stats_ext:
            _print_table(title + " (ext)", stats_ext[key])
        if key in stats:
            _print_table(title + " (loop)", stats[key])


if __name__ == "__main__":
    main()
