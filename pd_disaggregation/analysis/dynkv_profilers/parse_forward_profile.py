#!/usr/bin/env python3
"""
从 vLLM-Ascend 日志中解析 ``[DynamicKV][forward_profile] ...`` 行，
统计 forward 各阶段耗时：ctx_setup, kv_setup, dynkv_pre, model, dynkv_post, total。

用法::

    python parse_forward_profile.py /path/to/decode.log
    python parse_forward_profile.py decode_off.log decode_on.log

环境变量见 README；PA 日志带 ``[forward_profile][pa]``，PIA eager 为 ``[forward_profile]`` + FIA 字段。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

LINE_RE = re.compile(
    r"\[DynamicKV\]\[forward_profile\](?:\[pa\])?\s+(.+)$")
FIELD_RE = re.compile(r"(\w+)=([\d.]+)(ms)?")
WORKER_RE = re.compile(r"\(([^\s()]+)\s+pid=\d+\)")

Mode = Literal["pa", "pia", "pia_fia"]

# kind: cpu 墙钟分解 | npu NPU Event | metric 非时间 | ortho 同一次 replay/算子的交叉表，勿与同级相加
FieldKind = Literal["cpu", "npu", "metric", "ortho"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    depth: int
    kind: FieldKind
    modes: frozenset[Mode]


def _specs(*entries: tuple[str, int, FieldKind, frozenset[Mode]]) -> list[FieldSpec]:
    return [FieldSpec(n, d, k, m) for n, d, k, m in entries]


# 公共前缀 + 分模式字段（顺序即打印顺序）
_FIELD_SPECS: list[FieldSpec] = _specs(
    ("dynkv", 0, "metric", frozenset({"pa", "pia", "pia_fia"})),
    ("ctx_setup", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    ("kv_setup", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    ("dynkv_pre", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    # --- PA graph decode ---
    ("model_cpu", 0, "cpu", frozenset({"pa"})),
    ("model_acl", 1, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    ("model_npu_ms", 1, "npu", frozenset({"pa"})),
    ("graph_npu_ms", 2, "ortho", frozenset({"pa"})),
    ("graph_replay_wall", 2, "ortho", frozenset({"pa"})),
    ("model_graph_replay", 2, "ortho", frozenset({"pa", "pia", "pia_fia"})),
    ("fwd_block_npu_ms", 0, "npu", frozenset({"pa"})),
    ("pa_kv_tokens_avg", 0, "metric", frozenset({"pa"})),
    # --- PIA eager（enforce-eager + FIA）---
    ("model", 0, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_core", 1, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_embed", 2, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_norm", 2, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_attn", 2, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_attn_op", 3, "cpu", frozenset({"pia", "pia_fia"})),
    ("fia_ms_total", 4, "ortho", frozenset({"pia_fia"})),
    ("fia_kv_tokens_avg", 4, "metric", frozenset({"pia_fia"})),
    ("pa_ms_total", 4, "ortho", frozenset({"pia", "pia_fia"})),
    ("pa_kv_tokens_avg", 4, "metric", frozenset({"pia", "pia_fia"})),
    ("model_mlp", 2, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_layer_rms", 2, "cpu", frozenset({"pia", "pia_fia"})),
    ("model_sp_pcp", 1, "cpu", frozenset({"pia", "pia_fia"})),
    ("dynkv_post", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    ("profile_cpu_total", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
    ("total", 0, "cpu", frozenset({"pa", "pia", "pia_fia"})),
)

FIELD_SPECS: list[FieldSpec] = list(_FIELD_SPECS)
_SPEC_BY_NAME: dict[str, FieldSpec] = {s.name: s for s in FIELD_SPECS}

_INDENT_STEP = 4
_FIELD_COL_WIDTH = 26

_MODE_HINTS: dict[Mode, list[str]] = {
    "pa": [
        "CPU 分解: profile_cpu_total ≈ ctx_setup + kv_setup + dynkv_pre + model_cpu + dynkv_post",
        "model_cpu ≈ model_acl + model() 墙钟；model() 墙钟可用 (model_cpu - model_acl) 估算",
        "model_npu_ms ⊇ graph_npu_ms（均包 self.model() 内 replay，勿与 model_acl 相加）",
        "graph_replay_wall 与 graph_npu_ms 量同一次 replay()（CPU vs NPU），勿相加",
        "fwd_block_npu_ms ≈ Profile execute forward（整段 forward context NPU）",
    ],
    "pia": [
        "CPU 分解: profile_cpu_total ≈ ctx_setup + kv_setup + dynkv_pre + model + dynkv_post",
        "model ≈ model_core + model_sp_pcp；model_core ≈ embed + norm + attn + mlp + layer_rms（hook 累计）",
        "PIA eager 无 FULL graph：model_acl / graph_* 通常为 0 或缺失",
    ],
    "pia_fia": [
        "CPU 分解同 PIA；model_attn_op 为 Attention 算子 perf_counter 墙钟",
        "fia_ms_total 为 32 层 FIA NPU Event 合计，与 model_attn_op 交叉验证，勿相加",
        "fia_kv_tokens_avg 为 metadata 统计（非时间）",
    ],
}


def _parse_fields(rest: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in FIELD_RE.finditer(rest):
        key = m.group(1).strip()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        out[key] = val
    if "profile_cpu_total" not in out and "total" in out:
        out["profile_cpu_total"] = out["total"]
    if "model_cpu" not in out and "model" in out:
        out["model_cpu"] = out["model"]
    if "graph_replay_wall" not in out and "model_graph_replay" in out:
        out["graph_replay_wall"] = out["model_graph_replay"]
    return out


def _detect_mode(fields: dict[str, float], line: str) -> Mode:
    if "[forward_profile][pa]" in line:
        return "pa"
    if "model_cpu" in fields and "model" not in fields:
        return "pa"
    if "fia_ms_total" in fields:
        return "pia_fia"
    if "model" in fields and "model_core" in fields:
        return "pia"
    if "fwd_block_npu_ms" in fields and "graph_npu_ms" in fields:
        return "pa"
    return "pia"


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


def _ordered_fields(
    tag_vals: dict[str, list[float]],
    mode: Mode,
) -> list[tuple[str, int, FieldKind]]:
    ordered: list[tuple[str, int, FieldKind]] = []
    for spec in FIELD_SPECS:
        if mode not in spec.modes or spec.name not in tag_vals:
            continue
        ordered.append((spec.name, spec.depth, spec.kind))
    for name in sorted(tag_vals):
        if name not in _SPEC_BY_NAME:
            ordered.append((name, 0, "cpu"))
    return ordered


def _field_display(name: str, depth: int, kind: FieldKind) -> str:
    label = name
    if kind == "ortho":
        label = f"{name}*"  # 星号在脚注说明
    elif kind == "metric":
        label = f"{name}#"
    elif kind == "npu" and depth == 0:
        label = f"{name}†"
    if depth == 0:
        return label
    return f"{' ' * (_INDENT_STEP * depth)}{label}"


def parse_log(
    path: str,
    *,
    by_worker: bool,
) -> tuple[int, dict[tuple[str, ...], dict[str, list[float]]], dict[tuple[str, ...], Mode]]:
    stats: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    modes: dict[tuple[str, ...], Mode] = {}
    matched = 0
    status_printed = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not status_printed and "[DynamicKV][profile_status]" in line:
                print("== DynamicKV profile_status (from log) ==")
                print(line.strip())
                print()
                status_printed = True
            m = LINE_RE.search(line)
            if not m:
                continue
            fields = _parse_fields(m.group(1))
            if not fields:
                continue
            matched += 1
            mode = _detect_mode(fields, line)
            key: tuple[str, ...] = (mode,)
            if by_worker:
                key = (_worker_bucket(line), mode)
            modes[key] = mode
            bucket = stats[key]
            for fname, fval in fields.items():
                bucket[fname].append(fval)
    return matched, stats, modes


def _print_table(
    title: str,
    tag_vals: dict[str, list[float]],
    mode: Mode,
) -> None:
    rows: list[tuple[str, str, int, FieldKind, float, float, float, float, float, float, float]] = []
    ordered = _ordered_fields(tag_vals, mode)

    for name, depth, kind in ordered:
        vals = tag_vals.get(name)
        if not vals:
            continue
        n = len(vals)
        srt = sorted(vals)
        mean = sum(vals) / n
        vmin = srt[0]
        vmax = srt[-1]
        p50 = _percentile_linear(srt, 50.0)
        p90 = _percentile_linear(srt, 90.0)
        p95 = _percentile_linear(srt, 95.0)
        p99 = _percentile_linear(srt, 99.0)
        rows.append((
            name,
            _field_display(name, depth, kind),
            n,
            kind,
            mean,
            vmin,
            vmax,
            p50,
            p90,
            p95,
            p99,
        ))

    mode_label = {"pa": "PA graph", "pia": "PIA eager", "pia_fia": "PIA eager + FIA"}[mode]
    print(title)
    print(f"  mode={mode_label}  *ortho=勿与同级相加  #metric=非时间  †顶层NPU轴")
    for hint in _MODE_HINTS[mode]:
        print(f"  {hint}")
    col_w = max(_FIELD_COL_WIDTH, max((len(r[1]) for r in rows), default=0))
    hdr = (
        f"{'field':<{col_w}} {'cnt':>6} {'mean':>8} {'min':>8} "
        f"{'max':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        _name, label, n, _kind, mean, vmin, vmax, p50, p90, p95, p99 = row
        print(
            f"{label:<{col_w}} {n:>6} {mean:>8.3f} {vmin:>8.3f} "
            f"{vmax:>8.3f} {p50:>8.3f} {p90:>8.3f} {p95:>8.3f} {p99:>8.3f}",
        )
    print()


def _print_file_report(
    path: str,
    *,
    by_worker: bool,
    label: str | None = None,
) -> tuple[int, dict[tuple[str, ...], dict[str, list[float]]]]:
    matched, stats, modes = parse_log(path, by_worker=by_worker)
    header = label or path
    print(f"File: {header}")
    print(f"Matched [DynamicKV][forward_profile] lines: {matched}")
    if matched == 0:
        print(
            "  (no lines — need VLLM_DYNKV_PROFILE_FORWARD=1 on decode worker)",
        )
        print()
        return matched, stats
    keys = sorted(stats.keys())
    for key in keys:
        mode = modes[key]
        suffix = ""
        if by_worker and len(key) >= 2:
            suffix = f" | worker={key[0]} | {mode}"
        elif len(key) >= 1:
            suffix = f" | {mode}"
        title = f"== Forward Profile | {header}{suffix} =="
        _print_table(title, stats[key], mode)
    return matched, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse [DynamicKV][forward_profile] lines from vllm-ascend logs.",
    )
    ap.add_argument(
        "log_paths",
        nargs="+",
        help="One or more log files (e.g. decode_off.log decode_on.log).",
    )
    ap.add_argument(
        "--by-worker",
        action="store_true",
        help="Split stats by Ray-style worker prefix.",
    )
    args = ap.parse_args()

    any_matched = False
    for i, path in enumerate(args.log_paths):
        if len(args.log_paths) > 1:
            label = f"{path} [{i + 1}/{len(args.log_paths)}]"
        else:
            label = path
        matched, _ = _print_file_report(
            path,
            by_worker=args.by_worker,
            label=label,
        )
        any_matched = any_matched or matched > 0

    if not any_matched:
        sys.exit(1)


if __name__ == "__main__":
    main()
