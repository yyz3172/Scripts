#!/usr/bin/env python3
"""
统计目录下所有 jsonl 文件的字段均值（按文件分别统计）。

输入：
  python jsonl_mean_stats.py <input_path>

行为：
- 递归遍历 input_path 下所有 *.jsonl
- 每个 jsonl 文件按行读取 JSON
- 对每个文件分别统计以下字段的均值：
    length, input_tokens, output_tokens, ttft_ms, e2e_ms, tpot_ms

输出：
- 控制台打印表格（每个文件一行）
- 可选 --output 写入 JSON 汇总
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


FIELDS: Tuple[str, ...] = (
    "length",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "e2e_ms",
    "tpot_ms",
)


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return False
        return True
    return False


def iter_jsonl_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


@dataclass
class FileStats:
    path: str
    n_lines: int
    n_json_ok: int
    sum_by_field: Dict[str, float]
    cnt_by_field: Dict[str, int]

    def mean(self, field: str) -> Optional[float]:
        c = self.cnt_by_field.get(field, 0)
        if c == 0:
            return None
        return self.sum_by_field.get(field, 0.0) / c

    def to_dict(self) -> Dict[str, Any]:
        means = {k: self.mean(k) for k in FIELDS}
        return {
            "path": self.path,
            "n_lines": self.n_lines,
            "n_json_ok": self.n_json_ok,
            "means": means,
            "counts": {k: self.cnt_by_field.get(k, 0) for k in FIELDS},
        }


def summarize_jsonl(path: str) -> FileStats:
    sum_by_field: Dict[str, float] = {k: 0.0 for k in FIELDS}
    cnt_by_field: Dict[str, int] = {k: 0 for k in FIELDS}
    n_lines = 0
    n_json_ok = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            n_json_ok += 1
            for k in FIELDS:
                v = obj.get(k, None)
                if _is_number(v):
                    sum_by_field[k] += float(v)
                    cnt_by_field[k] += 1

    return FileStats(
        path=path,
        n_lines=n_lines,
        n_json_ok=n_json_ok,
        sum_by_field=sum_by_field,
        cnt_by_field=cnt_by_field,
    )


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if abs(x) >= 1000 or x == 0:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _print_table(stats: Iterable[FileStats], root: str) -> None:
    rows = list(stats)
    if not rows:
        print("未找到 jsonl 文件。")
        return

    rel_paths = [os.path.relpath(r.path, root) for r in rows]
    path_w = max(len("file"), max(len(p) for p in rel_paths))
    cols = ["file", "n", *FIELDS]
    header = (
        f"{cols[0]:<{path_w}}  {cols[1]:>6}  "
        + "  ".join(f"{c:>14}" for c in cols[2:])
    )
    print(header)
    print("-" * len(header))

    for r, rp in zip(rows, rel_paths):
        vals = [_fmt(r.mean(k)) for k in FIELDS]
        print(f"{rp:<{path_w}}  {r.n_json_ok:>6}  " + "  ".join(f"{v:>14}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="统计目录下所有 jsonl 文件的字段均值（按文件分别统计）")
    ap.add_argument("input_path", help="包含 jsonl 的目录（会递归遍历）")
    ap.add_argument("--output", "-o", default=None, help="可选：写出汇总 JSON 文件路径")
    args = ap.parse_args()

    root = os.path.abspath(args.input_path)
    files = iter_jsonl_files(root)
    stats = [summarize_jsonl(p) for p in files]

    _print_table(stats, root=root)

    if args.output:
        payload = {
            "root": root,
            "files": [s.to_dict() for s in stats],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

