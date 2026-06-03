#!/usr/bin/env python3
"""分析 vLLM Ascend torch_npu profiler 采集结果。

等价于：
  from torch_npu.profiler.profiler import analyse
  analyse("./vllm_profile/localhost.localdomain_*_ascend_pt/")

用法：
  # 传入 trace 目录本身（目录名须以 ascend_pt 结尾）
  python analyse_vllm_profile.py ./vllm_profile/prefill/localhost.localdomain_123_ascend_pt

  # 传入父目录，自动分析其下所有以 ascend_pt 结尾的子目录
  python analyse_vllm_profile.py ./vllm_profile/prefill
"""

from __future__ import annotations

import argparse
import os
import sys


def _ends_with_ascend_pt(name: str) -> bool:
    return name.endswith("ascend_pt")


def _resolve_targets(raw_path: str) -> list[str]:
    path = os.path.abspath(os.path.expanduser(raw_path))
    if not os.path.isdir(path):
        raise ValueError(f"目录不存在: {path}")

    if _ends_with_ascend_pt(os.path.basename(path.rstrip(os.sep))):
        return [path]

    children = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name)) and _ends_with_ascend_pt(name)
    )
    if not children:
        raise ValueError(f"未找到以 ascend_pt 结尾的子目录: {path}")
    return children


def main() -> None:
    parser = argparse.ArgumentParser(
        description="调用 torch_npu.profiler.analyse 分析 vLLM profile trace"
    )
    parser.add_argument(
        "profile_dir",
        help="trace 目录（须以 ascend_pt 结尾），或其父目录（分析其下所有 *ascend_pt 子目录）",
    )
    args = parser.parse_args()

    try:
        from torch_npu.profiler.profiler import analyse
    except ImportError as exc:
        print(
            "无法导入 torch_npu.profiler.profiler.analyse，"
            "请在已安装 torch_npu 的 Ascend 环境中运行。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        targets = _resolve_targets(args.profile_dir)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    exit_code = 0
    for target in targets:
        print(f"Analysing: {target}", file=sys.stderr)
        try:
            analyse(target)
        except Exception as exc:
            exit_code = 1
            print(f"分析失败: {target}\n  {exc}", file=sys.stderr)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
