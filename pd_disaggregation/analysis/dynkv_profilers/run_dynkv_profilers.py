#!/usr/bin/env python3
"""
一次性运行本目录下四个 DynamicKV 日志解析脚本。

顺序：
  1. parse_profile_execute_duration.py
  2. parse_dynkv_prepare_profile.py
  3. parse_forward_profile.py
  4. parse_model_acl_profile.py

用法::

    python run_dynkv_profilers.py /path/to/decode.log
    python run_dynkv_profilers.py decode.log --by-worker
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent

_SCRIPTS = (
    "parse_profile_execute_duration.py",
    "parse_dynkv_prepare_profile.py",
    "parse_forward_profile.py",
    "parse_model_acl_profile.py",
)

_BANNER = "=" * 78


def _run(
    script: str,
    args: list[str],
    *,
    title: str | None = None,
) -> int:
    path = _DIR / script
    if not path.is_file():
        print(f"error: missing {path}", file=sys.stderr)
        return 2
    if title:
        print()
        print(_BANNER)
        print(title)
        print(_BANNER)
        print()
    proc = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=str(_DIR),
    )
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all four dynkv_profilers parsers on one decode log.",
    )
    ap.add_argument(
        "log_path",
        help="Path to decode log (plain text).",
    )
    ap.add_argument(
        "--by-worker",
        action="store_true",
        help="Pass --by-worker to each parser that supports it.",
    )
    args = ap.parse_args()

    log = str(Path(args.log_path).resolve())
    if not Path(log).is_file():
        print(f"error: log not found: {log}", file=sys.stderr)
        sys.exit(2)

    common: list[str] = [log]
    if args.by_worker:
        common.append("--by-worker")

    failed = 0
    for i, script in enumerate(_SCRIPTS, start=1):
        rc = _run(script, common, title=f"{i}/4 {script}")
        failed += rc != 0

    print()
    print(_BANNER)
    if failed:
        print(f"Done with {failed} non-zero exit(s). Check sections above.")
        sys.exit(1)
    print("All four parsers finished successfully.")
    print(_BANNER)


if __name__ == "__main__":
    main()
