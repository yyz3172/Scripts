#!/usr/bin/env python3
"""一次运行本目录下四个 DynamicKV 日志解析脚本。"""

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


def _run(script: str, args: list[str]) -> int:
    path = _DIR / script
    if not path.is_file():
        print(f"error: missing {path}", file=sys.stderr)
        return 2
    print(f"[Running {script} with args: {args}]")
    proc = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=str(_DIR),
    )
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all four dynkv_profilers parsers on one decode log.",
    )
    ap.add_argument("log_path", help="Path to decode log (plain text).")
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

    run_args: list[str] = [log]
    if args.by_worker:
        run_args.append("--by-worker")

    failed = sum(_run(script, run_args) != 0 for script in _SCRIPTS)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
