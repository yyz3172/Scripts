#!/usr/bin/env python3
"""
PD 分离模式 LongBench 批量压测编排。

每轮：启动 prefill → 等待就绪 → 启动 decode → 可选代理 → 运行 LongBench `pred.py` → 回收进程。

说明：
- PD 栈启停由同目录 `pd_service_ctl.PdServiceCtl` 完成。
- LongBench 脚本为 `code/LongBench/pred.py`，通过环境变量 `VLLM_PORT`/`OPENAI_BASE_URL` 指向 vLLM(OpenAI兼容)服务。
- 本脚本默认“扫 n_proc”（并发进程数），用法上仍沿用 `run_sharegpt_sweep.py` 的 batch sweep 形态。
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pd_service_ctl as pdctl
from pd_service_ctl import PdRuntimeConfig, PdServiceCtl

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent.parent

VLLM_VENV = Path("/root/autodl-tmp/py_venv/vllm")
LOG_ROOT = PKG_DIR / "logs"
LOG_DIR_DEFAULT = LOG_ROOT / "longbench"
LONGBENCH_DIR_DEFAULT = REPO_ROOT / "LongBench"
PRED_PY_DEFAULT = LONGBENCH_DIR_DEFAULT / "pred.py"


def pd_mode_to_model_name(pd_mode: str) -> str:
    """由 pd_mode 推导模型名：取第一个 `/` 之前的部分作为模型名。"""
    mode = (pd_mode or "").strip().replace("\\", "/")
    if not mode:
        raise RuntimeError("pd_mode 为空，无法推导 model 名")
    if mode.startswith("./"):
        mode = mode[2:]
    mode = mode.lstrip("/")
    model_part = mode.split("/", 1)[0]
    if not model_part:
        raise RuntimeError(f"pd_mode 不包含有效模型名: {pd_mode!r}")
    return model_part


@dataclass
class SweepConfig:
    log_dir: Path = LOG_DIR_DEFAULT
    pd_mode: str = pdctl.PD_MODE
    # sweep 的“batch”在 LongBench 里对应 pred.py 的 n_proc（并发进程数）
    batch_sizes: List[int] = field(default_factory=lambda: [16, 24, 32])
    vllm_port: int = 9010
    proxy_port: int = 8000
    prefill_port: int = 9000
    ready_timeout_s: int = 300
    nic_name: str = pdctl.NIC_NAME
    local_ip: str = pdctl.LOCAL_IP
    vllm_venv: Path = VLLM_VENV
    longbench_dir: Path = LONGBENCH_DIR_DEFAULT
    pred_py: Path = PRED_PY_DEFAULT
    round_cooldown_s: int = 5

    # LongBench pred.py 关键参数
    data_path: str = ""
    rag: int = 0
    cot: bool = False
    no_context: bool = False
    measure_latency: bool = True

    @property
    def topo_dir(self) -> Path:
        rel = self.pd_mode.strip("/").replace("\\", "/")
        return PKG_DIR / rel

    @property
    def use_proxy(self) -> bool:
        return pdctl.pd_proxy_path(self.topo_dir) is not None


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_sweep(sweep_log: Path, msg: str) -> None:
    line = f"{ts()} [sweep] {msg}"
    print(line)
    sweep_log.parent.mkdir(parents=True, exist_ok=True)
    with sweep_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _sweep_log_fn(sweep_log: Path):
    return lambda msg: log_sweep(sweep_log, msg)


def _pd_rt(cfg: SweepConfig) -> PdRuntimeConfig:
    return PdRuntimeConfig(
        topo_dir=cfg.topo_dir,
        vllm_venv=cfg.vllm_venv,
        pd_mode=cfg.pd_mode,
        prefill_port=cfg.prefill_port,
        vllm_port=cfg.vllm_port,
        proxy_sleep_s=getattr(cfg, "proxy_sleep_s", 10),
        ready_timeout_s=cfg.ready_timeout_s,
        nic_name=cfg.nic_name,
        local_ip=cfg.local_ip,
    )


def parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def run_longbench_pred(
    cfg: SweepConfig,
    batch_size: int,
    bench_port: int,
    sweep_log: Path,
) -> int:
    round_dir = cfg.log_dir / f"batch_{batch_size}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.pred_py.is_file():
        log_sweep(sweep_log, f"ERROR: pred.py 不存在: {cfg.pred_py}")
        return 1

    save_dir = round_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_log = round_dir / "pred.log"

    model = pd_mode_to_model_name(cfg.pd_mode)
    args = [
        sys.executable,
        str(cfg.pred_py),
        "--data_path",
        cfg.data_path,
        "--save_dir",
        str(save_dir),
        "--out_file",
        "long_bench_output.jsonl",
        "--model",
        model,
        "--n_proc",
        str(batch_size),
        "--rag",
        str(cfg.rag),
    ]
    if cfg.cot:
        args.append("--cot")
    if cfg.no_context:
        args.append("--no_context")
    if cfg.measure_latency:
        args.append("--measure_latency")

    env = os.environ.copy()
    env["VLLM_PORT"] = str(bench_port)

    inner = f"""
set -euo pipefail
cd "{cfg.longbench_dir}"
{" ".join([subprocess.list2cmdline([a]) if " " in a else a for a in args])} >> "{pred_log}" 2>&1
"""
    log_sweep(
        sweep_log,
        f"Running LongBench pred.py (BATCH_SIZE={batch_size}, port={bench_port}, model={model})...",
    )
    r = subprocess.run(["/bin/bash", "-c", inner], cwd=str(cfg.longbench_dir), env=env)
    return r.returncode


def run_sweep(cfg: SweepConfig) -> int:
    sweep_log = cfg.log_dir / "sweep.log"
    use_proxy_sweep = cfg.use_proxy
    logf = _sweep_log_fn(sweep_log)
    pd_ctl = PdServiceCtl(_pd_rt(cfg), log=logf)

    def on_signal(signum: int, _frame) -> None:
        log_sweep(sweep_log, f"Interrupted (signal {signum}), cleaning up...")
        pd_ctl.stop(use_proxy=use_proxy_sweep)
        sys.exit(1)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log_sweep(
        sweep_log,
        "Sweep start (PD mode): "
        f"LOG_DIR={cfg.log_dir} TOPO_DIR={cfg.topo_dir} "
        f"BATCH_SIZES={cfg.batch_sizes} USE_PROXY={use_proxy_sweep} "
        f"NIC_NAME={cfg.nic_name} LOCAL_IP={cfg.local_ip} "
        f"MODEL={pd_mode_to_model_name(cfg.pd_mode)} DATA_PATH={cfg.data_path}",
    )

    for batch_size in cfg.batch_sizes:
        log_sweep(sweep_log, f"========== Round: BATCH_SIZE={batch_size} ==========")

        round_log_dir = cfg.log_dir / f"batch_{batch_size}"
        os.environ["LOG_DIR"] = str(round_log_dir)

        log_sweep(
            sweep_log,
            f"PdServiceCtl.start_stack (log_dir={round_log_dir}, with_proxy={use_proxy_sweep})...",
        )
        stack_rc = pd_ctl.start_stack(
            round_log_dir,
            with_proxy=use_proxy_sweep,
            wait_ready=True,
        )
        if stack_rc != 0:
            log_sweep(
                sweep_log,
                f"ERROR: PD 栈启动失败，跳过 BATCH_SIZE={batch_size}（ctl 已尝试回收）。",
            )
            continue

        bench_port = cfg.proxy_port if use_proxy_sweep else cfg.vllm_port
        rc = run_longbench_pred(cfg, batch_size, bench_port, sweep_log)
        if rc == 0:
            log_sweep(sweep_log, f"LongBench finished: BATCH_SIZE={batch_size}")
        else:
            log_sweep(
                sweep_log,
                f"WARNING: LongBench exited with error for BATCH_SIZE={batch_size}, continuing...",
            )

        pd_ctl.stop(use_proxy=use_proxy_sweep)
        log_sweep(
            sweep_log,
            f"vllm (PD) stopped. Waiting {cfg.round_cooldown_s}s before next round...",
        )
        time.sleep(cfg.round_cooldown_s)

    log_sweep(sweep_log, f"All rounds completed. Logs: {cfg.log_dir}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PD 分离 LongBench 批量压测编排")
    p.add_argument(
        "--log_dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help=f"本趟日志/输出根目录（sweep.log、batch_* 等；默认 {LOG_DIR_DEFAULT}）",
    )
    p.add_argument(
        "--vllm_venv",
        type=Path,
        default=None,
        help=f"vLLM 虚拟环境目录（默认未指定时用 {VLLM_VENV}）",
    )
    p.add_argument(
        "--nic_name",
        default=None,
        help=f"网卡名（默认未指定时用 pd_service_ctl 的 NIC_NAME，当前为 {pdctl.NIC_NAME!r}）",
    )
    p.add_argument(
        "--local_ip",
        default=None,
        help=f"本机可达 IP（默认未指定时用 pd_service_ctl 的 LOCAL_IP，当前为 {pdctl.LOCAL_IP!r}）",
    )
    p.add_argument(
        "--pd_mode",
        default=None,
        help=f"相对本目录的拓扑子路径（默认未指定时用 pd_service_ctl 的 PD_MODE，当前为 {pdctl.PD_MODE!r}）",
    )
    p.add_argument(
        "--batch_sizes",
        default=None,
        help="逗号分隔的 batch 列表（映射到 pred.py 的 --n_proc 并发进程数），如 \"8,16,32\"；默认 \"16\"",
    )
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="LongBench-v2 数据路径：data.json/.jsonl 文件或包含 data.json 的目录（同 pred.py 约定）",
    )
    p.add_argument("--rag", type=int, default=0, help="传给 pred.py 的 --rag")
    p.add_argument("--cot", action="store_true", help="传给 pred.py 的 --cot")
    p.add_argument("--no_context", action="store_true", help="传给 pred.py 的 --no_context")
    p.add_argument("--longbench_dir", type=Path, default=LONGBENCH_DIR_DEFAULT)
    p.add_argument("--pred_py", type=Path, default=PRED_PY_DEFAULT)
    args = p.parse_args(list(argv) if argv is not None else None)

    vllm_venv = args.vllm_venv.resolve() if args.vllm_venv is not None else VLLM_VENV
    nic_name = args.nic_name if args.nic_name is not None else pdctl.NIC_NAME
    local_ip = args.local_ip if args.local_ip is not None else pdctl.LOCAL_IP
    pd_mode = args.pd_mode if args.pd_mode is not None else pdctl.PD_MODE
    batch_sizes_s = args.batch_sizes if args.batch_sizes is not None else "16"

    cfg = SweepConfig(
        log_dir=args.log_dir.resolve(),
        pd_mode=pd_mode,
        batch_sizes=parse_int_list(batch_sizes_s),
        nic_name=nic_name,
        local_ip=local_ip,
        vllm_venv=vllm_venv,
        longbench_dir=args.longbench_dir.resolve(),
        pred_py=args.pred_py.resolve(),
        data_path=args.data_path,
        rag=args.rag,
        cot=args.cot,
        no_context=args.no_context,
    )
    return run_sweep(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

