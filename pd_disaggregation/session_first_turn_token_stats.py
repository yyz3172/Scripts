#!/usr/bin/env python3
"""统计每个 session 前 N 轮 messages 的 token 长度，并按 session 求和后汇总统计。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from jiuwen_affinity import load_session_data

MAX_TURNS_LIMIT = 10


def _extract_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    messages = data.get("message")
    if messages is None:
        messages = data.get("messages", [])
    if not isinstance(messages, list):
        return []
    return messages


def _extract_tools(data: dict[str, Any]) -> list[Any] | None:
    tools = data.get("tools")
    if tools is None:
        return None
    if isinstance(tools, list):
        return tools
    return None


def _char_estimate_token_count(messages: list[dict[str, Any]], tools: list[Any] | None) -> int:
    """无 tokenizer 时的粗估：字符数 / 4。"""
    char_count = 0

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            char_count += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    char_count += len(part.get("text", ""))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            char_count += len(json.dumps(tool_calls, ensure_ascii=False))

    if tools:
        char_count += len(json.dumps(tools, ensure_ascii=False))

    return max(1, char_count // 4) if char_count > 0 else 0


def _build_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "使用 huggingface tokenizer 需要安装 transformers，"
            "或改用 --method char_estimate"
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_turn_tokens(
    messages: list[dict[str, Any]],
    tools: list[Any] | None,
    *,
    tokenizer=None,
    method: str = "huggingface",
) -> int:
    if not messages and not tools:
        return 0

    if method == "char_estimate":
        return _char_estimate_token_count(messages, tools)

    if tokenizer is None:
        raise ValueError("method=huggingface 时必须提供 tokenizer")

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if tools:
        template_kwargs["tools"] = tools

    try:
        token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("tools", None)
        token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)

    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", [])
    return len(token_ids)


def _parse_max_turns(value: str) -> int:
    turns = int(value)
    if turns < 1 or turns > MAX_TURNS_LIMIT:
        raise argparse.ArgumentTypeError(f"--max-turns 取值范围为 1~{MAX_TURNS_LIMIT}")
    return turns


def get_session_turns(
    session_data: OrderedDict,
    max_turns: int,
) -> list[tuple[str, list[tuple[int, Path, dict[str, Any]]]]]:
    """返回每个 session 的前 max_turns 轮。"""
    session_turns_list = []
    for session_id, turns in session_data.items():
        if not turns:
            continue
        selected_turns = [
            (turn_num, file_path, data)
            for turn_num, file_path, data in turns[:max_turns]
        ]
        session_turns_list.append((session_id, selected_turns))
    return session_turns_list


def analyze_session_turn_tokens(
    data_folder: str,
    *,
    tokenizer_path: str | None = None,
    method: str = "huggingface",
    max_sessions: int | None = None,
    max_turns: int = 1,
) -> dict[str, Any]:
    session_data = load_session_data(data_folder)

    if max_sessions:
        session_ids = list(session_data.keys())[:max_sessions]
        session_data = OrderedDict((sid, session_data[sid]) for sid in session_ids)

    tokenizer = None
    if method == "huggingface":
        if not tokenizer_path:
            raise ValueError("method=huggingface 时必须指定 --tokenizer")
        tokenizer = _build_tokenizer(tokenizer_path)

    session_results = []
    session_token_sums: list[int] = []

    for session_id, turns in get_session_turns(session_data, max_turns):
        turn_results = []
        session_sum = 0
        for turn_num, _file_path, data in turns:
            messages = _extract_messages(data)
            tools = _extract_tools(data)
            token_count = count_turn_tokens(
                messages,
                tools,
                tokenizer=tokenizer,
                method=method,
            )
            turn_results.append(
                {
                    "turn_num": turn_num,
                    "token_count": token_count,
                }
            )
            session_sum += token_count

        session_token_sums.append(session_sum)
        session_results.append(
            {
                "session_id": session_id,
                "turns": turn_results,
                "token_count_sum": session_sum,
            }
        )

    summary: dict[str, Any] = {
        "data_folder": data_folder,
        "method": method,
        "tokenizer": tokenizer_path,
        "session_count": len(session_results),
        "max_turns": max_turns,
    }

    if session_token_sums:
        sorted_sums = sorted(session_token_sums)
        summary.update(
            {
                "token_count_total": sum(session_token_sums),
                "token_count_avg": statistics.mean(session_token_sums),
                "token_count_min": min(session_token_sums),
                "token_count_max": max(session_token_sums),
                "token_count_p50": sorted_sums[len(sorted_sums) // 2],
                "token_count_p90": sorted_sums[int(len(sorted_sums) * 0.9)],
                "token_count_p95": sorted_sums[int(len(sorted_sums) * 0.95)],
                "token_count_p99": sorted_sums[int(len(sorted_sums) * 0.99)],
            }
        )
    else:
        summary["token_count_avg"] = 0

    return {
        "summary": summary,
        "sessions": session_results,
    }


def _print_session_tokens(item: dict[str, Any], index: int) -> None:
    print(f"[{index}] {item['token_count_sum']}")


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    max_turns = summary["max_turns"]
    print(f"\n{'=' * 80}")
    print(f"Session 前 {max_turns} 轮 Token 统计")
    print(f"{'=' * 80}")
    print(f"数据目录: {summary['data_folder']}")
    print(f"统计方式: {summary['method']}")
    if summary.get("tokenizer"):
        print(f"Tokenizer: {summary['tokenizer']}")
    print(f"Session 数量: {summary['session_count']}")
    print(f"统计轮次: 每个 session 前 {max_turns} 轮求和")

    if summary["session_count"] == 0:
        print("未找到可统计的数据。")
        print(f"{'=' * 80}\n")
        return

    print("\nToken 之和统计（按 session 维度）:")
    print(f"  平均值: {summary['token_count_avg']:.1f}")
    print(f"  最小值: {summary['token_count_min']}")
    print(f"  最大值: {summary['token_count_max']}")
    print(f"  P50: {summary['token_count_p50']}")
    print(f"  P90: {summary['token_count_p90']}")
    print(f"  P95: {summary['token_count_p95']}")
    print(f"  P99: {summary['token_count_p99']}")

    print(f"\n{'=' * 80}")
    print(f"每个 Session 前 {max_turns} 轮 Token 之和")
    print(f"{'=' * 80}")
    for index, item in enumerate(report["sessions"], start=1):
        _print_session_tokens(item, index)

    print(f"\n{'=' * 80}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计每个 session 前 N 轮 messages 的 token 长度"
    )
    parser.add_argument(
        "data_folder",
        help="数据集根目录，结构与 jiuwen_affinity.load_session_data 一致",
    )
    parser.add_argument(
        "--tokenizer",
        help="HuggingFace tokenizer 路径或模型名（method=huggingface 时必填）",
    )
    parser.add_argument(
        "--method",
        choices=("huggingface", "char_estimate"),
        default="huggingface",
        help="token 统计方式，默认 huggingface",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="最多统计多少个 session",
    )
    parser.add_argument(
        "--max-turns",
        type=_parse_max_turns,
        default=1,
        help=f"每个 session 统计前多少个 turn，范围 1~{MAX_TURNS_LIMIT}，默认 1",
    )
    parser.add_argument(
        "--output",
        help="结果 JSON 输出路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_folder = args.data_folder
    if not Path(data_folder).is_dir():
        print(f"数据目录不存在: {data_folder}", file=sys.stderr)
        return 1

    try:
        report = analyze_session_turn_tokens(
            data_folder,
            tokenizer_path=args.tokenizer,
            method=args.method,
            max_sessions=args.max_sessions,
            max_turns=args.max_turns,
        )
    except Exception as exc:
        print(f"统计失败: {exc}", file=sys.stderr)
        return 1

    _print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
