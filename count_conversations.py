#!/usr/bin/env python3
"""
统计 JSON 中每个对象的 conversations 数组长度分布。
用法: python count_conversations.py <输入.json>
"""

import argparse
import json
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="统计每个对象 conversations 数组大小的分布")
    parser.add_argument("input", help="输入 JSON 文件路径")
    args = parser.parse_args()

    print(f"正在读取: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误: 输入 JSON 的根节点必须是数组", file=sys.stderr)
        sys.exit(1)

    # 统计每个对象的 conversations 长度
    lengths = []
    for obj in data:
        if not isinstance(obj, dict):
            lengths.append(0)
            continue
        conv = obj.get("conversations", [])
        lengths.append(len(conv) if isinstance(conv, list) else 0)

    counter = Counter(lengths)
    total = len(data)

    lines = [
        f"总对象数: {total}",
        "",
        "conversations 数组大小分布:",
        "-" * 40,
    ]
    for size in sorted(counter.keys()):
        count = counter[size]
        pct = 100.0 * count / total if total else 0
        lines.append(f"  数组大小为 {size} 的对象: {count} 个 ({pct:.2f}%)")
    lines.append("-" * 40)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
