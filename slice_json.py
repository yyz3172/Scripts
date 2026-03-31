#!/usr/bin/env python3
"""
从 JSON 文件中取数组的前 N 个元素并输出到新文件。
用法: python slice_json.py <输入.json> [输出.json] [--num N]
"""

import argparse
import json
import sys


def _is_valid_item(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    conv = obj.get("conversations")
    if not isinstance(conv, list):
        return False
    n = len(conv)
    if n == 0 or (n % 2) != 0:
        return False
    # if n < 28 or n > 60:
    #     return False
    # 要求每对消息: (human, gpt)
    for i in range(0, n, 2):
        a = conv[i]
        b = conv[i + 1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if a.get("from") != "human":
            print("human not found")
            return False
        if b.get("from") != "gpt":
            print("gpt not found")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="取 JSON 数组前 N 个元素")
    parser.add_argument("input", help="输入 JSON 文件路径")
    parser.add_argument("output", nargs="?", default="output_slice.json", help="输出 JSON 文件路径（默认: output_slice.json）")
    parser.add_argument("-n", "--num", type=int, default=100, help="取前 N 个元素（默认: 100）")
    args = parser.parse_args()

    print(f"正在读取: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误: 输入 JSON 的根节点必须是数组", file=sys.stderr)
        sys.exit(1)

    total = len(data)
    sliced = []
    traversed = 0
    for x in data:
        if len(sliced) >= args.num:
            break
        traversed += 1
        if _is_valid_item(x):
            sliced.append(x)
    print(f"原数组长度: {total}, 遍历 {traversed} 个后取到 {len(sliced)} 个符合条件者")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sliced, f, ensure_ascii=False, indent=2)

    print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
