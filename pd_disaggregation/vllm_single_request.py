#!/usr/bin/env python3
"""向 vLLM（OpenAI 兼容）发一条请求，用于单点连通性与输出检查。

默认走 POST {base_url}/chat/completions；也可用 --completions 走 {base_url}/completions。

-p / --prompt 支持内置模板：写成 @预设名（如 -p @short），自由文本不要加 @。
  预设列表见 --list-prompts。

环境变量（可被命令行覆盖）：
  OPENAI_API_KEY    默认 EMPTY（无鉴权时 vLLM 常用占位）

默认使用流式请求（stream=true）在客户端统计：
  TTFT_ms  首 Token 延迟（Time To First Token；口语/笔误里偶见「FFTF」）
  E2E_ms   从发起请求到收完流的端到端时间
  TPOT_ms  平均每输出 Token 耗时（毫秒/Token），用 (E2E_ms - TTFT_ms) / completion_tokens
           与 LongBench/pred_http.py 口径一致；completion_tokens 来自流末 usage（需 stream_options.include_usage）。
  加 --no-stream 则走单次非流式 JSON，仅打印 E2E_ms，TTFT/TPOT 为 null。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

PresetValue = str | Callable[[], str]

# 长上下文预设 @long20k：主体目标字数（多句拼接，减少「重复句」被分词极度压缩的情况）
# 20k tokens 为粗估
LONG20k_BODY_TARGET_CHARS = 22_000


def _build_preset_long20k(*, tail_header: bool = False) -> str:
    preamble_first = (
        "以下为长上下文压测文本：由多段互不相同的说明轮换拼接。请通读全文后，仅用「是」或「否」回答文末问题。\n\n"
    )
    preamble_last = (
        "以上为长上下文压测文本：由多段互不相同的说明轮换拼接。请通读全文后，仅用「是」或「否」回答文末问题。\n\n"
    )
    preamble = preamble_last if tail_header else preamble_first
    phrases = (
        "大模型推理分为预填充与解码两阶段；预填充对整段提示做单次前向并写入KV缓存。",
        "PagedAttention 将KV分页存放以降低显存碎片并提高批内并行度。",
        "连续批处理允许新请求插入正在运行的批次，用调度换整体吞吐。",
        "投机解码用小草稿模型或大步长草稿降低解码步数，需与接受率权衡。",
        "张量并行按列或行切分权重，多卡间通信成为扩展瓶颈之一。",
        "流水线并行把层分到不同设备，气泡时间与微批大小影响效率。",
        "数据并行复制模型分片数据，梯度聚合方式影响带宽与延迟。",
        "量化与稀疏化可减少权重量化比特与激活占用，但需校准精度。",
        "FlashAttention 通过分块与重算降低注意力显存访问量级。",
        "RoPE 与 ALiBi 等位置编码影响外推长度与长文稳定性。",
        "KV 传输在 PD 分离架构中占用网络带宽，序列化与压缩可优化。",
        "前缀缓存可复用相同系统提示或文档头的KV，降低首包延迟。",
        "调度器在 waiting 与 running 队列间分配，防止长请求饿死短请求。",
        "显存池与块分配策略影响是否能在高并发下仍接受新请求。",
        "温度与 top_p 等采样参数不改变预填充算量，主要影响解码路径。",
        "工具调用与 JSON 约束常在解码端加状态机或 logits 处理。",
        "多模态流水线在视觉编码后与文本嵌入拼接，再进入语言解码。",
        "推理服务常暴露 OpenAI 兼容 HTTP 接口以便压测与集成。",
        "健康检查与就绪探针用于编排系统滚动升级与流量切换。",
        "指标采集包括 TTFT、TPOT、吞吐与 GPU 利用率，用于调参对比。",
        "长提示的 tokenizer 开销与模型最大上下文长度共同限制可用输入。",
        "vLLM 等框架将引擎事件与周期指标写入日志，便于离线分析。",
    )
    parts: list[str] = []
    total = 0
    i = 0
    while total < LONG20k_BODY_TARGET_CHARS:
        s = phrases[i % len(phrases)]
        parts.append(s)
        total += len(s)
        i += 1
    body = "".join(parts)[:LONG20k_BODY_TARGET_CHARS]
    footer_first = "\n\n问题：下文是否明确论及「预填充」与「KV缓存」或同类键值缓存机制？只回答是或否。\n\n"
    footer_last = "\n\n问题：以上内容是否明确论及「预填充」与「KV缓存」或同类键值缓存机制？只回答是或否。"
    footer = footer_last if tail_header else footer_first
    if tail_header:
        # 对比组：正文在前，说明与问题在末尾附近（问题紧跟说明）。
        return body + "\n\n" + preamble + footer
    # 对比组：说明与问题在开头，正文在后（问题紧跟说明）。
    return preamble + footer + body


def _build_preset_long20k_question_first() -> str:
    return _build_preset_long20k(tail_header=False)


def _build_preset_long20k_question_last() -> str:
    return _build_preset_long20k(tail_header=True)


# -p @<key> 选用；key 为自由文本时不要用 @ 前缀
# 把所有内置模板集中在一个注册表里：字符串模板与运行时生成模板共用同一入口，便于维护与列出。
PRESETS: dict[str, PresetValue] = {
    # 短模板
    "short": "1+1等于几？只回答最终数字，不要其它解释。",
    "code": (
        "用 Python 写一个函数 `def is_prime(n: int) -> bool:`，判断正整数 n 是否为质数。"
        "只输出一个 Markdown 代码块，不要额外说明。"
    ),
    "repeat": "请只输出这一行原文，不要加引号或前后缀：VLLM_SINGLE_REQ_PING",
    # 长上下文压测：对比「问题在前」vs「问题在后」
    "long20k_question_first": _build_preset_long20k_question_first,
    "long20k_question_last": _build_preset_long20k_question_last,
}


def _all_preset_keys() -> list[str]:
    return sorted(PRESETS)


def _epilog_presets() -> str:
    rows: list[str] = []
    for k, v in sorted(PRESETS.items()):
        if isinstance(v, str):
            rows.append(f"{k} — {v[:48]}{'…' if len(v) > 48 else ''}")
    # 生成型模板单独给出更稳定的说明（避免预览文本随实现细节变化）
    rows.append(
        f"long20k_question_first — 约 {LONG20k_BODY_TARGET_CHARS} 字主体；preamble+问题在前，正文在后（问题紧跟说明）"
    )
    rows.append(
        f"long20k_question_last — 同主体；正文在前，preamble+问题在后（问题紧跟说明）"
    )
    rows.sort()
    return "内置提示（-p @<key>）：\n  " + "\n  ".join(rows)


def resolve_prompt(prompt: str) -> str:
    if not prompt.startswith("@") or prompt == "@":
        return prompt
    key = prompt[1:]
    preset = PRESETS.get(key)
    if callable(preset):
        return preset()
    if isinstance(preset, str):
        return preset
    names = ", ".join(_all_preset_keys())
    print(f"错误：未知内置提示 @{key}。可用：{names}", file=sys.stderr)
    raise SystemExit(2)


def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            content_type = resp.headers.get("Content-Type", "")
            preview = text[:1000] if text else "<empty response body>"
            raise ValueError(
                f"响应不是有效 JSON: HTTP {resp.status} {url}; "
                f"Content-Type={content_type!r}; body preview:\n{preview}"
            ) from e


def _post_json_stream_aggregate(
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
    *,
    completions: bool,
) -> tuple[dict, dict[str, float | None]]:
    """流式 POST，解析 SSE，聚合成与非流式相近的 body，并返回客户端时延（毫秒）。

    指标为客户端视角（含网络与排队），与 LongBench/pred_http.py 一致。
    """
    stream_payload = {
        **payload,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(stream_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    first_content_t: float | None = None
    text_parts: list[str] = []
    finish_reason: str | None = None
    resp_id: str | None = None
    model_name: str | None = None
    usage: dict | None = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].lstrip()
            if payload_text == "[DONE]":
                break
            try:
                obj = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if resp_id is None and isinstance(obj.get("id"), str):
                resp_id = obj["id"]
            if model_name is None and isinstance(obj.get("model"), str):
                model_name = obj["model"]
            now = time.perf_counter()
            for choice in obj.get("choices") or []:
                if completions:
                    piece = choice.get("text")
                    if piece:
                        if first_content_t is None:
                            first_content_t = now
                        text_parts.append(piece)
                else:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if first_content_t is None:
                            first_content_t = now
                        text_parts.append(piece)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
            u = obj.get("usage")
            if isinstance(u, dict):
                usage = u

    t_end = time.perf_counter()
    full_text = "".join(text_parts)
    e2e_ms = (t_end - t0) * 1000.0
    ttft_ms = (first_content_t - t0) * 1000.0 if first_content_t is not None else None

    completion_tokens: int | None = None
    if usage:
        ct = usage.get("completion_tokens")
        if isinstance(ct, int) and ct >= 0:
            completion_tokens = ct

    tpot_ms: float | None = None
    if (
        ttft_ms is not None
        and completion_tokens is not None
        and completion_tokens > 0
        and e2e_ms >= ttft_ms
    ):
        tpot_ms = (e2e_ms - ttft_ms) / float(completion_tokens)

    if completions:
        agg: dict = {
            "id": resp_id or "",
            "object": "text_completion",
            "model": model_name or payload.get("model", ""),
            "choices": [
                {
                    "text": full_text,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }
    else:
        agg = {
            "id": resp_id or "",
            "object": "chat.completion",
            "model": model_name or payload.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": finish_reason,
                }
            ],
        }
    if usage:
        agg["usage"] = usage

    metrics = {
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
    }
    return agg, metrics


def main() -> int:
    preset_help = "用户消息；写 @预设名 使用内置模板（如 @short、@code），预设见 --list-prompts"
    p = argparse.ArgumentParser(
        description="单点请求 vLLM OpenAI 兼容 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog_presets(),
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"含 /v1 的根地址（默认 {DEFAULT_BASE_URL}）",
    )
    p.add_argument("--model", "-m", required=True, help="模型 id（须与 vLLM 服务一致）")
    p.add_argument(
        "--prompt",
        "-p",
        default="@short",
        metavar="TEXT|@PRESET",
        help=preset_help,
    )
    p.add_argument(
        "--list-prompts",
        action="store_true",
        help="列出内置 @ 预设的全文并退出",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--timeout", type=float, default=600.0, help="请求超时（秒）")
    p.add_argument(
        "--completions",
        action="store_true",
        help="使用 /v1/completions 而非 chat（适合无 chat 模板的纯文本模型）",
    )
    p.add_argument("--raw", action="store_true", help="只打印完整 JSON，不额外打印 assistant 摘要")
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="非流式单次请求：仅能量化 E2E_ms，TTFT_ms/TPOT_ms 为 null（与旧行为一致）",
    )
    args = p.parse_args()

    if args.list_prompts:
        for key in _all_preset_keys():
            preset = PRESETS[key]
            if callable(preset):
                text = preset()
                print(f"=== @{key} ===（共 {len(text)} 字符；完整正文过长，仅首尾预览）")
                print(text[:400])
                print("\n...\n")
                print(text[-400:])
                print()
            else:
                print(f"=== @{key} ===\n{preset}\n")
        return 0

    user_text = resolve_prompt(args.prompt)

    base = args.base_url.rstrip("/")

    if args.completions:
        url = f"{base}/completions"
        payload = {
            "model": args.model,
            "prompt": user_text,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    else:
        url = f"{base}/chat/completions"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": user_text}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }

    latency: dict[str, float | None] = {}
    try:
        if args.no_stream:
            t_req = time.perf_counter()
            body = _post_json(url, payload, args.api_key, args.timeout)
            latency = {
                "ttft_ms": None,
                "e2e_ms": (time.perf_counter() - t_req) * 1000.0,
                "tpot_ms": None,
            }
        else:
            body, latency = _post_json_stream_aggregate(
                url,
                payload,
                args.api_key,
                args.timeout,
                completions=args.completions,
            )
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {url}\n{err}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"请求失败: {e.reason}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(json.dumps(body, ensure_ascii=False, indent=2))
    latency_lines = (
        "\n--- latency (client) ---",
        f"TTFT_ms: {latency.get('ttft_ms')!r}",
        f"TPOT_ms: {latency.get('tpot_ms')!r}",
        f"E2E_ms:  {latency.get('e2e_ms')!r}",
        "（TTFT=首 Token；TPOT=(E2E-TTFT)/completion_tokens；无 usage 时 TPOT 可能为 null）",
    )
    if args.raw:
        print(*latency_lines, sep="\n", file=sys.stderr)
    else:
        print(*latency_lines, sep="\n")
    if args.raw:
        return 0

    choices = body.get("choices") or []
    if not choices:
        return 0
    c0 = choices[0]
    if args.completions:
        text = c0.get("text", "")
        print("\n--- 生成文本 ---\n", text, sep="", end="")
    else:
        msg = c0.get("message") or {}
        content = msg.get("content", "")
        print("\n--- assistant ---\n", content, sep="", end="")
    print()
    usage = body.get("usage")
    if usage:
        print("\n--- usage ---", usage, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
