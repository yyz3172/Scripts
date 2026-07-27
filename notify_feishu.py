#!/usr/bin/env python3
"""飞书自定义机器人群通知。

优先使用参数传入的 webhook；否则读取环境变量 FEISHU_WEBHOOK_URL。
发送失败只记录错误，不抛异常，避免影响监控主循环。
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request


ENV_WEBHOOK = "FEISHU_WEBHOOK_URL"


def resolve_webhook(webhook_url: str | None = None) -> str | None:
    url = (webhook_url or os.environ.get(ENV_WEBHOOK) or "").strip()
    return url or None


def hostname() -> str:
    return socket.gethostname()


def notify_feishu(
    text: str,
    webhook_url: str | None = None,
    *,
    prefix_host: bool = True,
    timeout: float = 5.0,
    retries: int = 1,
) -> bool:
    """向飞书群自定义机器人发送文本消息。成功返回 True。"""
    url = resolve_webhook(webhook_url)
    if not url:
        return False

    content = text.strip()
    if not content:
        return False
    if prefix_host:
        content = f"[{hostname()}] {content}"

    payload = {
        "msg_type": "text",
        "content": {"text": content},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts = max(1, retries + 1)
    last_error = ""
    for _ in range(attempts):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    result = json.loads(body)
                except json.JSONDecodeError:
                    result = {}
                # 飞书成功通常 code=0 或 StatusCode=0
                code = result.get("code", result.get("StatusCode"))
                if code in (0, "0", None) and resp.status == 200:
                    return True
                last_error = body or f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail or exc.reason}"
        except Exception as exc:  # noqa: BLE001 - 通知失败不能打断监控
            last_error = str(exc)

    print(f"error: 飞书通知失败: {last_error}", file=sys.stderr, flush=True)
    return False


def add_feishu_args(parser) -> None:
    parser.add_argument(
        "--notify",
        action="store_true",
        help="开启飞书推送（默认关闭；需同时配置 webhook）",
    )
    parser.add_argument(
        "--feishu-webhook",
        default=None,
        help=(
            "飞书自定义机器人 webhook；"
            f"也可通过环境变量 {ENV_WEBHOOK} 配置"
        ),
    )


def should_notify(notify_enabled: bool, webhook_url: str | None = None) -> bool:
    return bool(notify_enabled and resolve_webhook(webhook_url))
