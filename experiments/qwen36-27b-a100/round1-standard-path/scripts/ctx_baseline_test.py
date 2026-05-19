#!/usr/bin/env python3
"""Long-context chat completion smoke test for vLLM on A100."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:50600/v1/chat/completions"
DEFAULT_MODEL = "/models/Qwen3.6-27B"
SENTENCE = "人工智能正在深刻改变人类社会的方方面面。"
SENTENCE_CHARS = len(SENTENCE)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Send a long-context request and print token usage.",
    )
    parser.add_argument(
        "target_chars",
        type=int,
        nargs="?",
        default=60000,
        help="Approximate Chinese character count in user message body.",
    )
    parser.add_argument(
        "label",
        type=str,
        nargs="?",
        default="test",
        help="Label printed in log output.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Chat completions endpoint.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id served by vLLM.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP timeout in seconds.",
    )
    return parser


def run_test(
    target_chars: int,
    label: str,
    url: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    """Run one long-context request and return summary dict."""
    repeat = max(1, target_chars // SENTENCE_CHARS)
    text = SENTENCE * repeat
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": text + "\n请用一句话总结以上内容。",
            }
        ],
        "max_tokens": 64,
        "stream": False,
    }
    started = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - started
    usage: dict[str, Any] = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            usage = response.json().get("usage", {})
        except ValueError:
            usage = {}
    summary = {
        "label": label,
        "status_code": response.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_sec": round(elapsed, 1),
        "approx_chars": len(text),
    }
    return summary


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    try:
        result = run_test(
            target_chars=args.target_chars,
            label=args.label,
            url=args.url,
            model=args.model,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        logger.exception("Request failed: %s", exc)
        return 1

    logger.info(
        "[%s] status=%s prompt_tokens=%s completion_tokens=%s elapsed=%ss chars=%s",
        result["label"],
        result["status_code"],
        result["prompt_tokens"],
        result["completion_tokens"],
        result["elapsed_sec"],
        result["approx_chars"],
    )
    return 0 if result["status_code"] == 200 else 2


if __name__ == "__main__":
    sys.exit(main())
