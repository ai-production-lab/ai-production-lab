#!/usr/bin/env python3
"""Round-2 unified benchmark suite (128K variants on A100)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:50600/v1/chat/completions"
DEFAULT_MODEL = "/models/Qwen3.6-27B"
LONG_SENTENCE = "人工智能正在深刻改变人类社会的方方面面。"
SYSTEM_LONG = "你是专业助手。" + ("请认真阅读以下背景材料。" * 900)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Round-2 benchmark for vLLM.")
    parser.add_argument("--label", required=True, help="Variant id, e.g. R2-A0.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "results" / "r2"),
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-b6", action="store_true", help="Run optional 2048-token test.")
    return parser


def post_chat(
    session: requests.Session,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> tuple[float, dict[str, Any], int]:
    """POST chat completion; return elapsed sec, json body, status."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.time()
    response = session.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - started
    body: dict[str, Any] = {}
    if "application/json" in response.headers.get("content-type", ""):
        try:
            body = response.json()
        except ValueError:
            body = {}
    return elapsed, body, response.status_code


def post_chat_official_thinking(
    session: requests.Session,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> tuple[float, dict[str, Any], int]:
    """POST chat with Qwen README thinking-mode general sampling."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
    }
    started = time.time()
    response = session.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - started
    body: dict[str, Any] = {}
    if "application/json" in response.headers.get("content-type", ""):
        try:
            body = response.json()
        except ValueError:
            body = {}
    return elapsed, body, response.status_code


def bench_b1(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """Short request latency."""
    elapsed, body, status = post_chat(
        session,
        url,
        model,
        [{"role": "user", "content": "你好"}],
        32,
        timeout,
    )
    usage = body.get("usage", {})
    return {
        "elapsed_sec": round(elapsed, 3),
        "status": status,
        "completion_tokens": usage.get("completion_tokens"),
    }


def warmup(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """Run one discarded short request to reduce cold first-token noise."""
    elapsed, body, status = post_chat(
        session,
        url,
        model,
        [{"role": "user", "content": "预热"}],
        8,
        timeout,
    )
    usage = body.get("usage", {})
    return {
        "elapsed_sec": round(elapsed, 3),
        "status": status,
        "completion_tokens": usage.get("completion_tokens"),
    }


def bench_b2(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """512-token generation throughput (Qwen README thinking-mode sampling)."""
    prompt = "请写一篇500字的文章介绍人工智能的发展历史"
    elapsed, body, status = post_chat_official_thinking(
        session,
        url,
        model,
        [{"role": "user", "content": prompt}],
        512,
        timeout,
    )
    usage = body.get("usage", {})
    completion = int(usage.get("completion_tokens") or 0)
    tps = round(completion / elapsed, 2) if elapsed > 0 and completion else None
    return {
        "elapsed_sec": round(elapsed, 3),
        "status": status,
        "completion_tokens": completion,
        "tokens_per_sec": tps,
        "sampling": "official_thinking_general",
    }


def bench_b3(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """Long prefill (~60k chars)."""
    text = LONG_SENTENCE * (60000 // len(LONG_SENTENCE))
    elapsed, body, status = post_chat(
        session,
        url,
        model,
        [{"role": "user", "content": text + "\n请用一句话总结以上内容。"}],
        64,
        timeout,
    )
    usage = body.get("usage", {})
    return {
        "elapsed_sec": round(elapsed, 3),
        "status": status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def bench_b4(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """Prefix cache: same long system prompt twice."""
    messages = [
        {"role": "system", "content": SYSTEM_LONG},
        {"role": "user", "content": "用一句话介绍人工智能"},
    ]
    e1, _, s1 = post_chat(session, url, model, messages, 32, timeout)
    e2, _, s2 = post_chat(session, url, model, messages, 32, timeout)
    reduction = round((1 - e2 / e1) * 100, 1) if e1 > 0 else None
    return {
        "first_sec": round(e1, 3),
        "second_sec": round(e2, 3),
        "reduction_pct": reduction,
        "status_first": s1,
        "status_second": s2,
    }


def bench_b5() -> dict[str, Any]:
    """GPU memory snapshot."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        )
        return {"nvidia_smi": out.strip()}
    except (subprocess.CalledProcessError, OSError) as exc:
        return {"error": str(exc)}


def bench_b6(session: requests.Session, url: str, model: str, timeout: int) -> dict[str, Any]:
    """Optional 2048-token generation with Qwen README thinking sampling."""
    elapsed, body, status = post_chat_official_thinking(
        session,
        url,
        model,
        [{"role": "user", "content": "详细解释深度学习的发展脉络"}],
        2048,
        max(timeout, 1800),
    )
    usage = body.get("usage", {})
    completion = int(usage.get("completion_tokens") or 0)
    tps = round(completion / elapsed, 2) if elapsed > 0 and completion else None
    return {
        "elapsed_sec": round(elapsed, 3),
        "status": status,
        "completion_tokens": completion,
        "tokens_per_sec": tps,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run all benchmarks and return report dict."""
    session = requests.Session()
    session.trust_env = False
    report: dict[str, Any] = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "url": args.url,
        "benchmarks": {},
    }
    report["benchmarks"]["warmup"] = warmup(session, args.url, args.model, args.timeout)
    report["benchmarks"]["b1_short"] = bench_b1(session, args.url, args.model, args.timeout)
    report["benchmarks"]["b2_gen512"] = bench_b2(session, args.url, args.model, args.timeout)
    report["benchmarks"]["b3_long_prefill"] = bench_b3(
        session,
        args.url,
        args.model,
        args.timeout,
    )
    report["benchmarks"]["b4_prefix"] = bench_b4(session, args.url, args.model, args.timeout)
    report["benchmarks"]["b5_memory"] = bench_b5()
    if args.run_b6:
        report["benchmarks"]["b6_gen2048"] = bench_b6(
            session,
            args.url,
            args.model,
            args.timeout,
        )
    return report


def failed_benchmarks(report: dict[str, Any]) -> list[str]:
    """Return benchmark names whose HTTP status indicates failure."""
    benchmarks = report.get("benchmarks", {})
    failures: list[str] = []
    for name in ("warmup", "b1_short", "b2_gen512", "b3_long_prefill", "b6_gen2048"):
        bench = benchmarks.get(name)
        if bench and bench.get("status") != 200:
            failures.append(name)

    b4 = benchmarks.get("b4_prefix", {})
    if b4:
        if b4.get("status_first") != 200 or b4.get("status_second") != 200:
            failures.append("b4_prefix")
    return failures


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{args.label}-{stamp}.json"

    try:
        report = run_suite(args)
    except requests.RequestException as exc:
        logger.exception("Benchmark failed: %s", exc)
        return 1

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    failures = failed_benchmarks(report)
    if failures:
        logger.error("Non-200 benchmark results: %s", ", ".join(failures))
        return 1
    b2 = report["benchmarks"].get("b2_gen512", {})
    b3 = report["benchmarks"].get("b3_long_prefill", {})
    logger.info(
        "summary label=%s B2_tps=%s B3_sec=%s B3_prompt=%s",
        args.label,
        b2.get("tokens_per_sec"),
        b3.get("elapsed_sec"),
        b3.get("prompt_tokens"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
