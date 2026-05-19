#!/usr/bin/env python3
"""Summarize round-2 JSON benchmarks into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_reports(r2_dir: Path) -> list[dict[str, Any]]:
    """Load all r2 JSON reports."""
    reports: list[dict[str, Any]] = []
    for path in sorted(r2_dir.glob("R2-*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            report["_source_path"] = str(path)
            reports.append(report)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skip %s: %s", path, exc)
    return reports


def latest_by_label(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the newest report per variant label."""
    latest: dict[str, dict[str, Any]] = {}
    for report in reports:
        label = str(report.get("label", ""))
        if not label:
            continue
        current = latest.get(label)
        if current is None or str(report.get("timestamp", "")) > str(current.get("timestamp", "")):
            latest[label] = report
    return [latest[label] for label in sorted(latest)]


def report_status(report: dict[str, Any]) -> str:
    """Return OK/FAIL/MISSING based on recorded HTTP status fields."""
    b = report.get("benchmarks", {})
    checks: list[int | None] = []
    for name in ("warmup", "b1_short", "b2_gen512", "b3_long_prefill"):
        if name not in b:
            return "MISSING"
        checks.append(b.get(name, {}).get("status"))

    b4 = b.get("b4_prefix", {})
    if not b4:
        return "MISSING"
    checks.extend([b4.get("status_first"), b4.get("status_second")])

    label = str(report.get("label", ""))
    if label in {"R2-A0", "R2-A2", "R2-A4"}:
        if "b6_gen2048" not in b:
            return "MISSING"
        checks.append(b.get("b6_gen2048", {}).get("status"))
    elif "b6_gen2048" in b:
        checks.append(b.get("b6_gen2048", {}).get("status"))

    if any(status is None for status in checks):
        return "MISSING"
    return "OK" if all(status == 200 for status in checks) else "FAIL"


def min_free_memory_mib(nvidia_smi: str) -> str:
    """Parse minimum free MiB from nvidia-smi CSV output."""
    free_values: list[int] = []
    for line in nvidia_smi.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            free_values.append(int(parts[2].split()[0]))
        except (ValueError, IndexError):
            continue
    return str(min(free_values)) if free_values else ""


def row_from_report(report: dict[str, Any]) -> dict[str, str]:
    """Flatten one report to CSV row."""
    b = report.get("benchmarks", {})
    b1 = b.get("b1_short", {})
    b2 = b.get("b2_gen512", {})
    b3 = b.get("b3_long_prefill", {})
    b4 = b.get("b4_prefix", {})
    b5 = b.get("b5_memory", {})
    b6 = b.get("b6_gen2048", {})
    return {
        "label": str(report.get("label", "")),
        "timestamp": str(report.get("timestamp", "")),
        "status": report_status(report),
        "b1_sec": str(b1.get("elapsed_sec", "")),
        "b2_tps": str(b2.get("tokens_per_sec", "")),
        "b2_sec": str(b2.get("elapsed_sec", "")),
        "b3_sec": str(b3.get("elapsed_sec", "")),
        "b3_prompt_tokens": str(b3.get("prompt_tokens", "")),
        "b4_reduction_pct": str(b4.get("reduction_pct", "")),
        "b4_second_sec": str(b4.get("second_sec", "")),
        "b5_min_free_mib": min_free_memory_mib(str(b5.get("nvidia_smi", ""))),
        "b6_tps": str(b6.get("tokens_per_sec", "")),
        "b6_sec": str(b6.get("elapsed_sec", "")),
        "source": str(report.get("_source_path", "")),
    }


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    _round2_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--r2-dir",
        default=str(_round2_root / "results"),
    )
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    r2_dir = Path(args.r2_dir)
    if not r2_dir.is_dir():
        logger.error("Directory not found: %s", r2_dir)
        return 1

    reports = latest_by_label(load_reports(r2_dir))
    if not reports:
        logger.error("No R2-*.json in %s", r2_dir)
        return 1

    out_csv = Path(args.out_csv) if args.out_csv else r2_dir / "summary.csv"
    fieldnames = [
        "label",
        "timestamp",
        "status",
        "b1_sec",
        "b2_tps",
        "b2_sec",
        "b3_sec",
        "b3_prompt_tokens",
        "b4_reduction_pct",
        "b4_second_sec",
        "b5_min_free_mib",
        "b6_tps",
        "b6_sec",
        "source",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(row_from_report(report))

    logger.info("Wrote %s (%d rows)", out_csv, len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
