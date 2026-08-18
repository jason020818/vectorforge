"""Render a Markdown comparison table from Phase 3 result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to summary.json")
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    lines = [
        (
            "| Engine | Recall@10 | Recall@100 | QPS | "
            "P50 | P95 | P99 | Build | Peak RSS | Index Size |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["engines"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("engine")),
                    _fmt(row.get("recall_at_10")),
                    _fmt(row.get("recall_at_100")),
                    _fmt(row.get("qps_median")),
                    _fmt(row.get("p50_ms_median")),
                    _fmt(row.get("p95_ms_median")),
                    _fmt(row.get("p99_ms_median")),
                    _fmt(row.get("build_time_s")),
                    _fmt(row.get("peak_rss_bytes")),
                    _fmt(row.get("index_size_bytes")),
                ]
            )
            + " |"
        )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
