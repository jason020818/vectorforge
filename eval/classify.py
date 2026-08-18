"""Placeholder performance classifier (REJECT / NONE / XS / S / M / L / XL).

Scoring rules will live in a version-controlled config once the harness exists.
"""

from __future__ import annotations

CLASSIFICATION = {
    "REJECT": "correctness or regression failure",
    "NONE": "<1% meaningful improvement",
    "XS": "1-3%",
    "S": "3-7%",
    "M": "7-15%",
    "L": "15-30%",
    "XL": ">30%",
}


def main() -> int:
    print("classify.py: not active in Phase 0/1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
