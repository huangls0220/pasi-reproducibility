#!/usr/bin/env python3
"""Validate a processed dataset against the unified schema (Section 8.3).

Usage:
  python scripts/validate_dataset.py --dir data/processed/google
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.trace_loader import (  # noqa: E402
    PROV_SCHEMA, STATIC_SCHEMA, TASK_SCHEMA, validate_schema,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate processed dataset")
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_absolute():
        d = ROOT / d
    problems: list[str] = []
    for fname in ["tasks.parquet", "providers.parquet", "provider_static.parquet", "meta.json"]:
        if not (d / fname).exists():
            problems.append(f"missing file: {fname}")
    if problems:
        sys.exit("FAILED:\n  " + "\n  ".join(problems))

    tasks = pd.read_parquet(d / "tasks.parquet")
    provs = pd.read_parquet(d / "providers.parquet")
    static = pd.read_parquet(d / "provider_static.parquet")
    problems += validate_schema(tasks, TASK_SCHEMA, "tasks")
    problems += validate_schema(provs, PROV_SCHEMA, "providers")
    problems += validate_schema(static, STATIC_SCHEMA, "provider_static")

    # Range sanity
    if len(tasks):
        if not ((tasks["L"] > 0).all() and (tasks["deadline"] > 0).all()):
            problems.append("tasks: non-positive L or deadline")
        if not ((tasks["min_quality"] < tasks["q_bar"]).all()):
            problems.append("tasks: min_quality >= q_bar rows present")
    if len(provs) and not provs["load_ratio"].between(0, 1).all():
        problems.append("providers: load_ratio outside [0,1]")

    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    if problems:
        sys.exit("FAILED:\n  " + "\n  ".join(problems))
    print(f"OK: {d}")
    print(f"  tasks={len(tasks)}  provider-slots={len(provs)}  "
          f"machines={len(static)}  slots={meta.get('n_slots')}")
    print(f"  source={meta.get('source')}  checksums={len(meta.get('raw_checksums', {}))} files")


if __name__ == "__main__":
    main()
