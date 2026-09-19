#!/usr/bin/env python3
"""Preprocess Google ClusterData2019 raw exports into the unified schema.

Usage:
  python scripts/preprocess_google.py --raw data/raw/google --out data/processed/google
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess Google trace")
    ap.add_argument("--raw", default="data/raw/google")
    ap.add_argument("--out", default="data/processed/google")
    ap.add_argument("--config", default="configs/datasets/google_trace.yaml")
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    from src.datasets.google import load_events
    from src.datasets.trace_common import events_to_episode, file_sha256
    from src.io_utils import load_config

    raw_dir = ROOT / args.raw if not Path(args.raw).is_absolute() else Path(args.raw)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    ds_cfg = load_config(ROOT / args.config).get("dataset", {})

    try:
        events = load_events(raw_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    checksums = {f.name: file_sha256(f) for f in sorted(raw_dir.glob("*.csv*"))}
    meta = events_to_episode(events, ds_cfg, "google", checksums, out_dir, seed=args.seed)
    print(f"OK: {meta['n_tasks']} tasks, {meta['n_machines']} machines, "
          f"{meta['n_slots']} slots -> {out_dir}")


if __name__ == "__main__":
    main()
