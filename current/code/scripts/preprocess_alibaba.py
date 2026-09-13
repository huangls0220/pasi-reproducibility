#!/usr/bin/env python3
"""Preprocess Alibaba cluster-trace-v2018 into the unified schema.

Usage:
  python scripts/preprocess_alibaba.py --raw data/raw/alibaba --out data/processed/alibaba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess Alibaba trace")
    ap.add_argument("--raw", default="data/raw/alibaba")
    ap.add_argument("--out", default="data/processed/alibaba")
    ap.add_argument("--config", default="configs/datasets/alibaba_trace.yaml")
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    from src.datasets.alibaba import load_events
    from src.datasets.trace_common import events_to_episode, file_sha256
    from src.io_utils import load_config

    raw_dir = ROOT / args.raw if not Path(args.raw).is_absolute() else Path(args.raw)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    ds_cfg = load_config(ROOT / args.config).get("dataset", {})

    try:
        events, machine_src = load_events(
            raw_dir, machine_subset=int(ds_cfg.get("machine_subset", 1000)))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    checksums = {f.name: file_sha256(f) for f in sorted(raw_dir.glob("*.csv*"))}
    meta = events_to_episode(events, ds_cfg, "alibaba", checksums, out_dir, seed=args.seed)
    # record the machine-assignment approximation honestly
    import json
    meta["machine_source"] = machine_src
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"OK: {meta['n_tasks']} tasks, {meta['n_machines']} machines, "
          f"{meta['n_slots']} slots (machine_source={machine_src}) -> {out_dir}")


if __name__ == "__main__":
    main()
