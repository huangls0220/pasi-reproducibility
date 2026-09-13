#!/usr/bin/env python3
"""Run a single PRIME experiment.

Usage:
  python scripts/run_experiment.py --config configs/experiments/correctness.yaml \
      --dataset synthetic --method PRIME --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_utils import (
    get_env_info,
    load_config,
    make_run_dir,
    save_config_snapshot,
    save_json,
    save_parquet,
    deep_merge,
)
from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single PRIME experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name")
    parser.add_argument("--method", required=True, help="Method name (PRIME, MOI, FR, etc.)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--pattern", default=None, help="Synthetic pattern (stationary, burst, dynamic); default from config")
    parser.add_argument("--force", action="store_true", help="Re-run even if summary exists")

    args = parser.parse_args()

    # Load config
    exp_cfg = load_config(args.config)
    default_cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    cfg = deep_merge(default_cfg, exp_cfg)

    # Method dispatch is fully handled by src/mechanisms.get_mechanism
    method = args.method
    # Pattern: CLI overrides config; config default in dataset section
    pattern = args.pattern or cfg.get("dataset", {}).get("pattern", "stationary")

    # Setup output paths
    exp_name = Path(args.config).stem
    results_root = Path(__file__).resolve().parent.parent / "results"
    run_dir = make_run_dir(results_root, exp_name, args.dataset, method, args.seed)

    # Check if already completed
    summary_file = run_dir / "summary.json"
    if summary_file.exists() and not args.force:
        print(f"Run already completed: {summary_file}")
        return

    # Generate dataset
    print(f"Generating {args.dataset} dataset (pattern={pattern}, seed={args.seed})...")
    if args.dataset == "synthetic":
        data = generate_synthetic_episode(cfg, seed=args.seed, pattern=pattern)
    elif args.dataset in ("google", "alibaba"):
        from src.datasets.trace_loader import load_trace_episode

        data = load_trace_episode(args.dataset, cfg, seed=args.seed)
    else:
        print(f"ERROR: Dataset '{args.dataset}' adapter not yet implemented.")
        sys.exit(1)

    # Save config snapshot
    save_config_snapshot(cfg, run_dir / "config_snapshot.yaml")

    # Save env info
    save_json(get_env_info(), run_dir / "env_info.json")

    # Run simulation
    print(f"Running {method} (seed={args.seed})...")
    sim = Simulator(cfg, data, method=method, seed=args.seed)
    result = sim.run()

    # Save outputs
    save_json(result["summary"], summary_file)
    save_json(result["diagnostics"], run_dir / "diagnostics.json")
    save_parquet(result["pair_log"], run_dir / "pair_log.parquet")
    save_parquet(result["slot_log"], run_dir / "slot_log.parquet")
    save_parquet(result["provider_log"], run_dir / "provider_log.parquet")

    # Print summary
    s = result["summary"]
    diag = result["diagnostics"]
    print(f"\n{'='*50}")
    print(f"Method: {method}  Seed: {args.seed}")
    print(f"Status: {diag['status']}")
    print(f"Assigned: {s['num_assigned']}  HQR: {s['HQR']:.4f}  Avg Q: {s['average_quality']:.4f}")
    print(f"Total payment: {s['cumulative_payment']:.2f}  Platform utility: {s['platform_utility']:.2f}")
    print(f"Cost/assigned: {s['payment_per_assigned']:.4f}  Cost/HQ: {s.get('payment_per_HQ', 'N/A')}")
    print(f"Runtime: {s['total_runtime']:.1f}s")
    if diag["issues"]:
        print(f"Issues: {diag['issues']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
