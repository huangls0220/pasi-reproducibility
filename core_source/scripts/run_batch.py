#!/usr/bin/env python3
"""Batch experiment runner with checkpoint/resume support.

Usage:
  python scripts/run_batch.py --config configs/experiments/overall.yaml \
      --dataset synthetic --seeds 1:30 --methods all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_seeds(spec: str) -> list[int]:
    """Parse seed specification: "1:30" -> [1..30], "1,3,5" -> [1,3,5]."""
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0])
        end = int(parts[1])
        return list(range(start, end + 1))
    return [int(s) for s in spec.split(",")]


def parse_methods(spec: str) -> list[str]:
    """Parse method specification. 'all' -> standard set."""
    if spec.lower() == "all":
        return ["PRIME", "MOI", "FR", "DP", "RAI", "LDI",
                "PRIME-w/o-PD", "PRIME-w/o-PW", "PRIME-w/o-LT",
                "PRIME-w/o-RC", "PRIME-Fixed"]
    return [m.strip() for m in spec.split(",")]


def is_completed(run_dir: Path) -> bool:
    """Check if a run has a valid summary."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path) as fh:
            data = json.load(fh)
        return data.get("status", "") == "completed" or "num_assigned" in data
    except (json.JSONDecodeError, KeyError):
        return False


def run_single(
    config_path: Path,
    dataset: str,
    method: str,
    seed: int,
    pattern: str,
    results_root: Path,
    force: bool = False,
) -> dict:
    """Run one experiment and return result status."""
    import subprocess

    exp_name = config_path.stem
    run_dir = results_root / exp_name / dataset / method / f"seed_{seed:04d}"

    if is_completed(run_dir) and not force:
        return {"method": method, "seed": seed, "status": "skipped", "run_dir": str(run_dir)}

    start = time.perf_counter()
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_experiment.py"),
            "--config", str(config_path),
            "--dataset", dataset,
            "--method", method,
            "--seed", str(seed),
        ]
        if pattern:
            cmd += ["--pattern", pattern]
        if force:
            cmd += ["--force"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.perf_counter() - start
        if result.returncode == 0:
            return {"method": method, "seed": seed, "status": "completed",
                    "run_dir": str(run_dir), "runtime": elapsed}
        else:
            return {"method": method, "seed": seed, "status": "failed",
                    "run_dir": str(run_dir), "stderr": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"method": method, "seed": seed, "status": "timeout", "run_dir": str(run_dir)}
    except Exception as e:
        return {"method": method, "seed": seed, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name")
    parser.add_argument("--seeds", default="1:3", help="Seed range (e.g., 1:30 or 1,2,3)")
    parser.add_argument("--methods", default="PRIME", help="Methods (comma-separated or 'all')")
    parser.add_argument("--pattern", default=None, help="Synthetic pattern (default: from config)")
    parser.add_argument("--force", action="store_true", help="Re-run completed runs")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = parse_methods(args.methods)
    config_path = Path(args.config)
    results_root = Path(__file__).resolve().parent.parent / "results"

    print(f"Batch run: {len(methods)} methods × {len(seeds)} seeds = {len(methods) * len(seeds)} runs")
    print(f"Config: {config_path}  Dataset: {args.dataset}  Pattern: {args.pattern}")

    # Generate run list
    runs = [(m, s) for m in methods for s in seeds]
    results = []
    failures = []

    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for method, seed in runs:
                fut = executor.submit(
                    run_single, config_path, args.dataset, method, seed,
                    args.pattern, results_root, args.force,
                )
                futures[fut] = (method, seed)

            for fut in as_completed(futures):
                method, seed = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    status = r["status"]
                    print(f"  {method} seed={seed}: {status}")
                    if status == "failed":
                        failures.append(r)
                except Exception as e:
                    results.append({"method": method, "seed": seed, "status": "error", "error": str(e)})
                    failures.append(results[-1])
                    print(f"  {method} seed={seed}: error ({e})")
    else:
        for method, seed in runs:
            r = run_single(config_path, args.dataset, method, seed, args.pattern, results_root, args.force)
            results.append(r)
            print(f"  {method} seed={seed}: {r['status']}")
            if r["status"] == "failed":
                failures.append(r)

    # Summary
    completed = sum(1 for r in results if r["status"] in ("completed", "skipped"))
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nDone: {completed}/{len(results)} completed, {failed} failed")

    if failures:
        import pandas as pd
        fail_path = results_root / config_path.stem / args.dataset / "failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"Failures saved to {fail_path}")


if __name__ == "__main__":
    main()
