#!/usr/bin/env python3
"""Sweep runner: variations × methods × seeds with resume + parallelism.

Consumes experiment configs with a `sweep:` section:

  sweep:
    methods: [PRIME, MOI]
    seeds: 20                       # -> seeds 1..20 (or "1:20"/"1,2,3")
    variations:
      - name: base
        overrides: {}               # deep-merged into the config
      - name: rhoB_0.4
        methods: [PRIME]            # optional per-variation method override
        overrides: {providers: {behavioral_fraction: 0.4}}

Output layout:
  results/<experiment>/<dataset>/<variation>/<method>/seed_%04d/
      summary.json, diagnostics.json, config_snapshot.yaml,
      slot_log.parquet [, provider_log.parquet, pair_log.parquet]

`simulation.save_logs` (list of pair/provider/slot) controls which log
levels are persisted; slot logs are always saved.  Runs whose summary.json
already exists are skipped unless --force.  Failures land in failures.csv.

Usage:
  python scripts/run_sweep.py --config configs/experiments/overall.yaml --parallel 14
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_seeds(spec) -> list[int]:
    if isinstance(spec, int):
        return list(range(1, spec + 1))
    s = str(spec)
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def run_one(job: dict) -> dict:
    """Worker: one (variation, method, seed) run.  Must be picklable."""
    import json

    t0 = time.perf_counter()
    out = {k: job[k] for k in ("experiment", "dataset", "variation", "method", "seed")}
    run_dir = Path(job["run_dir"])
    try:
        from src.datasets.synthetic import generate_synthetic_episode
        from src.io_utils import save_config_snapshot, save_json, save_parquet
        from src.simulator import Simulator

        cfg = job["cfg"]
        seed = job["seed"]
        pattern = (cfg.get("dataset", {}) or {}).get("pattern", "stationary")

        if job["dataset"] == "synthetic":
            data = generate_synthetic_episode(cfg, seed=seed, pattern=pattern)
        else:
            from src.datasets.trace_loader import load_trace_episode
            data = load_trace_episode(job["dataset"], cfg, seed=seed)

        sim = Simulator(cfg, data, method=job["method"], seed=seed)
        res = sim.run()

        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(res["summary"], run_dir / "summary.json")
        save_json(res["diagnostics"], run_dir / "diagnostics.json")
        save_config_snapshot(cfg, run_dir / "config_snapshot.yaml")
        save_logs = (cfg.get("simulation", {}) or {}).get(
            "save_logs", ["pair", "provider", "slot"])
        save_parquet(res["slot_log"], run_dir / "slot_log.parquet")
        if "provider" in save_logs and len(res["provider_log"]):
            save_parquet(res["provider_log"], run_dir / "provider_log.parquet")
        if "pair" in save_logs and len(res["pair_log"]):
            save_parquet(res["pair_log"], run_dir / "pair_log.parquet")

        out["status"] = ("completed" if res["diagnostics"]["status"] == "ok"
                         else "diagnostics_failed")
        out["diag_issues"] = "; ".join(res["diagnostics"].get("issues", []))
        out["runtime"] = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 — recorded in failures.csv
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
        out["runtime"] = time.perf_counter() - t0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep experiment.")
    parser.add_argument("--config", required=True, help="Experiment YAML with sweep section")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--parallel", type=int, default=1, help="Worker processes")
    parser.add_argument("--force", action="store_true", help="Re-run completed runs")
    parser.add_argument("--seeds", default=None, help="Override sweep seeds (e.g. 1:3)")
    parser.add_argument("--variations", default=None, help="Comma list to filter variations")
    parser.add_argument("--methods", default=None, help="Comma list to filter methods")
    args = parser.parse_args()

    from src.io_utils import deep_merge, load_config

    exp_cfg = load_config(args.config)
    if "analysis_of" in exp_cfg:
        print(f"'{Path(args.config).stem}' is analysis-only "
              f"(reuses experiment '{exp_cfg['analysis_of']}'). Nothing to run.")
        return
    default_cfg = load_config(ROOT / "configs" / "default.yaml")
    # Optional includes (e.g. configs/tuned/same_budget.yaml), merged
    # between defaults and the experiment body.
    for inc in exp_cfg.get("include", []) or []:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = ROOT / inc_path
        if inc_path.exists():
            default_cfg = deep_merge(default_cfg, load_config(inc_path))
        else:
            print(f"WARNING: include not found (run tune_baselines first?): {inc_path}")

    sweep = exp_cfg.get("sweep")
    if not sweep:
        print("ERROR: config has no sweep section; use run_batch.py instead.")
        sys.exit(1)
    base_cfg = {k: v for k, v in exp_cfg.items() if k not in ("sweep", "include")}

    exp_name = Path(args.config).stem
    seeds = parse_seeds(args.seeds or sweep.get("seeds", 3))
    var_filter = set(args.variations.split(",")) if args.variations else None
    meth_filter = set(args.methods.split(",")) if args.methods else None

    jobs, skipped = [], 0
    for var in sweep.get("variations", [{"name": "base", "overrides": {}}]):
        vname = var["name"]
        if var_filter and vname not in var_filter:
            continue
        methods = var.get("methods", sweep.get("methods", ["PRIME"]))
        cfg = deep_merge(deep_merge(default_cfg, base_cfg), var.get("overrides", {}) or {})
        for method in methods:
            if meth_filter and method not in meth_filter:
                continue
            for seed in seeds:
                run_dir = (ROOT / "results" / exp_name / args.dataset / vname
                           / method.replace("/", "_") / f"seed_{seed:04d}")
                if not args.force and (run_dir / "summary.json").exists():
                    skipped += 1
                    continue
                jobs.append({
                    "experiment": exp_name, "dataset": args.dataset,
                    "variation": vname, "method": method, "seed": seed,
                    "cfg": cfg, "run_dir": str(run_dir),
                })

    total = len(jobs) + skipped
    print(f"{exp_name}: {total} runs total, {skipped} already done, {len(jobs)} to run "
          f"({args.parallel} workers)")
    if not jobs:
        return

    results = []
    t0 = time.perf_counter()
    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                tag = f"{r['variation']}/{r['method']}/s{r['seed']}"
                if r["status"] != "completed":
                    print(f"  [{i}/{len(jobs)}] {tag}: {r['status']} {r.get('error', r.get('diag_issues', ''))}")
                elif i % 25 == 0 or i == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(jobs)}] {tag} ok  ({el:.0f}s elapsed, "
                          f"ETA {el / i * (len(jobs) - i):.0f}s)")
    else:
        for i, j in enumerate(jobs, 1):
            r = run_one(j)
            results.append(r)
            print(f"  [{i}/{len(jobs)}] {r['variation']}/{r['method']}/s{r['seed']}: {r['status']}")

    import pandas as pd

    ok = sum(1 for r in results if r["status"] == "completed")
    bad = [r for r in results if r["status"] != "completed"]
    print(f"\n{exp_name}: {ok}/{len(results)} completed, {len(bad)} failed, "
          f"{time.perf_counter() - t0:.0f}s")

    out_root = ROOT / "results" / exp_name / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "traceback"} for r in results]) \
        .to_csv(out_root / "sweep_run_log.csv", index=False)
    if bad:
        pd.DataFrame(bad).to_csv(out_root / "failures.csv", index=False)
        print(f"Failures -> {out_root / 'failures.csv'}")


if __name__ == "__main__":
    main()
