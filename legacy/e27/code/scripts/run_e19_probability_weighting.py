"""E19: fixed-parameter Prelec probability-weighting sensitivity.

The experiment reuses the frozen GeoLife low/medium/high event tapes.  It
changes only the provider probability-weighting parameter and compares PASI
with its matched MOI control.  Three nonlinear settings and the identity map
are evaluated with 30 paired seeds per workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e8_robustness import bootstrap_ci, configure, tape_hash
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


WORKLOADS = ("low", "medium", "high")
METHODS = ("MOI", "PASI")
ZETA_LEVELS = (0.65, 0.80, 1.00, 1.20)


def run_one(episode_root: Path, workload: str, method: str,
            zeta: float, seed: int) -> dict:
    folder = episode_root / workload / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    control = {"scenario": "control", "factor": "control", "level": 0.0}
    cfg = configure(control, int(meta["n_providers"]))
    cfg["dataset"] = {
        "processed_dir": str(folder), "use_region": "all",
        "service_radius_km": 3.0,
    }
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    cfg["simulation"]["log_level"] = "selected"
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)

    static = data["provider_static"].copy()
    static["zeta"] = float(zeta)
    data["provider_static"] = static

    started = time.perf_counter()
    result = Simulator(cfg, data, method=method, seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    pair = result["pair_log"]
    selected = pair[pair["selected"].astype(bool)] if not pair.empty else pair
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    return {
        "run_id": f"e19-{workload}-z{zeta:.2f}-{method}-seed-{seed:03d}",
        "dataset": "geolife", "workload": workload, "method": method,
        "zeta": float(zeta), "weighting": "identity" if zeta == 1.0 else "Prelec",
        "seed": int(seed), "N": int(meta["n_providers"]), "M": 4,
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "mean_probability": (float(selected["p_star"].mean())
                             if len(selected) else math.nan),
        "mean_bonus": (float(selected["D_star"].mean())
                       if len(selected) else math.nan),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": float(elapsed),
        "event_tape_hash": tape_hash(folder),
        "run_status": result["diagnostics"].get("status", "unknown"),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pps", "service_coverage", "qos_violation_rate",
               "under_incentive_rate", "target_miss_rate",
               "mean_probability", "mean_bonus"]
    rows = []
    for workload_index, ((workload, zeta), work) in enumerate(
            frame.groupby(["workload", "zeta"], sort=False), 1):
        reference = work[work["method"] == "MOI"].set_index("seed").sort_index()
        for method_index, (method, sub) in enumerate(work.groupby("method", sort=False), 1):
            sub = sub.set_index("seed").sort_index()
            base = reference.loc[sub.index]
            row = {
                "workload": workload, "zeta": float(zeta), "method": method,
                "reference": "MOI", "n_pairs": int(len(sub)),
                "all_runs_completed": bool(sub["run_status"].isin(
                    ["ok", "completed"]).all()),
            }
            for metric_index, metric in enumerate(metrics, 1):
                values = sub[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                mean_lo, mean_hi = bootstrap_ci(
                    values, 19000 + 100 * workload_index + 10 * method_index + metric_index)
                delta_lo, delta_hi = bootstrap_ci(
                    delta, 29000 + 100 * workload_index + 10 * method_index + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = mean_lo
                row[f"{metric}_ci_high"] = mean_hi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = delta_lo
                row[f"delta_{metric}_ci_high"] = delta_hi
            moi_pps = base["pps"].to_numpy(float)
            saving = (moi_pps - sub["pps"].to_numpy(float)) / np.maximum(moi_pps, 1e-12)
            saving_lo, saving_hi = bootstrap_ci(
                saving, 39000 + 100 * workload_index + 10 * method_index)
            row["mean_pps_saving_rate"] = float(saving.mean())
            row["pps_saving_rate_ci_low"] = saving_lo
            row["pps_saving_rate_ci_high"] = saving_hi
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(args.episodes, workload, method, zeta, seed)
            for workload in WORKLOADS for zeta in ZETA_LEVELS
            for method in METHODS for seed in seeds]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[{index:04d}/{len(jobs)}] {row['workload']} zeta={row['zeta']:.2f} "
                  f"{row['method']} seed={row['seed']}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["workload", "zeta", "method", "seed"])
    paired = summarize(frame)
    out = ROOT.parent / "results" / "e19_probability_weighting"
    out.mkdir(parents=True, exist_ok=True)
    prefix = "e19_pilot" if args.pilot else "e19_formal"
    frame.to_csv(out / f"{prefix}_seed_results.csv", index=False)
    paired.to_csv(out / f"{prefix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E19 fixed-parameter Prelec probability-weighting sensitivity",
        "weighting_function": "W(p)=exp(-(-ln p)^zeta)",
        "zeta_levels": list(ZETA_LEVELS),
        "nonlinear_levels": [z for z in ZETA_LEVELS if z != 1.0],
        "workloads": list(WORKLOADS), "methods": list(METHODS),
        "seeds": seeds, "runs": int(len(frame)),
        "changed_factor": "provider zeta only",
        "shared_inputs": ("frozen GeoLife event tape, provider/task attributes except zeta, "
                          "quality/cost model, budget, capacities, and exact lexicographic matcher"),
        "raw_or_derived_trace_redistributed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(paired[["workload", "zeta", "method", "mean_service_coverage",
                  "mean_pps", "mean_pps_saving_rate", "mean_qos_violation_rate"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
