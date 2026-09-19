"""E16: public QUAC-F contract baseline on shared frozen GeoLife tapes."""

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
METHODS = ("PASI", "MOI", "QUAC-F")
SOURCE_DOI = "10.1109/MASS.2017.45"
SOURCE_URL = "https://ecs.syr.edu/faculty/tang/Pub/Tang-MASS17.pdf"


def run_one(episode_root: Path, workload: str, method: str, seed: int) -> dict:
    folder = episode_root / workload / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    control = {"scenario": "control", "factor": "control", "level": 0.0}
    cfg = configure(control, int(meta["n_providers"]))
    cfg["dataset"] = {"processed_dir": str(folder), "use_region": "all",
                      "service_radius_km": 3.0}
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    cfg["simulation"]["log_level"] = "selected"
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)

    started = time.perf_counter()
    result = Simulator(cfg, data, method=method, seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    total_edges = float((slot["num_providers"] * slot["num_tasks"]).sum())
    feasible_edges = float(slot["num_feasible_pairs"].sum())
    selected = result["pair_log"]
    selected = selected[selected["selected"].astype(bool)]
    return {
        "run_id": f"e16-{workload}-{method}-seed-{seed:03d}",
        "dataset": "geolife", "workload": workload, "method": method,
        "seed": seed, "N": int(meta["n_providers"]), "M": 4,
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "mean_quality": float(slot["mean_quality"].dropna().mean()),
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "platform_utility": float(summary["platform_utility"]),
        "mean_base_payment": (float(selected["base_payment"].mean())
                              if len(selected) else math.nan),
        "negative_intercept_rate": (float((selected["base_payment"] < 0.0).mean())
                                    if len(selected) else math.nan),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "event_tape_hash": tape_hash(folder),
        "simulator_diagnostic_status": result["diagnostics"].get("status", "unknown"),
        "run_status": "completed",
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pps", "service_coverage", "mean_quality", "qos_violation_rate",
               "under_incentive_rate", "target_miss_rate", "legal_edge_rate",
               "platform_utility", "core_runtime_s"]
    rows = []
    for workload_index, (workload, work) in enumerate(
            frame.groupby("workload", sort=False), 1):
        reference = work[work["method"] == "PASI"].set_index("seed").sort_index()
        for method_index, (method, sub) in enumerate(work.groupby("method", sort=False), 1):
            sub = sub.set_index("seed").sort_index()
            base = reference.loc[sub.index]
            row = {"workload": workload, "method": method, "reference": "PASI",
                   "n_pairs": len(sub),
                   "all_runs_completed": bool((sub["run_status"] == "completed").all())}
            for metric_index, metric in enumerate(metrics, 1):
                values = sub[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                mean_lo, mean_hi = bootstrap_ci(
                    values, 1600 + 100 * workload_index + 10 * method_index + metric_index)
                delta_lo, delta_hi = bootstrap_ci(
                    delta, 2600 + 100 * workload_index + 10 * method_index + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = mean_lo
                row[f"{metric}_ci_high"] = mean_hi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = delta_lo
                row[f"delta_{metric}_ci_high"] = delta_hi
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
    jobs = [(args.episodes, workload, method, seed)
            for workload in WORKLOADS for method in METHODS for seed in seeds]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[{index:04d}/{len(jobs)}] {row['workload']} {row['method']} "
                  f"seed={row['seed']}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["workload", "method", "seed"])
    paired = summarize(frame)
    out = ROOT.parent / "results" / "e16_quac_public_baseline"
    out.mkdir(parents=True, exist_ok=True)
    prefix = "e16_pilot" if args.pilot else "e16_formal"
    frame.to_csv(out / f"{prefix}_seed_results.csv", index=False)
    paired.to_csv(out / f"{prefix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E16 public QUAC-F isomorphic contract baseline",
        "source_title": "QUAC: Quality-Aware Contract-Based Incentive Mechanisms for Crowdsensing",
        "source_doi": SOURCE_DOI, "source_pdf": SOURCE_URL,
        "implementation_status": "independent implementation from public equations; not author code",
        "mapping": ("risk-neutral QUAC-F A+Bq with q=q_bar*g(a), D=B*q_bar, "
                    "IR-binding signed intercept, shared quality/cost, provider capacity, "
                    "budget, frozen event tape, and coverage-first matching"),
        "workloads": list(WORKLOADS), "methods": list(METHODS), "seeds": seeds,
        "matching_solver": ("exact subset enumeration for at most 12 tasks; "
                            "exact two-stage binary MILP above 12 tasks; identical "
                            "coverage-first/payment-second objective"),
        "runs": len(frame), "elapsed_seconds": time.perf_counter() - started,
        "raw_or_derived_trace_redistributed": False,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(paired[["workload", "method", "mean_service_coverage", "mean_pps",
                  "mean_qos_violation_rate", "mean_under_incentive_rate",
                  "mean_target_miss_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
