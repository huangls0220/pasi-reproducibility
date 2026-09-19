"""Run PASI-U on the frozen E8 grid and compare it with retained PASI rows.

PASI-U changes contract construction only.  It protects the nominal target
against declared state, response, and cost uncertainty envelopes while all
realized outcomes continue to be evaluated with the frozen true-side model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e8_robustness import (bootstrap_ci, configure, joint_scenario,
                                       run_one, scenarios)


def configure_uncertainty(scenario: dict, n_providers: int) -> dict:
    cfg = configure(scenario, n_providers)
    factor = scenario["factor"]
    response = (float(scenario["response_error"]) if factor == "joint"
                else float(scenario["level"]) if factor == "response_error" else 0.0)
    cost = (float(scenario["cost_error"]) if factor == "joint"
            else float(scenario["level"]) if factor == "cost_error" else 0.0)
    cfg["uncertainty_aware"] = {
        "enabled": True,
        "state_confidence_z": 3.290527,
        "response_relative_bound": abs(response),
        "cost_relative_bound": abs(cost),
        "missing_state_floor_zero": True,
        "interpretation": (
            "99.95% one-sided Gaussian state bound; deterministic delay bound; "
            "zero state-credit floor when LOCF missingness is active; symmetric "
            "relative response/cost envelopes centred on design estimates"
        ),
    }
    return cfg


def run_one_u(episodes: Path, scenario: dict, seed: int) -> dict:
    # Reuse the frozen runner after wrapping its configure function locally.
    # The import-level function is replaced only inside this worker process.
    import scripts.run_e8_robustness as e8

    original = e8.configure
    e8.configure = configure_uncertainty
    try:
        row = run_one(episodes, scenario, seed)
    finally:
        e8.configure = original
    row["run_id"] = row["run_id"].replace("e8-", "e11-pasi-u-")
    row["mechanism"] = "PASI-U"
    row["uncertainty_response_bound"] = abs(float(row["response_error"]))
    row["uncertainty_cost_bound"] = abs(float(row["cost_error"]))
    row["uncertainty_state_confidence"] = 0.9995
    return row


def _bootstrap_mean_ci(values: np.ndarray, seed: int,
                       draws: int = 10000) -> tuple[float, float]:
    return bootstrap_ci(values, seed=seed, draws=draws)


def compare(robust: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pps", "service_coverage", "qos_violation_rate",
               "target_miss_rate", "under_incentive_rate", "legal_edge_rate",
               "platform_utility"]
    rows = []
    for idx, (variant, ru) in enumerate(robust.groupby("variant", sort=False), 1):
        ba = baseline[baseline["variant"] == variant]
        merged = ru.merge(ba, on="seed", suffixes=("_u", "_base"), validate="one_to_one")
        if len(merged) != len(ru):
            raise RuntimeError(f"missing retained PASI pairs for {variant}")
        row = {
            "variant": variant,
            "factor": ru["factor"].iloc[0],
            "level": float(ru["level"].iloc[0]),
            "n_pairs": len(merged),
            "all_runs_ok": bool(ru["run_status"].isin(["ok", "completed"]).all()),
        }
        for j, metric in enumerate(metrics, 1):
            u = merged[f"{metric}_u"].to_numpy(float)
            b = merged[f"{metric}_base"].to_numpy(float)
            delta = u - b
            d_lo, d_hi = bootstrap_ci(delta, seed=idx * 100 + j)
            a_lo, a_hi = _bootstrap_mean_ci(u, seed=idx * 1000 + j)
            row[f"mean_pasi_u_{metric}"] = float(u.mean())
            row[f"mean_pasi_{metric}"] = float(b.mean())
            row[f"delta_{metric}"] = float(delta.mean())
            row[f"delta_{metric}_ci_low"] = d_lo
            row[f"delta_{metric}_ci_high"] = d_hi
            row[f"pasi_u_{metric}_ci_low"] = a_lo
            row[f"pasi_u_{metric}_ci_high"] = a_hi
        row["safety_gate_pass"] = bool(
            row["pasi_u_under_incentive_rate_ci_high"] <= 0.005
            and row["pasi_u_qos_violation_rate_ci_high"] <= 0.005
            and row["pasi_u_target_miss_rate_ci_high"] <= 0.005
            and row["delta_service_coverage_ci_low"] >= -0.01
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline-formal", type=Path, required=True)
    parser.add_argument("--baseline-joint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    scens = scenarios() + [joint_scenario()]
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(scenario, seed) for scenario in scens for seed in seeds]
    out_dir = ROOT.parent / "results" / "e11_pasi_u"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_u, args.episodes, s, seed): (s["scenario"], seed)
                   for s, seed in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            variant, seed = futures[future]
            rows.append(future.result())
            print(f"[{done:04d}/{len(jobs)}] PASI-U {variant} seed={seed}", flush=True)

    robust = pd.DataFrame(rows).sort_values(["factor", "level", "seed"])
    baseline = pd.concat([
        pd.read_csv(args.baseline_formal),
        pd.read_csv(args.baseline_joint),
    ], ignore_index=True).drop_duplicates(["variant", "seed"], keep="first")
    summary = compare(robust, baseline)
    suffix = "pilot" if args.pilot else "formal"
    robust.to_csv(out_dir / f"e11_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out_dir / f"e11_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E11 PASI-U uncertainty-envelope repair",
        "dataset": "GeoLife GPS Trajectories 1.3 / medium workload",
        "scenario_count": len(scens),
        "seeds": seeds,
        "runs": len(robust),
        "paired_baseline": "retained frozen E8 PASI rows",
        "state_bound": "99.95% one-sided Gaussian; delay drift; zero LOCF floor",
        "model_bounds": "scenario-magnitude symmetric relative envelopes",
        "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
    }
    (out_dir / f"e11_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["variant", "mean_pasi_u_pps", "delta_pps",
                   "mean_pasi_u_service_coverage", "delta_service_coverage",
                   "mean_pasi_u_target_miss_rate", "mean_pasi_u_qos_violation_rate",
                   "safety_gate_pass"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
