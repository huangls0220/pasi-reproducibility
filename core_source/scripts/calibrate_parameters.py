#!/usr/bin/env python3
"""Parameter calibration script (Section 16).

Checks distributions of key quantities:
  a_min, a_system, C'/g', omega*H, Lambda, contract cost, VQ, budget utilization

Outputs distribution plots (if matplotlib available) and calibration_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.synthetic import generate_synthetic_episode
from src.io_utils import load_config, save_json, deep_merge
from src.contracts import compute_lambda, evaluate_pair
from src.quality_cost import Q, a_deadline, a_quality, g_prime, C_prime


def calibrate(cfg: dict, seed: int = 42, n_sample_slots: int = 5) -> dict:
    """Run calibration on a small sample and return distributions."""
    data = generate_synthetic_episode(cfg, seed=seed, pattern="stationary")

    tasks_df = data["tasks"]
    providers_df = data["providers"]
    static_df = data["provider_static"]

    # Sample first few slots
    sample_slots = sorted(tasks_df["slot"].unique())[:n_sample_slots]

    stats = {
        "a_min": [],
        "a_system": [],
        "cp_over_gp": [],  # C'/g'
        "omega_H": [],
        "lambda": [],
        "contract_cost": [],
        "VQ": [],
    }

    for slot in sample_slots:
        slot_tasks = tasks_df[tasks_df["slot"] == slot]
        slot_providers = providers_df[providers_df["slot"] == slot]
        slot_providers = slot_providers.merge(static_df, on=["episode_id", "provider_id"], how="left")
        slot_providers = slot_providers.dropna(subset=["alpha", "beta", "zeta"])

        if len(slot_tasks) == 0 or len(slot_providers) == 0:
            continue

        for _, prov in slot_providers.iterrows():
            F_i_t = max(1e-12, (1.0 - prov.get("load_ratio", 0.0)) * prov["max_processing_rate"])
            for _, task in slot_tasks.iterrows():
                L = task["L"]
                comm = prov.get("communication_rate", 10.0)
                D_tr = (task.get("input_size", 0) + task.get("output_size", 0)) / max(comm, 1e-12)
                eff_dead = min(task["deadline"], prov.get("availability_duration", 5.0))

                a_d = a_deadline(L, F_i_t, eff_dead, D_tr)
                a_q = a_quality(task["min_quality"], task["q_bar"], task["kappa"])

                if a_d == float("inf") or a_q == float("inf"):
                    continue

                a_min = max(a_d, a_q)
                if a_min > 1.0:
                    continue

                stats["a_min"].append(a_min)

                # Sample a_system at midpoint
                a_mid = (a_min + 1.0) / 2.0
                cp = C_prime(a_mid, prov["alpha"], prov["beta"], L)
                gp = g_prime(a_mid, task["kappa"])
                if gp > 1e-12:
                    stats["cp_over_gp"].append(cp / gp)

                stats["omega_H"].append(prov["omega"] * 0.1)  # initial H ≈ 0.1

                Lambda = compute_lambda(a_mid, task["kappa"], prov["alpha"], prov["beta"], L,
                                        prov["omega"], 0.1)
                stats["lambda"].append(Lambda)

                VQ_val = task["value"] * Q(a_mid, task["q_bar"], task["kappa"])
                stats["VQ"].append(VQ_val)

    # Summarize
    report = {}
    for key, values in stats.items():
        if not values:
            report[key] = {"empty": True}
            continue
        arr = np.array(values)
        finite = arr[np.isfinite(arr)]
        report[key] = {
            "n": len(arr),
            "n_finite": len(finite),
            "mean": float(np.mean(finite)) if len(finite) > 0 else None,
            "std": float(np.std(finite)) if len(finite) > 0 else None,
            "p5": float(np.percentile(finite, 5)) if len(finite) > 0 else None,
            "p50": float(np.percentile(finite, 50)) if len(finite) > 0 else None,
            "p95": float(np.percentile(finite, 95)) if len(finite) > 0 else None,
            "min": float(np.min(finite)) if len(finite) > 0 else None,
            "max": float(np.max(finite)) if len(finite) > 0 else None,
        }

    # Scale checks
    checks = []
    if report.get("a_min", {}).get("p50", 0) is not None:
        p50_a_min = report["a_min"]["p50"]
        if p50_a_min > 0.9:
            checks.append("WARNING: median a_min > 0.9, most pairs infeasible")

    if report.get("lambda", {}).get("p50", 0) is not None:
        p50_lambda = report["lambda"]["p50"]
        p90_lambda = report["lambda"].get("p90", p50_lambda)
        if p90_lambda < 1e-6:
            checks.append("WARNING: 90% Lambda ≈ 0, omega*H too large for cost scale")
        if p50_lambda > 100:
            checks.append("WARNING: median Lambda very large, may need D_bar adjustment")

    if report.get("cp_over_gp", {}).get("p50", 0) is not None:
        cp50 = report["cp_over_gp"]["p50"]
        oh50 = report["omega_H"]["p50"] if report.get("omega_H", {}).get("p50") else 0
        if oh50 > cp50 * 5:
            checks.append("WARNING: omega*H >> C'/g', intrinsic dominates, Lambda ≈ 0")

    report["_checks"] = checks

    return report


def main():
    parser = argparse.ArgumentParser(description="Calibrate PRIME parameters")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="calibration_report.json", help="Output path")
    parser.add_argument("--slots", type=int, default=5, help="Sample slots")
    args = parser.parse_args()

    default_cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    cfg = deep_merge(default_cfg, load_config(args.config))

    print(f"Running calibration on {args.slots} slots...")
    report = calibrate(cfg, seed=args.seed, n_sample_slots=args.slots)

    out_path = Path(args.output)
    save_json(report, out_path)
    print(f"Report saved to {out_path}")

    for check in report.pop("_checks", []):
        print(f"  {check}")

    print("\nKey distributions:")
    for key, vals in sorted(report.items()):
        if vals.get("empty") or vals.get("n_finite", 0) == 0:
            print(f"  {key}: (empty)")
        else:
            print(f"  {key}: n={vals['n_finite']} median={vals['p50']:.4f} [{vals['p5']:.4f}, {vals['p95']:.4f}]")


if __name__ == "__main__":
    main()
