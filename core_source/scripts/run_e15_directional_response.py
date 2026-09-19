"""E15: split-calibrated asymmetric response envelopes for PASI-U.

The first 20% of each frozen GeoLife episode is used only to calibrate the
ratio between realised and design response rates.  The interval is frozen
before the remaining 80% test region is loaded.  Test-side true parameters
are used only after execution for a fail-closed coverage audit and never enter
contract construction.

Completed calibration services provide two observable quantities: execution
time and realised normalised quality.  Given the task workload and provider
rate, execution time identifies effort; effort and quality then identify the
response rate in the paper's response model.  The original symmetric PASI-U
is retained as a comparator.
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

from scripts.run_e8_robustness import (bootstrap_ci, configure, joint_scenario,
                                       scenarios, tape_hash)
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

BASE_D_BAR = 10.0
BASE_BUDGET_RATIO = 0.70
CALIBRATION_ALPHA = 0.01
CALIBRATION_PAD = 1e-10
PROFILES = (
    ("symmetric", 1.0),
    ("symmetric", 5.0),
    ("directional", 1.0),
    ("directional", 5.0),
)


def response_scenarios() -> list[dict]:
    return [s for s in scenarios()
            if s["factor"] in {"control", "response_error"}] + [joint_scenario()]


def _errors(scenario: dict) -> tuple[float, float]:
    if scenario["factor"] == "joint":
        return float(scenario["response_error"]), float(scenario["cost_error"])
    response = float(scenario["level"]) if scenario["factor"] == "response_error" else 0.0
    cost = float(scenario["level"]) if scenario["factor"] == "cost_error" else 0.0
    return response, cost


def _configure_dataset(cfg: dict, folder: Path, region: str) -> dict:
    cfg["dataset"] = {
        "processed_dir": str(folder),
        "use_region": region,
        "service_radius_km": 3.0,
    }
    cfg["simulation"]["log_level"] = "selected"
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    return cfg


def _observable_response_ratios(result: dict, data: dict,
                                response_bias: float) -> np.ndarray:
    pair = result["pair_log"]
    if pair.empty:
        raise RuntimeError("calibration produced no selected service records")
    selected = pair[pair["selected"].astype(bool)].copy()
    tasks = data["tasks"][["slot", "task_id", "L", "input_size", "output_size",
                            "kappa"]].copy()
    providers = data["providers"][["slot", "provider_id", "communication_rate",
                                   "load_ratio"]].copy()
    static = data["provider_static"][["provider_id", "max_processing_rate"]].copy()
    merged = (selected.merge(tasks, on=["slot", "task_id"], how="left",
                             validate="many_to_one")
              .merge(providers, on=["slot", "provider_id"], how="left",
                     validate="many_to_one")
              .merge(static, on="provider_id", how="left", validate="many_to_one"))
    required = ["L", "input_size", "output_size", "kappa",
                "communication_rate", "load_ratio", "max_processing_rate", "total_delay",
                "normalized_quality"]
    if merged[required].isna().any().any():
        raise RuntimeError("calibration join contains missing observable fields")

    transit = ((merged["input_size"].to_numpy(float)
                + merged["output_size"].to_numpy(float))
               / np.maximum(merged["communication_rate"].to_numpy(float), 1e-12))
    compute_time = merged["total_delay"].to_numpy(float) - transit
    effective_rate = (merged["max_processing_rate"].to_numpy(float)
                      * (1.0 - merged["load_ratio"].to_numpy(float)))
    effort = (merged["L"].to_numpy(float)
              / np.maximum(effective_rate * compute_time, 1e-12))
    signal = np.clip(
        merged["normalized_quality"].to_numpy(float), 1e-12, 1.0 - 1e-12)
    inferred_kappa = -np.log1p(-signal) / np.maximum(effort, 1e-12)
    design_kappa = merged["kappa"].to_numpy(float) * max(0.01, 1.0 + response_bias)
    ratios = inferred_kappa / np.maximum(design_kappa, 1e-12)
    valid = (np.isfinite(ratios) & (ratios > 0.0)
             & np.isfinite(effort) & (effort > 1e-8) & (effort <= 1.0 + 1e-7))
    ratios = ratios[valid]
    if len(ratios) < 30:
        raise RuntimeError(f"only {len(ratios)} valid calibration services")
    return ratios


def _split_quantile_interval(values: np.ndarray,
                             alpha: float = CALIBRATION_ALPHA) -> tuple[float, float]:
    """Finite-sample order-statistic interval, padded only for round-off."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    lower_rank = max(1, int(math.floor((n + 1) * alpha / 2.0)))
    upper_rank = min(n, int(math.ceil((n + 1) * (1.0 - alpha / 2.0))))
    lower = max(1e-9, float(ordered[lower_rank - 1]) - CALIBRATION_PAD)
    upper = float(ordered[upper_rank - 1]) + CALIBRATION_PAD
    return lower, upper


def calibrate(folder: Path, scenario: dict, seed: int,
              n_providers: int) -> dict:
    response_error, _ = _errors(scenario)
    cfg = _configure_dataset(configure(scenario, n_providers), folder, "validation")
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    ratios = _observable_response_ratios(result, data, response_error)
    lower, upper = _split_quantile_interval(ratios)
    return {
        "calibration_n": int(len(ratios)),
        "ratio_lower": lower,
        "ratio_upper": upper,
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "ratio_expected_for_audit": 1.0 / max(0.01, 1.0 + response_error),
        "calibration_status": result["diagnostics"].get("status", "unknown"),
    }


def configure_test(scenario: dict, n_providers: int, envelope: str,
                   reserve_multiplier: float, calibration: dict) -> dict:
    cfg = configure(scenario, n_providers)
    response_error, cost_error = _errors(scenario)
    robust = {
        "enabled": True,
        "state_confidence_z": 3.290527,
        "cost_relative_bound": abs(cost_error),
        "missing_state_floor_zero": True,
    }
    if envelope == "directional":
        # The calibration point updates the design response used throughout
        # target construction.  The residual interval is expressed relative
        # to that corrected design value and remains available to the robust
        # feasibility/incentive calculations.  Only calibration-region
        # observables enter this update.
        ratio_point = math.sqrt(
            float(calibration["ratio_lower"])
            * float(calibration["ratio_upper"]))
        original_design_scale = max(0.01, 1.0 + response_error)
        corrected_design_scale = original_design_scale * ratio_point
        cfg["model_error"]["response_bias"] = corrected_design_scale - 1.0
        robust.update({
            "response_relative_bound": 0.0,
            "response_ratio_lower": float(calibration["ratio_lower"]) / ratio_point,
            "response_ratio_upper": float(calibration["ratio_upper"]) / ratio_point,
            "response_calibration_point": ratio_point,
            "response_corrected_design_scale": corrected_design_scale,
            "interpretation": (
                "response design and residual ratio interval frozen from the first 20% "
                "calibration region; state and cost bounds unchanged"),
        })
    elif envelope == "symmetric":
        robust.update({
            "response_relative_bound": abs(response_error),
            "interpretation": (
                "original scenario-magnitude symmetric response envelope; "
                "state and cost bounds unchanged"),
        })
    else:
        raise ValueError(f"unknown envelope: {envelope}")
    cfg["uncertainty_aware"] = robust
    cfg["contract"]["D_bar"] = BASE_D_BAR * reserve_multiplier
    cfg["matching"]["budget_ratio"] = BASE_BUDGET_RATIO * reserve_multiplier
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    return cfg


def run_profile(folder: Path, scenario: dict, seed: int, n_providers: int,
                envelope: str, reserve_multiplier: float,
                calibration: dict) -> dict:
    cfg = _configure_dataset(
        configure_test(scenario, n_providers, envelope, reserve_multiplier, calibration),
        folder, "test")
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    pair = result["pair_log"]
    selected = pair[pair["selected"].astype(bool)] if not pair.empty else pair
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    total_edges = float((slot["num_providers"] * slot["num_tasks"]).sum())
    feasible_edges = float(slot["num_feasible_pairs"].sum())
    response_error, cost_error = _errors(scenario)
    true_ratio = 1.0 / max(0.01, 1.0 + response_error)
    interval_contains_test_ratio = bool(
        calibration["ratio_lower"] - 1e-12 <= true_ratio
        <= calibration["ratio_upper"] + 1e-12)
    return {
        "run_id": (f"e15-{scenario['scenario']}-{envelope}-r{reserve_multiplier:g}"
                   f"-seed-{seed:03d}"),
        "variant": scenario["scenario"], "factor": scenario["factor"],
        "level": float(scenario["level"]), "seed": seed,
        "envelope": envelope, "reserve_multiplier": float(reserve_multiplier),
        "response_error": response_error, "cost_error": cost_error,
        "D_bar": float(cfg["contract"]["D_bar"]),
        "budget_ratio": float(cfg["matching"]["budget_ratio"]),
        "calibration_n": calibration["calibration_n"],
        "ratio_lower": calibration["ratio_lower"],
        "ratio_upper": calibration["ratio_upper"],
        "response_calibration_point": float(
            cfg["uncertainty_aware"].get("response_calibration_point", 1.0)),
        "corrected_response_bias": float(cfg["model_error"]["response_bias"]),
        "test_ratio_audit": true_ratio,
        "interval_contains_test_ratio": interval_contains_test_ratio,
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "platform_utility": float(summary["platform_utility"]),
        "reserve_activation_rate": (
            float(np.mean(selected["D_star"].to_numpy(float) > BASE_D_BAR + 1e-9))
            if len(selected) else 0.0),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "event_tape_hash": tape_hash(folder),
        "run_status": result["diagnostics"].get("status", "unknown"),
    }


def run_job(episodes: Path, scenario: dict, seed: int) -> tuple[dict, list[dict]]:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    calibration = calibrate(folder, scenario, seed, int(meta["n_providers"]))
    cal_row = dict(calibration)
    cal_row.update({
        "variant": scenario["scenario"], "factor": scenario["factor"],
        "level": float(scenario["level"]), "seed": seed,
        "event_tape_hash": tape_hash(folder),
    })
    rows = [run_profile(folder, scenario, seed, int(meta["n_providers"]),
                        envelope, reserve, calibration)
            for envelope, reserve in PROFILES]
    return cal_row, rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pps", "service_coverage", "qos_violation_rate",
               "target_miss_rate", "under_incentive_rate", "legal_edge_rate",
               "platform_utility", "reserve_activation_rate"]
    rows = []
    groups = frame.groupby(["envelope", "reserve_multiplier"], sort=False)
    for profile_index, ((envelope, reserve), profile) in enumerate(groups, 1):
        control = profile[profile["variant"] == "control"].set_index("seed")
        for variant_index, (variant, sub) in enumerate(
                profile.groupby("variant", sort=False), 1):
            sub = sub.set_index("seed").sort_index()
            base = control.loc[sub.index]
            row = {
                "variant": variant,
                "factor": sub["factor"].iloc[0],
                "level": float(sub["level"].iloc[0]),
                "envelope": envelope,
                "reserve_multiplier": float(reserve),
                "n_pairs": len(sub),
                "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all()),
                "all_test_ratios_contained": bool(sub["interval_contains_test_ratio"].all()),
                "mean_calibration_n": float(sub["calibration_n"].mean()),
                "mean_ratio_lower": float(sub["ratio_lower"].mean()),
                "mean_ratio_upper": float(sub["ratio_upper"].mean()),
            }
            for metric_index, metric in enumerate(metrics, 1):
                values = sub[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                lo, hi = bootstrap_ci(
                    delta, seed=profile_index * 10000 + variant_index * 100 + metric_index)
                alo, ahi = bootstrap_ci(
                    values, seed=profile_index * 20000 + variant_index * 100 + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = alo
                row[f"{metric}_ci_high"] = ahi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = lo
                row[f"delta_{metric}_ci_high"] = hi
            row["gate_pass"] = bool(
                row["all_runs_ok"]
                and row["all_test_ratios_contained"]
                and row["under_incentive_rate_ci_high"] <= 0.005
                and row["qos_violation_rate_ci_high"] <= 0.005
                and row["target_miss_rate_ci_high"] <= 0.005
                and row["delta_service_coverage_ci_low"] >= -0.01)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    scens = response_scenarios()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(scenario, seed) for scenario in scens for seed in seeds]
    out_dir = ROOT.parent / "results" / "e15_directional_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows: list[dict] = []
    result_rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, args.episodes, scenario, seed):
                   (scenario["scenario"], seed) for scenario, seed in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            variant, seed = futures[future]
            cal, rows = future.result()
            calibration_rows.append(cal)
            result_rows.extend(rows)
            print(f"[{done:04d}/{len(jobs)}] E15 {variant} seed={seed}", flush=True)

    calibration_frame = pd.DataFrame(calibration_rows).sort_values(
        ["factor", "level", "seed"])
    result_frame = pd.DataFrame(result_rows).sort_values(
        ["envelope", "reserve_multiplier", "factor", "level", "seed"])
    summary = summarize(result_frame)
    suffix = "pilot" if args.pilot else "formal"
    calibration_frame.to_csv(out_dir / f"e15_{suffix}_calibration.csv", index=False)
    result_frame.to_csv(out_dir / f"e15_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out_dir / f"e15_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E15 split-calibrated asymmetric response envelope",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium workload",
        "temporal_split": "first 20% calibration; remaining 80% test",
        "calibration_signal": (
            "response rate inferred from completed-service execution time and "
            "realised normalised quality"),
        "calibration_alpha": CALIBRATION_ALPHA,
        "profiles": [{"envelope": e, "reserve_multiplier": r} for e, r in PROFILES],
        "test_side_true_parameter_used_for_decision": False,
        "test_side_ratio_used_after_execution_for_audit_only": True,
        "scenario_count": len(scens), "seeds": seeds,
        "test_runs": len(result_frame),
        "calibration_runs": len(calibration_frame),
        "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / f"e15_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["variant", "envelope", "reserve_multiplier",
                   "mean_service_coverage", "delta_service_coverage",
                   "mean_pps", "delta_pps", "mean_target_miss_rate",
                   "mean_qos_violation_rate", "all_test_ratios_contained",
                   "gate_pass"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
