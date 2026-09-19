"""E20: response-envelope width versus safety and service coverage.

The original response rate is frozen as the design-side value.  Each declared
symmetric envelope is fixed before the held-out test, and the stress trajectory
is scaled to remain inside that envelope.  Contract construction receives only
the frozen point or the declared envelope; realised ratios enter only the
after-execution audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.experiment_protocol import protocol_fields

DEFAULT_WIDTHS = (0.05, 0.10, 0.20, 0.30)
PROFILES = ("frozen_point", "fixed_envelope")
SCENARIOS = (
    "stable", "down_step", "up_step", "down_ramp", "alternating",
    "hidden_blocks",
)


def slot_ratios(slots: np.ndarray, scenario: str, seed: int,
                lower: float, upper: float) -> np.ndarray:
    unique = np.sort(np.unique(slots))
    position = {int(slot): i for i, slot in enumerate(unique)}
    n = max(len(unique), 1)
    index = np.asarray([position[int(slot)] for slot in slots], dtype=int)
    fraction = index / max(n - 1, 1)
    if scenario == "stable":
        ratio = np.ones(len(slots))
    elif scenario == "down_step":
        ratio = np.where(fraction < 0.5, 1.0, lower)
    elif scenario == "up_step":
        ratio = np.where(fraction < 0.5, 1.0, upper)
    elif scenario == "down_ramp":
        ratio = 1.0 - (1.0 - lower) * fraction
    elif scenario == "alternating":
        ratio = np.where(index % 2 == 0, lower, upper)
    elif scenario == "hidden_blocks":
        rng = np.random.default_rng(900_000 + seed)
        block_values = rng.uniform(lower, upper,
                                   size=max(1, int(np.ceil(n / 8))))
        ratio = block_values[np.minimum(index // 8, len(block_values) - 1)]
    else:
        raise ValueError(scenario)
    return ratio.astype(float)


def build_config(folder: Path, n_providers: int, profile: str,
                 lower: float, upper: float) -> dict:
    control = {"scenario": "control", "factor": "control", "level": 0.0}
    cfg = configure(control, n_providers)
    cfg["dataset"] = {"processed_dir": str(folder), "use_region": "test",
                      "service_radius_km": 3.0}
    cfg["simulation"]["log_level"] = "selected"
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    if profile == "fixed_envelope":
        cfg["uncertainty_aware"] = {
            "enabled": True,
            "state_confidence_z": 3.290527,
            "cost_relative_bound": 0.0,
            "response_relative_bound": 0.0,
            "response_ratio_lower": lower,
            "response_ratio_upper": upper,
            "missing_state_floor_zero": True,
            "interpretation": (
                "fixed response envelope declared before the held-out test; "
                "no test-side response observation enters contract construction"),
        }
        cfg["contract"]["D_bar"] = 50.0
        cfg["matching"]["budget_ratio"] = 3.5
    elif profile != "frozen_point":
        raise ValueError(profile)
    return cfg


def run_one(episodes: Path, half_width: float, scenario: str,
            profile: str, seed: int) -> dict:
    lower = 1.0 - half_width
    upper = 1.0 + half_width
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), profile, lower, upper)
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed,
                         lower, upper)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks

    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    total_edges = float((slot["num_providers"] * slot["num_tasks"]).sum())
    feasible_edges = float(slot["num_feasible_pairs"].sum())
    within = bool(np.all((ratios >= lower - 1e-12)
                         & (ratios <= upper + 1e-12)))
    return {
        "run_id": (f"e20-w{half_width:.2f}-{scenario}-{profile}-"
                   f"seed-{seed:03d}"),
        "half_width": float(half_width),
        "bound_lower": float(lower),
        "bound_upper": float(upper),
        "scenario": scenario, "profile": profile, "seed": seed,
        "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()),
        "within_declared_envelope": within,
        "test_side_response_used_for_decision": False,
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "platform_utility": float(summary["platform_utility"]),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "event_tape_hash": tape_hash(folder),
        "run_status": result["diagnostics"].get("status", "unknown"),
        "reserved_budget": float(slot["budget_used"].sum()),
        "available_budget": float(slot["budget"].sum()),
        **protocol_fields(cfg, summary, result["diagnostics"], __file__),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("pps", "service_coverage", "qos_violation_rate",
               "under_incentive_rate", "target_miss_rate", "legal_edge_rate",
               "platform_utility")
    rows = []
    for cell_index, ((half_width, scenario), group) in enumerate(
            frame.groupby(["half_width", "scenario"], sort=True), 1):
        reference = group[group["profile"] == "frozen_point"].set_index("seed")
        for profile_index, (profile, sub) in enumerate(
                group.groupby("profile", sort=False), 1):
            sub = sub.set_index("seed").sort_index()
            base = reference.loc[sub.index]
            row = {
                "half_width": float(half_width),
                "bound_lower": float(1.0 - half_width),
                "bound_upper": float(1.0 + half_width),
                "scenario": scenario, "profile": profile, "n_pairs": len(sub),
                "within_declared_envelope": bool(sub["within_declared_envelope"].all()),
                "test_side_response_used_for_decision": bool(
                    sub["test_side_response_used_for_decision"].any()),
                "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all()),
                "ratio_min_audit": float(sub["ratio_min_audit"].min()),
                "ratio_max_audit": float(sub["ratio_max_audit"].max()),
            }
            for metric_index, metric in enumerate(metrics, 1):
                values = sub[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                if len(sub) < 2:
                    lo = hi = dlo = dhi = np.nan
                else:
                    lo, hi = bootstrap_ci(values, 31_000 + 100 * cell_index
                                          + 10 * profile_index + metric_index)
                    dlo, dhi = bootstrap_ci(delta, 41_000 + 100 * cell_index
                                            + 10 * profile_index + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = dlo
                row[f"delta_{metric}_ci_high"] = dhi
            row["safety_gate"] = bool(
                row["all_runs_ok"]
                and not row["test_side_response_used_for_decision"]
                and row["qos_violation_rate_ci_high"] <= 0.005
                and row["under_incentive_rate_ci_high"] <= 0.005
                and row["target_miss_rate_ci_high"] <= 0.005)
            # A bootstrap descriptive interval is not a risk certificate;
            # a single episode cannot supply a between-episode interval.
            row["population_risk_certified"] = False
            if len(sub) < 2:
                for key in list(row):
                    if key.endswith("_ci_low") or key.endswith("_ci_high"):
                        row[key] = np.nan
                row["safety_gate"] = False
            row["safety_gate_interpretation"] = "legacy_descriptive_screen_not_certification"
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--widths", type=float, nargs="+",
                        default=list(DEFAULT_WIDTHS))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    widths = sorted(set(float(value) for value in args.widths))
    if not widths or any(value <= 0.0 or value >= 0.5 for value in widths):
        raise ValueError("each envelope half-width must be in (0, 0.5)")
    jobs = [(args.episodes, width, scenario, profile, seed)
            for width in widths for scenario in SCENARIOS
            for profile in PROFILES for seed in seeds]
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, width, scenario, profile, seed = futures[future]
            rows.append(future.result())
            print((f"[{index:04d}/{len(jobs)}] E20 width={width:.2f} "
                   f"{scenario} {profile} seed={seed}"),
                  flush=True)
    frame = pd.DataFrame(rows).sort_values(
        ["half_width", "scenario", "profile", "seed"])
    summary = summarize(frame)
    out = ROOT.parent / "results" / "e20_response_envelope_tradeoff"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out / f"e20_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out / f"e20_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E20 response-envelope width and coverage tradeoff",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium test region",
        "design_response": "original task response rate frozen before test",
        "declared_envelope_half_widths": widths,
        "test_side_response_used_for_decision": False,
        "trajectory_construction": "each stress trajectory remains within its declared envelope",
        "scenarios": list(SCENARIOS), "profiles": list(PROFILES),
        "seeds": seeds, "runs": len(frame),
        "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_protocol_version": "r53-service-outcomes-v1",
        "contract_protocol_versions": sorted(frame["contract_protocol_version"].unique().tolist()),
        "core_sha256_values": sorted(frame["core_sha256"].unique().tolist()),
        "configuration_manifest": frame[["half_width", "profile", "config_sha256",
            "contract_D_bar", "matching_budget_ratio", "decision_cost_basis",
            "settlement_cost_basis", "guarantee_scope"]].drop_duplicates().to_dict("records"),
        "population_risk_certified": False,
    }
    (out / f"e20_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["half_width", "scenario", "profile", "within_declared_envelope",
                   "mean_service_coverage", "mean_pps", "mean_target_miss_rate",
                   "mean_qos_violation_rate", "safety_gate"]].to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
