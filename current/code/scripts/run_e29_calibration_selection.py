"""E29: calibration-only selection on the E23 safety--coverage frontier.

The odd GeoLife-medium seeds form the calibration split and the even seeds are
held out.  The selection file is written before any held-out job is launched.
See ``work/round45-e29-protocol.md`` for the frozen protocol.
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
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e20_envelope_tradeoff import SCENARIOS, build_config, slot_ratios
from scripts.run_e22_adaptive_geolife import ensure_episodes, metrics
from scripts.run_e23_guardrail_frontier import (
    GUARDRAIL_WEIGHTS,
    LOWER,
    PROFILES,
    UPPER,
    profile_weight,
)
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

ALPHA = 0.005
DELTA = 0.05
BOOTSTRAP_REPS = 10_000
CALIBRATION_SEEDS = tuple(range(1, 31, 2))
TEST_SEEDS = tuple(range(2, 31, 2))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clopper_pearson_upper(successes: int, trials: int,
                           tail_probability: float) -> float:
    """Exact one-sided binomial upper confidence endpoint."""
    if trials <= 0:
        return float("nan")
    if successes >= trials:
        return 1.0
    return float(beta.ppf(1.0 - tail_probability,
                          successes + 1, trials - successes))


def run_one(episodes: Path, split: str, scenario: str,
            profile: str, seed: int) -> dict:
    weight = profile_weight(profile)
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), "fixed_envelope",
                       LOWER, UPPER)
    cfg["uncertainty_aware"]["adaptive_response"] = {
        "enabled": True,
        "min_history": 24,
        "window": 96,
        "miscoverage_alpha": 0.05,
        "safety_margin": 0.02,
        "global_lower": LOWER,
        "global_upper": UPPER,
        "global_guardrail_weight": weight,
    }
    data = load_trace_episode("geolife", cfg, seed=seed,
                              processed_dir=folder)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed,
                         LOWER, UPPER)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks

    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    return {
        "run_id": f"e29-{split}-{scenario}-{profile}-seed-{seed:03d}",
        "split": split,
        "scenario": scenario,
        "profile": profile,
        "global_guardrail_weight": weight,
        "seed": seed,
        "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()),
        "within_global_envelope": bool(np.all(
            (ratios >= LOWER - 1e-12) & (ratios <= UPPER + 1e-12))),
        "test_side_lookahead": False,
        "runtime_s": time.perf_counter() - started,
        "event_tape_hash": tape_hash(folder),
        **metrics(result),
    }


def run_grid(episodes: Path, split: str, seeds: tuple[int, ...],
             workers: int) -> pd.DataFrame:
    jobs = [(episodes, split, scenario, profile, seed)
            for scenario in SCENARIOS for profile in PROFILES for seed in seeds]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, _, scenario, profile, seed = futures[future]
            rows.append(future.result())
            print(f"[{index:04d}/{len(jobs)}] E29 {split} {scenario} "
                  f"{profile} seed={seed}", flush=True)
    return pd.DataFrame(rows).sort_values(["scenario", "profile", "seed"])


def calibration_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    cell_tail = DELTA / (len(SCENARIOS) * len(GUARDRAIL_WEIGHTS))
    rows: list[dict] = []
    for (scenario, profile), cell in frame.groupby(
            ["scenario", "profile"], sort=False):
        assigned = int(cell["assigned"].sum())
        violations = int(cell["qos_violations"].sum())
        rows.append({
            "scenario": scenario,
            "profile": profile,
            "global_guardrail_weight": float(
                cell["global_guardrail_weight"].iloc[0]),
            "n_seeds": int(cell["seed"].nunique()),
            "assigned_total": assigned,
            "qos_violations_total": violations,
            "pooled_qos_violation_rate": violations / max(assigned, 1),
            "simultaneous_cp_upper": clopper_pearson_upper(
                violations, assigned, cell_tail),
            "cell_tail_probability": cell_tail,
        })
    return pd.DataFrame(rows).sort_values(
        ["global_guardrail_weight", "scenario"])


def cluster_bootstrap_upper(frame: pd.DataFrame, reps: int) -> dict[float, float]:
    """Paired seed-cluster sensitivity bound for max-scenario QoS risk."""
    seeds = np.sort(frame["seed"].unique())
    rng = np.random.default_rng(290_045)
    outputs: dict[float, float] = {}
    for weight in GUARDRAIL_WEIGHTS:
        cell = frame[frame["global_guardrail_weight"] == weight]
        indexed = {
            scenario: group.set_index("seed")[["qos_violations", "assigned"]]
            .reindex(seeds)
            for scenario, group in cell.groupby("scenario", sort=False)
        }
        boot_max = np.empty(reps, dtype=float)
        for rep in range(reps):
            draw = rng.integers(0, len(seeds), size=len(seeds))
            scenario_rates = []
            for scenario in SCENARIOS:
                values = indexed[scenario].iloc[draw]
                scenario_rates.append(float(values["qos_violations"].sum()) /
                                      max(float(values["assigned"].sum()), 1.0))
            boot_max[rep] = max(scenario_rates)
        outputs[float(weight)] = float(np.quantile(boot_max, 0.95))
    return outputs


def select_guardrail(bounds: pd.DataFrame,
                     cluster_upper: dict[float, float]) -> dict:
    worst = bounds.groupby("global_guardrail_weight")[
        "simultaneous_cp_upper"].max().to_dict()
    eligible: list[float] = []
    for weight in GUARDRAIL_WEIGHTS:
        if all(worst[float(candidate)] <= ALPHA
               for candidate in GUARDRAIL_WEIGHTS if candidate >= weight):
            eligible.append(float(weight))
    statistical_pass = bool(eligible)
    selected = min(eligible) if eligible else 1.0
    return {
        "selected_global_guardrail_weight": selected,
        "selection_basis": ("smallest candidate passing the monotone simultaneous "
                            "calibration bound" if statistical_pass else
                            "predeclared structural fallback"),
        "statistical_pass": statistical_pass,
        "risk_budget_alpha": ALPHA,
        "familywise_delta": DELTA,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "test_seeds_locked_before_test": list(TEST_SEEDS),
        "worst_scenario_simultaneous_cp_upper_by_g": {
            str(key): float(value) for key, value in worst.items()},
        "seed_cluster_bootstrap_max_scenario_upper95_by_g_sensitivity_only": {
            str(key): float(value) for key, value in cluster_upper.items()},
        "test_results_used_for_selection": False,
        "fallback_rule": "g=1.0 if no statistical candidate is eligible",
    }


def summarize_test(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for cell_index, ((scenario, profile), cell) in enumerate(frame.groupby(
            ["scenario", "profile"], sort=False), 1):
        assigned = int(cell["assigned"].sum())
        violations = int(cell["qos_violations"].sum())
        coverage = cell["service_coverage"].to_numpy(float)
        pps = cell["pps"].to_numpy(float)
        coverage_lo, coverage_hi = bootstrap_ci(coverage, 291_000 + cell_index)
        pps_lo, pps_hi = bootstrap_ci(pps, 292_000 + cell_index)
        rows.append({
            "scenario": scenario,
            "profile": profile,
            "global_guardrail_weight": float(
                cell["global_guardrail_weight"].iloc[0]),
            "n_seeds": int(cell["seed"].nunique()),
            "assigned_total": assigned,
            "qos_violations_total": violations,
            "qos_violation_rate": violations / max(assigned, 1),
            "qos_violation_cp_upper95": clopper_pearson_upper(
                violations, assigned, 0.05),
            "mean_service_coverage": float(coverage.mean()),
            "service_coverage_ci_low": coverage_lo,
            "service_coverage_ci_high": coverage_hi,
            "mean_pps": float(pps.mean()),
            "pps_ci_low": pps_lo,
            "pps_ci_high": pps_hi,
            "all_runs_ok": bool(cell["status"].isin(["ok", "completed"]).all()),
            "within_global_envelope": bool(cell["within_global_envelope"].all()),
            "test_side_lookahead": bool(cell["test_side_lookahead"].any()),
        })
    return pd.DataFrame(rows).sort_values(
        ["global_guardrail_weight", "scenario"])


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()

    out = PROJECT / "results" / "e29_calibration_selection"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.pilot:
        pilot_seeds = (1,)
        ensure_episodes(args.cache, args.episodes, list(pilot_seeds))
        pilot = run_grid(args.episodes, "pilot_calibration", pilot_seeds,
                         args.workers)
        pilot.to_csv(out / "e29_pilot_seed_results.csv", index=False)
        write_json(out / "e29_pilot_metadata.json", {
            "purpose": "seed_001 data-source and execution smoke test only",
            "selection_performed": False,
            "seeds": list(pilot_seeds),
            "runs": len(pilot),
            "position_cache_sha256": sha256(args.cache),
            "event_tape_hashes": sorted(pilot["event_tape_hash"].unique()),
            "all_runs_ok": bool(pilot["status"].isin(["ok", "completed"]).all()),
            "all_within_global_envelope": bool(
                pilot["within_global_envelope"].all()),
            "any_test_side_lookahead": bool(pilot["test_side_lookahead"].any()),
            "runner_sha256": sha256(Path(__file__)),
        })
        return

    all_seeds = list(CALIBRATION_SEEDS + TEST_SEEDS)
    ensure_episodes(args.cache, args.episodes, all_seeds)

    # Phase 1: run calibration, select, and persist the frozen decision.
    calibration = run_grid(args.episodes, "calibration", CALIBRATION_SEEDS,
                           args.workers)
    calibration_path = out / "e29_calibration_seed_results.csv"
    calibration.to_csv(calibration_path, index=False)
    bounds = calibration_bounds(calibration)
    cluster_upper = cluster_bootstrap_upper(calibration, args.bootstrap_reps)
    bounds["seed_cluster_bootstrap_max_scenario_upper95"] = bounds[
        "global_guardrail_weight"].map(cluster_upper)
    bounds_path = out / "e29_calibration_cell_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    selection = select_guardrail(bounds, cluster_upper)
    selection["calibration_results_sha256"] = sha256(calibration_path)
    selection["calibration_bounds_sha256"] = sha256(bounds_path)
    selection["selection_written_before_test_launch"] = True
    selection_path = out / "e29_selection_frozen_before_test.json"
    write_json(selection_path, selection)
    print("E29 selection frozen before held-out launch:",
          json.dumps(selection, indent=2), flush=True)

    # Phase 2: only now launch the held-out even-seed jobs.
    test = run_grid(args.episodes, "held_out_test", TEST_SEEDS, args.workers)
    test_path = out / "e29_test_seed_results_all_candidates.csv"
    test.to_csv(test_path, index=False)
    summary = summarize_test(test)
    summary_path = out / "e29_test_summary_all_candidates.csv"
    summary.to_csv(summary_path, index=False)
    selected = float(selection["selected_global_guardrail_weight"])
    selected_summary = summary[
        summary["global_guardrail_weight"] == selected].copy()
    selected_summary.to_csv(out / "e29_test_summary_selected_policy.csv",
                            index=False)

    metadata = {
        "experiment": "E29 calibration-only guardrail selection",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium episodes",
        "candidate_weights_predeclared": list(GUARDRAIL_WEIGHTS),
        "scenarios_predeclared": list(SCENARIOS),
        "risk_budget_alpha": ALPHA,
        "familywise_delta": DELTA,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "selection_file": selection_path.name,
        "selection_file_sha256": sha256(selection_path),
        "test_results_used_for_selection": False,
        "test_side_lookahead": False,
        "conditional_guarantee_assumption": (
            "independent Bernoulli assigned-service outcomes within each "
            "registered scenario/candidate cell"),
        "dependence_sensitivity": (
            "paired seed-cluster bootstrap; reported but not used to tune"),
        "structural_guarantee_scope": (
            "g=1 only, conditional on realised responses remaining inside "
            "the declared global envelope"),
        "position_cache_sha256": sha256(args.cache),
        "runner_sha256": sha256(Path(__file__)),
        "simulator_sha256": sha256(ROOT / "src" / "simulator.py"),
        "calibration_results_sha256": sha256(calibration_path),
        "test_results_sha256": sha256(test_path),
        "test_summary_sha256": sha256(summary_path),
        "raw_or_derived_trace_redistributed": False,
        "runs": len(calibration) + len(test),
        "elapsed_seconds": time.time() - started,
    }
    write_json(out / "e29_metadata.json", metadata)
    print(selected_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
