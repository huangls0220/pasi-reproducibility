"""E26: broaden public-participant dynamics across sources and activity strata.

ExtraSensory contributes observed sensor/network state for all 60 available
participants.  NetEaseCrowd contributes completed-answer quality dynamics for
90 workers sampled equally from low-, medium-, and high-activity strata.  The
frozen GeoLife episodes still supply mobility and candidate graphs; requests,
costs, capacities, contract responses, and payments remain model-generated.
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
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e20_envelope_tradeoff import build_config
from scripts.run_e22_adaptive_geolife import metrics
from scripts.run_e23_guardrail_frontier import wilson_upper
from scripts.run_e25_public_dynamics import _adaptive_config
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.public_dynamics import (
    load_extrasensory_drivers,
    load_netease_stratified_response_drivers,
    resample_trace,
    sha256_file,
)
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

LOWER = 0.70
UPPER = 1.30
EXTRA_METHODS = ("PASI", "MOI")
NETEASE_PROFILES = ("frozen_point", "guardrail_075", "fixed_envelope")


def _folder_and_data(episodes: Path, episode_seed: int, profile: str) -> tuple[Path, dict, dict]:
    folder = episodes / f"seed_{episode_seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    base = "frozen_point" if profile == "frozen_point" else "fixed_envelope"
    cfg = build_config(folder, int(meta["n_providers"]), base, LOWER, UPPER)
    data = load_trace_episode("geolife", cfg, seed=episode_seed, processed_dir=folder)
    return folder, cfg, data


def run_extra_source(
    episodes: Path,
    driver: dict,
    method: str,
    source_index: int,
    episode_seed: int,
) -> dict:
    folder, cfg, data = _folder_and_data(episodes, episode_seed, "frozen_point")
    providers = data["providers"].copy()
    slots = np.sort(providers["slot"].unique())
    availability = resample_trace(driver["availability_ratio"], len(slots))
    network = resample_trace(driver["network_ratio"], len(slots))
    slot_pos = {int(slot): index for index, slot in enumerate(slots)}
    index = providers["slot"].map(slot_pos).to_numpy(int)
    providers["availability_duration"] *= availability[index]
    providers["communication_rate"] *= network[index]
    data["providers"] = providers
    started = time.perf_counter()
    result = Simulator(cfg, data, method=method, seed=episode_seed).run()
    diagnostics = result["diagnostics"]
    return {
        "branch": "extrasensory_observed_state",
        "activity_stratum": "all",
        "profile": method,
        "source_index": source_index,
        "episode_seed": episode_seed,
        "source_token": driver["source_token"],
        "availability_ratio_min": float(availability.min()),
        "availability_ratio_max": float(availability.max()),
        "network_ratio_min": float(network.min()),
        "network_ratio_max": float(network.max()),
        "current_state_observed_before_decision": True,
        "test_side_response_used_for_decision": False,
        "event_tape_hash": tape_hash(folder),
        "runtime_s": time.perf_counter() - started,
        "execution_completed": True,
        "diagnostic_status": diagnostics.get("status", "unknown"),
        "diagnostic_ir_violations": int(diagnostics.get("ir_violations", 0)),
        "diagnostic_target_violations": int(diagnostics.get("target_violations", 0)),
        "diagnostic_budget_violations": int(diagnostics.get("budget_violations", 0)),
        "diagnostic_nan_rows": int(diagnostics.get("nan_rows", 0)),
        "diagnostic_inf_rows": int(diagnostics.get("inf_rows", 0)),
        "diagnostic_ub_lt_lb": int(diagnostics.get("ub_lt_lb", 0)),
        **metrics(result),
    }


def run_netease_source(
    episodes: Path,
    driver: dict,
    profile: str,
    source_index: int,
    episode_seed: int,
) -> dict:
    folder, cfg, data = _folder_and_data(episodes, episode_seed, profile)
    if profile == "guardrail_075":
        _adaptive_config(cfg, 0.75)
    tasks = data["tasks"].copy()
    slots = np.sort(tasks["slot"].unique())
    ratios_by_slot = resample_trace(driver["response_ratio"], len(slots))
    slot_pos = {int(slot): index for index, slot in enumerate(slots)}
    ratios = ratios_by_slot[tasks["slot"].map(slot_pos).to_numpy(int)]
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=episode_seed).run()
    diagnostics = result["diagnostics"]
    return {
        "branch": "netease_completed_quality",
        "activity_stratum": driver["activity_stratum"],
        "profile": profile,
        "source_index": source_index,
        "episode_seed": episode_seed,
        "source_token": driver["source_token"],
        "response_ratio_min": float(ratios.min()),
        "response_ratio_max": float(ratios.max()),
        "within_declared_envelope": bool(
            np.all((ratios >= LOWER - 1e-12) & (ratios <= UPPER + 1e-12))),
        "current_response_observed_before_decision": False,
        "observed_completion_only": True,
        "event_tape_hash": tape_hash(folder),
        "runtime_s": time.perf_counter() - started,
        "execution_completed": True,
        "diagnostic_status": diagnostics.get("status", "unknown"),
        "diagnostic_ir_violations": int(diagnostics.get("ir_violations", 0)),
        "diagnostic_target_violations": int(diagnostics.get("target_violations", 0)),
        "diagnostic_budget_violations": int(diagnostics.get("budget_violations", 0)),
        "diagnostic_nan_rows": int(diagnostics.get("nan_rows", 0)),
        "diagnostic_inf_rows": int(diagnostics.get("inf_rows", 0)),
        "diagnostic_ub_lt_lb": int(diagnostics.get("ub_lt_lb", 0)),
        **metrics(result),
    }


def _summarize_cell(group: pd.DataFrame, branch: str, stratum: str) -> list[dict]:
    reference_name = "MOI" if branch == "extrasensory_observed_state" else "fixed_envelope"
    reference = group[group["profile"] == reference_name].set_index("source_index")
    rows: list[dict] = []
    for profile_index, (profile, cell) in enumerate(group.groupby("profile", sort=False), 1):
        cell = cell.set_index("source_index").sort_index()
        ref = reference.loc[cell.index]
        assigned = int(cell["assigned"].sum())
        violations = int(cell["qos_violations"].sum())
        row = {
            "branch": branch,
            "activity_stratum": stratum,
            "profile": profile,
            "reference_profile": reference_name,
            "n_pairs": int(len(cell)),
            "unique_episode_count": int(cell["episode_seed"].nunique()),
            "all_executions_completed": bool(cell["execution_completed"].all()),
            "diagnostic_pass_runs": int(cell["diagnostic_status"].isin(["ok", "completed"]).sum()),
            "diagnostic_failed_runs": int((cell["diagnostic_status"] == "failed").sum()),
            "assigned_total": assigned,
            "qos_violations_total": violations,
            "pooled_qos_violation_rate": violations / max(assigned, 1),
            "qos_violation_wilson_upper_95": wilson_upper(violations, assigned),
        }
        for metric_index, metric in enumerate(("service_coverage", "pps", "qos_violation_rate"), 1):
            values = cell[metric].to_numpy(float)
            delta = values - ref[metric].to_numpy(float)
            base = 126_000 + 100 * profile_index + metric_index + 1_000 * len(rows)
            lo, hi = bootstrap_ci(values, base)
            dlo, dhi = bootstrap_ci(delta, base + 10_000)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"delta_reference_{metric}"] = float(delta.mean())
            row[f"delta_reference_{metric}_ci_low"] = dlo
            row[f"delta_reference_{metric}_ci_high"] = dhi
        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for branch, group in frame.groupby("branch", sort=False):
        rows.extend(_summarize_cell(group, branch, "all"))
        if branch == "netease_completed_quality":
            for stratum, cell in group.groupby("activity_stratum", sort=False):
                rows.extend(_summarize_cell(cell, branch, str(stratum)))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--extrasensory-dir", type=Path, required=True)
    parser.add_argument("--netease-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "e26_public_breadth")
    args = parser.parse_args()

    extra, extra_manifest = load_extrasensory_drivers(args.extrasensory_dir, seeds=60)
    netease, netease_manifest = load_netease_stratified_response_drivers(
        args.netease_csv, per_stratum=30, min_completions=80, block_size=8)
    if args.pilot:
        extra_indices = [1]
        netease_indices = [next(index for index, value in netease.items()
                                if value["activity_stratum"] == label)
                            for label in ("low", "medium", "high")]
    else:
        extra_indices = list(range(1, 61))
        netease_indices = list(range(1, 91))

    jobs: list[tuple] = []
    for source_index in extra_indices:
        episode_seed = (source_index - 1) % 30 + 1
        jobs.extend(("extra", extra[source_index], method, source_index, episode_seed)
                    for method in EXTRA_METHODS)
    for source_index in netease_indices:
        episode_seed = (source_index - 1) % 30 + 1
        jobs.extend(("netease", netease[source_index], profile, source_index, episode_seed)
                    for profile in NETEASE_PROFILES)

    rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for kind, driver, profile, source_index, episode_seed in jobs:
            fn = run_extra_source if kind == "extra" else run_netease_source
            futures[pool.submit(
                fn, args.episodes, driver, profile, source_index, episode_seed
            )] = (kind, profile, source_index, episode_seed)
        for index, future in enumerate(as_completed(futures), 1):
            kind, profile, source_index, episode_seed = futures[future]
            rows.append(future.result())
            print(
                f"[{index:03d}/{len(jobs)}] E26 {kind} {profile} "
                f"source={source_index} episode={episode_seed}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["branch", "profile", "source_index"])
    summary = summarize(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(args.out / f"e26_{suffix}_source_results.csv", index=False)
    summary.to_csv(args.out / f"e26_{suffix}_summary.csv", index=False)
    pd.DataFrame(extra_manifest + netease_manifest).to_csv(
        args.out / "e26_pseudonymous_source_manifest.csv", index=False)
    metadata = {
        "experiment": "E26 breadth extension using public participant dynamics",
        "runs": int(len(frame)),
        "elapsed_seconds": time.time() - started,
        "inference_unit": "public participant or worker; paired profiles within source",
        "extrasensory_sources": len(extra_indices),
        "netease_sources": len(netease_indices),
        "netease_activity_strata": "equal-rank tertiles among workers with at least 80 completed records",
        "netease_selection": "30 per stratum by SHA-256 of worker ID; no answer, truth, capability, or accuracy used",
        "episode_mapping": "source_index cycles over frozen GeoLife medium seed_001--seed_030",
        "episode_reuse": {"extrasensory": 2 if not args.pilot else 1,
                          "netease": 3 if not args.pilot else "pilot_subset"},
        "extrasensory_role": "current observed sensor/network availability multipliers only",
        "netease_role": "completed-worker correctness dynamics transformed with calibration-prefix quantiles only",
        "geolife_role": "frozen mobility and candidate graph",
        "model_generated_fields": ["requests", "costs", "capacities", "contract response mapping", "payments"],
        "not_claimed": ["population-representative sample", "real bids", "real private costs",
                        "causal payment response", "field deployment"],
        "netease_limit": "part 1, completed-annotation records only; no noncompletion or acceptance inference",
        "test_side_lookahead": False,
        "raw_participant_data_redistributed": False,
        "extrasensory_mirror_commit": "167e0a899cb73c6e20398e8af886517cc9357c3e",
        "netease_repository_commit": "2c3d461d8cfac70623da86110282876f6555121c",
        "netease_csv_sha256": sha256_file(args.netease_csv),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_dynamics_module_hash": hashlib.sha256(
            (ROOT / "src" / "datasets" / "public_dynamics.py").read_bytes()).hexdigest(),
        "simulator_hash": hashlib.sha256((ROOT / "src" / "simulator.py").read_bytes()).hexdigest(),
    }
    (args.out / f"e26_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
