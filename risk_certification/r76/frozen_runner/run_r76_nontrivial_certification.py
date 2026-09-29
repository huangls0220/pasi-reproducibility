"""R76 prespecified nontrivial-policy calibration--freeze--test study.

Restricted GeoLife caches and generated episodes must stay outside this folder.
Run actions in order: freeze, smoke, prepare-cal, calibrate, select,
prepare-test, test, summarize.  The test preparation gate requires a frozen
selection file, so test episodes cannot be inspected before selection.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CODE = WORK / "round73-repro-crosswindow" / "legacy" / "response_safety" / "current" / "code"
CONFIGS = WORK / "round73-repro-crosswindow" / "legacy" / "response_safety" / "current" / "configurations.json"
PLAN = ROOT / "preregistration.json"
PROTOCOL = ROOT / "protocol.md"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from scripts.run_e20_envelope_tradeoff import SCENARIOS, slot_ratios
from scripts.run_e22_adaptive_geolife import metrics
from src.datasets.mobility import positions_to_episode
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

ALPHA = 0.005
DELTA = 0.05
PLANNING_CANDIDATE = 0.75
WEIGHTS = (PLANNING_CANDIDATE,)
SMOKE_WEIGHTS = (PLANNING_CANDIDATE, 1.0)
CAL_SEEDS = tuple(range(770001, 771601))
TEST_SEEDS = tuple(range(780001, 781001))
PILOT_SEED = 769999
LOWER, UPPER = 0.70, 1.30
EPISODE_FILES = ("tasks.parquet", "providers.parquet", "provider_static.parquet", "meta.json")
SOURCE_FILES = (
    "src/simulator.py", "src/pair_eval.py", "src/matching.py",
    "src/datasets/mobility.py", "src/datasets/trace_loader.py",
    "scripts/run_e20_envelope_tradeoff.py",
    "scripts/run_e22_adaptive_geolife.py",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def core_hashes() -> dict[str, str]:
    hashes = {name: sha(CODE / name) for name in SOURCE_FILES}
    hashes["configurations.json"] = sha(CONFIGS)
    return hashes


def restricted_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT.resolve() or resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Restricted inputs and episodes must stay outside the public R76 folder")
    return resolved


def window_for_seed(seed: int) -> str:
    digest = hashlib.sha256(f"R76-window|{seed}".encode()).digest()
    return f"w{int.from_bytes(digest[:8], 'big') % 6 + 1:02d}"


def source_manifest(r73: Path) -> dict:
    r73 = restricted_path(r73)
    manifest = read(r73 / "input_manifest.json")
    if len(manifest["windows"]) != 6:
        raise ValueError("R73 source must contain all six frozen windows")
    for wid, entry in manifest["windows"].items():
        cache = r73 / wid / "positions.parquet"
        if sha(cache) != entry["cache_sha256"]:
            raise ValueError(f"Changed R73 position cache: {wid}")
    return manifest


def current_identity(r73: Path) -> dict:
    return {
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(PROTOCOL),
        "preregistration_sha256": sha(PLAN),
        "core_sha256": core_hashes(),
        "r73_input_manifest_sha256": sha(restricted_path(r73) / "input_manifest.json"),
    }


def require_frozen(results: Path, r73: Path) -> dict:
    frozen = read(results / "r76_frozen_before_outcomes.json")
    identity = current_identity(r73)
    for key, value in identity.items():
        if frozen[key] != value:
            raise ValueError(f"Frozen identity changed: {key}")
    source_manifest(r73)
    return frozen


def freeze(results: Path, r73: Path) -> None:
    source_manifest(r73)
    target = results / "r76_frozen_before_outcomes.json"
    if target.exists():
        raise FileExistsError("R76 is already frozen; do not overwrite")
    n_zero = math.ceil(math.log(DELTA / (len(SCENARIOS) * len(WEIGHTS))) /
                       math.log(1.0 - ALPHA))
    if len(CAL_SEEDS) < n_zero:
        raise AssertionError("Calibration sample is below the zero-loss threshold")
    payload = {
        **current_identity(r73),
        "frozen_unix_time": time.time(),
        "alpha": ALPHA,
        "delta": DELTA,
        "candidate_weights": WEIGHTS,
        "structural_fallback_weight": 1.0,
        "scenarios": SCENARIOS,
        "calibration_seed_first_last_count": [CAL_SEEDS[0], CAL_SEEDS[-1], len(CAL_SEEDS)],
        "test_seed_first_last_count": [TEST_SEEDS[0], TEST_SEEDS[-1], len(TEST_SEEDS)],
        "pilot_seed_excluded": PILOT_SEED,
        "zero_loss_minimum_n_per_cell": n_zero,
        "candidate_selected_from": "R75 calibration evidence only",
        "r75_test_outcomes_used_for_candidate_design": False,
        "test_outcomes_used_for_selection": False,
        "restricted_data_redistributed": False,
    }
    write(target, payload)


def episode_hashes(folder: Path) -> dict[str, str]:
    return {name: sha(folder / name) for name in EPISODE_FILES}


def prepare_episodes(r73: Path, restricted: Path, split: str,
                     seeds: tuple[int, ...]) -> None:
    r73 = restricted_path(r73)
    restricted = restricted_path(restricted)
    target = restricted / split
    if target.exists():
        raise FileExistsError(f"Choose a fresh restricted {split} directory")
    target.mkdir(parents=True)
    r73_manifest = source_manifest(r73)
    manifest = {"split": split, "seeds": [], "r73_input_manifest_sha256": sha(r73 / "input_manifest.json")}
    for index, seed in enumerate(seeds, 1):
        wid = window_for_seed(seed)
        cache = r73 / wid / "positions.parquet"
        folder = target / "episodes" / f"seed_{seed:06d}"
        meta = positions_to_episode(cache, folder, "medium", seed, service_radius_km=3.0)
        if int(meta["n_tasks"]) <= 0 or int(meta["n_providers"]) <= 0:
            raise ValueError(f"Empty episode {split}/{seed}")
        manifest["seeds"].append({
            "seed": seed, "window": wid, "cache_sha256": r73_manifest["windows"][wid]["cache_sha256"],
            "episode_meta": meta, "episode_file_sha256": episode_hashes(folder),
        })
        if index % 100 == 0 or index == len(seeds):
            print(f"prepared {split} {index}/{len(seeds)}", flush=True)
    write(target / "episode_manifest.json", manifest)


def verify_episode(restricted: Path, split: str, seed: int) -> tuple[Path, dict]:
    manifest = read(restricted / split / "episode_manifest.json")
    entries = {int(entry["seed"]): entry for entry in manifest["seeds"]}
    entry = entries[seed]
    folder = restricted / split / "episodes" / f"seed_{seed:06d}"
    if episode_hashes(folder) != entry["episode_file_sha256"]:
        raise ValueError(f"Changed episode {split}/{seed}")
    return folder, entry


def config_for(folder: Path, n_providers: int, weight: float) -> dict:
    template = read(CONFIGS)["2"]["fixed_envelope"]
    cfg = copy.deepcopy(template)
    cfg["dataset"]["processed_dir"] = str(folder)
    cfg["providers"]["N_mean"] = n_providers
    if (cfg["matching"]["objective"] != "coverage_first_payment_second" or
            float(cfg["contract"]["D_bar"]) != 50.0 or
            float(cfg["matching"]["budget_ratio"]) != 3.5):
        raise ValueError("Corrected formal protocol mismatch")
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
    return cfg


def run_cell(restricted_text: str, split: str, scenario: str,
             weight: float, seed: int) -> dict:
    restricted = Path(restricted_text)
    folder, entry = verify_episode(restricted, split, seed)
    meta = read(folder / "meta.json")
    cfg = config_for(folder, int(meta["n_providers"]), weight)
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    if data["meta"]["source"] != "geolife" or data["meta"]["redistributable"] is not False:
        raise ValueError("GeoLife source metadata mismatch")
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed, LOWER, UPPER)
    if not np.all((ratios >= LOWER - 1e-12) & (ratios <= UPPER + 1e-12)):
        raise ValueError("Response path outside frozen envelope")
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    execution_status = result["diagnostics"].get("execution_status")
    if execution_status != "completed":
        raise RuntimeError(f"Incomplete simulator execution: {execution_status}")
    row = metrics(result)
    assigned, violations = int(row["assigned"]), int(row["qos_violations"])
    if assigned < 0 or not 0 <= violations <= assigned:
        raise ValueError("Invalid episode counts")
    row.update({
        "run_id": f"r76-{split}-{scenario}-g{weight:.2f}-{seed}",
        "split": split, "scenario": scenario, "global_guardrail_weight": weight,
        "seed": seed, "window": entry["window"], "episode_loss": violations / max(assigned, 1),
        "zero_service_episode": assigned == 0, "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()), "within_global_envelope": True,
        "test_side_lookahead": False, "runtime_s": time.perf_counter() - started,
        "execution_status": execution_status,
        "safety_outcome_status": result["diagnostics"].get("service_outcome_status"),
        "episode_file_sha256": json.dumps(entry["episode_file_sha256"], sort_keys=True),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    })
    return row


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed line {number} in {path}") from exc
    return rows


def run_grid(restricted: Path, results: Path, split: str,
             seeds: tuple[int, ...], weights: tuple[float, ...], workers: int) -> pd.DataFrame:
    restricted = restricted_path(restricted)
    jsonl = results / f"r76_{split}_running.jsonl"
    existing = read_jsonl(jsonl)
    by_id = {row["run_id"]: row for row in existing}
    jobs = [(str(restricted), split, scenario, weight, seed)
            for scenario in SCENARIOS for weight in weights for seed in seeds]
    pending = [job for job in jobs
               if f"r76-{split}-{job[2]}-g{job[3]:.2f}-{job[4]}" not in by_id]
    print(f"{split}: {len(jobs) - len(pending)} recovered, {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, *job): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            by_id[row["run_id"]] = row
            with jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(pending):
                print(f"{split} completed {len(jobs) - len(pending) + index}/{len(jobs)}", flush=True)
    expected = {f"r76-{split}-{scenario}-g{weight:.2f}-{seed}"
                for scenario in SCENARIOS for weight in weights for seed in seeds}
    if set(by_id) != expected:
        raise RuntimeError("Incomplete or extra grid cells")
    frame = pd.DataFrame(by_id.values()).sort_values(["scenario", "global_guardrail_weight", "seed"])
    frame.to_csv(results / f"r76_{split}_seed_results.csv", index=False)
    return frame


def binary_kl(x: float, q: float) -> float:
    if x == q:
        return 0.0
    if q <= 0.0 or q >= 1.0:
        return math.inf
    left = 0.0 if x == 0.0 else x * math.log(x / q)
    right = 0.0 if x == 1.0 else (1.0 - x) * math.log((1.0 - x) / (1.0 - q))
    return left + right


def bounded_kl_upper(values, tail: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional loss vector required")
    if np.any((values < 0.0) | (values > 1.0)) or not 0.0 < tail < 1.0:
        raise ValueError("Loss/tail outside admissible range")
    mean = float(values.mean())
    if mean == 1.0:
        return 1.0
    if mean == 0.0:
        return -math.expm1(math.log(tail) / len(values))
    target = math.log(1.0 / tail) / len(values)
    lo, hi = mean, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if binary_kl(mean, mid) <= target:
            lo = mid
        else:
            hi = mid
    return hi


def summarize(frame: pd.DataFrame, tail: float) -> pd.DataFrame:
    cells = frame.groupby(["scenario", "global_guardrail_weight"]).ngroups
    coverage_tail = DELTA / (2 * cells)
    rows = []
    for (scenario, weight), cell in frame.groupby(["scenario", "global_guardrail_weight"]):
        n = len(cell)
        coverage = cell["service_coverage"].to_numpy(float)
        qualified = (cell["service_coverage"] * (1.0 - cell["qos_violation_rate"])).to_numpy(float)
        radius = math.sqrt(math.log(1.0 / coverage_tail) / (2.0 * n))
        assigned = int(cell["assigned"].sum())
        violations = int(cell["qos_violations"].sum())
        rows.append({
            "scenario": scenario, "global_guardrail_weight": float(weight), "n_episodes": n,
            "assigned_total": assigned, "qos_violations_total": violations,
            "pooled_violation_rate_descriptive": violations / max(assigned, 1),
            "mean_episode_loss": float(cell["episode_loss"].mean()),
            "episode_risk_upper": bounded_kl_upper(cell["episode_loss"], tail),
            "risk_cell_tail": tail, "mean_assignment_coverage": float(coverage.mean()),
            "assignment_coverage_lower": max(0.0, float(coverage.mean()) - radius),
            "assignment_coverage_upper": min(1.0, float(coverage.mean()) + radius),
            "mean_qualified_coverage": float(qualified.mean()),
            "mean_execution_pps": float(cell["pps"].mean()),
            "zero_service_episodes": int(cell["zero_service_episode"].sum()),
            "all_executions_completed": bool((cell["execution_status"] == "completed").all()),
            "all_inside_global_envelope": bool(cell["within_global_envelope"].all()),
            "test_side_lookahead": bool(cell["test_side_lookahead"].any()),
        })
    return pd.DataFrame(rows).sort_values(["global_guardrail_weight", "scenario"])


def paired_coverage_gain(frame: pd.DataFrame, candidate: float) -> pd.DataFrame:
    columns = [
        "scenario", "candidate_g", "n_pairs",
        "assignment_coverage_gain_pp", "assignment_gain_ci95_low_pp",
        "assignment_gain_ci95_high_pp", "qualified_coverage_gain_pp",
        "qualified_gain_ci95_low_pp", "qualified_gain_ci95_high_pp",
        "bootstrap_replicates", "bootstrap_seed",
    ]
    if candidate == 1.0:
        return pd.DataFrame(columns=columns)
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        cell = frame[frame["scenario"] == scenario].copy()
        policy = cell[cell["global_guardrail_weight"] == candidate].copy()
        baseline = cell[cell["global_guardrail_weight"] == 1.0].copy()
        joined = policy.merge(
            baseline, on=["seed", "window"], suffixes=("_candidate", "_g1"),
            validate="one_to_one",
        )
        if len(joined) != len(TEST_SEEDS):
            raise RuntimeError(f"Incomplete paired test comparison for {scenario}")
        assignment = (
            joined["service_coverage_candidate"].to_numpy(float)
            - joined["service_coverage_g1"].to_numpy(float)
        )
        qualified_candidate = (
            joined["service_coverage_candidate"].to_numpy(float)
            * (1.0 - joined["qos_violation_rate_candidate"].to_numpy(float))
        )
        qualified_g1 = (
            joined["service_coverage_g1"].to_numpy(float)
            * (1.0 - joined["qos_violation_rate_g1"].to_numpy(float))
        )
        qualified = qualified_candidate - qualified_g1
        rng_seed = 20260928 + scenario_index
        rng = np.random.default_rng(rng_seed)
        assignment_boot = []
        qualified_boot = []
        for _ in range(20):
            indices = rng.integers(0, len(joined), size=(500, len(joined)))
            assignment_boot.append(assignment[indices].mean(axis=1))
            qualified_boot.append(qualified[indices].mean(axis=1))
        assignment_ci = np.quantile(np.concatenate(assignment_boot), [0.025, 0.975])
        qualified_ci = np.quantile(np.concatenate(qualified_boot), [0.025, 0.975])
        rows.append({
            "scenario": scenario, "candidate_g": candidate, "n_pairs": len(joined),
            "assignment_coverage_gain_pp": float(100.0 * assignment.mean()),
            "assignment_gain_ci95_low_pp": float(100.0 * assignment_ci[0]),
            "assignment_gain_ci95_high_pp": float(100.0 * assignment_ci[1]),
            "qualified_coverage_gain_pp": float(100.0 * qualified.mean()),
            "qualified_gain_ci95_low_pp": float(100.0 * qualified_ci[0]),
            "qualified_gain_ci95_high_pp": float(100.0 * qualified_ci[1]),
            "bootstrap_replicates": 10000, "bootstrap_seed": rng_seed,
        })
    return pd.DataFrame(rows, columns=columns)


def select(results: Path) -> None:
    target = results / "r76_selection_frozen_before_test.json"
    if target.exists():
        raise FileExistsError("Selection already frozen; do not overwrite")
    frame = pd.read_csv(results / "r76_calibration_seed_results.csv")
    if len(frame) != len(CAL_SEEDS) * len(SCENARIOS) * len(WEIGHTS):
        raise ValueError("Incomplete calibration matrix")
    tail = DELTA / (len(SCENARIOS) * len(WEIGHTS))
    summary = summarize(frame, tail)
    summary.to_csv(results / "r76_calibration_summary.csv", index=False)
    worst = summary.groupby("global_guardrail_weight")["episode_risk_upper"].max()
    eligible = [float(weight) for weight, upper in worst.items() if upper <= ALPHA]
    selected = PLANNING_CANDIDATE if PLANNING_CANDIDATE in eligible else 1.0
    payload = {
        "selected_g": selected, "statistical_pass": bool(eligible),
        "selection_basis": "simultaneous_episode_KL" if eligible else "structural_global_envelope_fallback",
        "alpha": ALPHA, "delta": DELTA,
        "worst_episode_risk_upper_by_g": {str(k): float(v) for k, v in worst.items()},
        "test_outcomes_used_for_selection": False,
        "prespecified_candidate": PLANNING_CANDIDATE,
        "candidate_selected_from": "R75 calibration evidence only",
        "r75_test_outcomes_used_for_candidate_design": False,
        "calibration_results_sha256": sha(results / "r76_calibration_seed_results.csv"),
        "calibration_summary_sha256": sha(results / "r76_calibration_summary.csv"),
        "written_unix_time": time.time(),
    }
    write(target, payload)


def smoke(r73: Path, restricted: Path, results: Path) -> None:
    require_frozen(results, r73)
    pilot_root = restricted_path(restricted) / "pilot"
    if not pilot_root.exists():
        prepare_episodes(r73, restricted, "pilot", (PILOT_SEED,))
    rows = [run_cell(str(restricted), "pilot", scenario, weight, PILOT_SEED)
            for scenario in SCENARIOS for weight in SMOKE_WEIGHTS]
    passed = all(row["execution_status"] == "completed" and row["within_global_envelope"] for row in rows)
    write(results / "r76_smoke.json", {
        "passed": passed, "pilot_seed_excluded": PILOT_SEED, "cells": len(rows),
        "source": "GeoLife", "restricted_data_redistributed": False,
        "rows": rows,
    })
    if not passed:
        raise RuntimeError("R76 smoke failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "smoke", "prepare-cal", "calibrate", "select", "prepare-test", "test", "summarize"))
    parser.add_argument("--r73", required=True, type=Path, help="Restricted R73 six-window root")
    parser.add_argument("--restricted", required=True, type=Path, help="Fresh R76 restricted episode root")
    parser.add_argument("--results", required=True, type=Path, help="Non-sensitive R76 result root")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    if args.action == "freeze":
        freeze(args.results, args.r73)
        return
    require_frozen(args.results, args.r73)
    if args.action == "smoke":
        smoke(args.r73, args.restricted, args.results)
    elif args.action == "prepare-cal":
        if not read(args.results / "r76_smoke.json")["passed"]:
            raise RuntimeError("Successful excluded-seed smoke required")
        prepare_episodes(args.r73, args.restricted, "calibration", CAL_SEEDS)
    elif args.action == "calibrate":
        frame = run_grid(args.restricted, args.results, "calibration", CAL_SEEDS, WEIGHTS, args.workers)
        summary = summarize(frame, DELTA / (len(SCENARIOS) * len(WEIGHTS)))
        summary.to_csv(args.results / "r76_calibration_summary.csv", index=False)
    elif args.action == "select":
        select(args.results)
    elif args.action == "prepare-test":
        selection_path = args.results / "r76_selection_frozen_before_test.json"
        if not selection_path.exists():
            raise RuntimeError("Freeze selection before constructing held-out episodes")
        plan = {"selection_sha256": sha(selection_path), "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1], len(TEST_SEEDS)],
                "constructed_after_selection": True, "test_outcomes_used_for_selection": False}
        write(args.results / "r76_test_construction_gate.json", plan)
        prepare_episodes(args.r73, args.restricted, "test", TEST_SEEDS)
    elif args.action == "test":
        selection = read(args.results / "r76_selection_frozen_before_test.json")
        weights = tuple(sorted({float(selection["selected_g"]), 1.0}))
        frame = run_grid(args.restricted, args.results, "test", TEST_SEEDS, weights, args.workers)
        summary = summarize(frame, DELTA / (len(SCENARIOS) * len(weights)))
        summary.to_csv(args.results / "r76_test_summary.csv", index=False)
        gain = paired_coverage_gain(frame, float(selection["selected_g"]))
        gain.to_csv(args.results / "r76_test_paired_coverage_gain.csv", index=False)
    else:
        selection = read(args.results / "r76_selection_frozen_before_test.json")
        cal = pd.read_csv(args.results / "r76_calibration_seed_results.csv")
        test = pd.read_csv(args.results / "r76_test_seed_results.csv")
        test_weights = sorted(test["global_guardrail_weight"].unique().tolist())
        expected_cal = len(CAL_SEEDS) * len(SCENARIOS) * len(WEIGHTS)
        expected_test = len(TEST_SEEDS) * len(SCENARIOS) * len(test_weights)
        if len(cal) != expected_cal or len(test) != expected_test:
            raise RuntimeError("Do not summarize an incomplete R76 grid")
        completion = {
            "calibration_runs": len(cal), "test_runs": len(test),
            "selected_g": selection["selected_g"], "statistical_pass": selection["statistical_pass"],
            "selection_sha256": sha(args.results / "r76_selection_frozen_before_test.json"),
            "test_outcomes_used_for_selection": False,
            "all_executions_completed": bool((cal["execution_status"] == "completed").all() and (test["execution_status"] == "completed").all()),
            "all_inside_global_envelope": bool(cal["within_global_envelope"].all() and test["within_global_envelope"].all()),
            "claim_scope": "fixed six-week empirical GeoLife mixture plus modeled economic and response generator",
            "real_participant_economic_evidence": False,
            "restricted_data_redistributed": False,
            "references_changed": False,
            "identity": current_identity(args.r73),
        }
        write(args.results / "r76_completion.json", completion)
        print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
