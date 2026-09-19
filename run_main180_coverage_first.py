#!/usr/bin/env python3
"""R72 canonical Main-180 under the repaired coverage-first core.

This runner freezes the current core, reuses the exact archived JSON
configurations and stochastic episodes, explicitly selects lexicographic
coverage-first matching, and records decision-reservation and execution-
settlement quantities separately for the original 3x2x30 jobs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import shutil
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


ROUND = Path(__file__).resolve().parent
ROOT = ROUND.parents[1]
SOURCE = ROUND / "core_source"
CODE = ROUND / "code"
ARCHIVE = Path(r"C:\Users\huang\Desktop\8.14\第十二轮\04_实验附件\Main-180_release_safe.zip")
ARCHIVE_BASE = "sealed_evidence/pasi_final_evidence/"
SCENARIOS = ("stationary", "burst", "dynamic")
METHODS = ("MOI", "PASI")
SEEDS = tuple(range(1, 31))
QOS_TOL = 1e-9
TARGET_TOL = 1e-7


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_hashes() -> dict[str, str]:
    return {
        path.relative_to(CODE).as_posix(): sha256_file(path)
        for path in sorted(CODE.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def event_tape_hash(data: dict) -> str:
    payload = []
    for key in sorted(data):
        value = data[key]
        if not hasattr(value, "astype"):
            continue
        payload.append(key + ":" + value.astype(str).to_csv(index=False))
    return sha256_bytes("\n".join(payload).encode("utf-8"))


def frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return sha256_bytes(values.tobytes())


def key(job: dict) -> str:
    return f"{job['scenario']}__seed_{job['seed']:03d}__{job['method']}"


def prepare() -> None:
    plan_path = ROUND / "preregistration.json"
    if plan_path.exists():
        raise FileExistsError("Do not overwrite the frozen R72 preregistration")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)

    for source in SOURCE.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            target = CODE / source.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    config_records = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        historical = archive.read(ARCHIVE_BASE + "processed_results/run_metrics.csv")
        (ROUND / "historical_run_metrics.csv").write_bytes(historical)
        for scenario in SCENARIOS:
            member = ARCHIVE_BASE + f"frozen_configs/legacy_{scenario}_frozen.json"
            raw = archive.read(member)
            cfg_path = ROUND / "frozen_configs" / f"legacy_{scenario}_frozen.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_bytes(raw)
            archived_cfg = json.loads(raw.decode("utf-8-sig"))
            if archived_cfg.get("matching", {}).get("objective") is not None:
                raise AssertionError("Historical Main config unexpectedly declares an objective")
            cfg = copy.deepcopy(archived_cfg)
            cfg.setdefault("matching", {})["objective"] = "coverage_first_payment_second"
            config_records[scenario] = {
                "member": member,
                "raw_sha256": sha256_bytes(raw),
                "path": str(cfg_path),
                "archived_parsed_sha256": digest(archived_cfg),
                "effective_parsed_sha256": digest(cfg),
                "config": cfg,
            }

    hist = pd.read_csv(io.BytesIO(historical))
    expected_cells = {(s, m, seed) for s in SCENARIOS for m in METHODS for seed in SEEDS}
    actual_cells = set(zip(hist.scenario, hist.mechanism, hist.seed.astype(int)))
    if actual_cells != expected_cells or len(hist) != 180:
        raise AssertionError("Historical Main-180 cell matrix is not exact")

    jobs = [
        {"scenario": scenario, "method": method, "seed": seed}
        for scenario in SCENARIOS
        for seed in SEEDS
        for method in METHODS
    ]
    plan = {
        "created_utc": utc(),
        "round": 72,
        "scope": "canonical original Main-180 matrix under repaired coverage-first core",
        "interpretation": "new canonical-protocol evidence, not historical-program reproduction",
        "historical_source_available": False,
        "source_archive": str(ARCHIVE),
        "source_archive_sha256": sha256_file(ARCHIVE),
        "historical_metrics_sha256": sha256_bytes(historical),
        "jobs": jobs,
        "workers": 4,
        "config_records": config_records,
        "current_core_hashes": code_hashes(),
        "runner_sha256": sha256_file(Path(__file__)),
        "matching_policy": "lexicographically maximize qualified-service assignment cardinality, then minimize design-side decision cost",
        "matching_objective_injected": True,
        "matching_objective": "coverage_first_payment_second",
        "optimization_cost_basis": "decision_contract_cost (design-side reservation)",
        "settlement_reporting_basis": "expected_contract_cost at executed response, reported separately",
        "qos_tolerance": QOS_TOL,
        "target_tolerance": TARGET_TOL,
        "qos_event": "execution_quality < q_min-tol OR total_delay > deadline+tol",
        "smoke_cells": [
            {"scenario": "stationary", "method": "MOI", "seed": 1},
            {"scenario": "stationary", "method": "PASI", "seed": 1},
        ],
        "failure_policy": "preserve every completed/error cell and every negative outcome; do not tune or filter",
        "new_scenarios_or_seeds": False,
        "references_changed": False,
        "manuscript_changed": False,
        "restricted_or_nonredistributable_input": False,
    }
    dump(plan_path, plan)
    print(json.dumps({"frozen": True, "jobs": len(jobs), "runner_sha256": plan["runner_sha256"]}))


def verify(plan: dict) -> None:
    if sha256_file(Path(__file__)) != plan["runner_sha256"]:
        raise AssertionError("Runner changed after preregistration")
    if code_hashes() != plan["current_core_hashes"]:
        raise AssertionError("Frozen current core changed after preregistration")
    if sha256_file(ARCHIVE) != plan["source_archive_sha256"]:
        raise AssertionError("Historical source archive changed")
    if sha256_file(ROUND / "historical_run_metrics.csv") != plan["historical_metrics_sha256"]:
        raise AssertionError("Historical metric snapshot changed")
    for record in plan["config_records"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["raw_sha256"]:
            raise AssertionError(f"Frozen config changed: {path}")
        if digest(record["config"]) != record["effective_parsed_sha256"]:
            raise AssertionError("Effective coverage-first config changed")


def run_one(job: dict) -> dict:
    plan = load(ROUND / "preregistration.json")
    run_key = key(job)
    cell_path = ROUND / "cells" / f"{run_key}.json"
    if cell_path.exists():
        return load(cell_path)

    sys.path.insert(0, str(CODE))
    from src.datasets.synthetic import generate_synthetic_episode
    import src.simulator as simmodule

    record = plan["config_records"][job["scenario"]]
    cfg = copy.deepcopy(record["config"])
    cfg.setdefault("simulation", {})["log_level"] = "selected"
    cfg["simulation"].setdefault("state_reset", {"enabled": False})
    cfg["dataset"] = {"pattern": job["scenario"]}
    if cfg.get("matching", {}).get("objective") != "coverage_first_payment_second":
        raise AssertionError("Canonical Main requires coverage-first matching")
    data = generate_synthetic_episode(cfg, seed=job["seed"], pattern=job["scenario"])
    tape = event_tape_hash(data)
    fingerprints = {
        name: frame_hash(value)
        for name, value in data.items()
        if isinstance(value, pd.DataFrame)
    }

    counters = {
        "candidate_rows": 0,
        "legal_rows": 0,
        "legal_effort_below_a_min": 0,
        "legal_effort_at_one": 0,
        "legal_design_settlement_cost_diff": 0,
        "max_legal_design_settlement_cost_diff": 0.0,
        "zeta_above_one": 0,
        "selected_quality_violations": 0,
        "selected_deadline_violations": 0,
        "selected_qos_violations": 0,
        "selected_ir_violations": 0,
        "selected_target_violations": 0,
    }
    state: dict[str, dict] = {}
    candidate_hasher = hashlib.sha256()
    selection_hasher = hashlib.sha256()
    evaluator = simmodule.evaluate_pairs
    coverage_matcher = simmodule.coverage_first_matching
    matcher_audit = {"calls": 0, "objective_types": set(), "solver_counts": {}}

    def audited_coverage_matching(*args, **kwargs):
        result = coverage_matcher(*args, **kwargs)
        matcher_audit["calls"] += 1
        matcher_audit["objective_types"].add(result.get("objective_type"))
        solver = result.get("solver", "small_subset_or_empty")
        matcher_audit["solver_counts"][solver] = matcher_audit["solver_counts"].get(solver, 0) + 1
        if result.get("objective_type") != "coverage_first_payment_second":
            raise AssertionError("Unexpected matching objective type")
        return result

    def evaluate(mechanism, context):
        output = evaluator(mechanism, context)
        state["context"] = context
        feasible = np.asarray(output["feasible"], dtype=bool)
        effort = np.asarray(output["a_star"], dtype=float)
        minimum = np.asarray(output["a_min"], dtype=float)
        design_cost = np.asarray(output["decision_contract_cost"], dtype=float)
        settle_cost = np.asarray(output["expected_contract_cost"], dtype=float)
        diff = np.abs(design_cost - settle_cost)
        counters["candidate_rows"] += int(len(feasible))
        counters["legal_rows"] += int(feasible.sum())
        counters["legal_effort_below_a_min"] += int((feasible & (effort < minimum - 1e-9)).sum())
        counters["legal_effort_at_one"] += int((feasible & (np.abs(effort - 1.0) <= 1e-9)).sum())
        counters["legal_design_settlement_cost_diff"] += int((feasible & (diff > 1e-9)).sum())
        if feasible.any():
            counters["max_legal_design_settlement_cost_diff"] = max(
                counters["max_legal_design_settlement_cost_diff"],
                float(diff[feasible].max()),
            )
        zeta = np.broadcast_to(np.asarray(context["zeta_t"], dtype=float), feasible.shape)
        counters["zeta_above_one"] += int((zeta > 1.0 + 1e-12).sum())
        return output

    simulator = simmodule.Simulator(
        cfg,
        data,
        method="SAMI" if job["method"] == "PASI" else job["method"],
        seed=job["seed"],
    )
    original_log = simulator._log_pairs

    def log_pairs(slot, slot_providers, slot_tasks, provider_rows, task_rows, evaluation, selected):
        context = state["context"]
        selected = np.asarray(selected, dtype=bool)
        feasible = np.asarray(evaluation["feasible"], dtype=bool)
        candidate_hasher.update(np.asarray([slot], dtype=np.int64).tobytes())
        candidate_hasher.update(np.asarray(provider_rows, dtype=np.int64).tobytes())
        candidate_hasher.update(np.asarray(task_rows, dtype=np.int64).tobytes())
        candidate_hasher.update(feasible.astype(np.uint8).tobytes())
        selection_hasher.update(np.asarray([slot], dtype=np.int64).tobytes())
        selection_hasher.update(selected.astype(np.uint8).tobytes())
        if selected.any():
            quality_bad = (
                np.asarray(evaluation["execution_quality"], dtype=float)[selected]
                < np.asarray(context["q_min"], dtype=float)[selected] - QOS_TOL
            )
            deadline_bad = (
                np.asarray(evaluation["total_delay"], dtype=float)[selected]
                > np.asarray(context["deadline"], dtype=float)[selected] + QOS_TOL
            )
            qos_bad = quality_bad | deadline_bad
            ir_bad = (
                np.asarray(evaluation["experienced_utility"], dtype=float)[selected]
                < np.asarray(context["U_out"], dtype=float)[selected] - 1e-10
            )
            target_bad = (
                np.asarray(evaluation["implementation_gap"], dtype=float)[selected]
                < -TARGET_TOL
            )
            counters["selected_quality_violations"] += int(quality_bad.sum())
            counters["selected_deadline_violations"] += int(deadline_bad.sum())
            counters["selected_qos_violations"] += int(qos_bad.sum())
            counters["selected_ir_violations"] += int(ir_bad.sum())
            counters["selected_target_violations"] += int(target_bad.sum())
        return original_log(
            slot,
            slot_providers,
            slot_tasks,
            provider_rows,
            task_rows,
            evaluation,
            selected,
        )

    simulator._log_pairs = log_pairs
    started = time.perf_counter()
    with patch.object(simmodule, "evaluate_pairs", side_effect=evaluate), patch.object(
        simmodule, "coverage_first_matching", side_effect=audited_coverage_matching
    ):
        result = simulator.run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    pair_log = result["pair_log"]
    slot_log = result["slot_log"]
    selected = pair_log[pair_log["selected"].astype(bool)]
    assigned = int(summary["num_assigned"])
    tasks = int(summary["total_tasks"])
    qualified = assigned - counters["selected_qos_violations"]

    row = {
        **job,
        "tasks": tasks,
        "assigned": assigned,
        "qualified": qualified,
        "coverage": assigned / max(tasks, 1),
        "qualified_coverage": qualified / max(tasks, 1),
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "mean_quality": float(summary["average_quality"]),
        "high_quality_rate": float(summary["HQR"]),
        "platform_utility": float(summary["platform_utility"]),
        "quality_violations": counters["selected_quality_violations"],
        "deadline_violations": counters["selected_deadline_violations"],
        "qos_violations": counters["selected_qos_violations"],
        "qos_violation_rate": counters["selected_qos_violations"] / max(assigned, 1),
        "ir_violations": int(summary["ir_violations"]),
        "ir_violations_recomputed": counters["selected_ir_violations"],
        "target_violations": int(summary["target_violations"]),
        "target_violations_recomputed": counters["selected_target_violations"],
        "design_reserve": float(slot_log["budget_used"].sum()),
        "settlement_payment_from_slots": float(slot_log["total_payment"].sum()),
        "available_budget": float(slot_log["budget"].sum()),
        "reserve_over_budget_slots": int((slot_log["budget_used"] > slot_log["budget"] + 1e-7).sum()),
        "settlement_over_reserve_slots": int((slot_log["total_payment"] > slot_log["budget_used"] + 1e-7).sum()),
        "settlement_over_budget_slots": int((slot_log["total_payment"] > slot_log["budget"] + 1e-7).sum()),
        "selected_rows": int(len(selected)),
        "mean_base_payment": float(selected["base_payment"].mean()) if len(selected) else float("nan"),
        "negative_base_payment_rows": int((selected["base_payment"] < -1e-12).sum()),
        "event_tape_hash": tape,
        "input_fingerprints": fingerprints,
        "candidate_graph_hash": candidate_hasher.hexdigest(),
        "selection_hash": selection_hasher.hexdigest(),
        "config_raw_sha256": record["raw_sha256"],
        "config_effective_sha256": digest(cfg),
        "instrumented_runtime_s": elapsed,
        "diagnostic_status": result["diagnostics"].get("status"),
        "diagnostics": result["diagnostics"],
        "contract_protocol_version": summary.get("contract_protocol_version"),
        "decision_cost_basis": summary.get("decision_cost_basis"),
        "matching_objective_declared": cfg.get("matching", {}).get("objective"),
        "coverage_first_matcher_calls": matcher_audit["calls"],
        "observed_matching_objective_types": sorted(matcher_audit["objective_types"]),
        "coverage_first_solver_counts": matcher_audit["solver_counts"],
        **counters,
    }
    if row["ir_violations"] != row["ir_violations_recomputed"]:
        raise AssertionError(f"IR audit mismatch in {run_key}")
    if row["target_violations"] != row["target_violations_recomputed"]:
        raise AssertionError(f"Target audit mismatch in {run_key}")
    if abs(row["payment"] - row["settlement_payment_from_slots"]) > 1e-7:
        raise AssertionError(f"Payment audit mismatch in {run_key}")
    if row["selected_rows"] != assigned:
        raise AssertionError(f"Selected-count audit mismatch in {run_key}")
    if row["coverage_first_matcher_calls"] != int(summary["T"]):
        raise AssertionError(f"Coverage-first matcher was not used in every slot: {run_key}")
    if row["observed_matching_objective_types"] != ["coverage_first_payment_second"]:
        raise AssertionError(f"Coverage-first objective audit failed: {run_key}")

    if job == {"scenario": "stationary", "method": job["method"], "seed": 1}:
        target = ROUND / "restricted" / run_key
        target.mkdir(parents=True, exist_ok=True)
        pair_log.to_parquet(target / "pair_log.parquet", index=False)
        slot_log.to_parquet(target / "slot_log.parquet", index=False)
        result["provider_log"].to_parquet(target / "provider_log.parquet", index=False)
        row["restricted_log_sha256"] = {
            path.name: sha256_file(path) for path in sorted(target.glob("*.parquet"))
        }

    dump(cell_path, row)
    return row


def smoke() -> None:
    plan = load(ROUND / "preregistration.json")
    verify(plan)
    checks = []
    for job in plan["smoke_cells"]:
        row = run_one(job)
        checks.extend(
            [
                {"method": job["method"], "field": "matching_objective_declared", "pass": row["matching_objective_declared"] == "coverage_first_payment_second"},
                {"method": job["method"], "field": "observed_matching_objective_types", "pass": row["observed_matching_objective_types"] == ["coverage_first_payment_second"]},
                {"method": job["method"], "field": "coverage_first_matcher_calls", "pass": row["coverage_first_matcher_calls"] == 1000},
                {"method": job["method"], "field": "coverage_first_solver_counts", "pass": sum(row["coverage_first_solver_counts"].values()) == 1000},
                {"method": job["method"], "field": "decision_cost_basis", "pass": row["decision_cost_basis"] == "design_expected_payment"},
                {"method": job["method"], "field": "diagnostic_status", "pass": row["diagnostic_status"] == "ok"},
            ]
        )
    first = load(ROUND / "cells" / "stationary__seed_001__MOI.json")
    second = load(ROUND / "cells" / "stationary__seed_001__PASI.json")
    checks.extend([
        {"method": "paired", "field": "event_tape_hash", "pass": first["event_tape_hash"] == second["event_tape_hash"]},
        {"method": "paired", "field": "input_fingerprints", "pass": first["input_fingerprints"] == second["input_fingerprints"]},
        {"method": "paired", "field": "tasks", "pass": first["tasks"] == second["tasks"] == 80000},
    ])
    result = {"passed": all(item["pass"] for item in checks), "checks": checks, "formal_cells_completed": 2}
    dump(ROUND / "smoke_checks.json", result)
    verify(plan)
    if not result["passed"]:
        raise AssertionError("R72 smoke gate failed; preserve outputs and stop")
    print(json.dumps(result), flush=True)


def execute() -> None:
    plan = load(ROUND / "preregistration.json")
    verify(plan)
    if not load(ROUND / "smoke_checks.json")["passed"]:
        raise AssertionError("Smoke gate did not pass")
    started_path = ROUND / "execution_started.json"
    if not started_path.exists():
        dump(started_path, {"started_utc": utc(), "plan_sha256": sha256_file(ROUND / "preregistration.json")})

    rows = []
    jobs = []
    for job in plan["jobs"]:
        cell_path = ROUND / "cells" / f"{key(job)}.json"
        if cell_path.exists():
            rows.append(load(cell_path))
        else:
            jobs.append(job)
    errors = []
    with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                errors.append({"job": job, "error": traceback.format_exc()})
            progress = {"completed": len(rows), "expected": 180, "errors": errors, "updated_utc": utc()}
            dump(ROUND / "execution_progress.json", progress)
            if len(rows) % 10 == 0 or errors:
                print(json.dumps({"completed": len(rows), "expected": 180, "errors": len(errors)}), flush=True)

    verify(plan)
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key_: value for key_, value in row.items() if not isinstance(value, (dict, list))})
    pd.DataFrame(scalar_rows).sort_values(["scenario", "seed", "method"]).to_csv(
        ROUND / "seed_results.csv", index=False
    )
    completion = {
        "completed": len(rows),
        "expected": 180,
        "errors": errors,
        "finished_utc": utc(),
        "current_core_and_inputs_unchanged": True,
        "new_scenarios_or_seeds": False,
        "references_changed": False,
        "manuscript_changed": False,
    }
    dump(ROUND / "completion.json", completion)
    print(json.dumps(completion), flush=True)
    if len(rows) != 180 or errors:
        raise AssertionError("Incomplete R72 matrix; preserve evidence and do not claim completion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "execute"))
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "execute": execute}[args.action]()
