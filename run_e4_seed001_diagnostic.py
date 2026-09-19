#!/usr/bin/env python3
"""Seed-1 information-value timing diagnostic for legacy E4.

This is deliberately single-process and log-free. It covers the original 24
logical cells once; it is not a replacement for the former 3-repeat Full360.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROUND = Path(__file__).resolve().parent
CODE = ROUND / "core_source"
OUT = ROUND / "e4_seed001_diagnostic"
SCALES = ((25, 20), (50, 40), (100, 80), (200, 160))


def hash_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def run_one(scenario: str, n: int, m: int, method: str) -> dict:
    sys.path.insert(0, str(CODE))
    from src.datasets.synthetic import generate_synthetic_episode
    import src.simulator as simmodule

    plan = json.loads((ROUND / "preregistration.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(plan["config_records"][scenario]["config"])
    cfg["providers"]["N_mean"] = n
    cfg["tasks"]["M_mean"] = m
    cfg["simulation"]["log_level"] = "none"
    cfg["simulation"]["state_reset"] = {"enabled": False}
    cfg["dataset"] = {"pattern": scenario}
    if cfg["matching"].get("objective") != "coverage_first_payment_second":
        raise AssertionError("E4 diagnostic requires coverage-first matching")
    data = generate_synthetic_episode(cfg, seed=1, pattern=scenario)
    tape = {name: hash_frame(value) for name, value in data.items() if isinstance(value, pd.DataFrame)}
    matcher = simmodule.coverage_first_matching
    audit = {"calls": 0, "types": set(), "solvers": {}}

    def audited(*args, **kwargs):
        result = matcher(*args, **kwargs)
        audit["calls"] += 1
        audit["types"].add(result.get("objective_type"))
        solver = result.get("solver", "small_subset_or_empty")
        audit["solvers"][solver] = audit["solvers"].get(solver, 0) + 1
        return result

    simulator = simmodule.Simulator(cfg, data, method="SAMI" if method == "PASI" else method, seed=1)
    start = time.perf_counter()
    with patch.object(simmodule, "coverage_first_matching", side_effect=audited):
        result = simulator.run()
    elapsed = time.perf_counter() - start
    if audit["calls"] != int(result["summary"]["T"]):
        raise AssertionError("Matcher call count mismatch")
    if audit["types"] != {"coverage_first_payment_second"}:
        raise AssertionError(f"Unexpected objective type: {audit['types']}")
    return {
        "scenario": scenario,
        "N": n,
        "M": m,
        "seed": 1,
        "method": method,
        "core_time_s": elapsed,
        "assigned": int(result["summary"]["num_assigned"]),
        "execution_settlement": float(result["summary"]["cumulative_payment"]),
        "coverage_first_calls": audit["calls"],
        "solver_counts": json.dumps(audit["solvers"], sort_keys=True),
        "event_fingerprints": json.dumps(tape, sort_keys=True),
        "ir_violations": int(result["summary"]["ir_violations"]),
        "target_violations": int(result["summary"]["target_violations"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [
        (scenario, n, m, method)
        for scenario in ("stationary", "burst", "dynamic")
        for n, m in SCALES
        for method in ("MOI", "PASI")
    ]
    for index, job in enumerate(jobs, start=1):
        row = run_one(*job)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "seed001_runtime.csv", index=False)
        print(f"[{index:02d}/24] {job[0]} {job[1]}x{job[2]} {job[3]} {row['core_time_s']:.3f}s", flush=True)
    current = pd.DataFrame(rows)
    legacy_path = ROUND / "legacy_artifacts" / "E4" / "sealed_evidence" / "pasi_e4_full360_final_evidence_v1_0" / "logical_run_inventory" / "logical_run_inventory.csv"
    legacy = pd.read_csv(legacy_path)
    legacy = legacy[legacy["seed"] == 1].rename(columns={"mechanism": "method", "core_time_s_median": "legacy_core_time_s_median"})
    compare = current.merge(
        legacy[["scenario", "N", "M", "seed", "method", "legacy_core_time_s_median"]],
        on=["scenario", "N", "M", "seed", "method"], how="left", validate="one_to_one",
    )
    compare["current_over_legacy_ratio"] = compare["core_time_s"] / compare["legacy_core_time_s_median"]
    compare.to_csv(OUT / "seed001_runtime_comparison.csv", index=False)
    audit = {
        "logical_cells": len(current),
        "expected_cells": 24,
        "repeats_per_cell": 1,
        "not_a_full360_replacement": True,
        "all_coverage_first": bool((current["coverage_first_calls"] == 1000).all()),
        "ir_violations": int(current["ir_violations"].sum()),
        "target_violations": int(current["target_violations"].sum()),
        "reference_changes": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
