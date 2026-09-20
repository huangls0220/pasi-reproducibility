"""Portable exact-input replay of published E5/E16 outcome columns (not timings or private audit logs)."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "legacy" / "response_safety" / "current" / "code"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outside(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Reconstructed inputs/results must stay outside the public package")
    return path


def study_metadata(study: str) -> tuple[dict, dict]:
    if study not in ("e5", "e16"):
        raise ValueError(study)
    runner_dir = ROOT / "evidence_runners" / study
    plan_file = runner_dir / ("preregistration.json" if study == "e5" else "formal_preregistration.json")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    with (ROOT / "results" / study / "seed_results.csv").open(newline="", encoding="utf-8-sig") as stream:
        reference = {(r["workload"], int(r["seed"]), r["method"]): r for r in csv.DictReader(stream)}
    return plan, reference


def compare(study: str, episodes: Path, output: Path, workload: str, seeds: list[int]) -> None:
    output = outside(output)
    if output.exists():
        raise FileExistsError("Choose a new output file to retain all previous comparisons")
    episodes = outside(episodes)
    plan, reference = study_metadata(study)
    methods = ("MOI", "PASI") if study == "e5" else ("MOI", "PASI", "QUAC-F")
    rows, comparisons = [], []
    for seed in seeds:
        episode = episodes / workload / f"seed_{seed:03d}"
        original = plan["entries"][f"{workload}/{seed}"]
        expected = original["episode_sha256"]
        actual = {name: sha(episode / name) for name in expected}
        if actual != expected:
            raise ValueError(f"Frozen episode hash mismatch: {workload}/{seed}")
        for method in methods:
            cfg = copy.deepcopy(original["config"])
            cfg["dataset"]["processed_dir"] = str(episode)
            if cfg["matching"]["objective"] != "coverage_first_payment_second":
                raise ValueError("Published matching objective mismatch")
            data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=episode)
            if data["meta"]["source"] != "geolife" or data["meta"]["redistributable"] is not False:
                raise ValueError("Unexpected data provenance")
            result = Simulator(cfg, data, method=method, seed=seed).run()
            summary, slots = result["summary"], result["slot_log"]
            assigned = int(summary["num_assigned"])
            qualified = int(slots["num_qualified_completed"].sum())
            tasks = int(summary["total_tasks"])
            payment = float(summary["cumulative_payment"])
            row = {"study": study, "workload": workload, "seed": seed, "method": method,
                   "tasks": tasks, "assigned": assigned, "qualified": qualified,
                   "coverage": assigned / max(tasks, 1), "qos_violations": assigned - qualified,
                   "qos_violation_rate": (assigned - qualified) / max(assigned, 1),
                   "payment": payment, "pps": payment / max(assigned, 1),
                   "ir_violations": int(summary["ir_violations"]),
                   "target_violations": int(summary["target_violations"]),
                   "design_reserve": float(slots["budget_used"].sum()),
                   "run_status": result["diagnostics"].get("status"),
                   "episode_hashes": actual, "source_hash": sha(CODE / "src" / "simulator.py")}
            rows.append(row)
            ref = reference[(workload, seed, method)]
            mapping = {"tasks": "tasks", "assigned": "assigned", "qualified": "qualified",
                       "payment": "payment", "pps": "pps", "qos_violations": "qos_violations",
                       "ir_violations": "ir_violations", "target_violations": "target_violations",
                       "design_reserve": "design_reserve"}
            mapping["coverage"] = "coverage" if study == "e5" else "service_coverage"
            for key, refkey in mapping.items():
                got, want = row[key], float(ref[refkey])
                comparisons.append({"study": study, "workload": workload, "seed": seed, "method": method,
                                    "metric": key, "observed": got, "published": want,
                                    "pass": bool(np.isclose(got, want, rtol=0, atol=1e-8))})
            print(json.dumps({"study": study, "workload": workload, "seed": seed, "method": method,
                              "matches": all(x["pass"] for x in comparisons if x["seed"] == seed and x["method"] == method)}), flush=True)
    report = {"study": study, "cells": len(rows), "comparisons": comparisons,
              "all_passed": all(x["pass"] for x in comparisons),
              "not_reproduced": ["historical runtimes", "private candidate logs", "historical independent matching audit"],
              "no_real_economic_observations": True, "runner_sha256": sha(Path(__file__)), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_passed"]:
        raise AssertionError("Differences retained in output; inspect before citing reproduction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("e5", "e16"))
    parser.add_argument("--episodes", required=True, type=Path, help="Legally reconstructed GeoLife episodes root")
    parser.add_argument("--out", required=True, type=Path, help="Result JSON outside this package")
    parser.add_argument("--workload", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    arguments = parser.parse_args()
    compare(arguments.study, arguments.episodes, arguments.out, arguments.workload, arguments.seeds)
