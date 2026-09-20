"""Portable E7 current-core replay of published R69 cells; no old local snapshots required."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "legacy" / "response_safety" / "current" / "code"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import src.simulator as sim
from src.datasets.trace_loader import load_trace_episode


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def outside(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Inputs/results must remain outside the public package")
    return path


def apply_qos(tasks: pd.DataFrame, level: str) -> pd.DataFrame:
    """Exact QoS shift from original run_e7_phase_diagram.py."""
    shift = {"loose": -0.05, "moderate": 0.0, "strict": 0.05}[level]
    out = tasks.copy()
    out["min_quality"] = np.clip(out["min_quality"].to_numpy(float) + shift,
                                  0.05, out["q_bar"].to_numpy(float) - 0.01)
    return out


def policy(design, phase, slot, magnitude, duration, direction, method):
    """Exact R66 policy; only the contract-context state differs."""
    active = ((int(slot) + phase) % (5 * duration)) < duration
    sign = 1.0 if direction == "runtime_higher" else -1.0
    runtime = np.clip(design + sign * magnitude * active, 0.0, 1.0)
    return runtime, (runtime if method == "PASI" else design.copy()), active


class E7Simulator(sim.Simulator):
    def __init__(self, cfg, data, method, seed):
        super().__init__(cfg, data, method, seed)
        if self.robust_enabled or self.adaptive_response_enabled or self.state_reset_config is not None:
            raise ValueError("E7 must use its original unguarded state policy")
        self.e7 = copy.deepcopy(cfg["e7_mismatch"])
        if not self.e7["enabled"] or self.e7["duty_cycle"] != 0.20:
            raise ValueError("E7 policy mismatch")
        self.phase = np.random.default_rng([seed, 9707]).integers(
            0, 5 * self.e7["duration"], size=len(self.pid_list))
        self.design_equals_true = False
        self.e7_indices = None
        self.e7_slot = None

    def _get_budget(self, tasks):
        self.e7_slot = int(tasks.slot.iloc[0])
        return super()._get_budget(tasks)

    def _state_lower_bound(self, observed, provider_indices, delta_mult):
        self.e7_indices = np.array(provider_indices, copy=True)
        return super()._state_lower_bound(observed, provider_indices, delta_mult)

    def run(self):
        evaluator = sim.evaluate_pairs

        def wrapped(mech, ctx):
            indices = self.e7_indices
            design = self.H_design_state[indices].copy()
            runtime, contract, _ = policy(
                design, self.phase[indices], self.e7_slot,
                self.e7["magnitude"], self.e7["duration"],
                self.e7["direction"], self.method)
            changed = dict(ctx)
            changed.update(H_t=runtime, H_d=contract, H_lower=contract,
                           design_equals_true=False)
            return evaluator(mech, changed)

        with patch.object(sim, "evaluate_pairs", side_effect=wrapped):
            return super().run()


def replay(episodes: Path, output: Path, cell_id: str, seeds: list[int]) -> None:
    episodes, output = outside(episodes), outside(output)
    if output.exists():
        raise FileExistsError("Choose a fresh output; do not overwrite comparisons")
    plan = json.loads((ROOT / "evidence_runners" / "e7" / "preregistration.json").read_text(encoding="utf-8"))
    if cell_id not in plan["configs"]:
        raise ValueError("Cell was not in the published R69 matrix")
    entry = plan["configs"][cell_id]
    cell = entry["cell"]
    with (ROOT / "results" / "e7" / "seed_results.csv").open(newline="", encoding="utf-8-sig") as stream:
        reference = {(r["cell_id"], int(r["seed"]), r["method"]): r for r in csv.DictReader(stream)}
    rows, comparisons = [], []
    for seed in seeds:
        episode = episodes / cell["workload"] / f"seed_{seed:03d}"
        expected = plan["episodes"][f"{cell['workload']}/{seed}"]["sha256"]
        if {name: sha(episode / name) for name in expected} != expected:
            raise ValueError(f"Frozen episode hash mismatch: {seed}")
        cfg = copy.deepcopy(entry["config"])
        cfg["dataset"] = {"processed_dir": str(episode), "use_region": "all", "service_radius_km": 3.0}
        if cfg["matching"]["objective"] != "coverage_first_payment_second":
            raise ValueError("Matching protocol changed")
        data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=episode)
        if data["meta"]["source"] != "geolife" or data["meta"]["redistributable"] is not False:
            raise ValueError("Data provenance changed")
        data["tasks"] = apply_qos(data["tasks"], cell["qos"])
        for method in ("MOI", "PASI"):
            engine = E7Simulator(cfg, data, method, seed)
            result = engine.run()
            summary, slots = result["summary"], result["slot_log"]
            assigned = int(summary["num_assigned"])
            qualified = int(slots["num_qualified_completed"].sum())
            tasks = int(summary["total_tasks"])
            payment = float(summary["cumulative_payment"])
            row = {"cell_id": cell_id, "seed": seed, "method": method,
                   "tasks": tasks, "assigned": assigned,
                   "coverage": assigned / max(tasks, 1),
                   "qualified_coverage": qualified / max(tasks, 1),
                   "qos_violations": assigned - qualified,
                   "payment": payment, "pps": payment / max(assigned, 1),
                   "ir_violations": int(summary["ir_violations"]),
                   "target_violations": int(summary["target_violations"]),
                   "design_reserve": float(slots["budget_used"].sum()),
                   "window_phase_sha256": hashlib.sha256(engine.phase.tobytes()).hexdigest(),
                   "core_sha256": sha(CODE / "src" / "simulator.py")}
            rows.append(row)
            ref = reference[(cell_id, seed, method)]
            for key in ("tasks", "assigned", "coverage", "qualified_coverage", "qos_violations",
                        "payment", "pps", "ir_violations", "target_violations", "design_reserve"):
                got, want = row[key], float(ref[key])
                comparisons.append({"cell_id": cell_id, "seed": seed, "method": method,
                                    "metric": key, "observed": got, "published": want,
                                    "pass": bool(np.isclose(got, want, rtol=0, atol=1e-8))})
            comparisons.append({"cell_id": cell_id, "seed": seed, "method": method,
                                "metric": "window_phase_sha256", "observed": row["window_phase_sha256"],
                                "published": ref["window_phase_sha256"],
                                "pass": row["window_phase_sha256"] == ref["window_phase_sha256"]})
            print(json.dumps({"cell_id": cell_id, "seed": seed, "method": method,
                              "matches": all(x["pass"] for x in comparisons if x["seed"] == seed and x["method"] == method)}), flush=True)
    report = {"cell_id": cell_id, "seeds": seeds, "cells": len(rows), "all_passed": all(x["pass"] for x in comparisons),
              "comparisons": comparisons, "rows": rows, "runner_sha256": sha(Path(__file__)),
              "not_reproduced": ["historical runtime", "private candidate logs", "legacy simulator AST audit"],
              "no_real_economic_observations": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_passed"]:
        raise AssertionError("Differences retained; inspect before citing reproduction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    replay(args.episodes, args.out, args.cell, args.seeds)
