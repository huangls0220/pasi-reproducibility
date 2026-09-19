"""E24: robust-feasible-set coverage and payment certificates.

Under no current-slot response observation, a deterministic guarantee over a
declared response envelope can select only edges feasible for every response
in that envelope.  This runner checks that PASI's coverage-first/payment-
second solver attains the exact lexicographic optimum on that robust graph for
every slot, and independently re-solves five preregistered slots per seed with
SciPy/HiGHS MILP.
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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e20_envelope_tradeoff import build_config
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

LOWER = 0.70
UPPER = 1.30
PROFILES = ("frozen_point", "fixed_envelope")


def independent_lexicographic_milp(pair: pd.DataFrame, budget: float,
                                 cost_column: str = "decision_contract_cost") -> tuple[int, float]:
    """Optimize the supplied graph, NOT certify its contract-domain completeness.

    New runs must use decision_contract_cost; an archived graph may be
    audited with cost_column='expected_contract_cost' explicitly. Never
    silently mix a reserved-budget objective with ex-post settlement costs.
    """
    feasible = pair[pair["feasible_contract"]].copy()
    if feasible.empty or budget <= 0.0:
        return 0, 0.0
    p_codes, providers = pd.factorize(feasible["provider_id"], sort=True)
    t_codes, tasks = pd.factorize(feasible["task_id"], sort=True)
    costs = feasible[cost_column].to_numpy(float)
    n_edges = len(feasible)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for edge, (provider, task) in enumerate(zip(p_codes, t_codes)):
        rows.extend((int(provider), len(providers) + int(task)))
        cols.extend((edge, edge))
        data.extend((1.0, 1.0))
    assignment = coo_matrix(
        (data, (rows, cols)), shape=(len(providers) + len(tasks), n_edges)
    ).tocsr()
    base = vstack((assignment, coo_matrix(costs.reshape(1, -1))), format="csr")
    lb = np.full(base.shape[0], -np.inf)
    ub = np.r_[np.ones(len(providers) + len(tasks)), float(budget)]
    bounds = Bounds(np.zeros(n_edges), np.ones(n_edges))
    integrality = np.ones(n_edges, dtype=int)
    stage_one = milp(
        c=-np.ones(n_edges), integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(base, lb, ub), options={"disp": False},
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError(f"independent cardinality MILP failed: {stage_one.message}")
    cardinality = int(round(np.sum(stage_one.x > 0.5)))
    if cardinality == 0:
        return 0, 0.0
    cardinality_row = coo_matrix(np.ones((1, n_edges))).tocsr()
    stage_two_matrix = vstack((base, cardinality_row), format="csr")
    stage_two = milp(
        c=costs, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(
            stage_two_matrix,
            np.r_[lb, float(cardinality)],
            np.r_[ub, float(cardinality)],
        ),
        options={"disp": False},
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError(f"independent payment MILP failed: {stage_two.message}")
    selected = stage_two.x > 0.5
    return cardinality, float(costs[selected].sum())


def certificate_slots(slots: np.ndarray) -> set[int]:
    """Five fixed positions, chosen without looking at outcomes."""
    slots = np.asarray(sorted(set(int(value) for value in slots)), dtype=int)
    if len(slots) <= 5:
        return set(slots.tolist())
    positions = np.floor(np.linspace(0, len(slots) - 1, 5)).astype(int)
    return set(slots[positions].tolist())


def run_one(episodes: Path, profile: str, seed: int) -> tuple[dict, list[dict]]:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), profile, LOWER, UPPER)
    cfg["simulation"]["log_level"] = "full"
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    tasks["response_ratio_audit"] = np.ones(len(tasks), dtype=float)
    data["tasks"] = tasks
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    slot = result["slot_log"].copy()
    pair = result["pair_log"].copy()
    nonempty = slot[slot["num_providers"] > 0].copy()
    internal_cardinality = np.isclose(
        nonempty["num_assigned"].to_numpy(float),
        pd.to_numeric(nonempty["LB"], errors="coerce").to_numpy(float),
        equal_nan=False,
    )
    internal_gap = np.isclose(
        pd.to_numeric(nonempty["optimality_gap"], errors="coerce").fillna(0.0),
        0.0,
        atol=1e-12,
    )
    sampled = certificate_slots(nonempty["slot"].to_numpy(int)) if profile == "fixed_envelope" else set()
    slot_rows: list[dict] = []
    independent_passes = []
    for local_index, row in enumerate(nonempty.itertuples(index=False)):
        current = pair[pair["slot"] == row.slot]
        checked = int(row.slot) in sampled
        oracle_k = np.nan
        oracle_cost = np.nan
        oracle_pass = True
        if checked:
            oracle_k, oracle_cost = independent_lexicographic_milp(current, float(row.budget))
            oracle_pass = (
                oracle_k == int(row.num_assigned)
                and abs(oracle_cost - float(row.budget_used)) <= 1e-7
            )
            independent_passes.append(oracle_pass)
        if profile == "fixed_envelope":
            slot_rows.append({
                "seed": seed,
                "slot": int(row.slot),
                "robust_feasible_edges": int(row.num_feasible_pairs),
                "selected_cardinality": int(row.num_assigned),
                "selected_payment": float(row.budget_used),
                "settled_expected_payment": float(row.total_payment),
                "certificate_cost_basis": "decision_contract_cost",
                "budget": float(row.budget),
                "internal_exact_cardinality": bool(internal_cardinality[local_index]),
                "internal_zero_gap": bool(internal_gap[local_index]),
                "independent_milp_checked": checked,
                "independent_cardinality": oracle_k,
                "independent_payment": oracle_cost,
                "independent_milp_pass": bool(oracle_pass),
            })
    summary = result["summary"]
    assigned = int(summary["num_assigned"])
    run_row = {
        "profile": profile,
        "certificate_cost_basis": "decision_contract_cost",
        "seed": seed,
        "service_coverage": float(summary["assignment_ratio"]),
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "assigned": assigned,
        "slots_checked_internal": int(len(nonempty)),
        "all_slots_internal_exact": bool(internal_cardinality.all() and internal_gap.all()),
        "slots_checked_independent": int(len(independent_passes)),
        "all_sampled_independent_exact": bool(all(independent_passes)),
        "test_side_response_used_for_decision": False,
        "declared_response_lower": LOWER if profile == "fixed_envelope" else 1.0,
        "declared_response_upper": UPPER if profile == "fixed_envelope" else 1.0,
        "event_tape_hash": tape_hash(folder),
        "runtime_s": time.perf_counter() - started,
        "status": result["diagnostics"].get("status", "unknown"),
    }
    return run_row, slot_rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    frozen = frame[frame["profile"] == "frozen_point"].set_index("seed")
    rows = []
    for index, (profile, cell) in enumerate(frame.groupby("profile", sort=False), 1):
        cell = cell.set_index("seed").sort_index()
        ref = frozen.loc[cell.index]
        row = {
            "profile": profile,
            "n_pairs": int(len(cell)),
            "all_runs_ok": bool(cell["status"].isin(["ok", "completed"]).all()),
            "all_slots_internal_exact": bool(cell["all_slots_internal_exact"].all()),
            "all_sampled_independent_exact": bool(cell["all_sampled_independent_exact"].all()),
            "internal_slots_total": int(cell["slots_checked_internal"].sum()),
            "independent_slots_total": int(cell["slots_checked_independent"].sum()),
        }
        for metric_index, metric in enumerate(("service_coverage", "pps"), 1):
            values = cell[metric].to_numpy(float)
            delta = values - ref[metric].to_numpy(float)
            lo, hi = bootstrap_ci(values, 151_000 + 100 * index + metric_index)
            dlo, dhi = bootstrap_ci(delta, 152_000 + 100 * index + metric_index)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"delta_frozen_{metric}"] = float(delta.mean())
            row[f"delta_frozen_{metric}_ci_low"] = dlo
            row[f"delta_frozen_{metric}_ci_high"] = dhi
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "e24_robust_certificate")
    args = parser.parse_args()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(args.episodes, profile, seed) for profile in PROFILES for seed in seeds]
    rows: list[dict] = []
    slot_rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, profile, seed = futures[future]
            row, certificates = future.result()
            rows.append(row)
            slot_rows.extend(certificates)
            print(f"[{index:03d}/{len(jobs)}] E24 {profile} seed={seed}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["profile", "seed"])
    certificates = pd.DataFrame(slot_rows).sort_values(["seed", "slot"])
    summary = summarize(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(args.out / f"e24_{suffix}_seed_results.csv", index=False)
    certificates.to_csv(args.out / f"e24_{suffix}_slot_certificates.csv", index=False)
    summary.to_csv(args.out / f"e24_{suffix}_summary.csv", index=False)
    metadata = {
        "experiment": "E24 robust feasible-set lexicographic optimality certificate",
        "theorem_scope": (
            "deterministic policies with no current-slot response observation, "
            "declared response envelope [0.70,1.30], PASI bounded probability-bonus "
            "contract class, common candidate graph and slot budget"
        ),
        "certificate": (
            "all slots use the exact coverage-first/payment-second solver; five "
            "outcome-independent slots per seed are independently re-solved by HiGHS MILP"
        ),
        "seeds": seeds,
        "runs": int(len(frame)),
        "elapsed_seconds": time.time() - started,
        "test_side_response_used_for_decision": False,
        "raw_or_derived_geolife_redistributed": False,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matching_hash": hashlib.sha256((ROOT / "src" / "matching.py").read_bytes()).hexdigest(),
        "simulator_hash": hashlib.sha256((ROOT / "src" / "simulator.py").read_bytes()).hexdigest(),
    }
    (args.out / f"e24_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
