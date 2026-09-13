"""E27: source-native public payment baselines on frozen GeoLife tapes.

QIM-E and CSOPT are evaluated on the same single-minded, fixed-coverage
special case.  Both receive direct reserve bids, unlike PASI.  Candidate
edges have already passed PASI's physical, contract, IR, and QoS gates.
The experiment therefore supplies a payment-objective external anchor; it
does not claim identical information or contract semantics.
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

from scripts.run_a1_3_complete import legacy_cfg
from src.baselines.csopt import run_csopt
from src.baselines.qim_e import run_qim_e
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


WORKLOADS = ("low", "medium", "high")
_EPS = 1e-10


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tape_hash(processed: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "meta.json",
        "provider_static.parquet",
        "providers.parquet",
        "tasks.parquet",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((processed / name).read_bytes())
    return digest.hexdigest()


def _legal_home_pairs(pair_log: pd.DataFrame) -> pd.DataFrame:
    legal = pair_log[
        pair_log["feasible_contract"].astype(bool)
        & pair_log["ir_ok"].astype(bool)
        & (
            pair_log["execution_quality"] + _EPS
            >= pair_log["task_min_quality"]
        )
    ].copy()
    legal.sort_values(
        ["slot", "provider_id", "spatial_distance_km", "task_id"],
        inplace=True,
    )
    return legal.drop_duplicates(["slot", "provider_id"], keep="first")


def _evaluate_slot(home: pd.DataFrame) -> dict[str, float | int | bool]:
    counts = home.groupby("task_id", sort=True).size()
    comparable_ids = counts[counts >= 2].index
    comparable = home[home["task_id"].isin(comparable_ids)].copy()
    if comparable.empty:
        return {
            "comparable_tasks": 0,
            "candidate_bids": 0,
            "pasi_payment": 0.0,
            "qim_payment": 0.0,
            "csopt_payment": 0.0,
            "pasi_quality_sum": 0.0,
            "qim_quality_sum": 0.0,
            "csopt_quality_sum": 0.0,
            "qim_csopt_winner_equivalent": True,
            "qim_csopt_payment_gap": 0.0,
            "pasi_ir_violations": 0,
            "qim_ir_violations": 0,
            "csopt_ir_violations": 0,
            "pasi_qos_violations": 0,
            "qim_qos_violations": 0,
            "csopt_qos_violations": 0,
            "zero_reserve_bids": 0,
        }

    task_ids = sorted(comparable_ids.astype(str).tolist())
    task_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
    comparable.sort_values(["provider_id", "task_id"], inplace=True)
    comparable.reset_index(drop=True, inplace=True)
    costs = comparable["effective_reserve_cost"].to_numpy(dtype=float)
    if np.any(costs < -_EPS) or np.any(costs > 4.0 + _EPS):
        raise ValueError("A QIM-E reserve bid lies outside U(0,4]")
    costs = np.clip(costs, 0.0, 4.0)

    qualities = np.zeros((len(comparable), len(task_ids)), dtype=float)
    interests = np.zeros((len(comparable), len(task_ids)), dtype=bool)
    for row_idx, row in enumerate(comparable.itertuples(index=False)):
        column = task_index[str(row.task_id)]
        qualities[row_idx, column] = float(row.execution_quality)
        interests[row_idx, column] = True
    requirements = np.asarray([
        float(
            comparable.loc[
                comparable["task_id"].astype(str) == task_id,
                "task_min_quality",
            ].iloc[0]
        )
        for task_id in task_ids
    ])
    bidder_ids = comparable["provider_id"].astype(str).to_numpy()

    qim = run_qim_e(
        costs, qualities, requirements, bidder_ids=bidder_ids
    )
    csopt = run_csopt(
        costs, interests, redundancy=1, bidder_ids=bidder_ids
    )
    if len(qim.winners) != len(task_ids):
        raise AssertionError("QIM-E did not select one bidder per task")
    if int(csopt.assignment.sum()) != len(task_ids):
        raise AssertionError("CSOPT did not select one bidder per task")

    qim_rows = comparable.iloc[qim.winners]
    csopt_rows = comparable.iloc[csopt.winners]
    if qim_rows["task_id"].nunique() != len(task_ids):
        raise AssertionError("QIM-E selected duplicate task winners")
    if csopt_rows["task_id"].nunique() != len(task_ids):
        raise AssertionError("CSOPT selected duplicate task winners")

    pasi_rows = (
        comparable.sort_values(
            ["task_id", "expected_contract_cost", "provider_id"]
        )
        .drop_duplicates("task_id", keep="first")
    )
    if len(pasi_rows) != len(task_ids):
        raise AssertionError("PASI did not retain one legal winner per task")

    qim_payments = qim.payments[qim.winners]
    csopt_payments = csopt.payments[csopt.winners]
    qim_keys = set(
        zip(qim_rows["task_id"].astype(str), qim_rows["provider_id"].astype(str))
    )
    csopt_keys = set(
        zip(
            csopt_rows["task_id"].astype(str),
            csopt_rows["provider_id"].astype(str),
        )
    )
    csopt_quality = qualities[csopt.winners].max(axis=0)
    return {
        "comparable_tasks": len(task_ids),
        "candidate_bids": len(comparable),
        "pasi_payment": float(pasi_rows["expected_contract_cost"].sum()),
        "qim_payment": float(qim_payments.sum()),
        "csopt_payment": float(csopt_payments.sum()),
        "pasi_quality_sum": float(pasi_rows["execution_quality"].sum()),
        "qim_quality_sum": float(qim_rows["execution_quality"].sum()),
        "csopt_quality_sum": float(csopt_rows["execution_quality"].sum()),
        "qim_csopt_winner_equivalent": qim_keys == csopt_keys,
        "qim_csopt_payment_gap": float(qim_payments.sum() - csopt_payments.sum()),
        "pasi_ir_violations": int(
            np.sum(pasi_rows["experienced_utility"] < -_EPS)
        ),
        "qim_ir_violations": int(
            np.sum(qim_payments + _EPS < costs[qim.winners])
        ),
        "csopt_ir_violations": int(
            np.sum(csopt_payments + _EPS < costs[csopt.winners])
        ),
        "pasi_qos_violations": int(
            np.sum(
                pasi_rows["execution_quality"].to_numpy() + _EPS
                < pasi_rows["task_min_quality"].to_numpy()
            )
        ),
        "qim_qos_violations": int(
            np.sum(qim.achieved_quality + _EPS < requirements)
        ),
        "csopt_qos_violations": int(
            np.sum(csopt_quality + _EPS < requirements)
        ),
        "zero_reserve_bids": int(np.sum(costs <= _EPS)),
    }


def run_seed(episodes: Path, workload: str, seed: int) -> dict:
    processed = episodes / workload / f"seed_{seed:03d}"
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    cfg = legacy_cfg(
        T=int(meta["T"]),
        N=int(meta["n_providers"]),
        M=4,
        pattern="stationary",
    )
    cfg["simulation"]["log_level"] = "full"
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    cfg["dataset"] = {
        "processed_dir": str(processed),
        "use_region": "all",
        "service_radius_km": 3.0,
    }
    data = load_trace_episode(
        "geolife", cfg, seed=seed, processed_dir=processed
    )
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    home = _legal_home_pairs(result["pair_log"])
    task_counts = home.groupby(["slot", "task_id"]).size()
    comparable_index = task_counts[task_counts >= 2].index
    home_index = pd.MultiIndex.from_frame(home[["slot", "task_id"]])
    comparable_home = home[home_index.isin(comparable_index)]
    slot_results = [
        _evaluate_slot(slot)
        for _, slot in home.groupby("slot", sort=True)
    ]
    details = pd.DataFrame(slot_results)
    totals = details.sum(numeric_only=True)
    comparable = int(totals.get("comparable_tasks", 0))
    tasks_with_candidate = int(home.groupby(["slot", "task_id"]).ngroups)
    total_tasks = int(len(data["tasks"]))
    row = {
        "dataset": "GeoLife",
        "workload": workload,
        "seed": seed,
        "tape_sha256": tape_hash(processed),
        "total_tasks": total_tasks,
        "tasks_with_legal_candidate": tasks_with_candidate,
        "comparable_tasks": comparable,
        "excluded_lt_two_candidates": tasks_with_candidate - comparable,
        "excluded_no_legal_candidate": total_tasks - tasks_with_candidate,
        "candidate_bids": int(totals.get("candidate_bids", 0)),
        "zero_reserve_bids": int(totals.get("zero_reserve_bids", 0)),
        "min_reserve_bid": float(comparable_home["effective_reserve_cost"].min()),
        "max_reserve_bid": float(comparable_home["effective_reserve_cost"].max()),
        "pasi_payment": float(totals.get("pasi_payment", 0.0)),
        "qim_payment": float(totals.get("qim_payment", 0.0)),
        "csopt_payment": float(totals.get("csopt_payment", 0.0)),
        "pasi_pps": float(totals.get("pasi_payment", 0.0)) / max(comparable, 1),
        "qim_pps": float(totals.get("qim_payment", 0.0)) / max(comparable, 1),
        "csopt_pps": float(totals.get("csopt_payment", 0.0)) / max(comparable, 1),
        "pasi_mean_quality": float(totals.get("pasi_quality_sum", 0.0)) / max(comparable, 1),
        "qim_mean_quality": float(totals.get("qim_quality_sum", 0.0)) / max(comparable, 1),
        "csopt_mean_quality": float(totals.get("csopt_quality_sum", 0.0)) / max(comparable, 1),
        "qim_csopt_equivalent_slots": int(details["qim_csopt_winner_equivalent"].sum()),
        "evaluated_slots": int(len(details)),
        "qim_csopt_max_abs_payment_gap": float(details["qim_csopt_payment_gap"].abs().max()),
        "pasi_ir_violations": int(totals.get("pasi_ir_violations", 0)),
        "qim_ir_violations": int(totals.get("qim_ir_violations", 0)),
        "csopt_ir_violations": int(totals.get("csopt_ir_violations", 0)),
        "pasi_qos_violations": int(totals.get("pasi_qos_violations", 0)),
        "qim_qos_violations": int(totals.get("qim_qos_violations", 0)),
        "csopt_qos_violations": int(totals.get("csopt_qos_violations", 0)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    row["pasi_minus_qim_pps"] = row["pasi_pps"] - row["qim_pps"]
    row["pasi_minus_csopt_pps"] = row["pasi_pps"] - row["csopt_pps"]
    return row


def bootstrap_ci(
    values: np.ndarray, seed: int, draws: int = 20000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[
        rng.integers(0, len(values), size=(draws, len(values)))
    ].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for workload_idx, workload in enumerate(WORKLOADS, start=1):
        part = frame[frame["workload"] == workload].sort_values("seed")
        dq = part["pasi_minus_qim_pps"].to_numpy(dtype=float)
        dc = part["pasi_minus_csopt_pps"].to_numpy(dtype=float)
        qlo, qhi = bootstrap_ci(dq, 20260950 + workload_idx)
        clo, chi = bootstrap_ci(dc, 20260960 + workload_idx)
        rows.append({
            "workload": workload,
            "n_paired_seeds": len(part),
            "mean_comparable_tasks": part["comparable_tasks"].mean(),
            "mean_comparable_fraction_all_tasks": (
                part["comparable_tasks"] / part["total_tasks"]
            ).mean(),
            "pasi_mean_pps": part["pasi_pps"].mean(),
            "qim_e_mean_pps": part["qim_pps"].mean(),
            "csopt_mean_pps": part["csopt_pps"].mean(),
            "pasi_minus_qim_e_pps": dq.mean(),
            "pasi_minus_qim_e_ci_low": qlo,
            "pasi_minus_qim_e_ci_high": qhi,
            "pasi_minus_csopt_pps": dc.mean(),
            "pasi_minus_csopt_ci_low": clo,
            "pasi_minus_csopt_ci_high": chi,
            "pasi_lower_than_qim_seed_count": int((dq < 0).sum()),
            "pasi_lower_than_csopt_seed_count": int((dc < 0).sum()),
            "qim_csopt_all_slots_equivalent": bool(
                (
                    part["qim_csopt_equivalent_slots"]
                    == part["evaluated_slots"]
                ).all()
            ),
            "qim_csopt_max_abs_payment_gap": part[
                "qim_csopt_max_abs_payment_gap"
            ].max(),
            "total_ir_violations": int(
                part[[
                    "pasi_ir_violations",
                    "qim_ir_violations",
                    "csopt_ir_violations",
                ]].sum().sum()
            ),
            "total_qos_violations": int(
                part[[
                    "pasi_qos_violations",
                    "qim_qos_violations",
                    "csopt_qos_violations",
                ]].sum().sum()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "e27_public_payment_baselines",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(workload, seed) for workload in args.workloads for seed in seeds]
    rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_seed, args.episodes, workload, seed): (workload, seed)
            for workload, seed in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            workload, seed = futures[future]
            rows.append(future.result())
            print(
                f"[{done:03d}/{len(jobs)}] {workload} seed={seed}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values(["workload", "seed"])
    summary = summarize(frame)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(args.output / f"e27_{suffix}_seed_results.csv", index=False)
    summary.to_csv(args.output / f"e27_{suffix}_summary.csv", index=False)
    metadata = {
        "experiment": "E27 source-native public payment baselines",
        "sources": {
            "QIM-E": "Wang et al., IEEE ICDCS 2016, doi:10.1109/ICDCS.2016.30",
            "CSOPT": "Chatzopoulos et al., IEEE MASS 2018, arXiv:1808.04056",
        },
        "implementation": "independent implementations of published algorithms; not author code",
        "public_method_changes": "none to selection or payment rules",
        "published_special_case": {
            "QIM-E": "single-minded quality vectors with maximum capped QoC",
            "CSOPT": "r=1 and single-minded interest sets",
        },
        "shared_protocol": [
            "GeoLife episode",
            "task and candidate identity",
            "QoS-prequalified candidate set",
            "fixed one-provider-per-task coverage",
            "modeled economic state",
            "seed",
        ],
        "information_difference": "QIM-E and CSOPT observe direct effective reserve bids; PASI reconstructs its contract from design/runtime state",
        "contract_difference": "QIM-E and CSOPT use critical/externality payments; PASI uses its nonnegative-base probabilistic contract",
        "comparison_scope": "tasks with at least two single-minded QoS-legal candidates",
        "trace_fields": ["provider identity", "slot", "latitude", "longitude"],
        "model_generated_fields": [
            "tasks",
            "cost parameters",
            "capacity",
            "quality response",
            "reserve bids",
            "contracts",
        ],
        "payment_status": "mechanism-computed outcome, not an observed field",
        "raw_trace_redistributed": False,
        "seeds": seeds,
        "workloads": args.workloads,
        "runs": len(frame),
        "elapsed_seconds": time.time() - started,
        "source_hashes": {
            "runner": file_hash(Path(__file__)),
            "simulator": file_hash(ROOT / "src" / "simulator.py"),
            "pair_eval": file_hash(ROOT / "src" / "pair_eval.py"),
            "qim_e": file_hash(ROOT / "src" / "baselines" / "qim_e.py"),
            "csopt": file_hash(ROOT / "src" / "baselines" / "csopt.py"),
        },
    }
    (args.output / f"e27_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
