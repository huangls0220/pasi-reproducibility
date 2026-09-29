"""Plan R76 from R75 calibration evidence without reading R75 test outcomes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
R75 = ROOT.parent / "round75-risk-certification-v2" / "results"
ALPHA = 0.005
DELTA = 0.05
CANDIDATE = 0.75
SCENARIOS = ("stable", "down_step", "up_step", "down_ramp", "alternating", "hidden_blocks")
SAMPLE_SIZES = (1277, 1350, 1400, 1500, 1600)
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260928


def binary_kl(p: np.ndarray | float, q: float) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(values == 0.0, 0.0, values * np.log(values / q))
        right = np.where(values == 1.0, 0.0, (1.0 - values) * np.log((1.0 - values) / (1.0 - q)))
    return left + right


def minimum_n_for_observed_mean(mean: float, tail: float) -> int:
    divergence = float(binary_kl(mean, ALPHA))
    return math.ceil(math.log(1.0 / tail) / divergence)


def critical_mean(n: int, tail: float) -> float:
    target = math.log(1.0 / tail)
    lo, hi = 0.0, ALPHA
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if n * float(binary_kl(mid, ALPHA)) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def bootstrap_power(loss_matrix: np.ndarray, n: int, tail: float, rng: np.random.Generator) -> float:
    passed = 0
    completed = 0
    while completed < BOOTSTRAP_REPLICATES:
        batch = min(250, BOOTSTRAP_REPLICATES - completed)
        indices = rng.integers(0, len(loss_matrix), size=(batch, n))
        means = loss_matrix[indices].mean(axis=1)
        cell_pass = (means < ALPHA) & (n * binary_kl(means, ALPHA) >= math.log(1.0 / tail))
        passed += int(cell_pass.all(axis=1).sum())
        completed += batch
    return passed / BOOTSTRAP_REPLICATES


def main() -> None:
    calibration_path = R75 / "r75_calibration_seed_results.csv"
    summary_path = R75 / "r75_calibration_summary.csv"
    calibration = pd.read_csv(calibration_path)
    if "split" not in calibration or not calibration["split"].eq("calibration").all():
        raise ValueError("R75 calibration file is not calibration-only")
    candidate = calibration[calibration["global_guardrail_weight"] == CANDIDATE].copy()
    pivot = candidate.pivot(index="seed", columns="scenario", values="episode_loss")
    pivot = pivot.loc[:, list(SCENARIOS)]
    if pivot.isna().any().any() or len(pivot) != 1277:
        raise ValueError("Incomplete R75 calibration evidence")

    tail = DELTA / len(SCENARIOS)
    means = pivot.mean(axis=0)
    cell_plan = pd.DataFrame({
        "scenario": list(SCENARIOS),
        "r75_calibration_mean_episode_loss": [float(means[s]) for s in SCENARIOS],
        "minimum_n_at_observed_mean": [minimum_n_for_observed_mean(float(means[s]), tail) for s in SCENARIOS],
    })
    cell_plan.to_csv(ROOT / "r76_information_value_cells.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    loss_matrix = pivot.to_numpy(float)
    power_rows = []
    for n in SAMPLE_SIZES:
        power_rows.append({
            "n_per_scenario": n,
            "critical_sample_mean": critical_mean(n, tail),
            "estimated_simultaneous_pass_probability": bootstrap_power(loss_matrix, n, tail, rng),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })
    power = pd.DataFrame(power_rows)
    power.to_csv(ROOT / "r76_information_value_power.csv", index=False)

    summary = pd.read_csv(summary_path)
    coverage = summary[summary["global_guardrail_weight"].isin([CANDIDATE, 1.0])].pivot(
        index="scenario", columns="global_guardrail_weight", values="mean_assignment_coverage"
    )
    gains = (coverage[CANDIDATE] - coverage[1.0]) * 100.0
    payload = {
        "source": "R75 calibration only",
        "r75_test_outcomes_read": False,
        "candidate_g": CANDIDATE,
        "alpha": ALPHA,
        "delta": DELTA,
        "scenario_tail": tail,
        "chosen_n_per_scenario": 1600,
        "chosen_n_planned_pass_probability": float(
            power.loc[power["n_per_scenario"] == 1600, "estimated_simultaneous_pass_probability"].iloc[0]
        ),
        "mean_assignment_coverage_gain_pp_over_g1": float(gains.mean()),
        "scenario_assignment_coverage_gain_pp": {str(k): float(v) for k, v in gains.items()},
        "decision": "GO: freeze one nontrivial candidate and use fresh R76 calibration/test seeds",
        "interpretation": "Planning evidence only; not a risk certificate and not a test-set-selected policy",
    }
    (ROOT / "r76_information_value.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
