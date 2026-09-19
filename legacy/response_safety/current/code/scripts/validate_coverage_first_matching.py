"""Randomized exhaustive check of the exact coverage-first matcher."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matching import coverage_first_matching


def exhaustive(P: np.ndarray, T: np.ndarray, costs: np.ndarray,
               budget: float) -> tuple[int, float]:
    best_cardinality = 0
    best_cost = 0.0
    for size in range(1, len(P) + 1):
        for selected in itertools.combinations(range(len(P)), size):
            if len({int(P[k]) for k in selected}) != size:
                continue
            if len({int(T[k]) for k in selected}) != size:
                continue
            total = float(costs[list(selected)].sum())
            if total > budget + 1e-9:
                continue
            if size > best_cardinality or (size == best_cardinality and total < best_cost):
                best_cardinality, best_cost = size, total
    return best_cardinality, best_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(20260828)
    for case in range(args.cases):
        n_providers = int(rng.integers(1, 6))
        n_tasks = int(rng.integers(1, 5))
        all_edges = [(p, t) for p in range(n_providers) for t in range(n_tasks)]
        mask = rng.random(len(all_edges)) < 0.65
        edges = [edge for edge, keep in zip(all_edges, mask) if keep]
        if not edges:
            edges = [all_edges[int(rng.integers(0, len(all_edges)))]]
        P = np.asarray([p for p, _ in edges], dtype=int)
        T = np.asarray([t for _, t in edges], dtype=int)
        costs = rng.uniform(0.05, 3.0, size=len(edges))
        values = rng.uniform(0.1, 2.0, size=len(edges))
        budget = float(rng.uniform(0.01, 8.0))
        actual = coverage_first_matching(
            P, T, values, costs, n_providers, n_tasks, budget)
        expected_cardinality, expected_cost = exhaustive(P, T, costs, budget)
        if int(actual["maximum_cardinality"]) != expected_cardinality:
            raise AssertionError((case, "cardinality", actual, expected_cardinality))
        if not np.isclose(float(actual["total_cost"]), expected_cost, atol=1e-9):
            raise AssertionError((case, "cost", actual, expected_cost))
    print(f"PASS: {args.cases} randomized cases match exhaustive lexicographic optima")


if __name__ == "__main__":
    main()
