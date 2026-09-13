"""Matching P2 tests (Section 11.6).

- assignment constraints & budget satisfied
- small instances: Lagrangian LB <= MILP exact OPT <= Lagrangian UB
- zero budget -> empty matching
- edge cases: single pair, empty input, rectangular instances
"""

import numpy as np
import pytest

from src.matching import coverage_first_matching, lagrangian_matching, milp_matching


def _random_instance(rng, n_prov, n_task, feas_frac=0.8):
    pairs = []
    for p in range(n_prov):
        for t in range(n_task):
            if rng.random() < feas_frac:
                pairs.append((p, t, rng.uniform(0.1, 5.0), rng.uniform(0.1, 2.0)))
    if not pairs:
        pairs = [(0, 0, 1.0, 0.5)]
    P, T, V, C = map(np.asarray, zip(*pairs))
    return P.astype(int), T.astype(int), V.astype(float), C.astype(float)


def _check_assignment(P, T, sel):
    assert len(set(P[sel].tolist())) == len(sel), "provider used twice"
    assert len(set(T[sel].tolist())) == len(sel), "task used twice"


class TestLagrangianMatching:
    def test_constraints_and_budget(self):
        rng = np.random.default_rng(1)
        for trial in range(20):
            n_p, n_t = rng.integers(2, 12), rng.integers(2, 12)
            P, T, V, C = _random_instance(rng, n_p, n_t)
            budget = float(rng.uniform(0.5, C.sum()))
            res = lagrangian_matching(P, T, V, C, n_p, n_t, budget)
            sel = res["selected_idx"]
            _check_assignment(P, T, sel)
            assert res["total_cost"] <= budget + 1e-6, f"trial {trial}: budget violated"
            assert res["budget_gap"] >= -1e-6

    def test_zero_budget_empty(self):
        P = np.array([0, 1]); T = np.array([0, 1])
        V = np.array([1.0, 2.0]); C = np.array([0.5, 0.5])
        res = lagrangian_matching(P, T, V, C, 2, 2, budget=0.0)
        assert len(res["selected_idx"]) == 0

    def test_single_pair(self):
        res = lagrangian_matching(
            np.array([0]), np.array([0]), np.array([2.0]), np.array([1.0]),
            n_providers=1, n_tasks=1, budget=5.0)
        assert len(res["selected_idx"]) == 1
        assert res["total_value"] == pytest.approx(2.0)

    def test_empty_input(self):
        res = lagrangian_matching(
            np.array([], dtype=int), np.array([], dtype=int),
            np.array([]), np.array([]), 3, 3, budget=1.0)
        assert len(res["selected_idx"]) == 0

    def test_rectangular(self):
        """More providers than tasks and vice versa."""
        rng = np.random.default_rng(3)
        for n_p, n_t in [(10, 3), (3, 10), (1, 8), (8, 1)]:
            P, T, V, C = _random_instance(rng, n_p, n_t, feas_frac=1.0)
            res = lagrangian_matching(P, T, V, C, n_p, n_t, budget=100.0)
            sel = res["selected_idx"]
            _check_assignment(P, T, sel)
            assert len(sel) <= min(n_p, n_t)

    def test_ub_ge_lb(self):
        rng = np.random.default_rng(5)
        for _ in range(30):
            n_p, n_t = rng.integers(2, 10), rng.integers(2, 10)
            P, T, V, C = _random_instance(rng, n_p, n_t)
            budget = float(rng.uniform(0.3, C.sum() * 0.7))
            res = lagrangian_matching(P, T, V, C, n_p, n_t, budget)
            if res["lagrangian_upper_bound"] is not None:
                assert res["lagrangian_upper_bound"] >= res["feasible_lower_bound"] - 1e-7


class TestMILPExact:
    def test_lb_le_opt_le_ub(self):
        """Small instances: LB <= exact OPT <= UB (Section 6.3)."""
        rng = np.random.default_rng(11)
        n_checked = 0
        for trial in range(25):
            n_p, n_t = rng.integers(2, 9), rng.integers(2, 9)
            P, T, V, C = _random_instance(rng, n_p, n_t)
            budget = float(rng.uniform(0.3, C.sum() * 0.8))
            heur = lagrangian_matching(P, T, V, C, n_p, n_t, budget)
            exact = milp_matching(P, T, V, C, n_p, n_t, budget)
            if exact["status"] != "optimal":
                continue
            n_checked += 1
            opt = exact["total_value"]
            lb = heur["feasible_lower_bound"]
            ub = heur["lagrangian_upper_bound"]
            assert lb <= opt + 1e-6, f"trial {trial}: LB {lb} > OPT {opt}"
            if ub is not None:
                assert opt <= ub + 1e-6, f"trial {trial}: OPT {opt} > UB {ub}"
            assert exact["total_cost"] <= budget + 1e-6
        assert n_checked >= 20, "too few MILP instances solved"

    def test_milp_matches_bruteforce_tiny(self):
        """2x2 instance where the optimum is known by enumeration."""
        P = np.array([0, 0, 1, 1]); T = np.array([0, 1, 0, 1])
        V = np.array([3.0, 1.0, 1.0, 3.0]); C = np.array([1.0, 0.5, 0.5, 1.0])
        # Budget 2.0 allows both diagonal pairs: value 6
        res = milp_matching(P, T, V, C, 2, 2, budget=2.0)
        assert res["total_value"] == pytest.approx(6.0)
        # Budget 1.0 allows only one of the value-3 pairs
        res = milp_matching(P, T, V, C, 2, 2, budget=1.0)
        assert res["total_value"] == pytest.approx(3.0)

    def test_lagrangian_reaches_optimum_often(self):
        """Heuristic should be near-optimal on small instances (gap report)."""
        rng = np.random.default_rng(13)
        gaps = []
        for _ in range(20):
            n_p, n_t = 6, 6
            P, T, V, C = _random_instance(rng, n_p, n_t, feas_frac=1.0)
            budget = float(C.sum() * 0.5)
            heur = lagrangian_matching(P, T, V, C, n_p, n_t, budget)
            exact = milp_matching(P, T, V, C, n_p, n_t, budget)
            if exact["status"] == "optimal" and exact["total_value"] > 0:
                gaps.append(1 - heur["feasible_lower_bound"] / exact["total_value"])
        assert np.mean(gaps) < 0.05, f"mean heuristic gap too large: {np.mean(gaps):.4f}"


class TestCoverageFirstMILPFallback:
    def test_large_instance_preserves_lexicographic_objective(self):
        n = 15
        P, T = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        P = P.ravel(); T = T.ravel()
        C = 1.0 + 0.01 * np.abs(P - T)
        V = np.ones_like(C)
        result = coverage_first_matching(P, T, V, C, n, n, budget=15.0)
        assert result["maximum_cardinality"] == 15
        assert result["total_cost"] == pytest.approx(15.0)
        assert result["solver"] == "two_stage_exact_milp"
        _check_assignment(P, T, result["selected_idx"])

    def test_large_instance_respects_budget_before_payment_tie_break(self):
        n = 14
        P, T = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        P = P.ravel(); T = T.ravel()
        C = np.full(len(P), 2.0)
        V = np.ones(len(P))
        result = coverage_first_matching(P, T, V, C, n, n, budget=17.9)
        assert result["maximum_cardinality"] == 8
        assert result["total_cost"] == pytest.approx(16.0)
        _check_assignment(P, T, result["selected_idx"])
