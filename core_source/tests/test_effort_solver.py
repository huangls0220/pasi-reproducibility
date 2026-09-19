"""Tests for effort_solver.py — Sections 5.5–5.6, 5.10."""

import math
import random

import pytest

from src.effort_solver import (
    compute_effort_bounds,
    s_required,
    solve_a_reinforcement,
    solve_a_star,
    solve_a_system,
)
from src.quality_cost import C, Q, g


class TestSystemEffort:
    """a_system should be within dense-grid tolerance."""

    def test_boundary_a_min(self):
        """If f(a_min) <= 0, a_system = a_min."""
        # Very low value V → system doesn't want much effort
        a = solve_a_system(
            V=0.1, q_bar=0.9, kappa=2.0, alpha=0.5, beta=0.2, L=1.0, a_min=0.3,
        )
        assert math.isclose(a, 0.3, rel_tol=1e-6)

    def test_boundary_one(self):
        """If f(1) >= 0, a_system = 1."""
        # Very high value V → system wants maximum effort
        a = solve_a_system(
            V=100.0, q_bar=0.9, kappa=2.0, alpha=0.01, beta=0.01, L=1.0, a_min=0.1,
        )
        assert math.isclose(a, 1.0, rel_tol=1e-6)

    def test_interior_solution(self):
        """Test that interior solution beats boundaries."""
        # Use low V and high cost so optimum is interior (f(0.1)>0, f(1)<0)
        a = solve_a_system(
            V=1.0, q_bar=0.9, kappa=2.0, alpha=0.3, beta=0.2, L=1.0, a_min=0.1,
        )
        assert 0.1 < a < 1.0, f"Expected interior solution, got a={a}"
        # Verify optimality vs dense grid
        best_val = -float("inf")
        best_a_grid = 0.1
        for i in range(1001):
            ai = 0.1 + 0.9 * i / 1000
            val = 1.0 * Q(ai, 0.9, 2.0) - C(ai, 0.3, 0.2, 1.0)
            if val > best_val:
                best_val = val
                best_a_grid = ai
        assert abs(a - best_a_grid) < 1e-3, f"a_sys={a:.6f}, grid_best={best_a_grid:.6f}"

    def test_random_grid_validation(self):
        """Random 100 instances: a_system within 1e-3 of dense grid (5000 pts)."""
        rng = random.Random(42)
        max_err = 0.0
        for _ in range(100):
            V = 0.5 + rng.random() * 10.0
            q_bar = 0.85 + rng.random() * 0.15
            kappa = 1.0 + rng.random() * 4.0
            alpha = 0.05 + rng.random() * 0.2
            beta = 0.02 + rng.random() * 0.1
            L = 0.2 + rng.random() * 1.5
            a_min = max(0.01, rng.random() * 0.3)

            a_sys = solve_a_system(V, q_bar, kappa, alpha, beta, L, a_min)
            assert a_min - 1e-9 <= a_sys <= 1.0 + 1e-9

            # Dense grid (5000 points for higher precision)
            best_val = -float("inf")
            best_a = a_min
            n = 5000
            for i in range(n + 1):
                ai = a_min + (1.0 - a_min) * i / n
                val = V * Q(ai, q_bar, kappa) - C(ai, alpha, beta, L)
                if val > best_val:
                    best_val = val
                    best_a = ai
            err = abs(a_sys - best_a)
            max_err = max(max_err, err)
        assert max_err < 1e-3, f"Max grid error: {max_err:.2e}"


class TestReinforcementEffort:
    def test_s_required_formula(self):
        s = s_required(0.75, 0.08, 0.03)
        # s = 0.03*0.75 / (0.08*0.25 + 0.03*0.75) = 0.0225/0.0425 ≈ 0.529
        expected = 0.03 * 0.75 / (0.08 * 0.25 + 0.03 * 0.75)
        assert math.isclose(s, expected, rel_tol=1e-6)

    def test_s_required_at_boundary(self):
        """Theta → 1 gives s → 1."""
        s = s_required(0.999, 0.08, 0.03)
        assert s > 0.9

    def test_a_reinforcement_bounded(self):
        a = solve_a_reinforcement(0.75, 0.08, 0.03, 2.0, 0.05)
        assert 0.0 <= a <= 1.0

    def test_a_reinforcement_increases_with_theta(self):
        a1 = solve_a_reinforcement(0.6, 0.08, 0.03, 2.0, 0.05)
        a2 = solve_a_reinforcement(0.85, 0.08, 0.03, 2.0, 0.05)
        assert a2 >= a1, "Higher Theta_M should need higher effort"


class TestProviderBestResponse:
    """a_star: provider's optimal effort given contract."""

    def test_zero_incentive(self):
        """With free actions, Gamma=0 gives zero effort, not the QoS floor."""
        a = solve_a_star(Gamma=0.0, kappa=2.0, alpha=0.1, beta=0.05, L=1.0, a_min=0.2)
        assert a == 0.0

    def test_large_incentive(self):
        """Large Gamma → a_star = 1."""
        a = solve_a_star(Gamma=100.0, kappa=2.0, alpha=0.1, beta=0.05, L=1.0, a_min=0.1)
        assert math.isclose(a, 1.0, rel_tol=1e-6)

    def test_interior_solution(self):
        # Lower Gamma so optimum is interior
        a = solve_a_star(Gamma=0.5, kappa=2.0, alpha=0.1, beta=0.1, L=1.0, a_min=0.1)
        assert 0.1 < a < 1.0

    def test_a_star_vs_grid(self):
        """Random 100 instances: a_star within 1e-3 of dense grid (5000 pts)."""
        rng = random.Random(123)
        max_err = 0.0
        for _ in range(100):
            Gamma = 0.1 + rng.random() * 5.0
            kappa = 1.0 + rng.random() * 4.0
            alpha = 0.05 + rng.random() * 0.2
            beta = 0.02 + rng.random() * 0.1
            L = 0.2 + rng.random() * 1.5
            a_min = max(0.01, rng.random() * 0.3)

            a_star = solve_a_star(Gamma, kappa, alpha, beta, L, a_min)
            assert 0.0 <= a_star <= 1.0 + 1e-9

            # Dense grid (5000 points)
            best_val = -float("inf")
            best_a = 0.0
            n = 5000
            for i in range(n + 1):
                ai = i / n
                util = Gamma * g(ai, kappa) - C(ai, alpha, beta, L)
                if util > best_val:
                    best_val = util
                    best_a = ai
            err = abs(a_star - best_a)
            max_err = max(max_err, err)
        assert max_err < 1e-3, f"Max grid error: {max_err:.2e}"

    def test_strict_concavity(self):
        """Provider objective should be strictly concave in a."""
        Gamma, kappa, alpha, beta, L = 2.0, 2.0, 0.1, 0.05, 1.0
        vals = []
        for a in [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
            util = Gamma * g(a, kappa) - C(a, alpha, beta, L)
            vals.append(util)
        # Check single-peaked: rise then fall
        peak_idx = vals.index(max(vals))
        for i in range(peak_idx):
            assert vals[i] < vals[i + 1] or math.isclose(vals[i], vals[i + 1], rel_tol=1e-9)
        for i in range(peak_idx, len(vals) - 1):
            assert vals[i] > vals[i + 1] or math.isclose(vals[i], vals[i + 1], rel_tol=1e-9)


class TestComputeEffortBounds:
    def test_feasible_pair(self):
        r = compute_effort_bounds(
            L=1.0, F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
            q_min=0.7, q_bar=0.9, kappa=2.0,
            V=5.0, alpha=0.1, beta=0.05,
            Theta_M=0.75, xi=0.08, delta=0.03,
            reinforcement_margin=0.05, stage="cultivation",
        )
        assert r["feasible_physical"]
        assert r["a_min"] > 0
        assert r["a_system"] >= r["a_min"]
        assert r["a_target"] >= r["a_system"]

    def test_infeasible_deadline(self):
        r = compute_effort_bounds(
            L=10.0, F_i_t=1.0, effective_deadline=0.5, D_tr=0.6,
            q_min=0.7, q_bar=0.9, kappa=2.0,
            V=5.0, alpha=0.1, beta=0.05,
            Theta_M=0.75, xi=0.08, delta=0.03,
            reinforcement_margin=0.05, stage="cultivation",
        )
        assert not r["feasible_physical"]
        assert "deadline" in r["infeasible_reason"]

    def test_infeasible_quality(self):
        r = compute_effort_bounds(
            L=1.0, F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
            q_min=0.95, q_bar=0.9, kappa=2.0,
            V=5.0, alpha=0.1, beta=0.05,
            Theta_M=0.75, xi=0.08, delta=0.03,
            reinforcement_margin=0.05, stage="cultivation",
        )
        assert not r["feasible_physical"]
        assert "quality" in r["infeasible_reason"]

    def test_maintenance_target(self):
        """In maintenance, a_target = a_system (no reinforcement)."""
        r = compute_effort_bounds(
            L=1.0, F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
            q_min=0.7, q_bar=0.9, kappa=2.0,
            V=5.0, alpha=0.1, beta=0.05,
            Theta_M=0.75, xi=0.08, delta=0.03,
            reinforcement_margin=0.05, stage="maintenance",
        )
        assert r["feasible_physical"]
        assert math.isclose(r["a_target"], r["a_system"])
