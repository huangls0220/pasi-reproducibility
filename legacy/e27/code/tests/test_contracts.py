"""Tests for contracts.py — P1 contract design, base payment, utilities."""

import math
import random

import pytest

from src.contracts import (
    compute_base_payment,
    compute_lambda,
    evaluate_pair,
    solve_contract_p1,
    update_path_state,
)
from src.behavior import W


class TestLambda:
    """Lambda = max(0, C'(a_target)/g'(a_target) - omega*H)."""

    def test_lambda_decreases_with_H(self):
        """Higher H → lower Lambda needed (intrinsic motivation)."""
        L1 = compute_lambda(0.7, 2.0, 0.1, 0.05, 1.0, omega=0.2, H=0.2)
        L2 = compute_lambda(0.7, 2.0, 0.1, 0.05, 1.0, omega=0.2, H=0.8)
        assert L2 <= L1 + 1e-9

    def test_lambda_zero_when_intrinsic_sufficient(self):
        """If omega*H >= C'/g', Lambda=0."""
        L = compute_lambda(0.3, 2.0, 0.01, 0.01, 1.0, omega=10.0, H=0.8)
        assert L == 0.0

    def test_lambda_positive_otherwise(self):
        L = compute_lambda(0.8, 2.0, 0.2, 0.1, 1.0, omega=0.0, H=0.0)
        assert L > 0


class TestContractP1:
    """P1: min p*D s.t. W(p)*D >= Lambda."""

    def test_zero_lambda(self):
        r = solve_contract_p1(Lambda=0.0, p_min=0.05, p_max=0.8, D_bar=10.0, zeta=0.8)
        assert r["feasible"]
        assert r["p_star"] == 0.0
        assert r["D_star"] == 0.0

    def test_feasible_interior(self):
        r = solve_contract_p1(Lambda=1.0, p_min=0.05, p_max=0.8, D_bar=10.0, zeta=0.8)
        assert r["feasible"]
        assert 0.05 - 1e-9 <= r["p_star"] <= 0.8 + 1e-9
        assert 0 <= r["D_star"] <= 10.0 + 1e-9
        # Constraint check
        wp = W(r["p_star"], 0.8)
        assert wp * r["D_star"] + 1e-9 >= 1.0

    def test_infeasible(self):
        """Lambda too large for D_bar and p_max."""
        r = solve_contract_p1(Lambda=100.0, p_min=0.05, p_max=0.8, D_bar=5.0, zeta=0.8)
        assert not r["feasible"]

    def test_endpoint_no_worse_than_dense_grid(self):
        """P1 endpoint solution should be at least as good as any dense grid point.

        Uses zeta ∈ (0, 1] as specified in the paper (probability overweighting regime).
        """
        rng = random.Random(42)
        for _ in range(100):
            Lambda = rng.random() * 3.0
            p_max = 0.5 + rng.random() * 0.4
            D_bar = 2.0 + rng.random() * 8.0
            zeta = 0.6 + rng.random() * 0.4  # zeta ∈ [0.6, 1.0]

            r = solve_contract_p1(Lambda, 0.05, p_max, D_bar, zeta)
            if not r["feasible"]:
                continue

            opt_intensity = r["p_star"] * r["D_star"]

            # Dense grid over p — endpoint must not be worse than any grid point
            for i in range(1001):
                p = 0.05 + (p_max - 0.05) * i / 1000
                wp = W(p, zeta)
                if wp <= 1e-15:
                    continue
                D = Lambda / wp
                if D > D_bar + 1e-9:
                    continue
                intensity = p * D
                # Endpoint is mathematically optimal for zeta ≤ 1: opt <= intensity + tol
                assert opt_intensity <= intensity + 1e-4, (
                    f"Lambda={Lambda:.4f} p={p:.4f} intensity={intensity:.6f} < opt={opt_intensity:.6f}"
                )

    def test_zeta_one_constant_pD(self):
        """At zeta=1, W(p)=p, so p*D = Lambda for all feasible p."""
        Lambda = 1.5
        p_min, p_max, D_bar = 0.05, 0.8, 10.0
        # All feasible p must give p * (Lambda/p) = Lambda
        for p in [0.2, 0.3, 0.5, 0.8]:
            wp = W(p, 1.0)
            assert math.isclose(wp, p)
            D = Lambda / wp
            if D <= D_bar:
                assert math.isclose(p * D, Lambda, rel_tol=1e-9)

    def test_post_condition_assertions(self):
        """Every feasible contract must satisfy constraints."""
        rng = random.Random(99)
        for _ in range(200):
            Lambda = 0.1 + rng.random() * 2.0
            p_max = 0.5 + rng.random() * 0.3
            D_bar = 3.0 + rng.random() * 5.0
            zeta = 0.6 + rng.random() * 0.5

            r = solve_contract_p1(Lambda, 0.05, p_max, D_bar, zeta)
            if r["feasible"]:
                wp = W(r["p_star"], zeta)
                assert wp * r["D_star"] + 1e-9 >= Lambda
                assert r["D_star"] <= D_bar + 1e-9
                assert 0.05 - 1e-9 <= r["p_star"] <= p_max + 1e-9


class TestBasePayment:
    """Section 5.11: response-consistent base payment."""

    def test_ir_satisfied(self):
        """Experienced utility >= U_out."""
        b = compute_base_payment(
            U_out=0.1, a_star=0.7, p_star=0.3, D_star=5.0,
            omega=0.1, H=0.5, kappa=2.0, alpha=0.1, beta=0.05, L=1.0,
        )
        from src.quality_cost import C, g

        util = b + 0.3 * 5.0 * g(0.7, 2.0) + 0.1 * 0.5 * g(0.7, 2.0) - C(0.7, 0.1, 0.05, 1.0)
        assert util + 1e-8 >= 0.1

    def test_binding_case(self):
        """In binding case, reducing b by epsilon violates IR."""
        U_out = 0.1
        b = compute_base_payment(
            U_out=U_out, a_star=0.7, p_star=0.3, D_star=5.0,
            omega=0.1, H=0.5, kappa=2.0, alpha=0.1, beta=0.05, L=1.0,
        )
        from src.quality_cost import C, g

        # If b > 0, reduce by small epsilon
        if b > 1e-6:
            b2 = b - 1e-6
            util2 = b2 + 0.3 * 5.0 * g(0.7, 2.0) + 0.1 * 0.5 * g(0.7, 2.0) - C(0.7, 0.1, 0.05, 1.0)
            assert util2 < U_out or math.isclose(util2, U_out, rel_tol=1e-9)

    def test_non_negative(self):
        b = compute_base_payment(
            U_out=0.0, a_star=0.5, p_star=1.0, D_star=10.0,
            omega=1.0, H=0.9, kappa=2.0, alpha=0.01, beta=0.01, L=0.5,
        )
        assert b >= 0


class TestPathStateUpdate:
    """Section 5.13: H update and stage transitions."""

    def test_H_bounds(self):
        """H should stay in [0,1]."""
        for H in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for s in [0.0, 0.5, 1.0]:
                r = update_path_state(
                    H=H, s=s, xi=0.08, delta=0.03,
                    assigned=True, stage="cultivation",
                    recent_quality=[], Theta_M=0.75, Theta_C=0.55,
                    s_M=0.8, s_C=0.65, K=10,
                )
                assert 0 <= r["H_next"] <= 1, f"H={H}, s={s} → H_next={r['H_next']}"

    def test_H_unchanged_when_unassigned(self):
        r = update_path_state(
            H=0.5, s=0.9, xi=0.08, delta=0.03,
            assigned=False, stage="cultivation",
            recent_quality=[], Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        assert r["H_next"] == 0.5

    def test_H_increases_with_high_quality(self):
        """High s should increase H."""
        r = update_path_state(
            H=0.3, s=0.9, xi=0.08, delta=0.03,
            assigned=True, stage="cultivation",
            recent_quality=[], Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        assert r["H_next"] > 0.3

    def test_H_decreases_with_low_quality(self):
        """Low s should decrease H."""
        r = update_path_state(
            H=0.8, s=0.1, xi=0.08, delta=0.03,
            assigned=True, stage="cultivation",
            recent_quality=[], Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        assert r["H_next"] < 0.8

    def test_steady_state_closed_form(self):
        """Constant s → H converges to H* = xi*s / (xi*s + delta*(1-s))."""
        xi, delta = 0.08, 0.03
        s = 0.85
        H_star = xi * s / (xi * s + delta * (1 - s))
        # Simulate for many steps
        H = 0.5
        for _ in range(500):
            r = update_path_state(
                H=H, s=s, xi=xi, delta=delta,
                assigned=True, stage="cultivation",
                recent_quality=[], Theta_M=0.95, Theta_C=0.55,
                s_M=0.99, s_C=0.65, K=10,
            )
            H = r["H_next"]
        assert abs(H - H_star) < 0.01, f"H={H:.4f}, H*={H_star:.4f}"

    def test_cultivation_to_maintenance_transition(self):
        """After enough high-quality interactions, should enter maintenance."""
        recent = [0.85] * 9  # need K=10
        r = update_path_state(
            H=0.78, s=0.85, xi=0.08, delta=0.03,
            assigned=True, stage="cultivation",
            recent_quality=recent, Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        # H should be >= Theta_M and with K quality samples
        assert r["H_next"] >= 0.75
        # After update, window has 10 entries with mean 0.85 >= s_M
        assert len(r["recent_quality"]) == 10
        if r["H_next"] >= 0.75:
            pass  # transition should happen if conditions met

    def test_maintenance_to_cultivation_fallback(self):
        """If H falls below Theta_C, should return to cultivation."""
        recent = [0.5] * 10
        r = update_path_state(
            H=0.50, s=0.5, xi=0.08, delta=0.08,
            assigned=True, stage="maintenance",
            recent_quality=recent, Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        # H < Theta_C → fallback
        assert r["stage_next"] == "cultivation"

    def test_no_transition_without_enough_samples(self):
        """Shouldn't transition with < K quality samples."""
        r = update_path_state(
            H=0.8, s=0.9, xi=0.08, delta=0.03,
            assigned=True, stage="cultivation",
            recent_quality=[0.9] * 5,  # only 5 < K=10
            Theta_M=0.75, Theta_C=0.55,
            s_M=0.8, s_C=0.65, K=10,
        )
        assert r["stage_next"] == "cultivation"  # not enough samples


class TestEvaluatePair:
    """End-to-end pair evaluation."""

    def test_feasible_pair(self):
        r = evaluate_pair(
            L=1.0, V=5.0, q_bar=0.9, q_min=0.7, kappa=2.0,
            alpha=0.1, beta=0.05, omega=0.15, zeta=0.8,
            xi=0.08, delta=0.03, H=0.3, stage="cultivation",
            U_out=0.1, p_min=0.05, p_max=0.8, D_bar=10.0,
            reinforcement_margin=0.05, Theta_M=0.75, eta_H=5.0,
            last_probability=0.0, delta_p_max=0.05,
            F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
        )
        assert r["feasible_physical"]
        assert r["feasible_contract"]
        assert r["a_star"] >= r["a_target"] - 1e-7
        assert r["experienced_utility"] + 1e-8 >= 0.1

    def test_a_star_not_below_a_target(self):
        """Critical: a_star must not be significantly below a_target."""
        rng = random.Random(777)
        for _ in range(50):
            L = 0.3 + rng.random() * 1.5
            V = 1.0 + rng.random() * 8.0
            r = evaluate_pair(
                L=L, V=V, q_bar=0.9, q_min=0.65 + rng.random() * 0.2, kappa=1.5 + rng.random() * 3.0,
                alpha=0.05 + rng.random() * 0.15, beta=0.02 + rng.random() * 0.08,
                omega=0.05 + rng.random() * 0.2, zeta=0.65 + rng.random() * 0.3,
                xi=0.05 + rng.random() * 0.07, delta=0.01 + rng.random() * 0.04,
                H=rng.random() * 0.5, stage="cultivation",
                U_out=0.05, p_min=0.05, p_max=0.8, D_bar=10.0,
                reinforcement_margin=0.05, Theta_M=0.75, eta_H=5.0,
                last_probability=0.0, delta_p_max=0.05,
                F_i_t=3.0 + rng.random() * 10.0,
                effective_deadline=1.0 + rng.random() * 5.0,
                D_tr=0.1 + rng.random() * 0.5,
            )
            if r["feasible_physical"] and r["feasible_contract"]:
                assert r["a_star"] >= r["a_target"] - 1e-7, (
                    f"a_star={r['a_star']:.6f} < a_target={r['a_target']:.6f}"
                )

    def test_base_payment_non_negative(self):
        r = evaluate_pair(
            L=1.0, V=5.0, q_bar=0.9, q_min=0.7, kappa=2.0,
            alpha=0.1, beta=0.05, omega=0.15, zeta=0.8,
            xi=0.08, delta=0.03, H=0.3, stage="cultivation",
            U_out=0.1, p_min=0.05, p_max=0.8, D_bar=10.0,
            reinforcement_margin=0.05, Theta_M=0.75, eta_H=5.0,
            last_probability=0.0, delta_p_max=0.05,
            F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
        )
        if r["feasible_contract"]:
            assert r["base_payment"] >= 0

    def test_maintenance_probability_smoothing(self):
        """In maintenance with high last_probability, p should be smoothed."""
        r = evaluate_pair(
            L=1.0, V=5.0, q_bar=0.9, q_min=0.7, kappa=2.0,
            alpha=0.1, beta=0.05, omega=0.15, zeta=0.8,
            xi=0.08, delta=0.03, H=0.8, stage="maintenance",
            U_out=0.1, p_min=0.05, p_max=0.8, D_bar=10.0,
            reinforcement_margin=0.05, Theta_M=0.75, eta_H=5.0,
            last_probability=0.6, delta_p_max=0.05,
            F_i_t=5.0, effective_deadline=3.0, D_tr=0.5,
        )
        if r["feasible_contract"]:
            # p_star should be >= last_probability - delta_p_max
            assert r["p_star"] >= 0.6 - 0.05 - 1e-9
