"""Tests for quality_cost.py — quality function, cost function, delay calculations."""

import math

import pytest

from src.quality_cost import (
    C,
    C_prime,
    C_second,
    Q,
    Q_prime,
    a_deadline,
    a_quality,
    g,
    g_prime,
    g_second,
    provider_experienced_utility,
    total_delay,
)


class TestQualityFunction:
    """Section 5.3: Q(a) = q_bar * (1 - exp(-kappa*a))."""

    def test_Q_zero_effort(self):
        assert Q(0.0, 0.9, 2.0) == 0.0

    def test_Q_monotonic(self):
        """Q must be strictly increasing in a."""
        vals = [Q(a, 0.9, 2.0) for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], f"Q not monotonic at i={i}"

    def test_Q_concave(self):
        """Q'' < 0 (g'' < 0)."""
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert g_second(a, 2.0) < 0, f"g''({a}) >= 0"

    def test_Q_bounded_by_q_bar(self):
        for a in [0.0, 0.5, 1.0, 2.0]:
            assert Q(a, 0.9, 2.0) <= 0.9

    def test_g_identity_at_infinity(self):
        """g(a) → 1 as a → ∞."""
        assert math.isclose(g(1.0, 5.0), 0.9932, rel_tol=0.01)
        assert math.isclose(g(5.0, 1.0), 0.9932, rel_tol=0.01)

    def test_g_prime_positive(self):
        for a in [0.0, 0.2, 0.5, 0.8, 1.0]:
            assert g_prime(a, 2.0) > 0, f"g'({a}) <= 0"

    def test_Q_prime_formula(self):
        """Q'(a) = q_bar * g'(a)."""
        for a in [0.1, 0.5, 0.9]:
            assert math.isclose(Q_prime(a, 0.9, 2.0), 0.9 * g_prime(a, 2.0))


class TestCostFunction:
    """Section 5.4: C(a) = c_tr + alpha*L*a + beta*L*a^2."""

    def test_C_zero_effort(self):
        assert C(0.0, 0.1, 0.05, 1.0) == 0.0  # c_tr=0 by default

    def test_C_monotonic(self):
        """C must be strictly increasing in a."""
        vals = [C(a, 0.1, 0.05, 1.0) for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], f"C not monotonic at i={i}"

    def test_C_convex(self):
        """C'' = 2*beta*L > 0."""
        assert C_second(0.1, 0.05, 1.0) > 0

    def test_C_prime_linear(self):
        """C'(a) = alpha*L + 2*beta*L*a."""
        alpha, beta, L = 0.1, 0.05, 2.0
        for a in [0.0, 0.5, 1.0]:
            expected = alpha * L + 2.0 * beta * L * a
            assert math.isclose(C_prime(a, alpha, beta, L), expected)

    def test_C_with_c_tr(self):
        assert C(0.0, 0.1, 0.05, 1.0, c_tr=0.5) == 0.5


class TestDeadlineEffort:
    """a_d = L / [F * (deadline - D_tr)]."""

    def test_normal_case(self):
        a = a_deadline(L=10.0, F_i_t=5.0, effective_deadline=4.0, D_tr=1.0)
        # a = 10/(5*3) = 0.6667
        assert math.isclose(a, 2.0 / 3.0, rel_tol=1e-6)

    def test_infeasible_zero_slack(self):
        a = a_deadline(L=10.0, F_i_t=5.0, effective_deadline=1.0, D_tr=1.0)
        assert a == float("inf")

    def test_infeasible_negative_slack(self):
        a = a_deadline(L=10.0, F_i_t=5.0, effective_deadline=0.5, D_tr=1.0)
        assert a == float("inf")


class TestQualityEffort:
    """a_q = -(1/kappa)*ln(1 - q_min/q_bar)."""

    def test_normal_case(self):
        a = a_quality(q_min=0.7, q_bar=0.9, kappa=2.0)
        # ratio = 1 - 0.7/0.9 = 0.2222, -ln(0.2222)/2 ≈ 0.752
        assert 0.7 < a < 0.8

    def test_q_min_zero(self):
        a = a_quality(q_min=0.0, q_bar=0.9, kappa=2.0)
        assert a == 0.0

    def test_infeasible_q_min_ge_q_bar(self):
        a = a_quality(q_min=0.9, q_bar=0.9, kappa=2.0)
        assert a == float("inf")

    def test_infeasible_q_min_gt_q_bar(self):
        a = a_quality(q_min=0.95, q_bar=0.9, kappa=2.0)
        assert a == float("inf")


class TestDelay:
    def test_total_delay_positive(self):
        d = total_delay(1.0, 0.5, 10.0, 5.0, 0.8, 8.0)
        assert d > 0

    def test_total_delay_decreases_with_effort(self):
        d1 = total_delay(1.0, 0.5, 10.0, 5.0, 0.5, 8.0)
        d2 = total_delay(1.0, 0.5, 10.0, 5.0, 0.9, 8.0)
        assert d2 < d1, "higher effort should reduce computation delay"


class TestProviderUtility:
    def test_experienced_utility_binding_ir(self):
        """With exact base payment, experienced utility should meet U_out."""
        util = provider_experienced_utility(
            base_payment=0.5, p=0.3, D=5.0, omega=0.1, H=0.5,
            a=0.7, kappa=2.0, alpha=0.1, beta=0.05, L=1.0,
        )
        # Just a smoke test — exact IR test is in test_base_payment.py
        assert isinstance(util, float)
        assert not math.isnan(util)
