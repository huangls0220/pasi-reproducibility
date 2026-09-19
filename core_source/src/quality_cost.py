"""Quality and cost functions (Sections 5.3–5.4).

Core mathematical primitives shared by all mechanisms.

Quality:   Q(a) = q_bar * (1 - exp(-kappa * a))
           g(a) = 1 - exp(-kappa * a)

Cost:      C(a) = c_tr + alpha * L * a + beta * L * a^2
"""

from __future__ import annotations

import math

import numpy as np

# Numerical safety
_EPS = 1e-12
_LOG_CLIP_MIN = 1e-12


# ── Quality ─────────────────────────────────────────────────────

def q_bar_normalize(q_bar: float) -> float:
    """Clip q_bar to safe range."""
    return min(max(q_bar, 0.0), 1.0)


def g(a: float, kappa: float) -> float:
    """Normalised quality shape function g(a)=1-exp(-kappa*a)."""
    if a <= 0:
        return 0.0
    return 1.0 - math.exp(-kappa * a)


def g_prime(a: float, kappa: float) -> float:
    """First derivative g'(a) = kappa * exp(-kappa*a)."""
    if a <= 0:
        return kappa  # limit as a→0+
    return kappa * math.exp(-kappa * a)


def g_second(a: float, kappa: float) -> float:
    """Second derivative g''(a) = -kappa^2 * exp(-kappa*a)."""
    if a <= 0:
        return -(kappa**2)
    return -(kappa**2) * math.exp(-kappa * a)


def Q(a: float, q_bar: float, kappa: float) -> float:
    """Execution quality Q(a) = q_bar * g(a)."""
    return q_bar * g(a, kappa)


def Q_prime(a: float, q_bar: float, kappa: float) -> float:
    """Derivative Q'(a) = q_bar * g'(a)."""
    return q_bar * g_prime(a, kappa)


# ── Minimum effort from deadline ────────────────────────────────

def a_deadline(
    L: float,
    F_i_t: float,
    effective_deadline: float,
    D_tr: float,
) -> float:
    """Minimum effort to meet deadline (Section 5.3).

    a_d = L / [F_i_t * (effective_deadline - D_tr)]
    """
    slack = effective_deadline - D_tr
    if slack <= 0:
        return float("inf")  # physical infeasible
    return L / (F_i_t * slack)


# ── Minimum effort from quality ─────────────────────────────────

def a_quality(q_min: float, q_bar: float, kappa: float) -> float:
    """Minimum effort to meet quality requirement.

    a_q = -(1/kappa) * ln(1 - q_min/q_bar)
    """
    if q_min >= q_bar:
        return float("inf")  # physical infeasible
    ratio = max(1.0 - q_min / q_bar, _LOG_CLIP_MIN)
    return -math.log(ratio) / kappa


# ── Cost ────────────────────────────────────────────────────────

def C(a: float, alpha: float, beta: float, L: float, c_tr: float = 0.0) -> float:
    """Total execution cost C(a) = c_tr + alpha*L*a + beta*L*a^2."""
    return c_tr + alpha * L * a + beta * L * a * a


def C_prime(a: float, alpha: float, beta: float, L: float) -> float:
    """Marginal cost C'(a) = alpha*L + 2*beta*L*a."""
    return alpha * L + 2.0 * beta * L * a


def C_second(alpha: float, beta: float, L: float) -> float:
    """Second derivative C''(a) = 2*beta*L (constant in a)."""
    return 2.0 * beta * L


# ── Communication & computation delay ───────────────────────────

def compute_delays(
    input_size: float,
    output_size: float,
    communication_rate: float,
    L: float,
    a: float,
    F_i_t: float,
) -> tuple[float, float]:
    """Return (D_tr, D_cmp)."""
    D_tr = (input_size + output_size) / max(communication_rate, _EPS)
    D_cmp = L / max(a * F_i_t, _EPS)
    return D_tr, D_cmp


def total_delay(
    input_size: float,
    output_size: float,
    communication_rate: float,
    L: float,
    a: float,
    F_i_t: float,
) -> float:
    """Total delay D_total = D_tr + D_cmp."""
    D_tr, D_cmp = compute_delays(input_size, output_size, communication_rate, L, a, F_i_t)
    return D_tr + D_cmp


# ── Utility helpers ─────────────────────────────────────────────

def provider_experienced_utility(
    base_payment: float,
    p: float,
    D: float,
    omega: float,
    H: float,
    a: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    c_tr: float = 0.0,
) -> float:
    """Experienced utility = b + p*D*g(a) + omega*H*g(a) - C(a)."""
    g_val = g(a, kappa)
    return base_payment + p * D * g_val + omega * H * g_val - C(a, alpha, beta, L, c_tr)
