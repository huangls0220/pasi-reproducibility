"""Effort optimisation (Sections 5.5–5.6, 5.10).

Solves for:
  a_system       – system-optimal effort (max V*Q - C)
  a_reinforcement – effort needed for cultivation quality target
  a_star         – provider best-response effort given contract
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.optimize import brentq

from .quality_cost import (
    C,
    C_prime,
    C_second,
    Q,
    Q_prime,
    g,
    g_prime,
)

_EPS = 1e-12
_ROOT_TOL = 1e-10


# ── System-optimal effort a_system (Section 5.5) ────────────────

def solve_a_system(
    V: float,
    q_bar: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    a_min: float,
    c_tr: float = 0.0,
) -> float:
    """Find a ∈ [a_min, 1] that maximises Φ(a) = V*Q(a) - C(a).

    Solves f_system(a) = V*Q'(a) - C'(a) = 0 via brentq when interior solution exists.
    """
    a_min = max(0.0, min(a_min, 1.0))

    def f(a: float) -> float:
        return V * Q_prime(a, q_bar, kappa) - C_prime(a, alpha, beta, L)

    f_lo = f(a_min)
    f_hi = f(1.0)

    if f_lo <= 0:
        candidate = a_min
    elif f_hi >= 0:
        candidate = 1.0
    else:
        try:
            candidate = brentq(f, a_min, 1.0, xtol=_ROOT_TOL)
        except ValueError:
            # Fall back to boundary comparison
            phi_lo = V * Q(a_min, q_bar, kappa) - C(a_min, alpha, beta, L, c_tr)
            phi_hi = V * Q(1.0, q_bar, kappa) - C(1.0, alpha, beta, L, c_tr)
            candidate = a_min if phi_lo >= phi_hi else 1.0

    # Verify candidate is not worse than boundaries
    phi_c = V * Q(candidate, q_bar, kappa) - C(candidate, alpha, beta, L, c_tr)
    phi_lo = V * Q(a_min, q_bar, kappa) - C(a_min, alpha, beta, L, c_tr)
    phi_hi = V * Q(1.0, q_bar, kappa) - C(1.0, alpha, beta, L, c_tr)
    if phi_c + _ROOT_TOL < max(phi_lo, phi_hi):
        # Numerically suspicious — fall back to best boundary
        candidate = a_min if phi_lo >= phi_hi else 1.0

    return float(max(a_min, min(1.0, candidate)))


# ── Cultivation reinforcement effort (Section 5.6) ──────────────

def s_required(Theta: float, xi: float, delta: float) -> float:
    """Quality level s needed to asymptotically reach H >= Theta.

    s_required = delta*Theta / [xi*(1-Theta) + delta*Theta]
    """
    denom = xi * (1.0 - Theta) + delta * Theta
    if denom <= 0:
        return 1.0
    return min(1.0, delta * Theta / denom)


def solve_a_reinforcement(
    Theta_M: float,
    xi: float,
    delta: float,
    kappa: float,
    reinforcement_margin: float = 0.05,
) -> float:
    """Effort level that produces cultivation quality target.

    s_rein = min(1-ε, s_required(Theta_M) + margin)
    a_rein = -(1/kappa) * ln(1 - s_rein)
    """
    s_req = s_required(Theta_M, xi, delta)
    s_rein = min(1.0 - _EPS, s_req + reinforcement_margin)
    s_rein = max(0.0, s_rein)
    if s_rein >= 1.0:
        return 1.0
    ratio = max(1.0 - s_rein, _EPS)
    return float(-math.log(ratio) / kappa)


# ── Provider best-response effort a_star (Section 5.10) ─────────

def solve_a_star(
    Gamma: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    a_min: float,
) -> float:
    """Provider best-response to total incentive intensity Gamma.

    Maximises:  base + Gamma*g(a) - C(a)
    FOC:        f_provider(a) = Gamma*g'(a) - C'(a) = 0

    Gamma = W(p)*D + omega*H  (effective incentive intensity)
    """
    a_min = max(0.0, min(a_min, 1.0))

    # g'(a) = kappa*exp(-kappa*a), C'(a) = alpha*L + 2*beta*L*a
    def f(a: float) -> float:
        return Gamma * g_prime(a, kappa) - C_prime(a, alpha, beta, L)

    f_lo = f(a_min)
    f_hi = f(1.0)

    if f_lo <= 0:
        candidate = a_min
    elif f_hi >= 0:
        candidate = 1.0
    else:
        try:
            candidate = brentq(f, a_min, 1.0, xtol=_ROOT_TOL)
        except ValueError:
            candidate = _grid_search_a_star(Gamma, kappa, alpha, beta, L, a_min)

    return float(max(a_min, min(1.0, candidate)))


def _grid_search_a_star(
    Gamma: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    a_min: float,
    n: int = 10000,
) -> float:
    """Grid-search fallback for a_star."""
    from .quality_cost import C

    best_a = a_min
    best_val = -float("inf")
    for i in range(n + 1):
        a = a_min + (1.0 - a_min) * i / n
        util = Gamma * g(a, kappa) - C(a, alpha, beta, L)
        if util > best_val:
            best_val = util
            best_a = a
    return best_a


# ── Minimum / target effort calculation ─────────────────────────

def compute_effort_bounds(
    L: float,
    F_i_t: float,
    effective_deadline: float,
    D_tr: float,
    q_min: float,
    q_bar: float,
    kappa: float,
    V: float,
    alpha: float,
    beta: float,
    Theta_M: float,
    xi: float,
    delta: float,
    reinforcement_margin: float,
    stage: str,
) -> dict:
    """Compute all effort levels for a given provider–task pair.

    Returns dict with keys:
      feasible_physical, infeasible_reason,
      a_deadline, a_quality, a_min, a_system, a_reinforcement, a_target
    """
    from .quality_cost import a_deadline as _a_d, a_quality as _a_q

    result = {
        "feasible_physical": True,
        "infeasible_reason": None,
        "a_deadline": 0.0,
        "a_quality": 0.0,
        "a_min": 0.0,
        "a_system": 0.0,
        "a_reinforcement": 0.0,
        "a_target": 0.0,
    }

    # 1. Deadline effort
    a_d = _a_d(L, F_i_t, effective_deadline, D_tr)
    if a_d == float("inf"):
        result["feasible_physical"] = False
        result["infeasible_reason"] = "deadline_infeasible"
        result["a_deadline"] = float("nan")
        return result
    result["a_deadline"] = a_d

    # 2. Quality effort
    a_q = _a_q(q_min, q_bar, kappa)
    if a_q == float("inf"):
        result["feasible_physical"] = False
        result["infeasible_reason"] = "quality_infeasible"
        result["a_quality"] = float("nan")
        return result
    result["a_quality"] = a_q

    # 3. Combined minimum
    a_min = max(a_d, a_q)
    if a_min > 1.0:
        result["feasible_physical"] = False
        result["infeasible_reason"] = f"a_min={a_min:.4f}>1"
        result["a_min"] = a_min
        return result
    result["a_min"] = a_min

    # 4. System-optimal
    a_sys = solve_a_system(V, q_bar, kappa, alpha, beta, L, a_min)
    result["a_system"] = a_sys

    # 5. Reinforcement effort
    a_rein = solve_a_reinforcement(Theta_M, xi, delta, kappa, reinforcement_margin)
    result["a_reinforcement"] = a_rein

    # 6. Target effort
    if stage == "cultivation":
        a_target = max(a_sys, a_rein)
    else:
        a_target = a_sys
    result["a_target"] = max(a_min, min(1.0, a_target))

    return result
