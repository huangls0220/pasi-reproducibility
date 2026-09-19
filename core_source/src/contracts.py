"""Contract design P1 (Sections 5.8–5.9, 5.11–5.12).

Transparent bonus contract:
  min  p * D            (expected incentive intensity)
  s.t. W(p) * D >= Lambda
       p_min <= p <= p_max,  0 <= D <= D_bar

Plus response-consistent base payment (Section 5.11).
"""

from __future__ import annotations

import math
from typing import Optional

from .behavior import W, W_inv
from .quality_cost import C, g, g_prime
from .effort_solver import solve_a_star

_EPS = 1e-12


# ── Minimum incentive intensity Lambda (Section 5.8) ────────────

def compute_lambda(
    a_target: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    omega: float,
    H: float,
) -> float:
    """Lambda = max(0, C'(a_target)/g'(a_target) - omega*H).

    This is the minimum external incentive needed to induce a_target.
    """
    if a_target <= 0:
        return 0.0
    c_prime = alpha * L + 2.0 * beta * L * a_target
    gp = g_prime(a_target, kappa)
    if gp <= _EPS:
        return float("inf")  # avoid division by zero
    intrinsic = omega * H
    return max(0.0, c_prime / gp - intrinsic)


# ── P1: Transparent bonus contract ──────────────────────────────

def solve_contract_p1(
    Lambda: float,
    p_min: float,
    p_max: float,
    D_bar: float,
    zeta: float,
) -> dict:
    """Solve the transparent-probability bonus contract (P1).

    Returns dict:
      feasible: bool
      p_star, D_star: optimal contract parameters
      candidates: list of (p, D) candidates considered
      infeasible_reason: str or None
    """
    result = {
        "feasible": True,
        "p_star": 0.0,
        "D_star": 0.0,
        "candidates": [],
        "infeasible_reason": None,
    }

    # Zero Lambda → no incentive needed
    if Lambda <= 0:
        result["p_star"] = 0.0
        result["D_star"] = 0.0
        return result

    # Check if contract is possible at all
    w_max = W(p_max, zeta)
    if Lambda > D_bar * w_max:
        result["feasible"] = False
        result["infeasible_reason"] = (
            f"Lambda={Lambda:.6f} > D_bar*W(p_max)={D_bar*w_max:.6f}"
        )
        return result

    # p_L: lowest probability that can satisfy constraint at D=D_bar
    w_needed = Lambda / D_bar
    w_needed = min(w_needed, 1.0 - _EPS)
    try:
        p_L = W_inv(w_needed, zeta)
    except (ValueError, OverflowError):
        result["feasible"] = False
        result["infeasible_reason"] = "W_inv failed"
        return result
    p_L = max(p_min, p_L)

    # For zeta>1, log(p/W(p)) is convex in -log(p), with one
    # possible interior minimum. For zeta<=1 the endpoints suffice.
    candidates_p = sorted(set([p_L, p_max]))
    if zeta > 1.0:
        stationary = math.exp(-math.exp(-math.log(zeta)/(zeta-1.0)))
        candidates_p.append(min(p_max, max(p_L, stationary)))

    best_p = p_max
    best_D = Lambda / max(W(p_max, zeta), _EPS)
    best_intensity = best_p * best_D

    for p in candidates_p:
        wp = W(p, zeta)
        if wp <= _EPS:
            continue
        D = Lambda / wp
        if D > D_bar + 1e-9:
            continue
        expected_intensity = p * D
        result["candidates"].append((p, D))
        if expected_intensity < best_intensity - 1e-12:
            best_intensity = expected_intensity
            best_p = p
            best_D = D

    result["p_star"] = float(best_p)
    result["D_star"] = float(min(best_D, D_bar))

    # Post-condition assertions
    wp_check = W(result["p_star"], zeta)
    if wp_check * result["D_star"] + 1e-9 < Lambda:
        result["feasible"] = False
        result["infeasible_reason"] = "post-condition W(p)D >= Lambda failed"
        return result
    if result["D_star"] > D_bar + 1e-9:
        result["feasible"] = False
        result["infeasible_reason"] = "post-condition D <= D_bar failed"
        return result
    if not (p_min - 1e-9 <= result["p_star"] <= p_max + 1e-9):
        result["feasible"] = False
        result["infeasible_reason"] = "post-condition p bounds failed"
        return result

    return result


# ── Response-consistent base payment (Section 5.11) ─────────────

def compute_base_payment(
    U_out: float,
    a_star: float,
    p_star: float,
    D_star: float,
    omega: float,
    H: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
) -> float:
    """Response-consistent base payment.

    b_RC = max(0, U_out + C(a_star) - p*D*g(a_star) - omega*H*g(a_star))
    """
    cost = C(a_star, alpha, beta, L)
    g_val = g(a_star, kappa)
    incentive_term = (p_star * D_star + omega * H) * g_val
    b = U_out + cost - incentive_term
    return max(0.0, b)


def compute_utilities(
    base_payment: float,
    p_star: float,
    D_star: float,
    omega: float,
    H: float,
    a_star: float,
    kappa: float,
    alpha: float,
    beta: float,
    L: float,
    U_out: float,
    zeta: float,
) -> dict:
    """Compute experienced and perceived utilities for a pair.

    Returns dict with experienced_utility, perceived_utility, ir_satisfied.
    """
    cost = C(a_star, alpha, beta, L)
    g_val = g(a_star, kappa)

    experienced = base_payment + p_star * D_star * g_val + omega * H * g_val - cost
    perceived = base_payment + W(p_star, zeta) * D_star * g_val + omega * H * g_val - cost

    return {
        "experienced_utility": experienced,
        "perceived_utility": perceived,
        "ir_satisfied": experienced + 1e-8 >= U_out,
    }


# ── Full pair evaluation ────────────────────────────────────────

def evaluate_pair(
    L: float,
    V: float,
    q_bar: float,
    q_min: float,
    kappa: float,
    alpha: float,
    beta: float,
    omega: float,
    zeta: float,
    xi: float,
    delta: float,
    H: float,
    stage: str,
    U_out: float,
    p_min: float,
    p_max: float,
    D_bar: float,
    reinforcement_margin: float,
    Theta_M: float,
    eta_H: float,
    last_probability: float,
    delta_p_max: float,
    F_i_t: float,
    effective_deadline: float,
    D_tr: float,
    input_size: float = 0.0,
    output_size: float = 0.0,
    communication_rate: float = 1.0,
) -> dict:
    """Full evaluation of a provider–task pair under PRIME.

    This is the main orchestrator that chains all the mathematical
    components together in the correct order.  Returns a dict that
    maps 1:1 to PairResult fields.

    IMPORTANT: a_target is only used for contract design (Lambda);
    all utility / quality / cost computations use a_star.  See
    Section 2 rule 1 and Sections 5.10–5.12.
    """
    from .effort_solver import compute_effort_bounds
    from .quality_cost import Q, total_delay as _total_delay

    result = {
        "feasible_physical": True,
        "feasible_contract": True,
        "infeasible_reason": None,
        "a_deadline": 0.0,
        "a_quality": 0.0,
        "a_min": 0.0,
        "a_system": 0.0,
        "a_reinforcement": 0.0,
        "a_target": 0.0,
        "lambda_required": 0.0,
        "p_star": 0.0,
        "D_star": 0.0,
        "gamma_effective": 0.0,
        "a_star": 0.0,
        "implementation_gap": 0.0,
        "base_payment": 0.0,
        "expected_bonus": 0.0,
        "expected_contract_cost": 0.0,
        "experienced_utility": 0.0,
        "perceived_utility": 0.0,
        "execution_quality": 0.0,
        "normalized_quality": 0.0,
        "total_delay": 0.0,
        "delta_H_predicted": 0.0,
        "immediate_value": 0.0,
        "reinforcement_value": 0.0,
        "total_pair_value": 0.0,
    }

    # ── Step 1: Physical feasibility & effort bounds ─────────
    bounds = compute_effort_bounds(
        L, F_i_t, effective_deadline, D_tr,
        q_min, q_bar, kappa, V, alpha, beta,
        Theta_M, xi, delta, reinforcement_margin, stage,
    )
    if not bounds["feasible_physical"]:
        result["feasible_physical"] = False
        result["infeasible_reason"] = bounds["infeasible_reason"]
        return result
    for k in ("a_deadline", "a_quality", "a_min", "a_system", "a_reinforcement", "a_target"):
        result[k] = bounds[k]

    a_target = bounds["a_target"]
    a_min = bounds["a_min"]

    # ── Step 2: Lambda ───────────────────────────────────────
    Lambda = compute_lambda(a_target, kappa, alpha, beta, L, omega, H)
    result["lambda_required"] = Lambda

    # ── Step 3: P1 contract ──────────────────────────────────
    contract = solve_contract_p1(Lambda, p_min, p_max, D_bar, zeta)
    if not contract["feasible"]:
        result["feasible_contract"] = False
        result["infeasible_reason"] = contract.get("infeasible_reason", "P1_infeasible")
        return result
    p_star = contract["p_star"]
    D_star = contract["D_star"]
    result["p_star"] = p_star
    result["D_star"] = D_star

    # ── Step 3b: Maintenance probability smoothing (Section 5.13) ─
    if stage == "maintenance" and last_probability > 0:
        if p_star < last_probability:
            p_star = max(p_star, last_probability - delta_p_max)
            # Recompute D
            wp = W(p_star, zeta)
            if wp > _EPS:
                D_star = min(Lambda / wp, D_bar)
            else:
                result["feasible_contract"] = False
                result["infeasible_reason"] = "maintenance_smoothing_Wp_zero"
                return result
            if D_star > D_bar:
                result["feasible_contract"] = False
                result["infeasible_reason"] = "maintenance_smoothing_D_exceeds_bar"
                return result
            result["p_star"] = p_star
            result["D_star"] = D_star

    # ── Step 4: Gamma & a_star ───────────────────────────────
    Gamma = W(p_star, zeta) * D_star + omega * H
    result["gamma_effective"] = Gamma
    a_star = solve_a_star(Gamma, kappa, alpha, beta, L, a_min)
    result["a_star"] = a_star
    implementation_gap = a_star - a_target
    result["implementation_gap"] = implementation_gap

    # Critical check: a_star must NOT be significantly below a_target
    if implementation_gap < -1e-7:
        result["feasible_contract"] = False
        result["infeasible_reason"] = (
            f"a_star={a_star:.8f} < a_target={a_target:.8f} (gap={implementation_gap:.2e})"
        )
        return result

    # ── Step 5: Base payment ─────────────────────────────────
    b_rc = compute_base_payment(U_out, a_star, p_star, D_star, omega, H, kappa, alpha, beta, L)
    result["base_payment"] = b_rc

    # ── Step 6: Utilities ────────────────────────────────────
    utils = compute_utilities(b_rc, p_star, D_star, omega, H, a_star, kappa, alpha, beta, L, U_out, zeta)
    result["experienced_utility"] = utils["experienced_utility"]
    result["perceived_utility"] = utils["perceived_utility"]

    # ── Step 7: Quality & delay ──────────────────────────────
    result["execution_quality"] = Q(a_star, q_bar, kappa)
    result["normalized_quality"] = g(a_star, kappa)
    result["total_delay"] = _total_delay(
        input_size, output_size, communication_rate, L, a_star, F_i_t,
    )

    # ── Step 8: Platform costs & value ───────────────────────
    expected_bonus = p_star * D_star * g(a_star, kappa)
    expected_contract_cost = b_rc + expected_bonus
    result["expected_bonus"] = expected_bonus
    result["expected_contract_cost"] = expected_contract_cost
    immediate_value = V * Q(a_star, q_bar, kappa) - expected_contract_cost
    result["immediate_value"] = immediate_value

    # ── Step 9: Reinforcement value (Section 5.14) ───────────
    s_val = g(a_star, kappa)
    delta_H_pred = xi * s_val * (1.0 - H) - delta * (1.0 - s_val) * H
    result["delta_H_predicted"] = delta_H_pred

    if stage == "cultivation":
        reinforcement_value = eta_H * max(delta_H_pred, 0.0)
    else:
        reinforcement_value = 0.0
    result["reinforcement_value"] = reinforcement_value
    result["total_pair_value"] = immediate_value + reinforcement_value

    return result


# ── Path-state update (Section 5.13) ────────────────────────────

def update_path_state(
    H: float,
    s: float,
    xi: float,
    delta: float,
    assigned: bool,
    stage: str,
    recent_quality: list[float],
    Theta_M: float,
    Theta_C: float,
    s_M: float,
    s_C: float,
    K: int,
) -> dict:
    """Update path state H and stage after a slot.

    If assigned:  H_next = H + xi*s*(1-H) - delta*(1-s)*H
    If unassigned: H_next = H  (no change in quality signal)

    Returns dict with H_next, stage_next, recent_quality, stage_changed, events.
    """
    import statistics

    events = []

    # Update H
    if assigned:
        H_next = H + xi * s * (1.0 - H) - delta * (1.0 - s) * H
    else:
        H_next = H

    # Clip for floating-point protection
    if H_next < -1e-8 or H_next > 1.0 + 1e-8:
        raise ValueError(f"H_next={H_next} out of bounds [0,1]")
    H_next = max(0.0, min(1.0, H_next))

    # Update quality window (only when assigned)
    new_recent = list(recent_quality)
    if assigned:
        new_recent.append(s)
        if len(new_recent) > K * 2:  # keep bounded
            new_recent = new_recent[-K * 2:]

    # Stage transitions
    stage_next = stage
    if len(new_recent) >= K:
        window = new_recent[-K:]
        mean_q = statistics.mean(window)

        if stage == "cultivation":
            if H_next >= Theta_M and mean_q >= s_M:
                stage_next = "maintenance"
                events.append("cultivation→maintenance")
        elif stage == "maintenance":
            if H_next < Theta_C or mean_q < s_C:
                stage_next = "cultivation"
                events.append("maintenance→cultivation")

    return {
        "H_next": H_next,
        "stage_next": stage_next,
        "recent_quality": new_recent,
        "stage_changed": stage_next != stage,
        "events": events,
    }
