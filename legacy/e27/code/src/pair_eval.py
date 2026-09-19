"""Vectorised provider–task pair evaluation.

Evaluates ALL pairs of one slot simultaneously with NumPy.  The scalar
reference implementation lives in src/contracts.py / src/effort_solver.py;
tests/test_vectorized.py asserts both agree to <1e-6 on random instances,
so the auditable scalar formulas remain the ground truth (instruction §B.3).

Two parameter "sides" (see src/mechanisms.py):
  *_t  – true behavioural parameters (drive a_star, experienced utility,
         true path-state update).  Identical across all methods per §2.5.
  *_d  – design-side copies the PLATFORM uses (Lambda, P1, base payment,
         predicted response).  Equal to the true values for PRIME with
         perfect knowledge; H_d=0 for H-ignoring baselines (§7.5); perturbed
         for Experiment F (estimation error); conservative for PRIME-R.

Infeasibility reason codes (column "reason_code"):
  0 ok, 1 deadline, 2 quality, 3 a_min>1, 4 contract (Lambda too large),
  5 maintenance smoothing D>D_bar, 6 target not implementable,
  7 IR declined (fixed-base contracts), 8 outside mobility service radius
"""

from __future__ import annotations

import numpy as np

from .mechanisms import MechanismSpec

_EPS = 1e-12
_BISECT_ITERS = 60

REASON_LABELS = {
    0: None,
    1: "deadline_infeasible",
    2: "quality_infeasible",
    3: "a_min_gt_1",
    4: "contract_infeasible",
    5: "maintenance_smoothing_D_exceeds_bar",
    6: "target_not_implementable",
    7: "ir_declined",
    8: "spatial_out_of_range",
}


# ── Vectorised primitives ───────────────────────────────────────

def W_vec(p: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Prelec W(p) = exp(-(-ln p)^zeta), elementwise, with clipping."""
    p = np.clip(p, 0.0, 1.0)
    out = np.zeros_like(p, dtype=float)
    inner = (p > _EPS) & (p < 1.0 - 1e-15)
    ident = zeta <= 0
    lp = -np.log(np.clip(p, _EPS, 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.exp(-np.power(lp, np.where(zeta > 0, zeta, 1.0)))
    out = np.where(inner, w, np.where(p >= 1.0 - 1e-15, 1.0, 0.0))
    out = np.where(ident, p, out)
    return out


def W_inv_vec(y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Inverse Prelec, elementwise, with clipping."""
    y = np.clip(y, 0.0, 1.0)
    ly = -np.log(np.clip(y, _EPS, 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(-np.power(ly, 1.0 / np.where(zeta > 0, zeta, 1.0)))
    p = np.where(y <= _EPS, 0.0, np.where(y >= 1.0 - 1e-15, 1.0, p))
    p = np.where(zeta <= 0, y, p)
    return p


def g_vec(a: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return np.where(a <= 0, 0.0, 1.0 - np.exp(-kappa * np.maximum(a, 0.0)))


def g_prime_vec(a: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return kappa * np.exp(-kappa * np.maximum(a, 0.0))


def C_vec(a: np.ndarray, alpha: np.ndarray, beta: np.ndarray, L: np.ndarray) -> np.ndarray:
    return alpha * L * a + beta * L * a * a


def C_prime_vec(a: np.ndarray, alpha: np.ndarray, beta: np.ndarray, L: np.ndarray) -> np.ndarray:
    return alpha * L + 2.0 * beta * L * a


def response_marginal_lower_envelope(a: np.ndarray, kappa_lo: np.ndarray,
                                     kappa_hi: np.ndarray) -> np.ndarray:
    """Lower envelope of kappa*exp(-kappa*a) on a kappa interval.

    The response marginal is unimodal in kappa, hence its minimum on a
    closed interval is attained at one of the endpoints.
    """
    return np.minimum(g_prime_vec(a, kappa_lo), g_prime_vec(a, kappa_hi))


def _bisect_root(coef: np.ndarray, kappa: np.ndarray, alpha: np.ndarray,
                 beta: np.ndarray, L: np.ndarray, a_lo: np.ndarray) -> np.ndarray:
    """Solve coef * kappa * exp(-kappa a) = alpha L + 2 beta L a on [a_lo, 1].

    The LHS-RHS difference f(a) is strictly decreasing, so:
      f(a_lo) <= 0  ->  root = a_lo
      f(1)    >= 0  ->  root = 1
      else          ->  unique interior root via vectorised bisection.
    Works elementwise; every input is an array of equal shape.
    """
    a_lo = np.clip(a_lo, 0.0, 1.0)

    def f(a: np.ndarray) -> np.ndarray:
        return coef * kappa * np.exp(-kappa * a) - (alpha * L + 2.0 * beta * L * a)

    f_lo = f(a_lo)
    f_hi = f(np.ones_like(a_lo))

    lo = a_lo.copy()
    hi = np.ones_like(a_lo)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    interior = 0.5 * (lo + hi)

    root = np.where(f_lo <= 0, a_lo, np.where(f_hi >= 0, 1.0, interior))
    return np.clip(root, a_lo, 1.0)


# ── Main entry point ────────────────────────────────────────────

def evaluate_pairs(mech: MechanismSpec, ctx: dict) -> dict:
    """Evaluate all pairs of one slot.

    `ctx` maps names to 1-D float arrays (length n_pairs) or scalars:
      task side  : L, V, q_bar, q_min, kappa, input_size, output_size, deadline
      provider   : alpha, beta, omega_t, zeta_t, xi_t, delta_t, U_out,
                   F_i_t, communication_rate, availability, H_t,
                   in_cultivation (bool), last_p, n_interactions
      design side: alpha_d, beta_d, kappa_d, omega_d, zeta_d, xi_d,
                   delta_d, H_d.  The *_d cost/response entries default to
                   their true counterparts for backward compatibility.
      scalars    : p_min, p_max, D_bar, reinforcement_margin, Theta_M,
                   delta_p_max
    Returns dict of arrays incl. feasibility masks and reason codes.
    """
    L = ctx["L"]; V = ctx["V"]; q_bar = ctx["q_bar"]; q_min = ctx["q_min"]
    kappa_t = ctx["kappa"]
    alpha_t = ctx["alpha"]; beta_t = ctx["beta"]
    kappa_d = ctx.get("kappa_d", kappa_t)
    alpha_d = ctx.get("alpha_d", alpha_t); beta_d = ctx.get("beta_d", beta_t)
    omega_t = ctx["omega_t"]; zeta_t = ctx["zeta_t"]
    xi_t = ctx["xi_t"]; delta_t = ctx["delta_t"]
    omega_d = ctx["omega_d"]; zeta_d = ctx["zeta_d"]
    xi_d = ctx["xi_d"]; delta_d = ctx["delta_d"]
    H_t = ctx["H_t"]; H_d = ctx["H_d"]
    U_out = ctx["U_out"]; F = ctx["F_i_t"]
    comm = ctx["communication_rate"]
    avail = ctx["availability"]; deadline = ctx["deadline"]
    in_sz = ctx["input_size"]; out_sz = ctx["output_size"]
    cult = ctx["in_cultivation"].astype(bool)
    last_p = ctx["last_p"]; n_inter = ctx["n_interactions"]

    p_min = float(ctx["p_min"]); p_max = float(ctx["p_max"])
    D_bar = float(ctx["D_bar"])
    margin = float(ctx["reinforcement_margin"])
    Theta_M = float(ctx["Theta_M"])
    delta_p_max = float(ctx["delta_p_max"])

    n = len(L)
    reason = np.zeros(n, dtype=np.int8)

    # Optional PASI-U uncertainty envelope.  The frozen PASI branch is
    # unchanged when robust_enabled is absent/false.  Relative bounds are
    # centred on the platform's design copies; r<1 is required so the
    # corresponding true-parameter interval is finite.
    robust_enabled = bool(ctx.get("robust_enabled", False))
    if robust_enabled:
        r_response = float(ctx.get("response_relative_bound", 0.0))
        r_cost = float(ctx.get("cost_relative_bound", 0.0))
        if not 0.0 <= r_response < 1.0 or not 0.0 <= r_cost < 1.0:
            raise ValueError("robust relative bounds must lie in [0,1)")
        ratio_lo = ctx.get("response_ratio_lower")
        ratio_hi = ctx.get("response_ratio_upper")
        if (ratio_lo is None) != (ratio_hi is None):
            raise ValueError(
                "response_ratio_lower and response_ratio_upper must be supplied together")
        if ratio_lo is None:
            kappa_lo = np.maximum(_EPS, kappa_d / (1.0 + r_response))
            kappa_hi = np.maximum(kappa_lo, kappa_d / max(1.0 - r_response, _EPS))
        else:
            ratio_lo = float(ratio_lo)
            ratio_hi = float(ratio_hi)
            if ratio_lo <= 0.0 or ratio_hi < ratio_lo:
                raise ValueError(
                    "response ratio bounds must satisfy 0 < lower <= upper")
            kappa_lo = np.maximum(_EPS, kappa_d * ratio_lo)
            kappa_hi = np.maximum(kappa_lo, kappa_d * ratio_hi)
        alpha_hi = alpha_d / max(1.0 - r_cost, _EPS)
        beta_hi = beta_d / max(1.0 - r_cost, _EPS)
        H_lo = np.clip(ctx.get("H_lower", H_d), 0.0, 1.0)
    else:
        kappa_lo = kappa_d
        kappa_hi = kappa_d
        alpha_hi = alpha_d
        beta_hi = beta_d
        H_lo = H_d

    # ── 1. Physical feasibility & effort bounds (5.2–5.3) ────
    D_tr = (in_sz + out_sz) / np.maximum(comm, _EPS)
    eff_dl = np.minimum(deadline, avail)
    slack = eff_dl - D_tr

    ok = slack > 0
    reason[~ok] = 1

    a_d = np.where(ok, L / np.maximum(F * np.maximum(slack, _EPS), _EPS), np.nan)

    q_ok = q_min < q_bar - 1e-12
    reason[ok & ~q_ok] = 2
    ok &= q_ok
    ratio = np.clip(1.0 - q_min / np.maximum(q_bar, _EPS), _EPS, 1.0)
    # PASI-U protects physical quality against the least responsive model in
    # the declared envelope.  Baseline PASI retains the nominal expression.
    a_q = -np.log(ratio) / kappa_lo

    a_min = np.maximum(np.where(np.isnan(a_d), np.inf, a_d), a_q)
    amin_ok = a_min <= 1.0
    reason[ok & ~amin_ok] = 3
    ok &= amin_ok

    a_min_safe = np.clip(np.where(ok, a_min, 0.5), 0.0, 1.0)

    # ── 2. System-efficient effort (5.5): V q̄ g'(a) = C'(a) ──
    a_sys = _bisect_root(V * q_bar, kappa_d, alpha_d, beta_d, L, a_min_safe)

    # ── 3. Reinforcement effort (5.6, design-side xi/delta) ──
    denom = xi_d * (1.0 - Theta_M) + delta_d * Theta_M
    s_req = np.where(denom > _EPS, delta_d * Theta_M / np.maximum(denom, _EPS), 1.0)
    s_rein = np.clip(s_req + margin, 0.0, 1.0 - _EPS)
    a_rein = -np.log(np.maximum(1.0 - s_rein, _EPS)) / kappa_d

    # ── 4. Target effort ─────────────────────────────────────
    if mech.target_mode == "prime":
        a_target = np.where(cult, np.maximum(a_sys, a_rein), a_sys)
    elif mech.target_mode == "qos":
        # Protocol-aligned quality-constrained procurement (QIM-E mapping):
        # buy the least effort satisfying the common deadline/QoS constraint.
        a_target = np.minimum(1.0, a_min_safe + 1e-6)
    else:  # "system" and "none" both log a_sys as target
        a_target = a_sys.copy()
    a_target = np.clip(a_target, a_min_safe, 1.0)

    # ── 5. Minimum external incentive Lambda (5.8) ───────────
    gp_d = g_prime_vec(a_target, kappa_d)
    if robust_enabled:
        gp_floor = response_marginal_lower_envelope(a_target, kappa_lo, kappa_hi)
        Lambda = np.maximum(
            0.0,
            C_prime_vec(a_target, alpha_hi, beta_hi, L) / np.maximum(gp_floor, _EPS)
            - omega_d * H_lo,
        )
    else:
        Lambda = np.maximum(0.0, C_prime_vec(a_target, alpha_d, beta_d, L) / np.maximum(gp_d, _EPS)
                            - omega_d * H_d)

    # ── 6. Contract (5.9 / baselines §7) ─────────────────────
    p = np.zeros(n); D = np.zeros(n)
    if mech.contract_mode == "p1":
        w_pmax = W_vec(np.full(n, p_max), zeta_d)
        c_ok = Lambda <= D_bar * w_pmax + 1e-9
        reason[ok & ~c_ok] = 4

        y = np.clip(Lambda / D_bar, _EPS, 1.0 - _EPS)
        p_L = np.maximum(p_min, W_inv_vec(y, zeta_d))
        p_L = np.minimum(p_L, p_max)

        w_pL = W_vec(p_L, zeta_d)
        D_L = Lambda / np.maximum(w_pL, _EPS)
        D_M = Lambda / np.maximum(w_pmax, _EPS)
        int_L = np.where(D_L <= D_bar + 1e-9, p_L * D_L, np.inf)
        int_M = np.where(D_M <= D_bar + 1e-9, p_max * D_M, np.inf)

        # Strictly-better tie-break matches the scalar reference: on ties
        # (e.g. zeta=1 where p*D is constant) keep p_max.
        use_L = int_L < int_M - 1e-12
        p = np.where(use_L, p_L, p_max)
        D = np.minimum(np.where(use_L, D_L, D_M), D_bar)
        zero = Lambda <= 0
        p = np.where(zero, 0.0, p)
        D = np.where(zero, 0.0, D)
        ok &= c_ok

        # Maintenance probability smoothing (5.13), PRIME family only.
        # Matches the scalar reference: applies even when Lambda=0
        # (p is lifted along the smooth path, D stays Lambda/W = 0).
        if mech.maintenance_smoothing:
            smooth = ok & (~cult) & (last_p > 0) & (p < last_p)
            if smooth.any():
                p_s = np.maximum(p, last_p - delta_p_max)
                w_s = W_vec(p_s, zeta_d)
                D_s = Lambda / np.maximum(w_s, _EPS)
                bad = smooth & (D_s > D_bar + 1e-9)
                reason[bad] = 5
                ok &= ~bad
                app = smooth & ~bad
                p = np.where(app, p_s, p)
                D = np.where(app, D_s, D)
    elif mech.contract_mode == "risk_neutral":
        # Platform prices ignoring probability weighting: D = Lambda / p.
        p = np.full(n, float(mech.p_fixed))
        D = np.clip(Lambda / np.maximum(p, _EPS), 0.0, D_bar)
    elif mech.contract_mode == "fixed":
        p = np.full(n, float(np.clip(mech.p_fixed, p_min, p_max)))
        D = np.full(n, float(min(mech.D_fixed, D_bar)))
    elif mech.contract_mode == "ldi":
        frac = np.clip(n_inter / max(float(mech.decay_interactions), 1.0), 0.0, 1.0)
        p = mech.p_start - (mech.p_start - mech.p_end) * frac
        p = np.clip(p, p_min, p_max)
        D = np.full(n, float(min(mech.D_fixed, D_bar)))
    else:
        raise ValueError(f"unknown contract_mode {mech.contract_mode}")

    # Quality-Matched protocol knob (§13.3): scale the external incentive
    # of non-optimal contract families (never applied to P1 contracts).
    scale = float(getattr(mech, "incentive_scale", 1.0) or 1.0)
    if scale != 1.0 and mech.contract_mode in ("fixed", "risk_neutral", "ldi"):
        D = np.clip(D * scale, 0.0, D_bar)

    # ── 7. Induced effort a_star (5.10) ──────────────────────
    # True behavioural response (always uses true zeta/omega/H).
    Gamma_t = W_vec(p, zeta_t) * D + omega_t * H_t
    a_star = _bisect_root(Gamma_t, kappa_t, alpha_t, beta_t, L, a_min_safe)

    # Platform-predicted response (design side).  Equal when no error.
    design_equals_true = bool(ctx.get("design_equals_true", True))
    if design_equals_true and mech.design_H == "true":
        a_star_d = a_star
        Gamma_d = Gamma_t
    else:
        Gamma_d = W_vec(p, zeta_d) * D + omega_d * H_d
        a_star_d = _bisect_root(Gamma_d, kappa_d, alpha_d, beta_d, L, a_min_safe)

    gap_true = a_star - a_target
    gap_design = a_star_d - a_target

    if mech.enforce_target:
        # Platform only offers contracts it PREDICTS to be implementable.
        t_ok = gap_design >= -1e-7
        reason[ok & ~t_ok] = 6
        ok &= t_ok

    # ── 8. Base payment (5.11) ───────────────────────────────
    g_star_d = g_vec(a_star_d, kappa_d)
    if mech.base_mode == "fixed":
        scale_b = float(getattr(mech, "incentive_scale", 1.0) or 1.0)
        b = np.full(n, float(mech.b_fixed) * scale_b)
    elif mech.base_mode == "quac_ir":
        # QUAC-F's affine quality contract uses a signed intercept A.  The
        # paper's reported optimal contracts also contain negative A values;
        # IR applies to total payment at the induced quality, not to A alone.
        b = (U_out + C_vec(a_star_d, alpha_d, beta_d, L)
             - p * D * g_star_d - omega_d * H_d * g_star_d)
    else:
        # Response-consistent, computed at the platform's predicted response
        # with design-side omega*H (0 for H-ignoring baselines).
        b = np.maximum(0.0, U_out + C_vec(a_star_d, alpha_d, beta_d, L)
                       - p * D * g_star_d - omega_d * H_d * g_star_d)

        if robust_enabled:
            # It is sufficient to make the target action individually
            # rational under the worst admissible cost/response/state tuple:
            # the provider's actual optimum then has at least this utility.
            g_target_floor = g_vec(a_target, kappa_lo)
            b_floor = (U_out + C_vec(a_target, alpha_hi, beta_hi, L)
                       - (p * D + omega_d * H_lo) * g_target_floor)
            b = np.maximum(b, np.maximum(0.0, b_floor))

    # ── 9. Realised quantities (TRUE response) ───────────────
    g_star = g_vec(a_star, kappa_t)
    cost_star = C_vec(a_star, alpha_t, beta_t, L)
    experienced = b + p * D * g_star + omega_t * H_t * g_star - cost_star
    perceived = b + W_vec(p, zeta_t) * D * g_star + omega_t * H_t * g_star - cost_star
    # Audit-only procurement primitives used by public auction comparisons.
    # They do not enter PASI's feasibility, contract, or matching decisions.
    runtime_path_value = omega_t * H_t * g_star
    effective_reserve_cost = np.maximum(
        0.0, cost_star - runtime_path_value + U_out
    )
    ir_ok = experienced + 1e-8 >= U_out

    if mech.base_mode == "fixed":
        # Participation constraint: provider declines when IR violated.
        decl = ok & ~ir_ok
        reason[decl] = 7
        ok &= ir_ok

    quality = q_bar * g_star
    delay = D_tr + L / np.maximum(a_star * F, _EPS)
    bonus = p * D * g_star
    total_cost = b + bonus
    immediate = V * quality - total_cost

    # ── 10. Reinforcement value (5.14, design-side prediction) ─
    g_d = g_star_d
    dH_pred = xi_d * g_d * (1.0 - H_d) - delta_d * (1.0 - g_d) * H_d
    if mech.eta_H and mech.eta_H > 0:
        reinf = np.where(cult, mech.eta_H * np.maximum(dH_pred, 0.0), 0.0)
    else:
        reinf = np.zeros(n)
    pair_value = immediate + reinf

    # True predicted Delta H (for logging / state update preview)
    dH_true = xi_t * g_star * (1.0 - H_t) - delta_t * (1.0 - g_star) * H_t

    def _z(x: np.ndarray) -> np.ndarray:
        return np.where(ok, x, 0.0)

    return {
        "feasible": ok,
        "feasible_physical": reason < 4,  # physical checks are codes 1–3
        "reason_code": reason,
        "a_deadline": np.where(np.isnan(a_d), 0.0, a_d),
        "a_quality": a_q,
        "a_min": np.where(ok, a_min, 0.0),
        "a_system": _z(a_sys),
        "a_reinforcement": _z(a_rein),
        "a_target": _z(a_target),
        "lambda_required": _z(Lambda),
        "robust_contract": np.full(n, robust_enabled, dtype=bool),
        "H_design_lower": _z(H_lo),
        "p_star": _z(p),
        "D_star": _z(D),
        "gamma_effective": _z(Gamma_t),
        "a_star": _z(a_star),
        "a_star_design": _z(a_star_d),
        "implementation_gap": _z(gap_true),
        "implementation_gap_design": _z(gap_design),
        "base_payment": _z(b),
        "expected_bonus": _z(bonus),
        "expected_contract_cost": _z(total_cost),
        "experienced_utility": _z(experienced),
        "perceived_utility": _z(perceived),
        "realized_effort_cost": _z(cost_star),
        "runtime_path_value": _z(runtime_path_value),
        "outside_option": _z(U_out),
        "effective_reserve_cost": _z(effective_reserve_cost),
        "ir_ok": ir_ok | ~ok,
        "execution_quality": _z(quality),
        "normalized_quality": _z(g_star),
        "total_delay": _z(delay),
        "delta_H_predicted": _z(dH_true),
        "immediate_value": _z(immediate),
        "reinforcement_value": _z(reinf),
        "total_pair_value": _z(pair_value),
    }
