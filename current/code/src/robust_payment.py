"""Certified sufficient objective-IR reserve, not a minimum-payment solver.

Model: a in [0,1], g_k(a)=1-exp(-k*a), C=(alpha*a+beta*a*a)*L.
Known probability weighting; k >= k_lo, k <= k_hi, rho >= rho_lo,
alpha <= alpha_hi and beta <= beta_hi, all parameters nonnegative.
The comparison-action lower bound on perceived value and the convex
all-action upper bound are both valid on the entire continuous box.
"""
from __future__ import annotations

import numpy as np


def objective_ir_reserve(U, perceived_bonus, objective_bonus, rho_lo,
                         k_lo, k_hi, alpha_hi, beta_hi, L):
    """Broadcast inputs; return a nonnegative uniform IR reserve and g upper.

Let v_lo=max_a (perceived_bonus+rho_lo)*g_lo(a)-C_hi(a).
Actual perceived value >= v_lo. Actual objective value is at least
v_lo - max(perceived_bonus-objective_bonus,0)*g_hi(1).
Separately, C_hi(a)-(objective_bonus+rho_lo)*g_lo(a) is convex, so
its maximum on [0,1] is at 0 or 1. Take the tighter sufficient bound.
No current response, cost or realized effort may enter this function.
"""
    U, lam, t, rho, kl, kh, ah, bh, scale = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in
          (U, perceived_bonus, objective_bonus, rho_lo, k_lo, k_hi,
           alpha_hi, beta_hi, L)])
    if any(np.any(~np.isfinite(x)) for x in (U, lam, t, rho, kl, kh, ah, bh, scale)):
        raise ValueError('IR envelope inputs must be finite')
    if (np.any(kl <= 0) or np.any(kh < kl) or np.any(scale <= 0)
            or any(np.any(x < 0) for x in (lam, t, rho, ah, bh))):
        raise ValueError('invalid nonnegative exponential/quadratic IR envelope')
    lo = np.zeros_like(kl)
    hi = np.ones_like(kl)
    for _ in range(60):
        mid = (lo + hi) / 2
        derivative = (lam + rho)*kl*np.exp(-kl*mid) - scale*(ah+2*bh*mid)
        lo = np.where(derivative > 0, mid, lo)
        hi = np.where(derivative > 0, hi, mid)
    a = (lo + hi)/2
    value = (lam+rho)*(-np.expm1(-kl*a)) - scale*(ah*a+bh*a*a)
    # Include a=0 exactly; any comparison action is a valid lower bound.
    value = np.maximum(0.0, value)
    g_upper = -np.expm1(-kh)
    comparison_bound = U-value+np.maximum(lam-t, 0)*g_upper
    all_action_bound = U+np.maximum(0.0, scale*(ah+bh)-(t+rho)*(-np.expm1(-kl)))
    reserve = np.maximum(0.0, np.minimum(comparison_bound, all_action_bound))
    return reserve, g_upper


def objective_ir_reserve_scalar(*args):
    """Scalar entry for the same certified bound (not an independent oracle)."""
    reserve, upper = objective_ir_reserve(*args)
    return float(reserve), float(upper)
