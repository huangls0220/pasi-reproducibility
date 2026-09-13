"""Prelec probability weighting (Section 5.7).

W(p) = exp(-(-ln p)^zeta)
W(0) = 0, W(1) = 1

Inverse:
W_inv(y) = exp(-(-ln y)^(1/zeta))
"""

from __future__ import annotations

import math

_EPS = 1e-12


def _safe_clip(x: float, lo: float = _EPS, hi: float = 1.0) -> float:
    """Clip value to safe range for log operations."""
    return float(max(lo, min(hi, x)))


def W(p: float, zeta: float) -> float:
    """Prelec probability weighting function."""
    if p <= 0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # p in (0,1)
    if zeta <= 0:
        return p  # degenerate: identity
    lp = -math.log(p)  # lp > 0 since p < 1
    return math.exp(-(lp**zeta))


def W_inv(y: float, zeta: float) -> float:
    """Inverse Prelec function: given weighted prob y, return raw prob p."""
    if y <= 0:
        return 0.0
    if y >= 1.0:
        return 1.0
    if zeta <= 0:
        return y
    ly = -math.log(y)  # ly > 0
    raw = math.exp(-(ly ** (1.0 / zeta)))
    return float(raw)


def W_prime(p: float, zeta: float) -> float:
    """Derivative of W w.r.t p (for diagnostics)."""
    if p <= 0 or p >= 1.0 or zeta <= 0:
        return 1.0
    lp = -math.log(p)
    w = math.exp(-(lp**zeta))
    return w * zeta * (lp ** (zeta - 1.0)) / max(p, _EPS)
