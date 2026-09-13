"""Linear Decay Incentive (LDI) baseline — Section 7.4.

Bonus probability decays LINEARLY with the provider's assigned
interaction count n_i (a fixed schedule, NOT based on H or quality):
    p(n_i) = p_start - (p_start - p_end) * min(1, n_i / decay_interactions)
D is fixed.  If the decayed incentive is insufficient, real quality is
ALLOWED to drop (a_star is never forced, §7.4).

LDI mimics PRIME's "reduce incentives over time" on a blind schedule —
whether quality survives depends on whether the provider's own H grew
enough, which LDI neither measures nor manages.

Tunable: baselines.linear_decay.{p_start, p_end, decay_interactions, D}
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("LDI", cfg)
