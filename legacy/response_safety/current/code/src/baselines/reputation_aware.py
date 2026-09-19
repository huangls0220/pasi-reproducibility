"""Reputation-Aware Incentive (RAI) baseline — Section 7.3.

Reputation r_i is an EWMA of historical normalized quality:
    r_i <- (1 - a) * r_i + a * s_i           (after each assigned task)

Reputation multiplies the MATCHING weight (spec allows weight OR payment
coefficient; we use weight and document it):
    matching_weight = pair_value * (w0 + w1 * r_i)
New providers (fewer than `explore_interactions` assignments) get at
least `explore_multiplier` so they can build reputation (limited
exploration, §7.3).

Reputation is NOT a non-monetary utility: it never enters the provider's
utility function.  Contract pricing is risk-neutral with H=0 design
(same pricing family as DP) so the reputation channel is isolated.

Tunable: baselines.reputation_aware.{p, ewma_alpha, weight_base,
weight_scale, explore_interactions, explore_multiplier}
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("RAI", cfg)
