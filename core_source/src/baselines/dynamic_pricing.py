"""Dynamic Pricing (DP) baseline — Section 7.2.

Adapts the payment to the CURRENT task (value, cost, quality requirement)
by targeting the myopic system-efficient effort a_system, but:
  - ignores H entirely in contract design (H=0 each round, §7.2);
  - prices RISK-NEUTRALLY: D = Lambda / p_fixed (no probability-weighting
    model), clipped to D_bar — if the clip binds the pair is still offered
    and real quality may fall short of the myopic target;
  - no future reinforcement value (eta_H = 0).

The provider response is still behavioural (true W(p), true omega*H), so
DP can over- or under-incentivise depending on the Prelec regime.

Tunable: baselines.dynamic_pricing.p

Implementation: MechanismSpec(contract_mode="risk_neutral",
design_H="zero", target_mode="system") in src/mechanisms.py.
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("DP", cfg)
