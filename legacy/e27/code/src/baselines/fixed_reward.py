"""Fixed Reward (FR) baseline — Section 7.1.

Fixed (base_payment, p, D) for ALL pairs regardless of Lambda or H.
The provider still best-responds with its TRUE behavioural utility:
    Gamma = W(p_fixed) * D_fixed + omega * H
(the Prelec weighting belongs to the provider, not the mechanism).

Because the base payment is fixed, the participation constraint matters:
pairs whose experienced utility would fall below the outside option are
DECLINED by the provider (reason code "ir_declined").

Tunable parameters (validation grid, Section 13.1):
    baselines.fixed_reward.{base_payment, p, D}

Implementation: MechanismSpec(contract_mode="fixed", base_mode="fixed")
in src/mechanisms.py; evaluated by src/pair_eval.py.
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("FR", cfg)
