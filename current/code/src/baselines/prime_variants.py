"""PRIME ablation variants — Section 7.6.

- PRIME-w/o-PD : omega=0 in the MODEL (no path-dependent utility at all;
                 H still logged).  Behaviour-level ablation.
- PRIME-w/o-PW : zeta=1 (W(p)=p everywhere).  Behaviour-level ablation.
- PRIME-w/o-LT : eta_H=0 (no future reinforcement value in matching).
- PRIME-w/o-RC : maintenance never returns to cultivation.
- PRIME-Fixed  : switch to maintenance after a fixed number of assigned
                 interactions (baselines.prime_fixed.cultivation_interactions),
                 never switch back.
- PRIME-R      : robust variant for Experiment F — uses the most
                 conservative behavioural contribution within the
                 estimation-uncertainty interval (lower omega credit,
                 zeta toward 1 in the p<1/e regime).

All variants are resolved by src/mechanisms.get_mechanism; no config
mutation is needed anymore.
"""

from ..mechanisms import ABLATION_METHODS, get_mechanism


def spec(variant: str, cfg: dict):
    return get_mechanism(variant, cfg)


VARIANTS = [m for m in ABLATION_METHODS if m != "PRIME"] + ["PRIME-R"]
