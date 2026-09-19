"""Myopic Optimal Incentive (MOI) baseline — Section 7.5.

The most important, strongest non-path-dependent baseline.  Uses the SAME
machinery as PRIME — target effort solver, P1 optimal probability–bonus
contract, a_star response, response-consistent base payment, Lagrangian
matching — but the platform sets H = 0 and eta_H = 0 in contract design
and matching:

  - Lambda = C'(a_target)/g'(a_target)          (no omega*H credit)
  - a_target = a_system                          (no cultivation targets)
  - base payment computed with H = 0             (no intrinsic-utility discount)
  - matching value = immediate value only        (eta_H = 0)

Provider BEHAVIOUR is unchanged (§2.5): the true a_star response and the
experienced utility still include the provider's real omega*H, and the
true path state H keeps evolving — MOI simply never exploits it.  As a
result MOI providers weakly over-deliver (implementation_gap >= 0) while
the platform pays the full external incentive forever; the PRIME-vs-MOI
payment difference is the value of path dependence (Experiment C).

Implementation: MechanismSpec(design_H="zero", target_mode="system",
eta_H=0) in src/mechanisms.py.
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("MOI", cfg)
