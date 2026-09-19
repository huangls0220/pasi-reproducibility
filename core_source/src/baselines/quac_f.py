"""Isomorphic implementation of the public QUAC-F contract baseline.

Source: M. Li, J. Lin, D. Yang, G. Xue, and J. Tang, "QUAC:
Quality-Aware Contract-Based Incentive Mechanisms for Crowdsensing," IEEE
MASS, 2017, doi:10.1109/MASS.2017.45.

This is not author-released code.  It implements the paper's risk-neutral
linear quality contract in the shared simulator.  QUAC-F's payment A+Bq is
represented as b+D*g(a), with q=q_bar*g(a), D=B*q_bar, the slope chosen from
the published effort first-order condition, and b chosen so IR binds.  The
baseline then uses the exact same event tape, quality/cost primitives,
provider capacity, budget, and coverage-first matcher as PASI.
"""

from ..mechanisms import get_mechanism


def spec(cfg: dict):
    return get_mechanism("QUAC-F", cfg)
