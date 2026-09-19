"""CSOPT allocation and externality payments for mobile crowdsensing.

Source: D. Chatzopoulos et al., "Privacy Preserving and Cost Optimal
Mobile Crowdsensing using Smart Contracts on Blockchain," IEEE MASS 2018,
arXiv:1808.04056.  This independent implementation follows Algorithm 1.

Each bidder reports a homogeneous cost per allocated task and a set of tasks
it can perform.  The allocation rule repeatedly selects the lowest-cost
remaining bidder and assigns every still-uncovered task in its interest set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class CSOPTOutcome:
    assignment: np.ndarray
    winners: np.ndarray
    payments: np.ndarray
    allocation_cost: float


def allocate(
    costs: np.ndarray,
    interests: np.ndarray,
    redundancy: int = 1,
    bidder_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Implement CSOPT ALLOC-RULE and return a bidder-by-task matrix."""
    costs = np.asarray(costs, dtype=float)
    interests = np.asarray(interests, dtype=bool)
    if interests.ndim != 2 or interests.shape[0] != len(costs):
        raise ValueError("interests must be a bidder-by-task matrix")
    if np.any(~np.isfinite(costs)) or np.any(costs < -_EPS):
        raise ValueError("CSOPT costs must be finite and nonnegative")
    if int(redundancy) != redundancy or redundancy < 1:
        raise ValueError("redundancy must be a positive integer")

    ids = (
        np.arange(len(costs), dtype=int)
        if bidder_ids is None
        else np.asarray(bidder_ids)
    )
    if len(ids) != len(costs):
        raise ValueError("bidder_ids must match costs")

    demand = np.full(interests.shape[1], int(redundancy), dtype=int)
    assignment = np.zeros_like(interests, dtype=bool)
    remaining = list(range(len(costs)))
    while np.any(demand > 0):
        active = [
            idx
            for idx in remaining
            if np.any(interests[idx] & (demand > 0))
        ]
        if not active:
            raise ValueError("CSOPT candidate set cannot meet task demand")
        chosen = min(active, key=lambda idx: (costs[idx], str(ids[idx])))
        assigned = interests[chosen] & (demand > 0)
        assignment[chosen, assigned] = True
        demand[assigned] -= 1
        remaining.remove(chosen)
    return assignment


def allocation_cost(costs: np.ndarray, assignment: np.ndarray) -> float:
    costs = np.asarray(costs, dtype=float)
    assignment = np.asarray(assignment, dtype=bool)
    return float(np.dot(costs, assignment.sum(axis=1)))


def externality_payments(
    costs: np.ndarray,
    interests: np.ndarray,
    assignment: np.ndarray,
    redundancy: int = 1,
    bidder_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Implement CSOPT PAYMENT-RULE by removing each selected bidder."""
    costs = np.asarray(costs, dtype=float)
    interests = np.asarray(interests, dtype=bool)
    assignment = np.asarray(assignment, dtype=bool)
    ids = (
        np.arange(len(costs), dtype=int)
        if bidder_ids is None
        else np.asarray(bidder_ids)
    )
    total_cost = allocation_cost(costs, assignment)
    selected = np.flatnonzero(assignment.any(axis=1))
    payments = np.zeros(len(costs), dtype=float)
    for winner in selected:
        keep = np.arange(len(costs)) != winner
        without = allocate(
            costs[keep],
            interests[keep],
            redundancy=redundancy,
            bidder_ids=ids[keep],
        )
        cost_without = allocation_cost(costs[keep], without)
        own_cost = costs[winner] * int(assignment[winner].sum())
        payments[winner] = own_cost + cost_without - total_cost
        if payments[winner] + 1e-10 < own_cost:
            raise AssertionError("CSOPT externality payment violated IR")
    return payments


def run_csopt(
    costs: np.ndarray,
    interests: np.ndarray,
    redundancy: int = 1,
    bidder_ids: np.ndarray | None = None,
) -> CSOPTOutcome:
    assignment = allocate(
        costs,
        interests,
        redundancy=redundancy,
        bidder_ids=bidder_ids,
    )
    payments = externality_payments(
        costs,
        interests,
        assignment,
        redundancy=redundancy,
        bidder_ids=bidder_ids,
    )
    return CSOPTOutcome(
        assignment=assignment,
        winners=np.flatnonzero(assignment.any(axis=1)),
        payments=payments,
        allocation_cost=allocation_cost(costs, assignment),
    )
