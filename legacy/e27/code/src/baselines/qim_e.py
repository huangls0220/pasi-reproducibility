"""QIM-E winner selection and critical payments.

Source: J. Wang et al., "Quality-Aware and Fine-Grained Incentive
Mechanisms for Mobile Crowdsensing," IEEE ICDCS 2016,
doi:10.1109/ICDCS.2016.30.  This independent implementation follows
Algorithms 1 and 2 under the paper's U(0, 4] cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class QIMEOutcome:
    winners: np.ndarray
    payments: np.ndarray
    achieved_quality: np.ndarray


def virtual_cost_uniform(cost: np.ndarray, upper: float = 4.0) -> np.ndarray:
    """Return beta(c)=c+F(c)/f(c)=2c for c distributed U(0, upper]."""
    values = np.asarray(cost, dtype=float)
    if np.any(values < -_EPS) or np.any(values > upper + _EPS):
        raise ValueError("QIM-E costs must lie in the declared uniform support")
    return 2.0 * np.clip(values, 0.0, upper)


def inverse_virtual_cost_uniform(value: float, upper: float = 4.0) -> float:
    """Inverse beta on the declared support, saturated at its endpoint."""
    return float(np.clip(0.5 * float(value), 0.0, upper))


def _achieved_max_quality(qualities: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.zeros(qualities.shape[1], dtype=float)
    return np.max(qualities[np.asarray(selected, dtype=int)], axis=0)


def _normalized_marginal(
    qualities: np.ndarray,
    requirements: np.ndarray,
    selected: list[int],
) -> np.ndarray:
    current = _achieved_max_quality(qualities, selected)
    capped_current = np.minimum(current, requirements)
    expanded = np.maximum(current[None, :], qualities)
    gains = np.maximum(
        0.0,
        np.minimum(expanded, requirements) - capped_current[None, :],
    )
    return np.sum(gains / np.maximum(requirements[None, :], _EPS), axis=1)


def select_winners(
    costs: np.ndarray,
    qualities: np.ndarray,
    requirements: np.ndarray,
    bidder_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Algorithm 1: repeatedly select minimum virtual-cost/marginal-QoC."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    if qualities.ndim != 2 or qualities.shape[0] != len(costs):
        raise ValueError("qualities must be a bidder-by-subtask matrix")
    if qualities.shape[1] != len(requirements):
        raise ValueError("requirements must match the subtask dimension")
    if np.any(requirements <= 0.0) or np.any(requirements > 1.0 + _EPS):
        raise ValueError("QIM-E quality requirements must lie in (0, 1]")
    if np.any(qualities < -_EPS) or np.any(qualities > 1.0 + _EPS):
        raise ValueError("QIM-E quality scores must lie in [0, 1]")

    beta = virtual_cost_uniform(costs)
    ids = (
        np.arange(len(costs), dtype=int)
        if bidder_ids is None
        else np.asarray(bidder_ids)
    )
    remaining = list(range(len(costs)))
    selected: list[int] = []
    while remaining:
        marginal = _normalized_marginal(qualities, requirements, selected)
        active = [idx for idx in remaining if marginal[idx] > _EPS]
        if not active:
            break
        chosen = min(
            active,
            key=lambda idx: (beta[idx] / marginal[idx], str(ids[idx])),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(selected, dtype=int)


def critical_payments(
    costs: np.ndarray,
    qualities: np.ndarray,
    requirements: np.ndarray,
    winners: np.ndarray,
    bidder_ids: np.ndarray | None = None,
    support_upper: float = 4.0,
) -> np.ndarray:
    """Algorithm 2: replay without each winner to obtain critical bids."""
    costs = np.asarray(costs, dtype=float)
    qualities = np.asarray(qualities, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    ids = (
        np.arange(len(costs), dtype=int)
        if bidder_ids is None
        else np.asarray(bidder_ids)
    )
    beta = virtual_cost_uniform(costs, upper=support_upper)
    payments = np.zeros(len(costs), dtype=float)

    for winner in np.asarray(winners, dtype=int):
        remaining = [idx for idx in range(len(costs)) if idx != winner]
        selected_without: list[int] = []
        payment = 0.0
        while remaining:
            marginal = _normalized_marginal(
                qualities, requirements, selected_without
            )
            active = [idx for idx in remaining if marginal[idx] > _EPS]
            if not active:
                break
            chosen = min(
                active,
                key=lambda idx: (beta[idx] / marginal[idx], str(ids[idx])),
            )
            winner_marginal = marginal[winner]
            if winner_marginal > _EPS:
                threshold_beta = (
                    beta[chosen] / marginal[chosen]
                ) * winner_marginal
                payment = max(
                    payment,
                    inverse_virtual_cost_uniform(
                        threshold_beta, upper=support_upper
                    ),
                )
            selected_without.append(chosen)
            remaining.remove(chosen)
            if _normalized_marginal(
                qualities, requirements, selected_without
            )[winner] <= _EPS:
                break
        payments[winner] = payment
    return payments


def run_qim_e(
    costs: np.ndarray,
    qualities: np.ndarray,
    requirements: np.ndarray,
    bidder_ids: np.ndarray | None = None,
) -> QIMEOutcome:
    winners = select_winners(
        costs, qualities, requirements, bidder_ids=bidder_ids
    )
    payments = critical_payments(
        costs,
        qualities,
        requirements,
        winners,
        bidder_ids=bidder_ids,
    )
    achieved = _achieved_max_quality(qualities, winners.tolist())
    if np.any(achieved + 1e-10 < requirements):
        raise ValueError("QIM-E candidate set cannot meet every requirement")
    if np.any(payments[winners] + 1e-10 < costs[winners]):
        raise AssertionError("QIM-E critical payment violated IR")
    return QIMEOutcome(winners=winners, payments=payments, achieved_quality=achieved)
