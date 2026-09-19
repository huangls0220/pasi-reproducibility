"""Semantic tests for the QIM-E implementation."""

import numpy as np

from src.baselines.qim_e import (
    critical_payments,
    run_qim_e,
    select_winners,
    virtual_cost_uniform,
)


def test_uniform_virtual_cost_matches_paper_setting():
    costs = np.array([0.25, 1.0, 4.0])
    assert np.allclose(virtual_cost_uniform(costs), 2.0 * costs)


def test_single_minded_max_qoc_selects_one_bidder_per_subtask():
    costs = np.array([0.30, 0.50, 0.20, 0.45])
    qualities = np.array([
        [0.80, 0.00],
        [0.95, 0.00],
        [0.00, 0.75],
        [0.00, 0.90],
    ])
    requirements = np.array([0.70, 0.70])
    outcome = run_qim_e(costs, qualities, requirements)
    assert set(outcome.winners.tolist()) == {0, 2}
    assert np.all(outcome.achieved_quality >= requirements)
    assert np.all(outcome.payments[outcome.winners] >= costs[outcome.winners])


def test_critical_payment_uses_capped_qoc_at_the_requirement():
    costs = np.array([0.30, 0.50])
    qualities = np.array([[0.80], [1.00]])
    requirements = np.array([0.70])
    winners = select_winners(costs, qualities, requirements)
    payments = critical_payments(costs, qualities, requirements, winners)
    assert winners.tolist() == [0]
    assert np.allclose(payments[0], 0.50)


def test_critical_payment_scales_with_uncapped_marginal_qoc():
    costs = np.array([0.30, 0.50])
    qualities = np.array([[0.40], [0.50]])
    requirements = np.array([0.70])
    winners = select_winners(costs, qualities, requirements)
    payments = critical_payments(costs, qualities, requirements, winners)
    assert winners.tolist()[0] == 0
    assert np.allclose(payments[0], 0.50 * 0.40 / 0.50)
