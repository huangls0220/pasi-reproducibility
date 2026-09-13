"""Semantic tests for the CSOPT Algorithm 1 implementation."""

import numpy as np
import pytest

from src.baselines.csopt import allocate, run_csopt


def test_single_task_payment_is_second_bid():
    costs = np.array([0.30, 0.50, 0.80])
    interests = np.ones((3, 1), dtype=bool)
    outcome = run_csopt(costs, interests)
    assert outcome.winners.tolist() == [0]
    assert np.allclose(outcome.payments[0], 0.50)
    assert np.allclose(outcome.allocation_cost, 0.30)


def test_lowest_cost_bundle_receives_externality_payment():
    costs = np.array([0.50, 0.70, 0.80])
    interests = np.array([
        [True, True],
        [True, False],
        [False, True],
    ])
    outcome = run_csopt(costs, interests)
    assert outcome.winners.tolist() == [0]
    assert outcome.assignment[0].tolist() == [True, True]
    assert np.allclose(outcome.allocation_cost, 1.0)
    assert np.allclose(outcome.payments[0], 1.5)


def test_removal_without_competition_fails_closed():
    costs = np.array([0.30])
    interests = np.array([[True]])
    with pytest.raises(ValueError, match="cannot meet task demand"):
        run_csopt(costs, interests)


def test_allocation_is_deterministic_under_cost_ties():
    costs = np.array([0.30, 0.30])
    interests = np.ones((2, 1), dtype=bool)
    assignment = allocate(
        costs,
        interests,
        bidder_ids=np.array(["B", "A"]),
    )
    assert assignment[:, 0].tolist() == [False, True]
