"""E8 state-observation and model-error controls."""

import numpy as np
import pytest

from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator
from test_end_to_end import _small_config


def _sim(extra: dict | None = None, seed: int = 17) -> Simulator:
    cfg = _small_config()
    if extra:
        cfg.update(extra)
    data = generate_synthetic_episode(cfg, seed=seed, pattern="stationary")
    return Simulator(cfg, data, method="PASI", seed=seed)


def test_default_observer_is_identity():
    sim = _sim()
    assert sim.design_equals_true
    np.testing.assert_allclose(sim._observed_H(0), sim.H_design_state)


def test_noise_is_seeded_and_bounded():
    cfg = {"state_observation": {"noise_std": 0.2}}
    a = _sim(cfg)._observed_H(3)
    b = _sim(cfg)._observed_H(3)
    np.testing.assert_allclose(a, b)
    assert np.all((0.0 <= a) & (a <= 1.0))


def test_delay_and_locf_missingness():
    delayed = _sim({"state_observation": {"delay_slots": 1}})
    first = delayed._observed_H(0).copy()
    delayed.H_design_state[:] = 0.9
    np.testing.assert_allclose(delayed._observed_H(1), first)
    np.testing.assert_allclose(delayed._observed_H(2), 0.9)

    missing = _sim({"state_observation": {"missing_rate": 1.0}})
    initial = missing._observed_H(0).copy()
    missing.H_design_state[:] = 0.8
    np.testing.assert_allclose(missing._observed_H(1), initial)


def test_cost_and_response_bias_are_design_side_only():
    sim = _sim({"model_error": {"cost_bias": 0.30, "response_bias": -0.20}})
    np.testing.assert_allclose(sim.d_alpha, sim.s_alpha * 1.30)
    np.testing.assert_allclose(sim.d_beta, sim.s_beta * 1.30)
    assert sim.response_model_scale == 0.80
    assert not sim.design_equals_true


def test_uncertainty_aware_state_lower_bound():
    sim = _sim({
        "state_observation": {"noise_std": 0.05, "delay_slots": 2},
        "uncertainty_aware": {
            "enabled": True,
            "state_confidence_z": 2.0,
            "response_relative_bound": 0.2,
            "cost_relative_bound": 0.2,
        },
    })
    observed = np.full(3, 0.8)
    ids = np.arange(3)
    lower = sim._state_lower_bound(observed, ids, np.ones(3))
    expected = np.clip(0.8 - 0.10 - 2 * sim.d_delta[ids], 0.0, 1.0)
    np.testing.assert_allclose(lower, expected)


def test_uncertainty_aware_missingness_uses_zero_state_credit():
    sim = _sim({
        "state_observation": {"missing_rate": 0.1},
        "uncertainty_aware": {"enabled": True},
    })
    lower = sim._state_lower_bound(np.full(4, 0.8), np.arange(4), np.ones(4))
    np.testing.assert_allclose(lower, 0.0)


def test_directional_response_ratio_bounds_are_optional_and_validated():
    default = _sim({"uncertainty_aware": {"enabled": True}})
    assert default.robust_response_ratio_lower is None
    assert default.robust_response_ratio_upper is None

    directional = _sim({
        "uncertainty_aware": {
            "enabled": True,
            "response_ratio_lower": 1.20,
            "response_ratio_upper": 1.30,
        },
    })
    assert directional.robust_response_ratio_lower == pytest.approx(1.20)
    assert directional.robust_response_ratio_upper == pytest.approx(1.30)

    with pytest.raises(ValueError, match="must be supplied together"):
        _sim({
            "uncertainty_aware": {
                "enabled": True,
                "response_ratio_lower": 1.20,
            },
        })

    with pytest.raises(ValueError, match="0 < lower <= upper"):
        _sim({
            "uncertainty_aware": {
                "enabled": True,
                "response_ratio_lower": 1.30,
                "response_ratio_upper": 1.20,
            },
        })
