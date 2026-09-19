import numpy as np

from src.simulator import Simulator


def make_stub(observable: bool) -> Simulator:
    sim = Simulator.__new__(Simulator)
    sim.state_reset_config = {"observable_to_platform": observable}
    sim.reset_provider_ids = np.array([0, 2], dtype=int)
    sim.H_true = np.array([0.0, 0.4, 0.0])
    sim.H_design_state = np.array([0.8, 0.4, 0.7])
    sim._obs_last = np.array([0.8, 0.4, 0.7])
    sim._obs_cache_slot = 499
    sim._obs_cache = sim.H_design_state.copy()
    return sim


def test_observable_reset_is_available_to_same_slot_contract_design():
    sim = make_stub(observable=True)
    sim._synchronize_observable_state_reset()
    np.testing.assert_allclose(sim.H_design_state, [0.0, 0.4, 0.0])
    np.testing.assert_allclose(sim._obs_last, [0.0, 0.4, 0.0])
    assert sim._obs_cache_slot is None


def test_unobserved_reset_does_not_change_design_state():
    sim = make_stub(observable=False)
    sim._synchronize_observable_state_reset()
    np.testing.assert_allclose(sim.H_design_state, [0.8, 0.4, 0.7])
    assert sim._obs_cache_slot == 499
