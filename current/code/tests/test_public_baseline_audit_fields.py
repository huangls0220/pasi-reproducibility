"""The public-baseline reserve bid is logged without changing decisions."""

import numpy as np

from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator
from tests.test_mechanisms import _cfg


def test_effective_reserve_cost_identity_is_logged():
    cfg = _cfg(T=3, N=6, M=2)
    cfg["simulation"]["log_level"] = "full"
    data = generate_synthetic_episode(cfg, seed=3701)
    result = Simulator(cfg, data, method="PASI", seed=3701).run()
    pairs = result["pair_log"]
    required = {
        "realized_effort_cost",
        "runtime_path_value",
        "outside_option",
        "effective_reserve_cost",
    }
    assert required.issubset(pairs.columns)
    legal = pairs[pairs["feasible_contract"].astype(bool)]
    expected = np.maximum(
        0.0,
        legal["realized_effort_cost"].to_numpy()
        - legal["runtime_path_value"].to_numpy()
        + legal["outside_option"].to_numpy(),
    )
    assert np.allclose(legal["effective_reserve_cost"], expected)
