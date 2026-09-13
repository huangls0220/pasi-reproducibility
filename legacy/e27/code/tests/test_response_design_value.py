from unittest.mock import patch

import numpy as np

from src.datasets.synthetic import generate_synthetic_episode
from src.pair_eval import evaluate_pairs as real_evaluate_pairs
from src.simulator import Simulator
from tests.test_mechanisms import _cfg


def test_contract_uses_frozen_design_response_value():
    cfg = _cfg(T=2, N=5, M=2)
    cfg["simulation"]["log_level"] = "selected"
    data = generate_synthetic_episode(cfg, seed=404)
    data["tasks"] = data["tasks"].copy()
    data["tasks"]["kappa_design"] = 1.75
    data["tasks"]["kappa"] = 0.70

    captured = []

    def capture(mechanism, ctx):
        captured.append(ctx["kappa_d"].copy())
        return real_evaluate_pairs(mechanism, ctx)

    with patch("src.simulator.evaluate_pairs", side_effect=capture):
        Simulator(cfg, data, method="PASI", seed=404).run()

    assert captured
    assert all(np.allclose(values, 1.75) for values in captured)
