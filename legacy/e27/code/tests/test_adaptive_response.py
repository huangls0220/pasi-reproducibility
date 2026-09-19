import numpy as np

from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator
from tests.test_mechanisms import _cfg


def test_adaptive_response_uses_past_only_and_shrinks_after_warmup():
    cfg = _cfg(T=12, N=8, M=4)
    cfg["uncertainty_aware"] = {
        "enabled": True,
        "cost_relative_bound": 0.0,
        "response_ratio_lower": 0.70,
        "response_ratio_upper": 1.30,
        "adaptive_response": {
            "enabled": True, "min_history": 8, "window": 40,
            "miscoverage_alpha": 0.05, "safety_margin": 0.02,
            "global_lower": 0.70, "global_upper": 1.30,
        },
    }
    data = generate_synthetic_episode(cfg, seed=101)
    data["tasks"] = data["tasks"].copy()
    data["tasks"]["kappa_design"] = data["tasks"]["kappa"]
    data["tasks"]["response_ratio_audit"] = 1.0
    result = Simulator(cfg, data, method="PASI", seed=101).run()
    slots = result["slot_log"]
    assert slots.iloc[0]["response_bound_lower"] == 0.70
    assert slots.iloc[0]["response_bound_upper"] == 1.30
    assert slots.iloc[-1]["response_bound_lower"] >= 0.98 - 1e-12
    assert slots.iloc[-1]["response_bound_upper"] <= 1.02 + 1e-12
    assert slots.iloc[0]["response_history_n"] > 0
    assert slots.iloc[0]["response_history_n_used"] == 0
    assert slots.iloc[0]["response_bound_mode"] == "global_warmup"
    assert slots.iloc[-1]["response_bound_mode"] == "adaptive_stable"
    assert "response_global_fallback_candidates" in slots
    assert "response_global_fallback_selected" in slots


def test_adaptive_global_guardrail_is_nested_and_has_fixed_endpoint():
    bounds = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfg = _cfg(T=2, N=8, M=4)
        cfg["uncertainty_aware"] = {
            "enabled": True,
            "response_ratio_lower": 0.70,
            "response_ratio_upper": 1.30,
            "adaptive_response": {
                "enabled": True, "min_history": 4, "window": 20,
                "miscoverage_alpha": 0.05, "safety_margin": 0.02,
                "global_lower": 0.70, "global_upper": 1.30,
                "global_guardrail_weight": weight,
            },
        }
        data = generate_synthetic_episode(cfg, seed=202)
        sim = Simulator(cfg, data, method="PASI", seed=202)
        sim._response_ratio_history = [1.0] * 8
        bounds.append(sim._adaptive_response_bounds())

    lowers = np.asarray([item[0] for item in bounds])
    uppers = np.asarray([item[1] for item in bounds])
    assert np.all(np.diff(lowers) <= 1e-12)
    assert np.all(np.diff(uppers) >= -1e-12)
    assert bounds[0] == (0.98, 1.02)
    assert bounds[-1] == (0.70, 1.30)


def test_adaptive_global_guardrail_rejects_invalid_weight():
    cfg = _cfg(T=2, N=8, M=4)
    cfg["uncertainty_aware"] = {
        "enabled": True,
        "response_ratio_lower": 0.70,
        "response_ratio_upper": 1.30,
        "adaptive_response": {
            "enabled": True, "global_lower": 0.70, "global_upper": 1.30,
            "global_guardrail_weight": 1.01,
        },
    }
    data = generate_synthetic_episode(cfg, seed=303)
    try:
        Simulator(cfg, data, method="PASI", seed=303)
    except ValueError as exc:
        assert "global_guardrail_weight" in str(exc)
    else:
        raise AssertionError("invalid guardrail weight was accepted")
