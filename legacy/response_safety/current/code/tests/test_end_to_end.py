"""End-to-end smoke test: small-scale run with PRIME and all baselines (Section 11.8)."""

import pytest

from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator


def _small_config() -> dict:
    return {
        "simulation": {"T": 20, "log_level": "full", "checkpoint_interval": 10},
        "providers": {
            "N_mean": 10,
            "max_processing_rate": [8.0, 15.0],
            "alpha": [0.05, 0.15],
            "beta": [0.02, 0.08],
            "zeta": [0.65, 0.90],
            "omega": [0.05, 0.20],
            "xi": [0.06, 0.10],
            "delta": [0.02, 0.04],
            "outside_option": 0.01,
            "initial_H": 0.1,
            "initial_stage": "cultivation",
        },
        "tasks": {
            "M_mean": 8,
            "cpu_cycles": [0.2, 1.0],
            "input_size": [0.1, 1.5],
            "output_size": [0.05, 0.8],
            "deadline_factor": [1.2, 2.5],
            "min_quality": [0.65, 0.85],
            "q_bar": [0.90, 0.95],
            "kappa": [1.5, 4.0],
            "value_base": [1.0, 4.0],
        },
        "contract": {
            "p_min": 0.05, "p_max": 0.80, "D_bar": 8.0,
            "reinforcement_margin": 0.05,
        },
        "path_state": {
            "Theta_M": 0.75, "Theta_C": 0.55,
            "s_M": 0.80, "s_C": 0.65,
            "K": 5, "delta_p_max": 0.05,
        },
        "matching": {
            "budget_ratio": 0.70,
            "max_iter": 50,
            "budget_tol": 1e-4,
            "stagnation_limit": 5,
            "initial_lambda_B": 0.1,
            "step_scale": 0.1,
        },
        "prime": {
            "eta_H": 5.0,
            "use_path_dependence": True,
            "use_probability_weighting": True,
        },
    }


class TestEndToEnd:
    """10 providers, 8 tasks, 20 slots — PRIME and baselines."""

    def test_synthetic_data_generation(self):
        """Verify synthetic data can be generated."""
        cfg = _small_config()
        cfg["simulation"]["T"] = 20
        data = generate_synthetic_episode(cfg, seed=42, pattern="stationary")
        assert "tasks" in data
        assert "providers" in data
        assert "provider_static" in data
        assert len(data["tasks"]) > 0
        assert len(data["providers"]) > 0
        assert len(data["provider_static"]) == 10

    def test_prime_smoke(self):
        """PRIME method: 20-slot smoke test."""
        cfg = _small_config()
        data = generate_synthetic_episode(cfg, seed=42, pattern="stationary")

        sim = Simulator(cfg, data, method="PRIME", seed=42)
        result = sim.run()

        assert result["diagnostics"]["status"] == "ok", (
            f"Diagnostics failed: {result['diagnostics']['issues']}"
        )
        assert result["summary"]["num_assigned"] > 0, "No tasks assigned"
        assert result["summary"]["HQR"] is not None
        assert len(result["pair_log"]) > 0
        assert len(result["slot_log"]) == 20
        assert len(result["provider_log"]) > 0

    def test_moi_smoke(self):
        """Myopic Optimal Incentive method: 20-slot smoke test."""
        cfg = _small_config()
        data = generate_synthetic_episode(cfg, seed=42, pattern="stationary")

        sim = Simulator(cfg, data, method="MOI", seed=42)
        result = sim.run()

        assert result["diagnostics"]["status"] == "ok", (
            f"Diagnostics failed: {result['diagnostics']['issues']}"
        )
        assert result["summary"]["num_assigned"] > 0

    def test_fixed_reward_smoke(self):
        """Fixed Reward baseline: 20-slot smoke test."""
        cfg = _small_config()
        data = generate_synthetic_episode(cfg, seed=42, pattern="stationary")

        sim = Simulator(cfg, data, method="FR", seed=42)
        result = sim.run()

        assert result["diagnostics"]["status"] == "ok", (
            f"Diagnostics failed: {result['diagnostics']['issues']}"
        )

    def test_reproducibility(self):
        """Same config + seed → identical results."""
        cfg = _small_config()
        data1 = generate_synthetic_episode(cfg, seed=99, pattern="stationary")
        data2 = generate_synthetic_episode(cfg, seed=99, pattern="stationary")

        sim1 = Simulator(cfg, data1, method="PRIME", seed=99)
        result1 = sim1.run()

        sim2 = Simulator(cfg, data2, method="PRIME", seed=99)
        result2 = sim2.run()

        # Key metrics should match
        assert result1["summary"]["num_assigned"] == result2["summary"]["num_assigned"]
        assert abs(result1["summary"]["cumulative_payment"] - result2["summary"]["cumulative_payment"]) < 1e-9
        assert abs(result1["summary"]["platform_utility"] - result2["summary"]["platform_utility"]) < 1e-9
