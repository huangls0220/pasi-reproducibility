"""Mechanism dispatch + baseline behaviour tests (Section 7, 11.8).

Verifies:
- every method resolves and runs end-to-end on a tiny episode
- same seed -> identical dataset across methods (fair comparison §2.5)
- MOI does NOT modify provider behavioural parameters (§7.5 fix)
- FR uses fixed contract parameters; declined pairs never assigned
- LDI probability decays with interactions
- RAI reputation updates and stays in [0,1]
- PRIME-Fixed switches stage after the configured interaction count
- PRIME-w/o-RC never re-cultivates
"""

import numpy as np
import pandas as pd
import pytest

from src.datasets.synthetic import generate_synthetic_episode
from src.mechanisms import ABLATION_METHODS, ALL_METHODS, get_mechanism
from src.simulator import Simulator


def _cfg(T=25, N=10, M=8, **over):
    cfg = {
        "simulation": {"T": T, "log_level": "full"},
        "providers": {
            "N_mean": N, "behavioral_fraction": 1.0,
            "max_processing_rate": [8.0, 15.0],
            "alpha": [0.05, 0.15], "beta": [0.02, 0.08],
            "zeta": [0.65, 0.90], "omega": [0.05, 0.20],
            "xi": [0.06, 0.10], "delta": [0.02, 0.04],
            "outside_option": 0.01, "initial_H": 0.1,
        },
        "tasks": {
            "M_mean": M, "cpu_cycles": [0.2, 1.0],
            "input_size": [0.1, 1.5], "output_size": [0.05, 0.8],
            "deadline_factor": [1.2, 2.5], "min_quality": [0.65, 0.85],
            "q_bar": [0.90, 0.95], "kappa": [1.5, 4.0],
            "value_base": [1.0, 4.0],
        },
        "contract": {"p_min": 0.05, "p_max": 0.80, "D_bar": 8.0,
                     "reinforcement_margin": 0.05},
        "path_state": {"Theta_M": 0.75, "Theta_C": 0.55, "s_M": 0.80,
                       "s_C": 0.65, "K": 5, "delta_p_max": 0.05},
        "matching": {"budget_ratio": 0.70, "max_iter": 50},
        "prime": {"eta_H": 5.0},
        "baselines": {
            "fixed_reward": {"base_payment": 0.05, "p": 0.4, "D": 3.0},
            "dynamic_pricing": {"p": 0.4},
            "reputation_aware": {"p": 0.4, "ewma_alpha": 0.2},
            "linear_decay": {"p_start": 0.8, "p_end": 0.1,
                             "decay_interactions": 10, "D": 2.0},
            "prime_fixed": {"cultivation_interactions": 5},
        },
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


ALL = ALL_METHODS + [m for m in ABLATION_METHODS if m != "PRIME"]


class TestDispatch:
    def test_all_methods_resolve(self):
        cfg = _cfg()
        for m in ALL + ["PRIME-R"]:
            spec = get_mechanism(m, cfg)
            assert spec.name == m or spec.name in m

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            get_mechanism("NOPE", {})

    def test_public_mapped_baseline_targets(self):
        cfg = _cfg()
        assert get_mechanism("QUAC-I-MAPPED", cfg).target_mode == "system"
        assert get_mechanism("QIM-E-MAPPED", cfg).target_mode == "qos"


class TestAllMethodsRun:
    @pytest.mark.parametrize("method", ALL)
    def test_method_smoke(self, method):
        cfg = _cfg()
        data = generate_synthetic_episode(cfg, seed=42)
        sim = Simulator(cfg, data, method=method, seed=42)
        res = sim.run()
        diag = res["diagnostics"]
        assert diag["execution_status"] == "completed"
        assert diag["nan_rows"] == diag["inf_rows"] == diag["budget_violations"] == 0
        if method == "LDI":
            # This fixture really misses QoS twice; successful execution
            # must not be confused with constraint satisfaction.
            assert diag["qos_violations"] == 2
            assert diag["status"] == "failed"
        else:
            assert diag["status"] == "ok", f"{method}: {diag['issues']}"
        assert res["summary"]["num_assigned"] >= 0
        # PRIME and MOI must assign a healthy share
        if method in ("PRIME", "MOI"):
            assert res["summary"]["num_assigned"] > 0

    def test_same_seed_same_dataset(self):
        cfg = _cfg()
        d1 = generate_synthetic_episode(cfg, seed=7)
        d2 = generate_synthetic_episode(cfg, seed=7)
        pd.testing.assert_frame_equal(d1["tasks"], d2["tasks"])
        pd.testing.assert_frame_equal(d1["provider_static"], d2["provider_static"])


class TestMOIBehaviourPreserved:
    def test_moi_keeps_true_omega(self):
        """§7.5: H=0 only in DESIGN — provider behaviour untouched."""
        cfg = _cfg()
        data = generate_synthetic_episode(cfg, seed=5)
        true_omega = data["provider_static"]["omega"].to_numpy()
        sim = Simulator(cfg, data, method="MOI", seed=5)
        assert np.allclose(sim.s_omega, true_omega), "MOI must not zero behavioural omega"
        res = sim.run()
        sel = res["pair_log"][res["pair_log"]["selected"]]
        # Design ignores H -> Lambda has no omega*H credit, so realised
        # a_star (with true omega*H) weakly over-delivers.
        assert (sel["implementation_gap"] >= -1e-7).all()

    def test_moi_lambda_ge_prime_lambda(self):
        """With H>0, PRIME's Lambda (credited omega*H) <= MOI's Lambda
        on identical maintenance-stage pairs (statistical check)."""
        cfg = _cfg(T=40)
        data = generate_synthetic_episode(cfg, seed=11)
        r_pr = Simulator(cfg, data, method="PRIME", seed=11).run()
        r_moi = Simulator(cfg, data, method="MOI", seed=11).run()
        lp = r_pr["pair_log"]; lm = r_moi["pair_log"]
        key = ["slot", "provider_id", "task_id"]
        merged = lp.merge(lm, on=key, suffixes=("_pr", "_moi"))
        both = merged[(merged["feasible_contract_pr"]) & (merged["feasible_contract_moi"])]
        late = both[both["slot"] > 20]
        # PRIME may target HIGHER effort in cultivation (a_rein), so compare
        # only pairs with equal targets.
        same_t = late[np.isclose(late["a_target_pr"], late["a_target_moi"], atol=1e-9)]
        assert len(same_t) > 50
        assert (same_t["lambda_required_pr"] <= same_t["lambda_required_moi"] + 1e-9).all()


class TestFixedReward:
    def test_fixed_contract_params(self):
        cfg = _cfg()
        data = generate_synthetic_episode(cfg, seed=3)
        res = Simulator(cfg, data, method="FR", seed=3).run()
        sel = res["pair_log"][res["pair_log"]["selected"]]
        if len(sel):
            assert np.allclose(sel["p_star"], 0.4)
            assert np.allclose(sel["D_star"], 3.0)
            assert np.allclose(sel["base_payment"], 0.05)

    def test_declined_pairs_not_assigned(self):
        cfg = _cfg()
        # Absurdly low fixed payment: many providers decline
        cfg["baselines"]["fixed_reward"] = {"base_payment": 0.0, "p": 0.05, "D": 0.1}
        data = generate_synthetic_episode(cfg, seed=3)
        res = Simulator(cfg, data, method="FR", seed=3).run()
        log = res["pair_log"]
        declined = log[log["infeasible_reason"] == "ir_declined"]
        assert len(declined) > 0, "expected IR-declined pairs at absurd payment"
        assert not declined["selected"].any()
        assert res["summary"]["ir_violations"] == 0


class TestLDI:
    def test_probability_decays(self):
        cfg = _cfg(T=40)
        data = generate_synthetic_episode(cfg, seed=9)
        res = Simulator(cfg, data, method="LDI", seed=9).run()
        pl = res["provider_log"]
        heavy = pl.groupby("provider_id")["assigned"].sum()
        pid = heavy.idxmax()
        assert heavy.max() >= 12, "test needs a heavily assigned provider"
        rows = pl[(pl["provider_id"] == pid) & pl["assigned"]]
        p_first = rows.iloc[0]["p_last"] if rows.iloc[0]["p_last"] > 0 else 0.8
        p_late = rows.iloc[-1]["p_last"]
        assert p_late <= 0.1 + 1e-9 + 0.0, (
            f"p should decay to p_end after decay_interactions: {p_late}"
        )
        assert p_late < p_first or p_first <= 0.1 + 1e-9


class TestRAI:
    def test_reputation_updates_and_bounded(self):
        cfg = _cfg(T=30)
        data = generate_synthetic_episode(cfg, seed=13)
        res = Simulator(cfg, data, method="RAI", seed=13).run()
        pl = res["provider_log"]
        assert ((pl["reputation"] >= 0) & (pl["reputation"] <= 1)).all()
        assigned_prov = pl[pl["assigned"]]["provider_id"].unique()
        final = pl[pl["provider_id"].isin(assigned_prov)].groupby("provider_id")["reputation"].last()
        assert (np.abs(final - 0.5) > 1e-6).any(), "reputation never moved"


class TestPrimeVariants:
    def test_prime_fixed_switches_by_count(self):
        cfg = _cfg(T=40)
        data = generate_synthetic_episode(cfg, seed=17)
        res = Simulator(cfg, data, method="PRIME-Fixed", seed=17).run()
        pl = res["provider_log"]
        exceeded = pl[pl["n_interactions"] >= 5]
        assert len(exceeded) > 0
        assert (exceeded["stage_after"] == "maintenance").all()

    def test_wo_rc_never_recultivates(self):
        cfg = _cfg(T=60)
        data = generate_synthetic_episode(cfg, seed=19)
        res = Simulator(cfg, data, method="PRIME-w/o-RC", seed=19).run()
        assert res["summary"]["recultivation_events"] == 0

    def test_wo_pd_zero_omega(self):
        cfg = _cfg()
        data = generate_synthetic_episode(cfg, seed=21)
        sim = Simulator(cfg, data, method="PRIME-w/o-PD", seed=21)
        assert np.allclose(sim.s_omega, 0.0)
        res = sim.run()
        assert res["diagnostics"]["status"] == "ok"

    def test_wo_pw_zeta_one(self):
        cfg = _cfg()
        data = generate_synthetic_episode(cfg, seed=23)
        sim = Simulator(cfg, data, method="PRIME-w/o-PW", seed=23)
        assert np.allclose(sim.s_zeta, 1.0)
        res = sim.run()
        sel = res["pair_log"][res["pair_log"]["selected"]]
        # zeta=1 -> perceived == experienced utility
        assert np.allclose(sel["perceived_utility"], sel["experienced_utility"], atol=1e-9)


class TestEstimationError:
    def test_perturbation_same_across_methods(self):
        cfg = _cfg()
        cfg["estimation_error"] = {"level": 0.3, "params": "all", "mode": "random"}
        data = generate_synthetic_episode(cfg, seed=31)
        s1 = Simulator(cfg, data, method="PRIME", seed=31)
        s2 = Simulator(cfg, data, method="MOI", seed=31)
        assert np.allclose(s1.d_omega, s2.d_omega)
        assert np.allclose(s1.d_zeta, s2.d_zeta)
        assert not np.allclose(s1.d_omega, s1.s_omega), "perturbation missing"

    def test_error_run_reports_constraint_failure_without_execution_error(self):
        cfg = _cfg()
        cfg["estimation_error"] = {"level": 0.5, "params": "omega", "mode": "over"}
        data = generate_synthetic_episode(cfg, seed=33)
        res = Simulator(cfg, data, method="PRIME", seed=33).run()
        # Deliberate estimation error is a completed run with negative
        # outcomes, not a reason to relabel those outcomes as successful.
        diag = res["diagnostics"]
        assert diag["execution_status"] == "completed"
        assert diag["status"] == "failed"
        assert diag["qos_violations"] == 2
        assert diag["target_violations"] == 172
        assert diag["nan_rows"] == diag["inf_rows"] == diag["budget_violations"] == 0

    def test_prime_r_more_conservative(self):
        cfg = _cfg()
        cfg["estimation_error"] = {"level": 0.4, "params": "omega", "mode": "over"}
        data = generate_synthetic_episode(cfg, seed=35)
        plain = Simulator(cfg, data, method="PRIME", seed=35)
        robust = Simulator(cfg, data, method="PRIME-R", seed=35)
        assert (robust.d_omega <= plain.d_omega + 1e-12).all()
