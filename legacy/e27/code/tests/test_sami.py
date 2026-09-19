"""SAMI (State-Aware Myopic Incentive) — hardened unit tests.

Phase 1.1 hardening: exact numerical equivalence checks at 1e-8 tolerance
(replacing the original 5% relative tolerance from Phase 1).

SAMI uses current H in contract design (Lambda = [C'(a_sys)/g'(a_sys) - omega*H]^+)
but targets only a_sys with no cultivation, no future reinforcement value,
no stage transitions, no recultivation, and no probability smoothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.datasets.synthetic import generate_synthetic_episode
from src.mechanisms import MechanismSpec, get_mechanism
from src.pair_eval import evaluate_pairs
from src.simulator import Simulator

_ATOL = 1e-8
_RTOL = 1e-6


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


# ── Single-pair deterministic context for exact comparison ──────────

def _single_pair_ctx(H_t=0.0, H_d=None, omega_t=0.0, omega_d=None,
                     zeta_t=1.0, zeta_d=None, in_cult=True, last_p=0.0,
                     n_inter=0):
    """Build a single-pair evaluation context with fully fixed parameters.

    All random variables are set to deterministic values so the comparison
    between mechanisms can be exact.
    """
    if H_d is None:
        H_d = H_t
    if omega_d is None:
        omega_d = omega_t
    if zeta_d is None:
        zeta_d = zeta_t
    n = 1
    return {
        "L": np.array([1.0], dtype=float),
        "V": np.array([3.0], dtype=float),
        "q_bar": np.array([0.95], dtype=float),
        "q_min": np.array([0.70], dtype=float),
        "kappa": np.array([3.0], dtype=float),
        "input_size": np.array([0.5], dtype=float),
        "output_size": np.array([0.3], dtype=float),
        "deadline": np.array([5.0], dtype=float),
        "alpha": np.array([0.1], dtype=float),
        "beta": np.array([0.05], dtype=float),
        "omega_t": np.array([omega_t], dtype=float),
        "zeta_t": np.array([zeta_t], dtype=float),
        "xi_t": np.array([0.08], dtype=float),
        "delta_t": np.array([0.03], dtype=float),
        "omega_d": np.array([omega_d], dtype=float),
        "zeta_d": np.array([zeta_d], dtype=float),
        "xi_d": np.array([0.08], dtype=float),
        "delta_d": np.array([0.03], dtype=float),
        "H_t": np.array([H_t], dtype=float),
        "H_d": np.array([H_d], dtype=float),
        "U_out": np.array([0.01], dtype=float),
        "F_i_t": np.array([10.0], dtype=float),
        "communication_rate": np.array([5.0], dtype=float),
        "availability": np.array([100.0], dtype=float),
        "in_cultivation": np.array([in_cult], dtype=bool),
        "last_p": np.array([last_p], dtype=float),
        "n_interactions": np.array([n_inter], dtype=float),
        "p_min": 0.05,
        "p_max": 0.80,
        "D_bar": 8.0,
        "reinforcement_margin": 0.05,
        "Theta_M": 0.75,
        "delta_p_max": 0.05,
        "design_equals_true": True,
    }


# ── Spec tests (unchanged from Phase 1, still required) ────────────

class TestSAMISpec:
    def test_sami_uses_current_H(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.design_H == "true"

    def test_sami_target_is_always_system_effort(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.target_mode == "system"

    def test_sami_has_no_reinforcement_target(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.target_mode != "prime"

    def test_sami_eta_H_is_zero(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.eta_H == 0.0

    def test_sami_has_no_stage_transition(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.use_stage_logic is False

    def test_sami_has_no_recultivation(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.allow_recultivation is False

    def test_sami_has_no_probability_smoothing(self):
        spec = get_mechanism("SAMI", _cfg())
        assert spec.maintenance_smoothing is False


# ── Hardened exact equivalence tests ───────────────────────────────

class TestSAMIExactEquivalence:
    """Deterministic single-pair comparisons at 1e-8 absolute tolerance."""

    def test_sami_equals_moi_exactly_when_H_zero_and_rational(self):
        """When H=0, omega=0, zeta=1: SAMI and MOI are NUMERICALLY IDENTICAL.

        Both compute Lambda = C'/g' (no omega*H credit),
        target a_sys, use P1 contract identically.
        Expected contract compared, NOT random realized bonus.
        """
        ctx = _single_pair_ctx(H_t=0.0, omega_t=0.0, zeta_t=1.0)

        # SAMI spec: design_H="true" but H_d=0, so effectively same as MOI
        sami_spec = MechanismSpec(
            name="SAMI", contract_mode="p1", design_H="true",
            target_mode="system", eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False, allow_recultivation=False,
        )
        # MOI spec: design_H="zero"
        moi_spec = MechanismSpec(
            name="MOI", contract_mode="p1", design_H="zero",
            target_mode="system", eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False,
        )

        # Also set H_d=0 so SAMI's H has no effect
        ctx_h0 = dict(ctx)
        ctx_h0["H_d"] = np.array([0.0], dtype=float)

        ev_sami = evaluate_pairs(sami_spec, ctx_h0)
        ev_moi = evaluate_pairs(moi_spec, ctx)

        # Both must be feasible
        assert ev_sami["feasible"][0], f"SAMI infeasible: {ev_sami['reason_code'][0]}"
        assert ev_moi["feasible"][0], f"MOI infeasible: {ev_moi['reason_code'][0]}"

        # Exact numerical equality
        for field in ["a_target", "lambda_required", "p_star", "D_star",
                       "base_payment", "a_star", "expected_contract_cost"]:
            v_s = float(ev_sami[field][0])
            v_m = float(ev_moi[field][0])
            assert abs(v_s - v_m) <= _ATOL, (
                f"{field}: SAMI={v_s:.15e} MOI={v_m:.15e} diff={abs(v_s-v_m):.2e}"
            )

    def test_sami_equals_prime_when_active_modules_are_disabled(self):
        """SAMI ≈ PRIME when all active modules are disabled.

        Construct a scenario where:
        - a_rein <= a_sys (no cultivation target elevation)
        - in_cultivation = True (no maintenance smoothing)
        - eta_H = 0 (no future reinforcement value)
        - Same H, no stage transitions, no recultivation
        """
        ctx = _single_pair_ctx(H_t=0.3, omega_t=0.15, zeta_t=0.80)

        sami_spec = get_mechanism("SAMI", _cfg())

        # Manually construct a PRIME-equivalent with ALL modules disabled
        prime_inactive_spec = MechanismSpec(
            name="PRIME-inactive", contract_mode="p1", design_H="true",
            target_mode="system",  # system only, no cultivation target
            eta_H=0.0,             # no future reinforcement value
            maintenance_smoothing=False,
            use_stage_logic=False,
            allow_recultivation=False,
        )

        ev_sami = evaluate_pairs(sami_spec, ctx)
        ev_prime_off = evaluate_pairs(prime_inactive_spec, ctx)

        assert ev_sami["feasible"][0], f"SAMI infeasible: {ev_sami['reason_code'][0]}"
        assert ev_prime_off["feasible"][0], f"PRIME-inactive infeasible: {ev_prime_off['reason_code'][0]}"

        for field in ["a_target", "lambda_required", "p_star", "D_star",
                       "base_payment", "a_star", "expected_contract_cost",
                       "total_pair_value"]:
            v_s = float(ev_sami[field][0])
            v_p = float(ev_prime_off[field][0])
            assert abs(v_s - v_p) <= _ATOL, (
                f"{field}: SAMI={v_s:.15e} PRIME-inactive={v_p:.15e} diff={abs(v_s-v_p):.2e}"
            )

    def test_sami_Lambda_uses_H_while_MOI_does_not(self):
        """With H>0, SAMI's Lambda < MOI's Lambda (omega*H credit in design).

        The simulator zeros H_d for MOI (design_H="zero"), but evaluate_pairs
        receives H_d directly from context — so we must explicitly pass H_d=0
        for MOI to replicate what the simulator does.
        """
        ctx_sami = _single_pair_ctx(H_t=0.5, H_d=0.5, omega_t=0.2, omega_d=0.2,
                                    zeta_t=0.85)
        ctx_moi = _single_pair_ctx(H_t=0.5, H_d=0.0, omega_t=0.2, omega_d=0.2,
                                   zeta_t=0.85)

        sami_spec = get_mechanism("SAMI", _cfg())
        moi_spec = get_mechanism("MOI", _cfg())

        ev_sami = evaluate_pairs(sami_spec, ctx_sami)
        ev_moi = evaluate_pairs(moi_spec, ctx_moi)

        assert ev_sami["feasible"][0], f"SAMI infeasible: {ev_sami['reason_code'][0]}"
        assert ev_moi["feasible"][0], f"MOI infeasible: {ev_moi['reason_code'][0]}"

        lam_s = float(ev_sami["lambda_required"][0])
        lam_m = float(ev_moi["lambda_required"][0])

        # SAMI's Lambda must be strictly smaller: Lambda_SAMI = C'/g' - omega*H
        # vs Lambda_MOI = C'/g'
        assert lam_s < lam_m - _ATOL, (
            f"SAMI Lambda ({lam_s:.6f}) should be < MOI Lambda ({lam_m:.6f}) "
            f"due to omega*H credit"
        )
        # Verify: Lambda_MOI - Lambda_SAMI ≈ omega_d * H_d
        expected_diff = 0.2 * 0.5  # omega*H = 0.10
        actual_diff = lam_m - lam_s
        assert abs(actual_diff - expected_diff) <= _ATOL, (
            f"Lambda diff ({actual_diff:.6f}) should be omega*H ({expected_diff:.6f})"
        )


# ── SAMI environmental H update test ───────────────────────────────

class TestSAMIHUpdate:
    """SAMI updates environmental H without using stage logic."""

    def test_sami_updates_environmental_H_without_using_stage_logic(self):
        """SAMI: H updates by path equation, but stage never changes."""
        cfg = _cfg(T=40)
        cfg["providers"]["initial_H"] = 0.2
        cfg["providers"]["xi"] = [0.08, 0.08]
        cfg["providers"]["delta"] = [0.03, 0.03]
        cfg["providers"]["omega"] = [0.10, 0.10]

        data = generate_synthetic_episode(cfg, seed=201)
        res = Simulator(cfg, data, method="SAMI", seed=201).run()

        # H must be updated
        prov = res["provider_log"]
        assigned = prov[prov["assigned"]]
        assert len(assigned) > 0, "Need assigned providers to verify H update"
        # H should change for at least some providers
        h_changes = (assigned["H_before"] != assigned["H_after"])
        assert h_changes.any(), "H must be updated for assigned providers"

        # future_reinforcement_value always 0
        sel_pairs = res["pair_log"][res["pair_log"]["selected"]]
        if len(sel_pairs) > 0 and "reinforcement_value" in sel_pairs.columns:
            assert (sel_pairs["reinforcement_value"] == 0).all(), (
                "SAMI reinforcement_value must always be 0"
            )

        # Stage never switches
        assert prov["recultivation_flag"].sum() == 0
        assert prov["maintenance_entry_flag"].sum() == 0
        assert (prov["stage_after"] == "cultivation").all()

        # No violations
        assert res["diagnostics"]["ir_violations"] == 0
        assert res["diagnostics"]["budget_violations"] == 0


# ── Runtime validation tests (retained from Phase 1) ────────────────

class TestSAMIRuntime:
    def test_sami_response_consistent_IR(self):
        cfg = _cfg(T=40)
        data = generate_synthetic_episode(cfg, seed=103)
        res = Simulator(cfg, data, method="SAMI", seed=103).run()
        assert res["diagnostics"]["ir_violations"] == 0

    def test_sami_budget_feasible(self):
        cfg = _cfg(T=40)
        data = generate_synthetic_episode(cfg, seed=104)
        res = Simulator(cfg, data, method="SAMI", seed=104).run()
        assert res["diagnostics"]["budget_violations"] == 0

    def test_sami_induced_effort_meets_target(self):
        cfg = _cfg(T=60)
        data = generate_synthetic_episode(cfg, seed=105)
        res = Simulator(cfg, data, method="SAMI", seed=105).run()
        sel = res["pair_log"][res["pair_log"]["selected"]]
        if len(sel) > 0:
            assert (sel["implementation_gap"] >= -1e-7).all()
