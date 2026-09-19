"""Semantic mapping tests for the public QUAC-F baseline."""

import numpy as np

from src.mechanisms import get_mechanism
from src.pair_eval import C_vec, evaluate_pairs, g_vec


def _ctx():
    a = np.array([1.0])
    z = np.array([0.0])
    return {
        "L": a, "V": np.array([8.0]), "q_bar": np.array([1.0]),
        "q_min": np.array([0.2]), "kappa": np.array([1.4]),
        "input_size": z, "output_size": z, "deadline": np.array([10.0]),
        "alpha": np.array([0.2]), "beta": np.array([0.4]),
        "alpha_d": np.array([0.2]), "beta_d": np.array([0.4]),
        "kappa_d": np.array([1.4]),
        "omega_t": z, "zeta_t": a, "xi_t": np.array([0.1]),
        "delta_t": np.array([0.1]), "omega_d": z, "zeta_d": a,
        "xi_d": np.array([0.1]), "delta_d": np.array([0.1]),
        "H_t": z, "H_d": z, "U_out": np.array([0.05]),
        "F_i_t": np.array([10.0]), "communication_rate": np.array([10.0]),
        "availability": np.array([10.0]), "in_cultivation": np.array([False]),
        "last_p": z, "n_interactions": z, "p_min": 0.05, "p_max": 0.9,
        "D_bar": 100.0, "reinforcement_margin": 0.0, "Theta_M": 0.8,
        "delta_p_max": 0.1, "design_equals_true": True,
    }


def test_quac_f_registry_and_public_contract_semantics():
    spec = get_mechanism("QUAC-F", {})
    assert spec.contract_mode == "risk_neutral"
    assert spec.design_H == "zero"
    assert spec.target_mode == "system"
    assert spec.base_mode == "quac_ir"
    assert spec.enforce_target
    assert spec.p_fixed == 1.0


def test_quac_f_linear_quality_payment_binds_ir_at_design_optimum():
    ctx = _ctx()
    ev = evaluate_pairs(get_mechanism("QUAC-F", {}), ctx)
    assert ev["feasible"][0]
    assert abs(ev["p_star"][0] - 1.0) < 1e-12
    assert abs(ev["implementation_gap_design"][0]) < 1e-7

    effort = ev["a_star_design"]
    design_quality = g_vec(effort, ctx["kappa_d"])
    design_cost = C_vec(effort, ctx["alpha_d"], ctx["beta_d"], ctx["L"])
    expected = ctx["U_out"] + design_cost
    realised_contract = ev["base_payment"] + ev["D_star"] * design_quality
    assert np.allclose(realised_contract, expected, atol=1e-9, rtol=0.0)
    assert np.allclose(ev["experienced_utility"], ctx["U_out"],
                       atol=1e-9, rtol=0.0)
