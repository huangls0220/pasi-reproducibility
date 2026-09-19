"""Vectorised evaluation vs scalar reference (auditability, §B.3).

The scalar implementation in src/contracts.py is the audited ground
truth; src/pair_eval.py must reproduce it to tight tolerance on random
instances for the PRIME mechanism (design side == true side).
"""

import numpy as np
import pytest

from src.contracts import evaluate_pair
from src.mechanisms import get_mechanism
from src.pair_eval import evaluate_pairs

RTOL = 1e-6
ATOL = 1e-7

FIELDS = [
    "a_min", "a_system", "a_reinforcement", "a_target", "lambda_required",
    "p_star", "D_star", "gamma_effective", "a_star", "implementation_gap",
    "base_payment", "expected_bonus", "expected_contract_cost",
    "experienced_utility", "perceived_utility", "execution_quality",
    "normalized_quality", "total_delay", "delta_H_predicted",
    "immediate_value", "reinforcement_value", "total_pair_value",
]


def _random_ctx(rng, n):
    stage_maint = rng.random(n) < 0.4
    ctx = {
        "L": rng.uniform(0.1, 1.0, n),
        "V": rng.uniform(1.0, 5.0, n),
        "q_bar": rng.uniform(0.9, 1.0, n),
        "q_min": rng.uniform(0.65, 0.85, n),
        "kappa": rng.uniform(1.0, 5.0, n),
        "input_size": rng.uniform(0.1, 2.0, n),
        "output_size": rng.uniform(0.05, 1.0, n),
        "deadline": rng.uniform(0.5, 6.0, n),
        "alpha": rng.uniform(0.05, 0.2, n),
        "beta": rng.uniform(0.02, 0.1, n),
        "omega_t": rng.uniform(0.0, 0.25, n),
        "zeta_t": rng.uniform(0.65, 1.0, n),
        "xi_t": rng.uniform(0.05, 0.12, n),
        "delta_t": rng.uniform(0.01, 0.05, n),
        "U_out": np.full(n, 0.01),
        "F_i_t": rng.uniform(3.0, 20.0, n),
        "communication_rate": rng.uniform(5.0, 20.0, n),
        "availability": rng.uniform(2.0, 10.0, n),
        "H_t": rng.uniform(0.0, 1.0, n),
        "in_cultivation": ~stage_maint,
        "last_p": np.where(rng.random(n) < 0.5, rng.uniform(0.05, 0.8, n), 0.0),
        "n_interactions": rng.integers(0, 50, n).astype(float),
        "p_min": 0.05, "p_max": 0.80, "D_bar": 10.0,
        "reinforcement_margin": 0.05, "Theta_M": 0.75, "delta_p_max": 0.05,
        "design_equals_true": True,
    }
    for k in ["omega", "zeta", "xi", "delta"]:
        ctx[f"{k}_d"] = ctx[f"{k}_t"].copy()
    ctx["H_d"] = ctx["H_t"].copy()
    return ctx


def _scalar_eval(ctx, i, eta_H):
    D_tr = (ctx["input_size"][i] + ctx["output_size"][i]) / ctx["communication_rate"][i]
    eff_dl = min(ctx["deadline"][i], ctx["availability"][i])
    return evaluate_pair(
        L=ctx["L"][i], V=ctx["V"][i], q_bar=ctx["q_bar"][i],
        q_min=ctx["q_min"][i], kappa=ctx["kappa"][i],
        alpha=ctx["alpha"][i], beta=ctx["beta"][i],
        omega=ctx["omega_t"][i], zeta=ctx["zeta_t"][i],
        xi=ctx["xi_t"][i], delta=ctx["delta_t"][i],
        H=ctx["H_t"][i],
        stage="cultivation" if ctx["in_cultivation"][i] else "maintenance",
        U_out=ctx["U_out"][i],
        p_min=ctx["p_min"], p_max=ctx["p_max"], D_bar=ctx["D_bar"],
        reinforcement_margin=ctx["reinforcement_margin"],
        Theta_M=ctx["Theta_M"], eta_H=eta_H,
        last_probability=ctx["last_p"][i], delta_p_max=ctx["delta_p_max"],
        F_i_t=ctx["F_i_t"][i], effective_deadline=eff_dl, D_tr=D_tr,
        input_size=ctx["input_size"][i], output_size=ctx["output_size"][i],
        communication_rate=ctx["communication_rate"][i],
    )


class TestVectorizedVsScalar:
    def test_prime_random_1000(self):
        """1000 random pairs: vectorised PRIME == scalar reference."""
        rng = np.random.default_rng(2026)
        n = 1000
        ctx = _random_ctx(rng, n)
        mech = get_mechanism("PRIME", {"prime": {"eta_H": 5.0}})
        vec = evaluate_pairs(mech, ctx)

        n_feasible = 0
        for i in range(n):
            sc = _scalar_eval(ctx, i, eta_H=5.0)
            v_feas = bool(vec["feasible"][i])
            s_feas = bool(sc["feasible_physical"] and sc["feasible_contract"])
            assert v_feas == s_feas, (
                f"pair {i}: feasibility mismatch vec={v_feas} scalar={s_feas} "
                f"(reason vec={vec['reason_code'][i]}, scalar={sc['infeasible_reason']})"
            )
            if not s_feas:
                continue
            n_feasible += 1
            for f in FIELDS:
                sv = sc[f]
                vv = float(vec[f][i])
                assert vv == pytest.approx(sv, rel=RTOL, abs=ATOL), (
                    f"pair {i} field {f}: vec={vv} scalar={sv}"
                )
        # The random ranges are chosen so a substantial share is feasible
        assert n_feasible > 300, f"only {n_feasible} feasible pairs — bad test setup"

    def test_infeasible_reasons_match(self):
        """Deadline/quality/a_min infeasibility flagged identically."""
        rng = np.random.default_rng(7)
        n = 400
        ctx = _random_ctx(rng, n)
        # Force a mix of infeasibilities
        ctx["deadline"][:100] = 0.01           # deadline infeasible
        ctx["q_min"][100:150] = 0.99           # q_min >= q_bar
        ctx["q_bar"][100:150] = 0.95
        ctx["F_i_t"][150:200] = 0.11           # a_min > 1 likely
        mech = get_mechanism("PRIME", {"prime": {"eta_H": 5.0}})
        vec = evaluate_pairs(mech, ctx)
        for i in range(200):
            sc = _scalar_eval(ctx, i, eta_H=5.0)
            s_feas = bool(sc["feasible_physical"] and sc["feasible_contract"])
            assert bool(vec["feasible"][i]) == s_feas, f"pair {i}"
