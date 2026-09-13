"""Mechanism registry — unified BaseMechanism interface (Section 7).

Every method (PRIME, baselines, ablations) is described by a MechanismSpec
that parameterises the SAME audited evaluation pipeline (src/pair_eval.py):

  contract_mode:
    "p1"           – transparent probability–bonus contract P1 (optimal
                     endpoint solution, Section 5.9).  Used by PRIME / MOI.
    "risk_neutral" – adaptive price with fixed p and D = Lambda / p
                     (platform ignores probability weighting).  Used by
                     DP / RAI.
    "fixed"        – fixed (p, D) for every pair.  Used by FR.
    "ldi"          – fixed D, per-provider linearly decaying p.  Used by LDI.

  design_H:
    "true" – platform uses the provider's true path state H in contract
             design (Lambda, base payment).  PRIME family.
    "zero" – platform ignores H in design (Lambda and base payment computed
             with H=0).  Provider BEHAVIOUR is unchanged: a_star and the
             experienced utility always use the true omega*H.  This is the
             §7.5 requirement ("contract design and matching use H=0") and
             keeps behaviour parameters identical across methods (§2.5).

  target_mode:
    "prime"  – cultivation: max(a_system, a_reinforcement); maintenance: a_system
    "system" – always a_system (myopic-efficient)
    "none"   – no target (contract not derived from a target; Lambda unused)

  base_mode:
    "response_consistent" – b = [U_out + C(a*) - pDg(a*) - omega_d*H_d*g(a*)]^+
                            with DESIGN-side omega_d*H_d (true for PRIME,
                            0 for H-ignoring baselines).
    "fixed"               – fixed b for all pairs (FR).  Providers decline
                             pairs whose experienced utility < U_out
                             (participation constraint).
    "quac_ir"             – signed QUAC-F intercept A that makes IR bind at
                             the intended quality; total intended payment is
                             nonnegative even when A itself is negative.

  enforce_target: if True, a_star < a_target - 1e-7 marks the pair
             infeasible (PRIME / MOI guarantee implementability).  Methods
             that allow real quality degradation (FR/DP/RAI/LDI) set False.

All provider-side quantities (a_star response, experienced utility, path
state update) ALWAYS use the true behavioural parameters — mechanisms only
differ in what the PLATFORM computes.  Estimation error (Experiment F) is
expressed by perturbing the design-side copies of omega/zeta/xi/delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MechanismSpec:
    name: str
    contract_mode: str = "p1"            # p1 | risk_neutral | fixed | ldi
    design_H: str = "true"               # true | zero
    target_mode: str = "prime"           # prime | system | none
    base_mode: str = "response_consistent"  # response_consistent | fixed | quac_ir
    enforce_target: bool = True
    eta_H: Optional[float] = None        # None -> use cfg prime.eta_H
    maintenance_smoothing: bool = True   # PRIME probability smoothing (5.13)
    use_stage_logic: bool = True         # track cultivation/maintenance stages
    allow_recultivation: bool = True     # w/o-RC sets False
    fixed_cultivation_count: Optional[int] = None  # PRIME-Fixed: switch after N assigned
    reputation_weighting: bool = False   # RAI matching weight multiplier
    # fixed-contract parameters (FR / LDI / risk-neutral p)
    p_fixed: Optional[float] = None
    D_fixed: Optional[float] = None
    b_fixed: Optional[float] = None
    # LDI schedule
    p_start: Optional[float] = None
    p_end: Optional[float] = None
    decay_interactions: Optional[int] = None
    # behaviour overrides for ablations of the MODEL itself
    behavior_omega_zero: bool = False    # w/o-PD: no path-dependent utility at all
    behavior_zeta_one: bool = False      # w/o-PW: no probability weighting at all
    # Quality-Matched protocol knob (§13.3): scales the external incentive
    # (D, and the fixed base for FR) of fixed/risk-neutral/ldi contracts.
    incentive_scale: float = 1.0
    notes: str = ""


def _baseline_cfg(cfg: dict, key: str) -> dict:
    return (cfg.get("baselines", {}) or {}).get(key, {}) or {}


def get_mechanism(method: str, cfg: dict) -> MechanismSpec:
    """Resolve a method name into a fully-populated MechanismSpec."""
    prime_cfg = cfg.get("prime", {}) or {}
    eta_H = float(prime_cfg.get("eta_H", 5.0))

    if method == "PRIME":
        return MechanismSpec(name="PRIME", eta_H=eta_H)

    # ── Ablation variants (Section 7.6) ──────────────────────
    if method == "PRIME-w/o-PD":
        # omega = 0 in the MODEL: no path-dependent utility anywhere.
        # H is still logged but enters neither design nor behaviour.
        return MechanismSpec(
            name=method, eta_H=eta_H, behavior_omega_zero=True,
            notes="omega=0 (model ablation); H logged but utility-free",
        )
    if method == "PRIME-w/o-PW":
        return MechanismSpec(
            name=method, eta_H=eta_H, behavior_zeta_one=True,
            notes="zeta=1: W(p)=p everywhere",
        )
    if method == "PRIME-w/o-LT":
        return MechanismSpec(name=method, eta_H=0.0,
                             notes="eta_H=0: no future reinforcement value")
    if method == "PRIME-w/o-RC":
        return MechanismSpec(name=method, eta_H=eta_H, allow_recultivation=False,
                             notes="maintenance never returns to cultivation")
    if method == "PRIME-Fixed":
        n_fix = int((cfg.get("baselines", {}).get("prime_fixed", {}) or {})
                    .get("cultivation_interactions", 15))
        return MechanismSpec(
            name=method, eta_H=eta_H, fixed_cultivation_count=n_fix,
            allow_recultivation=False,
            notes=f"switch to maintenance after {n_fix} assigned interactions",
        )
    if method == "PRIME-R":
        # Robust PRIME (Experiment F): same spec as PRIME; the conservative
        # design-parameter handling is applied by the simulator when
        # estimation_error.robust is true.
        return MechanismSpec(name=method, eta_H=eta_H,
                             notes="conservative design params under uncertainty")

    # ── Baselines (Sections 7.1–7.5) ─────────────────────────
    if method in ("SAMI", "PASI"):
        # PASI (paper name) = SAMI (code name): Path-State-Aware Incentive.
        # Uses current H in contract design (Lambda = [C'(a_sys)/g'(a_sys)
        # - omega_i*H_i^t]^+) with no cultivation, no future reinforcement
        # value, no stage transitions, no recultivation, no probability
        # smoothing.  PASI exploits naturally-formed H without active mgmt.
        # PASI vs MOI → value of exploiting current H.
        # PRIME vs PASI → incremental value of active cultivation + long-term mgmt.
        return MechanismSpec(
            name="SAMI", contract_mode="p1", design_H="true",
            target_mode="system", eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False, allow_recultivation=False,
            notes="PASI: path-state-aware incentive — uses current H, no active cultivation",
        )

    if method in ("MOI", "myopic_optimal"):
        # Same target machinery / P1 / a_star / base payment as PRIME but
        # H=0 and eta_H=0 in contract design and matching (§7.5).
        return MechanismSpec(
            name="MOI", contract_mode="p1", design_H="zero",
            target_mode="system", eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False,
            notes="strongest non-path-dependent baseline",
        )
    if method in ("QUAC-F", "QUACF", "quac_f"):
        # Li et al., MASS 2017, QUAC-F under the risk-neutral case.  Their
        # public linear quality contract A+Bq maps exactly to this pipeline as
        # b + D*g(a), because q=q_bar*g(a), D=B*q_bar, the slope satisfies
        # D=C'(a*)/g'(a*), and the intercept binds IR.  The public method has
        # no path-state term, so design_H=zero; provider behaviour and the
        # common task/capacity/budget/matching environment remain unchanged.
        return MechanismSpec(
            name="QUAC-F", contract_mode="risk_neutral", design_H="zero",
            target_mode="system", base_mode="quac_ir", enforce_target=True, eta_H=0.0,
            maintenance_smoothing=False, use_stage_logic=False,
            p_fixed=1.0,
            notes=("Li et al. MASS 2017 QUAC-F risk-neutral linear-quality "
                   "contract mapped to shared q(a), C(a), IR, capacity, budget, "
                   "and matching semantics"),
        )
    if method in ("QUAC-I-MAPPED", "QUAC-I", "quac_i"):
        # Protocol-aligned, independent implementation of QUAC-I.  The
        # incomplete-information envelope itself is supplied by the runner
        # through ``uncertainty_aware``; this spec retains QUAC's affine
        # quality contract and IR-binding signed intercept.  It must not be
        # described as the authors' implementation.
        return MechanismSpec(
            name="QUAC-I-MAPPED", contract_mode="risk_neutral",
            design_H="zero", target_mode="system", base_mode="quac_ir",
            enforce_target=True, eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False, p_fixed=1.0,
            notes=("protocol-aligned QUAC-I: affine quality contract under a "
                   "preregistered cost/response type envelope; shared QoS, "
                   "capacity, budget, candidate graph and matching objective"),
        )
    if method in ("QIM-E-MAPPED", "QIM-E", "qim_e"):
        # QIM-E is a quality-constrained procurement auction rather than an
        # effort contract.  The shared-protocol mapping therefore purchases
        # exactly the minimum feasible QoS effort and minimizes the resulting
        # IR payment under the common coverage-first/payment-second matcher.
        return MechanismSpec(
            name="QIM-E-MAPPED", contract_mode="risk_neutral",
            design_H="zero", target_mode="qos", base_mode="quac_ir",
            enforce_target=True, eta_H=0.0, maintenance_smoothing=False,
            use_stage_logic=False, p_fixed=1.0,
            notes=("protocol-aligned QIM-E quality-constrained procurement: "
                   "minimum-QoS target, modeled cost bid, IR payment, shared "
                   "information, candidate graph, capacity and budget"),
        )
    if method in ("FR", "fixed_reward"):
        b = _baseline_cfg(cfg, "fixed_reward")
        return MechanismSpec(
            name="FR", contract_mode="fixed", design_H="zero",
            target_mode="none", base_mode="fixed", enforce_target=False,
            eta_H=0.0, maintenance_smoothing=False, use_stage_logic=False,
            p_fixed=float(b.get("p", 0.4)),
            D_fixed=float(b.get("D", 3.0)),
            b_fixed=float(b.get("base_payment", 0.05)),
            incentive_scale=float(b.get("incentive_scale", 1.0)),
            notes="fixed (b,p,D); provider declines if IR violated",
        )
    if method in ("DP", "dynamic_pricing"):
        b = _baseline_cfg(cfg, "dynamic_pricing")
        return MechanismSpec(
            name="DP", contract_mode="risk_neutral", design_H="zero",
            target_mode="system", enforce_target=False, eta_H=0.0,
            maintenance_smoothing=False, use_stage_logic=False,
            p_fixed=float(b.get("p", 0.4)),
            incentive_scale=float(b.get("incentive_scale", 1.0)),
            notes="adaptive price D=Lambda/p (no behaviour model); D clipped to D_bar",
        )
    if method in ("RAI", "reputation_aware"):
        b = _baseline_cfg(cfg, "reputation_aware")
        return MechanismSpec(
            name="RAI", contract_mode="risk_neutral", design_H="zero",
            target_mode="system", enforce_target=False, eta_H=0.0,
            maintenance_smoothing=False, use_stage_logic=False,
            reputation_weighting=True,
            p_fixed=float(b.get("p", 0.4)),
            incentive_scale=float(b.get("incentive_scale", 1.0)),
            notes="reputation EWMA scales matching weight; exploration for new providers",
        )
    if method in ("LDI", "linear_decay"):
        b = _baseline_cfg(cfg, "linear_decay")
        return MechanismSpec(
            name="LDI", contract_mode="ldi", design_H="zero",
            target_mode="none", enforce_target=False, eta_H=0.0,
            maintenance_smoothing=False, use_stage_logic=False,
            p_start=float(b.get("p_start", 0.8)),
            p_end=float(b.get("p_end", 0.1)),
            decay_interactions=int(b.get("decay_interactions", 30)),
            D_fixed=float(b.get("D", 2.0)),
            incentive_scale=float(b.get("incentive_scale", 1.0)),
            notes="p decays linearly with assigned interactions; quality may drop",
        )

    if method == "PRIME_ORIGINAL":
        return MechanismSpec(
            name="PRIME_ORIGINAL", eta_H=eta_H,
            notes="Original PRIME (alias for PRIME) — preserved for backward compatibility",
        )

    if method in ("SFPRIME", "SFPRIME_ORACLE"):
        # SF-PRIME: Selective Finite-Horizon cultivation with exact SAMI fallback.
        # The MechanismSpec describes the BASELINE (SAMI-equivalent) behavior.
        # Actual cultivation decisions are made by the SFPRIMESimulator controller.
        return MechanismSpec(
            name=method,
            contract_mode="p1",
            design_H="true",
            target_mode="sf_prime",       # Per-pair target dispatch by simulator
            eta_H=0.0,                     # NO future reinforcement value in pair score
            maintenance_smoothing=False,   # SF-PRIME has no probability smoothing
            use_stage_logic=False,         # SF-PRIME manages modes independently
            allow_recultivation=False,     # No automatic recultivation
            notes="SAMI baseline + finite-horizon selective cultivation with exact fallback",
        )

    if method == "SFPRIME_ESTIMATED":
        return MechanismSpec(
            name="SFPRIME_ESTIMATED",
            contract_mode="p1",
            design_H="true",
            target_mode="sf_prime",
            eta_H=0.0,
            maintenance_smoothing=False,
            use_stage_logic=False,
            allow_recultivation=False,
            notes="SF-PRIME with estimated (non-oracle) future stream prediction",
        )

    if method == "SFPRIME_ALWAYS_ADMIT":
        # Ablation: always admit regardless of LCB/opportunity check
        return MechanismSpec(
            name="SFPRIME_ALWAYS_ADMIT",
            contract_mode="p1",
            design_H="true",
            target_mode="sf_prime",
            eta_H=0.0,
            maintenance_smoothing=False,
            use_stage_logic=False,
            allow_recultivation=False,
            notes="Ablation: SF-PRIME with forced admission (no selectivity)",
        )

    raise ValueError(f"Unknown method: {method!r}")


ALL_METHODS = ["PRIME", "SAMI", "MOI", "QUAC-F", "QUAC-I-MAPPED",
               "QIM-E-MAPPED", "FR", "DP", "RAI", "LDI",
               "SFPRIME", "PRIME_ORIGINAL"]
ABLATION_METHODS = ["PRIME", "PRIME-w/o-PD", "PRIME-w/o-PW", "PRIME-w/o-LT",
                    "PRIME-w/o-RC", "PRIME-Fixed"]
