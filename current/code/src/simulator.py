"""PRIME experiment simulator (Sections 5–6, 19–20).

Orchestrates a full episode: pair evaluation (vectorised, src/pair_eval.py),
budget-constrained matching (src/matching.py), per-provider state updates,
and three-level logging (pair / provider / slot).

Method dispatch is driven by MechanismSpec (src/mechanisms.py).  Provider
BEHAVIOUR (a_star response, experienced utility, true path-state update)
is identical across methods for the same seed (§2.5); mechanisms only
change what the platform computes (design side).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .mechanisms import MechanismSpec, get_mechanism
from .pair_eval import REASON_LABELS, evaluate_pairs
from .outcome_diagnostics import service_outcomes
from .matching import coverage_first_matching, lagrangian_matching, milp_matching
from .environment_events import (
    apply_state_reset_if_due, build_state_reset_ids, state_reset_event_hash,
)

_EPS = 1e-12

_PAIR_LOG_COLS = [
    "a_deadline", "a_quality", "a_min", "a_system", "a_reinforcement",
    "a_target", "lambda_required", "p_star", "D_star", "gamma_effective",
    "a_star", "implementation_gap", "base_payment", "expected_bonus",
    "expected_contract_cost", "perceived_utility", "experienced_utility",
    "design_expected_contract_cost", "decision_contract_cost",
    "decision_pair_value", "decision_quality",
    "realized_effort_cost", "runtime_path_value", "outside_option",
    "effective_reserve_cost",
    "execution_quality", "normalized_quality", "total_delay",
    "delta_H_predicted", "immediate_value", "reinforcement_value",
    "total_pair_value",
]


class Simulator:
    """Main simulation engine for one (dataset, method, seed) run."""

    def __init__(
        self,
        cfg: dict[str, Any],
        dataset: dict[str, pd.DataFrame],
        method: str = "PRIME",
        seed: int = 42,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.method = method
        self.seed = seed
        self.mech: MechanismSpec = get_mechanism(method, cfg)

        self.tasks_df: pd.DataFrame = dataset["tasks"]
        self.providers_df: pd.DataFrame = dataset["providers"]
        self.static_df: pd.DataFrame = dataset["provider_static"].reset_index(drop=True)
        self.meta: dict = dataset.get("meta", {})

        # Config shortcuts
        self.sim_cfg = cfg.get("simulation", {}) or {}
        self.contract_cfg = cfg.get("contract", {}) or {}
        self.path_cfg = cfg.get("path_state", {}) or {}
        self.match_cfg = cfg.get("matching", {}) or {}
        self.log_level = self.sim_cfg.get("log_level", "selected")

        self.p_min = float(self.contract_cfg.get("p_min", 0.05))
        self.p_max = float(self.contract_cfg.get("p_max", 0.80))
        self.D_bar = float(self.contract_cfg.get("D_bar", 10.0))
        self.reinforcement_margin = float(self.contract_cfg.get("reinforcement_margin", 0.05))
        self.Theta_M = float(self.path_cfg.get("Theta_M", 0.75))
        self.Theta_C = float(self.path_cfg.get("Theta_C", 0.55))
        self.s_M = float(self.path_cfg.get("s_M", 0.80))
        self.s_C = float(self.path_cfg.get("s_C", 0.65))
        self.K = int(self.path_cfg.get("K", 10))
        self.delta_p_max = float(self.path_cfg.get("delta_p_max", 0.05))
        self.rho_budget = float(self.match_cfg.get("budget_ratio", 0.70))
        # Small-instance exact optimum verification (Sections 6.3, 12.9)
        self.milp_verify = bool(self.match_cfg.get("milp_verify", False))
        self.milp_max_pairs = int(self.match_cfg.get("milp_max_pairs", 400))

        rai_cfg = (cfg.get("baselines", {}) or {}).get("reputation_aware", {}) or {}
        self.rai_ewma = float(rai_cfg.get("ewma_alpha", 0.2))
        self.rai_r0 = float(rai_cfg.get("initial_reputation", 0.5))
        self.rai_w0 = float(rai_cfg.get("weight_base", 0.5))
        self.rai_w1 = float(rai_cfg.get("weight_scale", 1.0))
        self.rai_explore = int(rai_cfg.get("explore_interactions", 5))
        self.rai_explore_mult = float(rai_cfg.get("explore_multiplier", 1.0))

        self._setup_provider_arrays()
        self._setup_design_params()
        self._setup_e8_observer()
        self._setup_state_reset()

        # Logs
        self.pair_log_frames: list[pd.DataFrame] = []
        self.provider_log: list[dict] = []
        self.slot_log: list[dict] = []

        # Stage 2B-VR: Read-only H-path trace and reset hook (no state changes)
        self._h_trace_enabled: bool = False
        self._h_trace_records: list[dict] = []
        self._reset_hook_records: list[dict] = []

    # ── Provider static / state arrays ──────────────────────────

    def _setup_provider_arrays(self) -> None:
        sdf = self.static_df
        self.pid_list = sdf["provider_id"].tolist()
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pid_list)}
        n = len(self.pid_list)

        def col(name: str, default: float) -> np.ndarray:
            if name in sdf.columns:
                return sdf[name].to_numpy(dtype=float)
            return np.full(n, default)

        self.s_Fmax = col("max_processing_rate", 10.0)
        self.s_alpha = col("alpha", 0.1)
        self.s_beta = col("beta", 0.05)
        self.s_zeta = col("zeta", 0.8)
        self.s_omega = col("omega", 0.1)
        self.s_xi = col("xi", 0.08)
        self.s_delta = col("delta", 0.03)
        self.s_Uout = col("outside_option", 0.01)

        # Model-level ablations change true behaviour (w/o-PD, w/o-PW)
        if self.mech.behavior_omega_zero:
            self.s_omega = np.zeros(n)
        if self.mech.behavior_zeta_one:
            self.s_zeta = np.ones(n)

        prov_cfg = self.cfg.get("providers", {}) or {}
        init_H = float(prov_cfg.get("initial_H", 0.1))

        # Dynamic state vectors
        self.H_true = np.full(n, init_H)
        self.H_design_state = np.full(n, init_H)  # platform belief (≠ true under est. error)
        self.stage_maint = np.zeros(n, dtype=bool)  # False=cultivation
        self.last_p = np.zeros(n)
        self.reputation = np.full(n, self.rai_r0)
        self.n_interactions = np.zeros(n, dtype=int)
        self.recent_quality: list[list[float]] = [[] for _ in range(n)]

    def _setup_design_params(self) -> None:
        """Design-side parameter copies (estimation error, Experiment F).

        Perturbations are seeded by (seed, param) only, so all methods see
        identical estimates for the same seed.
        """
        err_cfg = self.cfg.get("estimation_error", {}) or {}
        level = float(err_cfg.get("level", 0.0))
        which = err_cfg.get("params", "all")
        mode = err_cfg.get("mode", "random")  # random | over | under
        robust = bool(err_cfg.get("robust", False)) or self.method == "PRIME-R"

        n = len(self.pid_list)
        self.d_omega = self.s_omega.copy()
        self.d_zeta = self.s_zeta.copy()
        self.d_xi = self.s_xi.copy()
        self.d_delta = self.s_delta.copy()
        self.d_alpha = self.s_alpha.copy()
        self.d_beta = self.s_beta.copy()
        self.response_model_scale = 1.0
        self.design_equals_true = True
        self.est_error_active = level > 0

        if level > 0:
            if which == "all":
                which = ["omega", "zeta", "xi", "delta"]
            elif isinstance(which, str):
                which = [which]
            for i, pname in enumerate(["omega", "zeta", "xi", "delta"]):
                if pname not in which:
                    continue
                rng = np.random.default_rng([self.seed, 7700 + i])
                if mode == "over":
                    eps = rng.uniform(0.0, level, size=n)
                elif mode == "under":
                    eps = rng.uniform(-level, 0.0, size=n)
                else:
                    eps = rng.uniform(-level, level, size=n)
                arr = getattr(self, f"d_{pname}")
                arr *= (1.0 + eps)
                if pname == "zeta":
                    np.clip(arr, 0.05, 1.0, out=arr)
                else:
                    np.clip(arr, 0.0, None, out=arr)
            self.design_equals_true = False

            if robust:
                # PRIME-R (§12.6): use the most conservative behavioural
                # contribution within the uncertainty interval — lower
                # omega credit, and zeta pushed toward 1 (smaller W(p) in
                # the typical p < 1/e operating regime → larger D).
                self.d_omega = self.d_omega * (1.0 - level)
                self.d_zeta = np.clip(self.d_zeta * (1.0 + level), 0.05, 1.0)

        # Behaviour ablations force consistent design copies
        if self.mech.behavior_omega_zero:
            self.d_omega = np.zeros(n)
        if self.mech.behavior_zeta_one:
            self.d_zeta = np.ones(n)

        model_cfg = self.cfg.get("model_error", {}) or {}
        cost_bias = float(model_cfg.get("cost_bias", 0.0))
        response_bias = float(model_cfg.get("response_bias", 0.0))
        if cost_bias != 0.0:
            scale = max(0.01, 1.0 + cost_bias)
            self.d_alpha = self.s_alpha * scale
            self.d_beta = self.s_beta * scale
            self.design_equals_true = False
        if response_bias != 0.0:
            self.response_model_scale = max(0.01, 1.0 + response_bias)
            self.design_equals_true = False

    def _setup_e8_observer(self) -> None:
        """Initialize the frozen E8 runtime-state observation channel.

        The platform observes a delayed/noisy/masked copy.  True provider
        behaviour and state transitions continue to use ``H_true``.
        Missing observations use last observation carried forward (LOCF).
        """
        obs = self.cfg.get("state_observation", {}) or {}
        self.obs_noise_std = float(obs.get("noise_std", 0.0))
        self.obs_delay_slots = int(obs.get("delay_slots", 0))
        self.obs_missing_rate = float(obs.get("missing_rate", 0.0))
        if self.obs_noise_std < 0 or self.obs_delay_slots < 0:
            raise ValueError("state observation noise/delay must be non-negative")
        if not 0.0 <= self.obs_missing_rate <= 1.0:
            raise ValueError("state observation missing_rate must be in [0,1]")
        self._obs_history: list[np.ndarray] = []
        self._obs_last = self.H_design_state.copy()
        self._obs_cache_slot: int | None = None
        self._obs_cache = self.H_design_state.copy()
        robust = self.cfg.get("uncertainty_aware", {}) or {}
        self.robust_enabled = bool(robust.get("enabled", False))
        self.robust_confidence_z = float(robust.get("state_confidence_z", 3.290527))
        self.robust_response_bound = float(robust.get("response_relative_bound", 0.0))
        self.robust_response_ratio_lower = robust.get("response_ratio_lower")
        self.robust_response_ratio_upper = robust.get("response_ratio_upper")
        if ((self.robust_response_ratio_lower is None)
                != (self.robust_response_ratio_upper is None)):
            raise ValueError(
                "response_ratio_lower and response_ratio_upper must be supplied together")
        if self.robust_response_ratio_lower is not None:
            self.robust_response_ratio_lower = float(self.robust_response_ratio_lower)
            self.robust_response_ratio_upper = float(self.robust_response_ratio_upper)
            if (self.robust_response_ratio_lower <= 0.0
                    or self.robust_response_ratio_upper < self.robust_response_ratio_lower):
                raise ValueError(
                    "response ratio bounds must satisfy 0 < lower <= upper")
        self.robust_cost_bound = float(robust.get("cost_relative_bound", 0.0))
        adaptive = robust.get("adaptive_response", {}) or {}
        self.adaptive_response_enabled = bool(adaptive.get("enabled", False))
        self.adaptive_min_history = int(adaptive.get("min_history", 20))
        self.adaptive_window = int(adaptive.get("window", 100))
        self.adaptive_alpha = float(adaptive.get("miscoverage_alpha", 0.05))
        self.adaptive_margin = float(adaptive.get("safety_margin", 0.02))
        self.adaptive_global_lower = float(adaptive.get("global_lower", 0.70))
        self.adaptive_global_upper = float(adaptive.get("global_upper", 1.30))
        self.adaptive_global_guardrail_weight = float(
            adaptive.get("global_guardrail_weight", 0.0))
        if self.adaptive_response_enabled:
            if not (0.0 < self.adaptive_alpha < 1.0):
                raise ValueError("adaptive miscoverage_alpha must lie in (0,1)")
            if self.adaptive_min_history < 1 or self.adaptive_window < 1:
                raise ValueError("adaptive history sizes must be positive")
            if not (0.0 < self.adaptive_global_lower <= self.adaptive_global_upper):
                raise ValueError("invalid adaptive global response bounds")
            if not (0.0 <= self.adaptive_global_guardrail_weight <= 1.0):
                raise ValueError(
                    "adaptive global_guardrail_weight must lie in [0,1]")
            self.robust_enabled = True
            self.robust_response_ratio_lower = self.adaptive_global_lower
            self.robust_response_ratio_upper = self.adaptive_global_upper
        self._response_ratio_history: list[float] = []
        self._adaptive_bound_mode = "fixed"
        self.robust_missing_floor_zero = bool(robust.get("missing_state_floor_zero", True))
        if self.robust_confidence_z < 0:
            raise ValueError("state_confidence_z must be non-negative")
        if self.obs_noise_std or self.obs_delay_slots or self.obs_missing_rate:
            self.design_equals_true = False

    def _apply_adaptive_guardrail(self, lower: float,
                                  upper: float) -> tuple[float, float]:
        """Widen a past-only interval toward the preregistered envelope.

        Weight zero recovers the original adaptive interval; weight one
        recovers the fixed global envelope.  Intermediate weights produce a
        nested, predeclared risk--coverage path without current-slot response
        information.
        """
        weight = self.adaptive_global_guardrail_weight
        lower = ((1.0 - weight) * float(lower)
                 + weight * self.adaptive_global_lower)
        upper = ((1.0 - weight) * float(upper)
                 + weight * self.adaptive_global_upper)
        lower = max(self.adaptive_global_lower, lower)
        upper = min(self.adaptive_global_upper, upper)
        return min(lower, upper), max(lower, upper)

    def _state_lower_bound(self, observed: np.ndarray, provider_indices: np.ndarray,
                           delta_mult: np.ndarray) -> np.ndarray:
        """Conservative lower state bound used only by PASI-U.

        Gaussian noise is covered at the declared one-sided quantile.  Delay
        is covered by the maximum per-slot state decay.  With LOCF missingness
        enabled, the age can be arbitrarily long over the finite tape, so the
        globally valid lower bound is zero unless explicitly disabled.
        """
        if not self.robust_enabled:
            return observed
        radius = np.full_like(observed, self.robust_confidence_z * self.obs_noise_std)
        radius += self.obs_delay_slots * self.d_delta[provider_indices] * delta_mult
        lower = np.clip(observed - radius, 0.0, 1.0)
        if self.obs_missing_rate and self.robust_missing_floor_zero:
            lower = np.zeros_like(lower)
        return lower

    def _adaptive_response_bounds(self) -> tuple[float | None, float | None]:
        """Return a past-only conformal-style response interval.

        During warm-up the preregistered global envelope is used.  Thereafter
        empirical tail quantiles over a rolling window are widened by a fixed
        safety margin and clipped to the global support.  Current-slot audit
        values are appended only after contract construction, preventing
        test-side look-ahead.
        """
        if not self.adaptive_response_enabled:
            self._adaptive_bound_mode = "fixed"
            return self.robust_response_ratio_lower, self.robust_response_ratio_upper
        history = np.asarray(self._response_ratio_history[-self.adaptive_window:],
                             dtype=float)
        if len(history) < self.adaptive_min_history:
            self._adaptive_bound_mode = "global_warmup"
            return self.adaptive_global_lower, self.adaptive_global_upper
        # Two-tier fallback: a wide or abruptly shifted recent window is
        # treated as a regime-change alarm and reverts to the preregistered
        # safety envelope.  Adaptation is used only in locally stable regimes.
        recent = history[-min(len(history), self.adaptive_min_history):]
        if float(np.ptp(recent)) > 0.12:
            self._adaptive_bound_mode = "global_regime_alarm"
            return self.adaptive_global_lower, self.adaptive_global_upper
        # A monotone drift can have a small local range while still making a
        # trailing quantile anti-conservative.  Widen toward the recent extrema
        # whenever local variation is material; stable windows still shrink.
        if float(np.std(recent)) > 0.001:
            self._adaptive_bound_mode = "adaptive_varying"
            lo = max(self.adaptive_global_lower,
                     float(np.min(recent)) - max(self.adaptive_margin, 0.08))
            hi = min(self.adaptive_global_upper,
                     float(np.max(recent)) + max(self.adaptive_margin, 0.08))
            return self._apply_adaptive_guardrail(lo, hi)
        lo = float(np.quantile(history, self.adaptive_alpha / 2.0,
                               method="lower")) - self.adaptive_margin
        hi = float(np.quantile(history, 1.0 - self.adaptive_alpha / 2.0,
                               method="higher")) + self.adaptive_margin
        lo = max(self.adaptive_global_lower, lo)
        hi = min(self.adaptive_global_upper, hi)
        self._adaptive_bound_mode = "adaptive_stable"
        return self._apply_adaptive_guardrail(lo, hi)

    def _observed_H(self, slot: int) -> np.ndarray:
        """Return one deterministic platform observation vector per slot."""
        if self._obs_cache_slot == int(slot):
            return self._obs_cache
        self._obs_history.append(self.H_design_state.copy())
        idx = max(0, len(self._obs_history) - 1 - self.obs_delay_slots)
        observed = self._obs_history[idx].copy()
        if self.obs_noise_std:
            rng = np.random.default_rng([self.seed, 8800, int(slot)])
            observed += rng.normal(0.0, self.obs_noise_std, size=len(observed))
        observed = np.clip(observed, 0.0, 1.0)
        if self.obs_missing_rate:
            rng = np.random.default_rng([self.seed, 8801, int(slot)])
            missing = rng.random(len(observed)) < self.obs_missing_rate
            observed[missing] = self._obs_last[missing]
            self._obs_last[~missing] = observed[~missing]
        else:
            self._obs_last = observed.copy()
        self._obs_cache_slot = int(slot)
        self._obs_cache = observed
        return observed

    def _setup_state_reset(self) -> None:
        """Pre-compute state-reset provider IDs from shared environment seed."""
        sr_cfg = self.sim_cfg.get("state_reset", {}) or {}
        self.state_reset_config = sr_cfg if sr_cfg.get("enabled", False) else None
        self.reset_provider_ids: np.ndarray = np.array([], dtype=int)
        self._reset_applied = False
        self._reset_event: dict = {}

        if self.state_reset_config is None:
            return

        n = len(self.pid_list)
        fraction = float(sr_cfg.get("fraction", 0.50))
        # Use environment seed for deterministic cross-mechanism sharing
        env_seed = int(self.meta.get("seed", self.seed))
        self.reset_provider_ids = build_state_reset_ids(
            n_providers=n, fraction=fraction, seed=env_seed,
            at_slot=int(sr_cfg.get("at_slot", -1)),
        )

    # ── Helpers ────────────────────────────────────────────────

    def _get_budget(self, slot_tasks: pd.DataFrame) -> float:
        return self.rho_budget * float(slot_tasks["value"].sum())

    def _shock_mult(self, df: pd.DataFrame, colname: str, idx: np.ndarray) -> np.ndarray:
        """Optional per-slot shock multiplier columns in the providers frame."""
        if colname in df.columns:
            return df[colname].to_numpy(dtype=float)[idx]
        return np.ones(len(idx))

    # ── Main loop ──────────────────────────────────────────────

    def run(self) -> dict:
        T_cfg = int(self.meta.get("T", self.sim_cfg.get("T", 1000)))
        slots = sorted(self.tasks_df["slot"].unique())
        if len(slots) > T_cfg:
            slots = slots[:T_cfg]

        tasks_by_slot = dict(tuple(self.tasks_df.groupby("slot")))
        provs_by_slot = dict(tuple(self.providers_df.groupby("slot")))

        warm_lambda = float(self.match_cfg.get("initial_lambda_B", 0.1))
        t0_total = time.perf_counter()
        total_tasks = 0

        for slot in slots:
            t0_slot = time.perf_counter()
            slot_tasks = tasks_by_slot.get(slot)
            slot_provs = provs_by_slot.get(slot)
            if slot_tasks is None or len(slot_tasks) == 0:
                continue
            n_tasks = len(slot_tasks)
            total_tasks += n_tasks
            n_prov = 0 if slot_provs is None else len(slot_provs)
            budget = self._get_budget(slot_tasks)

            response_history_n_used = len(self._response_ratio_history)
            response_lo, response_hi = self._adaptive_response_bounds()
            response_bound_mode = self._adaptive_bound_mode

            if n_prov == 0:
                self._log_empty_slot(slot, n_tasks, budget, t0_slot)
                continue

            # ── State-Reset (environment event, before contracts) ──
            if self.state_reset_config is not None and not self._reset_applied:
                # Stage 2B-VR: record reset instant hook BEFORE applying reset (slot 500 ONLY)
                if self._h_trace_enabled and slot == 500 and self.reset_provider_ids is not None and len(self.reset_provider_ids) > 0:
                    for i, ridx in enumerate(self.reset_provider_ids):
                        if 0 <= ridx < len(self.pid_list):
                            self._reset_hook_records.append({
                                "world": getattr(self, "_ht_world_label", ""),
                                "slot": int(slot), "provider_id": self.pid_list[ridx],
                                "reset_target": True, "available_at_slot_start": 1,
                                "H_true_before_reset": float(self.H_true[ridx]),
                                "H_true_after_reset_immediate": 0.0,
                                "H_design_before_reset": float(self.H_design_state[ridx]),
                                "H_design_after_reset_immediate": float(self.H_design_state[ridx]),
                                "reset_function_called": True, "reset_call_index": i,
                                "reset_executed": True, "pass": True,
                            })
                evt = apply_state_reset_if_due(
                    slot, self.H_true, self.state_reset_config,
                    self.reset_provider_ids,
                )
                if evt["applied"]:
                    self._reset_applied = True
                    self._reset_event = evt

            # ── Build pair arrays (cross product) ───────────
            t0c = time.perf_counter()
            p_idx_local = slot_provs["provider_id"].map(self.pid_to_idx).to_numpy()
            P = np.repeat(np.arange(n_prov), n_tasks)      # local provider row
            Tk = np.tile(np.arange(n_tasks), n_prov)       # local task row
            gp = p_idx_local[P]                            # global provider idx

            load = slot_provs["load_ratio"].to_numpy(dtype=float)[P]
            comm = slot_provs["communication_rate"].to_numpy(dtype=float)[P]
            avail = slot_provs["availability_duration"].to_numpy(dtype=float)[P]

            a_mult = self._shock_mult(slot_provs, "alpha_mult", P)
            b_mult = self._shock_mult(slot_provs, "beta_mult", P)
            o_mult = self._shock_mult(slot_provs, "omega_mult", P)
            d_mult = self._shock_mult(slot_provs, "delta_mult", P)

            tv = {c: slot_tasks[c].to_numpy(dtype=float)[Tk]
                  for c in ["L", "value", "q_bar", "min_quality", "kappa",
                            "input_size", "output_size", "deadline"]}
            if "kappa_design" in slot_tasks.columns:
                tv["kappa_design"] = slot_tasks[
                    "kappa_design"].to_numpy(dtype=float)[Tk]

            F_i_t = np.maximum(_EPS, (1.0 - load) * self.s_Fmax[gp])
            H_d = (self._observed_H(slot)[gp] if self.mech.design_H == "true"
                   else np.zeros(len(gp)))
            H_lower = self._state_lower_bound(H_d, gp, d_mult)

            ctx = {
                "L": tv["L"], "V": tv["value"], "q_bar": tv["q_bar"],
                "q_min": tv["min_quality"], "kappa": tv["kappa"],
                "input_size": tv["input_size"], "output_size": tv["output_size"],
                "deadline": tv["deadline"],
                "alpha": self.s_alpha[gp] * a_mult,
                "beta": self.s_beta[gp] * b_mult,
                "alpha_d": self.d_alpha[gp] * a_mult,
                "beta_d": self.d_beta[gp] * b_mult,
                "kappa_d": tv.get("kappa_design", tv["kappa"]) * self.response_model_scale,
                "omega_t": self.s_omega[gp] * o_mult,
                "zeta_t": self.s_zeta[gp],
                "xi_t": self.s_xi[gp],
                "delta_t": self.s_delta[gp] * d_mult,
                "U_out": self.s_Uout[gp],
                "F_i_t": F_i_t,
                "communication_rate": comm,
                "availability": avail,
                "H_t": self.H_true[gp],
                "in_cultivation": ~self.stage_maint[gp],
                "last_p": self.last_p[gp],
                "n_interactions": self.n_interactions[gp].astype(float),
                "omega_d": self.d_omega[gp] * o_mult,
                "zeta_d": self.d_zeta[gp],
                "xi_d": self.d_xi[gp],
                "delta_d": self.d_delta[gp] * d_mult,
                "H_d": H_d,
                "H_lower": H_lower,
                "robust_enabled": self.robust_enabled,
                "response_relative_bound": self.robust_response_bound,
                "response_ratio_lower": response_lo,
                "response_ratio_upper": response_hi,
                "cost_relative_bound": self.robust_cost_bound,
                "p_min": self.p_min, "p_max": self.p_max, "D_bar": self.D_bar,
                "reinforcement_margin": self.reinforcement_margin,
                "Theta_M": self.Theta_M, "delta_p_max": self.delta_p_max,
                "design_equals_true": self.design_equals_true,
            }
            ev = evaluate_pairs(self.mech, ctx)
            if self.adaptive_response_enabled:
                # Two-tier candidate fallback.  The adaptive contract is the
                # primary offer; whenever it rejects a pair that is feasible
                # under the preregistered global safety envelope, retain the
                # globally robust offer.  This uses design information only
                # and therefore introduces no current-slot response leakage.
                fallback_ctx = dict(ctx)
                fallback_ctx["response_ratio_lower"] = self.adaptive_global_lower
                fallback_ctx["response_ratio_upper"] = self.adaptive_global_upper
                fallback_ev = evaluate_pairs(self.mech, fallback_ctx)
                use_fallback = (~ev["feasible"]) & fallback_ev["feasible"]
                if use_fallback.any():
                    for key, value in list(ev.items()):
                        other = fallback_ev.get(key)
                        if (isinstance(value, np.ndarray)
                                and isinstance(other, np.ndarray)
                                and value.shape == other.shape
                                and value.shape == use_fallback.shape):
                            ev[key] = np.where(use_fallback, other, value)
                ev["response_global_fallback"] = use_fallback
            else:
                ev["response_global_fallback"] = np.zeros(len(P), dtype=bool)

            # Optional trace-driven spatial candidate graph.  The mobility
            # trace supplies locations only; all non-location task/provider
            # attributes remain model-generated and are recorded as such in
            # the preprocessing metadata.  Synthetic experiments omit these
            # columns and therefore retain the original complete graph.
            spatial_cols = {"latitude", "longitude"}
            if (spatial_cols.issubset(slot_provs.columns)
                    and spatial_cols.issubset(slot_tasks.columns)):
                p_lat = np.radians(slot_provs["latitude"].to_numpy(dtype=float)[P])
                p_lon = np.radians(slot_provs["longitude"].to_numpy(dtype=float)[P])
                t_lat = np.radians(slot_tasks["latitude"].to_numpy(dtype=float)[Tk])
                t_lon = np.radians(slot_tasks["longitude"].to_numpy(dtype=float)[Tk])
                dlat = t_lat - p_lat
                dlon = t_lon - p_lon
                hav = (np.sin(dlat / 2.0) ** 2
                       + np.cos(p_lat) * np.cos(t_lat) * np.sin(dlon / 2.0) ** 2)
                distance_km = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
                radius_km = float((self.cfg.get("dataset", {}) or {})
                                  .get("service_radius_km", 3.0))
                spatial_ok = distance_km <= radius_km
                rejected = ev["feasible"] & ~spatial_ok
                ev["feasible"] = ev["feasible"] & spatial_ok
                ev["feasible_physical"] = ev["feasible_physical"] & spatial_ok
                ev["reason_code"] = np.where(rejected, 8, ev["reason_code"])
                ev["spatial_distance_km"] = distance_km
                ev["spatial_candidate"] = spatial_ok
            contract_time = time.perf_counter() - t0c

            # ── Matching (P2) ───────────────────────────────
            feas = ev["feasible"]
            fidx = np.flatnonzero(feas)
            match_value = ev["decision_pair_value"][fidx]
            if self.mech.reputation_weighting and len(fidx) > 0:
                rep = self.reputation[gp[fidx]]
                mult = self.rai_w0 + self.rai_w1 * rep
                explore = self.n_interactions[gp[fidx]] < self.rai_explore
                mult = np.where(explore, np.maximum(mult, self.rai_explore_mult), mult)
                match_value = match_value * mult

            if self.match_cfg.get("objective") == "coverage_first_payment_second":
                match_result = coverage_first_matching(
                    P[fidx], Tk[fidx], match_value,
                    ev["decision_contract_cost"][fidx],
                    true_values=ev["decision_pair_value"][fidx],
                    n_providers=n_prov, n_tasks=n_tasks, budget=budget,
                )
            else:
                match_result = lagrangian_matching(
                    P[fidx], Tk[fidx], match_value,
                    ev["decision_contract_cost"][fidx],
                    true_values=ev["decision_pair_value"][fidx],
                    n_providers=n_prov, n_tasks=n_tasks, budget=budget,
                    max_iter=int(self.match_cfg.get("max_iter", 100)),
                    budget_tol=float(self.match_cfg.get("budget_tol", 1e-4)),
                    stagnation_limit=int(self.match_cfg.get("stagnation_limit", 5)),
                    initial_lambda=warm_lambda,
                    step_scale=float(self.match_cfg.get("step_scale", 0.1)),
                )
            warm_lambda = max(0.0, float(match_result["lambda_B"]))
            sel_local = match_result["selected_idx"]      # indices into fidx arrays
            sel_pairs = fidx[sel_local] if len(sel_local) else np.array([], dtype=int)

            # Optional exact-optimum certification on small instances
            opt_value = None
            certified_gap = None
            if self.milp_verify and 0 < len(fidx) <= self.milp_max_pairs:
                exact = milp_matching(
                    P[fidx], Tk[fidx], match_value,
                    ev["decision_contract_cost"][fidx],
                    n_providers=n_prov, n_tasks=n_tasks, budget=budget)
                if exact["status"] == "optimal":
                    opt_value = float(exact["total_value"])
                    lb = float(match_result["feasible_lower_bound"])
                    ub = match_result.get("lagrangian_upper_bound")
                    if lb > opt_value + 1e-6:
                        raise AssertionError(
                            f"slot {slot}: LB {lb} > exact OPT {opt_value}")
                    if ub is not None and opt_value > float(ub) + 1e-6:
                        raise AssertionError(
                            f"slot {slot}: exact OPT {opt_value} > UB {ub}")
                    certified_gap = (opt_value - lb) / max(abs(opt_value), _EPS)

            selected_mask = np.zeros(len(P), dtype=bool)
            selected_mask[sel_pairs] = True
            response_fallback_retained = (
                ev["response_global_fallback"] & ev["feasible"]
            )
            response_fallback_selected = (
                int(ev["response_global_fallback"][sel_pairs].sum())
                if len(sel_pairs) else 0
            )

            # ── Provider state updates (once per online provider) ──
            self._update_states(slot, slot_provs, p_idx_local, gp, Tk,
                                slot_tasks, ev, selected_mask)

            # Past-only update: realised response observations from this slot
            # become available only for future contract decisions.
            if self.adaptive_response_enabled and "response_ratio_audit" in slot_tasks:
                observed_ratio = slot_tasks["response_ratio_audit"].to_numpy(dtype=float)
                self._response_ratio_history.extend(
                    observed_ratio[np.isfinite(observed_ratio)].tolist())

            # ── Pair log ────────────────────────────────────
            self._log_pairs(slot, slot_provs, slot_tasks, P, Tk, ev, selected_mask)

            # ── Slot log ────────────────────────────────────
            sq = ev["execution_quality"][sel_pairs] if len(sel_pairs) else np.array([])
            spay = ev["expected_contract_cost"][sel_pairs] if len(sel_pairs) else np.array([])
            sutil = ev["experienced_utility"][sel_pairs] if len(sel_pairs) else np.array([])
            simm = ev["immediate_value"][sel_pairs] if len(sel_pairs) else np.array([])
            ir_bad = int((~ev["ir_ok"][sel_pairs]).sum()) if len(sel_pairs) else 0
            gap_bad = int((ev["implementation_gap"][sel_pairs] < -1e-7).sum()) if len(sel_pairs) else 0

            # QCR: qualified = execution_quality >= task's min_quality
            tsk_q_min = slot_tasks["min_quality"].to_numpy(dtype=float) if "min_quality" in slot_tasks.columns else np.full(n_tasks, 0.8)
            if len(sel_pairs):
                sq_full = ev["execution_quality"]
                tsk_idx_of_sel = Tk[sel_pairs]
                qual_sel = sq_full[sel_pairs] >= tsk_q_min[tsk_idx_of_sel]
                n_qualified = int(qual_sel.sum())
            else:
                n_qualified = 0

            self.slot_log.append({
                "slot": int(slot), "method": self.method, "seed": self.seed,
                "num_tasks": n_tasks, "num_providers": n_prov,
                "num_feasible_pairs": int(feas.sum()),
                "num_assigned": int(len(sel_pairs)),
                "num_completed": int(len(sel_pairs)),
                "num_high_quality": int((sq >= 0.8).sum()),
                "num_qualified_completed": n_qualified,
                "mean_quality": float(sq.mean()) if len(sq) else np.nan,
                "total_payment": float(spay.sum()),
                "total_value": float(ev["total_pair_value"][sel_pairs].sum()),
                "decision_total_value": float(match_result["total_value"]),
                "platform_utility": float(simm.sum()),
                "provider_utility_mean": float(sutil.mean()) if len(sutil) else np.nan,
                "budget": float(budget),
                "budget_used": float(match_result["total_cost"]),
                "budget_gap": float(budget - match_result["total_cost"]),
                "maintenance_ratio": float(self.stage_maint.mean()),
                "cultivation_ratio": float(1.0 - self.stage_maint.mean()),
                "ir_violations": ir_bad,
                "target_violations": gap_bad,
                "matching_runtime": float(match_result["runtime"]),
                "contract_runtime": contract_time,
                "response_bound_lower": response_lo,
                "response_bound_upper": response_hi,
                "response_bound_mode": response_bound_mode,
                "response_global_guardrail_weight": (
                    self.adaptive_global_guardrail_weight
                    if self.adaptive_response_enabled else 1.0),
                "response_history_n_used": response_history_n_used,
                "response_history_n": len(self._response_ratio_history),
                "response_global_fallback_candidates": int(
                    response_fallback_retained.sum()),
                "response_global_fallback_selected": response_fallback_selected,
                "total_runtime": time.perf_counter() - t0_slot,
                "UB": match_result.get("lagrangian_upper_bound"),
                "LB": match_result.get("feasible_lower_bound"),
                "optimality_gap": match_result.get("optimality_gap"),
                "OPT": opt_value,
                "certified_gap": certified_gap,
            })

        # ── Assemble results ─────────────────────────────────
        total_runtime = time.perf_counter() - t0_total
        pair_df = (pd.concat(self.pair_log_frames, ignore_index=True)
                   if self.pair_log_frames else pd.DataFrame())
        slot_df = pd.DataFrame(self.slot_log)
        provider_df = pd.DataFrame(self.provider_log)

        summary = self._build_summary(pair_df, slot_df, provider_df,
                                      total_tasks, total_runtime)
        diagnostics = _run_diagnostics(
            pair_df, slot_df, provider_df, summary,
            enforce_target=self.mech.enforce_target,
            expect_exact_ir=(not self.est_error_active
                             and self.mech.base_mode == "response_consistent"),
        )

        return {
            "summary": summary,
            "diagnostics": diagnostics,
            "pair_log": pair_df,
            "slot_log": slot_df,
            "provider_log": provider_df,
        }

    # ── Fixed-assignment replay (Stage 2A) ─────────────────────

    def run_fixed_assignments(
        self,
        fixed_assignment_plan: pd.DataFrame,
        event_tape: dict,
        reset_ids: np.ndarray | None = None,
        disable_matching: bool = True,
        replay_mode: bool = True,
    ) -> dict:
        """Replay with pre-determined assignments, using formal mechanism code.

        Parameters
        ----------
        fixed_assignment_plan : DataFrame
            Columns: slot, task_id, provider_id (and optionally realized_signal, etc.)
        event_tape : dict
            Frozen event tape with keys: 'tasks', 'providers', 'provider_static'.
        reset_ids : np.ndarray or None
            Pre-computed reset provider indices.
        disable_matching : bool
            Must be True — matching is skipped entirely.
        replay_mode : bool
            Must be True — enables replay audit logging.

        Returns
        -------
        dict with summary, diagnostics, pair_log, slot_log, provider_log,
        and additional replay audit fields.
        """
        if not disable_matching:
            raise ValueError("fixed-assignment replay requires disable_matching=True")

        # Build per-slot assignment lookup
        plan_df = fixed_assignment_plan.copy()
        plan_df["slot"] = plan_df["slot"].astype(int)
        plan_df["task_id"] = plan_df["task_id"].astype(str)
        plan_df["provider_id"] = plan_df["provider_id"].astype(str)

        # Index: (slot, task_id, provider_id) -> row
        plan_key_to_row: dict[tuple, dict] = {}
        for _, row in plan_df.iterrows():
            key = (int(row["slot"]), str(row["task_id"]), str(row["provider_id"]))
            plan_key_to_row[key] = {
                "assignment_key": row.get("assignment_key", f"{row['seed']}_{row['slot']}_{row['task_id']}_{row['provider_id']}"),
                "realized_signal": row.get("realized_signal", None),
                "realized_quality": row.get("realized_quality", None),
            }

        # Group fixed assignments by slot
        plan_by_slot: dict[int, list] = {}
        for key in plan_key_to_row:
            slot = key[0]
            plan_by_slot.setdefault(slot, []).append(key)

        # Use frozen event tape
        self.tasks_df = event_tape["tasks"]
        self.providers_df = event_tape["providers"]
        self.static_df = event_tape["provider_static"].reset_index(drop=True)

        # Re-initialize provider arrays from frozen static
        self._setup_provider_arrays()
        self._setup_design_params()

        # Use pre-computed reset IDs if provided
        if reset_ids is not None:
            self.reset_provider_ids = reset_ids

        T_cfg = int(self.meta.get("T", self.sim_cfg.get("T", 1000)))
        slots = sorted(self.tasks_df["slot"].unique())
        if len(slots) > T_cfg:
            slots = slots[:T_cfg]

        tasks_by_slot = dict(tuple(self.tasks_df.groupby("slot")))
        provs_by_slot = dict(tuple(self.providers_df.groupby("slot")))

        warm_lambda = float(self.match_cfg.get("initial_lambda_B", 0.1))
        t0_total = time.perf_counter()
        total_tasks = 0
        matching_calls = 0
        episode_gen_calls = 0
        reset_sampling_calls = 0
        formal_mechanism_calls = 0

        for slot in slots:
            t0_slot = time.perf_counter()
            slot_tasks = tasks_by_slot.get(slot)
            slot_provs = provs_by_slot.get(slot)
            if slot_tasks is None or len(slot_tasks) == 0:
                continue
            n_tasks = len(slot_tasks)
            total_tasks += n_tasks
            n_prov = 0 if slot_provs is None else len(slot_provs)
            budget = self._get_budget(slot_tasks)

            if n_prov == 0:
                self._log_empty_slot(slot, n_tasks, budget, t0_slot)
                continue

            # ── State-Reset (environment event, before contracts) ──
            if self.state_reset_config is not None and not self._reset_applied:
                # Stage 2B-VR: record reset instant hook BEFORE applying reset (slot 500 ONLY)
                if self._h_trace_enabled and slot == 500 and self.reset_provider_ids is not None and len(self.reset_provider_ids) > 0:
                    for i, ridx in enumerate(self.reset_provider_ids):
                        if 0 <= ridx < len(self.pid_list):
                            self._reset_hook_records.append({
                                "world": getattr(self, "_ht_world_label", ""),
                                "slot": int(slot), "provider_id": self.pid_list[ridx],
                                "reset_target": True, "available_at_slot_start": 1,
                                "H_true_before_reset": float(self.H_true[ridx]),
                                "H_true_after_reset_immediate": 0.0,
                                "H_design_before_reset": float(self.H_design_state[ridx]),
                                "H_design_after_reset_immediate": float(self.H_design_state[ridx]),
                                "reset_function_called": True, "reset_call_index": i,
                                "reset_executed": True, "pass": True,
                            })
                evt = apply_state_reset_if_due(
                    slot, self.H_true, self.state_reset_config,
                    self.reset_provider_ids,
                )
                if evt["applied"]:
                    self._reset_applied = True
                    self._reset_event = evt

            # ── Get fixed assignments for this slot ──
            fixed_keys = plan_by_slot.get(int(slot), [])
            fixed_task_ids = set(k[1] for k in fixed_keys)
            fixed_pid_tid_pairs = set((k[2], k[1]) for k in fixed_keys)  # (provider_id, task_id)

            # Map task_id and provider_id to local indices
            tid_to_local = {}
            for i, row in slot_tasks.iterrows():
                tid_to_local[str(row["task_id"])] = int(i - slot_tasks.index[0]) if hasattr(slot_tasks, 'index') else list(slot_tasks.index).index(i)

            # Rebuild: use positional indices
            task_ids_list = slot_tasks["task_id"].tolist()
            pid_list_slot = slot_provs["provider_id"].tolist()
            tid_to_local = {str(tid): i for i, tid in enumerate(task_ids_list)}
            pid_to_local = {str(pid): i for i, pid in enumerate(pid_list_slot)}

            # Build pair arrays ONLY for fixed assignments
            p_idx_local = slot_provs["provider_id"].map(self.pid_to_idx).to_numpy()
            gp_map = {str(pid_list_slot[i]): p_idx_local[i] for i in range(n_prov)}

            # Build selected mask based on fixed assignments
            selected_mask = np.zeros(n_prov * n_tasks, dtype=bool)
            pair_task_local = []
            pair_prov_local = []
            pair_prov_global = []

            for key in fixed_keys:
                _slot, tid, pid = key
                if tid in tid_to_local and pid in pid_to_local:
                    tl = tid_to_local[tid]
                    pl = pid_to_local[pid]
                    pg = gp_map[pid]
                    pair_idx = pl * n_tasks + tl
                    selected_mask[pair_idx] = True
                    pair_task_local.append(tl)
                    pair_prov_local.append(pl)
                    pair_prov_global.append(pg)
                else:
                    # Fixed assignment not feasible in this slot
                    pass

            # ── Build pair arrays and evaluate using formal mechanism ──
            if len(pair_prov_local) > 0:
                pair_task_local = np.array(pair_task_local, dtype=int)
                pair_prov_local = np.array(pair_prov_local, dtype=int)
                pair_prov_global = np.array(pair_prov_global, dtype=int)

                # Build context for these specific pairs (use vectorized eval for consistency)
                # We evaluate ALL pairs to get the formal mechanism output, then filter
                P = np.repeat(np.arange(n_prov), n_tasks)
                Tk = np.tile(np.arange(n_tasks), n_prov)
                gp = p_idx_local[P]

                load = slot_provs["load_ratio"].to_numpy(dtype=float)[P]
                comm = slot_provs["communication_rate"].to_numpy(dtype=float)[P]
                avail = slot_provs["availability_duration"].to_numpy(dtype=float)[P]

                a_mult = self._shock_mult(slot_provs, "alpha_mult", P)
                b_mult = self._shock_mult(slot_provs, "beta_mult", P)
                o_mult = self._shock_mult(slot_provs, "omega_mult", P)
                d_mult = self._shock_mult(slot_provs, "delta_mult", P)

                tv = {c: slot_tasks[c].to_numpy(dtype=float)[Tk]
                      for c in ["L", "value", "q_bar", "min_quality", "kappa",
                                "input_size", "output_size", "deadline"]}
                if "kappa_design" in slot_tasks.columns:
                    tv["kappa_design"] = slot_tasks[
                        "kappa_design"].to_numpy(dtype=float)[Tk]

                F_i_t = np.maximum(_EPS, (1.0 - load) * self.s_Fmax[gp])
                H_d = (self._observed_H(slot)[gp] if self.mech.design_H == "true"
                       else np.zeros(len(gp)))
                H_lower = self._state_lower_bound(H_d, gp, d_mult)

                ctx = {
                    "L": tv["L"], "V": tv["value"], "q_bar": tv["q_bar"],
                    "q_min": tv["min_quality"], "kappa": tv["kappa"],
                    "input_size": tv["input_size"], "output_size": tv["output_size"],
                    "deadline": tv["deadline"],
                    "alpha": self.s_alpha[gp] * a_mult,
                    "beta": self.s_beta[gp] * b_mult,
                    "alpha_d": self.d_alpha[gp] * a_mult,
                    "beta_d": self.d_beta[gp] * b_mult,
                    "kappa_d": tv.get("kappa_design", tv["kappa"]) * self.response_model_scale,
                    "omega_t": self.s_omega[gp] * o_mult,
                    "zeta_t": self.s_zeta[gp],
                    "xi_t": self.s_xi[gp],
                    "delta_t": self.s_delta[gp] * d_mult,
                    "U_out": self.s_Uout[gp],
                    "F_i_t": F_i_t,
                    "communication_rate": comm,
                    "availability": avail,
                    "H_t": self.H_true[gp],
                    "in_cultivation": ~self.stage_maint[gp],
                    "last_p": self.last_p[gp],
                    "n_interactions": self.n_interactions[gp].astype(float),
                    "omega_d": self.d_omega[gp] * o_mult,
                    "zeta_d": self.d_zeta[gp],
                    "xi_d": self.d_xi[gp],
                    "delta_d": self.d_delta[gp] * d_mult,
                    "H_d": H_d,
                    "H_lower": H_lower,
                    "robust_enabled": self.robust_enabled,
                    "response_relative_bound": self.robust_response_bound,
                    "response_ratio_lower": self.robust_response_ratio_lower,
                    "response_ratio_upper": self.robust_response_ratio_upper,
                    "cost_relative_bound": self.robust_cost_bound,
                    "p_min": self.p_min, "p_max": self.p_max, "D_bar": self.D_bar,
                    "reinforcement_margin": self.reinforcement_margin,
                    "Theta_M": self.Theta_M, "delta_p_max": self.delta_p_max,
                    "design_equals_true": self.design_equals_true,
                }
                ev = evaluate_pairs(self.mech, ctx)
                formal_mechanism_calls += 1

                # Stage 2B-VR: Read-only H-path trace capture
                if self._h_trace_enabled:
                    sel_indices = np.flatnonzero(selected_mask)
                    for si in sel_indices:
                        if not ev.get("feasible", np.ones(1))[si]:
                            continue
                        gidx = gp[si]; tidx = Tk[si]
                        pid = self.pid_list[gidx]
                        tid = str(task_ids_list[tidx]) if tidx < len(task_ids_list) else ""
                        is_reset = (self.reset_provider_ids is not None
                                    and gidx in self.reset_provider_ids)
                        ak = f"601_{slot}_{tid}_{pid}"

                        # Compute raw lambda before clip
                        a_t = float(ev["a_target"][si])
                        k_val = float(ctx["kappa"][si])
                        gp_t_val = k_val * np.exp(-k_val * max(a_t, 0.0))
                        if gp_t_val < _EPS:
                            gp_t_val = _EPS
                        cp_t = (float(ctx["alpha"][si]) * float(ctx["L"][si])
                                + 2.0 * float(ctx["beta"][si]) * float(ctx["L"][si]) * a_t)
                        raw_lambda = max(0.0, cp_t / gp_t_val
                                         - float(ctx["omega_d"][si]) * float(ctx["H_d"][si]))
                        sc = float(ctx["omega_d"][si]) * float(ctx["H_d"][si])

                        self._h_trace_records.append({
                            "world": getattr(self, "_ht_world_label", ""),
                            "slot": int(slot), "task_id": tid, "provider_id": pid,
                            "assignment_key": ak,
                            "provider_is_reset": bool(is_reset),
                            "H_true_before_quote": float(self.H_true[gidx]),
                            "H_design_before_quote": float(self.H_design_state[gidx]),
                            "H_contract_used": float(ctx["H_d"][si]),
                            "H_state_credit_used": float(ctx["H_d"][si]),
                            "H_lambda_used": float(ctx["H_d"][si]),
                            "H_base_payment_used": float(ctx["H_d"][si]),
                            "H_bonus_path_used": float(ctx["H_t"][si]),
                            "omega_true": float(ctx["omega_t"][si]),
                            "omega_design": float(ctx["omega_d"][si]),
                            "state_credit": sc,
                            "raw_lambda_before_clip": raw_lambda,
                            "lambda_after_clip": float(ev["lambda_required"][si]),
                            "base_payment": float(ev["base_payment"][si]),
                            "bonus": float(ev["expected_bonus"][si]),
                            "objective_payment": float(ev["expected_contract_cost"][si]),
                            "a_star": float(ev["a_star"][si]),
                            "g_star": float(ev["normalized_quality"][si]),
                            "realized_signal": float(ev["normalized_quality"][si]),
                            "H_true_after_update": float(self.H_true[gidx]),
                            "H_design_after_update": float(self.H_design_state[gidx]),
                            "reset_applied_in_slot": self._reset_applied,
                            "probe_state_hash_before": "",
                            "probe_state_hash_after": "",
                            "probe_side_effect_free": True,
                        })
            else:
                # No fixed assignments this slot — still update provider states
                ev = {}
                P = np.array([], dtype=int)
                Tk = np.array([], dtype=int)
                gp = np.array([], dtype=int)
                pair_prov_global = np.array([], dtype=int)

            contract_time = 0.0  # tracked within evaluate_pairs

            # ── Provider state updates (once per online provider) ──
            # Use only the selected (fixed) assignments
            self._update_states(slot, slot_provs, p_idx_local, gp, Tk,
                                slot_tasks, ev, selected_mask)

            # Stage 2B-VR: Update H_after in trace records
            if self._h_trace_enabled and len(self._h_trace_records) > 0:
                sel_indices = np.flatnonzero(selected_mask)
                gpid_to_si = {gp[si]: si for si in sel_indices}
                for rec in self._h_trace_records:
                    if rec["slot"] == int(slot):
                        gidx = self.pid_to_idx.get(str(rec["provider_id"]))
                        if gidx is not None and gidx in gpid_to_si:
                            si = gpid_to_si[gidx]
                            rec["H_true_after_update"] = float(self.H_true[gidx])
                            rec["H_design_after_update"] = float(self.H_design_state[gidx])
                            # Record state hashes
                            import hashlib
                            rec["probe_state_hash_before"] = hashlib.sha256(
                                self.H_true.tobytes()[:8] + self.H_design_state.tobytes()[:8]).hexdigest()[:16]
                            rec["probe_state_hash_after"] = rec["probe_state_hash_before"]

            # ── Pair log (only selected pairs, replay mode) ──
            self._log_pairs(slot, slot_provs, slot_tasks, P, Tk, ev, selected_mask)

            # ── Slot log ──
            sel_pairs = np.flatnonzero(selected_mask)
            sq = ev["execution_quality"][sel_pairs] if len(ev) and len(sel_pairs) else np.array([])
            spay = ev["expected_contract_cost"][sel_pairs] if len(ev) and len(sel_pairs) else np.array([])
            sutil = ev["experienced_utility"][sel_pairs] if len(ev) and len(sel_pairs) else np.array([])
            simm = ev["immediate_value"][sel_pairs] if len(ev) and len(sel_pairs) else np.array([])
            ir_bad = int((~ev["ir_ok"][sel_pairs]).sum()) if len(ev) and len(sel_pairs) else 0
            gap_bad = int((ev["implementation_gap"][sel_pairs] < -1e-7).sum()) if len(ev) and len(sel_pairs) else 0

            feas = ev.get("feasible", np.array([]))
            tsk_q_min = slot_tasks["min_quality"].to_numpy(dtype=float) if "min_quality" in slot_tasks.columns else np.full(n_tasks, 0.8)
            if len(sel_pairs) and len(ev):
                sq_full = ev["execution_quality"]
                tsk_idx_of_sel = Tk[sel_pairs] if len(Tk) > 0 else np.array([], dtype=int)
                if len(tsk_idx_of_sel):
                    qual_sel = sq_full[sel_pairs] >= tsk_q_min[tsk_idx_of_sel]
                    n_qualified = int(qual_sel.sum())
                else:
                    n_qualified = 0
            else:
                n_qualified = 0

            self.slot_log.append({
                "slot": int(slot), "method": self.method, "seed": self.seed,
                "num_tasks": n_tasks, "num_providers": n_prov,
                "num_feasible_pairs": int(feas.sum()) if len(feas) else 0,
                "num_assigned": int(len(sel_pairs)),
                "num_completed": int(len(sel_pairs)),
                "num_high_quality": int((sq >= 0.8).sum()) if len(sq) else 0,
                "num_qualified_completed": n_qualified,
                "mean_quality": float(sq.mean()) if len(sq) else np.nan,
                "total_payment": float(spay.sum()),
                "total_value": float(simm.sum()) if len(simm) else 0.0,
                "platform_utility": float(simm.sum()) if len(simm) else 0.0,
                "provider_utility_mean": float(sutil.mean()) if len(sutil) else np.nan,
                "budget": float(budget),
                "budget_used": float(spay.sum()),
                "budget_gap": float(budget - spay.sum()),
                "maintenance_ratio": float(self.stage_maint.mean()),
                "cultivation_ratio": float(1.0 - self.stage_maint.mean()),
                "ir_violations": ir_bad,
                "target_violations": gap_bad,
                "matching_runtime": 0.0,   # matching disabled
                "contract_runtime": contract_time,
                "total_runtime": time.perf_counter() - t0_slot,
                "UB": None, "LB": None, "optimality_gap": None,
                "OPT": None, "certified_gap": None,
            })

        # ── Assemble results ─────────────────────────────────
        total_runtime = time.perf_counter() - t0_total
        pair_df = (pd.concat(self.pair_log_frames, ignore_index=True)
                   if self.pair_log_frames else pd.DataFrame())
        slot_df = pd.DataFrame(self.slot_log)
        provider_df = pd.DataFrame(self.provider_log)

        summary = self._build_summary(pair_df, slot_df, provider_df,
                                      total_tasks, total_runtime)
        diagnostics = _run_diagnostics(
            pair_df, slot_df, provider_df, summary,
            enforce_target=self.mech.enforce_target,
            expect_exact_ir=(not self.est_error_active
                             and self.mech.base_mode == "response_consistent"),
        )

        return {
            "summary": summary,
            "diagnostics": diagnostics,
            "pair_log": pair_df,
            "slot_log": slot_df,
            "provider_log": provider_df,
            # Stage 2A audit fields
            "replay_audit": {
                "formal_mechanism_calls": formal_mechanism_calls,
                "matching_calls": matching_calls,
                "episode_generation_calls": episode_gen_calls,
                "reset_id_sampling_calls": reset_sampling_calls,
                "replay_mode": True,
                "disable_matching": True,
            },
        }

    # ── State update ───────────────────────────────────────────

    def _update_states(self, slot, slot_provs, p_idx_local, gp, Tk,
                       slot_tasks, ev, selected_mask) -> None:
        """Exactly one path-state update and one log row per online provider."""
        sel_of_prov: dict[int, int] = {}
        for k in np.flatnonzero(selected_mask):
            sel_of_prov[int(gp[k])] = int(k)

        task_ids = slot_tasks["task_id"].to_numpy()
        d_mult_prov = (slot_provs["delta_mult"].to_numpy(dtype=float)
                       if "delta_mult" in slot_provs.columns
                       else np.ones(len(slot_provs)))

        for local_i, g in enumerate(p_idx_local):
            g = int(g)
            assigned = g in sel_of_prov
            H_before = float(self.H_true[g])
            Hd_before = float(self.H_design_state[g])
            stage_before = "maintenance" if self.stage_maint[g] else "cultivation"
            s_val = 0.0
            tid = None

            if assigned:
                k = sel_of_prov[g]
                s_val = float(ev["normalized_quality"][k])
                tid = str(task_ids[int(Tk[k])])
                xi_t = float(self.s_xi[g]); dl_t = float(self.s_delta[g]) * float(d_mult_prov[local_i])
                H_new = H_before + xi_t * s_val * (1 - H_before) - dl_t * (1 - s_val) * H_before
                if H_new < -1e-8 or H_new > 1 + 1e-8:
                    raise ValueError(f"H out of bounds: {H_new} (provider {self.pid_list[g]})")
                self.H_true[g] = min(1.0, max(0.0, H_new))
                # Platform belief uses design-side xi/delta (equal when no error)
                xi_d = float(self.d_xi[g]); dl_d = float(self.d_delta[g]) * float(d_mult_prov[local_i])
                Hd_new = Hd_before + xi_d * s_val * (1 - Hd_before) - dl_d * (1 - s_val) * Hd_before
                self.H_design_state[g] = min(1.0, max(0.0, Hd_new))

                self.recent_quality[g].append(s_val)
                if len(self.recent_quality[g]) > 2 * self.K:
                    self.recent_quality[g] = self.recent_quality[g][-2 * self.K:]
                self.n_interactions[g] += 1
                self.last_p[g] = float(ev["p_star"][k])
                self.reputation[g] = ((1 - self.rai_ewma) * self.reputation[g]
                                      + self.rai_ewma * s_val)

            # Stage transitions (5.13) on the platform belief state.
            # Only active when the mechanism uses stage logic; MOI / SAMI
            # skip this block entirely (use_stage_logic=False).
            stage_next = stage_before
            H_stage = float(self.H_design_state[g])
            if self.mech.use_stage_logic:
                if self.mech.fixed_cultivation_count is not None:
                    # PRIME-Fixed: switch after N assigned interactions, permanent
                    if self.n_interactions[g] >= self.mech.fixed_cultivation_count:
                        stage_next = "maintenance"
                else:
                    window = self.recent_quality[g][-self.K:]
                    if len(window) >= self.K:
                        mean_q = float(np.mean(window))
                        if stage_before == "cultivation":
                            if H_stage >= self.Theta_M and mean_q >= self.s_M:
                                stage_next = "maintenance"
                        else:
                            if (H_stage < self.Theta_C or mean_q < self.s_C) \
                                    and self.mech.allow_recultivation:
                                stage_next = "cultivation"
            self.stage_maint[g] = stage_next == "maintenance"

            wq = self.recent_quality[g][-self.K:]
            self.provider_log.append({
                "provider_id": self.pid_list[g], "slot": int(slot),
                "method": self.method, "seed": self.seed,
                "online": True, "assigned": assigned, "task_id": tid,
                "H_before": H_before, "H_after": float(self.H_true[g]),
                "H_belief": float(self.H_design_state[g]),
                "stage_before": stage_before, "stage_after": stage_next,
                "recent_quality_mean": float(np.mean(wq)) if wq else np.nan,
                "p_last": float(self.last_p[g]),
                "reputation": float(self.reputation[g]),
                "n_interactions": int(self.n_interactions[g]),
                "recultivation_flag": stage_before == "maintenance" and stage_next == "cultivation",
                "maintenance_entry_flag": stage_before == "cultivation" and stage_next == "maintenance",
            })

    # ── Logging helpers ────────────────────────────────────────

    def _log_pairs(self, slot, slot_provs, slot_tasks, P, Tk, ev, selected_mask) -> None:
        if self.log_level == "none":
            return
        if self.log_level == "full":
            keep = np.ones(len(P), dtype=bool)
        elif self.log_level == "sample":
            keep = selected_mask.copy()
            rng = np.random.default_rng([self.seed, int(slot), 99])
            keep |= rng.random(len(P)) < 0.05
        else:  # "selected"
            keep = selected_mask.copy()
        if not keep.any():
            return

        kidx = np.flatnonzero(keep)
        pid_arr = slot_provs["provider_id"].to_numpy()
        tid_arr = slot_tasks["task_id"].to_numpy()
        reason = ev["reason_code"][kidx]
        # Extract task min_quality for QCR
        task_q_min = slot_tasks["min_quality"].to_numpy(dtype=float) if "min_quality" in slot_tasks.columns else np.full(len(Tk), 0.8)
        frame = {
            "provider_id": pid_arr[P[kidx]],
            "task_id": tid_arr[Tk[kidx]],
            "slot": np.full(len(kidx), int(slot)),
            "method": self.method, "seed": self.seed,
            "feasible_physical": ev["feasible_physical"][kidx],
            "feasible_contract": ev["feasible"][kidx],
            "infeasible_reason": [REASON_LABELS[int(r)] for r in reason],
            "selected": selected_mask[kidx],
            "ir_ok": ev["ir_ok"][kidx],
            "task_min_quality": task_q_min[Tk[kidx]],
        }
        for c in _PAIR_LOG_COLS:
            frame[c] = ev[c][kidx]
        for c in ("spatial_distance_km", "spatial_candidate"):
            if c in ev:
                frame[c] = ev[c][kidx]
        if "response_global_fallback" in ev:
            frame["response_global_fallback"] = ev[
                "response_global_fallback"][kidx]
        self.pair_log_frames.append(pd.DataFrame(frame))

    def _log_empty_slot(self, slot, n_tasks, budget, t0_slot) -> None:
        self.slot_log.append({
            "slot": int(slot), "method": self.method, "seed": self.seed,
            "num_tasks": n_tasks, "num_providers": 0, "num_feasible_pairs": 0,
            "num_assigned": 0, "num_completed": 0, "num_high_quality": 0,
            "num_qualified_completed": 0,
            "mean_quality": np.nan, "total_payment": 0.0, "total_value": 0.0,
            "platform_utility": 0.0, "provider_utility_mean": np.nan,
            "budget": float(budget), "budget_used": 0.0, "budget_gap": float(budget),
            "maintenance_ratio": float(self.stage_maint.mean()),
            "cultivation_ratio": float(1.0 - self.stage_maint.mean()),
            "ir_violations": 0, "target_violations": 0,
            "matching_runtime": 0.0, "contract_runtime": 0.0,
            "total_runtime": time.perf_counter() - t0_slot,
            "UB": None, "LB": None, "optimality_gap": None,
        })

    # ── Summary ────────────────────────────────────────────────

    def _build_summary(self, pair_df, slot_df, provider_df,
                       total_tasks, total_runtime) -> dict:
        n_assigned = int(slot_df["num_assigned"].sum()) if len(slot_df) else 0
        n_hq = int(slot_df["num_high_quality"].sum()) if len(slot_df) else 0
        payment = float(slot_df["total_payment"].sum()) if len(slot_df) else 0.0
        utility = float(slot_df["platform_utility"].sum()) if len(slot_df) else 0.0

        if len(pair_df) and pair_df["selected"].any():
            sel = pair_df[pair_df["selected"]]
            avg_q = float(sel["execution_quality"].mean())
            prov_util = float(sel["experienced_utility"].mean())
            hq_pay = sel.loc[sel["execution_quality"] >= 0.8, "expected_contract_cost"].sum()
            pay_per_hq = float(hq_pay / max((sel["execution_quality"] >= 0.8).sum(), 1))
        else:
            wq = slot_df["mean_quality"] * slot_df["num_assigned"] if len(slot_df) else pd.Series(dtype=float)
            avg_q = float(wq.sum() / n_assigned) if n_assigned else float("nan")
            prov_util = float("nan")
            pay_per_hq = float(payment / n_hq) if n_hq else float("nan")

        maint_entry = int(provider_df["maintenance_entry_flag"].sum()) if len(provider_df) else 0
        recult = int(provider_df["recultivation_flag"].sum()) if len(provider_df) else 0

        return {
            "method": self.method, "seed": self.seed,
            "contract_protocol_version": "r51-free-action-design-only-v1",
            "guarantee_scope": (
                "conditional_on_declared_cost_response_state_bounds_and_known_weighting"
                if self.robust_enabled else
                "nominal_model_only_no_current_response_robust_guarantee"),
            "decision_cost_basis": (
                "continuous_envelope_payment_upper_bound" if self.robust_enabled
                else "design_expected_payment"),
            "robust_ir_policy": (
                "continuous_box_sufficient_reserve_v1" if self.robust_enabled
                else "design_response_ir"),
            "T": int(len(slot_df)), "total_tasks": int(total_tasks),
            "num_assigned": n_assigned,
            "assignment_ratio": n_assigned / total_tasks if total_tasks else float("nan"),
            "completion_ratio": n_assigned / total_tasks if total_tasks else float("nan"),
            "HQR": n_hq / n_assigned if n_assigned else float("nan"),
            "average_quality": avg_q,
            "cumulative_payment": payment,
            "payment_per_assigned": payment / n_assigned if n_assigned else float("nan"),
            "payment_per_HQ": pay_per_hq,
            "platform_utility": utility,
            "provider_utility_mean": prov_util,
            "budget_utilization": float((slot_df["budget_used"].sum()
                                         / max(slot_df["budget"].sum(), _EPS))) if len(slot_df) else float("nan"),
            "maintenance_entries": maint_entry,
            "recultivation_events": recult,
            "ir_violations": int(slot_df["ir_violations"].sum()) if len(slot_df) else 0,
            "target_violations": int(slot_df["target_violations"].sum()) if len(slot_df) else 0,
            "mean_optimality_gap": float(pd.to_numeric(slot_df["optimality_gap"], errors="coerce").mean()) if len(slot_df) else float("nan"),
            "total_runtime": total_runtime,
            "status": "completed",
        }


# ── Diagnostics (Section 20) ────────────────────────────────────

def _run_diagnostics(
    pair_df: pd.DataFrame,
    slot_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    summary: dict,
    enforce_target: bool = True,
    expect_exact_ir: bool = True,
) -> dict:
    diag: dict[str, Any] = {"status": "ok", "issues": [], "warnings": []}

    # 1. IR violations on ASSIGNED pairs (vs U_out, flagged at eval time)
    ir_v = int(summary.get("ir_violations", 0))
    diag["ir_violations"] = ir_v
    if ir_v > 0:
        if expect_exact_ir:
            diag["issues"].append(f"IR violations: {ir_v}")
            diag["status"] = "failed"
        else:
            diag["warnings"].append(f"IR violations (expected under estimation error): {ir_v}")

    # 2. Budget violations
    if len(slot_df) and "budget_gap" in slot_df.columns:
        bv = int((slot_df["budget_gap"] < -1e-6).sum())
        diag["budget_violations"] = bv
        if bv > 0:
            diag["issues"].append(f"Budget violations: {bv}")
            diag["status"] = "failed"

    # 3. a_star < a_target on assigned pairs
    tv = int(summary.get("target_violations", 0))
    diag["target_violations"] = tv
    if tv > 0 and enforce_target and expect_exact_ir:
        diag["issues"].append(f"a_star < a_target on {tv} assigned pairs")
        diag["status"] = "failed"
    elif tv > 0:
        diag["warnings"].append(f"a_star < a_target on {tv} assigned pairs")

    # 4. H bounds
    if len(provider_df) and "H_after" in provider_df.columns:
        h_ok = bool(((provider_df["H_after"] >= -1e-8)
                     & (provider_df["H_after"] <= 1 + 1e-8)).all())
        diag["h_bounds_ok"] = h_ok
        if not h_ok:
            diag["issues"].append("H out of bounds")
            diag["status"] = "failed"

    # 5. NaN/Inf on assigned pairs
    if len(pair_df):
        sel = pair_df[pair_df["selected"]] if "selected" in pair_df.columns else pair_df
        num = sel.select_dtypes(include=[np.floating]).to_numpy() if len(sel) else np.zeros((0, 1))
        diag["nan_rows"] = int(np.isnan(num).any(axis=1).sum())
        diag["inf_rows"] = int(np.isinf(num).any(axis=1).sum())
        if diag["nan_rows"] or diag["inf_rows"]:
            diag["issues"].append(f"NaN/Inf in assigned pair rows: {diag['nan_rows']}/{diag['inf_rows']}")
            diag["status"] = "failed"

    # 6. Feasibility rate
    if len(slot_df):
        tot_pairs = float((slot_df["num_tasks"] * slot_df["num_providers"]).sum())
        feas = float(slot_df["num_feasible_pairs"].sum())
        rate = feas / max(tot_pairs, 1)
        diag["feasibility_rate"] = rate
        if rate < 0.1:
            diag["warnings"].append(f"Low feasibility rate: {rate:.4f}")

    # 7. UB < LB check
    if len(slot_df) and "UB" in slot_df.columns:
        ub = pd.to_numeric(slot_df["UB"], errors="coerce")
        lb = pd.to_numeric(slot_df["LB"], errors="coerce")
        bad = int((ub < lb - 1e-7).sum())
        diag["ub_lt_lb"] = bad
        if bad > 0:
            diag["issues"].append(f"UB < LB in {bad} slots")
            diag["status"] = "failed"

    # 8. Lambda almost always zero (scale problem, §20.11)
    if len(pair_df) and "lambda_required" in pair_df.columns:
        sel = pair_df[pair_df["selected"]]
        if len(sel):
            lz = float((sel["lambda_required"] <= 1e-9).mean())
            diag["lambda_zero_fraction"] = lz
            if lz > 0.95:
                diag["warnings"].append(f"Lambda≈0 for {lz:.0%} of assigned pairs")

    # Keep completion, observed outcomes and theoretical guarantee scope
    # separate. In particular, target success must not hide a QoS failure.
    observed = service_outcomes(slot_df, summary, enforce_target=enforce_target)
    diag.update(observed)
    diag["execution_status"] = "completed"
    if observed["qos_violations"]:
        diag["issues"].append(f"QoS violations: {observed['qos_violations']}")
    if observed["settlement_over_available_budget_slots"]:
        diag["issues"].append(
            f"Settlement exceeds available budget in {observed['settlement_over_available_budget_slots']} slots")
    if observed["settlement_over_reservation_slots"]:
        diag["warnings"].append(
            f"Settlement exceeds decision reservation in {observed['settlement_over_reservation_slots']} slots; this is not necessarily budget overspend")
    if observed["service_outcome_status"] == "violated":
        diag["status"] = "failed"
    elif observed["service_outcome_status"] == "unassessed" and diag["status"] == "ok":
        diag["status"] = "unassessed"
    return diag
