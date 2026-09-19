"""Synthetic dataset generator (Section 8.2).

Workload patterns: stationary, burst, dynamic.
Controls: slots, provider/task counts, behavioural fraction rho_B,
provider lifetime/churn, parameter distributions, and shock scenarios
(Experiment G).

Shock config (cfg["shocks"]: list of dicts), all applied inside
[start, end) slots:
  {type: cost_shock,        start, end, magnitude, fraction}  alpha/beta ×(1+m)
  {type: workload_burst,    start, end, magnitude}            task count ×(1+m)
  {type: provider_churn,    start, end, magnitude}            per-slot leave prob = m
  {type: behavior_decay,    start, end, magnitude, fraction}  delta ×(1+m), omega ×(1-m)
  {type: comm_degradation,  start, end, magnitude}            comm rate ×(1-m)

Shock effects are baked into the DATASET so every method sees identical
realisations for the same seed (§2.5).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator


def _shocks_of(cfg: dict, stype: str) -> list[dict]:
    return [s for s in (cfg.get("shocks") or []) if s.get("type") == stype]


def _active(shocks: list[dict], t: int) -> list[dict]:
    return [s for s in shocks if s.get("start", 0) <= t < s.get("end", 0)]


def generate_synthetic_episode(
    cfg: dict[str, Any],
    seed: int,
    pattern: str = "stationary",
    episode_id: str = "synth_001",
) -> dict[str, pd.DataFrame]:
    """Generate one synthetic episode.

    Returns dict with keys: tasks, providers, provider_static, meta.
    """
    rng = np.random.default_rng(seed)

    sim_cfg = cfg.get("simulation", {}) or {}
    prov_cfg = cfg.get("providers", {}) or {}
    task_cfg = cfg.get("tasks", {}) or {}

    T = int(sim_cfg.get("T", cfg.get("T", 1000)))
    N_mean = int(prov_cfg.get("N_mean", cfg.get("N_mean", 100)))
    M_mean = int(task_cfg.get("M_mean", cfg.get("M_mean", 80)))
    rho_B = float(prov_cfg.get("behavioral_fraction", 0.70))
    lifetime_mean = prov_cfg.get("lifetime_mean")  # None -> persistent

    max_rate = prov_cfg.get("max_processing_rate", [5.0, 20.0])
    alpha_r = prov_cfg.get("alpha", [0.05, 0.20])
    beta_r = prov_cfg.get("beta", [0.02, 0.10])
    zeta_r = prov_cfg.get("zeta", [0.65, 0.95])
    omega_r = prov_cfg.get("omega", [0.05, 0.25])
    xi_r = prov_cfg.get("xi", [0.05, 0.12])
    delta_r = prov_cfg.get("delta", [0.01, 0.05])
    U_out = float(prov_cfg.get("outside_option", 0.01))

    static_rows: list[dict] = []
    pid_counter = 0

    def spawn_provider(birth_slot: int) -> dict:
        nonlocal pid_counter
        pid = f"P{pid_counter:05d}"
        pid_counter += 1
        behavioral = bool(rng.random() < rho_B)
        if lifetime_mean:
            lifetime = int(max(1, rng.geometric(1.0 / float(lifetime_mean))))
        else:
            lifetime = T + 1
        row = {
            "episode_id": episode_id,
            "provider_id": pid,
            "max_processing_rate": float(rng.uniform(*max_rate)),
            "alpha": float(rng.uniform(*alpha_r)),
            "beta": float(rng.uniform(*beta_r)),
            "zeta": float(rng.uniform(*zeta_r)) if behavioral else 1.0,
            "omega": float(rng.uniform(*omega_r)) if behavioral else 0.0,
            "xi": float(rng.uniform(*xi_r)),
            "delta": float(rng.uniform(*delta_r)),
            "outside_option": U_out,
            "behavioral": behavioral,
            "birth_slot": birth_slot,
            "death_slot": birth_slot + lifetime,
        }
        static_rows.append(row)
        return row

    active: list[dict] = [spawn_provider(0) for _ in range(N_mean)]

    # ── Task-side ranges ──────────────────────────────────────
    cpu_r = task_cfg.get("cpu_cycles", [0.1, 1.0])
    in_r = task_cfg.get("input_size", [0.1, 2.0])
    out_r = task_cfg.get("output_size", [0.05, 1.0])
    dlf_r = task_cfg.get("deadline_factor", [1.2, 2.5])
    qmin_r = task_cfg.get("min_quality", [0.65, 0.85])
    qbar_r = task_cfg.get("q_bar", [0.90, 1.00])
    kappa_r = task_cfg.get("kappa", [1.0, 5.0])
    val_r = task_cfg.get("value_base", [1.0, 5.0])

    if pattern == "burst":
        base_count = lambda t: M_mean * (1.5 if t % 100 < 20 else 0.7)  # noqa: E731
    elif pattern == "dynamic":
        base_count = lambda t: M_mean * (0.75 + 0.5 * math.sin(2 * math.pi * t / 200))  # noqa: E731
    else:
        base_count = lambda t: float(M_mean)  # noqa: E731

    sh_cost = _shocks_of(cfg, "cost_shock")
    sh_load = _shocks_of(cfg, "workload_burst")
    sh_churn = _shocks_of(cfg, "provider_churn")
    sh_decay = _shocks_of(cfg, "behavior_decay")
    sh_comm = _shocks_of(cfg, "comm_degradation")
    any_mult_shock = bool(sh_cost or sh_decay)

    # Pre-assign shock membership per provider id (stable across slots)
    shock_membership: dict[str, dict[str, bool]] = {}

    def in_shock_group(pid: str, shock: dict, key: str) -> bool:
        frac = float(shock.get("fraction", 1.0))
        m = shock_membership.setdefault(pid, {})
        if key not in m:
            m[key] = bool(rng.random() < frac)
        return m[key]

    tasks_rows: list[dict] = []
    dyn_rows: list[dict] = []
    task_counter = 0

    for slot in range(T):
        # ── churn: natural deaths + shock-forced leaves ──────
        churn_p = max((float(s.get("magnitude", 0.0)) for s in _active(sh_churn, slot)), default=0.0)
        survivors = []
        for prov in active:
            dead = slot >= prov["death_slot"]
            if not dead and churn_p > 0 and rng.random() < churn_p:
                dead = True
            if dead:
                survivors.append(spawn_provider(slot))
            else:
                survivors.append(prov)
        active = survivors

        # ── online subset ────────────────────────────────────
        online_ratio = 0.7 + 0.3 * rng.random()
        n_online = max(5, int(len(active) * online_ratio))
        online_idx = rng.choice(len(active), size=min(n_online, len(active)), replace=False)

        comm_mult = 1.0
        for s in _active(sh_comm, slot):
            comm_mult *= max(0.05, 1.0 - float(s.get("magnitude", 0.0)))

        for i in online_idx:
            prov = active[int(i)]
            pid = prov["provider_id"]
            row = {
                "episode_id": episode_id,
                "slot": slot,
                "provider_id": pid,
                "load_ratio": float(rng.uniform(0.0, 0.5)),
                "communication_rate": float(rng.uniform(5.0, 20.0)) * comm_mult,
                "availability_duration": float(rng.uniform(2.0, 10.0)),
            }
            if any_mult_shock:
                am = bm = dm = om = 1.0
                for s in _active(sh_cost, slot):
                    if in_shock_group(pid, s, f"cost{s.get('start')}"):
                        am *= 1.0 + float(s.get("magnitude", 0.0))
                        bm *= 1.0 + float(s.get("magnitude", 0.0))
                for s in _active(sh_decay, slot):
                    if in_shock_group(pid, s, f"decay{s.get('start')}"):
                        dm *= 1.0 + float(s.get("magnitude", 0.0))
                        om *= max(0.0, 1.0 - float(s.get("magnitude", 0.0)))
                row.update({"alpha_mult": am, "beta_mult": bm,
                            "delta_mult": dm, "omega_mult": om})
            dyn_rows.append(row)

        # ── tasks ────────────────────────────────────────────
        m_slot = base_count(slot)
        for s in _active(sh_load, slot):
            m_slot *= 1.0 + float(s.get("magnitude", 0.0))
        m_slot = max(1, int(round(m_slot)))

        for _ in range(m_slot):
            raw_dur = float(rng.uniform(0.5, 3.0))
            tasks_rows.append({
                "episode_id": episode_id,
                "slot": slot,
                "task_id": f"T{task_counter:06d}",
                "L": float(rng.uniform(*cpu_r)),
                "input_size": float(rng.uniform(*in_r)),
                "output_size": float(rng.uniform(*out_r)),
                "deadline": float(rng.uniform(*dlf_r)) * raw_dur,
                "min_quality": float(rng.uniform(*qmin_r)),
                "q_bar": float(rng.uniform(*qbar_r)),
                "kappa": float(rng.uniform(*kappa_r)),
                "value": float(rng.uniform(*val_r)),
                "raw_duration": raw_dur,
            })
            task_counter += 1

    df_tasks = pd.DataFrame(tasks_rows)
    df_dyn = pd.DataFrame(dyn_rows)
    df_static = pd.DataFrame(static_rows)

    meta = {
        "source": "synthetic",
        "pattern": pattern,
        "episode_id": episode_id,
        "seed": seed,
        "T": T,
        "n_providers_total": len(df_static),
        "n_tasks_total": len(df_tasks),
        "behavioral_fraction": rho_B,
        "lifetime_mean": lifetime_mean,
        "shocks": cfg.get("shocks") or [],
        "slots": T,
    }

    return {
        "tasks": df_tasks,
        "providers": df_dyn,
        "provider_static": df_static,
        "meta": meta,
    }
