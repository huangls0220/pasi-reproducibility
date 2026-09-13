#!/usr/bin/env python3
"""Paper-quality figure suite (Section 15).

matplotlib only (no seaborn); every figure saved as PDF + PNG, with the
plotted data exported to results/figures/data/<name>.csv.  Captions and
titles describe variables only — no conclusions are hardcoded.

Usage:
  python scripts/make_figures.py --experiment overall
  python scripts/make_figures.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "results" / "figures"
DATA_DIR = FIG_DIR / "data"

METHOD_COLORS = {
    "PRIME": "#1f77b4", "MOI": "#ff7f0e", "FR": "#2ca02c", "DP": "#d62728",
    "RAI": "#9467bd", "LDI": "#8c564b", "PRIME-R": "#17becf",
    "PRIME-w_o-PD": "#17becf", "PRIME-w_o-PW": "#bcbd22",
    "PRIME-w_o-LT": "#e377c2", "PRIME-w_o-RC": "#7f7f7f", "PRIME-Fixed": "#b5bd61",
    "PRIME-w/o-PD": "#17becf", "PRIME-w/o-PW": "#bcbd22",
    "PRIME-w/o-LT": "#e377c2", "PRIME-w/o-RC": "#7f7f7f",
}
METHOD_MARKERS = {"PRIME": "o", "MOI": "s", "FR": "D", "DP": "^",
                  "RAI": "v", "LDI": "<", "PRIME-R": ">"}
METHOD_ORDER = ["PRIME", "MOI", "FR", "DP", "RAI", "LDI",
                "PRIME-w/o-PD", "PRIME-w/o-PW", "PRIME-w/o-LT",
                "PRIME-w/o-RC", "PRIME-Fixed", "PRIME-R"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def _mcolor(m: str) -> str:
    return METHOD_COLORS.get(m, "#999999")


def _order(methods) -> list[str]:
    return sorted(methods, key=lambda m: (METHOD_ORDER.index(m)
                                          if m in METHOD_ORDER else 99, m))


def _save(fig, name: str, data: pd.DataFrame | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    if data is not None:
        data.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"  {name}.pdf/.png" + ("" if data is None else " (+data csv)"))


def _by_seed(experiment: str, dataset: str = "synthetic") -> pd.DataFrame:
    p = ROOT / "results" / "summaries" / f"{experiment}_{dataset}_by_seed.csv"
    if not p.exists():
        sys.exit(f"Missing {p}; run aggregate_results.py --experiment {experiment} first")
    return pd.read_csv(p)


def _mean_ci_cols(g: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(g, errors="coerce").dropna().to_numpy()
    if len(v) == 0:
        return np.nan, 0.0
    if len(v) == 1:
        return float(v[0]), 0.0
    from scipy import stats
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * se)


def _bars(ax, df, metric, ylabel, methods=None):
    methods = methods or _order(df["method"].unique())
    means, cis = [], []
    for m in methods:
        mu, ci = _mean_ci_cols(df.loc[df["method"] == m, metric])
        means.append(mu); cis.append(ci)
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=cis, capsize=2.5,
           color=[_mcolor(m) for m in methods], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("PRIME-", "P-") for m in methods],
                       rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)
    return pd.DataFrame({"method": methods, f"{metric}_mean": means, f"{metric}_ci": cis})


def _load_slot_logs(experiment, variation, methods, dataset="synthetic",
                    columns=None) -> dict[str, pd.DataFrame]:
    out = {}
    for m in methods:
        mdir = (ROOT / "results" / experiment / dataset / variation
                / m.replace("/", "_"))
        frames = []
        for f in sorted(mdir.glob("seed_*/slot_log.parquet")):
            df = pd.read_parquet(f, columns=columns)
            df["seed"] = int(f.parent.name.replace("seed_", ""))
            frames.append(df)
        if frames:
            out[m] = pd.concat(frames, ignore_index=True)
    return out


def _load_provider_logs(experiment, variation, methods, dataset="synthetic",
                        columns=None) -> dict[str, pd.DataFrame]:
    out = {}
    for m in methods:
        mdir = (ROOT / "results" / experiment / dataset / variation
                / m.replace("/", "_"))
        frames = []
        for f in sorted(mdir.glob("seed_*/provider_log.parquet")):
            df = pd.read_parquet(f, columns=columns)
            df["seed"] = int(f.parent.name.replace("seed_", ""))
            frames.append(df)
        if frames:
            out[m] = pd.concat(frames, ignore_index=True)
    return out


def _ts_plot(ax, per_seed: pd.DataFrame, method: str, window: int = 25):
    """per_seed: columns [slot, seed, y]. Mean ± 95% CI across seeds, rolled."""
    piv = per_seed.pivot_table(index="slot", columns="seed", values="y")
    mean = piv.mean(axis=1)
    n = piv.notna().sum(axis=1).clip(lower=1)
    se = piv.std(axis=1, ddof=1) / np.sqrt(n)
    ci = 1.96 * se
    mean_r = mean.rolling(window, min_periods=1).mean()
    ci_r = ci.rolling(window, min_periods=1).mean()
    c = _mcolor(method)
    ax.plot(mean_r.index, mean_r, color=c, lw=1.1, label=method)
    ax.fill_between(mean_r.index, mean_r - ci_r, mean_r + ci_r, color=c, alpha=0.15)
    return pd.DataFrame({"slot": mean_r.index, "method": method,
                         "mean": mean_r.to_numpy(), "ci95": ci_r.to_numpy()})


# ═════════════════ Experiment figures ═══════════════════════════

def fig_overall(dataset="synthetic"):
    df = _by_seed("overall", dataset)
    for var in sorted(df["variation"].unique()):
        sub = df[df["variation"] == var]
        tag = "" if var == "stationary" else f"_{var}"
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
        d1 = _bars(axes[0], sub, "HQR", "High-quality ratio")
        axes[0].set_ylim(0, 1.0)
        d2 = _bars(axes[1], sub, "cumulative_payment", "Cumulative payment")
        d3 = _bars(axes[2], sub, "platform_utility", "Platform utility")
        fig.tight_layout()
        _save(fig, f"fig_overall_main{tag}", pd.concat([d1, d2, d3], axis=1))

        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
        d1 = _bars(axes[0], sub, "payment_per_HQ", "Payment per HQ task")
        d2 = _bars(axes[1], sub, "completion_ratio", "Task completion ratio")
        axes[1].set_ylim(0, 1.0)
        d3 = _bars(axes[2], sub, "provider_utility_mean", "Mean provider utility")
        fig.tight_layout()
        _save(fig, f"fig_overall_costs{tag}", pd.concat([d1, d2, d3], axis=1))

    # Fairness (needs aggregate --with-fairness)
    sub = df[df["variation"] == "stationary"]
    if "jain_assignments" in sub.columns and sub["jain_assignments"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
        d1 = _bars(axes[0], sub, "jain_assignments", "Jain index (assignments)")
        axes[0].set_ylim(0, 1.0)
        if "jain_utilities" in sub.columns and sub["jain_utilities"].notna().any():
            d2 = _bars(axes[1], sub, "jain_utilities", "Jain index (utilities)")
            axes[1].set_ylim(0, 1.0)
        fig.tight_layout()
        _save(fig, "fig_overall_fairness", d1)


def fig_quality_matched(dataset="synthetic"):
    df = _by_seed("overall_quality_matched", dataset)
    sub = df[df["variation"] == "stationary"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    d1 = _bars(axes[0], sub, "HQR", "High-quality ratio (matched)")
    axes[0].set_ylim(0.8, 1.0)
    d2 = _bars(axes[1], sub, "cumulative_payment", "Cumulative payment")
    d3 = _bars(axes[2], sub, "payment_per_HQ", "Payment per HQ task")
    fig.tight_layout()
    _save(fig, "fig_quality_matched", pd.concat([d1, d2, d3], axis=1))


def fig_long_term(dataset="synthetic"):
    cfg_path = ROOT / "configs" / "experiments" / "long_term.yaml"
    import yaml
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    src_exp = cfg.get("analysis_of", "overall")
    var = cfg.get("analysis_variation", "stationary")
    methods = cfg.get("analysis_methods", ["PRIME", "MOI", "LDI"])
    window = int(cfg.get("rolling_window", 25))

    slot = _load_slot_logs(src_exp, var, methods, dataset,
                           columns=["slot", "num_assigned", "num_high_quality",
                                    "total_payment", "maintenance_ratio", "mean_quality"])
    prov = _load_provider_logs(src_exp, var, methods, dataset,
                               columns=["slot", "H_after", "p_last", "assigned"])

    # H over time
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = []
    for m, df in prov.items():
        g = df.groupby(["seed", "slot"])["H_after"].mean().reset_index()
        rows.append(_ts_plot(ax, g.rename(columns={"H_after": "y"}), m, window))
    ax.set_xlabel("Slot"); ax.set_ylabel("Mean path state H")
    ax.set_ylim(0, 1); ax.legend()
    _save(fig, "fig_dynamics_H", pd.concat(rows) if rows else None)

    # Bonus probability among assigned providers
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = []
    for m, df in prov.items():
        a = df[df["assigned"] & (df["p_last"] > 0)]
        g = a.groupby(["seed", "slot"])["p_last"].mean().reset_index()
        if len(g):
            rows.append(_ts_plot(ax, g.rename(columns={"p_last": "y"}), m, window))
    ax.set_xlabel("Slot"); ax.set_ylabel("Mean bonus probability p")
    ax.legend()
    _save(fig, "fig_dynamics_p", pd.concat(rows) if rows else None)

    # Maintenance ratio
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = []
    for m, df in slot.items():
        g = df.rename(columns={"maintenance_ratio": "y"})[["seed", "slot", "y"]]
        rows.append(_ts_plot(ax, g, m, window))
    ax.set_xlabel("Slot"); ax.set_ylabel("Maintenance-stage ratio")
    ax.set_ylim(0, 1); ax.legend()
    _save(fig, "fig_dynamics_stage", pd.concat(rows) if rows else None)

    # HQR over time
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = []
    for m, df in slot.items():
        df = df.copy()
        df["y"] = df["num_high_quality"] / df["num_assigned"].clip(lower=1)
        rows.append(_ts_plot(ax, df[["seed", "slot", "y"]], m, window))
    ax.set_xlabel("Slot"); ax.set_ylabel("High-quality ratio")
    ax.set_ylim(0, 1.05); ax.legend()
    _save(fig, "fig_dynamics_hqr", pd.concat(rows) if rows else None)

    # Per-slot payment
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = []
    for m, df in slot.items():
        g = df.rename(columns={"total_payment": "y"})[["seed", "slot", "y"]]
        rows.append(_ts_plot(ax, g, m, window))
    ax.set_xlabel("Slot"); ax.set_ylabel("Payment per slot")
    ax.legend()
    _save(fig, "fig_dynamics_payment", pd.concat(rows) if rows else None)


def _break_even_slot(delta: np.ndarray, window: int = 50) -> float:
    """First slot where DeltaC <= 0 and stays <= 0 for `window` slots."""
    T = len(delta)
    neg = delta <= 0
    for t in range(T):
        end = min(t + window, T)
        if neg[t:end].all() and (end - t) >= min(window, T - t):
            return float(t)
    return float("nan")


def fig_break_even(dataset="synthetic"):
    exp_dir = ROOT / "results" / "break_even" / dataset
    variations = sorted([d.name for d in exp_dir.iterdir() if d.is_dir()]) \
        if exp_dir.exists() else []
    groups = {
        "lifetime": ["base", "lifetime_300", "lifetime_150"],
        "rhoB": ["base", "rhoB_0.4", "rhoB_1.0"],
        "xi": ["base", "xi_low", "xi_high"],
        "delta": ["base", "delta_low", "delta_high"],
    }
    be_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    plot_data = []
    for ax, (gname, vars_) in zip(axes.flat, groups.items()):
        for i, var in enumerate(v for v in vars_ if v in variations):
            logs = _load_slot_logs("break_even", var, ["PRIME", "MOI"], dataset,
                                   columns=["slot", "total_payment"])
            if "PRIME" not in logs or "MOI" not in logs:
                continue
            cp = logs["PRIME"].pivot_table(index="slot", columns="seed",
                                           values="total_payment").cumsum()
            cm = logs["MOI"].pivot_table(index="slot", columns="seed",
                                         values="total_payment").cumsum()
            common = [c for c in cp.columns if c in cm.columns]
            delta = cp[common] - cm[common]
            mean = delta.mean(axis=1)
            ci = 1.96 * delta.std(axis=1, ddof=1) / np.sqrt(len(common))
            ls = ["-", "--", ":"][i % 3]
            ax.plot(mean.index, mean, ls=ls, lw=1.1, color="#1f77b4", label=var)
            ax.fill_between(mean.index, mean - ci, mean + ci,
                            color="#1f77b4", alpha=0.10)
            for s in common:
                be_rows.append({"group": gname, "variation": var, "seed": s,
                                "break_even_slot": _break_even_slot(delta[s].to_numpy())})
            plot_data.append(pd.DataFrame({
                "group": gname, "variation": var, "slot": mean.index,
                "delta_mean": mean.to_numpy(), "ci95": ci.to_numpy()}))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Slot")
        ax.set_ylabel(r"$\Delta C(t)$ = cumPay(PRIME) - cumPay(MOI)")
        ax.set_title(f"varying {gname}")
        ax.legend()
    fig.tight_layout()
    _save(fig, "fig_break_even", pd.concat(plot_data) if plot_data else None)
    if be_rows:
        be = pd.DataFrame(be_rows)
        be.to_csv(DATA_DIR / "break_even_slots.csv", index=False)
        print("  break_even_slots.csv:",
              be.groupby("variation")["break_even_slot"].median().to_dict())


def _cultivation_duration(experiment, variation, methods, dataset="synthetic"):
    """Mean slots from first appearance to first maintenance entry, per run."""
    rows = []
    for m in methods:
        mdir = ROOT / "results" / experiment / dataset / variation / m.replace("/", "_")
        for f in sorted(mdir.glob("seed_*/provider_log.parquet")):
            df = pd.read_parquet(f, columns=["provider_id", "slot",
                                             "maintenance_entry_flag", "assigned"])
            first_seen = df.groupby("provider_id")["slot"].min()
            entry = df[df["maintenance_entry_flag"]].groupby("provider_id")["slot"].min()
            dur = (entry - first_seen.reindex(entry.index)).dropna()
            # assigned interactions before entry
            rows.append({"method": m, "seed": int(f.parent.name.replace("seed_", "")),
                         "cultivation_slots_mean": float(dur.mean()) if len(dur) else np.nan,
                         "n_entered": int(len(entry))})
    return pd.DataFrame(rows)


def fig_ablation(dataset="synthetic"):
    df = _by_seed("ablation", dataset)
    sub = df[df["variation"] == "stationary"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    d1 = _bars(axes[0], sub, "HQR", "High-quality ratio")
    axes[0].set_ylim(0, 1.0)
    d2 = _bars(axes[1], sub, "cumulative_payment", "Cumulative payment")
    cult = _cultivation_duration("ablation", "stationary",
                                 sorted(sub["method"].unique()), dataset)
    d3 = _bars(axes[2], cult, "cultivation_slots_mean", "Cultivation duration (slots)")
    fig.tight_layout()
    _save(fig, "fig_ablation", pd.concat([d1, d2, d3], axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))
    d1 = _bars(axes[0], sub, "platform_utility", "Platform utility")
    d2 = _bars(axes[1], sub, "recultivation_events", "Re-cultivation events")
    fig.tight_layout()
    _save(fig, "fig_ablation_extra", pd.concat([d1, d2], axis=1))


def _sweep_lines(df, parse, xlabel, metrics, fname, methods=None, xlog=False):
    """Generic sweep figure: parse(variation) -> x or None."""
    sub = df.copy()
    sub["x"] = sub["variation"].map(parse)
    sub = sub[sub["x"].notna()]
    if sub.empty:
        return
    methods = methods or _order(sub["method"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 2.4))
    if len(metrics) == 1:
        axes = [axes]
    data = []
    for ax, (metric, ylabel) in zip(axes, metrics):
        for m in methods:
            g = sub[sub["method"] == m].groupby("x")[metric]
            mu = g.mean(); n = g.count()
            ci = 1.96 * g.std() / np.sqrt(n.clip(lower=1))
            ax.errorbar(mu.index, mu, yerr=ci, color=_mcolor(m),
                        marker=METHOD_MARKERS.get(m, "o"), ms=3.5, lw=1.1,
                        capsize=2, label=m)
            data.append(pd.DataFrame({"method": m, "x": mu.index,
                                      f"{metric}_mean": mu.to_numpy(),
                                      f"{metric}_ci": ci.to_numpy()}))
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if xlog:
            ax.set_xscale("log")
        ax.legend()
    fig.tight_layout()
    _save(fig, fname, pd.concat(data) if data else None)


def fig_heterogeneity(dataset="synthetic"):
    df = _by_seed("heterogeneity", dataset)

    def parse_rho(v):
        m = re.match(r"rhoB_([\d.]+)$", v)
        return float(m.group(1)) if m else None

    _sweep_lines(df, parse_rho, r"Behavioural fraction $\rho_B$",
                 [("HQR", "High-quality ratio"),
                  ("cumulative_payment", "Cumulative payment"),
                  ("platform_utility", "Platform utility")],
                 "fig_hetero_rhoB")

    # Categorical panels for omega / xi / delta / zeta
    cats = {
        "omega": ["omega_x0.5", "omega_x1.5", "omega_x2.0"],
        "xi": ["xi_low", "xi_high"],
        "delta": ["delta_low", "delta_high"],
        "zeta": ["zeta_strong", "zeta_weak"],
    }
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.6))
    data = []
    for j, (pname, vars_) in enumerate(cats.items()):
        for i, metric in enumerate(["HQR", "cumulative_payment"]):
            ax = axes[i][j]
            sub = df[df["variation"].isin(vars_)]
            for m in _order(sub["method"].unique()):
                g = sub[sub["method"] == m].groupby("variation")[metric]
                mu = g.mean().reindex(vars_)
                ci = (1.96 * g.std() / np.sqrt(g.count().clip(lower=1))).reindex(vars_)
                x = np.arange(len(vars_))
                ax.errorbar(x, mu, yerr=ci, color=_mcolor(m),
                            marker=METHOD_MARKERS.get(m, "o"), ms=3.5, lw=1.1,
                            capsize=2, label=m)
                data.append(pd.DataFrame({"param": pname, "metric": metric,
                                          "method": m, "variation": vars_,
                                          "mean": mu.to_numpy(), "ci": ci.to_numpy()}))
            ax.set_xticks(np.arange(len(vars_)))
            ax.set_xticklabels([v.split("_", 1)[1] for v in vars_], rotation=20)
            if i == 0:
                ax.set_title(pname)
            ax.set_ylabel(metric if j == 0 else "")
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    _save(fig, "fig_hetero_params", pd.concat(data) if data else None)


def fig_parameter_error(dataset="synthetic"):
    df = _by_seed("parameter_error", dataset)
    params = ["omega", "zeta", "xi", "delta", "all"]
    fig, axes = plt.subplots(3, len(params), figsize=(2.1 * len(params), 6.2))
    data = []
    for j, p in enumerate(params):
        def parse(v, p=p):
            if v == "err_0.0":
                return 0.0
            m = re.match(rf"{p}_([\d.]+)$", v)
            return float(m.group(1)) if m else None
        sub = df.copy()
        sub["x"] = sub["variation"].map(parse)
        sub = sub[sub["x"].notna()]
        for i, (metric, ylab) in enumerate([
                ("HQR", "HQR"), ("cumulative_payment", "Cum. payment"),
                ("ir_violations", "IR violations")]):
            ax = axes[i][j]
            for m in _order(sub["method"].unique()):
                g = sub[sub["method"] == m].groupby("x")[metric]
                mu = g.mean()
                ci = 1.96 * g.std() / np.sqrt(g.count().clip(lower=1))
                ax.errorbar(mu.index, mu, yerr=ci, color=_mcolor(m),
                            marker=METHOD_MARKERS.get(m, "o"), ms=3, lw=1.0,
                            capsize=2, label=m)
                data.append(pd.DataFrame({"param": p, "metric": metric, "method": m,
                                          "level": mu.index, "mean": mu.to_numpy(),
                                          "ci": ci.to_numpy()}))
            if i == 0:
                ax.set_title(f"perturbed: {p}")
            if i == 2:
                ax.set_xlabel("error level")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    _save(fig, "fig_param_error", pd.concat(data) if data else None)

    # Systematic omega over/under-estimation
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    data = []
    for i, (metric, ylab) in enumerate([
            ("HQR", "HQR"), ("cumulative_payment", "Cum. payment"),
            ("ir_violations", "IR violations")]):
        ax = axes[i]
        for mode, ls in [("over", "-"), ("under", "--")]:
            def parse(v, mode=mode):
                if v == "err_0.0":
                    return 0.0
                m = re.match(rf"omega_{mode}_([\d.]+)$", v)
                return float(m.group(1)) if m else None
            sub = df[df["method"] == "PRIME"].copy()
            sub["x"] = sub["variation"].map(parse)
            sub = sub[sub["x"].notna()]
            g = sub.groupby("x")[metric]
            mu = g.mean()
            ci = 1.96 * g.std() / np.sqrt(g.count().clip(lower=1))
            ax.errorbar(mu.index, mu, yerr=ci, ls=ls, color="#1f77b4",
                        marker="o", ms=3, lw=1.0, capsize=2,
                        label=f"omega {mode}-estimated")
            data.append(pd.DataFrame({"mode": mode, "metric": metric,
                                      "level": mu.index, "mean": mu.to_numpy()}))
        ax.set_xlabel("error level"); ax.set_ylabel(ylab); ax.legend()
    fig.tight_layout()
    _save(fig, "fig_param_error_omega_bias", pd.concat(data) if data else None)


def fig_dynamic_shock(dataset="synthetic"):
    scenarios = ["cost_shock", "workload_burst", "provider_churn",
                 "behavior_decay", "comm_degradation"]
    methods = ["PRIME", "MOI", "LDI"]
    rec_rows = []
    for sc in scenarios:
        logs = _load_slot_logs("dynamic_shock", sc, methods, dataset,
                               columns=["slot", "num_assigned", "num_high_quality",
                                        "total_payment", "mean_quality"])
        if not logs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
        data = []
        for ax, (metric, ylab) in zip(axes, [
                ("hqr", "High-quality ratio"), ("total_payment", "Payment per slot")]):
            for m, df in logs.items():
                df = df.copy()
                df["y"] = (df["num_high_quality"] / df["num_assigned"].clip(lower=1)
                           if metric == "hqr" else df[metric])
                d = _ts_plot(ax, df[["seed", "slot", "y"]], m, window=10)
                d["metric"] = metric; d["scenario"] = sc
                data.append(d)
            ax.axvspan(300, 350, color="red", alpha=0.08)
            ax.set_xlabel("Slot"); ax.set_ylabel(ylab); ax.legend()
        fig.suptitle(f"scenario: {sc}", y=1.02)
        fig.tight_layout()
        _save(fig, f"fig_shock_{sc}", pd.concat(data))

        # Recovery time (95% of pre-shock 100-slot mean, 20-slot persistence)
        for m, df in logs.items():
            piv = df.assign(y=df["num_high_quality"] / df["num_assigned"].clip(lower=1)) \
                .pivot_table(index="slot", columns="seed", values="y")
            for s in piv.columns:
                y = piv[s].rolling(10, min_periods=1).mean()
                base = y.loc[200:299].mean()
                rec = np.nan
                after = y.loc[350:]
                okv = after >= 0.95 * base
                run = 0
                for t, flag in okv.items():
                    run = run + 1 if flag else 0
                    if run >= 20:
                        rec = t - 19 - 350
                        break
                rec_rows.append({"scenario": sc, "method": m, "seed": int(s),
                                 "recovery_slots": rec, "pre_shock_hqr": float(base)})
    if rec_rows:
        rec = pd.DataFrame(rec_rows)
        rec.to_csv(DATA_DIR / "recovery_times.csv", index=False)
        print("  recovery_times.csv (median):")
        print(rec.groupby(["scenario", "method"])["recovery_slots"]
              .median().unstack().round(1).to_string())


def fig_sensitivity(dataset="synthetic"):
    df = _by_seed("sensitivity", dataset)
    base = df[df["variation"] == "base"]
    panels = [
        ("ThetaM", r"$\Theta_M$", r"ThetaM_([\d.]+)", 0.75),
        ("ThetaC", r"$\Theta_C$", r"ThetaC_([\d.]+)", 0.55),
        ("sM", r"$s_M$", r"sM_([\d.]+)", 0.80),
        ("sC", r"$s_C$", r"sC_([\d.]+)", 0.65),
        ("K", "K", r"K_(\d+)", 10),
        ("dpmax", r"$\Delta p_{max}$", r"dpmax_([\d.]+)", 0.05),
        ("etaH", r"$\eta_H$", r"etaH_([\d.]+)", 5.0),
        ("Dbar", r"$\bar D$", r"Dbar_([\d.]+)", 10.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.6))
    data = []
    for ax, (pname, plabel, pat, base_x) in zip(axes.flat, panels):
        sub = df[df["method"] == "PRIME"].copy()
        sub["x"] = sub["variation"].str.extract(pat)[0].astype(float)
        sub = sub[sub["x"].notna()]
        b = base.copy(); b["x"] = base_x
        sub = pd.concat([sub[["x", "HQR", "cumulative_payment"]],
                         b[b["method"] == "PRIME"][["x", "HQR", "cumulative_payment"]]])
        for metric, color, mk in [("HQR", "#1f77b4", "o"),
                                  ("cumulative_payment", "#d62728", "s")]:
            g = sub.groupby("x")[metric]
            mu, ci = g.mean(), 1.96 * g.std() / np.sqrt(g.count().clip(lower=1))
            axx = ax if metric == "HQR" else ax.twinx()
            axx.errorbar(mu.index, mu, yerr=ci, color=color, marker=mk, ms=3,
                         lw=1.0, capsize=2, label=metric)
            if metric == "HQR":
                axx.set_ylabel("HQR", color=color)
            else:
                axx.set_ylabel("payment", color=color)
                axx.grid(False)
            data.append(pd.DataFrame({"param": pname, "metric": metric,
                                      "x": mu.index, "mean": mu.to_numpy(),
                                      "ci": ci.to_numpy()}))
        ax.set_xlabel(plabel)
        ax.axvline(base_x, color="gray", lw=0.6, ls=":")
    fig.tight_layout()
    _save(fig, "fig_sensitivity", pd.concat(data))

    def parse_budget(v):
        m = re.match(r"budget_([\d.]+)$", v)
        return float(m.group(1)) if m else (0.7 if v == "base" else None)

    _sweep_lines(df, parse_budget, r"Budget ratio $\rho_{budget}$",
                 [("HQR", "High-quality ratio"),
                  ("cumulative_payment", "Cumulative payment"),
                  ("budget_utilization", "Budget utilization")],
                 "fig_sensitivity_budget")


def fig_scalability(dataset="synthetic"):
    sizes = ["N20_M20", "N50_M50", "N100_M100", "N200_M150", "N500_M300"]
    rows = []
    for var in sizes:
        logs = _load_slot_logs("scalability", var, ["PRIME"], dataset)
        if "PRIME" not in logs:
            continue
        df = logs["PRIME"]
        gaps = pd.to_numeric(df["optimality_gap"], errors="coerce").dropna()
        cert = (pd.to_numeric(df["certified_gap"], errors="coerce").dropna()
                if "certified_gap" in df.columns else pd.Series(dtype=float))
        rows.append({
            "size": var,
            "n_pairs_mean": float((df["num_providers"] * df["num_tasks"]).mean()),
            "contract_ms_mean": float(df["contract_runtime"].mean() * 1000),
            "matching_ms_mean": float(df["matching_runtime"].mean() * 1000),
            "total_ms_mean": float(df["total_runtime"].mean() * 1000),
            "total_ms_p90": float(df["total_runtime"].quantile(0.9) * 1000),
            "total_ms_max": float(df["total_runtime"].max() * 1000),
            "gap_mean": float(gaps.mean()) if len(gaps) else np.nan,
            "gap_p90": float(gaps.quantile(0.9)) if len(gaps) else np.nan,
            "gap_max": float(gaps.max()) if len(gaps) else np.nan,
            "certified_gap_mean": float(cert.mean()) if len(cert) else np.nan,
            "certified_gap_max": float(cert.max()) if len(cert) else np.nan,
        })
    if not rows:
        return
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(d))
    axes[0].bar(x, d["contract_ms_mean"], label="contract", color="#1f77b4")
    axes[0].bar(x, d["matching_ms_mean"], bottom=d["contract_ms_mean"],
                label="matching", color="#ff7f0e")
    other = (d["total_ms_mean"] - d["contract_ms_mean"] - d["matching_ms_mean"]).clip(lower=0)
    axes[0].bar(x, other, bottom=d["contract_ms_mean"] + d["matching_ms_mean"],
                label="state/log", color="#2ca02c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(d["size"], rotation=20)
    axes[0].set_ylabel("Mean time per slot (ms)")
    axes[0].legend()
    axes[1].plot(x, d["gap_mean"] * 100, marker="o", label="mean gap", color="#1f77b4")
    axes[1].plot(x, d["gap_p90"] * 100, marker="s", label="P90 gap", color="#ff7f0e")
    if d["certified_gap_mean"].notna().any():
        axes[1].plot(x, d["certified_gap_mean"] * 100, marker="^",
                     label="certified gap (MILP)", color="#2ca02c")
    axes[1].set_xticks(x); axes[1].set_xticklabels(d["size"], rotation=20)
    axes[1].set_ylabel("Optimality gap (%)")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "fig_scalability", d)


ALL_FIGS = {
    "overall": fig_overall,
    "overall_quality_matched": fig_quality_matched,
    "long_term": fig_long_term,
    "break_even": fig_break_even,
    "ablation": fig_ablation,
    "heterogeneity": fig_heterogeneity,
    "parameter_error": fig_parameter_error,
    "dynamic_shock": fig_dynamic_shock,
    "sensitivity": fig_sensitivity,
    "scalability": fig_scalability,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    targets = list(ALL_FIGS) if args.all else [args.experiment]
    if not targets or targets == [None]:
        sys.exit("Specify --experiment <name> or --all")
    for t in targets:
        fn = ALL_FIGS.get(t)
        if fn is None:
            print(f"Unknown experiment {t}; choices: {list(ALL_FIGS)}")
            continue
        print(f"[{t}]")
        try:
            fn(args.dataset)
        except SystemExit as e:
            print(f"  skipped: {e}")


if __name__ == "__main__":
    main()
