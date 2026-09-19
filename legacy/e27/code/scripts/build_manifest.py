#!/usr/bin/env python3
"""Build final deliverables (Section 21):

  EXPERIMENT_MANIFEST.csv — every run: experiment, variation, method,
                            dataset, seed, status, path
  RESULT_INDEX.md         — index of summaries, statistics and figures
  PAPER_RESULTS_TEMPLATE.md — tables filled ONLY with actually-produced
                            numbers (means ± 95% CI); no conclusions.

Usage:  python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def build_manifest() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.rglob("summary.json")):
        rel = f.relative_to(RESULTS).parts
        if len(rel) != 5:  # experiment/dataset/variation/method/seed_x/summary.json
            continue
        exp, dataset, variation, method, seed_dir = rel[0], rel[1], rel[2], rel[3], rel[4]
        try:
            with open(f, encoding="utf-8") as fh:
                s = json.load(fh)
            diag = f.parent / "diagnostics.json"
            status = "completed"
            if diag.exists():
                with open(diag, encoding="utf-8") as fh:
                    d = json.load(fh)
                status = "completed" if d.get("status") == "ok" else "diagnostics_failed"
        except (json.JSONDecodeError, OSError):
            status = "corrupt"
            s = {}
        rows.append({
            "experiment": exp, "dataset": dataset, "variation": variation,
            "method": method, "seed": s.get("seed", seed_dir),
            "status": status, "T": s.get("T"),
            "config": str((f.parent / "config_snapshot.yaml").relative_to(ROOT)),
            "path": str(f.parent.relative_to(ROOT)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "EXPERIMENT_MANIFEST.csv", index=False)
    return df


def build_index(manifest: pd.DataFrame) -> None:
    lines = ["# Result Index", "",
             f"Runs: {len(manifest)} "
             f"({(manifest['status'] == 'completed').sum()} completed)", ""]
    lines.append("## Per-experiment run counts\n")
    cnt = manifest.groupby(["experiment", "status"]).size().unstack(fill_value=0)
    lines.append(cnt.to_markdown())
    lines.append("\n## Summaries (results/summaries/)\n")
    for f in sorted((RESULTS / "summaries").glob("*.csv")):
        lines.append(f"- `{f.relative_to(ROOT)}`")
    lines.append("\n## Statistical tests (results/statistics/)\n")
    for f in sorted((RESULTS / "statistics").glob("*.csv")):
        lines.append(f"- `{f.relative_to(ROOT)}`")
    lines.append("\n## Figures (results/figures/)\n")
    for f in sorted((RESULTS / "figures").glob("*.pdf")):
        lines.append(f"- `{f.relative_to(ROOT)}` (data: figures/data/{f.stem}.csv)")
    lines.append("\n## Tuning\n")
    for f in [RESULTS / "tuning" / "tuning_results.csv",
              ROOT / "configs" / "tuned" / "same_budget.yaml",
              ROOT / "configs" / "tuned" / "quality_matched.yaml"]:
        if f.exists():
            lines.append(f"- `{f.relative_to(ROOT)}`")
    (ROOT / "RESULT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(mu, ci):
    if pd.isna(mu):
        return "—"
    return f"{mu:.4g} ± {ci:.2g}" if not pd.isna(ci) else f"{mu:.4g}"


def build_paper_template() -> None:
    lines = [
        "# Paper Results (auto-generated from real runs)",
        "",
        "All numbers are mean ± 95% CI across seeds from the runs listed in",
        "EXPERIMENT_MANIFEST.csv.  Dataset: SYNTHETIC workloads (real traces",
        "not downloaded; adapters ready).  This file records values only —",
        "interpretation belongs to the paper text.",
        "",
    ]
    for exp in ["overall", "overall_quality_matched", "ablation"]:
        p = RESULTS / "summaries" / f"{exp}_synthetic_summary.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        for var in sorted(s["variation"].unique()):
            sub = s[s["variation"] == var]
            lines.append(f"## {exp} / {var}\n")
            cols = [("HQR", "HQR"), ("average_quality", "Avg quality"),
                    ("completion_ratio", "Completion"),
                    ("cumulative_payment", "Cum. payment"),
                    ("payment_per_HQ", "Pay / HQ task"),
                    ("platform_utility", "Platform utility"),
                    ("provider_utility_mean", "Provider utility")]
            hdr = "| Method | n | " + " | ".join(c[1] for c in cols) + " |"
            lines.append(hdr)
            lines.append("|" + "---|" * (len(cols) + 2))
            for _, r in sub.iterrows():
                cells = [_fmt(r.get(f"{c}_mean"), r.get(f"{c}_ci_hi", float("nan"))
                              - r.get(f"{c}_mean", 0) if f"{c}_ci_hi" in r else float("nan"))
                         for c, _ in cols]
                lines.append(f"| {r['method']} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
            lines.append("")
    # Statistics digest
    lines.append("## Statistical tests (Holm-Bonferroni adjusted)\n")
    for f in sorted((RESULTS / "statistics").glob("*_tests.csv")):
        t = pd.read_csv(f)
        key = t[t["metric"].isin(["HQR", "cumulative_payment"])]
        if len(key) == 0:
            continue
        lines.append(f"### {f.stem}\n")
        lines.append("| Variation | Metric | vs | mean diff [95% CI] | p_adj | effect | test |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in key.iterrows():
            lines.append(
                f"| {r['variation']} | {r['metric']} | {r['method_b']} | "
                f"{r['mean_diff']:+.4g} [{r['ci_lower']:.4g}, {r['ci_upper']:.4g}] | "
                f"{r['adjusted_p']:.2e} | {r['effect_size']:.2f} | {r['test_name']} |")
        lines.append("")
    (ROOT / "PAPER_RESULTS_TEMPLATE.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")


def main() -> None:
    m = build_manifest()
    print(f"EXPERIMENT_MANIFEST.csv: {len(m)} runs")
    build_index(m)
    print("RESULT_INDEX.md written")
    build_paper_template()
    print("PAPER_RESULTS_TEMPLATE.md written")


if __name__ == "__main__":
    main()
