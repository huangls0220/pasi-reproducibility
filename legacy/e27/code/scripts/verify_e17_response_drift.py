"""Verify E17 response-drift evidence and declared failure boundary."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e17_response_drift"


def main() -> None:
    seeds = pd.read_csv(RESULTS / "e17_formal_seed_results.csv")
    summary = pd.read_csv(RESULTS / "e17_formal_paired_summary.csv")
    assert len(seeds) == 8 * 2 * 30
    assert not seeds["test_side_response_used_for_decision"].any()
    assert seeds.groupby(["scenario", "profile"])["seed"].nunique().eq(30).all()
    in_bound = summary[
        summary["within_declared_envelope"]
        & (summary["profile"] == "fixed_envelope")
    ]
    assert len(in_bound) == 6
    assert in_bound["safety_gate"].all()
    outside = summary[
        (~summary["within_declared_envelope"])
        & (summary["profile"] == "fixed_envelope")
    ]
    assert set(outside["scenario"]) == {"outside_low", "outside_high"}
    print("E17 verification passed: in-bound drift is safe without test-side response observation; outside-bound probes retained as boundary audits.")


if __name__ == "__main__":
    main()
