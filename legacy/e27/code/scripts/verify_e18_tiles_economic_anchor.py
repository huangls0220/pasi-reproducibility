"""Fail-closed checks for the E18 TILES compensation anchor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e18_tiles_economic_anchor"


def main() -> None:
    weekly = pd.read_csv(RESULTS / "e18_tiles_weekly_cutoffs.csv")
    summary = pd.read_csv(RESULTS / "e18_tiles_anchor_summary.csv")
    metadata = json.loads((RESULTS / "e18_tiles_metadata.json").read_text())
    assert np.allclose(weekly["usd_per_point"], 0.1, atol=1e-12)
    assert len(summary) == 9
    assert set(summary["workload"]) == {"low", "medium", "high"}
    assert set(summary["method"]) == {"PASI", "MOI", "QUAC-F"}
    assert (summary["n_runs"] == 30).all()
    assert "participant bids" in metadata["not_claimed"]
    assert metadata["mapping_type"].startswith("post-hoc")
    print("E18 verification passed: real TILES schedule retained and bid claim excluded.")


if __name__ == "__main__":
    main()
