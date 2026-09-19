import pandas as pd

from scripts.run_e29_calibration_selection import (
    ALPHA,
    CALIBRATION_SEEDS,
    TEST_SEEDS,
    clopper_pearson_upper,
    select_guardrail,
)


def test_registered_seed_splits_are_complete_and_disjoint():
    assert set(CALIBRATION_SEEDS).isdisjoint(TEST_SEEDS)
    assert sorted(CALIBRATION_SEEDS + TEST_SEEDS) == list(range(1, 31))


def test_exact_upper_endpoint_special_cases():
    assert 0.0 < clopper_pearson_upper(0, 100, 0.05) < 0.05
    assert clopper_pearson_upper(100, 100, 0.05) == 1.0


def test_selector_uses_smallest_monotone_eligible_candidate():
    rows = []
    worst = {0.0: ALPHA + 0.001, 0.25: ALPHA - 0.001,
             0.5: ALPHA - 0.002, 0.75: ALPHA - 0.003,
             1.0: ALPHA - 0.004}
    for weight, upper in worst.items():
        rows.append({"global_guardrail_weight": weight,
                     "simultaneous_cp_upper": upper})
    selection = select_guardrail(pd.DataFrame(rows),
                                 {weight: upper for weight, upper in worst.items()})
    assert selection["statistical_pass"] is True
    assert selection["selected_global_guardrail_weight"] == 0.25


def test_selector_falls_back_structurally_when_none_passes():
    rows = [{"global_guardrail_weight": weight,
             "simultaneous_cp_upper": ALPHA + 0.001}
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]
    sensitivity = {weight: ALPHA + 0.001
                   for weight in (0.0, 0.25, 0.5, 0.75, 1.0)}
    selection = select_guardrail(pd.DataFrame(rows), sensitivity)
    assert selection["statistical_pass"] is False
    assert selection["selected_global_guardrail_weight"] == 1.0
    assert "fallback" in selection["selection_basis"]
