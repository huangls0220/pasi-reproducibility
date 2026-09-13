import pandas as pd

from scripts.run_e24_robust_optimality_certificate import (
    certificate_slots,
    independent_lexicographic_milp,
)


def test_independent_lexicographic_milp_prefers_min_cost_at_max_coverage():
    pairs = pd.DataFrame({
        "provider_id": ["p0", "p0", "p1", "p1"],
        "task_id": ["t0", "t1", "t0", "t1"],
        "feasible_contract": [True, True, True, True],
        "expected_contract_cost": [2.0, 1.0, 1.0, 4.0],
    })
    cardinality, payment = independent_lexicographic_milp(pairs, budget=5.0)
    assert cardinality == 2
    assert payment == 2.0


def test_certificate_slots_are_fixed_positions():
    assert certificate_slots(list(range(21))) == {0, 5, 10, 15, 20}
