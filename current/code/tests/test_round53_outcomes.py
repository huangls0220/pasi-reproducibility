"""Evaluation repair: missing/negative outcomes cannot become safety passes."""
import numpy as np
import pandas as pd
import pytest

from src.outcome_diagnostics import service_outcomes
from src.simulator import _run_diagnostics


def slot(**changes):
    row=dict(num_tasks=10,num_providers=3,num_assigned=5,num_qualified_completed=5,
             ir_violations=0,target_violations=0,total_payment=2.,budget_used=2.,
             budget=10.,budget_gap=8.,num_feasible_pairs=8)
    row.update(changes)
    return pd.DataFrame([row])


def test_qos_failure_is_not_hidden_by_target_success():
    data=slot(num_qualified_completed=4)
    diag=_run_diagnostics(pd.DataFrame(),data,pd.DataFrame(),
                          {'num_assigned':5,'ir_violations':0,'target_violations':0})
    assert diag['status']=='failed'
    assert diag['execution_status']=='completed'
    assert diag['qos_violations']==1 and diag['target_violations']==0
    assert diag['service_outcome_status']=='violated'


def test_target_failure_is_not_hidden_by_qos_success():
    out=service_outcomes(slot(target_violations=3),{})
    assert out['qos_violations']==0 and out['target_violations']==3
    assert out['service_outcome_status']=='violated'


def test_point_forecast_excess_is_not_budget_overspend():
    out=service_outcomes(slot(total_payment=3.),{})
    assert out['settlement_over_reservation_slots']==1
    assert out['settlement_over_available_budget_slots']==0
    assert out['service_outcome_status']=='passed_observed'
    assert out['reserve_bound_observation']=='exceeded'
    assert not out['population_risk_certified']


def test_actual_budget_overspend_is_failed():
    assert service_outcomes(slot(total_payment=11.),{})['service_outcome_status']=='violated'


def test_missing_quality_observation_is_not_zero_violations():
    out=service_outcomes(slot().drop(columns='num_qualified_completed'),{})
    assert out['qos_violations'] is None
    assert out['service_outcome_status']=='unassessed'


def test_zero_assigned_is_not_risk_certification():
    out=service_outcomes(slot(num_assigned=0,num_qualified_completed=0),{})
    assert out['qos_violation_rate'] is None
    assert out['service_outcome_status']=='no_assigned_contracts'


def test_invalid_counts_raise_not_clamp():
    with pytest.raises(ValueError,match='invalid qualified'):
        service_outcomes(slot(num_qualified_completed=6),{})


def test_target_not_required_still_logged():
    out=service_outcomes(slot(target_violations=4),{},enforce_target=False)
    assert out['target_violations']==4
    assert out['service_outcome_status']=='passed_observed'


def test_no_bootstrap_or_certification_with_one_episode(monkeypatch):
    import scripts.run_e20_envelope_tradeoff as runner
    def forbidden(*args,**kwargs):raise AssertionError('bootstrap for n=1')
    monkeypatch.setattr(runner,'bootstrap_ci',forbidden)
    rows=[]
    for profile in ['frozen_point','fixed_envelope']:
        row=dict(half_width=.3,scenario='stable',profile=profile,seed=1,
            within_declared_envelope=True,test_side_response_used_for_decision=False,
            run_status='ok',ratio_min_audit=1.,ratio_max_audit=1.,pps=1.,
            service_coverage=.5,qos_violation_rate=0.,under_incentive_rate=0.,
            target_miss_rate=0.,legal_edge_rate=.5,platform_utility=1.)
        rows.append(row)
    result=runner.summarize(pd.DataFrame(rows))
    assert result['qos_violation_rate_ci_high'].isna().all()
    assert not result['safety_gate'].any()
    assert not result['population_risk_certified'].any()
