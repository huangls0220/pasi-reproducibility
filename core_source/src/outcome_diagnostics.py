"""Observed service outcomes, separate from execution and model guarantees.

No outcome in this module is a population risk certificate. A point forecast
exceeded by settlement is distinct from exceeding the available slot budget.
"""
from __future__ import annotations

import pandas as pd

DIAGNOSTICS_VERSION='r53-service-outcomes-v1'


def service_outcomes(slot: pd.DataFrame, summary: dict, enforce_target=True) -> dict:
    def total(column, fallback=None):
        if column not in slot:
            return fallback
        values=pd.to_numeric(slot[column],errors='coerce')
        return None if values.isna().any() else int(values.sum())

    def excess(left,right):
        if left not in slot or right not in slot:return None
        a=pd.to_numeric(slot[left],errors='coerce')
        b=pd.to_numeric(slot[right],errors='coerce')
        if a.isna().any() or b.isna().any():return None
        return int((a>b+1e-7).sum())

    assigned=total('num_assigned',summary.get('num_assigned'))
    qualified=total('num_qualified_completed')
    qos=None if assigned is None or qualified is None else assigned-qualified
    if qos is not None and (qos<0 or qualified<0):
        raise ValueError('invalid qualified/assigned counts; do not clamp negative violations')
    ir=total('ir_violations',summary.get('ir_violations'))
    target=total('target_violations',summary.get('target_violations'))
    available_over=excess('total_payment','budget')
    reservation_over=excess('budget_used','budget')
    forecast_over=excess('total_payment','budget_used')
    required=[qos,ir,available_over,reservation_over]
    if enforce_target:required.append(target)
    if any(v is not None and v>0 for v in required):status='violated'
    elif any(v is None for v in required):status='unassessed'
    elif assigned==0:status='no_assigned_contracts'
    else:status='passed_observed'
    rate=lambda v:None if v is None or not assigned else v/assigned
    return {
        'diagnostics_version':DIAGNOSTICS_VERSION,
        'service_outcome_status':status,
        'assigned_contracts':assigned,'qualified_contracts':qualified,
        'qos_violations':qos,'qos_violation_rate':rate(qos),
        'ir_violations':ir,'ir_violation_rate':rate(ir),
        'target_violations':target,'target_violation_rate':rate(target),
        'target_required_for_outcome':bool(enforce_target),
        'settlement_over_available_budget_slots':available_over,
        'reservation_over_available_budget_slots':reservation_over,
        'settlement_over_reservation_slots':forecast_over,
        'reserve_bound_observation':('unassessed' if forecast_over is None else
                                    'exceeded' if forecast_over else 'not_exceeded_observed'),
        'guarantee_scope':summary.get('guarantee_scope','unspecified_not_certified'),
        'population_risk_certified':False,
    }
