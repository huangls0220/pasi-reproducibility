"""Provenance and cost semantics for result rows; no decision logic."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def protocol_fields(cfg, summary, diagnostics, runner_path):
    def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    core={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((ROOT/'src').rglob('*.py'))}
    return {
        'contract_protocol_version':summary.get('contract_protocol_version','unspecified'),
        'evaluation_protocol_version':'r53-service-outcomes-v1',
        'decision_cost_basis':summary.get('decision_cost_basis','unspecified'),
        'settlement_cost_basis':'objective_expected_payment_at_executed_response',
        'robust_ir_policy':summary.get('robust_ir_policy','unspecified'),
        'guarantee_scope':summary.get('guarantee_scope','unspecified_not_certified'),
        'service_outcome_status':diagnostics.get('service_outcome_status','unassessed'),
        'execution_status':diagnostics.get('execution_status','unknown'),
        'settlement_over_reservation_slots':diagnostics.get('settlement_over_reservation_slots'),
        'settlement_over_available_budget_slots':diagnostics.get('settlement_over_available_budget_slots'),
        'reservation_over_available_budget_slots':diagnostics.get('reservation_over_available_budget_slots'),
        'population_risk_certified':False,
        'config_sha256':hashlib.sha256(canonical(cfg).encode()).hexdigest(),
        'core_sha256':hashlib.sha256(canonical(core).encode()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(runner_path).read_bytes()).hexdigest(),
        'contract_D_bar':float(cfg['contract']['D_bar']),
        'matching_budget_ratio':float(cfg['matching']['budget_ratio']),
    }
