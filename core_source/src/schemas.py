"""Core data structures for PRIME experiment framework.

All structures use Pydantic for validation. Paper notation is
referenced in docstrings.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Task (Section 4) ───────────────────────────────────────────

class Task(BaseModel):
    """A single task / job arriving at slot t.

    Paper notation:
      L_j^t   = cpu_cycles
      V_j^t   = value
      q_min   = min_quality
    """

    task_id: str
    slot: int = Field(ge=0)
    cpu_cycles: float = Field(gt=0, description="L_j^t")
    input_size: float = Field(ge=0)
    output_size: float = Field(ge=0)
    deadline: float = Field(gt=0)
    min_quality: float = Field(ge=0, le=1, description="q_min ∈ [0,1]")
    value: float = Field(gt=0, description="V_j^t")
    raw_duration: Optional[float] = None
    raw_cpu_request: Optional[float] = None

    @field_validator("min_quality")
    @classmethod
    def check_q_min(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError(f"min_quality must be in [0,1], got {v}")
        return v


# ── Provider ────────────────────────────────────────────────────

class ProviderStatic(BaseModel):
    """Time-invariant provider characteristics.

    Paper notation:
      F_i,max = max_processing_rate
      alpha, beta = cost parameters
      zeta       = Prelec weighting parameter
      omega      = intrinsic motivation weight
      xi         = learning rate
      delta      = decay rate
      U_out      = outside_option
    """

    provider_id: str
    max_processing_rate: float = Field(gt=0, description="F_i,max")
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    zeta: float = Field(gt=0, le=2.0, description="Prelec ζ")
    omega: float = Field(ge=0)
    xi: float = Field(gt=0, le=1.0)
    delta: float = Field(gt=0, le=1.0)
    outside_option: float = Field(ge=0, description="U_out")
    max_bonus_preference_group: Optional[str] = None


class ProviderDynamic(BaseModel):
    """Per-slot dynamic provider state.

    Paper notation:
      ell_i^t = load_ratio
      R_i^t   = communication_rate
      w_i^t   = availability_duration
      H_i^t   = path_state
    """

    provider_id: str
    slot: int = Field(ge=0)
    load_ratio: float = Field(ge=0, le=1, description="ell_i^t")
    communication_rate: float = Field(gt=0, description="R_i^t")
    availability_duration: float = Field(gt=0, description="w_i^t")
    path_state: float = Field(ge=0, le=1, description="H_i^t")
    stage: str = Field(default="cultivation", pattern="^(cultivation|maintenance)$")
    recent_quality: list[float] = Field(default_factory=list)
    last_probability: float = Field(default=0.0, ge=0, le=1)
    reputation: float = Field(default=0.5, ge=0, le=1)


# ── Pair-level results ──────────────────────────────────────────

class PairResult(BaseModel):
    """Detailed result for a single provider–task pair."""

    provider_id: str
    task_id: str
    slot: int

    # Feasibility
    feasible_physical: bool
    feasible_contract: bool
    infeasible_reason: Optional[str] = None

    # Effort breakdown (all in [0,1] or NaN)
    a_deadline: float
    a_quality: float
    a_min: float
    a_system: float
    a_reinforcement: float
    a_target: float

    # Contract
    lambda_required: float
    p_star: float
    D_star: float
    gamma_effective: float

    # Realised effort
    a_star: float
    implementation_gap: float

    # Payments
    base_payment: float
    expected_bonus: float
    expected_contract_cost: float

    # Utilities
    experienced_utility: float
    perceived_utility: float

    # Quality
    execution_quality: float
    normalized_quality: float
    total_delay: float

    # Reinforcement
    delta_H_predicted: float
    immediate_value: float
    reinforcement_value: float
    total_pair_value: float

    # Selected in matching?
    selected: bool = False


# ── Matching result ─────────────────────────────────────────────

class MatchingResult(BaseModel):
    """Output of one slot's matching procedure (P2)."""

    selected_pairs: list[str] = Field(default_factory=list)  # (pid, tid) keys
    total_value: float = 0.0
    total_cost: float = 0.0
    budget: float
    budget_gap: float = 0.0
    lambda_B: float
    lagrangian_upper_bound: Optional[float] = None
    feasible_lower_bound: Optional[float] = None
    optimality_gap: Optional[float] = None
    iterations: int = 0
    runtime: float = 0.0
