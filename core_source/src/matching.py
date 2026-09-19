"""Budget-constrained matching P2 (Section 6).

Array-based API: pairs are given as parallel arrays
(provider_row, task_row, matching_value, expected_cost).

Implements:
  - Lagrangian relaxation + Hungarian assignment (Section 6.1)
  - Lagrangian upper bound tracking (Section 6.2)
  - Budget repair (drop worst value/cost) + greedy top-up improvement
  - Small-instance MILP exact optimum via scipy.optimize.milp (Section 6.3)
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_EPS = 1e-12


def coverage_first_matching(
    P: np.ndarray,
    T: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    n_providers: int,
    n_tasks: int,
    budget: float,
    true_values: Optional[np.ndarray] = None,
) -> dict:
    """Lexicographically maximize legal-service count, then minimize payment.

    The E8 workload has four tasks per slot.  Enumerating task subsets and
    solving a rectangular linear assignment for each subset is therefore both
    exact and much cheaper than invoking a general MILP in every slot.  The
    first cardinality whose minimum-cost assignment fits the slot budget is
    the budget-feasible maximum; the chosen assignment is minimum-payment at
    that cardinality.

    For larger task sets the number of subsets grows exponentially.  Those
    instances use an exact two-stage binary MILP with the same lexicographic
    objective: stage one maximizes assignment cardinality and stage two fixes
    that cardinality while minimizing payment.  The small-instance subset
    solver is retained as an independent oracle for regression tests.
    """
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=int)
    T = np.asarray(T, dtype=int)
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if true_values is None:
        true_values = values
    else:
        true_values = np.asarray(true_values, dtype=float)
    empty = {
        "selected_idx": np.array([], dtype=int), "total_value": 0.0,
        "total_cost": 0.0, "budget": float(budget), "budget_gap": float(budget),
        "matching_objective": 0.0, "objective_type": "coverage_first_payment_second",
        "maximum_cardinality": 0, "lagrangian_upper_bound": None,
        "feasible_lower_bound": 0.0, "optimality_gap": 0.0,
        "iterations": 0, "runtime": time.perf_counter() - t0,
        "lambda_B": 0.0,
    }
    if len(P) == 0 or n_tasks == 0 or n_providers == 0 or budget <= 0:
        return empty
    if n_tasks > 12:
        # Exact fast path for the common dense case.  A feasible assignment of
        # min(n_providers, n_tasks) pairs has the largest cardinality possible.
        # The rectangular Hungarian solution is the minimum-cost assignment
        # at that cardinality.  If it fits the budget, it is therefore exactly
        # the two-stage coverage-first/payment-second optimum.  Structural
        # infeasibility or a budget failure falls back to the general MILP.
        lookup = np.full((n_providers, n_tasks), -1, dtype=int)
        edge_cost = np.full((n_providers, n_tasks), np.inf, dtype=float)
        for k, (provider, task) in enumerate(zip(P, T)):
            if costs[k] < edge_cost[provider, task]:
                lookup[provider, task] = k
                edge_cost[provider, task] = costs[k]
        try:
            row_ind, col_ind = linear_sum_assignment(edge_cost)
            assigned_costs = edge_cost[row_ind, col_ind]
            full_cardinality = min(n_providers, n_tasks)
            if (
                len(row_ind) == full_cardinality
                and np.all(np.isfinite(assigned_costs))
                and float(assigned_costs.sum()) <= budget + 1e-9
            ):
                selected = np.asarray(
                    [lookup[int(provider), int(task)]
                     for provider, task in zip(row_ind, col_ind)],
                    dtype=int,
                )
                if np.all(selected >= 0):
                    total_cost = float(costs[selected].sum())
                    return {
                        "selected_idx": np.sort(selected),
                        "total_value": float(true_values[selected].sum()),
                        "total_cost": total_cost,
                        "budget": float(budget),
                        "budget_gap": float(budget - total_cost),
                        "matching_objective": float(full_cardinality),
                        "objective_type": "coverage_first_payment_second",
                        "maximum_cardinality": int(full_cardinality),
                        "lagrangian_upper_bound": None,
                        "feasible_lower_bound": float(full_cardinality),
                        "optimality_gap": 0.0,
                        "iterations": 1,
                        "runtime": time.perf_counter() - t0,
                        "lambda_B": 0.0,
                        "solver": "full_cardinality_hungarian_exact",
                    }
        except ValueError:
            pass
        return _coverage_first_kcardinality(
            P, T, values, costs, n_providers, n_tasks, budget,
            true_values=true_values, started_at=t0,
        )

    lookup = np.full((n_providers, n_tasks), -1, dtype=int)
    edge_cost = np.full((n_providers, n_tasks), np.inf, dtype=float)
    for k, (provider, task) in enumerate(zip(P, T)):
        if costs[k] < edge_cost[provider, task]:
            lookup[provider, task] = k
            edge_cost[provider, task] = costs[k]

    max_k = min(n_providers, n_tasks)
    iterations = 0
    for cardinality in range(max_k, 0, -1):
        best_sel: Optional[np.ndarray] = None
        best_cost = np.inf
        for task_subset in combinations(range(n_tasks), cardinality):
            iterations += 1
            sub = edge_cost[:, task_subset]
            if np.any(np.all(~np.isfinite(sub), axis=0)):
                continue
            try:
                row_ind, col_ind = linear_sum_assignment(sub)
            except ValueError:
                # Each task can have at least one edge while Hall's condition
                # still fails for the subset; such a subset has no complete
                # one-to-one assignment and is therefore skipped.
                continue
            if len(col_ind) != cardinality:
                continue
            assigned_costs = sub[row_ind, col_ind]
            if not np.all(np.isfinite(assigned_costs)):
                continue
            total_cost = float(assigned_costs.sum())
            if total_cost > budget + 1e-9 or total_cost >= best_cost - 1e-12:
                continue
            selected = np.asarray([
                lookup[int(provider), int(task_subset[int(local_task)])]
                for provider, local_task in zip(row_ind, col_ind)
            ], dtype=int)
            if np.any(selected < 0):
                continue
            best_sel = np.sort(selected)
            best_cost = total_cost
        if best_sel is not None:
            return {
                "selected_idx": best_sel,
                "total_value": float(true_values[best_sel].sum()),
                "total_cost": best_cost,
                "budget": float(budget), "budget_gap": float(budget - best_cost),
                "matching_objective": float(cardinality),
                "objective_type": "coverage_first_payment_second",
                "maximum_cardinality": int(cardinality),
                "lagrangian_upper_bound": None,
                "feasible_lower_bound": float(cardinality),
                "optimality_gap": 0.0, "iterations": iterations,
                "runtime": time.perf_counter() - t0, "lambda_B": 0.0,
            }
    empty["iterations"] = iterations
    empty["runtime"] = time.perf_counter() - t0
    return empty


def _minimum_cost_k_matching(
    lookup: np.ndarray,
    edge_cost: np.ndarray,
    cardinality: int,
) -> Optional[tuple[np.ndarray, float]]:
    """Return the minimum-cost matching with exactly ``cardinality`` edges.

    The reduction adds ``n_tasks-k`` dummy providers and
    ``n_providers-k`` dummy tasks.  Dummy-to-real and real-to-dummy edges
    have zero cost, while dummy-to-dummy edges are forbidden.  Any perfect
    assignment in the augmented square matrix therefore contains exactly
    ``k`` real provider--task edges.  Hungarian assignment then minimizes
    their total cost.
    """
    n_providers, n_tasks = edge_cost.shape
    k = int(cardinality)
    if k < 0 or k > min(n_providers, n_tasks):
        return None
    if k == 0:
        return np.array([], dtype=int), 0.0

    dummy_providers = n_tasks - k
    dummy_tasks = n_providers - k
    size = n_providers + dummy_providers
    augmented = np.full((size, size), np.inf, dtype=float)
    augmented[:n_providers, :n_tasks] = edge_cost
    if dummy_tasks:
        augmented[:n_providers, n_tasks:] = 0.0
    if dummy_providers:
        augmented[n_providers:, :n_tasks] = 0.0

    try:
        row_ind, col_ind = linear_sum_assignment(augmented)
    except ValueError:
        return None
    real = (row_ind < n_providers) & (col_ind < n_tasks)
    real_rows = row_ind[real]
    real_cols = col_ind[real]
    if len(real_rows) != k:
        return None
    selected = lookup[real_rows, real_cols]
    if np.any(selected < 0) or not np.all(np.isfinite(edge_cost[real_rows, real_cols])):
        return None
    return np.sort(selected.astype(int)), float(edge_cost[real_rows, real_cols].sum())


def _coverage_first_kcardinality(
    P: np.ndarray,
    T: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    n_providers: int,
    n_tasks: int,
    budget: float,
    true_values: Optional[np.ndarray] = None,
    started_at: Optional[float] = None,
) -> dict:
    """Solve coverage-first/payment-second by exact k-cardinality search.

    Let C(k) be the minimum cost of a matching with k edges.  With
    nonnegative contract costs, C(k) is nondecreasing: removing one edge
    from any (k+1)-matching yields a feasible k-matching of no larger cost.
    Binary search therefore finds the largest k whose minimum cost fits the
    budget.  The retained Hungarian solution is the minimum-cost matching at
    that maximum feasible cardinality.
    """
    t0 = time.perf_counter() if started_at is None else started_at
    P = np.asarray(P, dtype=int)
    T = np.asarray(T, dtype=int)
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if true_values is None:
        true_values = values
    else:
        true_values = np.asarray(true_values, dtype=float)
    empty = {
        "selected_idx": np.array([], dtype=int), "total_value": 0.0,
        "total_cost": 0.0, "budget": float(budget), "budget_gap": float(budget),
        "matching_objective": 0.0,
        "objective_type": "coverage_first_payment_second",
        "maximum_cardinality": 0, "lagrangian_upper_bound": None,
        "feasible_lower_bound": 0.0, "optimality_gap": 0.0,
        "iterations": 0, "runtime": time.perf_counter() - t0,
        "lambda_B": 0.0, "solver": "k_cardinality_hungarian_exact",
    }
    if len(P) == 0 or n_tasks == 0 or n_providers == 0 or budget <= 0:
        return empty
    if np.any(costs < -1e-12):
        raise ValueError("coverage-first k-cardinality solver requires nonnegative costs")

    lookup = np.full((n_providers, n_tasks), -1, dtype=int)
    edge_cost = np.full((n_providers, n_tasks), np.inf, dtype=float)
    for edge, (provider, task) in enumerate(zip(P, T)):
        if costs[edge] < edge_cost[provider, task]:
            lookup[provider, task] = edge
            edge_cost[provider, task] = costs[edge]

    low, high = 0, min(n_providers, n_tasks)
    best_selected = np.array([], dtype=int)
    best_cost = 0.0
    iterations = 0
    while low <= high:
        k = (low + high) // 2
        candidate = _minimum_cost_k_matching(lookup, edge_cost, k)
        iterations += 1
        if candidate is None:
            high = k - 1
            continue
        selected, total_cost = candidate
        if total_cost <= budget + 1e-9:
            best_selected = selected
            best_cost = total_cost
            low = k + 1
        else:
            high = k - 1

    cardinality = int(len(best_selected))
    return {
        "selected_idx": best_selected,
        "total_value": float(true_values[best_selected].sum()),
        "total_cost": float(best_cost),
        "budget": float(budget),
        "budget_gap": float(budget - best_cost),
        "matching_objective": float(cardinality),
        "objective_type": "coverage_first_payment_second",
        "maximum_cardinality": cardinality,
        "lagrangian_upper_bound": None,
        "feasible_lower_bound": float(cardinality),
        "optimality_gap": 0.0,
        "iterations": iterations,
        "runtime": time.perf_counter() - t0,
        "lambda_B": 0.0,
        "solver": "k_cardinality_hungarian_exact",
    }


def _coverage_first_milp(
    P: np.ndarray,
    T: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    n_providers: int,
    n_tasks: int,
    budget: float,
    true_values: Optional[np.ndarray] = None,
    started_at: Optional[float] = None,
) -> dict:
    """Solve coverage-first/payment-second matching exactly by two MILPs."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, vstack

    t0 = time.perf_counter() if started_at is None else started_at
    P = np.asarray(P, dtype=int)
    T = np.asarray(T, dtype=int)
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if true_values is None:
        true_values = values
    else:
        true_values = np.asarray(true_values, dtype=float)
    n_edges = len(P)
    empty = {
        "selected_idx": np.array([], dtype=int), "total_value": 0.0,
        "total_cost": 0.0, "budget": float(budget), "budget_gap": float(budget),
        "matching_objective": 0.0,
        "objective_type": "coverage_first_payment_second",
        "maximum_cardinality": 0, "lagrangian_upper_bound": None,
        "feasible_lower_bound": 0.0, "optimality_gap": 0.0,
        "iterations": 0, "runtime": time.perf_counter() - t0,
        "lambda_B": 0.0, "solver": "two_stage_exact_milp",
    }
    if n_edges == 0 or n_tasks == 0 or n_providers == 0 or budget <= 0:
        return empty

    row = []
    col = []
    data = []
    for edge, (provider, task) in enumerate(zip(P, T)):
        row.extend((int(provider), n_providers + int(task)))
        col.extend((edge, edge))
        data.extend((1.0, 1.0))
    assignment = coo_matrix(
        (data, (row, col)), shape=(n_providers + n_tasks, n_edges)
    ).tocsr()
    budget_row = coo_matrix(costs.reshape(1, -1)).tocsr()
    base_matrix = vstack((assignment, budget_row), format="csr")
    base_lb = np.full(n_providers + n_tasks + 1, -np.inf)
    base_ub = np.r_[np.ones(n_providers + n_tasks), float(budget)]
    bounds = Bounds(np.zeros(n_edges), np.ones(n_edges))
    integrality = np.ones(n_edges, dtype=int)
    options = {"disp": False}

    stage_one = milp(
        c=-np.ones(n_edges), integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(base_matrix, base_lb, base_ub),
        options=options,
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError(f"coverage-first MILP stage one failed: {stage_one.message}")
    cardinality = int(round(float(np.sum(stage_one.x > 0.5))))
    if cardinality == 0:
        empty["iterations"] = 1
        empty["runtime"] = time.perf_counter() - t0
        return empty

    cardinality_row = coo_matrix(np.ones((1, n_edges))).tocsr()
    stage_two_matrix = vstack((base_matrix, cardinality_row), format="csr")
    stage_two_lb = np.r_[base_lb, float(cardinality)]
    stage_two_ub = np.r_[base_ub, float(cardinality)]
    stage_two = milp(
        c=costs, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(stage_two_matrix, stage_two_lb, stage_two_ub),
        options=options,
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError(f"coverage-first MILP stage two failed: {stage_two.message}")
    selected = np.flatnonzero(stage_two.x > 0.5).astype(int)
    total_cost = float(costs[selected].sum())
    return {
        "selected_idx": selected,
        "total_value": float(true_values[selected].sum()),
        "total_cost": total_cost, "budget": float(budget),
        "budget_gap": float(budget - total_cost),
        "matching_objective": float(cardinality),
        "objective_type": "coverage_first_payment_second",
        "maximum_cardinality": cardinality, "lagrangian_upper_bound": None,
        "feasible_lower_bound": float(cardinality), "optimality_gap": 0.0,
        "iterations": 2, "runtime": time.perf_counter() - t0,
        "lambda_B": 0.0, "solver": "two_stage_exact_milp",
    }


def _selection_from_assignment(W: np.ndarray, lookup: np.ndarray,
                               n_providers: int, n_tasks: int) -> np.ndarray:
    """Run Hungarian on padded square weight matrix, return pair indices."""
    row_ind, col_ind = linear_sum_assignment(-W)
    sel = []
    for r, c in zip(row_ind, col_ind):
        if r < n_providers and c < n_tasks and W[r, c] > 0:
            k = lookup[r * n_tasks + c]
            if k >= 0:
                sel.append(k)
    return np.asarray(sel, dtype=int)


def _greedy_topup(sel: np.ndarray, P: np.ndarray, T: np.ndarray,
                  values: np.ndarray, costs: np.ndarray,
                  budget: float) -> np.ndarray:
    """Add unused positive-value pairs that fit assignment + budget.

    Simple local improvement after Lagrangian/repair (Section 6.1 'local
    swap'): fills leftover budget in value/cost-ratio order.
    """
    used_p = set(P[sel].tolist())
    used_t = set(T[sel].tolist())
    spent = float(costs[sel].sum())
    order = np.argsort(-(values / np.maximum(costs, _EPS)))
    out = list(sel)
    for k in order:
        if values[k] <= 0:
            break
        p, t = int(P[k]), int(T[k])
        if p in used_p or t in used_t:
            continue
        c = float(costs[k])
        if spent + c > budget + 1e-9:
            continue
        out.append(int(k))
        used_p.add(p); used_t.add(t); spent += c
    return np.asarray(sorted(set(out)), dtype=int)


def _repair(sel: np.ndarray, values: np.ndarray, costs: np.ndarray,
            budget: float) -> np.ndarray:
    """Drop lowest value/cost pairs until the budget is satisfied."""
    sel = list(sel)
    spent = float(costs[sel].sum()) if sel else 0.0
    if spent <= budget + 1e-9:
        return np.asarray(sel, dtype=int)
    ratio_order = sorted(sel, key=lambda k: values[k] / max(costs[k], _EPS))
    keep = set(sel)
    for k in ratio_order:
        if spent <= budget + 1e-9:
            break
        keep.discard(k)
        spent -= float(costs[k])
    return np.asarray(sorted(keep), dtype=int)


def lagrangian_matching(
    P: np.ndarray,
    T: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    n_providers: int,
    n_tasks: int,
    budget: float,
    true_values: Optional[np.ndarray] = None,
    max_iter: int = 100,
    budget_tol: float = 1e-4,
    stagnation_limit: int = 5,
    initial_lambda: float = 0.1,
    step_scale: float = 0.1,
) -> dict:
    """Lagrangian-relaxation matching over pair arrays.

    `values` is the MATCHING objective (may include reputation weighting);
    `true_values` (default: values) is what `total_value` reports.
    Returns dict with `selected_idx` — indices into the given arrays.
    """
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=int); T = np.asarray(T, dtype=int)
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if true_values is None:
        true_values = values
    n_pairs = len(P)

    empty = {
        "selected_idx": np.array([], dtype=int), "total_value": 0.0,
        "total_cost": 0.0, "budget": float(budget), "budget_gap": float(budget),
        "lambda_B": float(initial_lambda), "lagrangian_upper_bound": 0.0,
        "feasible_lower_bound": 0.0, "optimality_gap": 0.0,
        "iterations": 0, "runtime": time.perf_counter() - t0,
    }
    if n_pairs == 0 or n_tasks == 0 or n_providers == 0:
        return empty
    if budget <= 0:
        return empty  # zero budget -> empty matching (tested edge case)

    # (provider,task) -> pair index lookup
    lookup = np.full(n_providers * n_tasks, -1, dtype=int)
    lookup[P * n_tasks + T] = np.arange(n_pairs)

    n = max(n_providers, n_tasks)
    lam = max(0.0, float(initial_lambda))
    best_sel: Optional[np.ndarray] = None
    best_obj = -np.inf
    last_sel_key = None
    stagnation = 0
    UB = np.inf
    step0 = step_scale * budget / max(n_tasks, 1) if budget > 0 else step_scale
    it = 0

    for it in range(1, max_iter + 1):
        adj = values - lam * costs
        W = np.zeros((n, n))
        pos = adj > 0
        W[P[pos], T[pos]] = adj[pos]

        sel = _selection_from_assignment(W, lookup, n_providers, n_tasks)
        tot_cost = float(costs[sel].sum()) if len(sel) else 0.0
        assign_obj = float(adj[sel].sum()) if len(sel) else 0.0

        # Lagrangian upper bound (6.2)
        UB = min(UB, lam * budget + assign_obj)

        if tot_cost <= budget + 1e-9:
            obj = float(values[sel].sum()) if len(sel) else 0.0
            if obj > best_obj:
                best_obj = obj
                best_sel = sel

        # Subgradient update with diminishing step
        step = step0 / np.sqrt(it)
        lam = max(0.0, lam + step * (tot_cost - budget) / max(budget, _EPS))

        key = frozenset(sel.tolist())
        if key == last_sel_key:
            stagnation += 1
            if stagnation >= stagnation_limit:
                break
        else:
            stagnation = 0
        last_sel_key = key

        if abs(tot_cost - budget) / max(budget, _EPS) < budget_tol and tot_cost <= budget + 1e-9:
            break

    if best_sel is None:
        best_sel = _repair(sel, values, costs, budget)

    # Greedy top-up: use leftover budget (improves LB, never violates budget)
    best_sel = _greedy_topup(best_sel, P, T, values, costs, budget)

    LB = float(values[best_sel].sum()) if len(best_sel) else 0.0
    UB = max(UB, LB)  # numerical guard: UB may be loose but never < LB
    gap = (UB - LB) / max(abs(UB), _EPS) if np.isfinite(UB) else None

    tot_cost = float(costs[best_sel].sum()) if len(best_sel) else 0.0
    return {
        "selected_idx": best_sel,
        "total_value": float(true_values[best_sel].sum()) if len(best_sel) else 0.0,
        "matching_objective": LB,
        "total_cost": tot_cost,
        "budget": float(budget),
        "budget_gap": float(budget - tot_cost),
        "lambda_B": float(lam),
        "lagrangian_upper_bound": float(UB) if np.isfinite(UB) else None,
        "feasible_lower_bound": LB,
        "optimality_gap": float(gap) if gap is not None else None,
        "iterations": it,
        "runtime": time.perf_counter() - t0,
    }


def milp_matching(
    P: np.ndarray,
    T: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    n_providers: int,
    n_tasks: int,
    budget: float,
) -> dict:
    """Exact optimum for small instances via scipy.optimize.milp (HiGHS).

    max Σ v_k x_k  s.t.  Σ_{k∈provider i} x_k ≤ 1,  Σ_{k∈task j} x_k ≤ 1,
                          Σ c_k x_k ≤ B,  x binary.
    """
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import lil_matrix

    P = np.asarray(P, dtype=int); T = np.asarray(T, dtype=int)
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n_pairs = len(P)
    if n_pairs == 0 or budget <= 0:
        return {"selected_idx": np.array([], dtype=int), "total_value": 0.0,
                "total_cost": 0.0, "status": "empty"}

    n_rows = n_providers + n_tasks + 1
    A = lil_matrix((n_rows, n_pairs))
    for k in range(n_pairs):
        A[P[k], k] = 1.0
        A[n_providers + T[k], k] = 1.0
        A[n_rows - 1, k] = costs[k]
    ub = np.ones(n_rows)
    ub[-1] = budget
    lc = LinearConstraint(A.tocsr(), -np.inf, ub)

    res = milp(c=-values, constraints=[lc],
               integrality=np.ones(n_pairs),
               bounds=(0, 1))
    if res.status != 0 or res.x is None:
        return {"selected_idx": np.array([], dtype=int), "total_value": 0.0,
                "total_cost": 0.0, "status": f"milp_status_{res.status}"}
    x = np.round(res.x).astype(bool)
    sel = np.flatnonzero(x)
    return {
        "selected_idx": sel,
        "total_value": float(values[sel].sum()),
        "total_cost": float(costs[sel].sum()),
        "status": "optimal",
    }
