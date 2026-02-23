"""Optimization engine with MILP and greedy fallback."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import pulp


def _build_breakdown(alloc_df: pd.DataFrame, open_sites: pd.DataFrame) -> Dict[str, float]:
    transport = float((alloc_df.get("allocated_qty", 0) * alloc_df.get("unit_cost", 0)).sum()) if not alloc_df.empty else 0.0
    fixed = float(open_sites["fixed_cost"].sum()) if not open_sites.empty and "fixed_cost" in open_sites.columns else 0.0
    return {
        "transport": transport,
        "fixed": fixed,
        "tax": float((alloc_df.get("allocated_qty", 0) * alloc_df.get("tax_rate", 0)).sum()) if not alloc_df.empty else 0.0,
        "inventory": 0.0,
        "time_penalty": float((alloc_df.get("allocated_qty", 0) * alloc_df.get("delivery_time_h", 0) * 0.1).sum()) if not alloc_df.empty else 0.0,
    }


def solve_milp(candidates_df: pd.DataFrame, demand_df: pd.DataFrame, cost_matrix_df: pd.DataFrame, constraints: dict, objective_weights: dict) -> Dict[str, Any]:
    customers = demand_df["customer_id"].tolist()
    availability = (
        candidates_df["availability"]
        if "availability" in candidates_df.columns
        else pd.Series(True, index=candidates_df.index)
    )
    sites = candidates_df[availability == True]["site_id"].tolist()  # noqa: E712

    demand = demand_df.set_index("customer_id")["demand"].to_dict()
    if "max_capacity_units" in candidates_df.columns:
        capacity = candidates_df.set_index("site_id")["max_capacity_units"].fillna(1e9).to_dict()
    else:
        capacity = {row["site_id"]: 1e9 for _, row in candidates_df.iterrows()}
    if "rent_monthly" in candidates_df.columns:
        fixed_cost = candidates_df.set_index("site_id")["rent_monthly"].fillna(0).astype(float).to_dict()
    else:
        fixed_cost = {row["site_id"]: 0.0 for _, row in candidates_df.iterrows()}

    max_open = int(constraints.get("max_sites", len(sites)))
    coverage = float(constraints.get("coverage", 0.99))

    cost = {(r.customer_id, r.site_id): float(r.total_cost) for r in cost_matrix_df.itertuples()}
    ttime = {(r.customer_id, r.site_id): float(r.travel_time_h) for r in cost_matrix_df.itertuples()}
    sla_h = demand_df.set_index("customer_id")["sla_days"].astype(float).mul(24).to_dict()

    model = pulp.LpProblem("cd_locator", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [(i, j) for i in customers for j in sites], lowBound=0, cat="Continuous")
    open_j = pulp.LpVariable.dicts("open", sites, lowBound=0, upBound=1, cat="Binary")

    wc = float(objective_weights.get("cost", 0.7))
    wt = float(objective_weights.get("time", 0.3))
    model += (
        pulp.lpSum(wc * cost[(i, j)] * x[(i, j)] + wt * ttime[(i, j)] * x[(i, j)] for i in customers for j in sites)
        + pulp.lpSum(fixed_cost.get(j, 0.0) * open_j[j] for j in sites)
    )

    for i in customers:
        model += pulp.lpSum(x[(i, j)] for j in sites) >= coverage * float(demand[i])
        for j in sites:
            if ttime[(i, j)] > sla_h[i]:
                model += x[(i, j)] == 0

    for j in sites:
        model += pulp.lpSum(x[(i, j)] for i in customers) <= float(capacity.get(j, 1e9)) * open_j[j]

    model += pulp.lpSum(open_j[j] for j in sites) <= max_open

    status = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=int(constraints.get("time_limit", 15))))
    if pulp.LpStatus[status] not in {"Optimal", "Not Solved", "Undefined", "Integer Feasible"}:
        out = fallback_greedy(candidates_df, demand_df, cost_matrix_df, constraints)
        out["solver_status"] = "failed; used_fallback"
        return out

    alloc: List[Dict[str, Any]] = []
    for i in customers:
        for j in sites:
            q = float(x[(i, j)].value() or 0.0)
            if q > 1e-6:
                row = cost_matrix_df[(cost_matrix_df["customer_id"] == i) & (cost_matrix_df["site_id"] == j)].iloc[0]
                alloc.append(
                    {
                        "customer_id": i,
                        "site_id": j,
                        "allocated_qty": q,
                        "unit_cost": float(row["total_cost"]),
                        "delivery_time_h": float(row["travel_time_h"]),
                    }
                )

    alloc_df = pd.DataFrame(alloc)
    open_rows = []
    for j in sites:
        if float(open_j[j].value() or 0) > 0.5:
            used = float(alloc_df.loc[alloc_df["site_id"] == j, "allocated_qty"].sum()) if not alloc_df.empty else 0.0
            cap = float(capacity.get(j, 1e9))
            open_rows.append({"site_id": j, "fixed_cost": float(fixed_cost.get(j, 0.0)), "utilization_pct": used / cap if cap else 0.0})

    open_df = pd.DataFrame(open_rows)
    breakdown = _build_breakdown(alloc_df, open_df)
    total = float(sum(breakdown.values()))
    return {
        "open_sites": open_rows,
        "allocations": alloc,
        "total_cost": total,
        "objective_breakdown": breakdown,
        "solver_status": pulp.LpStatus[status],
    }


def fallback_greedy(
    candidates_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    cost_matrix_df: pd.DataFrame,
    constraints: dict,
) -> Dict[str, Any]:
    """
    Capacity- and coverage-aware greedy fallback.
    """
    eps = 1e-9
    max_open = int(constraints.get("max_sites", 1))
    coverage_target = float(constraints.get("coverage", 0.99))

    avg_cost = cost_matrix_df.groupby("site_id")["total_cost"].mean().to_dict()
    scores: List[Tuple[str, float]] = []
    for _, r in candidates_df.iterrows():
        risk = 1 - (float(r.get("flood_risk_score", 0.0)) + float(r.get("crime_risk_score", 0.0))) / 2.0
        av_penalty = 1.0 if bool(r.get("availability", True)) else 0.01
        site_id = str(r["site_id"])
        score = (1.0 / (avg_cost.get(site_id, 1e9) + eps)) * av_penalty * (1.0 + risk)
        scores.append((site_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    open_site_ids = [s for s, _ in scores[:max_open]]

    open_candidates_df = candidates_df[candidates_df["site_id"].astype(str).isin(open_site_ids)].copy()
    if "max_capacity_units" in open_candidates_df.columns:
        remaining_capacity: Dict[str, float] = {
            str(row["site_id"]): float(row["max_capacity_units"])
            for _, row in open_candidates_df.iterrows()
        }
    else:
        remaining_capacity = {str(site_id): float("inf") for site_id in open_site_ids}

    allocations: List[Dict[str, Any]] = []
    total_cost = 0.0
    total_demand = 0.0
    unserved_demand = 0.0

    open_cost_matrix = cost_matrix_df[cost_matrix_df["site_id"].astype(str).isin(open_site_ids)]
    demand_id_col = "customer_id" if "customer_id" in demand_df.columns else ("demand_id" if "demand_id" in demand_df.columns else None)
    qty_col = "demand" if "demand" in demand_df.columns else ("demand_units" if "demand_units" in demand_df.columns else "qty")

    for _, d_row in demand_df.iterrows():
        demand_qty = float(d_row[qty_col]) * coverage_target
        total_demand += demand_qty

        if demand_id_col is not None:
            did = d_row[demand_id_col]
            candidate_rows = open_cost_matrix[open_cost_matrix[demand_id_col] == did].copy()
        else:
            candidate_rows = open_cost_matrix.copy()

        if candidate_rows.empty:
            unserved_demand += demand_qty
            continue

        candidate_rows = candidate_rows.sort_values("total_cost")
        remaining = demand_qty

        for _, c_row in candidate_rows.iterrows():
            site_id = str(c_row["site_id"])
            if remaining <= 0:
                break

            site_cap = remaining_capacity.get(site_id, 0.0)
            if site_cap <= 0:
                continue

            qty_to_allocate = min(remaining, site_cap)
            if qty_to_allocate <= 0:
                continue

            unit_cost = float(c_row["total_cost"])
            allocations.append(
                {
                    "site_id": site_id,
                    (demand_id_col or "demand_id"): d_row.get(demand_id_col, d_row.get("id", None)),
                    "allocated_qty": qty_to_allocate,
                    "unit_cost": unit_cost,
                    "delivery_time_h": float(c_row.get("travel_time_h", 0.0)),
                    "total_cost": qty_to_allocate * unit_cost,
                }
            )
            remaining_capacity[site_id] = site_cap - qty_to_allocate
            total_cost += qty_to_allocate * unit_cost
            remaining -= qty_to_allocate

        if remaining > 0:
            unserved_demand += remaining

    served_demand = total_demand - unserved_demand
    achieved_coverage = served_demand / total_demand if total_demand > 0 else 0.0

    open_sites = []
    for _, row in open_candidates_df.iterrows():
        sid = str(row["site_id"])
        init_cap = float(row.get("max_capacity_units", float("inf")))
        rem_cap = remaining_capacity.get(sid, 0.0)
        util = 0.0 if init_cap in (0.0, float("inf")) else (init_cap - rem_cap) / init_cap
        open_sites.append({"site_id": sid, "fixed_cost": float(row.get("rent_monthly", 0.0)), "utilization_pct": util})

    objective_breakdown = {
        "transport": total_cost,
        "fixed": float(open_candidates_df.get("rent_monthly", pd.Series(dtype=float)).fillna(0).sum()) if "rent_monthly" in open_candidates_df.columns else 0.0,
        "tax": 0.0,
        "inventory": 0.0,
        "time_penalty": 0.0,
        "served_demand_units": served_demand,
        "unserved_demand_units": unserved_demand,
        "achieved_coverage": achieved_coverage,
        "coverage_target": coverage_target,
    }

    return {
        "open_sites": open_sites,
        "allocations": allocations,
        "total_cost": total_cost + objective_breakdown["fixed"],
        "objective_breakdown": objective_breakdown,
        "solver_status": "FALLBACK_GREEDY",
    }
