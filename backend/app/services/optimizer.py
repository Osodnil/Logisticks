"""Optimization engine with MILP and greedy fallback."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pulp


def _build_breakdown(alloc_df: pd.DataFrame, open_sites: pd.DataFrame) -> Dict[str, float]:
    transport = float((alloc_df["allocated_qty"] * alloc_df["unit_cost"]).sum())
    fixed = float(open_sites["fixed_cost"].sum()) if not open_sites.empty else 0.0
    return {
        "transport": transport,
        "fixed": fixed,
        "tax": float((alloc_df["allocated_qty"] * alloc_df.get("tax_rate", 0)).sum()),
        "inventory": 0.0,
        "time_penalty": float((alloc_df["allocated_qty"] * alloc_df["delivery_time_h"] * 0.1).sum()),
    }


def solve_milp(candidates_df: pd.DataFrame, demand_df: pd.DataFrame, cost_matrix_df: pd.DataFrame, constraints: dict, objective_weights: dict) -> Dict[str, Any]:
    customers = demand_df["customer_id"].tolist()
    sites = candidates_df[candidates_df.get("availability", True) == True]["site_id"].tolist()  # noqa
    demand = demand_df.set_index("customer_id")["demand"].to_dict()
    capacity = candidates_df.set_index("site_id")["max_capacity_units"].fillna(1e9).to_dict()
    fixed_cost = candidates_df.set_index("site_id")["rent_monthly"].fillna(0).to_dict()
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
    model += pulp.lpSum(wc * cost[(i, j)] * x[(i, j)] + wt * ttime[(i, j)] * x[(i, j)] for i in customers for j in sites) + pulp.lpSum(fixed_cost.get(j, 0.0) * open_j[j] for j in sites)

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

    alloc = []
    for i in customers:
        for j in sites:
            q = float(x[(i, j)].value() or 0.0)
            if q > 1e-6:
                row = cost_matrix_df[(cost_matrix_df["customer_id"] == i) & (cost_matrix_df["site_id"] == j)].iloc[0]
                alloc.append({"customer_id": i, "site_id": j, "allocated_qty": q, "unit_cost": float(row["total_cost"]), "delivery_time_h": float(row["travel_time_h"])})

    alloc_df = pd.DataFrame(alloc)
    open_rows = []
    for j in sites:
        if float(open_j[j].value() or 0) > 0.5:
            used = float(alloc_df.loc[alloc_df["site_id"] == j, "allocated_qty"].sum()) if not alloc_df.empty else 0.0
            cap = float(capacity.get(j, 1e9))
            open_rows.append({"site_id": j, "fixed_cost": float(fixed_cost.get(j, 0.0)), "utilization_pct": used / cap if cap else 0.0})

    open_df = pd.DataFrame(open_rows)
    breakdown = _build_breakdown(alloc_df if not alloc_df.empty else pd.DataFrame(columns=["allocated_qty", "unit_cost", "delivery_time_h"]), open_df)
    total = float(sum(breakdown.values()))
    return {
        "open_sites": open_rows,
        "allocations": alloc,
        "total_cost": total,
        "objective_breakdown": breakdown,
        "solver_status": pulp.LpStatus[status],
    }


def fallback_greedy(candidates_df: pd.DataFrame, demand_df: pd.DataFrame, cost_matrix_df: pd.DataFrame, constraints: dict) -> Dict[str, Any]:
    eps = 1e-9
    max_open = int(constraints.get("max_sites", 1))
    coverage = float(constraints.get("coverage", 0.99))

    avg_cost = cost_matrix_df.groupby("site_id")["total_cost"].mean().to_dict()
    scores = []
    for _, r in candidates_df.iterrows():
        risk = 1 - (float(r.get("flood_risk_score", 0)) + float(r.get("crime_risk_score", 0))) / 2
        av_penalty = 1.0 if bool(r.get("availability", True)) else 0.01
        score = (1 / (avg_cost.get(r["site_id"], 1e9) + eps)) * av_penalty * (1 + risk)
        scores.append((r["site_id"], score))
    selected = [s for s, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:max_open]]

    alloc = []
    total_demand = demand_df["demand"].sum()
    allocated = 0.0
    for _, d in demand_df.sort_values("demand", ascending=False).iterrows():
        subset = cost_matrix_df[(cost_matrix_df["customer_id"] == d["customer_id"]) & (cost_matrix_df["site_id"].isin(selected))].sort_values("total_cost")
        if subset.empty:
            continue
        row = subset.iloc[0]
        qty = float(d["demand"]) * coverage
        allocated += qty
        alloc.append({"customer_id": d["customer_id"], "site_id": row["site_id"], "allocated_qty": qty, "unit_cost": float(row["total_cost"]), "delivery_time_h": float(row["travel_time_h"])})

    alloc_df = pd.DataFrame(alloc)
    open_rows = [{"site_id": s, "fixed_cost": float(candidates_df.loc[candidates_df["site_id"] == s, "rent_monthly"].fillna(0).iloc[0]), "utilization_pct": float(alloc_df.loc[alloc_df["site_id"] == s, "allocated_qty"].sum() / max(float(candidates_df.loc[candidates_df["site_id"] == s, "max_capacity_units"].fillna(1e9).iloc[0]), 1e-9)) if not alloc_df.empty else 0.0} for s in selected]
    open_df = pd.DataFrame(open_rows)
    breakdown = _build_breakdown(alloc_df if not alloc_df.empty else pd.DataFrame(columns=["allocated_qty", "unit_cost", "delivery_time_h"]), open_df)
    total = float(sum(breakdown.values()))
    return {"open_sites": open_rows, "allocations": alloc, "total_cost": total, "objective_breakdown": breakdown, "solver_status": "greedy"}
