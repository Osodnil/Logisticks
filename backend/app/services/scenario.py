"""Scenario simulation service."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.services.optimizer import solve_milp


def _run_one(args: Dict[str, Any]) -> Dict[str, Any]:
    return solve_milp(args["sites"], args["demand"], args["cost"], args["constraints"], args["weights"])


def run_scenarios(sites_df: pd.DataFrame, demand_df: pd.DataFrame, cost_matrix_df: pd.DataFrame, constraints: Dict[str, Any], objective_weights: Dict[str, Any], n_scenarios: int = 8) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for _ in range(n_scenarios):
        d = demand_df.copy()
        c = cost_matrix_df.copy()
        d["demand"] = d["demand"] * rng.uniform(0.9, 1.1)
        c["total_cost"] = c["total_cost"] * rng.uniform(0.9, 1.1)
        jobs.append({"sites": sites_df, "demand": d, "cost": c, "constraints": constraints, "weights": objective_weights})

    with mp.Pool(processes=min(4, n_scenarios)) as pool:
        results = pool.map(_run_one, jobs)

    totals = np.array([r["total_cost"] for r in results], dtype=float)
    site_counter: Dict[str, int] = {}
    for r in results:
        for s in r["open_sites"]:
            site_counter[s["site_id"]] = site_counter.get(s["site_id"], 0) + 1
    recurring = [s for s, c in site_counter.items() if c / n_scenarios >= 0.7]

    summary = {
        "mean": float(np.mean(totals)),
        "std": float(np.std(totals)),
        "p10": float(np.quantile(totals, 0.1)),
        "p90": float(np.quantile(totals, 0.9)),
        "sites_ge_70pct": recurring,
    }
    Path("example/output").mkdir(parents=True, exist_ok=True)
    Path("example/output/scenario_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
