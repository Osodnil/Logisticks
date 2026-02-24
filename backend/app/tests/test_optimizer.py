import pandas as pd

from app.services.cost_engine import calc_cost_matrix
from app.services.optimizer import fallback_greedy, solve_milp


def test_milp_not_worse_than_greedy() -> None:
    customers = pd.read_csv("example/customers_toy.csv")
    customers[["lat", "lon"]] = customers[["lat", "lon"]].fillna(method="ffill")
    sites = pd.read_csv("example/sites_toy.csv")
    tax = pd.read_json("example/tax_rules_toy.json")
    cm, _ = calc_cost_matrix(sites, customers, {"cost_per_km": 2.2, "avg_speed_kmh": 50, "handling_cost_per_order": 3.0}, tax)
    demand = customers[["customer_id", "sla_days"]].copy()
    demand["demand"] = customers["volume_month"].astype(float)
    constraints = {"max_sites": 3, "coverage": 0.99, "time_limit": 10}
    milp = solve_milp(sites, demand, cm, constraints, {"cost": 0.7, "time": 0.3})
    greedy = fallback_greedy(sites, demand, cm, constraints)
    assert greedy["total_cost"] + 1e-6 >= milp["total_cost"]
