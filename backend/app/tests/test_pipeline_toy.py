import pandas as pd

from app.services.cost_engine import calc_cost_matrix
from app.services.forecast import forecast_demand
from app.services.optimizer import solve_milp
from app.services.validator import validate_customers, validate_sites, validate_tax_rules


def test_pipeline_toy_sync() -> None:
    cv = validate_customers("example/customers_toy.csv")
    sv = validate_sites("example/sites_toy.csv")
    tv = validate_tax_rules("example/tax_rules_toy.json")
    customers = pd.read_parquet(cv["cleaned_dataframe"])
    sites = pd.read_parquet(sv["cleaned_dataframe"])
    tax = pd.read_parquet(tv["cleaned_dataframe"])

    fcst, _ = forecast_demand(customers, horizon_months=6, cluster_k=2)
    demand = fcst.groupby("customer_id")["forecast_volume"].sum().reset_index().rename(columns={"forecast_volume": "demand"})
    demand = demand.merge(customers[["customer_id", "sla_days"]], on="customer_id", how="left")
    cm, _ = calc_cost_matrix(sites, customers, {"cost_per_km": 2.2, "avg_speed_kmh": 50, "handling_cost_per_order": 3.0}, tax)
    result = solve_milp(sites, demand, cm, {"max_sites": 3, "coverage": 0.99, "time_limit": 20}, {"cost": 0.7, "time": 0.3})
    assert result["open_sites"]
    assert isinstance(result["total_cost"], float)
    assert result["total_cost"] > 0
