import pytest

pd = pytest.importorskip("pandas")
from fastapi.testclient import TestClient

from app.main import app
from app.services.cost_engine import calc_cost_matrix
from app.services.forecast import forecast_demand
from app.services.optimizer import solve_milp
from app.services.validator import validate_customers, validate_sites, validate_tax_rules

client = TestClient(app)


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


def _upload_all_toy_inputs(role: str = "admin") -> None:
    with open("example/customers_toy.csv", "rb") as f:
        resp = client.post(
            "/upload/customers",
            headers={"X-User-Role": role, "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("customers_toy.csv", f, "text/csv")},
        )
    assert resp.status_code == 200

    with open("example/sites_toy.csv", "rb") as f:
        resp = client.post(
            "/upload/sites",
            headers={"X-User-Role": role, "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("sites_toy.csv", f, "text/csv")},
        )
    assert resp.status_code == 200

    with open("example/tax_rules_toy.json", "rb") as f:
        resp = client.post(
            "/upload/tax_rules",
            headers={"X-User-Role": role, "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("tax_rules_toy.json", f, "application/json")},
        )
    assert resp.status_code == 200


def test_upload_endpoints_rbac_allowed() -> None:
    with open("example/customers_toy.csv", "rb") as f:
        resp = client.post(
            "/upload/customers",
            headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("customers_toy.csv", f, "text/csv")},
        )
    assert resp.status_code == 200

    with open("example/sites_toy.csv", "rb") as f:
        resp = client.post(
            "/upload/sites",
            headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("sites_toy.csv", f, "text/csv")},
        )
    assert resp.status_code == 200

    with open("example/tax_rules_toy.json", "rb") as f:
        resp = client.post(
            "/upload/tax_rules",
            headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
            files={"file": ("tax_rules_toy.json", f, "application/json")},
        )
    assert resp.status_code == 200


def test_upload_endpoints_rbac_forbidden() -> None:
    with open("example/customers_toy.csv", "rb") as f:
        resp = client.post(
            "/upload/customers",
            headers={"X-User-Role": "viewer", "X-User-Scopes": "cd:write"},
            files={"file": ("customers_toy.csv", f, "text/csv")},
        )
    assert resp.status_code in (401, 403)


def test_run_analysis_sync_with_consent() -> None:
    _upload_all_toy_inputs(role="admin")

    resp = client.post(
        "/run/analysis",
        headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
        params={"sync": True},
        json={"consent_to_use_sensitive_data": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    run_id = body["run_id"]

    status_resp = client.get(f"/status/{run_id}", headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"})
    assert status_resp.status_code == 200

    results_resp = client.get(f"/results/{run_id}", headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"})
    assert results_resp.status_code == 200


def test_run_analysis_async_with_consent() -> None:
    _upload_all_toy_inputs(role="admin")

    resp = client.post(
        "/run/analysis",
        headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
        params={"sync": False},
        json={"consent_to_use_sensitive_data": True},
    )
    # async runs commonly return 202; allow 200 as well if implementation differs
    assert resp.status_code in (200, 202)
    body = resp.json()
    assert "run_id" in body
    run_id = body["run_id"]

    status_resp = client.get(f"/status/{run_id}", headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"})
    assert status_resp.status_code == 200


def test_run_analysis_rejected_without_consent_when_sensitive_data_present() -> None:
    _upload_all_toy_inputs(role="admin")

    resp = client.post(
        "/run/analysis",
        headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
        params={"sync": True},
        json={"consent_to_use_sensitive_data": False},
    )
    assert resp.status_code in (400, 403)


def test_status_and_results_missing_run() -> None:
    status_resp = client.get("/status/nonexistent-run-id", headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"})
    assert status_resp.status_code in (404, 400)

    results_resp = client.get("/results/nonexistent-run-id", headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"})
    assert results_resp.status_code in (404, 400)


def test_metrics_after_uploads_and_run() -> None:
    _upload_all_toy_inputs(role="admin")

    run_resp = client.post(
        "/run/analysis",
        headers={"X-User-Role": "admin", "X-User-Scopes": "cd:write,cd:run"},
        params={"sync": True},
        json={"consent_to_use_sensitive_data": True},
    )
    assert run_resp.status_code in (200, 202)

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    content_type = metrics_resp.headers.get("content-type", "")
    assert content_type.startswith("text/plain")
    body = metrics_resp.text
    # basic sanity checks that some counters are present, without depending
    # on exact metric names
    assert "upload" in body or "run" in body
