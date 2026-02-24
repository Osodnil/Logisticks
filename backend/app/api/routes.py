"""API routes for upload and analysis."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.audit import append_audit_event
from app.services.cost_engine import calc_cost_matrix
from app.services.forecast import forecast_demand
from app.services.optimizer import solve_milp
from app.services.report_builder import build_reports
from app.services.state_store import StateStore
from app.services.validator import validate_customers, validate_sites, validate_suppliers, validate_tax_rules
from app.utils.logging_setup import log_event

router = APIRouter()
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)
PROJECTS: Dict[str, Dict[str, str]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}
METRICS = {"uploads_total": 0, "runs_total": 0}
SETTINGS = get_settings()
STORE = StateStore(SETTINGS.db_url)


class AnalysisRequest(BaseModel):
    project_id: str = "proj_toy"
    params: Dict[str, Any] = Field(default_factory=dict)
    consent_to_use_sensitive_data: bool = False


def _save_upload(project_id: str, name: str, upload: UploadFile) -> str:
    pdir = DATA_ROOT / project_id
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / name
    with out.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    PROJECTS.setdefault(project_id, {})[name] = str(out)
    STORE.save_project(project_id, PROJECTS[project_id])
    METRICS["uploads_total"] += 1
    return str(out)


def _get_role(request: Request) -> str:
    return request.headers.get("X-User-Role", "viewer")


def _has_scope(request: Request, required_scope: str) -> bool:
    scopes = {s.strip() for s in request.headers.get("X-User-Scopes", "").split(",") if s.strip()}
    return required_scope in scopes if required_scope else True


def _has_sensitive_columns(project_id: str) -> bool:
    files = PROJECTS.get(project_id, {})
    candidate_files = [files.get("customers.csv"), files.get("sites.csv")]
    sensitive_cols = {"salary", "internal_cost", "rent_monthly"}
    for raw_path in candidate_files:
        if not raw_path:
            continue
        try:
            cols = {c.lower() for c in pd.read_csv(raw_path, nrows=1).columns}
        except Exception:
            continue
        if cols.intersection(sensitive_cols):
            return True
    return False


def _run_pipeline(run_id: str, payload: AnalysisRequest, logger) -> None:
    RUNS[run_id]["status"] = "running"
    project_files = PROJECTS.get(payload.project_id) or STORE.get_project(payload.project_id)
    if not project_files:
        RUNS[run_id]["status"] = "failed"
        RUNS[run_id]["error"] = "project not found"
        return

    customers = pd.read_parquet(project_files["customers_cleaned.parquet"])
    sites = pd.read_parquet(project_files["sites_cleaned.parquet"])
    tax = pd.read_parquet(project_files["tax_rules_cleaned.parquet"])

    fcst, metrics = forecast_demand(customers, int(payload.params.get("horizon_months", 12)))
    demand = fcst.groupby("customer_id")["forecast_volume"].sum().reset_index().rename(columns={"forecast_volume": "demand"})
    demand = demand.merge(customers[["customer_id", "sla_days"]], on="customer_id", how="left")

    cost_matrix, _ = calc_cost_matrix(sites, customers, payload.params.get("transport_params", {}), tax)
    result = solve_milp(
        sites,
        demand,
        cost_matrix,
        {"max_sites": int(payload.params.get("max_sites", 3)), "coverage": 0.99, "time_limit": 20},
        payload.params.get("objective_weights", {"cost": 0.7, "time": 0.3}),
    )

    artifacts = build_reports(result)
    RUNS[run_id].update({"status": "completed", "result": result, "artifacts": artifacts, "fit_metrics": metrics.to_dict(orient="records")})
    STORE.save_run(run_id, RUNS[run_id])
    append_audit_event(SETTINGS.audit_log_path, {"event": "run_completed", "run_id": run_id, "solver_status": result.get("solver_status")})
    log_event(logger, "info", "analysis completed", run_id=run_id)


@router.post("/upload/customers")
async def upload_customers(request: Request, file: UploadFile = File(...), project_id: str = "proj_toy") -> Dict[str, Any]:
    if _get_role(request) not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="role not allowed")
    if not _has_scope(request, SETTINGS.required_scope_upload):
        raise HTTPException(status_code=403, detail="missing required scope")
    path = _save_upload(project_id, "customers.csv", file)
    out = validate_customers(path)
    if out["errors"]:
        raise HTTPException(status_code=400, detail=out)
    PROJECTS.setdefault(project_id, {})["customers_cleaned.parquet"] = out["cleaned_dataframe"]
    STORE.save_project(project_id, PROJECTS[project_id])
    return {"status": "ok", "errors": out["errors"], "warnings": out["warnings"], "project_id": project_id}


@router.post("/upload/sites")
async def upload_sites(request: Request, file: UploadFile = File(...), project_id: str = "proj_toy") -> Dict[str, Any]:
    if _get_role(request) not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="role not allowed")
    if not _has_scope(request, SETTINGS.required_scope_upload):
        raise HTTPException(status_code=403, detail="missing required scope")
    path = _save_upload(project_id, "sites.csv", file)
    out = validate_sites(path)
    if out["errors"]:
        raise HTTPException(status_code=400, detail=out)
    PROJECTS.setdefault(project_id, {})["sites_cleaned.parquet"] = out["cleaned_dataframe"]
    STORE.save_project(project_id, PROJECTS[project_id])
    return {"status": "ok", "errors": out["errors"], "warnings": out["warnings"], "project_id": project_id}


@router.post("/upload/suppliers")
async def upload_suppliers(request: Request, file: UploadFile = File(...), project_id: str = "proj_toy") -> Dict[str, Any]:
    if _get_role(request) not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="role not allowed")
    if not _has_scope(request, SETTINGS.required_scope_upload):
        raise HTTPException(status_code=403, detail="missing required scope")
    path = _save_upload(project_id, "suppliers.csv", file)
    out = validate_suppliers(path)
    if out["errors"]:
        raise HTTPException(status_code=400, detail=out)
    PROJECTS.setdefault(project_id, {})["suppliers_cleaned.parquet"] = out["cleaned_dataframe"]
    STORE.save_project(project_id, PROJECTS[project_id])
    return {"status": "ok", "errors": out["errors"], "warnings": out["warnings"], "project_id": project_id}


@router.post("/upload/tax_rules")
async def upload_tax_rules(request: Request, file: UploadFile = File(...), project_id: str = "proj_toy") -> Dict[str, Any]:
    if _get_role(request) not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="role not allowed")
    if not _has_scope(request, SETTINGS.required_scope_upload):
        raise HTTPException(status_code=403, detail="missing required scope")
    path = _save_upload(project_id, "tax_rules.json", file)
    out = validate_tax_rules(path)
    if out["errors"]:
        raise HTTPException(status_code=400, detail=out)
    PROJECTS.setdefault(project_id, {})["tax_rules_cleaned.parquet"] = out["cleaned_dataframe"]
    STORE.save_project(project_id, PROJECTS[project_id])
    return {"status": "ok", "errors": out["errors"], "warnings": out["warnings"], "project_id": project_id}


@router.post("/run/analysis")
async def run_analysis(payload: AnalysisRequest, request: Request, background_tasks: BackgroundTasks, sync: bool = Query(default=False)) -> Dict[str, Any]:
    if _get_role(request) not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="role not allowed")
    if not _has_scope(request, SETTINGS.required_scope_run):
        raise HTTPException(status_code=403, detail="missing required scope")
    has_sensitive = _has_sensitive_columns(payload.project_id)
    if has_sensitive and not payload.consent_to_use_sensitive_data:
        raise HTTPException(status_code=400, detail="consent_to_use_sensitive_data must be true")

    run_id = f"run_{uuid4().hex[:8]}"
    METRICS["runs_total"] += 1
    raw = json.dumps(payload.dict(), sort_keys=True).encode("utf-8")
    RUNS[run_id] = {
        "status": "queued",
        "project_id": payload.project_id,
        "params": payload.params,
        "input_hash": hashlib.sha256(raw).hexdigest(),
        "model_versions": {"optimizer": "pulp-2.7.0", "forecast": "prophet|sarimax"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_audit_event(SETTINGS.audit_log_path, {"event": "run_created", "run_id": run_id, "project_id": payload.project_id, "params": payload.params})
    STORE.save_run(run_id, RUNS[run_id])
    logger = request.app.state.logger
    if sync:
        _run_pipeline(run_id, payload, logger)
        return {"run_id": run_id, "status": RUNS[run_id]["status"]}
    background_tasks.add_task(_run_pipeline, run_id, payload, logger)
    return {"run_id": run_id, "status": "running", "job_backend": SETTINGS.job_backend}


@router.get("/status/{run_id}")
def status(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        restored = STORE.get_run(run_id)
        if not restored:
            raise HTTPException(status_code=404, detail="run not found")
        RUNS[run_id] = restored
    r = RUNS[run_id].copy()
    r.pop("result", None)
    return r


@router.get("/results/{run_id}")
def results(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        restored = STORE.get_run(run_id)
        if not restored:
            raise HTTPException(status_code=404, detail="run not found")
        RUNS[run_id] = restored
    if RUNS[run_id].get("status") != "completed":
        return {"run_id": run_id, "status": RUNS[run_id].get("status")}
    return {"run_id": run_id, "status": "completed", "result": RUNS[run_id]["result"], "artifacts": RUNS[run_id]["artifacts"]}


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "\n".join([f"uploads_total {METRICS['uploads_total']}", f"runs_total {METRICS['runs_total']}"])
