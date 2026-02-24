"""Validation services for incoming files."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import pandas as pd
from pydantic import BaseModel, ValidationError

from app.models.schemas import Customer, Site, Supplier, TaxRule

CEP_RE = re.compile(r"^\d{5}-?\d{3}$")


def _coerce_nan(value: Any) -> Any:
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, (bool,)):  # scalar path
            return None if is_na else value
    except Exception:
        return value
    return value


def mock_geocode_from_cep(cep: str) -> Tuple[float, float]:
    base = sum(ord(c) for c in cep if c.isdigit())
    lat = -33 + (base % 1000) / 100
    lon = -55 + (base % 1300) / 100
    return max(-90, min(90, lat)), max(-180, min(180, lon))


def _save_clean(df: pd.DataFrame, prefix: str) -> str:
    path = Path(tempfile.gettempdir()) / f"{prefix}_cleaned.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def _validate_df(df: pd.DataFrame, schema: Type[BaseModel], id_field: str) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    cleaned_rows: List[Dict[str, Any]] = []

    required = {
        name
        for name, field in schema.__fields__.items()
        if getattr(field, "required", False)
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        for c in sorted(missing_cols):
            errors.append({"line": 0, "field": c, "message": f"missing required column: {c}", "severity": "error"})
        return {"errors": errors, "warnings": warnings, "cleaned_dataframe": None}

    seen_ids = set()
    seen_coords: List[Tuple[float, float]] = []
    for idx, row in df.iterrows():
        payload = {k: _coerce_nan(v) for k, v in row.to_dict().items()}
        if "cep" in payload and payload.get("cep") is not None and not CEP_RE.match(str(payload["cep"]).strip('"')):
            errors.append({"line": int(idx + 2), "field": "cep", "message": f"invalid cep '{payload['cep']}'", "severity": "error"})
        if payload.get("lat") is None and payload.get("lon") is None and payload.get("cep"):
            lat, lon = mock_geocode_from_cep(str(payload["cep"]).strip('"'))
            payload["lat"], payload["lon"] = lat, lon
            warnings.append({"line": int(idx + 2), "field": "lat/lon", "message": "geocoding_estimated", "severity": "warning"})
        try:
            obj = schema(**payload)
            data = obj.dict()
            if id_field in data:
                if data[id_field] in seen_ids:
                    errors.append({"line": int(idx + 2), "field": id_field, "message": f"duplicate {id_field}", "severity": "error"})
                seen_ids.add(data[id_field])
            if data.get("lat") is not None:
                coords = (float(data["lat"]), float(data["lon"]))
                for c in seen_coords:
                    if abs(c[0] - coords[0]) <= 1e-6 and abs(c[1] - coords[1]) <= 1e-6:
                        warnings.append({"line": int(idx + 2), "field": "lat/lon", "message": "duplicate coordinates", "severity": "warning"})
                seen_coords.append(coords)
            cleaned_rows.append(data)
        except ValidationError as exc:
            for err in exc.errors():
                errors.append({"line": int(idx + 2), "field": ".".join(str(v) for v in err["loc"]), "message": err["msg"], "severity": "error"})

    cleaned_df = pd.DataFrame(cleaned_rows)
    clean_path = _save_clean(cleaned_df, id_field) if not cleaned_df.empty else None
    return {"errors": errors, "warnings": warnings, "cleaned_dataframe": clean_path}


def validate_customers(file_path: str) -> Dict[str, Any]:
    return _validate_df(pd.read_csv(file_path), Customer, "customer_id")


def validate_sites(file_path: str) -> Dict[str, Any]:
    return _validate_df(pd.read_csv(file_path), Site, "site_id")


def validate_suppliers(file_path: str) -> Dict[str, Any]:
    return _validate_df(pd.read_csv(file_path), Supplier, "supplier_id")


def validate_tax_rules(file_path: str) -> Dict[str, Any]:
    if file_path.endswith(".json"):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    return _validate_df(df, TaxRule, "uf")
