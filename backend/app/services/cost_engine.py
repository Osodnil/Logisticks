"""Cost matrix calculation engine."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import requests

from app.core.config import get_settings


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371
    p1, p2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return float(2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


@lru_cache(maxsize=20000)
def route_via_osrm(orig: Tuple[float, float], dest: Tuple[float, float]) -> Dict[str, float]:
    """OSRM hook with safe fallback to haversine when unavailable."""
    settings = get_settings()
    url = f"{settings.osrm_url}/route/v1/driving/{orig[1]},{orig[0]};{dest[1]},{dest[0]}"
    try:
        resp = requests.get(url, params={"overview": "false"}, timeout=2)
        resp.raise_for_status()
        payload = resp.json()
        route = payload["routes"][0]
        return {"distance": float(route["distance"]), "time": float(route["duration"]) / 3600.0}
    except Exception:
        dist_km = haversine_km(orig[0], orig[1], dest[0], dest[1])
        return {"distance": dist_km * 1000, "time": 0.0}




@lru_cache(maxsize=20000)
def route_via_graphhopper(orig: Tuple[float, float], dest: Tuple[float, float]) -> Dict[str, float]:
    """GraphHopper hook with safe fallback to haversine when unavailable."""
    settings = get_settings()
    url = f"{settings.graphhopper_url}/route"
    try:
        resp = requests.get(
            url,
            params={"point": [f"{orig[0]},{orig[1]}", f"{dest[0]},{dest[1]}"], "profile": "car"},
            timeout=2,
        )
        resp.raise_for_status()
        payload = resp.json()
        path = payload["paths"][0]
        return {"distance": float(path["distance"]), "time": float(path["time"]) / 3600000.0}
    except Exception:
        dist_km = haversine_km(orig[0], orig[1], dest[0], dest[1])
        return {"distance": dist_km * 1000, "time": 0.0}


def _mock_uf_from_cep(cep: str | None) -> str:
    if not cep:
        return "SP"
    cep = str(cep).strip('"')
    if cep.startswith("2"):
        return "RJ"
    if cep.startswith("8"):
        return "PR"
    return "SP"


def calc_cost_matrix(sites_df: pd.DataFrame, customers_df: pd.DataFrame, transport_params: Dict[str, Any], tax_rules: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    settings = get_settings()
    road_factor = float(transport_params.get("road_factor", 1.3))
    avg_speed = float(transport_params.get("avg_speed_kmh", 50.0))
    cost_per_km = float(transport_params.get("cost_per_km", 2.2))
    handling = float(transport_params.get("handling_cost_per_order", 3.0))
    use_osrm = transport_params.get("use_osrm", False) or settings.routing_backend == "osrm"
    use_graphhopper = transport_params.get("use_graphhopper", False) or settings.routing_backend == "graphhopper"

    rows = []
    tax_idx = {str(r["uf"]): float(r["icms_rate"]) for _, r in tax_rules.iterrows()} if not tax_rules.empty else {}

    for _, s in sites_df.iterrows():
        for _, c in customers_df.iterrows():
            if use_osrm:
                route = route_via_osrm((float(s["lat"]), float(s["lon"])), (float(c["lat"]), float(c["lon"])))
                dist = (float(route["distance"]) / 1000.0) * road_factor
                time_h = float(route["time"]) if route["time"] > 0 else dist / max(avg_speed, 1e-9)
            elif use_graphhopper:
                route = route_via_graphhopper((float(s["lat"]), float(s["lon"])), (float(c["lat"]), float(c["lon"])))
                dist = (float(route["distance"]) / 1000.0) * road_factor
                time_h = float(route["time"]) if route["time"] > 0 else dist / max(avg_speed, 1e-9)
            else:
                dist = haversine_km(float(s["lat"]), float(s["lon"]), float(c["lat"]), float(c["lon"])) * road_factor
                time_h = dist / max(avg_speed, 1e-9)

            transport = dist * cost_per_km + handling * float(c.get("orders_per_month", 0))
            uf = _mock_uf_from_cep(c.get("cep"))
            tax_rate = tax_idx.get(uf, 0.12)
            tax_cost = transport * tax_rate
            total = transport + tax_cost
            rows.append({
                "site_id": s["site_id"],
                "customer_id": c["customer_id"],
                "distance_km": dist,
                "travel_time_h": time_h,
                "transport_cost": transport,
                "tax_cost": tax_cost,
                "total_cost": total,
            })

    out = pd.DataFrame(rows)
    Path("example/output").mkdir(parents=True, exist_ok=True)
    out.to_parquet("example/output/cost_matrix.parquet", index=False)
    summary = {
        "mean_cost": float(out["total_cost"].mean()),
        "p10": float(out["total_cost"].quantile(0.1)),
        "p90": float(out["total_cost"].quantile(0.9)),
    }
    return out, summary
