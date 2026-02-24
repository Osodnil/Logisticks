"""Pydantic data contracts for CD locator."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, condecimal, confloat, conint, constr, root_validator, validator


class _LatLonMixin(BaseModel):
    lat: Optional[float]
    lon: Optional[float]

    @validator("lat")
    def validate_lat(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (-90 <= value <= 90):
            raise ValueError("lat must be between -90 and 90")
        return value

    @validator("lon")
    def validate_lon(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (-180 <= value <= 180):
            raise ValueError("lon must be between -180 and 180")
        return value

    @root_validator
    def validate_pair(cls, values):
        lat, lon = values.get("lat"), values.get("lon")
        if (lat is None) != (lon is None):
            raise ValueError("lat/lon must be both present or both absent")
        return values


class Customer(_LatLonMixin):
    customer_id: constr(strip_whitespace=True, min_length=1)
    cep: Optional[str]
    volume_month: condecimal(gt=-0.0001)
    weight_kg: Optional[condecimal(ge=0)]
    cube_m3: Optional[condecimal(ge=0)]
    orders_per_month: conint(ge=0)
    sla_days: conint(gt=0)
    growth_rate_annual: Optional[confloat(ge=0)]

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_A",
                "lat": -23.55,
                "lon": -46.63,
                "cep": "01001-000",
                "volume_month": 1000,
                "weight_kg": 10,
                "cube_m3": 0.02,
                "orders_per_month": 120,
                "sla_days": 2,
                "growth_rate_annual": 0.05,
            }
        }


class Site(_LatLonMixin):
    site_id: constr(strip_whitespace=True, min_length=1)
    area_m2: Optional[confloat(ge=0)]
    rent_monthly: Optional[condecimal(ge=0)]
    iptu_annual: Optional[condecimal(ge=0)]
    availability: bool = True
    max_capacity_units: Optional[conint(ge=0)]
    flood_risk_score: confloat(ge=0, le=1) = 0.0
    crime_risk_score: confloat(ge=0, le=1) = 0.0

    class Config:
        schema_extra = {
            "example": {
                "site_id": "SITE_1",
                "lat": -23.56,
                "lon": -46.65,
                "area_m2": 5000,
                "rent_monthly": 20000,
                "iptu_annual": 5000,
                "availability": True,
                "max_capacity_units": 20000,
                "flood_risk_score": 0.1,
                "crime_risk_score": 0.2,
            }
        }


class Supplier(_LatLonMixin):
    supplier_id: constr(strip_whitespace=True, min_length=1)
    lead_time_days: conint(ge=0)
    modal: Optional[str]
    cost_per_km: Optional[condecimal(ge=0)]

    class Config:
        schema_extra = {"example": {"supplier_id": "SUP_1", "lat": -23.5, "lon": -46.6, "lead_time_days": 3, "modal": "road", "cost_per_km": 2.3}}


class TaxRule(BaseModel):
    uf: constr(min_length=2, max_length=2)
    operation_type: constr(min_length=3)
    icms_rate: confloat(ge=0, le=1)
    diferencial_aliquota: Optional[confloat(ge=0, le=1)] = 0.0
    incentives: Optional[List[dict]] = []

    class Config:
        schema_extra = {"example": {"uf": "SP", "operation_type": "sale", "icms_rate": 0.18, "diferencial_aliquota": 0.0, "incentives": []}}
