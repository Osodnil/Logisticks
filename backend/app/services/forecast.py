"""Demand forecasting with Prophet fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet  # type: ignore
except Exception:  # noqa
    Prophet = None


def _pick_k(coords: np.ndarray) -> int:
    max_k = min(50, max(1, len(coords)))
    if len(coords) <= 2:
        return 1
    inertias = []
    for k in range(1, min(max_k, 8) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(coords)
        inertias.append(model.inertia_)
    if len(inertias) < 3:
        return 1
    diffs = np.diff(inertias)
    return int(np.argmin(np.abs(diffs[1:] - diffs[:-1])) + 2)


def _build_series(volume: float, months: int = 18) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="MS")
    trend = np.linspace(0.9, 1.1, months)
    return pd.DataFrame({"ds": idx, "y": volume * trend})


def forecast_demand(customers_df: pd.DataFrame, horizon_months: int, cluster_k: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = customers_df.copy()
    coords = df[["lat", "lon"]].fillna(0).to_numpy()
    k = cluster_k or _pick_k(coords)
    km = KMeans(n_clusters=max(1, min(k, len(df))), random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(coords)

    rows = []
    metrics = []
    Path("example/output").mkdir(parents=True, exist_ok=True)
    for cluster_id, g in df.groupby("cluster_id"):
        total_volume = float(g["volume_month"].astype(float).sum())
        hist = _build_series(total_volume)
        future_idx = pd.date_range(hist["ds"].max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

        if Prophet is not None and len(hist) >= 12:
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            model.fit(hist)
            fut = pd.DataFrame({"ds": future_idx})
            pred = model.predict(fut)["yhat"].clip(lower=0).to_numpy()
            fitted = model.predict(hist[["ds"]])["yhat"].to_numpy()
        elif len(hist) >= 12:
            m = SARIMAX(hist["y"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
            pred = np.maximum(m.forecast(horizon_months).to_numpy(), 0)
            fitted = m.fittedvalues.to_numpy()
        else:
            avg = float(hist["y"].tail(3).mean())
            pred = np.array([avg] * horizon_months)
            fitted = hist["y"].to_numpy()

        mae = float(np.mean(np.abs(hist["y"].to_numpy() - fitted[: len(hist)])))
        mape = float(np.mean(np.abs((hist["y"].to_numpy() - fitted[: len(hist)]) / np.maximum(hist["y"].to_numpy(), 1e-9))))
        metrics.append({"cluster_id": int(cluster_id), "mae": mae, "mape": mape})

        plt.figure(figsize=(6, 3))
        plt.plot(hist["ds"], hist["y"], label="real")
        plt.plot(hist["ds"], fitted[: len(hist)], label="pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"example/output/forecast_cluster_{cluster_id}.png")
        plt.close()

        for _, cust in g.iterrows():
            share = float(cust["volume_month"]) / max(total_volume, 1e-9)
            for dt, val in zip(future_idx, pred):
                rows.append({"customer_id": cust["customer_id"], "month": dt.strftime("%Y-%m"), "forecast_volume": float(val * share)})

    return pd.DataFrame(rows), pd.DataFrame(metrics)
