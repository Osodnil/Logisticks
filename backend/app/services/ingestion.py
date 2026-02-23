"""Data ingestion helpers."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def read_csv_to_df(path: str, schema: Type[BaseModel]) -> pd.DataFrame:
    """Read csv, normalize column names and coerce supported types."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    field_names = set(schema.__fields__.keys())
    missing = field_names - set(df.columns)
    for col in missing:
        df[col] = None
    for name, field in schema.__fields__.items():
        if name not in df.columns:
            continue
        if field.type_ in (int, float):
            df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def export_template(schema: Type[BaseModel], out_path: str) -> None:
    """Export an empty CSV template from a pydantic schema."""
    columns: Iterable[str] = schema.__fields__.keys()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(columns))
