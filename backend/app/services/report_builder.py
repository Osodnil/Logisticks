"""Build final reports from optimizer output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_reports(result: Dict[str, Any], out_dir: str = "example/output") -> Dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results_json = Path(out_dir) / "results.json"
    summary_csv = Path(out_dir) / "summary.csv"
    ppt_json = Path(out_dir) / "presentation_stub.json"

    results_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    alloc = pd.DataFrame(result.get("allocations", []))
    if alloc.empty:
        ranking = pd.DataFrame(columns=["site_id", "allocated_qty", "mean_unit_cost"])
    else:
        ranking = alloc.groupby("site_id").agg(allocated_qty=("allocated_qty", "sum"), mean_unit_cost=("unit_cost", "mean")).reset_index().sort_values("allocated_qty", ascending=False)
    ranking.to_csv(summary_csv, index=False)

    ppt_payload = {
        "slides": [
            {"title": "Resumo Executivo", "content": "CD Locator results"},
            {"title": "Sites Abertos", "content": result.get("open_sites", [])},
            {"title": "Alocações", "content": result.get("allocations", [])[:10]},
            {"title": "Custos", "content": result.get("objective_breakdown", {})},
            {"title": "Riscos", "content": "flood/crime"},
            {"title": "Próximos Passos", "content": "refinar com OSRM e PostGIS"},
        ]
    }
    ppt_json.write_text(json.dumps(ppt_payload, indent=2), encoding="utf-8")
    return {"results_json": str(results_json), "summary_csv": str(summary_csv), "presentation_stub": str(ppt_json)}
