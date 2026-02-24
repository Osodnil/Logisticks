"""Persistent state store for projects and runs (SQLite baseline)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_settings


class StateStore:
    """Small persistence layer to prepare migration from in-memory to DB-backed state."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.db_url = db_url or settings.db_url
        self.sqlite_path = self._to_sqlite_path(self.db_url)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @staticmethod
    def _to_sqlite_path(db_url: str) -> str:
        if db_url.startswith("sqlite:///"):
            return db_url.replace("sqlite:///", "", 1)
        # Placeholder for postgres URL support in future iterations.
        return "data/state.db"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.sqlite_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    files_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS od_cache (
                    route_key TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    distance_km REAL NOT NULL,
                    time_h REAL NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_project(self, project_id: str, files: Dict[str, str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects(project_id, files_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(project_id) DO UPDATE SET
                    files_json=excluded.files_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (project_id, json.dumps(files)),
            )

    def get_project(self, project_id: str) -> Dict[str, str]:
        with self._connect() as conn:
            row = conn.execute("SELECT files_json FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if not row:
            return {}
        return json.loads(row[0])

    def save_run(self, run_id: str, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, payload_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (run_id, json.dumps(payload)),
            )

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return {}
        return json.loads(row[0])


    def save_od_cache(self, route_key: str, backend: str, distance_km: float, time_h: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO od_cache(route_key, backend, distance_km, time_h, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(route_key) DO UPDATE SET
                    backend=excluded.backend,
                    distance_km=excluded.distance_km,
                    time_h=excluded.time_h,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (route_key, backend, distance_km, time_h),
            )

    def get_od_cache(self, route_key: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT backend, distance_km, time_h FROM od_cache WHERE route_key=?",
                (route_key,),
            ).fetchone()
        if not row:
            return {}
        return {"backend": row[0], "distance_km": float(row[1]), "time_h": float(row[2])}
