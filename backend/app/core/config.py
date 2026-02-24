"""Application configuration."""
from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = "CD Locator"
    app_env: str = Field(default="dev", env="APP_ENV")
    app_secret_key: str = Field(default="change-me", env="APP_SECRET_KEY")
    db_url: str = Field(default="sqlite:///tmp.db", env="DB_URL")
    data_dir: str = "data"
    output_dir: str = "example/output"
    mask_sensitive: bool = Field(default=True, env="MASK_SENSITIVE")

    routing_backend: str = Field(default="haversine", env="ROUTING_BACKEND")
    osrm_url: str = Field(default="http://osrm:5000", env="OSRM_URL")
    graphhopper_url: str = Field(default="http://graphhopper:8989", env="GRAPHHOPPER_URL")
    solver_backend: str = Field(default="pulp", env="SOLVER_BACKEND")
    job_backend: str = Field(default="inline", env="JOB_BACKEND")

    enable_oidc: bool = Field(default=False, env="ENABLE_OIDC")
    required_scope_upload: str = Field(default="cd:write", env="REQUIRED_SCOPE_UPLOAD")
    required_scope_run: str = Field(default="cd:run", env="REQUIRED_SCOPE_RUN")

    audit_log_path: str = Field(default="data/audit/events.jsonl", env="AUDIT_LOG_PATH")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
