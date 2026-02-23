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

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
