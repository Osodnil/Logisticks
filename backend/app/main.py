"""FastAPI entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.utils.logging_setup import setup_logging

app = FastAPI(title="CD Locator API", version="0.1.0")
app.include_router(router)
app.state.logger = setup_logging()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "cd-locator"}
