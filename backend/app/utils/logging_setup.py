"""Structured JSON logging utilities."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


SENSITIVE_KEYS = {"rent_monthly", "salary", "internal_cost"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "masked", "params"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def mask_payload(payload: Dict[str, Any], enabled: bool = True) -> Dict[str, Any]:
    if not enabled:
        return payload
    out = {}
    for k, v in payload.items():
        if k in SENSITIVE_KEYS and v is not None:
            out[k] = "***MASKED***"
        else:
            out[k] = v
    return out


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("cd_locator")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, message: str, run_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None, masked: bool = False) -> None:
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
        extra={"run_id": run_id, "params": params or {}, "masked": masked},
    )
