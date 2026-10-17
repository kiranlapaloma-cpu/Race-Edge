"""Race Edge analysis API: read-only sectional calculations.

Mount from app.py using `app.include_router(analysis_router)`.
Does not modify prediction endpoints or write to Supabase.
"""
from __future__ import annotations

import math
import os
import secrets
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
import sys

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
from metrics_engine import build_metrics_and_shape

def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("RACE_EDGE_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="API access is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


analysis_router = APIRouter(prefix="/api/analysis", tags=["Race Analysis"], dependencies=[Depends(require_api_key)])

class AnalysisRequest(BaseModel):
    distance_m: float = Field(gt=0, le=3600)
    step: Literal[100, 200]
    rows: list[dict[str, Any]] = Field(min_length=2)
    use_cg: bool = True
    dampen_cg: bool = True
    use_race_shape: bool = True
    going_type: str = "Good"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@analysis_router.post("/calculate")
def calculate_analysis(payload: AnalysisRequest):
    frame = pd.DataFrame(payload.rows)
    if "Finish_Time" not in frame.columns:
        raise HTTPException(status_code=422, detail="Finish_Time column is required")
    if not any(str(c).endswith("_Time") and c != "Finish_Time" for c in frame.columns):
        raise HTTPException(status_code=422, detail="At least one sectional marker (_Time) is required")
    try:
        result, markers = build_metrics_and_shape(
            frame, D_actual_m=payload.distance_m, step=payload.step,
            use_cg=payload.use_cg, dampen_cg=payload.dampen_cg,
            use_race_shape=payload.use_race_shape, debug=False,
            going_type=payload.going_type,
        )
        return {
            "distance_m": payload.distance_m,
            "step": payload.step,
            "markers": [int(m) for m in markers],
            "race_diagnostics": _json_safe(dict(result.attrs)),
            "rows": _json_safe(result.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")),
            "saved": False,
        }
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=422, detail=f"Sectional calculation failed: {exc}") from exc
