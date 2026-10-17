"""Race Edge 2.0 Phase 1: private, data-in/data-out prediction API.

No Supabase credentials are required or accepted. Historical rows are supplied
by the caller for this first integration milestone.
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
from race_prediction import build_race_predictions
from wfa import get_wfa_lb

app = FastAPI(title="Race Edge 2.0 — Python Engine", version="0.1.0")

class PredictionRequest(BaseModel):
    race_date: date
    distance_m: float = Field(gt=0, le=3600)
    runners: list[dict[str, Any]] = Field(min_length=1)
    histories: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    apply_apprentice_claims: bool = False

def clean(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [clean(r) for r in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if value is pd.NA or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value

@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "uploaded-current-race-prediction"}

@app.get("/api/wfa")
def wfa(race_date: date, distance_m: float, age: int):
    try:
        pounds = get_wfa_lb(race_date, distance_m, age)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"wfa_lb": pounds, "wfa_kg": pounds * 0.5, "mr_points": pounds}

@app.post("/api/predictions")
def predictions(payload: PredictionRequest):
    # Deliberately do not apply apprentice claims by default: worksheet rules.
    histories = {name.strip().upper(): pd.DataFrame(rows)
                 for name, rows in payload.histories.items()}
    def load_history(horse: str) -> pd.DataFrame:
        return histories.get(horse.strip().upper(), pd.DataFrame()).copy()
    try:
        result = build_race_predictions(
            pd.DataFrame(payload.runners),
            race_date=payload.race_date,
            distance_m=payload.distance_m,
            history_loader=load_history,
            apply_apprentice_claims=payload.apply_apprentice_claims,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output = clean(result)
    output.setdefault("apply_apprentice_claims", payload.apply_apprentice_claims)
    return output
