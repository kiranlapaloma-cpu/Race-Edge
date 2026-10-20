"""Race Edge 2.0 Phase 1: private, data-in/data-out prediction API.

No Supabase credentials are required or accepted. Historical rows are supplied
by the caller for this first integration milestone.
"""
from __future__ import annotations

import math
import os
import secrets
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
from race_prediction import build_race_predictions
from wfa import get_wfa_lb
from database_readonly import horse_history, race_date_runs, historical_race_explorer, search_saved_horses
from sahr import (SAHRError, get_meetings_for_date, get_fields_meeting,
                  meeting_race_options, race_to_race_edge_card)
from race_card_service import prediction_runners

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


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("RACE_EDGE_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="API access is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/api/db/horse-search", dependencies=[Depends(require_api_key)])
def db_horse_search(q: str = "", limit: int = 30):
    """Search horse names saved in Race Edge, independent of live race cards."""
    if len(q.strip()) < 2:
        return {"query": q, "horses": []}
    if not 1 <= limit <= 50:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 50")
    try:
        return {"query": q, "horses": search_saved_horses(q, limit=limit)}
    except Exception:
        raise HTTPException(status_code=502, detail="Horse search failed; check server logs")

@app.get("/api/db/horse/{horse}", dependencies=[Depends(require_api_key)])
def db_horse(horse: str):
    try:
        return {"horse": horse, "runs": horse_history(horse)}
    except Exception:
        raise HTTPException(status_code=502, detail="Database query failed; check server logs")

@app.get("/api/db/date/{race_date}", dependencies=[Depends(require_api_key)])
def db_date(race_date: date):
    try:
        return {"race_date": race_date.isoformat(), "runs": race_date_runs(race_date.isoformat())}
    except Exception:
        raise HTTPException(status_code=502, detail="Database query failed; check server logs")

@app.get("/api/db/historical-race", dependencies=[Depends(require_api_key)])
def db_historical_race(
    race_date: date,
    track: str,
    course: str,
    race_number: int,
    subsequent_limit: int = 5,
):
    """Full saved field and subsequent Race Edge runs, read-only."""
    if not track.strip() or not course.strip():
        raise HTTPException(status_code=422, detail="Track and course are required")
    if race_number < 1 or not 1 <= subsequent_limit <= 20:
        raise HTTPException(status_code=422, detail="Invalid race number or subsequent limit")
    try:
        return clean(historical_race_explorer(
            day=race_date.isoformat(),
            track=track,
            course=course,
            race_number=race_number,
            subsequent_limit=subsequent_limit,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=502, detail="Historical race query failed; check server logs")

class DatabasePredictionRequest(BaseModel):
    race_date: date
    distance_m: float = Field(gt=0, le=3600)
    runners: list[dict[str, Any]] = Field(min_length=1)
    # Claims are never applied in the prediction worksheet.

@app.post("/api/db/predictions", dependencies=[Depends(require_api_key)])
def db_predictions(payload: DatabasePredictionRequest):
    histories = {}
    try:
        for runner in payload.runners:
            horse = str(runner.get("Horse", "")).strip()
            if horse:
                histories[horse] = pd.DataFrame(horse_history(horse))
        result = build_race_predictions(
            pd.DataFrame(payload.runners), race_date=payload.race_date,
            distance_m=payload.distance_m,
            history_loader=lambda horse: histories.get(horse, pd.DataFrame()).copy(),
            apply_apprentice_claims=False,
        )
        return clean(result)
    except Exception:
        raise HTTPException(status_code=502, detail="Prediction or database query failed; check server logs")


# Race-card endpoints: all protected, because these are part of a private tool.
@app.get("/api/cards/meetings", dependencies=[Depends(require_api_key)])
def card_meetings(race_date: date):
    try:
        meetings = get_meetings_for_date(race_date)
        return {"date": race_date.isoformat(), "meetings": [
            {"date": item["date"], "club": item["club"], "name": item["name"]}
            for item in meetings]}
    except SAHRError as exc:
        raise HTTPException(status_code=502, detail=f"Race-card feed unavailable: {exc}") from exc


def _load_card(race_date: date, club: int, race: int):
    try:
        meeting = get_fields_meeting(race_date, club)
        card = race_to_race_edge_card(meeting, str(race))
        if not card.get("distance"):
            raise HTTPException(status_code=422, detail="Race distance is missing from the feed")
        return card
    except SAHRError as exc:
        raise HTTPException(status_code=502, detail=f"Race-card feed unavailable: {exc}") from exc


@app.get("/api/cards/races", dependencies=[Depends(require_api_key)])
def card_races(race_date: date, club: int):
    try:
        meeting = get_fields_meeting(race_date, club)
        return {"date": race_date.isoformat(), "club": club, "races": [
            {"key": key, "label": label}
            for key, label in meeting_race_options(meeting)]}
    except SAHRError as exc:
        raise HTTPException(status_code=502, detail=f"Race-card feed unavailable: {exc}") from exc


@app.get("/api/cards/race", dependencies=[Depends(require_api_key)])
def card_race(race_date: date, club: int, race: int):
    card = _load_card(race_date, club, race)
    return {"card": card, "prediction_runners": prediction_runners(card)}


@app.get("/api/cards/predictions", dependencies=[Depends(require_api_key)])
def card_predictions(race_date: date, club: int, race: int, include_peak: bool = False):
    """Automatically load the card and histories, then call original maths.

    Peak is calculated by the original engine but omitted from the default
    API response; this does not alter any projection or ranking.
    """
    card = _load_card(race_date, club, race)
    runners = prediction_runners(card)
    if not runners:
        raise HTTPException(status_code=422, detail="No runners returned by race-card feed")
    histories = {}
    try:
        for runner in runners:
            if runner["Status"] != "Runner":
                continue
            horse = runner["Horse"]
            # Exclude races on/after target date to prevent look-ahead leakage.
            rows = horse_history(horse)
            histories[horse] = pd.DataFrame([
                row for row in rows if str(row.get("race_date") or "")[:10] < race_date.isoformat()
            ])
        result = build_race_predictions(
            pd.DataFrame(runners), race_date=race_date,
            distance_m=float(card["distance"]),
            history_loader=lambda horse: histories.get(horse, pd.DataFrame()).copy(),
            apply_apprentice_claims=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Historical data or prediction calculation failed; check server logs") from exc
    response = clean(result)
    if not include_peak:
        response.get("scenarios", {}).pop("Peak Ability", None)
    return {
        "meeting": {"date": race_date.isoformat(), "club": club, "race": race},
        "card": card,
        "prediction_runners": runners,
        "prediction": response,
        "visible_scenarios": ["Recent Best", "Latest Form"] + (["Peak Ability"] if include_peak else []),
        "apprentice_claims_applied": False,
    }

from race_analysis_api import analysis_router
app.include_router(analysis_router)

from race_plane_api import race_plane_router
from database_save_api import database_save_router
app.include_router(database_save_router, dependencies=[Depends(require_api_key)])
app.include_router(race_plane_router)
