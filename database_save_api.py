"""Race Edge 2.0: authenticated, atomic whole-race Supabase RPC saving.

Mount in app.py with:
    from database_save_api import database_save_router
    app.include_router(database_save_router, dependencies=[Depends(require_api_key)])

Requires Render environment variables SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""
from __future__ import annotations

import math
import os
import re
from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from supabase import create_client


database_save_router = APIRouter(prefix="/api/db", tags=["Database Saving"])


class RaceRun(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    horse: str = Field(min_length=1, max_length=250)
    finish_position: str = Field(min_length=1, max_length=40)
    race_date: date
    track: str = Field(min_length=1, max_length=120)
    course: str = Field(min_length=1, max_length=120)
    race_number: int = Field(ge=1, le=30)
    distance: int = Field(ge=400, le=4000)
    rpss: float | None = None
    race_test: str | None = None
    official_mr: int | None = Field(default=None, ge=0, le=250)
    mr_achieved: float | None = None
    sustain_residual: float | None = None
    sustain_verdict: str | None = None
    analyst_note: str | None = None

    @field_validator("rpss", "mr_achieved", "sustain_residual")
    @classmethod
    def finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Numeric values must be finite")
        return value

    @field_validator("race_test", "sustain_verdict", "analyst_note", mode="before")
    @classmethod
    def empty_optional_text(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("analyst_note")
    @classmethod
    def validate_note(cls, value: str | None) -> str | None:
        if value is None:
            return None  # Preserve existing note on a revised upload.
        # Notes may contain further analyst commentary, but must identify all four metrics.
        for label in ("F", "T", "A", "CG"):
            if not re.search(rf"(?<!\w){label}\s*[:=]?\s*[-+]?\d+(?:\.\d+)?\b", value, re.I):
                raise ValueError(f"Analyst Note must include a numeric {label} metric")
        return value


class SaveRaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: bool = Field(description="Explicit confirmation from the verification page")
    runners: list[RaceRun] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_race(self):
        if not self.confirm:
            raise ValueError("Explicit confirmation is required to save or update a race")
        race_keys = {
            (r.race_date, r.track, r.course, r.race_number, r.distance)
            for r in self.runners
        }
        if len(race_keys) != 1:
            raise ValueError("All runners must belong to the same race")
        horses = [r.horse for r in self.runners]
        if len(horses) != len(set(horses)):
            raise ValueError("The race contains duplicate horse names")
        return self


@lru_cache(maxsize=1)
def _write_client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Supabase database saving is not configured")
    return create_client(url, key)


@database_save_router.post("/save-race")
def save_race(payload: SaveRaceRequest):
    """One Supabase RPC call: PostgreSQL handles the atomic upsert."""
    try:
        rows = [r.model_dump(mode="json") for r in payload.runners]
        response = _write_client().rpc(
            "save_race_edge_race", {"p_rows": rows}
        ).execute()
        result = response.data
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("Database did not confirm the race save")
        return result
    except RuntimeError as exc:
        if "not configured" in str(exc):
            raise HTTPException(status_code=503, detail="Database saving is not configured") from exc
        raise HTTPException(status_code=502, detail="Database save could not be confirmed") from exc
    except Exception as exc:
        # Never leak database credentials, SQL, or internal errors to the browser.
        raise HTTPException(status_code=502, detail="Database save failed; check Render logs") from exc
