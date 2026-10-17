"""SAHorseracing public race-card feed integration for Race Edge.

Observed fields endpoint:
    https://www.sahorseracing.co.za/sahr-php/public.php
        ?feed=fields&date=YYYYMMDD&club=CLUB_ID

Meeting discovery uses the companion public fixtures feed when available:
    ?feed=fixtures&country=ALL

The endpoint can return either a JSON object or a JSON-encoded string containing
that object, so decoding handles both forms.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

import requests

SAHR_FIELDS_URL = "https://www.sahorseracing.co.za/sahr-php/public.php"
SAHR_PUBLIC_PAGE = "https://www.sahorseracing.co.za/sahr/public.html"


class SAHRError(RuntimeError):
    """Raised when the SAHorseracing feed cannot be loaded or parsed."""


def _sahr_is_scratched(runner: dict[str, Any]) -> bool:
    """Recognise scratchings without confusing reserves (R) with non-runners."""
    status = str(runner.get("status") or "").strip().upper()
    if status in {"L", "S", "SCR", "SCRATCH", "SCRATCHED", "SCRATCHING", "NON-RUNNER", "NON RUNNER", "NR"}:
        return True
    for key in ("scratched", "isScratched", "is_scratched", "scratch"):
        value = runner.get(key)
        if isinstance(value, bool) and value:
            return True
        if str(value or "").strip().upper() in {"TRUE", "YES", "Y", "1", "S", "SCR", "SCRATCHED"}:
            return True
    return False


def _apprentice_claim_kg(runner: dict[str, Any]) -> float:
    """Return the apprentice claim in kg from the SAHR runner payload.

    SAHR commonly appends the claim to the jockey text, e.g.
    "*Divesh Ramkhalawonn -2.5".  Explicit claim fields are also accepted
    so this remains compatible if the feed exposes one directly later.
    """
    for key in ("apprenticeClaim", "jockeyClaim", "claim"):
        raw = runner.get(key)
        if raw not in (None, ""):
            try:
                claim = abs(float(raw))
                return claim if claim > 0 else 0.0
            except Exception:
                pass

    for key in ("jockey", "jockeyFull"):
        text = str(runner.get(key) or "").strip()
        match = re.search(r"-\s*(\d+(?:\.\d+)?)\s*$", text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                pass
    return 0.0


def _date_key(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value or "").strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return digits
    raise ValueError("Race date must resolve to YYYYMMDD.")


def _decode_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload: Any = response.json()
    except Exception:
        text = response.text.strip()
        try:
            payload = json.loads(text)
        except Exception as exc:
            preview = " ".join(text[:220].split())
            raise SAHRError(
                "SAHorseracing did not return JSON. "
                f"Response started with: {preview or '<empty response>'}"
            ) from exc

    # The observed feed is sometimes a JSON string whose contents are another
    # JSON object: "{\"date\":...}".
    for _ in range(2):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception as exc:
                raise SAHRError("SAHorseracing returned an invalid encoded JSON payload.") from exc
        else:
            break

    if not isinstance(payload, dict):
        raise SAHRError("SAHorseracing returned an unexpected payload type.")
    return payload


def get_fields_meeting(
    race_date: date | datetime | str,
    club: int,
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch the complete meeting (all races and runners) for a date + club."""
    params = {
        "feed": "fields",
        "date": _date_key(race_date),
        "club": int(club),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Race Edge private form study)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": SAHR_PUBLIC_PAGE,
    }
    try:
        response = requests.get(
            SAHR_FIELDS_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SAHRError(f"Could not connect to SAHorseracing: {exc}") from exc

    payload = _decode_payload(response)
    races = payload.get("races")
    if not isinstance(races, dict) or not races:
        heading = str(payload.get("heading") or "").strip()
        msg = "No races were returned for that date and club."
        if heading:
            msg += f" Meeting response: {heading}"
        raise SAHRError(msg)
    return payload



def _decode_any_payload(response: requests.Response) -> Any:
    """Decode SAHR responses that may be plain JSON or JSON encoded as a string."""
    try:
        payload: Any = response.json()
    except Exception:
        text = response.text.strip()
        try:
            payload = json.loads(text)
        except Exception as exc:
            preview = " ".join(text[:220].split())
            raise SAHRError(
                "SAHorseracing did not return JSON. "
                f"Response started with: {preview or '<empty response>'}"
            ) from exc
    for _ in range(2):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception as exc:
                raise SAHRError("SAHorseracing returned invalid encoded JSON.") from exc
        else:
            break
    return payload


def _request_public(params: dict[str, Any], *, timeout: float = 20.0) -> Any:
    headers = {
        "User-Agent": "Mozilla/5.0 (Race Edge private form study)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": SAHR_PUBLIC_PAGE,
    }
    try:
        response = requests.get(
            SAHR_FIELDS_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SAHRError(f"Could not connect to SAHorseracing: {exc}") from exc
    return _decode_any_payload(response)


def _fixture_records(payload: Any) -> list[dict[str, Any]]:
    """Normalise likely fixtures-feed shapes into a flat list of meeting records."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    # Common API wrapper keys.
    for key in ("fixtures", "meetings", "results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            rows = [x for x in value.values() if isinstance(x, dict)]
            if rows:
                return rows

    # Some feeds use numeric/object keys directly.
    rows = [x for x in payload.values() if isinstance(x, dict)]
    return rows


def _fixture_date(record: dict[str, Any]) -> str:
    for key in ("date", "raceDate", "race_date", "meetingDate", "meeting_date"):
        value = record.get(key)
        if value in (None, ""):
            continue
        digits = re.sub(r"\D", "", str(value))
        if len(digits) >= 8:
            return digits[:8]
    return ""


def _fixture_club(record: dict[str, Any]) -> int | None:
    for key in ("club", "clubId", "club_id", "venueId", "venue_id"):
        value = record.get(key)
        try:
            return int(float(value))
        except Exception:
            pass
    return None


def _fixture_name(record: dict[str, Any]) -> str:
    for key in ("clubName", "club_name", "meeting", "meetingName", "meeting_name", "venue", "venueName", "venue_name", "name"):
        text = str(record.get(key) or "").strip()
        if text:
            return text
    return ""


def get_meetings_for_date(
    race_date: date | datetime | str,
    *,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Discover available SA meetings for a selected date.

    The SAHR public interface shares the fixtures-style feed naming used by the
    public racing pages.  We normalise the response defensively because the
    payload shape can vary.  Each returned item has date, club and name.
    """
    target = _date_key(race_date)
    payload = _request_public(
        {"feed": "fixtures", "country": "ALL"},
        timeout=timeout,
    )
    records = _fixture_records(payload)
    meetings: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        d = _fixture_date(record)
        club = _fixture_club(record)
        if d != target or club is None:
            continue
        name = _fixture_name(record) or f"Club {club}"
        key = (d, club)
        if key in seen:
            continue
        seen.add(key)
        meetings.append({
            "date": d,
            "club": club,
            "name": name,
            "raw": record,
        })

    meetings.sort(key=lambda x: (str(x.get("name") or ""), int(x.get("club") or 0)))
    return meetings


def meeting_display_label(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip() or f"Club {item.get('club')}"
    return name

def meeting_race_options(meeting: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(race_key, display_label), ...] in race-number order."""
    races = meeting.get("races") or {}
    options: list[tuple[str, str]] = []
    for key, race in races.items():
        if not isinstance(race, dict):
            continue
        summary = str(race.get("summary") or "").strip()
        label = summary or f"Race {str(key).lstrip('0') or key}"
        options.append((str(key), label))

    def sort_key(item: tuple[str, str]):
        try:
            return int(item[0])
        except Exception:
            return 999

    return sorted(options, key=sort_key)


def _summary_meta(summary: str) -> dict[str, Any]:
    """Extract surface, stake, distance and time from a SAHR race summary."""
    text = str(summary or "").strip()
    out: dict[str, Any] = {}

    m_surface = re.search(r"\(([^)]+)\)", text)
    if m_surface:
        out["surfaceDescr"] = m_surface.group(1).strip()

    m_stake = re.search(r"\bR\s*([\d\s,]+)\s+(\d{3,4})m\s+(\d{1,2}:\d{2})\b", text, re.I)
    if m_stake:
        out["currency"] = "R"
        out["stake"] = re.sub(r"\s+", "", m_stake.group(1)).replace(",", "")
        out["distance"] = int(m_stake.group(2))
        out["time"] = m_stake.group(3)
    else:
        m_dist = re.search(r"\b(\d{3,4})m\b", text, re.I)
        if m_dist:
            out["distance"] = int(m_dist.group(1))
        m_time = re.search(r"\b(\d{1,2}:\d{2})\b", text)
        if m_time:
            out["time"] = m_time.group(1)
    return out


def _format_date(date_key: str) -> str:
    try:
        return datetime.strptime(str(date_key), "%Y%m%d").strftime("%d %B %Y")
    except Exception:
        return str(date_key or "")


def _meeting_name(heading: str) -> str:
    text = str(heading or "").strip()
    # e.g. "Hollywoodbets Greyville Polytrack - Tuesday 18 August 2026"
    if " - " in text:
        return text.split(" - ", 1)[0].strip()
    return text


def race_to_race_edge_card(meeting: dict[str, Any], race_key: str) -> dict[str, Any]:
    """Convert one SAHR race into the existing Race Edge race-card schema."""
    races = meeting.get("races") or {}
    race = races.get(str(race_key))
    if not isinstance(race, dict):
        # Tolerate numeric race keys supplied without zero padding.
        try:
            race = races.get(f"{int(race_key):02d}")
        except Exception:
            race = None
    if not isinstance(race, dict):
        raise SAHRError(f"Race {race_key} was not found in the loaded meeting.")

    try:
        race_no = int(str(race_key))
    except Exception:
        race_no = None

    meta = _summary_meta(race.get("summary"))
    date_key = str(meeting.get("date") or "")
    heading = str(meeting.get("heading") or "")

    card: dict[str, Any] = {
        "date": date_key,
        "dateFormat": _format_date(date_key),
        "club": meeting.get("club"),
        "itw": meeting.get("itw"),
        "clubName": _meeting_name(heading),
        "race": race_no,
        "event": race.get("ref"),
        "name": race.get("name") or "",
        "description": race.get("description") or "",
        "WFA": race.get("WFA") or "",
        "currency": race.get("currency") or meta.get("currency") or "R",
        "stake": meta.get("stake") or "",
        "distance": meta.get("distance"),
        "time": meta.get("time") or "",
        "surfaceDescr": meta.get("surfaceDescr") or "",
        "meetingStatus": meeting.get("meetingStatus"),
        "source": "SAHorseracing",
        "sourceFooter": meeting.get("footer") or "",
        "runners": [],
    }

    for r in race.get("runners") or []:
        if not isinstance(r, dict):
            continue
        apprentice_claim = _apprentice_claim_kg(r)
        card["runners"].append({
            "saddleNo": r.get("sno"),
            "horseSeq": r.get("seq"),
            "horseName": r.get("horse") or "",
            "status": r.get("status") or "F",
            "scratched": _sahr_is_scratched(r),
            "draw": r.get("draw"),
            "officialDraw": r.get("odraw"),
            "age": r.get("age"),
            "colour": r.get("colour"),
            "sex": r.get("sex"),
            "weight": r.get("weight"),
            "MR": r.get("MR"),
            "jockeyName": r.get("jockey") or "",
            "jockeyFull": r.get("jockeyFull") or "",
            "apprenticeClaim": apprentice_claim,
            "trainerName": r.get("trainer") or "",
            "trainerEst": r.get("trainerEst") or "",
            "owner": r.get("ownerName") or "",
            "equipment": "".join([
                "A" if str(r.get("alumites") or "").strip() else "",
                "B" if str(r.get("blinkers") or "").strip() else "",
                "T" if str(r.get("tongueties") or "").strip() else "",
                "E" if str(r.get("earmuffs") or "").strip() else "",
            ]),
            "runs": r.get("runs"),
            "wins": r.get("wins"),
            "places": r.get("places"),
            "stakes": r.get("stakes"),
            "trainerComments": r.get("tcmts") or "",
            # Not provided by this feed; keep existing Race Edge schema stable.
            "horseWeight": "",
            "horseWeightDelta": None,
            "restDays": None,
            "odds": "",
            "openBet": "",
        })

    return card
