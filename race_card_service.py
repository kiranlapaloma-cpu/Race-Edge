"""Pure race-card mapping extracted from the current Streamlit race_card module."""
from __future__ import annotations
import math
import re
from typing import Any


def _number(value: Any, *, integer=False):
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if integer else number
    except (TypeError, ValueError):
        return None


def prediction_runners(card: dict) -> list[dict]:
    """Mirror the original _racecard_runner_frame prediction inputs.

    No database queries, Streamlit dependencies or calculation changes.
    Reserve and scratched runners remain visible but are excluded by engine.
    """
    rows = []
    for runner in card.get("runners", []):
        horse = str(runner.get("horseName") or "").strip()
        if not horse:
            continue
        status = str(runner.get("status") or "").strip().upper()
        scratched = bool(runner.get("scratched"))
        reserve = not scratched and (status == "R" or
            str(runner.get("jockeyName") or "").strip().lower().startswith("reserve"))
        claim = _number(runner.get("apprenticeClaim"))
        if claim is None:
            for key in ("jockeyName", "jockeyFull"):
                match = re.search(r"-\s*(\d+(?:\.\d+)?)\s*$", str(runner.get(key) or ""))
                if match:
                    claim = float(match.group(1))
                    break
        mr = _number(runner.get("MR"), integer=True)
        rows.append({
            "No.": _number(runner.get("saddleNo"), integer=True),
            "Horse": horse,
            "Draw": _number(runner.get("draw"), integer=True),
            "Age": _number(runner.get("age"), integer=True),
            "Sex": str(runner.get("sex") or "").strip().upper(),
            "Weight": _number(runner.get("weight")),
            "Claim": abs(claim or 0.0),
            "Official MR": mr if mr is not None and mr > 0 else None,
            "Jockey": str(runner.get("jockeyName") or ""),
            "Trainer": str(runner.get("trainerName") or ""),
            "Equipment": str(runner.get("equipment") or ""),
            "Status": "Scratched" if scratched else "Reserve" if reserve else "Runner",
        })
    return sorted(rows, key=lambda r: ({"Runner": 0, "Reserve": 1, "Scratched": 2}[r["Status"]], r["No."] if r["No."] is not None else 9999))
