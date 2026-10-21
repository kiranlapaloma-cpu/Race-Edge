"""Race Edge 2.0 — read-only horse telemetry and reusable comparison data.

Uses existing horse_runs.analyst_note. No prediction calculations, writes,
or average MR calculations. Sectional readings are descriptive indices.
"""
from __future__ import annotations

import math
import re
from statistics import median
from typing import Any

from database_readonly import horse_history, canon_horse

METRICS = ('F200', 'tsSPI', 'Accel', 'CG')
LABELS = {'F200': 'Opening speed', 'tsSPI': 'Mid-race speed',
          'Accel': 'Acceleration', 'CG': 'Corrected Grind'}
# Matches labelled values in saved analyst_note; avoids accidental numbers in prose.
_TOKEN = re.compile(
    r'(?<![A-Za-z0-9])(?P<label>F200|F|tsSPI|T|Accel|A|CG|Corrected\s+Grind)'
    r'\s*[:=]?\s*(?P<value>[+-]?\d+(?:\.\d+)?)\b', re.IGNORECASE)
_ALIAS = {'F200': 'F200', 'F': 'F200', 'TSPI': 'tsSPI', 'T': 'tsSPI',
          'ACCEL': 'Accel', 'A': 'Accel', 'CG': 'CG', 'CORRECTEDGRIND': 'CG'}


def extract_telemetry(note: Any) -> dict[str, float]:
    """Parse the saved metric line, e.g. 'F 100 | T 102 | A 103 | CG 98'.

    Only the first line is parsed: free-form analyst commentary must not be
    misinterpreted as saved metrics. Missing/invalid values stay missing.
    """
    first_line = str(note or '').splitlines()[0]
    result: dict[str, float] = {}
    for match in _TOKEN.finditer(first_line):
        key = _ALIAS[re.sub(r'\s+', '', match.group('label')).upper()]
        value = float(match.group('value'))
        if math.isfinite(value) and key not in result:
            result[key] = value
    return result


def _valid_mr(run: dict[str, Any]) -> float | None:
    try:
        value = float(run.get('mr_achieved'))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _summary(runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    readings = [(run, run['telemetry'][metric]) for run in runs if metric in run['telemetry']]
    values = [value for _, value in readings]
    latest = max(readings, key=lambda item: (str(item[0].get('race_date') or ''),
                                             int(item[0].get('race_number') or 0))) if readings else None
    above = sum(value > 101 for value in values)  # Strictly greater than 101.
    return {
        'metric': metric, 'label': LABELS[metric], 'readings': len(values),
        'runs_considered': len(runs), 'latest': latest[1] if latest else None,
        'latest_date': latest[0].get('race_date') if latest else None,
        'median': median(values) if values else None,
        'best': max(values) if values else None,
        'above_101': above,
        'above_101_denominator': len(values),
        'above_101_percent': round(100 * above / len(values), 1) if values else None,
    }


def _run_view(run: dict[str, Any]) -> dict[str, Any]:
    return {key: run.get(key) for key in (
        'horse', 'race_date', 'track', 'course', 'race_number', 'distance',
        'finish_position', 'mr_achieved', 'official_mr', 'rpss', 'sustain_residual') } | {
        'telemetry': extract_telemetry(run.get('analyst_note')),
    }


def horse_telemetry(horse: str, before_date: str | None = None) -> dict[str, Any]:
    """Summarise saved telemetry by horse and exact race distance.

    before_date is an exclusive cutoff for prospective use (prevents look-ahead).
    No averaging, no inferred race positions, no new composite scores.
    """
    target = canon_horse(horse)
    if not target:
        raise ValueError('Horse name required')
    raw = horse_history(horse)
    if before_date:
        from datetime import date
        cutoff = date.fromisoformat(before_date).isoformat()
        raw = [r for r in raw if str(r.get('race_date') or '') < cutoff]
    runs = [_run_view(r) for r in raw]
    runs.sort(key=lambda r: (str(r.get('race_date') or ''), int(r.get('race_number') or 0)), reverse=True)
    distances = sorted({int(r['distance']) for r in runs if r.get('distance') is not None})
    by_distance = {}
    for distance in distances:
        subset = [r for r in runs if r.get('distance') is not None and int(r['distance']) == distance]
        mr_values = [value for r in subset if (value := _valid_mr(r)) is not None]
        by_distance[str(distance)] = {
            'distance': distance, 'recorded_runs': len(subset),
            'mr_achieved': {'recorded': len(mr_values), 'peak': max(mr_values) if mr_values else None,
                            'median': median(mr_values) if mr_values else None,
                            'latest': next((_valid_mr(r) for r in subset if _valid_mr(r) is not None), None)},
            'metrics': {m: _summary(subset, m) for m in METRICS},
        }
    return {
        'horse': horse, 'recorded_runs': len(runs), 'before_date': before_date,
        'metrics': {m: _summary(runs, m) for m in METRICS},
        'by_distance': by_distance, 'runs': runs,
        'interpretation_note': ('Sectional indices describe measured performance phases, not literal '
                                'running position or proof of stamina. Above 101 means strictly >101. '
                                'Only saved runs and successfully parsed readings are included.'),
    }


def compare_horse_telemetry(horses: list[str], before_date: str | None = None) -> dict[str, Any]:
    """Shared data for future 2–4 horse comparison; no ranking or score."""
    if not 2 <= len(horses) <= 4:
        raise ValueError('Select between 2 and 4 horses')
    normalized = [canon_horse(h) for h in horses]
    if any(not h for h in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError('Select distinct, nonempty horse names')
    return {'horses': [horse_telemetry(h, before_date) for h in horses],
            'before_date': before_date,
            'note': 'Side-by-side evidence only; no new composite score or prediction ranking.'}
