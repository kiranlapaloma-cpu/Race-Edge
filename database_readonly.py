"""Read-only Supabase access derived from the original Race Edge database module."""
import os
import re
from functools import lru_cache
from supabase import create_client

COLUMNS = ('id,horse,finish_position,race_date,track,course,race_number,distance,'
           'rpss,race_test,official_mr,mr_achieved,sustain_residual,sustain_verdict,analyst_note')

def canon_horse(name: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(name).upper().strip())).strip()

@lru_cache(maxsize=1)
def client():
    url = os.environ.get('SUPABASE_URL', '').strip()
    key = os.environ.get('SUPABASE_KEY', '').strip()
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be set in Render environment')
    return create_client(url, key)

def _all_rows(columns=COLUMNS, order='race_date', desc=True):
    output = []
    offset = 0
    while True:
        batch = (client().table('horse_runs').select(columns)
                 .order(order, desc=desc).range(offset, offset + 999).execute().data or [])
        output.extend(batch)
        if len(batch) < 1000:
            return output
        offset += 1000

def horse_history(horse: str):
    target = canon_horse(horse)
    if not target:
        return []
    # Original module tries a direct canonical lookup then a paginated legacy fallback.
    # Paginate the direct match as well, avoiding truncation for horses with many runs.
    rows = []
    offset = 0
    while True:
        batch = (client().table('horse_runs').select(COLUMNS)
                 .eq('horse', target).order('race_date', desc=True)
                 .range(offset, offset + 999).execute().data or [])
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    if rows:
        return rows
    return [r for r in _all_rows() if canon_horse(r.get('horse', '')) == target]


def search_saved_horses(query: str, limit: int = 30):
    """Search distinct horse names saved in horse_runs (never live race cards)."""
    term = str(query or '').strip()
    if len(term) < 2:
        return []
    limit = max(1, min(int(limit), 50))
    # Canonical names are generally stored uppercase. Search is case insensitive,
    # and the canonical fallback handles punctuation differences in legacy rows.
    normalized = canon_horse(term)
    if not normalized:
        return []
    # Escape LIKE wildcards so input is treated as literal horse-name text.
    escaped = normalized.replace('\\', '\\\\').replace('%', r'\%').replace('_', r'\_')
    results = []
    seen = set()
    offset = 0
    while True:
        batch = (client().table('horse_runs').select('horse')
                 .ilike('horse', f'%{escaped}%')
                 .order('horse').range(offset, offset + 999).execute().data or [])
        for row in batch:
            name = str(row.get('horse') or '').strip()
            canonical = canon_horse(name)
            if canonical and canonical not in seen:
                seen.add(canonical)
                results.append(name)
                if len(results) >= limit:
                    return results
        if len(batch) < 1000:
            break
        offset += 1000
    return results

def race_date_runs(day: str):
    # All rows for the date, paginated to preserve the original database semantics.
    rows = []
    offset = 0
    while True:
        batch = (client().table('horse_runs').select(COLUMNS)
                 .eq('race_date', day).order('race_number')
                 .range(offset, offset + 999).execute().data or [])
        rows.extend(batch)
        if len(batch) < 1000:
            return rows
        offset += 1000


def _finish_place(value):
    """Parse a stored finish such as '3/11' for display ordering."""
    match = re.match(r'^\s*(\d+)\s*(?:/|$)', str(value or ''))
    return int(match.group(1)) if match else None


def historical_race_explorer(day: str, track: str, course: str, race_number: int,
                             subsequent_limit: int = 5):
    """Return a saved race field and later *recorded* runs for each runner.

    Read-only. A race is identified by date, track, course and race number.
    Subsequent runs are strictly later dates (not the same-day race).
    No claims are made about races absent from the Race Edge database.
    """
    from datetime import date

    day = date.fromisoformat(str(day)).isoformat()
    track = str(track or '').strip()
    course = str(course or '').strip()
    if not track or not course:
        raise ValueError('Track and course are required')
    if isinstance(race_number, bool) or int(race_number) < 1:
        raise ValueError('A positive race number is required')
    race_number = int(race_number)
    if not 1 <= int(subsequent_limit) <= 20:
        raise ValueError('subsequent_limit must be between 1 and 20')
    subsequent_limit = int(subsequent_limit)

    # Filter by the stable race identity, not just the date or horse name.
    field = []
    offset = 0
    while True:
        batch = (client().table('horse_runs').select(COLUMNS)
                 .eq('race_date', day)
                 .eq('track', track).eq('course', course)
                 .eq('race_number', race_number)
                 .range(offset, offset + 999).execute().data or [])
        field.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    field.sort(key=lambda row: (
        _finish_place(row.get('finish_position')) is None,
        _finish_place(row.get('finish_position')) or 9999,
        canon_horse(row.get('horse', '')),
    ))

    runners = []
    for row in field:
        history = horse_history(row.get('horse', ''))
        later = [run for run in history if str(run.get('race_date') or '') > day]
        later.sort(key=lambda run: (
            str(run.get('race_date') or ''),
            str(run.get('track') or ''),
            str(run.get('course') or ''),
            int(run.get('race_number') or 0),
        ))
        runners.append({
            'race_run': row,
            'subsequent_runs': later[:subsequent_limit],
            'subsequent_runs_recorded': len(later),
            'next_recorded_run': later[0] if later else None,
        })

    return {
        'race': {'race_date': day, 'track': track, 'course': course,
                 'race_number': race_number,
                 'distance': field[0].get('distance') if field else None},
        'runners': runners,
        'saved_runner_count': len(runners),
        'subsequent_limit': subsequent_limit,
        'coverage_note': ('Only races and subsequent runs saved in Race Edge are shown.'),
    }
