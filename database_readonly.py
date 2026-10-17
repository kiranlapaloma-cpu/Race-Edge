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
