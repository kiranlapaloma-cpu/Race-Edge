“””
db.py - SQLite persistence for Race Edge.

Provides:

- init_db(path)          - create/verify schema
- save_race(conn, …)   - insert race + all performances
- query_horse(conn, name) - retrieve horse history
  “””
  import sqlite3
  from datetime import datetime

import numpy as np
import pandas as pd

from utils import canon_horse, sha1

# –––––––––––––––––––––––

# Schema

# –––––––––––––––––––––––

_SCHEMA = “””
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS races(
race_id        TEXT PRIMARY KEY,
date           TEXT,
track          TEXT,
race_no        INTEGER,
distance_m     INTEGER NOT NULL,
split_step     INTEGER CHECK(split_step IN (100,200)) NOT NULL,
fsr            REAL,
collapse       REAL,
shape_tag      TEXT,
sci            REAL,
fra_applied    INTEGER,
going          TEXT,
app_version    TEXT,
created_ts     TEXT DEFAULT (datetime(‘now’)),
src_hash       TEXT
);

CREATE TABLE IF NOT EXISTS performances(
perf_id         TEXT PRIMARY KEY,
race_id         TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
horse           TEXT NOT NULL,
horse_canon     TEXT NOT NULL,
finish_pos      INTEGER,
race_time_s     REAL,
f200_idx        REAL,
tsspi           REAL,
accel           REAL,
grind           REAL,
grind_cg        REAL,
delta_g         REAL,
finisher_factor REAL,
grind_adj_pts   REAL,
pi              REAL,
pi_rs           REAL,
gci             REAL,
gci_rs          REAL,
hidden          REAL,
ability         REAL,
ability_tier    TEXT,
iai             REAL,
bal             REAL,
comp            REAL,
iai_pct         REAL,
hid_pct         REAL,
bal_pct         REAL,
comp_pct        REAL,
dir_hint        TEXT,
confidence      TEXT,
winning_dna     REAL,
xwin            REAL,
pwr400          REAL,
fatigue_score   REAL,
car             REAL,
inserted_ts     TEXT DEFAULT (datetime(‘now’))
);

CREATE INDEX IF NOT EXISTS idx_perf_horse ON performances(horse_canon);
CREATE INDEX IF NOT EXISTS idx_perf_race  ON performances(race_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
“””

def init_db(path: str) -> tuple[bool, str]:
“””
Create or verify the database.
Returns (success, message).
“””
try:
conn = sqlite3.connect(path)
conn.executescript(_SCHEMA)
conn.commit()
conn.close()
return True, f”DB ready at {path}”
except Exception as e:
return False, f”DB init failed: {e}”

# –––––––––––––––––––––––

# Helpers

# –––––––––––––––––––––––

def _f(x) -> float | None:
“”“Float or None (for DB nullability).”””
try:
v = float(x)
return v if np.isfinite(v) else None
except Exception:
return None

def _i(x) -> int | None:
try:
v = int(x)
return v
except Exception:
return None

# –––––––––––––––––––––––

# Save race

# –––––––––––––––––––––––

def save_race(
db_path: str,
metrics: pd.DataFrame,
distance_m: float,
split_step: int,
date: str = “”,
track: str = “”,
race_no: int | None = None,
app_version: str = “3.2”,
src_hash: str = “”,
extra_cols: dict | None = None,      # {horse_name: {col: value}} for any extra per-horse cols
) -> tuple[bool, str]:
“””
Upsert one race + all performances into the DB.
Returns (success, message).

```
extra_cols lets callers pass in WinningDNA, xWin, PWR400, FatigueScore, CAR
keyed by horse name.
"""
attrs = metrics.attrs
race_id = sha1(f"{src_hash}{distance_m}{split_step}{date}{track}{race_no}")

race_row = (
    race_id,
    date or datetime.now().strftime("%Y-%m-%d"),
    track or "",
    _i(race_no),
    _i(distance_m),
    _i(split_step),
    _f(attrs.get("FSR")),
    _f(attrs.get("CollapseSeverity")),
    str(attrs.get("SHAPE_TAG", "")),
    _f(attrs.get("SCI")),
    _i(attrs.get("FRA_APPLIED", 0)),
    str(attrs.get("GOING", "Good")),
    app_version,
    datetime.now().isoformat(),
    src_hash,
)

perf_rows = []
for _, r in metrics.iterrows():
    horse = str(r.get("Horse", ""))
    horse_canon_ = canon_horse(horse)
    perf_id = sha1(f"{race_id}{horse_canon_}")

    # optional extra columns (WinningDNA, xWin, etc.)
    ex = (extra_cols or {}).get(horse, {})

    perf_rows.append((
        perf_id, race_id,
        horse, horse_canon_,
        _i(r.get("Finish_Pos")),
        _f(r.get("RaceTime_s")),
        _f(r.get("F200_idx")),
        _f(r.get("tsSPI")),
        _f(r.get("Accel")),
        _f(r.get("Grind")),
        _f(r.get("Grind_CG")),
        _f(r.get("DeltaG")),
        _f(r.get("FinisherFactor")),
        _f(r.get("GrindAdjPts")),
        _f(r.get("PI")),
        _f(r.get("PI_RS")),
        _f(r.get("GCI")),
        _f(r.get("GCI_RS")),
        _f(r.get("HiddenScore")),
        None, None,                  # ability / ability_tier (not computed here)
        None, None, None,            # iai / bal / comp
        None, None, None, None,      # pct cols
        None, None,                  # dir_hint / confidence
        _f(ex.get("WinningDNA")),
        _f(ex.get("xWin")),
        _f(ex.get("PWR400")),
        _f(ex.get("FatigueScore")),
        _f(ex.get("CAR")),
        datetime.now().isoformat(),
    ))

try:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute("""
        INSERT OR REPLACE INTO races
        (race_id,date,track,race_no,distance_m,split_step,
         fsr,collapse,shape_tag,sci,fra_applied,going,
         app_version,created_ts,src_hash)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, race_row)

    conn.executemany("""
        INSERT OR REPLACE INTO performances
        (perf_id,race_id,horse,horse_canon,finish_pos,race_time_s,
         f200_idx,tsspi,accel,grind,grind_cg,delta_g,finisher_factor,
         grind_adj_pts,pi,pi_rs,gci,gci_rs,hidden,ability,ability_tier,
         iai,bal,comp,iai_pct,hid_pct,bal_pct,comp_pct,
         dir_hint,confidence,winning_dna,xwin,pwr400,fatigue_score,car,
         inserted_ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, perf_rows)

    conn.commit()
    conn.close()
    return True, f"Saved race {race_id[:8]}… with {len(perf_rows)} performances."
except Exception as e:
    return False, f"DB write failed: {e}"
```

# –––––––––––––––––––––––

# Query helpers

# –––––––––––––––––––––––

def query_horse(db_path: str, name: str) -> pd.DataFrame:
“”“Return all performance rows for a horse (fuzzy canonical match).”””
canon = canon_horse(name)
try:
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(
“””
SELECT p.*, r.date, r.track, r.distance_m, r.going, r.shape_tag
FROM performances p
JOIN races r ON p.race_id = r.race_id
WHERE p.horse_canon LIKE ?
ORDER BY r.date DESC
“””,
conn,
params=(f”%{canon}%”,),
)
conn.close()
return df
except Exception:
return pd.DataFrame()

def query_recent_races(db_path: str, limit: int = 20) -> pd.DataFrame:
“”“Return recent race summaries.”””
try:
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(
“SELECT * FROM races ORDER BY date DESC LIMIT ?”,
conn, params=(limit,),
)
conn.close()
return df
except Exception:
return pd.DataFrame()
