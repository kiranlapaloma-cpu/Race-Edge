“””
data_io.py — file loading, header normalisation, split-step detection,
integrity scanning.  No Streamlit imports here.
“””
import re
import numpy as np
import pandas as pd

from utils import as_num

# ──────────────────────────────────────────────

# Header normalisation

# ──────────────────────────────────────────────

def normalize_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
“””
Map common column-name variants to canonical forms:
• ‘<n>_time’ / ‘<n>m_time’ / ‘<n>*split’  →  ‘<n>*Time’
• ‘finish_time’ / ‘finish_split’ / ‘finish’ →  ‘Finish_Time’
• ‘finish_pos’                               →  ‘Finish_Pos’
Returns (df_with_aliases, list_of_alias_notes).
“””
notes: list[str] = []
lmap = {
c.lower().strip().replace(” “, “*”).replace(”-”, “*”): c
for c in df.columns
}

```
def _alias(src_key: str, alias_col: str):
    if src_key in lmap and alias_col not in df.columns:
        df[alias_col] = df[lmap[src_key]]
        notes.append(f"Aliased `{lmap[src_key]}` → `{alias_col}`")

for k in ("finish_time", "finish_split", "finish"):
    _alias(k, "Finish_Time")
_alias("finish_pos", "Finish_Pos")

pat = re.compile(r"^(\d{2,4})m?_(time|split)$")
for lk, orig in lmap.items():
    m = pat.match(lk)
    if m:
        alias_col = f"{m.group(1)}_Time"
        if alias_col not in df.columns:
            df[alias_col] = df[orig]
            notes.append(f"Aliased `{orig}` → `{alias_col}`")

return df, notes
```

# ──────────────────────────────────────────────

# Split-step detection

# ──────────────────────────────────────────────

def detect_step(df: pd.DataFrame) -> int:
“”“Return 100 or 200 based on gaps between **Time marker columns.”””
markers = sorted(
{int(c.split(”*”)[0]) for c in df.columns
if c.endswith(”*Time”) and c != “Finish_Time”
and c.split(”*”)[0].isdigit()},
reverse=True,
)
if len(markers) < 2:
return 100
diffs = [markers[i] - markers[i + 1] for i in range(len(markers) - 1)]
cnt100 = sum(60 <= d <= 140 for d in diffs)
cnt200 = sum(160 <= d <= 240 for d in diffs)
return 200 if cnt200 > cnt100 else 100

# ──────────────────────────────────────────────

# Column helpers

# ──────────────────────────────────────────────

def collect_markers(df: pd.DataFrame) -> list[int]:
“”“Return sorted-descending list of numeric markers from *_Time columns.”””
marks = set()
for c in df.columns:
if c.endswith(”*Time”) and c != “Finish_Time”:
try:
marks.add(int(c.split(”*”)[0]))
except ValueError:
pass
return sorted(marks, reverse=True)

def expected_segments(distance_m: float, step: int) -> list[str]:
cols = [f”{m}_Time” for m in range(int(distance_m) - step, step - 1, -step)]
cols.append(“Finish_Time”)
return cols

# ──────────────────────────────────────────────

# Integrity scan

# ──────────────────────────────────────────────

def integrity_scan(
df: pd.DataFrame,
distance_m: float,
step: int,
) -> tuple[str, list[str], dict[str, int]]:
“””
Checks only columns that actually exist in df.
Returns (summary_text, missing_cols, {col: n_invalid}).
“””
exp = expected_segments(distance_m, step)
missing = [c for c in exp if c not in df.columns]

```
invalid: dict[str, int] = {}
for c in exp:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        invalid[c] = int(((s <= 0) | s.isna()).sum())

msgs: list[str] = []
if missing:
    msgs.append("Missing: " + ", ".join(missing))
bads = [f"{k} ({v} rows)" for k, v in invalid.items() if v > 0]
if bads:
    msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))

return (" • ".join(msgs) or "OK"), missing, invalid
```

# ──────────────────────────────────────────────

# File loader

# ──────────────────────────────────────────────

def load_file(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
“””
Read CSV or Excel upload, normalise headers.
Returns (work_df, alias_notes).
Raises on parse failure.
“””
if uploaded_file.name.lower().endswith(”.csv”):
raw = pd.read_csv(uploaded_file)
else:
raw = pd.read_excel(uploaded_file)

```
# Coerce obvious numeric columns
for c in list(raw.columns):
    lc = c.lower().strip()
    if any(lc.endswith(s) for s in ("_time", "_pos", "_split")) \
            or lc in ("race time", "finish_time", "finish_pos",
                       "horse weight", "weight allocated"):
        raw[c] = as_num(raw[c])

if "Finish_Pos" not in raw.columns:
    raw["Finish_Pos"] = np.arange(1, len(raw) + 1)

work, notes = normalize_headers(raw.copy())
return work, notes
```
