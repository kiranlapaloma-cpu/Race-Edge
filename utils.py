“””
utils.py - shared helpers for Race Edge.
All functions here are pure (no Streamlit, no side effects).
“””
import math
import re
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# –––––––––––––––––––––––

# Numeric helpers

# –––––––––––––––––––––––

def as_num(x):
“”“Coerce to numeric Series / scalar with NaN on failure.”””
return pd.to_numeric(x, errors=“coerce”)

def safe_num(x, default=0.0):
“”“Return a finite float; fall back to default.”””
try:
v = float(x)
return v if np.isfinite(v) else float(default)
except Exception:
return float(default)

def clamp(v, lo, hi):
return max(lo, min(hi, float(v)))

def mad_std(x):
“”“Robust sigma via MAD (1.4826 _ median |x_ _ median(x)|).”””
x = np.asarray(x, dtype=float)
x = x[np.isfinite(x)]
if x.size == 0:
return np.nan
med = np.median(x)
return 1.4826 * np.median(np.abs(x - med))

def winsorize(s, p_lo=0.10, p_hi=0.90):
try:
lo = s.quantile(p_lo)
hi = s.quantile(p_hi)
return s.clip(lower=lo, upper=hi)
except Exception:
return s

def lerp(a, b, t):
return a + (b - a) * float(t)

def pct_at_or_above(s, thr):
s = pd.to_numeric(s, errors=“coerce”).dropna()
return 0.0 if s.empty else float((s >= thr).mean())

# –––––––––––––––––––––––

# String / identity helpers

# –––––––––––––––––––––––

def sha1(s: str) -> str:
return hashlib.sha1(s.encode(“utf-8”)).hexdigest()

def canon_horse(name: str) -> str:
if not isinstance(name, str):
return “”
s = name.upper().strip()
s = re.sub(r”[^\w\s]”, “ “, s)
return re.sub(r”\s+”, “ “, s)

def norm_str(x: str) -> str:
“”“Lowercase, collapse spaces - for fuzzy key matching.”””
return re.sub(r”\s+”, “ “, str(x).strip().lower())

# –––––––––––––––––––––––

# NaN / Inf sanitisation (for Streamlit / Arrow)

# –––––––––––––––––––––––

def _is_nanlike(x):
try:
return (x is None)   
or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))   
or (isinstance(x, np.floating) and (np.isnan(x) or np.isinf(x)))
except Exception:
return False

def _san_df(df: pd.DataFrame) -> pd.DataFrame:
clean = df.replace([np.inf, -np.inf], np.nan).where(lambda d: d.notna(), None)
clean.index   = [None if _is_nanlike(v) else v for v in clean.index.tolist()]
clean.columns = [None if _is_nanlike(v) else v for v in clean.columns.tolist()]
return clean.astype(“object”)

def _san_ser(s: pd.Series) -> pd.Series:
ss = s.replace([np.inf, -np.inf], np.nan).where(s.notna(), None)
ss.index = [None if _is_nanlike(v) else v for v in ss.index.tolist()]
return ss.astype(“object”)

def sanitize(obj):
“”“Recursively replace NaN/Inf with None; safe for Arrow / JSON.”””
if isinstance(obj, pd.DataFrame):
return _san_df(obj).reset_index(drop=True)
if isinstance(obj, pd.Series):
return _san_ser(obj).reset_index(drop=True)
try:
from pandas.io.formats.style import Styler
if isinstance(obj, Styler):
obj.data = _san_df(obj.data).reset_index(drop=True)
return obj
except Exception:
pass
if isinstance(obj, np.ndarray):
return [sanitize(v) for v in obj.tolist()]
if isinstance(obj, dict):
return {k: sanitize(v) for k, v in obj.items()}
if isinstance(obj, (list, tuple)):
return type(obj)(sanitize(v) for v in obj)
return None if _is_nanlike(obj) else obj

def sanitize_jsonable(obj, ndigits=3):
“”“Recursively convert NaN/Inf _ None and round floats.”””
if obj is None:
return None
if isinstance(obj, (float, np.floating)):
return None if not np.isfinite(obj) else round(float(obj), ndigits)
if isinstance(obj, (int, np.integer, str, bool)):
return obj
if isinstance(obj, dict):
return {k: sanitize_jsonable(v, ndigits) for k, v in obj.items()}
if isinstance(obj, (list, tuple, set)):
return [sanitize_jsonable(v, ndigits) for v in obj]
if isinstance(obj, pd.Series):
return [sanitize_jsonable(v, ndigits) for v in obj.tolist()]
if isinstance(obj, pd.DataFrame):
return [sanitize_jsonable(r, ndigits) for _, r in obj.iterrows()]
try:
return sanitize_jsonable(float(obj), ndigits)
except Exception:
return None

# –––––––––––––––––––––––

# Matplotlib helpers

# –––––––––––––––––––––––

def color_cycle(n: int) -> list:
base = plt.rcParams[“axes.prop_cycle”].by_key().get(
“color”, [f”C{i}” for i in range(10)]
)
out, i = [], 0
while len(out) < n:
out.append(base[i % len(base)])
i += 1
return out

def safe_bal_norm(series, center=100.0, pad=0.5) -> TwoSlopeNorm:
“”“TwoSlopeNorm that always satisfies vmin < center < vmax.”””
arr = pd.to_numeric(series, errors=“coerce”).astype(float).to_numpy()
arr = arr[np.isfinite(arr)]
if arr.size == 0:
return TwoSlopeNorm(vcenter=center, vmin=center - 5, vmax=center + 5)
vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
if vmin == vmax:
vmin -= max(pad, 0.1)
vmax += max(pad, 0.1)
if vmax <= center:
vmax = center + max(pad, (center - vmin) * 0.05 + 0.1)
if vmin >= center:
vmin = center - max(pad, (vmax - center) * 0.05 + 0.1)
if not (vmin < center < vmax):
vmin = center - 0.1 if vmin >= center else vmin
vmax = center + 0.1 if vmax <= center else vmax
return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

# –––––––––––––––––––––––

# Weight / mass parsing  (single authoritative copy)

# –––––––––––––––––––––––

def parse_mass_to_kg(v) -> float:
“””
Parse a horse weight value to kg.
Accepts: plain kg float/str, ‘NNkg’, ‘NNlb/lbs’, stones-lbs (‘9-4’, ‘9st4lb’),
bare number (>200 assumed lb, 30-120 assumed kg).
Returns np.nan on failure.
“””
if v is None or (isinstance(v, float) and not np.isfinite(v)):
return np.nan
s = str(v).strip().lower()
if not s or s == “nan”:
return np.nan

```
# stones-lbs: "9-4", "9 st 4", "9st4lb"
m = re.match(r"^\s*(\d+)\s*(?:st|stone)?\s*[-\s]\s*(\d+)\s*(?:lb|lbs)?\s*$", s)
if m:
    return (float(m.group(1)) * 14.0 + float(m.group(2))) * 0.45359237

# explicit kg
m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(?:kg|kilogram[s]?)\s*$", s)
if m:
    return float(m.group(1))

# explicit lb
m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(?:lb|lbs|pound[s]?)\s*$", s)
if m:
    return float(m.group(1)) * 0.45359237

# bare number
m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
if m:
    x = float(m.group(1))
    if "lb" in s or "lbs" in s or x > 200:
        return x * 0.45359237
    if 30 <= x <= 120:
        return x
    if 100 <= x <= 200:
        return x * 0.45359237

return np.nan
```

def pick_mass_column(df_cols) -> str | None:
“”“Return the best weight column name from a list of columns, or None.”””
keys = {re.sub(r”[\s_]+”, “”, c.lower()): c for c in df_cols}
for k in (“carriedkg”, “horseweight”, “weight”, “wt”, “bodyweight”,
“bw”, “mass”, “declaredweight”, “officialweight”, “carriedweight”,
“carried”, “weightcarried”):
if k in keys:
return keys[k]
cands = [c for c in df_cols if “weight” in c.lower()]
return sorted(cands, key=len)[0] if cands else None

def mass_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
“””
Return (mass_kg Series, source_column_name).
Falls back to NaN series if no weight column found.
“””
src = pick_mass_column(df.columns)
if src is None:
return pd.Series(np.nan, index=df.index, dtype=float), “none”
kg = pd.to_numeric(pd.Series(df[src]).map(parse_mass_to_kg), errors=“coerce”)
return kg, src
