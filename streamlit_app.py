# ======================= Batch 1 — Core + UI + I/O + DB bootstrap =======================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io, math, re, os, sqlite3, hashlib
from datetime import datetime

# ======================= Global NaN/Inf → None guard (JSON-safe, index-safe) =======================
import math, numpy as np, pandas as pd, streamlit as st
pd.options.mode.use_inf_as_na = True

def _is_nanlike(x):
    try:
        return (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) \
               or (isinstance(x, (np.floating,)) and (np.isnan(x) or np.isinf(x)))
    except Exception:
        return False

def _san_df(df: pd.DataFrame) -> pd.DataFrame:
    # values
    clean = df.replace([np.inf, -np.inf], np.nan).where(lambda d: d.notna(), None)
    # index/columns too
    clean.index   = [None if _is_nanlike(v) else v for v in clean.index.tolist()]
    clean.columns = [None if _is_nanlike(v) else v for v in clean.columns.tolist()]
    # force object dtype so Arrow doesn’t re-infer with NaNs
    return clean.astype("object")

def _san_ser(s: pd.Series) -> pd.Series:
    ss = s.replace([np.inf, -np.inf], np.nan).where(s.notna(), None)
    ss.index = [None if _is_nanlike(v) else v for v in ss.index.tolist()]
    return ss.astype("object")

def _sanitize(obj):
    # pandas
    if isinstance(obj, pd.DataFrame): return _san_df(obj).reset_index(drop=True)
    if isinstance(obj, pd.Series):    return _san_ser(obj).reset_index(drop=True)
    # pandas Styler
    try:
        from pandas.io.formats.style import Styler
        if isinstance(obj, Styler):
            sty = obj
            sty.data = _san_df(sty.data).reset_index(drop=True)
            return sty
    except Exception:
        pass
    # numpy
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    # dict / list / tuple (recursive)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize(v) for v in obj)
    # scalars
    return None if _is_nanlike(obj) else obj

# ---- Patch common emitters (incl. data_editor) ----
_orig_write = st.write
def _safe_write(*args, **kwargs):
    return _orig_write(*(_sanitize(a) for a in args),
                       **{k: _sanitize(v) for k, v in kwargs.items()})
st.write = _safe_write

_orig_json = st.json
st.json = lambda obj, *a, **k: _orig_json(_sanitize(obj), *a, **k)

_orig_metric = st.metric
def _safe_metric(label, value, delta=None, *a, **k):
    v = _sanitize(value); d = _sanitize(delta)
    return _orig_metric(label, "-" if v is None else v, "-" if (delta is not None and d is None) else d, *a, **k)
st.metric = _safe_metric

_orig_dataframe = st.dataframe
def _safe_dataframe(data=None, *a, **k):
    data = _sanitize(data)
    # final guard: if it’s still a DataFrame-like, ensure a clean RangeIndex
    try:
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)
    except Exception:
        pass
    return _orig_dataframe(data, *a, **k)
st.dataframe = _safe_dataframe

_orig_table = st.table
st.table = lambda data=None, *a, **k: _orig_table(_sanitize(data), *a, **k)

# (optional but helpful)
if hasattr(st, "data_editor"):
    _orig_editor = st.data_editor
    st.data_editor = lambda data=None, *a, **k: _orig_editor(_sanitize(data), *a, **k)

_orig_download_button = st.download_button
def _safe_download_button(*a, **k):
    if "data" in k:
        k["data"] = _sanitize(k["data"])
        if isinstance(k["data"], (pd.DataFrame, pd.Series)):
            k["data"] = k["data"].to_csv(index=False).encode("utf-8")
    return _orig_download_button(*a, **k)
st.download_button = _safe_download_button
# ======================= /Global guard =======================
# ----------------------- Page config -----------------------
st.set_page_config(
    page_title="Race Edge — PI v3.2 + Hidden v2 + Ability v2 + CG + Race Shape + DB",
    layout="wide"
)

# ----------------------- Globals ---------------------------
DB_DEFAULT_PATH = "race_edge.db"
APP_VERSION = "3.2"

# ----------------------- Small helpers ---------------------
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def mad_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    try:
        lo = s.quantile(p_lo); hi = s.quantile(p_hi)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def canon_horse(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out, i = [], 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

# ---------- Safety helpers (drop-in) ----------
def to_num(x):
    """Coerce to numeric with NaN on failure (alias for consistency)."""
    return pd.to_numeric(x, errors="coerce")

def safe_num(x, default=0.0):
    """Return a finite float; else default."""
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def sanitize_jsonable(obj, ndigits=3):
    """
    Recursively convert NaN/Inf -> None and round floats.
    Use before st.json/vega/altair or writing dicts to state.
    """
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj): return None
        return round(float(obj), ndigits)
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
    # fallback
    try:
        return sanitize_jsonable(float(obj), ndigits)
    except Exception:
        return None

from matplotlib.colors import TwoSlopeNorm

def _safe_bal_norm(series, center=100.0, pad=0.5):
    """Return a TwoSlopeNorm that always satisfies vmin < center < vmax.
    Falls back to a tiny padded range around the data if it's one-sided or flat."""
    arr = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        vmin, vmax = center - 5.0, center + 5.0
        return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    # If all values are the same, make a tiny symmetric band around the value
    if vmin == vmax:
        vmin = vmin - max(pad, 0.1)
        vmax = vmax + max(pad, 0.1)

    # Ensure the center sits strictly inside (vmin, vmax)
    if vmax <= center:
        vmax = center + max(pad, (center - vmin) * 0.05 + 0.1)
    if vmin >= center:
        vmin = center - max(pad, (vmax - center) * 0.05 + 0.1)

    # Final guard: if still touching, nudge a hair
    if not (vmin < center < vmax):
        if vmin >= center:
            vmin = center - 0.1
        if vmax <= center:
            vmax = center + 0.1

    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

def render_profile_badge(label: str, color_hex: str):
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{color_hex};
            color:white;
            padding:4px 10px;
            border-radius:8px;
            font-weight:600;
            font-size:0.9rem;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(f"### Race Edge v{APP_VERSION}")
    st.caption("PI v3.2 with Race Shape (SED/FRA/SCI), CG, Hidden v2, Ability v2, DB")

    st.markdown("#### Upload race")
    up = st.file_uploader(
        "Upload CSV/XLSX with **100 m** or **200 m** splits.\n"
        "Finish column variants accepted: `Finish_Time`, `Finish_Split`, or `Finish`.",
        type=["csv","xlsx","xls"]
    )
    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1600)

    st.markdown("#### Toggles")
    USE_CG = st.toggle("Use Corrected Grind (CG)", value=True, help="Adjust Grind when the field finish collapses; preserves finisher credit.")
    DAMPEN_CG = st.toggle("Dampen Grind weight if collapsed", value=True, help="Shift a little weight Grind→Accel/tsSPI on collapse races.")
    USE_RACE_SHAPE = st.toggle("Use Race Shape module (SED/FRA/SCI)", value=True,
                               help="Detect slow-early/sprint-home and apply False-Run Adjustment and consistency guardrails.")
        # --- Going (affects PI weighting only) ---
    USE_GOING_ADJUST = st.toggle(
        "Use Going Adjustment",
        value=True,
        help="Adjust PI weighting based on track going (Firm/Good/Soft/Heavy)"
    )
    GOING_TYPE = st.selectbox(
        "Track Going",
        options=["Good", "Firm", "Soft", "Heavy"],
        index=0,
        help="Only affects PI weights; GCI stays independent"
    ) if USE_GOING_ADJUST else "Good"
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

    # --- Wind (display-only for now) ---
    WIND_AFFECTED = st.toggle("Wind affected race?", value=False, help="Purely informational (disclaimer only).")
    WIND_TAG = st.selectbox(
        "Wind note",
        options=["Headwind", "Tailwind", "Crosswind", "Negligible"],
        index=0,
        disabled=not WIND_AFFECTED,
    ) if WIND_AFFECTED else "None"

    st.markdown("---")
    st.markdown("#### Database")
    db_path = st.text_input("Database path", value=DB_DEFAULT_PATH)
    init_btn = st.button("Initialise / Check DB")

# ----------------------- DB init (races + performances) -------------------
if init_btn:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript("""
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
  going          TEXT,       -- NEW: Track going used for PI weighting (Firm/Good/Soft/Heavy)
  app_version    TEXT,
  created_ts     TEXT DEFAULT (datetime('now')),
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
  pi_rs           REAL,    -- NEW: PI after race-shape adjustments (if any)
  gci             REAL,
  gci_rs          REAL,    -- NEW: GCI after race-shape adjustments (if any)
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
  inserted_ts     TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_perf_horse ON performances(horse_canon);
CREATE INDEX IF NOT EXISTS idx_perf_race  ON performances(race_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
""")
        conn.commit()
        conn.close()
        st.success(f"DB ready at {db_path}")
    except Exception as e:
        st.error(f"DB init failed: {e}")

# ----------------------- Stop until a file is uploaded --------------------
if not up:
    st.stop()

# ----------------------- Header normalization / Aliases -------------------
def normalize_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize common variants (case-insensitive):
      • '<meters>_time' or '<meters>m_time'      -> '<meters>_Time'
      • '<meters>_split' or '<meters>m_split'    -> '<meters>_Time'
      • 'finish_time' / 'finish_split' / 'finish'-> 'Finish_Time'
      • 'finish_pos'                              -> 'Finish_Pos'
      • pass-through every other column
    """
    notes = []
    lmap = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    def alias(src_key, alias_col):
        nonlocal df, notes
        if src_key in lmap and alias_col not in df.columns:
            df[alias_col] = df[lmap[src_key]]
            notes.append(f"Aliased `{lmap[src_key]}` → `{alias_col}`")

    # Finish variants (for 200m data, this is 200→0)
    for k in ("finish_time", "finish_split", "finish"):
        alias(k, "Finish_Time")
    alias("finish_pos", "Finish_Pos")

    # Segment columns: accept optional 'm' before the underscore
    pat = re.compile(r"^(\d{2,4})m?_(time|split)$")
    for lk, orig in lmap.items():
        m = pat.match(lk)
        if m:
            alias_col = f"{m.group(1)}_Time"
            if alias_col not in df.columns:
                df[alias_col] = df[orig]
                notes.append(f"Aliased `{orig}` → `{alias_col}`")

    return df, notes

# ----------------------- Split-step detection -----------------------------
def detect_step(df: pd.DataFrame) -> int:
    """
    Detect whether the splits are 100m or 200m based on gaps between *_Time columns.
    """
    markers = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                markers.append(int(c.split("_")[0]))
            except Exception:
                pass
    markers = sorted(set(markers), reverse=True)
    if len(markers) < 2:
        return 100
    diffs = [markers[i] - markers[i+1] for i in range(len(markers)-1)]
    cnt100 = sum(60 <= d <= 140 for d in diffs)
    cnt200 = sum(160 <= d <= 240 for d in diffs)
    return 200 if cnt200 > cnt100 else 100

def normalize_200m_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace("\u2013","-").replace("\u2014","-") for c in df.columns]
    # coerce obvious numeric fields
    for c in df.columns:
        if c.endswith("_Time") or c.endswith("_Pos") or c in ("Race Time","800-400","400-Finish","Horse Weight","Weight Allocated","Finish_Time","Finish_Pos"):
            df[c] = to_num(df[c])
    if "Finish_Pos" not in df.columns:
        df["Finish_Pos"] = np.arange(1, len(df) + 1)
    return df

# ----------------------- File load & preview ------------------------------
try:
    raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    work, alias_notes = normalize_headers(raw.copy())
    st.success("File loaded.")
except Exception as e:
    st.error("Failed to read file.")
    st.exception(e)
    st.stop()

split_step = detect_step(work)
st.markdown(f"**Detected split step:** {split_step} m")
if alias_notes and SHOW_WARNINGS:
    st.info("Header aliases applied: " + "; ".join(alias_notes))

st.markdown("### Raw Table")
st.dataframe(work.head(12), use_container_width=True)

# ----------------------- Integrity helpers (odds-aware) -------------------
def expected_segments_from_df(df: pd.DataFrame) -> list[str]:
    """Use only *_Time columns that actually exist (highest→lowest) + Finish_Time if present."""
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    marks = sorted(set(marks), reverse=True)
    cols = [f"{m}_Time" for m in marks if f"{m}_Time" in df.columns]
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    """
    Validate only the columns that truly exist in the file.
    Reports rows where times are <=0 or NaN (treated as missing).
    """
    exp_cols = expected_segments_from_df(df)
    missing = []  # by construction we only check columns that exist
    invalid_counts = {}
    for c in exp_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    bads = [f"{k} ({v} rows)" for k, v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs), missing, invalid_counts


# Quick integrity line (display-only; full use comes after metrics)
integrity_text, _miss, _bad = integrity_scan(work, race_distance_input, split_step)
st.caption(f"Integrity: {integrity_text or 'OK'}")
if split_step == 200:
    st.caption("Finish column assumed to be the 200→0 segment (`Finish_Time`).")

# -------------------------------------------------------------------------
# Hand-off: Batch 2 will compute metrics and the Race Shape module (SED/FRA/SCI)
# ======================= Batch 2 — Metrics Engine + Race Shape (SED/SCI/FRA) =======================
# (Self-contained drop-in)

import math
import numpy as np
import pandas as pd

# ---- tiny helpers we rely on from Batch 1 (already defined there) ----
# as_num(x), clamp(v,lo,hi), mad_std(x) must exist above from Batch 1.

# -------- Stage helpers (works for 100m and 200m data) --------
def _collect_markers(df: pd.DataFrame):
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    return sorted(set(marks), reverse=True)

def _sum_times(row, cols):
    vals = [pd.to_numeric(row.get(c), errors="coerce") for c in cols]
    vals = [float(v) for v in vals if pd.notna(v) and v > 0]
    return np.sum(vals) if vals else np.nan

def _make_range_cols(D, start_incl, end_incl, step):
    if start_incl < end_incl:  # defensive
        return []
    want = list(range(int(start_incl), int(end_incl)-1, -int(step)))
    return [f"{m}_Time" for m in want]

def _stage_speed(row, cols, meters_per_split):
    if not cols: return np.nan
    tsum = _sum_times(row, cols)
    if not (pd.notna(tsum) and tsum > 0): return np.nan
    valid = [c for c in cols if pd.notna(row.get(c)) and pd.to_numeric(row.get(c), errors="coerce") > 0]
    dist = meters_per_split * len(valid)
    return np.nan if dist <= 0 else dist / tsum

def _grind_speed(row, step):
    # 100m: last 100 + Finish (100); 200m: Finish (200)
    if step == 100:
        t100 = pd.to_numeric(row.get("100_Time"), errors="coerce")
        tfin = pd.to_numeric(row.get("Finish_Time"), errors="coerce")
        parts, dist = [], 0.0
        if pd.notna(t100) and t100 > 0: parts.append(float(t100)); dist += 100.0
        if pd.notna(tfin) and tfin > 0: parts.append(float(tfin)); dist += 100.0
        return np.nan if not parts or dist <= 0 else dist / sum(parts)
    else:
        tfin = pd.to_numeric(row.get("Finish_Time"), errors="coerce")
        return np.nan if (pd.isna(tfin) or tfin <= 0) else 200.0 / float(tfin)

def _pct_at_or_above(s, thr):
    s = pd.to_numeric(s, errors="coerce")
    s = s[s.notna()]
    return 0.0 if s.empty else float((s >= thr).mean())

# -------- Adaptive F-window + tsSPI start --------
def _adaptive_f_cols_and_dist(D, step, markers, frame_cols):
    """
    Returns (f_cols, f_dist_m) for the 'early panel'.
    Rules (your spec):
      100m:  D ending with ..50  → F150 = [D-50, D-150]
             else                → F200 = [D-100, D-200]
      200m:  infer first-span from D - first_marker
             ~100 → F100 ; ~160 → F160 ; ~200 → F200 ; ~250 → F250
             (implemented by bucketing first-span)
    """
    if not markers:
        return [], 0.0

    D = float(D); step = int(step)
    if step == 100:
        if int(D) % 100 == 50:
            wanted = [int(D-50), int(D-150)]
            cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in frame_cols]
            dist = 150.0 if len(cols) == 2 else 100.0 * len(cols)
        else:
            wanted = [int(D-100), int(D-200)]
            cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in frame_cols]
            dist = 100.0 * len(cols)
        return cols, float(dist)

    # 200m data
    m1 = int(markers[0])
    first_span = max(1.0, D - m1)  # m
    c = f"{m1}_Time"
    cols = [c] if c in frame_cols else []
    # bucket by span (loose thresholds are robust to rounding)
    if first_span <= 120:    dist = 100.0
    elif first_span <= 180:  dist = 160.0
    elif first_span <= 220:  dist = 200.0
    else:                    dist = 250.0
    return cols, float(dist)

def _adaptive_tssp_start(D, step, markers):
    """tsSPI start per your spec."""
    D = float(D); step = int(step)
    if step == 100:
        return int(D - (150 if int(D) % 100 == 50 else 300))
    if not markers:
        return int(D - 400)
    first_span = D - int(markers[0])
    if first_span <= 120:  return int(D - 100)
    if first_span <= 180:  return int(D - 150)
    if first_span <= 220:  return int(D - 400)
    return int(D - 250)

# -------- Speed→Index mapping (robust to small fields) --------
def _shrink_center(idx_series):
    x = pd.to_numeric(idx_series, errors="coerce").dropna().values
    if x.size == 0: return 100.0, 0
    med_race = float(np.median(x))
    alpha = x.size / (x.size + 6.0)
    return alpha * med_race + (1 - alpha) * 100.0, x.size

def _dispersion_equalizer(delta_series, n_eff, N_ref=10, beta=0.20, cap=1.20):
    gamma = 1.0 + beta * max(0, N_ref - n_eff) / N_ref
    return delta_series * min(gamma, cap)

def _variance_floor(idx_series, floor=1.5, cap=1.25):
    deltas = idx_series - 100.0
    sigma = mad_std(deltas)
    if not np.isfinite(sigma) or sigma <= 0: return idx_series
    if sigma < floor:
        factor = min(cap, floor / sigma)
        return 100.0 + deltas * factor
    return idx_series

def _speed_to_idx(spd_series):
    s = pd.to_numeric(spd_series, errors="coerce")
    med = s.median(skipna=True)
    idx_raw = 100.0 * (s / med)
    center, n_eff = _shrink_center(idx_raw)
    idx = 100.0 * (s / (center/100.0 * med))
    idx = 100.0 + _dispersion_equalizer(idx - 100.0, n_eff)
    idx = _variance_floor(idx)
    return idx

# -------- Distance/context weights for PI --------
def _lerp(a, b, t): return a + (b - a) * float(t)

def _interp_weights(dm, a_dm, a_w, b_dm, b_w):
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

def pi_weights_distance_and_context(
    distance_m: float,
    acc_med: float|None,
    grd_med: float|None,
    going: str = "Good",
    field_n: int | None = None,
    return_meta: bool = False
) -> dict | tuple[dict, dict]:
    """
    Returns PI weights (sum=1). If return_meta=True, also returns a meta dict
    describing the going multipliers actually applied.
    Going affects PI weighting ONLY (not indices or GCI).
    """

    dm = float(distance_m or 1200)

    # ---- Base by distance (your current logic) ----
    if dm <= 1000:
        F200, ts, ACC, GR = 0.15, 0.20, 0.40, 0.25

    elif dm <= 1200:
        F200, ts, ACC, GR = 0.10, 0.25, 0.40, 0.25

    elif dm <= 1400:
        F200, ts, ACC, GR = 0.10, 0.30, 0.35, 0.25

    elif dm <= 2200:
        F200, ts, ACC, GR = 0.08, 0.35, 0.35, 0.22

    else:  # 2400–3200
        F200, ts, ACC, GR = 0.05, 0.40, 0.35, 0.20

    base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    # ---- Mild context nudge (your bias step) ----
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale
        F200, ts, ACC, GR = base["F200_idx"], base["tsSPI"], base["Accel"], base["Grind"]
        if bias > 0:
            delta = min(max_shift, ACC - 0.26); ACC -= delta; GR += delta
        elif bias < 0:
            delta = min(max_shift, GR - 0.18); GR -= delta; ACC += delta
        GR = min(GR, 0.40); ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":GR}

    # ---- Going modulation (affects PI weighting ONLY) ----
    # Field-size damper: full effect by 12+ runners; scale down in small fields
    n = max(1, int(field_n or 12))
    field_scale = min(1.0, n / 12.0)

    # Going multipliers (before renormalization)
    # Good = neutral (all 1.0)
    # Firm: reward Accel & F200; soften Grind & tsSPI
    # Soft: reward Grind & tsSPI; soften Accel & F200
    # Heavy: stronger version of Soft
    if going == "Firm":
        amp_main, amp_side = 0.06, 0.03
        mult = {
            "Accel":    1.0 + amp_main * field_scale,
            "F200_idx": 1.0 + amp_side * field_scale,
            "Grind":    1.0 - amp_main * field_scale,
            "tsSPI":    1.0 - amp_side * field_scale,
        }
    elif going == "Soft":
        amp_main, amp_side = 0.06, 0.03
        mult = {
            "Accel":    1.0 - amp_main * field_scale,
            "F200_idx": 1.0 - amp_side * field_scale,
            "Grind":    1.0 + amp_main * field_scale,
            "tsSPI":    1.0 + amp_side * field_scale,
        }
    elif going == "Heavy":
        amp_main, amp_side = 0.10, 0.05
        mult = {
            "Accel":    1.0 - amp_main * field_scale,
            "F200_idx": 1.0 - amp_side * field_scale,
            "Grind":    1.0 + amp_main * field_scale,
            "tsSPI":    1.0 + amp_side * field_scale,
        }
    else:  # "Good" or unknown
        mult = {"F200_idx":1.0, "tsSPI":1.0, "Accel":1.0, "Grind":1.0}

    weighted = {k: base[k] * mult[k] for k in base.keys()}
    s = sum(weighted.values()) or 1.0
    out = {k: v / s for k, v in weighted.items()}

    if not return_meta:
        return out
    meta = {
        "going": going,
        "field_n": n,
        "multipliers": mult,
        "base": base.copy(),
        "final": out.copy()
    }
    return out, meta

# -------- Core builder --------
def build_metrics_and_shape(df_in: pd.DataFrame,
                            D_actual_m: float,
                            step: int,
                            use_cg: bool,
                            dampen_cg: bool,
                            use_race_shape: bool,
                            debug: bool):
    w = df_in.copy()
    seg_markers = _collect_markers(w)
    D = float(D_actual_m); step = int(step)

    # per-segment speeds (raw)
    for m in seg_markers:
        w[f"spd_{m}"] = (step * 1.0) / pd.to_numeric(w.get(f"{m}_Time"), errors="coerce")
    w["spd_Finish"] = ((100.0 if step == 100 else 200.0) /
                       pd.to_numeric(w.get("Finish_Time"), errors="coerce")) if "Finish_Time" in w.columns else np.nan

    # RaceTime = sum of segments (incl Finish)
    if seg_markers:
        wanted = [f"{m}_Time" for m in range(int(D)-step, step-1, -step) if f"{m}_Time" in w.columns]
        if "Finish_Time" in w.columns: wanted += ["Finish_Time"]
        w["RaceTime_s"] = w[wanted].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0,np.nan).sum(axis=1)
    else:
        w["RaceTime_s"] = pd.to_numeric(w.get("Race Time"), errors="coerce")

    # ----- Composite speeds -----
    f_cols, f_dist = _adaptive_f_cols_and_dist(D, step, seg_markers, w.columns)
    w["_F_spd"]   = w.apply(lambda r: (f_dist / _sum_times(r, f_cols)) if (f_cols and pd.notna(_sum_times(r,f_cols)) and _sum_times(r,f_cols)>0) else np.nan, axis=1)

    tssp_start    = _adaptive_tssp_start(D, step, seg_markers)
    mid_cols      = [c for c in _make_range_cols(D, tssp_start, 600, step) if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: _stage_speed(r, mid_cols, float(step)), axis=1)

    if step == 100:
        acc_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]
    else:
        acc_cols = [c for c in [f"{m}_Time" for m in [600,400]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: _stage_speed(r, acc_cols, float(step)), axis=1)

    w["_GR_spd"]  = w.apply(lambda r: _grind_speed(r, step), axis=1)

    # ----- Speed → Indices -----
    w["F200_idx"] = _speed_to_idx(w["_F_spd"])
    w["tsSPI"]    = _speed_to_idx(w["_MID_spd"])
    w["Accel"]    = _speed_to_idx(w["_ACC_spd"])
    w["Grind"]    = _speed_to_idx(w["_GR_spd"])

    # ----- Corrected Grind (CG) -----
    ACC_field = pd.to_numeric(w["_ACC_spd"], errors="coerce").mean(skipna=True)
    GR_field  = pd.to_numeric(w["_GR_spd"],  errors="coerce").mean(skipna=True)
    FSR = float(GR_field / ACC_field) if (math.isfinite(ACC_field) and ACC_field > 0 and math.isfinite(GR_field)) else np.nan
    if not np.isfinite(FSR): FSR = 1.0
    CollapseSeverity = float(min(10.0, max(0.0, (0.95 - FSR) * 100.0)))

    def _delta_g_row(r):
        mid, grd = float(r.get("_MID_spd", np.nan)), float(r.get("_GR_spd", np.nan))
        if not (math.isfinite(mid) and math.isfinite(grd) and mid > 0): return np.nan
        return 100.0 * (grd / mid)
    w["DeltaG"] = w.apply(_delta_g_row, axis=1)

    w["FinisherFactor"] = w["DeltaG"].map(lambda dg: 0.0 if not math.isfinite(dg) else float(clamp((dg-98.0)/4.0, 0.0, 1.0)))
    w["GrindAdjPts"]    = (CollapseSeverity * (1.0 - w["FinisherFactor"])).round(2)

    w["Grind_CG"] = (w["Grind"] - w["GrindAdjPts"]).clip(lower=90.0, upper=110.0)
    def _fade_cap(g, dg):
        if not (math.isfinite(g) and math.isfinite(dg)): return g
        return 100.0 + 0.5*(g-100.0) if (dg < 97.0 and g > 100.0) else g
    w["Grind_CG"] = [ _fade_cap(g, dg) for g, dg in zip(w["Grind_CG"], w["DeltaG"]) ]

    # ----- PI v3.2 -----
    GR_COL = "Grind_CG" if use_cg else "Grind"
    acc_med = pd.to_numeric(w["Accel"], errors="coerce").median(skipna=True)
    grd_med = pd.to_numeric(w[GR_COL], errors="coerce").median(skipna=True)

    # Pull going context from the outer scope (set in sidebar)
    global GOING_TYPE, USE_GOING_ADJUST
    going_for_pi = GOING_TYPE if ('USE_GOING_ADJUST' in globals() and USE_GOING_ADJUST) else "Good"

    PI_W, PI_META = pi_weights_distance_and_context(
        D,
        acc_med,
        grd_med,
        going=going_for_pi,
        field_n=len(w),
        return_meta=True
    )
    # ---- MASS (horse weight) awareness for PI ---------------------------------
    # Detect/parse a mass column, default 60 kg. Accepts: kg, lb, st-lb ("9-4"), or strings.
    def _parse_mass_to_kg(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return np.nan
        s = str(v).strip().lower()
        if not s: return np.nan
        # st-lb like "9-4" or "9st 4lb"
        m = re.match(r"^\s*(\d+)\s*st[\s\-]*([0-9]{1,2})\s*(lb|lbs)?\s*$", s)
        if m:
            stn = float(m.group(1)); lb = float(m.group(2))
            return (stn*14.0 + lb) * 0.45359237
        # plain "9-4"
        m = re.match(r"^\s*(\d+)\s*[\-\/]\s*([0-9]{1,2})\s*$", s)
        if m:
            stn = float(m.group(1)); lb = float(m.group(2))
            return (stn*14.0 + lb) * 0.45359237
        # numbers with unit
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(kg|kilogram|kilograms)\s*$", s)
        if m:
            return float(m.group(1))
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(lb|lbs|pound|pounds)\s*$", s)
        if m:
            return float(m.group(1)) * 0.45359237
        # bare number → guess unit
        try:
            x = float(s)
            # > 200 is almost surely lb; 30..120 plausible kg
            if x > 200:      return x * 0.45359237
            if 30 <= x <= 120: return x
            # 100–200 could be lb; bias to lb if <= 90 kg equiv
            if 100 <= x <= 200: return x * 0.45359237
            return np.nan
        except Exception:
            return np.nan

    def _pick_mass_column(df_cols):
        # try common names (case-insensitive, strip spaces/underscores)
        keys = { re.sub(r"[\s_]+","", c.lower()): c for c in df_cols }
        for k in (
            "horseweight","weight","wt","bodyweight","bw","mass",
            "declaredweight","officialweight","carriedweight"
        ):
            if k in keys: return keys[k]
        # fallbacks: any column containing "weight" (shortest name wins)
        cands = [c for c in df_cols if "weight" in c.lower()]
        return sorted(cands, key=len)[0] if cands else None

    MASS_REF_KG = 60.0  # neutral reference
    mass_col = _pick_mass_column(w.columns)
    if mass_col:
        mass_kg = w[mass_col].map(_parse_mass_to_kg)
    else:
        mass_kg = pd.Series(np.nan, index=w.index)

    # Fill with median when available; else 60
    if mass_kg.notna().any():
        mass_kg = mass_kg.fillna(mass_kg.median(skipna=True))
    mass_kg = mass_kg.fillna(MASS_REF_KG)

    # Distance-sensitive per-kg penalty (in PI_pts space, pre-scaling)
    # Calibrated so that ~1600m charges ~0.16 pts per kg around the reference.
    def _perkg_pts(dm):
        dm = float(dm)
        # anchors: (1000m→0.10), (1200→0.12), (1400→0.14), (1600→0.16), (2000→0.20), (2400→0.24)
        knots = [(1000,0.10),(1200,0.12),(1400,0.14),(1600,0.16),(2000,0.20),(2400,0.24)]
        if dm <= knots[0][0]: return knots[0][1]
        if dm >= knots[-1][0]: return knots[-1][1]
        for (a,va),(b,vb) in zip(knots, knots[1:]):
            if a <= dm <= b:
                t = (dm-a)/(b-a)
                return va + (vb - va)*t
        return 0.16  # fallback near a mile

    perkg = _perkg_pts(D)
    # Persist a short note for UI/PDF (source + per-kg effect)
    w.attrs["PI_MASS_NOTE"] = {
        "mass_col": mass_col or "(none → 60 kg default)",
        "ref_kg": MASS_REF_KG,
        "perkg_pts": round(perkg, 3),
        "distance_m": int(D),
    }

    # Signed adjustment vs reference (positive mass = penalty)
    mass_delta = (mass_kg - MASS_REF_KG)
                               

    # Store for captions/DB/PDF
    w.attrs["GOING"] = going_for_pi
    w.attrs["PI_GOING_META"] = PI_META
    if use_cg and dampen_cg and CollapseSeverity >= 3.0:
        d = min(0.02 + 0.01*(CollapseSeverity-3.0), 0.08)
        shift = min(d, PI_W["Grind"])
        PI_W["Grind"] -= shift
        PI_W["Accel"] += shift*0.5
        PI_W["tsSPI"] += shift*0.5

    # ---------- Mass-aware PI points (drop-in) ----------
    # 1) Pull carried mass (kg) from the file, robustly

    def _mass_kg_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
        """
        Returns (mass_kg, source_name).
        Accepts any of these columns if present (first match wins):
          Carried_kg, Weight, Wt, Carried, WeightCarried
        Parses:
          • pure kg: 57.0, "57", "57kg"
          • pounds: 126, "126lb"
          • stones-lbs: "9-4", "9st4lb", "9 st 4"
        """
        candidates = ["Carried_kg", "Weight", "Wt", "Carried", "WeightCarried"]
        src = next((c for c in candidates if c in df.columns), None)
        if src is None:
            return pd.Series(np.nan, index=df.index, dtype=float), "none"

        def _to_kg(v):
            if v is None:
                return np.nan
            s = str(v).strip().lower()
            if s == "" or s == "nan":
                return np.nan

            # stones-lbs: "9-4", "9 st 4", "9st4lb"
            m = re.match(r"^\s*(\d+)\s*(?:st|stone|)\s*[-\s]?\s*(\d+)\s*(?:lb|lbs|)\s*$", s)
            if m:
                st = float(m.group(1)); lb = float(m.group(2))
                return (st * 14.0 + lb) * 0.45359237

            # any number in the string
            m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
            if not m:
                return np.nan
            val = float(m.group(1))

            # unit hint
            if "lb" in s or "lbs" in s or val > 130:   # 130+ → almost surely pounds
                return val * 0.45359237
            # default kg
            return val

        mass_kg = pd.to_numeric(pd.Series(df[src]).map(_to_kg), errors="coerce")
        return mass_kg, src

    mass_kg, mass_src = _mass_kg_series(w)
    w.attrs["MASS_SRC"] = mass_src

    # Reference mass = median of available masses in this race
    mass_ref = float(np.nanmedian(mass_kg)) if np.isfinite(np.nanmedian(mass_kg)) else np.nan
    mass_delta = (mass_kg - mass_ref) if np.isfinite(mass_ref) else pd.Series(0.0, index=w.index, dtype=float)
    # Positive mass_delta = carried HEAVIER than reference → penalty

    # 2) Per-kg impact in PI points (tweakable)
    PER_KG_PTS = 0.14  # subtract 0.14 PI_pts per extra kg carried vs race median

    # 3) Base (sectional) PI points as you already do, then apply mass adjustment
    def _pi_pts_row(r):
        acc = r.get("Accel"); mid = r.get("tsSPI"); f = r.get("F200_idx"); gr = r.get(GR_COL)
        parts = []
        if pd.notna(f):   parts.append(PI_W["F200_idx"] * (f   - 100.0))
        if pd.notna(mid): parts.append(PI_W["tsSPI"]    * (mid - 100.0))
        if pd.notna(acc): parts.append(PI_W["Accel"]    * (acc - 100.0))
        if pd.notna(gr):  parts.append(PI_W["Grind"]    * (gr  - 100.0))

        if not parts:
            return np.nan
        base = sum(parts) / (sum(PI_W.values()) or 1.0)

        # mass penalty in PI_pts (kg heavier than ref lowers PI_pts)
        idx = r.name
        md = float(mass_delta.loc[idx]) if idx in mass_delta.index else 0.0
        return base - (PER_KG_PTS * md)

    w["PI_pts"] = w.apply(_pi_pts_row, axis=1)

    # 4) Convert PI_pts → PI 0..10 (same centering/scaling you use)
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    sigma = 0.75 if (not np.isfinite(sigma) or sigma < 0.75) else sigma
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)
# ---------- /Mass-aware PI points ----------

    # ----- GCI (time + shape + efficiency) -----
    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try:
            winner_time = float(w["RaceTime_s"].min())
        except Exception:
            winner_time = None

    wT = 0.25
    Wg = pi_weights_distance_and_context(
        D,
        pd.to_numeric(w["Accel"], errors="coerce").median(skipna=True),
        pd.to_numeric(w[GR_COL], errors="coerce").median(skipna=True)
    )  # (no going arg -> defaults to Good)

    wPACE = Wg["Accel"] + Wg["Grind"]
    wSS   = Wg["tsSPI"]
    wEFF  = max(0.0, 1.0 - (wT + wPACE + wSS))

    def _map_pct(x, lo=98.0, hi=104.0):
        x = float(x) if pd.notna(x) else np.nan
        return 0.0 if not np.isfinite(x) else clamp((x-lo)/(hi-lo), 0.0, 1.0)

    gci_vals = []
    for _, r in w.iterrows():
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"]) - winner_time
            T = 1.0 if d <= 0.30 else (0.7 if d <= 0.60 else (0.4 if d <= 1.00 else 0.2))
        LQ = 0.6*_map_pct(r.get("Accel")) + 0.4*_map_pct(r.get(GR_COL))
        SS = _map_pct(r.get("tsSPI"))
        acc, grd_eff = r.get("Accel"), r.get(GR_COL)
        if pd.isna(acc) or pd.isna(grd_eff):
            EFF = 0.0
        else:
            dev = (abs(acc-100.0) + abs(grd_eff-100.0))/2.0
            EFF = clamp(1.0 - dev/8.0, 0.0, 1.0)
        gci_vals.append(round(10.0 * (wT*T + wPACE*LQ + wSS*SS + wEFF*EFF), 3))
    w["GCI"] = gci_vals

    # ---- GCI refinement by Race Shape (GCI_RS) ----
    # RSI sign: + = slow-early (late favoured), - = fast-early (early favoured)
    RSI_val = float(w.attrs.get("RSI", 0.0))
    SCI_val = float(w.attrs.get("SCI", 0.0))

    # per-horse exposure along same axis (late-minus-mid already computed for RS_Component)
    dLM = (pd.to_numeric(w["Accel"], errors="coerce") -
           pd.to_numeric(w["tsSPI"], errors="coerce"))

    # normalise exposure into [-1, +1] softly (bigger fields → slightly stronger signal)
    field_n = max(1, len(w))
    expo = np.tanh((dLM / 6.0)) * np.sign(RSI_val)

    # gate by consensus; reduce if consensus is weak
    consensus = 0.60 + 0.40 * max(0.0, min(1.0, SCI_val))
    expo *= consensus

    # soft-sigmoid attenuation: trim “with-shape”, boost “against-shape”
    # cap adjustment to ±0.60 GCI points
    with_shape  = np.clip(expo, 0.0, 1.0)
    against     = np.clip(-expo, 0.0, 1.0)
    adj = 0.35*against - 0.22*with_shape

    # small distance seasoning (mile gets a touch more)
    Dm = float(D_actual_m)
    dist_gain = 1.00 + (0.06 if 1400 <= Dm <= 1800 else 0.00)
    adj *= dist_gain

    # SCI failsafe: damp everything if SCI is very low
    if SCI_val < 0.40:
        adj *= 0.5

    w["GCI_RS"] = (pd.to_numeric(w["GCI"], errors="coerce") + adj).clip(0.0, 10.0).round(3)                           
        # ----- EARLY/LATE (blended, for display only) -----
    w["EARLY_idx"] = (0.65*pd.to_numeric(w["F200_idx"], errors="coerce") +
                      0.35*pd.to_numeric(w["tsSPI"],    errors="coerce"))
    w["LATE_idx"]  = (0.60*pd.to_numeric(w["Accel"],    errors="coerce") +
                      0.40*pd.to_numeric(w[GR_COL],     errors="coerce"))

         # --- put this right BEFORE the "Race Shape gates (UNIVERSAL, eased)" block ---

    # Which grind column are we using?
    w.attrs["GR_COL"] = GR_COL
    w.attrs["STEP"]   = step
    w.attrs["FSR"]    = FSR
    w.attrs["CollapseSeverity"] = CollapseSeverity

    # Series for shape calculations
    acc = pd.to_numeric(w["Accel"],  errors="coerce")
    mid = pd.to_numeric(w["tsSPI"],  errors="coerce")
    grd = pd.to_numeric(w[GR_COL],   errors="coerce")

    # Deltas: late vs mid, and finish vs late
    dLM = acc - mid   # +ve = late stronger than mid  → SLOW_EARLY candidate
    dLG = grd - acc   # +ve = grind tougher than late → Attritional finish

        # ========================= Race Shape (Eureka RSI) =========================
    # Uses the SAME primitives you already calculated: tsSPI, Accel, Grind[_CG]
    # Sign convention:
    #   RSI > 0  → Slow-Early / Sprint-home bias (late favoured)
    #   RSI < 0  → Fast-Early / Attritional bias (early favoured)
    # Scale: approximately -10..+10 for human sense-making.

    acc = pd.to_numeric(w["Accel"], errors="coerce")
    mid = pd.to_numeric(w["tsSPI"], errors="coerce")
    grd = pd.to_numeric(w[GR_COL], errors="coerce")

    # Core axis (late minus mid): positive = slow-early; negative = fast-early
    dLM = (acc - mid)

    # Finish flavour axis (grind minus accel): positive = attritional finish; negative = sprint finish
    dLG = (grd - acc)

    def _madv(s):
        v = mad_std(pd.to_numeric(s, errors="coerce"))
        return 0.0 if (not np.isfinite(v)) else float(v)

    # 1) Consensus (SCI) on the shape direction using dLM signs
    sgn = np.sign(dLM.dropna().to_numpy())
    if sgn.size:
        sgn_med = int(np.sign(np.median(dLM.dropna())))
        sci = float((sgn == sgn_med).mean()) if sgn_med != 0 else 0.0
    else:
        sgn_med = 0
        sci = 0.0

    # 2) Directional centre and robust scale
    med_dLM = float(np.nanmedian(dLM))
    mad_dLM = _madv(dLM)
    if not np.isfinite(mad_dLM) or mad_dLM <= 0:
        mad_dLM = 1.0  # safety

    # 3) Distance sensitivity (mile & 7f get a touch more lift)
    D = float(D_actual_m)
    if   D <= 1100: dist_gain = 0.95
    elif D <= 1400: dist_gain = 1.05
    elif D <= 1800: dist_gain = 1.12
    elif D <= 2000: dist_gain = 1.05
    else:           dist_gain = 0.98

    # 4) Finish flavour adds gentle seasoning to magnitude only
    mad_dLG = _madv(dLG)
    fin_strength = 0.0 if mad_dLG == 0 else clamp(abs(np.nanmedian(dLG)) / max(mad_dLG, 1e-6), 0.0, 2.0)
    # mapped ~0..+0.6
    fin_bonus = 0.30 * fin_strength

    # 5) RSI raw → scaled to ≈ [-10, +10], with SCI gating
    # Base scale ~3.2 chosen to make |RSI|~6–8 for notably biased races.
    base_scale = 3.2
    rsi_signed = (med_dLM / mad_dLM) * base_scale
    rsi_signed *= (0.60 + 0.40 * sci)   # respect consensus
    rsi_signed *= dist_gain
    # flavour magnifies magnitude only
    rsi = np.sign(rsi_signed) * min(10.0, abs(rsi_signed) * (1.0 + fin_bonus))
    rsi = float(np.round(rsi, 2))

    # 6) Tag (human label) from RSI only
    if abs(rsi) < 1.2:
        shape_tag = "EVEN"
    elif rsi > 0:
        shape_tag = "SLOW_EARLY"
    else:
        shape_tag = "FAST_EARLY"

    # 7) RSI strength index (0..10) = |RSI|, capped
    rsi_strength = float(min(10.0, abs(rsi)))

    # 8) Per-horse exposure along the same axis (late-minus-mid)
    # Positive RS_Component = ran like late-favoured type; negative = early-favoured type
    w["RS_Component"] = (acc - mid).round(3)

    # Alignment cue: +1 with shape, -1 against shape, 0 neutral
    def _align_row(val, rsi_val, eps=0.25):
        if not (np.isfinite(val) and np.isfinite(rsi_val)) or abs(rsi_val) < 1.2:
            return 0
        if val > +eps and rsi_val > 0: return +1
        if val < -eps and rsi_val < 0: return +1
        if val > +eps and rsi_val < 0: return -1
        if val < -eps and rsi_val > 0: return -1
        return 0

    w["RSI_Align"] = [ _align_row(v, rsi) for v in w["RS_Component"] ]

    # Pretty cue for tables
    def _align_icon(a):
        if a > 0:  return "🔵 ➜ with shape"
        if a < 0:  return "🔴 ⇦ against shape"
        return "⚪ neutral"

    w["RSI_Cue"] = [ _align_icon(a) for a in w["RSI_Align"] ]

    # Save attrs you already expose / use elsewhere
    w.attrs["RSI"]         = float(rsi)
    w.attrs["RSI_STRENGTH"]= float(rsi_strength)
    w.attrs["SCI"]         = float(sci)
    w.attrs["SHAPE_TAG"]   = shape_tag
    # Informational finish flavour (kept from your previous UX)
    fin_flav = "Balanced Finish"
    med_dLG = float(np.nanmedian(dLG))
    gLG_gate = max(1.40, 0.50 * _madv(dLG))  # keep your eased threshold
    if   med_dLG >= +gLG_gate: fin_flav = "Attritional Finish"
    elif med_dLG <= -gLG_gate: fin_flav = "Sprint Finish"
    w.attrs["FINISH_FLAV"] = fin_flav
    return w, seg_markers


# ---- Compute metrics + race shape now ----
try:
    metrics, seg_markers = build_metrics_and_shape(
        work,
        float(race_distance_input),
        int(split_step),
        USE_CG,
        DAMPEN_CG,
        USE_RACE_SHAPE,
        DEBUG
    )
except Exception as e:
    st.error("Metric computation failed.")
    st.exception(e)
    st.stop()

# ----------------------- Race Quality Score (RQS v2) -----------------------
def compute_rqs(df: pd.DataFrame, attrs: dict) -> float:
    """
    RQS v2 (0..100): single-race class/quality indicator, handicap-friendly.

    Ingredients (no GCI in the core):
      • Peak talent  (S1): p90(PI_RS or PI) scaled to 0..100
      • Depth        (S2): share with PI_RS >= 6.8 (good/strong mark)
      • Dispersion   (S3): how healthy the spread is (MAD target ≈ 1.0)
      • Field trust  (mult): small lift for bigger fields, small trim for tiny fields
      • False-run penalty: if FRA applied with real conviction (SCI high)
    """
    if df is None or len(df) == 0:
        return 0.0

    # Prefer RS-adjusted; fall back cleanly
    pi = pd.to_numeric(df.get("PI_RS", df.get("PI")), errors="coerce")
    pi = pi[pi.notna()]
    if pi.empty:
        return 0.0

    # ---------- S1: Peak talent (p90 of PI) ----------
    p90 = float(np.nanpercentile(pi.values, 90))
    S1  = 10.0 * p90                    # 0..100

    # ---------- S2: Depth above a strong bar ----------
    # 6.8 is a “solid black-type” effort in your scale.
    depth_prop = _pct_at_or_above(pi, 6.8)
    S2 = 100.0 * depth_prop             # 0..100

    # --- S3: Dispersion “health” (use RAW MAD, not mad_std) ---
    center = float(np.nanmedian(pi))
    mad_raw = float(np.nanmedian(np.abs(pi - center)))  # <-- raw MAD (no 1.4826)
    if not np.isfinite(mad_raw): mad_raw = 1.2
    # peak at 1.2; falls linearly to 0 at 0 or 2.4
    S3 = 100.0 * max(0.0, 1.0 - abs(mad_raw - 1.2) / 1.2)

    # ---------- field-size trust (multiplier 0.85..1.15) ----------
    n = int(len(df.index))
    # 6 → 0.85 ; 10 → ~1.02 ; 12 → 1.08 ; 14+ → 1.15 (capped)
    trust = 0.85 + 0.30 * max(0.0, min(1.0, (n - 6) / 8.0))
    trust = float(min(1.15, max(0.85, trust)))

    # ---------- False-run penalty (only when FRA actually applied & SCI meaningful) ----------
    fra_applied = int(attrs.get("FRA_APPLIED", 0) or 0)
    sci         = float(attrs.get("SCI", 1.0))
    penalty = 0.0
    if fra_applied == 1 and sci >= 0.60:
        # Up to 10 pts at SCI=1.0, scaled from 0 at 0.60
        penalty = 10.0 * (sci - 0.60) / 0.40
        penalty = float(min(10.0, max(0.0, penalty)))

    # ---------- Blend (weights sum to 1.0 before multiplier) ----------
    # Handicap-friendly: put most weight on peak & depth; dispersion still matters.
    w1, w2, w3 = 0.55, 0.25, 0.20
    base = w1*S1 + w2*S2 + w3*S3

    rqs = base * trust - penalty
    return float(np.clip(round(rqs, 1), 0.0, 100.0))

# Compute once and store into attrs (used by header / PDF / DB)
metrics.attrs["RQS"] = compute_rqs(metrics, metrics.attrs)

def compute_rps(df: pd.DataFrame) -> float:
    """
    RPS (0..100): star-aware peak strength.
    Blends p95(PI_RS) with the true peak based on dominance and field-size trust.
    """
    if df is None or len(df) == 0:
        return 0.0

    pi = pd.to_numeric(df.get("PI_RS", df.get("PI")), errors="coerce").dropna()
    n = int(pi.size)
    if n == 0:
        return 0.0

    p95 = float(np.nanpercentile(pi, 95)) if n >= 5 else float(np.nanmax(pi))
    p90 = float(np.nanpercentile(pi, 90)) if n >= 4 else p95
    pmax = float(np.nanmax(pi))
    # second-best (for Gap2)
    if n >= 2:
        top2 = np.partition(pi.values, -2)[-2]
    else:
        top2 = p90

    # Dominance signals (in PI points, not ×10)
    gap_top = max(0.0, pmax - p90)       # how far top is beyond the elite band
    gap_2   = max(0.0, pmax - float(top2))  # separation to 2nd

    # Field-size trust: 0 at ≤6; 1 at ≥12 (same shape as you used elsewhere)
    trust = max(0.0, min(1.0, (n - 6) / 6.0))

    # Turn dominance into [0..1] via a smooth ramp that saturates ~3pts
    # (≈ 3 PI points above the pack is huge)
    def smooth_saturate(x, mid=1.8, span=1.2):
        # 0 at x=0; ~0.5 at mid; →1 near mid+span (~3.0)
        return max(0.0, min(1.0, (x / (mid + span))))

    dom = max(smooth_saturate(gap_top), smooth_saturate(gap_2))

    # Final blend weight toward the max:
    # base 0.30 (slight pull to max), + up to +0.40 from dominance*trust
    w_star = 0.30 + 0.40 * dom * trust
    w_star = max(0.0, min(0.95, w_star))  # safety clamp

    rps_pi = (1.0 - w_star) * p95 + w_star * pmax
    return float(np.clip(round(10.0 * rps_pi, 1), 0.0, 100.0))

# ----------------------- Race Peak Strength (RPS) + Race Profile -----------------------
def compute_rps(df: pd.DataFrame) -> float:
    """
    RPS (0..100): star-aware peak strength.
    Blends p95(PI_RS) with the true peak based on dominance and field-size trust.
    """
    if df is None or len(df) == 0:
        return 0.0

    pi = pd.to_numeric(df.get("PI"), errors="coerce").dropna()
    n = int(pi.size)
    if n == 0:
        return 0.0

    p95 = float(np.nanpercentile(pi, 95)) if n >= 5 else float(np.nanmax(pi))
    p90 = float(np.nanpercentile(pi, 90)) if n >= 4 else p95
    pmax = float(np.nanmax(pi))
    # second-best (for Gap2)
    if n >= 2:
        top2 = np.partition(pi.values, -2)[-2]
    else:
        top2 = p90

    # Dominance signals (in PI points, not ×10)
    gap_top = max(0.0, pmax - p90)       # how far top is beyond the elite band
    gap_2   = max(0.0, pmax - float(top2))  # separation to 2nd

    # Field-size trust: 0 at ≤6; 1 at ≥12 (same shape as you used elsewhere)
    trust = max(0.0, min(1.0, (n - 6) / 6.0))

    # Turn dominance into [0..1] via a smooth ramp that saturates ~3pts
    # (≈ 3 PI points above the pack is huge)
    def smooth_saturate(x, mid=1.8, span=1.2):
        # 0 at x=0; ~0.5 at mid; →1 near mid+span (~3.0)
        return max(0.0, min(1.0, (x / (mid + span))))

    dom = max(smooth_saturate(gap_top), smooth_saturate(gap_2))

    # Final blend weight toward the max:
    # base 0.30 (slight pull to max), + up to +0.40 from dominance*trust
    w_star = 0.30 + 0.40 * dom * trust
    w_star = max(0.0, min(0.95, w_star))  # safety clamp

    rps_pi = (1.0 - w_star) * p95 + w_star * pmax
    return float(np.clip(round(10.0 * rps_pi, 1), 0.0, 100.0))

def classify_race_profile(rqs: float, rps: float) -> tuple[str, str]:
    """
    Return (label, color_hex) for the badge.
    - 🔴 Top-Heavy  when RPS - RQS ≥ 18
    - 🟢 Deep Field when RQS ≥ RPS - 10
    - ⚪ Average Profile otherwise
    """
    if not (math.isfinite(rqs) and math.isfinite(rps)):
        return ("Unknown", "#7f8c8d")
    delta = rps - rqs
    if delta >= 18.0:
        return ("🔴 Top-Heavy", "#e74c3c")
    elif rqs >= (rps - 10.0):
        return ("🟢 Deep Field", "#2ecc71")
    else:
        return ("⚪ Average Profile", "#95a5a6")

metrics.attrs["RPS"] = compute_rps(metrics)
_profile_label, _profile_color = classify_race_profile(
    float(metrics.attrs.get("RQS", np.nan)),
    float(metrics.attrs.get("RPS", np.nan))
)
metrics.attrs["RACE_PROFILE"] = _profile_label
metrics.attrs["RACE_PROFILE_COLOR"] = _profile_color

# ======================= Data Integrity & Header (post compute) ==========================
def _expected_segments(distance_m: float, step:int) -> list[str]:
    cols = [f"{m}_Time" for m in range(int(distance_m)-step, step-1, -step)]
    cols.append("Finish_Time")
    return cols

def _integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    exp_cols = _expected_segments(distance_m, step)
    missing = [c for c in exp_cols if c not in df.columns]
    invalid_counts = {}
    for c in exp_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    if missing: msgs.append("Missing: " + ", ".join(missing))
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    if bads: msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs), missing, invalid_counts

integrity_text, missing_cols, invalid_counts = _integrity_scan(work, race_distance_input, split_step)

# ======================= Header with RQS + RPS + Badge =======================
_hdr = (
    f"## Race Distance: **{int(race_distance_input)}m**  |  "
    f"Split step: **{split_step}m**  |  "
    f"Shape: **{metrics.attrs.get('SHAPE_TAG','EVEN')}**  |  "
    f"RSI: **{metrics.attrs.get('RSI',0.0):+.2f} / 10**  |  "
    f"Finish: **{metrics.attrs.get('FINISH_FLAV','Balanced Finish')}**  |  "
    f"SCI: **{metrics.attrs.get('SCI',0.0):.2f}**  |  "
    f"FRA: **{'Yes' if metrics.attrs.get('FRA_APPLIED',0)==1 else 'No'}**"
)
rqs_v = metrics.attrs.get("RQS", None)
rps_v = metrics.attrs.get("RPS", None)
if rqs_v is not None:
    _hdr += f"  |  **RQS:** {float(rqs_v):.1f}/100"
if rps_v is not None:
    _hdr += f"  |  **RPS:** {float(rps_v):.1f}/100"

metrics.attrs["WIND_AFFECTED"] = bool(WIND_AFFECTED)
metrics.attrs["WIND_TAG"] = str(WIND_TAG)

st.markdown(_hdr)

# Badge + short legend line
render_profile_badge(
    metrics.attrs.get("RACE_PROFILE","Unknown"),
    metrics.attrs.get("RACE_PROFILE_COLOR","#7f8c8d")
)
st.caption("RQS = field depth/consistency • RPS = peak performance • Badge = depth vs dominance")
# ----------------------- RQS Badge -----------------------
rqs_val = float(metrics.attrs.get("RQS", 0.0))
if rqs_val >= 80:
    badge_color, badge_label = "#27AE60", "Elite Class"
elif rqs_val >= 65:
    badge_color, badge_label = "#F39C12", "Competitive Field"
elif rqs_val >= 45:
    badge_color, badge_label = "#E67E22", "Moderate Class"
else:
    badge_color, badge_label = "#C0392B", "Weak Field"

st.markdown(
    f"<div style='display:inline-block;padding:4px 10px;border-radius:6px;"
    f"background-color:{badge_color};color:white;font-weight:bold;'>"
    f"RQS {rqs_val:.1f} / 100 — {badge_label}</div>",
    unsafe_allow_html=True
)
if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    warn = []
    if missing_cols: warn.append("Missing: " + ", ".join(missing_cols))
    if bads: warn.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    if warn: st.markdown(f"*(⚠ {' • '.join(warn)})*")
if split_step == 200:
    st.caption("First panel & F-window adapt to odd 200m distances (e.g., 1160→F160, 1450→F250, 1100→F100). Finish is the 200→0 split.")

st.markdown("## Sectional Metrics (PI v3.2 & GCI + CG + Race Shape)")

GR_COL = metrics.attrs.get("GR_COL", "Grind")

show_cols = [
    "Horse","Finish_Pos","RaceTime_s",
    "F200_idx","tsSPI","Accel","Grind","Grind_CG",
    "EARLY_idx","LATE_idx",
    "GrindAdjPts","DeltaG",
    "PI","GCI","GCI_RS",
    "RSI","RS_Component","RSI_Cue"
]

# ---- make the column pick robust (no KeyError if some are missing) ----
tmp = metrics.copy()
for c in show_cols:
    if c not in tmp.columns:
        tmp[c] = np.nan
display_df = tmp[show_cols].copy()

# prefer sorting by finish as secondary key when present
_finish_sort = pd.to_numeric(display_df["Finish_Pos"], errors="coerce").fillna(1e9)
display_df = display_df.assign(_FinishSort=_finish_sort).sort_values(
    ["PI","_FinishSort"], ascending=[False, True]
).drop(columns=["_FinishSort"])

st.dataframe(display_df, use_container_width=True)

# Now (optionally) backfill RSI/exposure cue columns from attrs if they were missing
if "RSI" in metrics.attrs and display_df["RSI"].isna().all():
    display_df["RSI"] = float(metrics.attrs["RSI"])
if "RS_Component" in metrics.columns and display_df["RS_Component"].isna().all():
    display_df["RS_Component"] = metrics["RS_Component"]
if "RSI_Cue" in metrics.columns and display_df["RSI_Cue"].isna().all():
    display_df["RSI_Cue"] = metrics["RSI_Cue"]
# ----- Add RSI & exposure columns to the Sectional Metrics view -----
display_df["RSI"]          = metrics.attrs.get("RSI", np.nan)
display_df["RS_Component"] = metrics.get("RS_Component", np.nan)
display_df["RSI_Cue"]      = metrics.get("RSI_Cue", "")
# Going note (PI only)
pi_meta = metrics.attrs.get("PI_GOING_META", {})
if pi_meta:
    g = str(pi_meta.get("going","Good"))
    n = int(pi_meta.get("field_n", len(display_df)))
    mult = pi_meta.get("multipliers", {})
    # Compact summary (only show components that actually moved)
    moved = [f"{k}×{mult[k]:.3f}" for k in ["Accel","F200_idx","tsSPI","Grind"] if abs(mult.get(k,1.0)-1.0) >= 0.005]
    if moved:
        st.caption(f"Going: {g} — PI weight multipliers: " + ", ".join(moved) + f" (field={n}).")
        st.caption("RSI: + = slow-early (late favoured), − = fast-early (early favoured).  RS_Component per horse uses the same axis.  🔵 with shape · 🔴 against shape.")

# ======================= Race Class Summary (pure stats) =======================
st.markdown("## Race Class Summary")

# Prefer GCI_RS, fall back to GCI
gci_col = "GCI_RS" if ("GCI_RS" in metrics.columns and pd.to_numeric(metrics["GCI_RS"], errors="coerce").notna().any()) else "GCI"
s = pd.to_numeric(metrics.get(gci_col, pd.Series(dtype=float)), errors="coerce").dropna()

if s.empty:
    st.info("No valid GCI values found for this race.")
else:
    mean_v   = float(s.mean())
    med_v    = float(s.median())
    # Robust spread stats (raw MAD, not 1.4826 scaled; and IQR)
    mad_raw  = float(np.nanmedian(np.abs(s - med_v)))
    try:
        q75, q25 = np.nanpercentile(s, 75), np.nanpercentile(s, 25)
        iqr_v    = float(q75 - q25)
    except Exception:
        iqr_v    = float("nan")

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.metric("Mean GCI", f"{mean_v:.2f}")
    with c2:
        st.metric("Median GCI", f"{med_v:.2f}")
    with c3:
        st.metric("Spread (MAD)", f"{mad_raw:.2f}")
    with c4:
        st.metric("IQR", f"{iqr_v:.2f}" if np.isfinite(iqr_v) else "—")

    st.caption(f"Source: **{gci_col}** (pure stats; no class labels).")
# =================== /Race Class Summary ===================

# ======================= Ahead of the Handicap — One-Run Weight Intelligence =======================
st.markdown("## Ahead of the Handicap — One-Run (Race-local)")

# ---- Safe helpers (local, conflict-free) ----
def _as_num(s):
    return pd.to_numeric(s, errors="coerce")

def _field_corr(x, y):
    x = _as_num(x); y = _as_num(y)
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 6:  # tiny fields → correlation not reliable
        return np.nan
    c = df["x"].corr(df["y"])
    try:
        return float(c) if np.isfinite(c) else np.nan
    except Exception:
        return np.nan

def _beta_base(distance_m: float) -> float:
    """Baseline PI-per-kg by trip; conservative, race-agnostic."""
    d = float(distance_m)
    if d <= 1200: return 0.30
    if d <= 1600: return 0.35
    if d <= 2000: return 0.40
    if d <= 2400: return 0.45
    return 0.50

def _shape_adjust(beta: float, rsi: float, sci: float) -> float:
    """Light-touch modulation by race shape (one-run safe)."""
    if np.isfinite(rsi) and np.isfinite(sci) and sci >= 0.5:
        if rsi < -0.6:  # fast-early
            beta *= 1.10
        elif rsi > 0.6: # slow-early
            beta *= 0.90
    return beta

# ---- Inputs / guards ----
# ---------------- Safe Horse Weight Resolver ----------------
weight_col_candidates = ["Horse Weight", "Horse_Weight", "Wt", "Weight", "Weight (kg)"]
weight_col = next((c for c in weight_col_candidates if c in metrics.columns), None)

# if still missing, create it directly before loc[]
if weight_col is None:
    st.warning("No weight column found — assigning 60 kg baseline for all horses.")
    metrics["Horse Weight"] = 60.0
    weight_col = "Horse Weight"
else:
    metrics["Horse Weight"] = pd.to_numeric(metrics[weight_col], errors="coerce").fillna(60.0)
    weight_col = "Horse Weight"  # standardize name
# ------------------------------------------------------------

# now this will never KeyError
df = metrics.loc[:, ["Horse", "PI", weight_col]].copy()

need_cols = {"Horse", "PI"}
if weight_col: need_cols.add(weight_col)
missing = [c for c in need_cols if c not in metrics.columns]

if missing:
    st.warning("Ahead of the Handicap: missing columns → " + ", ".join(missing))
else:
    df = metrics.loc[:, ["Horse", "PI", weight_col]].copy()
    df["PI"] = _as_num(df["PI"])
    df[weight_col] = _as_num(df[weight_col])
    df = df.dropna(subset=["PI", weight_col])
    if df.empty:
        st.info("No valid PI/weight rows to evaluate.")
    else:
        # Distance & shape inputs (safe defaults)
        D_m  = float(race_distance_input)
        RSI  = float(metrics.attrs.get("RSI", np.nan))
        SCI  = float(metrics.attrs.get("SCI", np.nan))

        # 1) Baseline slope (PI per kg)
        beta0 = _beta_base(D_m)

        # 2) Race-local realism: how much did weight correlate with performance here?
        #    Use magnitude of correlation as a realism knob, damped for tiny fields.
        corr_w_pi = _field_corr(df[weight_col], df["PI"])
        n = df["PI"].notna().sum()
        tiny_dampen = 0.0 if n < 6 else min(1.0, (n - 5) / 7.0)  # ramps in from n=6 to ~n=12
        corr_mag = 0.0 if not np.isfinite(corr_w_pi) else abs(corr_w_pi)

        # Blend: mostly baseline, plus up to +40% from race-local signal (scaled by field size)
        beta_local = beta0 * (1.0 + 0.40 * corr_mag * tiny_dampen)

        # 3) Shape modulation (light touch)
        beta_eff = _shape_adjust(beta_local, RSI, SCI)

        # 4) Safety rails (avoid crazy kg if β too tiny or huge)
        beta_eff = float(np.clip(beta_eff, 0.22, 0.70))  # keep within realistic PI/kg bounds

        # 5) Convert each horse’s PI to ΔPI vs field median, then to kg & lb
        PI_med = float(np.nanmedian(df["PI"]))
        df["ΔPI_vs_med"] = df["PI"] - PI_med
        df["RanAbove_kg"] = df["ΔPI_vs_med"] / beta_eff
        df["RanAbove_MR"] = df["RanAbove_kg"] * 2
        
        # 6) Friendly view
        view = df.copy()
        view = view.rename(columns={
            weight_col: "Wt (kg)"
        })
        view["β_eff (PI/kg)"] = beta_eff
        view = view[["Horse", "Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg", "RanAbove_MR", "β_eff (PI/kg)"]]
        view = view.sort_values("RanAbove_kg", ascending=False)

        # Round for display only (keep raw in df if you need later)
        for c in ["Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg", "RanAbove_MR", "β_eff (PI/kg)"]:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(2)

        st.dataframe(view, use_container_width=True)

        # Key readout + tiny legend
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            st.metric("Field median PI", f"{PI_med:.2f}")
        with colB:
            corr_str = "n/a" if not np.isfinite(corr_w_pi) else f"{corr_w_pi:+.2f}"
            st.metric("Weight↔PI correlation (|r| used)", corr_str)
        with colC:
            st.metric("β_eff (this race)", f"{beta_eff:.2f} PI per kg")

        st.caption(
            "Interpretation: **RanAbove (kg)** estimates how many kilograms a horse effectively ran above/below the "
            "field median, *within this single race*. Positive = ran as if it could carry more and still match median. "
            "Slope (β_eff) is distance-based, gently adjusted by (i) how weight correlated with PI in this field and "
            "(ii) race shape (fast-early increases weight bite; slow-early reduces it)."
        )

        # Optional CSV download (small footprint)
        csv_bytes = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download Ahead-of-Handicap table (CSV)", csv_bytes,
                           file_name="ahead_of_handicap_one_run.csv", mime="text/csv", use_container_width=True)
# ======================= /Ahead of the Handicap =======================
# ======================= End of Batch 2 =======================

# ======================= Batch 3 — Visuals + Hidden v2 + Ability v2 =======================
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ----------------------- Label repel (built-in fallback) -----------------------
def _repel_labels_builtin(ax, x, y, labels, *, init_shift=0.18, k_attract=0.006, k_repel=0.012, max_iter=250):
    trans=ax.transData; renderer=ax.figure.canvas.get_renderer()
    xy=np.column_stack([x,y]).astype(float); offs=np.zeros_like(xy)
    for i,(xi,yi) in enumerate(xy):
        offs[i]=[init_shift if xi>=0 else -init_shift, init_shift if yi>=0 else -init_shift]
    texts,lines=[],[]
    for (xi,yi),(dx,dy),lab in zip(xy,offs,labels):
        t=ax.text(xi+dx, yi+dy, lab, fontsize=8.4, va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
        texts.append(t)
        ln=Line2D([xi,xi+dx],[yi,yi+dy], lw=0.75, color="black", alpha=0.9)
        ax.add_line(ln); lines.append(ln)
    inv=ax.transData.inverted()
    for _ in range(max_iter):
        moved=False
        bbs=[t.get_window_extent(renderer=renderer).expanded(1.02,1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1,len(texts)):
                if not bbs[i].overlaps(bbs[j]): continue
                ci=((bbs[i].x0+bbs[i].x1)/2,(bbs[i].y0+bbs[i].y1)/2)
                cj=((bbs[j].x0+bbs[j].x1)/2,(bbs[j].y0+bbs[j].y1)/2)
                vx,vy=ci[0]-cj[0],ci[1]-cj[1]
                if vx==0 and vy==0: vx=1.0
                n=(vx**2+vy**2)**0.5; dx,dy=(vx/n)*k_repel*72,(vy/n)*k_repel*72
                for t,s in ((texts[i],+1),(texts[j],-1)):
                    tx,ty=t.get_position()
                    px=trans.transform((tx,ty))+s*np.array([dx,dy])
                    t.set_position(inv.transform(px)); moved=True
        if not moved: break
    for t,ln,(xi,yi) in zip(texts,lines,xy):
        tx,ty=t.get_position(); ln.set_data([xi,tx],[yi,ty])

def label_points_neatly(ax, x, y, names):
    try:
        from adjustText import adjust_text
        texts=[ax.text(xi,yi,nm,fontsize=8.4,
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
               for xi,yi,nm in zip(x,y,names)]
        adjust_text(texts, x=x, y=y, ax=ax,
                    only_move={'points':'y','text':'xy'},
                    force_points=0.6, force_text=0.7,
                    expand_text=(1.05,1.15), expand_points=(1.05,1.15),
                    arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9,
                                    shrinkA=0, shrinkB=3))
    except Exception:
        _repel_labels_builtin(ax, x, y, names)

# ======================= Visual 1: Sectional Shape Map =======================
st.markdown("## Sectional Shape Map — Accel (home drive) vs Grind (finish)")
shape_map_png = None
GR_COL = metrics.attrs.get("GR_COL","Grind")

need_cols={"Horse","Accel",GR_COL,"tsSPI","PI"}
if not need_cols.issubset(metrics.columns):
    st.warning("Shape Map: required columns missing: " + ", ".join(sorted(need_cols - set(metrics.columns))))
else:
    dfm = metrics.loc[:, ["Horse","Accel",GR_COL,"tsSPI","PI"]].copy()
    for c in ["Accel",GR_COL,"tsSPI","PI"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["Accel",GR_COL,"tsSPI"])
    if dfm.empty:
        st.info("Not enough data to draw the shape map.")
    else:
        dfm["AccelΔ"]=dfm["Accel"]-100.0
        dfm["GrindΔ"]=dfm[GR_COL]-100.0
        dfm["tsSPIΔ"]=dfm["tsSPI"]-100.0
        names=dfm["Horse"].astype(str).to_list()
        xv=dfm["AccelΔ"].to_numpy(); yv=dfm["GrindΔ"].to_numpy()
        cv=dfm["tsSPIΔ"].to_numpy(); piv=dfm["PI"].fillna(0).to_numpy()

        span=max(4.5,float(np.nanmax(np.abs(np.concatenate([xv,yv])))))
        lim=np.ceil(span/1.5)*1.5

        DOT_MIN, DOT_MAX = 40.0, 140.0
        pmin,pmax=np.nanmin(piv),np.nanmax(piv)
        sizes=np.full_like(xv,DOT_MIN) if not np.isfinite(pmin) or not np.isfinite(pmax) \
               else DOT_MIN+(piv-pmin)/(pmax-pmin+1e-9)*(DOT_MAX-DOT_MIN)

        fig, ax = plt.subplots(figsize=(7.8,6.2))
        # quadrant tint (stronger alpha)
        TINT=0.12
        ax.add_patch(Rectangle((0,0),lim,lim,facecolor="#4daf4a",alpha=TINT,zorder=0))
        ax.add_patch(Rectangle((-lim,0),lim,lim,facecolor="#377eb8",alpha=TINT,zorder=0))
        ax.add_patch(Rectangle((0,-lim),lim,lim,facecolor="#ff7f00",alpha=TINT,zorder=0))
        ax.add_patch(Rectangle((-lim,-lim),lim,lim,facecolor="#984ea3",alpha=TINT,zorder=0))
        ax.axvline(0,color="gray",lw=1.3,ls=(0,(3,3)),zorder=1)
        ax.axhline(0,color="gray",lw=1.3,ls=(0,(3,3)),zorder=1)

        vmin,vmax=np.nanmin(cv),np.nanmax(cv)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax:
            vmin,vmax=-1.0,1.0
        norm=TwoSlopeNorm(vcenter=0.0,vmin=vmin,vmax=vmax)

        sc=ax.scatter(xv,yv,s=sizes,c=cv,cmap="coolwarm",norm=norm,
                      edgecolor="black",linewidth=0.6,alpha=0.95)
        label_points_neatly(ax,xv,yv,names)

        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_xlabel("Acceleration vs field (points) →")
        ax.set_ylabel(("Corrected " if USE_CG else "")+"Grind vs field (points) ↑")
        ax.set_title("Quadrants: +X=Accel  · +Y="+("Corrected Grind" if USE_CG else "Grind")+" · Colour=tsSPIΔ")
        s_ex=[DOT_MIN,0.5*(DOT_MIN+DOT_MAX),DOT_MAX]
        h_ex=[Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',
                     markersize=np.sqrt(s/np.pi),markeredgecolor='black') for s in s_ex]
        ax.legend(h_ex,["PI low","PI mid","PI high"],loc="upper left",frameon=False,fontsize=8)
        cbar=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.04); cbar.set_label("tsSPI − 100")
        ax.grid(True,linestyle=":",alpha=0.25)
        st.pyplot(fig)
        buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=300,bbox_inches="tight")
        shape_map_png=buf.getvalue()
        st.download_button("Download shape map (PNG)",shape_map_png,file_name="shape_map.png",mime="image/png")
        st.caption(("Y uses Corrected Grind (CG). " if USE_CG else "")+"Size=PI; X=Accel; Colour=tsSPIΔ.")


# ======================= Pace Curve — field average (lean version) =======================
st.markdown("## Pace Curve — field average (black) + Top 10 finishers")

# Optional: toggle PNG export (saves memory if off)
make_png = st.toggle("Prepare PNG for download (uses more memory)", value=False)
pace_png = None

step = int(metrics.attrs.get("STEP", 100))
D = float(race_distance_input)
marks = _collect_markers(work)

# ---- Build segments ----
segs = []
if marks:
    m1 = int(marks[0])
    L0 = max(1.0, D - m1)
    if f"{m1}_Time" in work.columns:
        segs.append((f"{int(D)}→{m1}", float(L0), f"{m1}_Time"))
    for a, b in zip(marks, marks[1:]):
        src = f"{int(b)}_Time"
        if src in work.columns:
            segs.append((f"{int(a)}→{int(b)}", float(a - b), src))
if "Finish_Time" in work.columns:
    segs.append((f"{step}→0 (Finish)", float(step), "Finish_Time"))

if not segs:
    st.info("Not enough *_Time columns to draw the pace curve.")
else:
    # ---- Compute speeds efficiently ----
    seg_keys = [f"s{i}" for i in range(len(segs))]
    arr = np.full((len(work), len(segs)), np.nan, dtype="float32")

    for j, (_, L, col) in enumerate(segs):
        if col in work.columns:
            t = pd.to_numeric(work[col], errors="coerce").astype("float32")
            t = np.where((t > 0) & np.isfinite(t), t, np.nan)
            arr[:, j] = L / t

    # Field average once
    field_avg = np.nanmean(arr, axis=0)
    if not np.isfinite(np.nanmean(field_avg)):
        st.info("Pace curve: all segments missing/invalid.")
    else:
        # ---- Pick Top 10 ----
        if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
            top10 = metrics.nsmallest(10, "Finish_Pos")
            rule = "Top-10 by Finish_Pos"
        else:
            top10 = metrics.nlargest(10, "PI")
            rule = "Top-10 by PI"

        # ---- Build plot ----
        x_idx = np.arange(len(segs))
        x_labels = [lbl for (lbl, _, _) in segs]

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot(x_idx, field_avg, color="black", lw=2.0, label="Field average")

        palette = color_cycle(len(top10))
        for i, (_, r) in enumerate(top10.iterrows()):
            speeds = np.full(len(segs), np.nan, dtype="float32")
            for j, (_, L, col) in enumerate(segs):
                t = pd.to_numeric(r.get(col, np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    speeds[j] = L / float(t)
            if np.any(np.isfinite(speeds)):
                ax.plot(x_idx, speeds, lw=1.0, marker="o", ms=2.5,
                        color=palette[i], label=str(r.get("Horse", "")))

        # ---- Axes and styling ----
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Pace over segments (left = early, right = home straight)")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, frameon=False, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        # ---- Optional PNG export ----
        if make_png:
            buf = io.BytesIO()
            fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
            ax2.plot(x_idx, field_avg, color="black", lw=2.0)
            for i, (_, r) in enumerate(top10.iterrows()):
                speeds = np.full(len(segs), np.nan, dtype="float32")
                for j, (_, L, col) in enumerate(segs):
                    t = pd.to_numeric(r.get(col, np.nan), errors="coerce")
                    if np.isfinite(t) and t > 0:
                        speeds[j] = L / float(t)
                ax2.plot(x_idx, speeds, lw=1.0, color=palette[i])
            ax2.set_xticks(x_idx)
            ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Speed (m/s)")
            ax2.set_title("Pace Curve")
            ax2.grid(True, ls="--", alpha=0.3)
            fig2.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
            plt.close(fig2)
            pace_png = buf.getvalue()
            st.download_button("Download pace curve (PNG)",
                               pace_png, file_name="pace_curve.png", mime="image/png")

        st.caption(f"Top-10 plotted: {rule}. Finish segment included explicitly.")
# ======================= /Pace Curve (lean version) =======================

# ======================= Winning DNA Matrix — distance-aware & report cards =======================
st.markdown("## Winning DNA Matrix")

WD = metrics.copy()
gr_col = metrics.attrs.get("GR_COL", "Grind")
D_m  = float(race_distance_input)
RSI  = float(metrics.attrs.get("RSI", 0.0))      # + = slow-early, - = fast-early
SCI  = float(metrics.attrs.get("SCI", 0.0))      # 0..1 (consensus/strength)

# ---- safety: make sure the columns exist ----
for c in ["Horse", "F200_idx", "tsSPI", "Accel", gr_col]:
    if c not in WD.columns:
        WD[c] = np.nan

# ---- helpers ----
def _clamp01(x): 
    return float(max(0.0, min(1.0, x)))

def _lerp(a, b, t): 
    return a + (b - a) * float(max(0.0, min(1.0, t)))

def _band_knots(metric: str):
    """
    Priors for (lo, hi) index bands that map to 0..1.
    Distance knots: 1000,1100,1200,1400,1600,1800,2000+
    """
    if metric == "EZ":  # F200_idx
        return [
            (1000, (96.0, 106.0)), (1100, (96.5, 105.5)), (1200, (97.0, 105.0)),
            (1400, (98.0, 104.0)), (1600, (98.5, 103.5)), (1800, (99.0, 103.0)),
            (2000, (99.2, 102.8)),
        ]
    if metric == "MC":  # tsSPI
        return [
            (1000, (98.0, 102.0)), (1100, (98.0, 102.0)), (1200, (98.0, 102.0)),
            (1400, (98.0, 102.2)), (1600, (97.8, 102.4)), (1800, (97.6, 102.6)),
            (2000, (97.5, 102.7)),
        ]
    if metric == "LP":  # Accel
        return [
            (1000, (96.0, 104.0)), (1100, (96.5, 103.8)), (1200, (97.0, 103.5)),
            (1400, (97.5, 103.0)), (1600, (98.0, 102.5)), (1800, (98.3, 102.3)),
            (2000, (98.5, 102.0)),
        ]
    if metric == "LL":  # Grind (or Grind_CG)
        return [
            (1000, (98.5, 101.5)), (1100, (98.0, 102.0)), (1200, (98.0, 102.0)),
            (1400, (97.5, 102.5)), (1600, (97.0, 103.0)), (1800, (96.5, 103.5)),
            (2000, (96.0, 104.0)),
        ]
    return [(1200, (98.0, 102.0)), (2000, (98.0, 102.0))]

def _prior_band(distance_m: float, metric: str):
    """Piecewise-linear interpolation over the metric's distance knots."""
    knots = _band_knots(metric)
    dm = float(distance_m)
    if dm <= knots[0][0]: 
        return knots[0][1]
    if dm >= knots[-1][0]:
        return knots[-1][1]
    for (ad, (alo, ahi)), (bd, (blo, bhi)) in zip(knots, knots[1:]):
        if ad <= dm <= bd:
            t = (dm - ad) / (bd - ad)
            return (_lerp(alo, blo, t), _lerp(ahi, bhi, t))
    return knots[-1][1]

def _shape_shift(lo, hi, metric: str, rsi: float, sci: float):
    """
    Shift band centers by ±(0.2 * SCI) depending on RSI sign.
    Fast-early (RSI<0): EZ/LP shift up; LL/MC shift down.
    Slow-early (RSI>0): LL/MC shift up; EZ/LP shift down.
    """
    if not np.isfinite(sci) or sci <= 0: 
        return lo, hi
    shift = 0.2 * float(min(1.0, max(0.0, sci)))
    center = 0.5*(lo+hi)
    half   = 0.5*(hi-lo)
    if rsi < -1e-9:   # fast-early
        if metric in ("EZ","LP"): center += shift
        if metric in ("LL","MC"): center -= shift
    elif rsi >  1e-9: # slow-early
        if metric in ("LL","MC"): center += shift
        if metric in ("EZ","LP"): center -= shift
    return (center - half, center + half)

def _blend_with_field(lo, hi, series: pd.Series):
    """Blend priors (70%) with field 10th/90th (30%) if available."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        q10, q90 = np.nanpercentile(s.dropna(), [10, 90])
        lo = 0.7*lo + 0.3*float(q10)
        hi = 0.7*hi + 0.3*float(q90)
    if hi <= lo: 
        hi = lo + 1.0
    return lo, hi

def _score01_from_band(x, lo, hi):
    try:
        xv = float(x)
        if not np.isfinite(xv): return 0.0
        return _clamp01((xv - lo) / (hi - lo))
    except Exception:
        return 0.0

# ---- compute distance/shape-aware 0..1 components ----
bands = {}
for metric_key, col in [("EZ","F200_idx"), ("MC","tsSPI"), ("LP","Accel"), ("LL",gr_col)]:
    lo0, hi0 = _prior_band(D_m, metric_key)
    lo1, hi1 = _shape_shift(lo0, hi0, metric_key, RSI, SCI)
    lo, hi   = _blend_with_field(lo1, hi1, WD[col])
    bands[metric_key] = (lo, hi)

WD["EZ01"] = WD["F200_idx"].map(lambda v: _score01_from_band(v, *bands["EZ"]))
WD["MC01"] = WD["tsSPI"].map(lambda v: _score01_from_band(v, *bands["MC"]))
WD["LP01"] = WD["Accel" ].map(lambda v: _score01_from_band(v, *bands["LP"]))
WD["LL01"] = WD[gr_col   ].map(lambda v: _score01_from_band(v, *bands["LL"]))

# ---- SOS (0..1) using your robust z-blend idea ----
def _sos01_series(df: pd.DataFrame) -> pd.Series:
    ts = winsorize(pd.to_numeric(df["tsSPI"], errors="coerce"))
    ac = winsorize(pd.to_numeric(df["Accel"], errors="coerce"))
    gr = winsorize(pd.to_numeric(df[gr_col],  errors="coerce"))
    def rz(s):
        mu, sd = np.nanmedian(s), mad_std(s)
        sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
        return (s - mu) / sd
    raw = 0.45*rz(ts) + 0.35*rz(ac) + 0.20*rz(gr)
    if raw.notna().any():
        q5, q95 = np.nanpercentile(raw.dropna(), [5, 95])
        denom = max(q95 - q5, 1.0)
        return ((raw - q5) / denom).clip(0.0, 1.0)
    return pd.Series(0.0, index=df.index)

WD["SOS01"] = _sos01_series(WD)

# ---- weights (distance + shape nudges) ----
def _wdna_base_weights(distance_m: float) -> dict:
    knots = [
        (1000, {"EZ":0.25,"MC":0.21,"LP":0.28,"LL":0.11}),
        (1100, {"EZ":0.22,"MC":0.22,"LP":0.27,"LL":0.14}),
        (1200, {"EZ":0.20,"MC":0.22,"LP":0.25,"LL":0.18}),
        (1400, {"EZ":0.15,"MC":0.24,"LP":0.24,"LL":0.22}),
        (1600, {"EZ":0.10,"MC":0.26,"LP":0.23,"LL":0.26}),
        (1800, {"EZ":0.06,"MC":0.28,"LP":0.22,"LL":0.29}),
        (2000, {"EZ":0.03,"MC":0.30,"LP":0.20,"LL":0.32}),
    ]
    dm = float(distance_m)
    if dm <= knots[0][0]:  return knots[0][1]
    if dm >= knots[-1][0]: return knots[-1][1]
    for (ad, aw), (bd, bw) in zip(knots, knots[1:]):
        if ad <= dm <= bd:
            t = (dm - ad) / (bd - ad)
            return {k: _lerp(aw[k], bw[k], t) for k in aw}
    return knots[-1][1]

def _apply_shape_nudges(w: dict, rsi: float, sci: float) -> dict:
    w = w.copy()
    mag = 0.01 * max(0.0, min(1.0, sci))
    if rsi < -1e-9:   # fast-early
        take = mag/2.0
        w["EZ"] += mag; w["LP"] += mag
        w["LL"] = max(0.0, w["LL"] - take); w["MC"] = max(0.0, w["MC"] - take)
    elif rsi >  1e-9: # slow-early
        take = mag/2.0
        w["LL"] += mag; w["MC"] += mag
        w["EZ"] = max(0.0, w["EZ"] - take); w["LP"] = max(0.0, w["LP"] - take)
    s = sum(w.values()) or 1.0
    return {k: (v/s)*0.85 for k,v in w.items()}  # SOS fixed outside

def _wdna_weights(distance_m: float, rsi: float, sci: float) -> dict:
    base = _wdna_base_weights(distance_m)
    shaped = _apply_shape_nudges(base, rsi, sci)
    shaped["SOS"] = 0.15
    S = sum(shaped.values()) or 1.0
    return {k: v / S for k, v in shaped.items()}

W = _wdna_weights(D_m, RSI, SCI)  # EZ/MC/LP/LL/SOS sum to 1.0

# ---- composite (0..1) -> re-centered 0..10 with median ≈ 5 ----
comp01 = (
    W["EZ"]  * WD["EZ01"].fillna(0.0)  +
    W["MC"]  * WD["MC01"].fillna(0.0)  +
    W["LP"]  * WD["LP01"].fillna(0.0)  +
    W["LL"]  * WD["LL01"].fillna(0.0)  +
    W["SOS"] * WD["SOS01"].fillna(0.0)
)
med = float(np.nanmedian(comp01)) if comp01.notna().any() else 0.5
q10, q90 = (np.nanpercentile(comp01.dropna(), [10,90]) if comp01.notna().any() else (0.3,0.7))
den = max(q90 - q10, 1e-6)

WD["WinningDNA"] = (5.0 + 5.0 * ((comp01 - med) / den)).clip(0.0, 10.0).round(2)

# ---- tags, tiers, summaries ----
def _top_traits_row(r):
    pairs = [("Early Zip", r.get("EZ01",0.0)), ("Mid Control", r.get("MC01",0.0)),
             ("Late Punch", r.get("LP01",0.0)), ("Lasting Lift", r.get("LL01",0.0))]
    pairs = [(n, float(v if np.isfinite(v) else 0.0)) for n,v in pairs]
    pairs.sort(key=lambda x: x[1], reverse=True)
    keep = [n for n,v in pairs[:2] if v >= 0.55]
    # specialist flags
    ez, ll = float(r.get("EZ01",0.0)), float(r.get("LL01",0.0))
    if ez >= 0.70 and ll <= 0.45: keep.append("Sprinter-leaning")
    if ll >= 0.70 and ez <= 0.45: keep.append("Stayer-leaning")
    return " · ".join(keep)

def _tier_badge(score):
    if not np.isfinite(score): return ("", "")
    if score >= 8.0: return ("🔥 Prime", "A")
    if score >= 7.0: return ("🟢 Live", "B")
    if score >= 6.0: return ("⚪ Comp", "C")
    return ("⚪ Setup", "D")

def _summary_row(r):
    name = str(r.get("Horse","")).strip()
    sc   = float(r.get("WinningDNA",0.0))
    traits = r.get("DNA_TopTraits","")
    badge,_ = _tier_badge(sc)
    parts = [f"{badge} DNA {sc:.2f}/10"]
    if traits: parts.append(traits)
    if SCI >= 0.6 and abs(RSI) >= 1.2:
        with_shape = np.sign(RSI)*(float(r.get("LP01",0.0))-float(r.get("MC01",0.0))) >= 0
        parts.append("with race shape" if with_shape else "against race shape")
    return f"{name}: " + " — ".join(parts) + "."

WD["DNA_TopTraits"] = WD.apply(_top_traits_row, axis=1)
WD["DNA_Summary"]   = WD.apply(_summary_row,    axis=1)

# ---- Top-6 report cards ----
cards = WD.sort_values("WinningDNA", ascending=False).head(6).copy()
if len(cards) > 0:
    st.markdown("**Top profiles (report cards)**")
    cols = st.columns(3)
    for i, (_, r) in enumerate(cards.iterrows()):
        col = cols[i % 3]
        nm  = str(r["Horse"])
        sc  = float(r["WinningDNA"])
        ez,mc,lp,ll,sos = [float(r[k]) for k in ["EZ01","MC01","LP01","LL01","SOS01"]]
        badge, grade = _tier_badge(sc)
        with col:
            st.markdown(
                f"""
<div style="border:1px solid rgba(255,255,255,0.15); border-radius:10px; padding:10px; margin-bottom:10px;">
  <div style="font-weight:700; font-size:1.05rem;">{nm}</div>
  <div style="margin:4px 0;"><span style="font-weight:700;">{badge}</span> · Grade <b>{grade}</b> · DNA <b>{sc:.2f}</b>/10</div>
  <div style="font-size:0.9rem; opacity:0.9;">{r['DNA_TopTraits']}</div>
  <div style="margin-top:6px; font-size:0.85rem; opacity:0.8;">
    EZ {ez:.2f} · MC {mc:.2f} · LP {lp:.2f} · LL {ll:.2f} · SOS {sos:.2f}
  </div>
</div>
""",
                unsafe_allow_html=True
            )

# ---- Full table ----
show_cols = ["Horse","WinningDNA","EZ01","MC01","LP01","LL01","SOS01","DNA_TopTraits"]
WD_view = WD.sort_values(["WinningDNA","Accel",gr_col], ascending=[False, False, False])[show_cols]
st.dataframe(WD_view, use_container_width=True)

# ---- Per-horse summaries ----
with st.expander("Race Pulse — per-horse summaries"):
    for _, r in WD.sort_values("WinningDNA", ascending=False).iterrows():
        st.write("• " + r["DNA_Summary"])

# ---- footnote: weights & band hints ----
w_note = ", ".join([f"{k} {W[k]:.2f}" for k in ["EZ","MC","LP","LL","SOS"]])
st.caption(
    f"Weights — EZ fades with distance; LL grows; shape nudges via RSI×SCI. Final weights: {w_note}. "
    f"Bands are distance-aware, shape-shifted, and blended 70/30 with field 10th/90th percentiles; "
    f"DNA recentered so race median ≈ 5."
)
# ======================= /Winning DNA Matrix =======================
    
        # ======================= Hidden Horses (v2, shape-aware) =======================
st.markdown("## Hidden Horses v2 (Shape-aware)")

hh = metrics.copy()
gr_col = metrics.attrs.get("GR_COL", "Grind")

# --- SOS (robust z-score blend) ---
need_cols = {"tsSPI", "Accel", gr_col}
if need_cols.issubset(hh.columns) and len(hh) > 0:
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh[gr_col], errors="coerce"))

    def rz(s):
        mu, sd = np.nanmedian(s), mad_std(s)
        return (s - mu) / (sd if np.isfinite(sd) and sd > 0 else 1.0)

    z_ts, z_ac, z_gr = rz(ts_w), rz(ac_w), rz(gr_w)
    hh["SOS_raw"] = 0.45*z_ts + 0.35*z_ac + 0.20*z_gr
    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = max(q95 - q5, 1.0)
    hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(0, 2)
else:
    hh["SOS"] = 0.0

# --- TFS (trip friction) ---  (moved above ASI so ASI can use TFS_plus)
def tfs_row(r):
    last_cols = [c for c in ["300_Time", "200_Time", "100_Time"] if c in r.index]
    spds = [metrics.attrs.get("STEP", 100) / as_num(r.get(c))
            for c in last_cols if pd.notna(r.get(c)) and as_num(r.get(c)) > 0]
    if len(spds) < 2:
        return np.nan
    sigma = np.std(spds, ddof=0)
    mid = as_num(r.get("_MID_spd"))
    return np.nan if not np.isfinite(mid) or mid <= 0 else 100.0 * (sigma / mid)

hh["TFS"] = hh.apply(tfs_row, axis=1)
D_rounded = int(np.ceil(float(race_distance_input) / 200.0) * 200)
_gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
hh["TFS_plus"] = hh["TFS"].apply(lambda x: 0.0 if pd.isna(x) or x < _gate else min(0.6, (x - _gate) / 3.0))


# --- ASI (Against-Shape Index, v3; race-local, 0–2 scale) ---
def _rz(s):
    s = winsorize(pd.to_numeric(s, errors="coerce"))
    mu = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - mu))
    sd = 1.4826 * mad if mad > 0 else np.nanstd(s)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (s - mu) / sd

# 1) Flow strength (FS) from RacePulse if available, else a safe proxy
RSI = metrics.attrs.get("RSI", np.nan)
SCI = metrics.attrs.get("SCI", np.nan)
collapse = float(metrics.attrs.get("CollapseSeverity", 0.0) or 0.0)

if not np.isfinite(RSI) or not np.isfinite(SCI):
    # Fallback proxy using early/late distribution
    zE = _rz(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0, index=hh.index)
    zL = _rz(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0, index=hh.index)
    RSI = float(np.nanmedian(zE) - np.nanmedian(zL))  # >0 early tilt
    SCI = 0.50  # neutral clarity if unknown

_dir = 0 if (not np.isfinite(RSI) or abs(RSI) < 1e-6) else (1 if RSI > 0 else -1)
FS = 0.0 if _dir == 0 else (0.6 + 0.4 * max(0.0, min(1.0, float(SCI)))) * min(1.0, abs(float(RSI)) / 2.0)
if collapse >= 3.0:
    FS *= 0.75  # collapse guard

# 2) Style opposition (SO): early vs late style using Accel vs Grind
zA, zG = _rz(hh.get("Accel")), _rz(hh.get(gr_col))
if _dir == 1:   # early-favoured race
    SO = (zG - zA).clip(lower=0)
elif _dir == -1:  # late-favoured race
    SO = (zA - zG).clip(lower=0)
else:
    SO = pd.Series(0.0, index=hh.index)

# 3) Segment execution opposition (XO): EARLY_idx vs LATE_idx
zE = _rz(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0, index=hh.index)
zL = _rz(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0, index=hh.index)
if _dir == 1:
    XO = (zL - zE).clip(lower=0)
elif _dir == -1:
    XO = (zE - zL).clip(lower=0)
else:
    XO = pd.Series(0.0, index=hh.index)

# 4) False-positive dampeners (trip friction & grind anomalies)
tfs_plus = pd.to_numeric(hh.get("TFS_plus"), errors="coerce").fillna(0.0)
gr_adj  = pd.to_numeric(hh.get("GrindAdjPts"), errors="coerce").fillna(1.0)

D1 = 1.0 - np.minimum(0.35, tfs_plus.clip(lower=0.0))                   # up to -35%
D2 = 1.0 - np.minimum(0.25, ((gr_adj - 1.0).clip(lower=0.0) / 3.0))      # up to -25%
D  = D1 * D2

# Combine (more weight on style than execution), scale to 0–10, then to 0–2
Opp   = 0.6 * SO + 0.4 * XO
ASI10 = 10.0 * FS * Opp * D
hh["ASI2"] = (0.2 * ASI10).clip(0.0, 2.0).fillna(0.0)
# --- UEI (underused engine) ---
def uei_row(r):
    ts, ac, gr = [as_num(r.get(k)) for k in ("tsSPI", "Accel", gr_col)]
    if any(pd.isna([ts,ac,gr])): return 0.0
    val = 0.0
    if ts >= 102 and ac <= 98 and gr <= 98:
        val = 0.3 + 0.3 * min((ts-102)/3.0, 1.0)
    if ts >= 102 and gr >= 102 and ac <= 100:
        val = max(val, 0.3 + 0.3 * min(((ts-102)+(gr-102))/6.0, 1.0))
    return round(val, 3)
hh["UEI"] = hh.apply(uei_row, axis=1)

# --- HiddenScore ---
hidden = (0.55*hh["SOS"] + 0.30*hh["ASI2"] + 0.10*hh["TFS_plus"] + 0.05*hh["UEI"]).fillna(0.0)
if len(hh) <= 6: hidden *= 0.9
h_med, h_mad = float(np.nanmedian(hidden)), float(np.nanmedian(np.abs(hidden - np.nanmedian(hidden))))
h_sigma = max(1e-6, 1.4826*h_mad)
hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5*h_sigma)).clip(0.0, 3.0)

# --- Tier logic (race-shape-aware) ---
def hh_tier_row(r):
    """Return a tier label for Hidden Horses v2."""
    hs = as_num(r.get("HiddenScore"))
    if not np.isfinite(hs):
        return ""

    # Baseline performance gates (robust to missing GCI_RS)
    pi_val = as_num(r.get("PI"))
    gci_rs = as_num(r.get("GCI_RS")) if pd.notna(r.get("GCI_RS")) else as_num(r.get("GCI"))

    # Mild gates so we don't crown complete outliers with zero baseline
    def baseline_ok_for(top: bool) -> bool:
        if top:
            return (
                (np.isfinite(pi_val)  and pi_val  >= 5.4) or
                (np.isfinite(gci_rs) and gci_rs >= 4.8)
            )
        else:
            return (
                (np.isfinite(pi_val)  and pi_val  >= 4.8) or
                (np.isfinite(gci_rs) and gci_rs >= 4.2)
            )

    if hs >= 1.8 and baseline_ok_for(top=True):
        return "🔥 Top Hidden"
    if hs >= 1.2 and baseline_ok_for(top=False):
        return "🟡 Notable Hidden"
    return ""
hh["Tier"] = hh.apply(hh_tier_row, axis=1)

# --- Descriptive note ---
def hh_note(r):
    pi, gci_rs = as_num(r.get("PI")), as_num(r.get("GCI_RS"))
    bits=[]
    if np.isfinite(pi) and np.isfinite(gci_rs):
        bits.append(f"PI {pi:.2f}, GCI_RS {gci_rs:.2f}")
    else:
        if as_num(r.get("SOS")) >= 1.2: bits.append("sectionals superior")
        asi2 = as_num(r.get("ASI2"))
        if asi2 >= 0.8: bits.append("ran against strong bias")
        elif asi2 >= 0.4: bits.append("ran against bias")
        if as_num(r.get("TFS_plus")) > 0: bits.append("trip friction late")
        if as_num(r.get("UEI")) >= 0.5: bits.append("latent potential if shape flips")
    return "; ".join(bits).capitalize()+"."
hh["Note"] = hh.apply(hh_note, axis=1)

# ---- Build ranked, presentation-friendly Hidden Horses table ----
cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel",gr_col,
           "SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

# numeric hygiene
num_cols = ["PI","GCI","tsSPI","Accel",gr_col,"SOS","ASI2","TFS","UEI","HiddenScore"]
for c in num_cols:
    hh[c] = pd.to_numeric(hh[c], errors="coerce")

# explicit tier ordering (for secondary sort / grouping)
_tier_order = {"🔥 Top Hidden": 0, "🟡 Notable Hidden": 1, "": 2}
hh["_tier_order"] = hh["Tier"].map(_tier_order).fillna(2)

# primary sort = HiddenScore (desc), then Tier order, then PI (desc)
hh_ranked = (
    hh.sort_values(["HiddenScore", "_tier_order", "PI"],
                   ascending=[False, True, False])
      .reset_index(drop=True)
)

# --- Build final table -------------------------------------------------
cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel",gr_col,"SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

# Tier order: put the best at the top
tier_order = pd.CategoricalDtype(categories=["🔥 Top Hidden","🟡 Notable Hidden",""], ordered=True)
hh["Tier"] = hh["Tier"].astype(tier_order)

hh_view = hh[cols_hh].copy()
hh_view = hh_view.sort_values(["Tier","HiddenScore","PI"], ascending=[True, False, False])

# ---- Safe numeric casting & rounding (no helper, no NameError) ----
def _cast_round(df, col):
    if col not in df.columns:
        return
    # Make a 1-D Series aligned to df.index regardless of the source shape/type
    s = pd.Series(np.ravel(df[col].values), index=df.index)
    df[col] = pd.to_numeric(s, errors="coerce").round(2)

for c in ["PI","GCI","ASI2","SOS","TFS","UEI","Accel",gr_col,"tsSPI","HiddenScore"]:
    _cast_round(hh_view, c)

st.dataframe(hh_view, use_container_width=True)
st.caption("Hidden Horses v2 — sorted by Tier, then HiddenScore, then PI (Top first).")

# ======================= PWX + EFI — Robust final build =======================
st.markdown("## PWX + EFI — Burst vs. Efficiency (field-calibrated)")

import re, numpy as np, pandas as pd

def _nz(s): return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
def _mad(x): x=np.asarray(x,float); m=np.nanmedian(x); return 1.4826*np.nanmedian(np.abs(x-m))
def _robust_z(s):
    v=_nz(s); m=np.nanmedian(v); d=_mad(v); 
    d = d if (np.isfinite(d) and d>1e-9) else 1.0
    return (v-m)/d
def _pct_rank(s):
    v=_nz(s).to_numpy(); order=np.argsort(v,kind="mergesort")
    ranks=np.empty_like(order,float); ranks[order]=np.arange(1,len(v)+1)
    pr=(ranks-0.5)/max(len(v),1); pr[~np.isfinite(v)]=0.5
    return pd.Series(pr,index=s.index)
def _cal_0_10(raw):
    pr=_pct_rank(raw); z=_robust_z(raw); sig=1/(1+np.exp(-0.85*z)); q=0.6*pr+0.4*sig
    q_src=[0.01,0.05,0.25,0.5,0.75,0.9,0.985,0.995]; y_tgt=[0.6,1.6,3.6,5,6.6,8,9.5,9.9]
    s=np.interp(q,q_src,y_tgt).clip(0,10)
    n=len(raw); 
    if n<10: s=(1-min(0.5,(10-n)/10))*s+min(0.5,(10-n)/10)*5
    return pd.Series(s,index=raw.index)
def _norm(x): return re.sub(r"\s+"," ",str(x).strip().lower())

GR_COL = metrics.attrs.get("GR_COL","Grind")
need={"Horse","Accel",GR_COL,"tsSPI"}
if not need.issubset(metrics.columns):
    st.warning("PWX/EFI: missing Accel/Grind/tsSPI")
else:
    df=metrics.loc[:,["Horse","Accel",GR_COL,"tsSPI"]].copy()
    df["HorseKey"]=df["Horse"].map(_norm)
    df["TSI_tmp"]=np.nan; df["Onset_tmp"]=np.nan

    # --- Safe optional VP merge ---
    try:
        if "VP" in globals() and isinstance(VP,pd.DataFrame) and len(VP):
            vpj=VP.copy()
            if "Horse" not in vpj.columns: vpj=vpj.reset_index().rename(columns={"index":"Horse"})
            vpj["HorseKey"]=vpj["Horse"].map(_norm)
            merge_cols=["HorseKey"]
            if "TSI" in vpj.columns: merge_cols.append("TSI")
            if "Onset_from_home_m" in vpj.columns: merge_cols.append("Onset_from_home_m")
            vpj=vpj[merge_cols].drop_duplicates("HorseKey")
            df=df.merge(vpj.rename(columns={"TSI":"TSI_tmp","Onset_from_home_m":"Onset_tmp"}),on="HorseKey",how="left")
    except Exception: pass

    for c in ["Accel",GR_COL,"tsSPI","TSI_tmp","Onset_tmp"]:
        if c in df.columns: df[c]=_nz(df[c])

    # --- compute ---
    has_tsi = "TSI_tmp" in df.columns and df["TSI_tmp"].notna().any()
    has_on  = "Onset_tmp" in df.columns and df["Onset_tmp"].notna().any()
    z_acc=_robust_z(df["Accel"])
    z_tsi=_robust_z(df["TSI_tmp"]) if has_tsi else _robust_z(df["tsSPI"])
    z_on=-_robust_z(df["Onset_tmp"]) if has_on else pd.Series(0.0,index=df.index)
    PWX_raw=0.70*z_acc+0.25*z_tsi+0.05*z_on

    sm=1-(df["Accel"]-df[GR_COL]).abs()/((df["Accel"]+df[GR_COL]).replace(0,np.nan)/2)
    z_tss=_robust_z(df["tsSPI"]); z_gr=_robust_z(df[GR_COL]); z_sm=_robust_z(sm)
    EFI_raw=0.45*z_tss+0.35*z_gr+0.20*z_sm

    df["PWX"]=_cal_0_10(PWX_raw); df["EFI"]=_cal_0_10(EFI_raw)
    out=df[["Horse","PWX","EFI"]].sort_values(["PWX","EFI"],ascending=[False,False]).reset_index(drop=True)
    st.dataframe(out,use_container_width=True)
# ======================= /PWX + EFI =======================

# ======================= Fatigue Gradient — One-Run (refined, RAM-light) =======================
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# ---------- tiny helpers (scalar / vector, RAM-light) ----------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").astype(np.float32)

def _nz(x, alt=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(alt)
    except Exception:
        return float(alt)

def _late_markers_from_cols(cols):
    """Decide late-window markers from available *_Time columns.
    100m splits: prefer 500,400,300,200,100,Finish. 200m splits: 400,200,Finish.
    Returns list of (label, length_m, src_col) in left→right order (early→late)."""
    want100 = ["500_Time","400_Time","300_Time","200_Time","100_Time","Finish_Time"]
    want200 = ["400_Time","200_Time","Finish_Time"]
    have = set(cols)

    # Pick the fuller scheme we can actually support
    if all(c in have for c in ["200_Time","Finish_Time"]) and any(c in have for c in ["300_Time","100_Time","500_Time","400_Time"]):
        # Prefer 100m if we can get >=3 points
        got = [c for c in want100 if c in have]
        # Require at least 4 points to use the 100m scheme, else fall back to 200m
        if len(got) >= 4:
            # segment lengths: 100 for 100_Time legs, Finish leg = STEP (100/200)
            return [(c.replace("_Time",""), 100.0, c) if c!="Finish_Time" else ("Finish", np.nan, c) for c in got]

    # fallback: 200m
    got = [c for c in want200 if c in have]
    if len(got) >= 3:
        return [(c.replace("_Time",""), 200.0, c) if c!="Finish_Time" else ("Finish", np.nan, c) for c in got]

    # last resort: minimal late pair if exists
    if "200_Time" in have and "Finish_Time" in have:
        return [("200", 200.0, "200_Time"), ("Finish", np.nan, "Finish_Time")]
    return []  # not enough to compute

def _extract_late_speeds(row, segs, step_default=100.0, bend_weight_first=True):
    """Return np.array of late segment speeds (m/s) for a single horse.
       segs: list[(label, length_m, src_col)] where length_m for 'Finish' is resolved via step_default.
       bend_weight_first: if True, we down-weight first late point later (not here)."""
    spd = []
    for (lab, L, src) in segs:
        Lm = float(L) if np.isfinite(L) else float(step_default)  # Finish leg length = STEP
        t = row.get(src, np.nan)
        t = _nz(t, np.nan)
        if np.isfinite(t) and t > 0:
            spd.append(Lm / t)
        else:
            spd.append(np.nan)
    return np.asarray(spd, dtype=np.float32)

def _theil_sen_slope_per200(x, y):
    """Robust slope per 200m using Theil–Sen (very small n, so O(n^2) is fine). x is distance index (0..n-1)."""
    # Use only finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    n = len(xx)
    if n < 2:
        return np.nan
    # pairwise slopes
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = (xx[j] - xx[i])
            if dx == 0:
                continue
            slopes.append( (yy[j] - yy[i]) / (dx * 200.0) )  # scale to per-200m
    if not slopes:
        return np.nan
    return float(np.median(slopes))

def _ols_slope_per200(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    n = len(xx)
    if n < 2:
        return np.nan
    xm = float(np.mean(xx)); ym = float(np.mean(yy))
    num = float(np.sum((xx - xm) * (yy - ym)))
    den = float(np.sum((xx - xm)**2))
    if den <= 0:
        return np.nan
    b = num / den  # per x-step
    return float(b / 200.0)  # per 200m

def _mono_score(v):
    """Monotonicity of late speeds: 1 - (#direction flips)/(n-1) in sign of diffs."""
    w = v[np.isfinite(v)]
    if len(w) <= 2:
        return 0.5
    dif = np.diff(w)
    sgn = np.sign(dif)
    flips = np.sum(np.abs(np.diff(sgn)) > 0)
    denom = max(1, len(sgn) - 1)
    return float(1.0 - (flips / denom))

def _trip_bucket(dm):
    dm = float(dm)
    if dm <= 1200: return "SPRINT"
    if dm <= 1650: return "MILE"
    if dm <= 2050: return "MIDDLE"
    return "STAY"

def _late_scalars_by_trip(dm):
    """Distance-aware constants (small, stable)."""
    bucket = _trip_bucket(dm)
    # late-lift scale (not used directly as m/s scaling here, but reserved)
    S_norm = {"SPRINT":0.70, "MILE":0.60, "MIDDLE":0.55, "STAY":0.50}[bucket]
    kappa  = {"SPRINT":0.18, "MILE":0.20, "MIDDLE":0.22, "STAY":0.24}[bucket]
    return S_norm, kappa

# ---------- main builder ----------
def build_fatigue_table_refined(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    # try to locate raw timing table (preferred for *_Time)
    work = None
    if st is not None:
        work = st.session_state.get("work", None)
    if work is None or not isinstance(work, pd.DataFrame):
        work = df

    # basics
    step = _nz(metrics.attrs.get("STEP", 100), 100.0)
    dm   = _nz(metrics.attrs.get("race_distance_m", st.session_state.get("race_distance_m", 1200) if st else 1200), 1200.0)
    RSI  = metrics.attrs.get("RSI", np.nan)
    SCI  = metrics.attrs.get("SCI", np.nan)
    bend = bool(metrics.attrs.get("LATE_BEND", True))  # default True: many SA tracks bend into the straight

    # late segments
    segs0 = _late_markers_from_cols(work.columns)
    if not segs0:
        # abort gracefully
        view = pd.DataFrame({"Horse": df.get("Horse", pd.Series(range(len(df))))})
        view["Note"] = "Not enough *_Time columns to compute fatigue."
        return view

    # distance-aware constants
    S_norm, kappa = _late_scalars_by_trip(dm)

    # pull per-horse late speeds
    late_speeds = []
    for _, r in work.iterrows():
        late_speeds.append(_extract_late_speeds(r, segs0, step_default=step, bend_weight_first=bend))
    late_speeds = np.asarray(late_speeds, dtype=object)  # ragged ok; each row tiny

    # one-split outlier guard & robust slope per horse
    x_idx = np.arange(len(segs0), dtype=np.float32)
    slopes = np.full(len(df), np.nan, dtype=np.float32)
    monos  = np.full(len(df), np.nan, dtype=np.float32)

    for i in range(len(df)):
        vi = np.asarray(late_speeds[i], dtype=np.float32)
        if vi.size == 0 or not np.isfinite(vi).any():
            continue

        # bend-awareness: down-weight first point (applied in slope by nudging toward neighbors)
        v = vi.copy()
        if bend and np.isfinite(v[0]) and v.size >= 2 and np.isfinite(v[1]):
            v[0] = 0.8 * v[0] + 0.2 * v[1]  # small soften; avoids fabricating speed

        # outlier guard: mask the single worst split if >12% from median of finite
        vf = v[np.isfinite(v)]
        if vf.size >= 3:
            med = float(np.median(vf))
            dev = np.abs(v - med) / (med if med > 0 else 1.0)
            j = int(np.nanargmax(dev))
            if np.isfinite(dev[j]) and dev[j] > 0.12:
                v[j] = np.nan

        # slope per 200m (Theil–Sen if 5+ finite points, else OLS)
        msk = np.isfinite(v)
        nfin = int(np.sum(msk))
        if nfin >= 2:
            if nfin >= 5:
                slopes[i] = _theil_sen_slope_per200(x_idx[msk], v[msk])
            else:
                slopes[i] = _ols_slope_per200(x_idx[msk], v[msk])
            monos[i] = _mono_score(v[msk])
        else:
            slopes[i] = np.nan
            monos[i]  = 0.5

    # FG_rate/200m: slope; more negative = more fade (fatigue)
    # Convert to "vs field" semantics (positive = MORE fade vs field, to match your earlier sign logic):
    med_slope = np.nanmedian(slopes)
    FG_vs_field = -(slopes - med_slope)  # flip so + means "faded more than field"
    # small-field shrinkage
    if len(df) <= 7:
        FG_vs_field = 0.8 * FG_vs_field

    # Line_vs_field: use provided if exists, else proxy from LATE_idx - EARLY_idx (median-centered)
    if "Line_vs_field" in df.columns:
        line_raw = _to_num(df["Line_vs_field"])
    else:
        zE = _to_num(df.get("EARLY_idx", np.nan))
        zL = _to_num(df.get("LATE_idx", np.nan))
        line_raw = (zL - zE) - np.nanmedian(zL - zE)  # raw sign: negative = stronger late
    Line_vs_field = line_raw.copy()  # keep raw sign in table for continuity
    LineEff = (-line_raw).astype(np.float32)         # internal: bigger = stronger late

    # Early heat (small, fair)
    EARLY_idx = _to_num(df.get("EARLY_idx", np.nan))
    EHeat = np.maximum(0.0, EARLY_idx - 104.0)

    # Race-shape factor (bounded, SCI-gated)
    if np.isfinite(RSI) and np.isfinite(SCI) and SCI >= 0.6:
        FS = 1.0 + np.sign(RSI) * min(0.12, 0.08 * (abs(float(RSI)) ** 0.8))
    else:
        FS = 1.0

    # Class anchor (tiny): tsSPI → normalize slow/fast run illusions
    tsSPI = _to_num(df.get("tsSPI", np.nan))
    ClassAdj = np.clip((tsSPI - 100.0) / 8.0, -0.2, 0.2)

    # Efficiency (late minus fade), small penalty for early burn; distance aware captured via slope per 200 already
    FG_eff = (LineEff - 0.50 * FG_vs_field - 0.03 * EHeat).astype(np.float32)

    # Composite fatigue score: late engine vs front-spent (positive = late engine, negative = front-spent)
    LateCloserScore  = (0.60 * LineEff + 0.30 * (-FG_vs_field)).astype(np.float32)
    FrontSpentScore  = (0.60 * (-LineEff) + 0.30 * (FG_vs_field)).astype(np.float32)
    FatigueScore_raw = (LateCloserScore - FrontSpentScore).astype(np.float32)

    # shape + class calibration (light touch)
    FatigueScore = (FS * FatigueScore_raw - 0.10 * ClassAdj).astype(np.float32)

    # Reliability = 0.5 completeness + 0.5 monotonicity
    need_cols = [
        "Accel", metrics.attrs.get("GR_COL","Grind"),
        "EARLY_idx", "LATE_idx"
    ]
    comp_bits = []
    for c in need_cols:
        if c in df.columns:
            comp_bits.append(np.isfinite(_to_num(df[c])))
    if comp_bits:
        comp = np.mean(np.column_stack(comp_bits), axis=1).astype(np.float32)
    else:
        comp = np.full(len(df), 0.5, dtype=np.float32)
    Mono = np.nan_to_num(monos, nan=0.5).astype(np.float32)
    Reliability = (0.5 * comp + 0.5 * np.clip(Mono, 0.0, 1.0)).astype(np.float32)

    # ---------- NEW TAG RULE: based ONLY on FatigueScore ----------
    FS = np.asarray(FatigueScore, float)

    Tag = np.full(len(FS), "balanced", dtype=object)
    Tag[FS < -3.0] = "late engine"
    Tag[FS >  3.0] = "front-spent"

    Cue = np.full(len(FS), "repeats", dtype=object)
    Cue[FS < -3.0] = "↑ trip / stronger pace"
    Cue[FS >  3.0] = "↓ trip / easier early"

    # Assemble view
    view = pd.DataFrame({
        "Horse": df.get("Horse", pd.Series(np.arange(len(df)), dtype=object)),
        "FG_rate/200m": np.round(slopes, 4),                 # raw slope (m/s per 200m) — diagnostic
        "FG_vs_field":  np.round(FG_vs_field, 3),            # + = faded more than field
        "Line_vs_field": np.round(Line_vs_field, 3),         # keep raw sign convention
        "FG_eff(adjusted)": np.round(FG_eff, 3),
        "Reliability": np.round(Reliability, 3),
        "Tag": Tag,
        "Cue": Cue,
        "FatigueScore": np.round(FatigueScore, 3),
    })

    # Order: late engines first, then balanced, then front-spent; within group by score desc
    ord_grp = np.where(view["Tag"] == "late engine", 0,
                np.where(view["Tag"] == "balanced", 1, 2)).astype(np.int16)
    view = view.assign(_g=ord_grp, _s=-view["FatigueScore"].astype(float))\
               .sort_values(["_g","_s","Reliability"], ascending=[True, True, False])\
               .drop(columns=["_g","_s"])\
               .reset_index(drop=True)

    return view

# ---------- render ----------
_ftab = build_fatigue_table_refined(metrics)
if st is not None:
    st.markdown("## Fatigue Gradient — refined (one-run)")
    st.dataframe(_ftab, use_container_width=True)
# ======================= /Fatigue Gradient (refined) =======================

# ======================= PWR400 MODULE — Late Power under Load (400m) =======================
# Pure power lens: last 400m speed + weight, no efficiency.
# PWR400 = distance-aware blend of Speed Index (S400) and Weight Index (W_idx).
# Uses "Horse Weight" as carried kg and renders its own Streamlit table.

import numpy as np
import pandas as pd
import streamlit as st

def compute_pwr400(
    df: pd.DataFrame,
    distance_m: float,
    step: int,
    weight_col: str = "Horse Weight"
) -> pd.DataFrame:
    """
    Compute PWR400 (Late Power Index over last 400m):

      • S400      = Speed Index over last 400m (median = 100)
      • W_idx     = Weight Index (field-relative carried weight, median = 100)
      • alpha(D)  = distance-aware weight importance
      • PWR400_raw = S400 + alpha(D) * (W_idx - 100)
      • PWR400    = PWR400_raw reindexed so race median = 100

    step = split size (100 or 200).
    distance_m = race distance in meters.
    """

    w = df.copy()
    step = int(step)
    D = float(distance_m)

    # ---------- 1. Last-400m time & speed (v400) ----------
    fin = pd.to_numeric(w.get("Finish_Time"), errors="coerce")

    if fin.isna().all():
        # No finish times → can't do anything
        w["PWR400_v400"] = np.nan
        w["PWR400_SIdx"] = np.nan
        w["PWR400_WIdx"] = np.nan
        w["PWR400_raw"]  = np.nan
        w["PWR400"]      = np.nan
        w.attrs["PWR400_NOTE"] = {"error": "Missing Finish_Time"}
        return w

    # Build last-400m time
    if step == 200:
        # Preferred: 400 + Finish (400m window)
        if "200_Time" in w.columns:
            t400 = pd.to_numeric(w["200_Time"], errors="coerce")
            t_last400 = t400 + fin
            source = "200_Time + Finish_Time"
        else:
            # Fallback: only Finish_Time available (best-effort; effectively 200m)
            t_last400 = fin.copy()
            source = "Finish_Time only (fallback; <400m)"
    else:
        # 100m splits: sum last four 100m segments where available
        segs = []
        used_cols = []
        for col in ["300_Time", "200_Time", "100_Time", "Finish_Time"]:
            if col in w.columns:
                segs.append(pd.to_numeric(w[col], errors="coerce"))
                used_cols.append(col)
        if segs:
            t_last400 = sum(segs)
            source = " + ".join(used_cols)
        else:
            t_last400 = fin.copy()
            source = "Finish_Time only (fallback)"

    t_last400 = t_last400.where(t_last400 > 0, np.nan)

    v400 = 400.0 / t_last400
    v400 = v400.replace([np.inf, -np.inf], np.nan)
    w["PWR400_v400"] = v400

    # ---------- 2. Convert speed to km/h ----------
    # 400 meters = 0.4 km
    # t_hours = t_seconds / 3600
    v400_kmh = 0.4 / (t_last400 / 3600.0)
    v400_kmh = v400_kmh.replace([np.inf, -np.inf], np.nan)
    w["PWR400_v400"] = v400_kmh

    # ---------- 3. Speed Index from km/h ----------
    v_med = float(np.nanmedian(v400_kmh))
    if np.isfinite(v_med) and v_med > 0:
        w["PWR400_SIdx"] = 100.0 * (v400_kmh / v_med)
    else:
        w["PWR400_SIdx"] = np.nan

    # ---------- 4. Weight Index W_idx (field-relative carried weight) ----------
    W = pd.to_numeric(w.get(weight_col), errors="coerce")
    W = W.where(W > 0, np.nan)

    if W.notna().any():
        W_med = float(np.nanmedian(W))
        if np.isfinite(W_med) and W_med > 0:
            w["PWR400_WIdx"] = 100.0 * (W / W_med)
        else:
            w["PWR400_WIdx"] = np.nan
    else:
        W_med = np.nan
        w["PWR400_WIdx"] = np.nan

    # ---------- 5. Distance-aware alpha(D) ----------
    # Linear Option A:
    #   alpha(D) = 0.20 + 0.00025 * (D - 1000), clamped to [0.20, 0.70]
    alpha = 0.20 + 0.00025 * (D - 1000.0)
    alpha = max(0.20, min(0.70, alpha))

    # ---------- 6. Raw Power score & indexed PWR400 ----------
    S400 = pd.to_numeric(w["PWR400_SIdx"], errors="coerce")
    Widx = pd.to_numeric(w["PWR400_WIdx"], errors="coerce")

    # Need both indices to build raw power
    mask_valid = S400.notna() & Widx.notna()
    PWR_raw = pd.Series(np.nan, index=w.index, dtype=float)
    PWR_raw[mask_valid] = S400[mask_valid] + alpha * (Widx[mask_valid] - 100.0)

    w["PWR400_raw"] = PWR_raw

    raw_med = float(np.nanmedian(PWR_raw))
    if np.isfinite(raw_med) and raw_med > 0:
        w["PWR400"] = 100.0 * (PWR_raw / raw_med)
    else:
        w["PWR400"] = np.nan

    # ---------- 7. Metadata for UI / PDF / DB ----------
    w.attrs["PWR400_NOTE"] = {
        "desc": "PWR400 = distance-aware late power: Speed Index (last 400m) + alpha(D) × (Weight Index − 100); race median = 100.",
        "distance_m": int(D),
        "alpha": round(alpha, 4),
        "weight_col": weight_col,
        "v400_source": source,
        "v400_med": round(v_med, 5) if np.isfinite(v_med) else None,
        "W_med": round(W_med, 3) if np.isfinite(W_med) else None,
    }

    return w


def render_pwr400_table(df: pd.DataFrame):
    """
    Render the standalone PWR400 table in Streamlit.
    """

    note = df.attrs.get("PWR400_NOTE", {}) or {}
    if "error" in note:
        st.markdown("## Late Power Index — **PWR400**")
        st.warning(f"PWR400 could not be computed: {note.get('error')}")
        return

    trip = note.get("distance_m", None)
    alpha = note.get("alpha", None)
    v_src = note.get("v400_source", "last 400m window")
    weight_col = note.get("weight_col", "Horse Weight")

    st.markdown("## Late Power Index — **PWR400** (Last 400m under Load)")
    if trip and alpha is not None:
        st.caption(
            f"Blend of Speed Index (last 400m) and Weight Index with distance-aware α. "
            f"Trip: **{trip}m**, α ≈ **{alpha:.3f}**, v₄₀₀ window: **{v_src}**, weight: **{weight_col}**."
        )
    else:
        st.caption("Blend of Speed Index (last 400m) and Weight Index with distance-aware α. Median PWR400 = 100.")

    # Build display columns
    cols = []
    for c in [
        "Horse", "Finish_Pos",
        weight_col,
        "PWR400_v400",
        "PWR400"
    ]:
        if c in df.columns:
            cols.append(c)

    if not cols:
        st.info("No PWR400 columns to display.")
        return

    show = df.copy()
    if "PWR400" in show.columns:
        show = show.sort_values("PWR400", ascending=False)

    st.dataframe(show[cols], use_container_width=True)
    st.caption(
        "Interpretation: PWR400 ≈ 100 = typical late power for this field; 110+ = strong late engine under meaningful weight; "
        "90–95 = weak late power."
    )

# ======================= /PWR400 MODULE =======================

# ======================= MODULE INVOCATION EXAMPLE =======================
# Call this AFTER you have `metrics`, `race_distance_input`, and `split_step` available.

try:
    metrics = compute_pwr400(
        metrics,
        float(race_distance_input),
        int(split_step),
        weight_col="Horse Weight"  # adjust if your column name differs
    )
    render_pwr400_table(metrics)
except Exception as e:
    st.warning("PWR400 module failed.")
    st.exception(e)
# ======================= /INVOCATION EXAMPLE =======================

# ======================= Visual: Power–Freshness Map (PWR400 vs Fatigue) =======================
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import io

st.markdown("## Power–Freshness Map — Late Power vs Fatigue")

# We need from the modules:
#   metrics: Horse, PI, PWR400
#   _ftab:   Horse, FatigueScore, Tag   (from build_fatigue_table_refined)
need_p  = {"Horse", "PI", "PWR400"}
need_fg = {"Horse", "FatigueScore", "Tag"}

missing_p  = need_p  - set(metrics.columns)
missing_fg = need_fg - set(_ftab.columns)

if missing_p:
    st.warning(
        "Power–Freshness Map: metrics missing columns: "
        + ", ".join(sorted(missing_p))
    )
elif missing_fg:
    st.warning(
        "Power–Freshness Map: fatigue table missing columns: "
        + ", ".join(sorted(missing_fg))
    )
else:
    # Merge power + PI with fatigue table
    dfm = metrics.loc[:, ["Horse", "PI", "PWR400"]].merge(
        _ftab.loc[:, ["Horse", "FatigueScore", "Tag"]],
        on="Horse",
        how="inner",
    )

    # numeric casting
    for c in ["PI", "PWR400", "FatigueScore"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    dfm = dfm.dropna(subset=["PWR400", "FatigueScore"])
    if dfm.empty:
        st.info("Not enough overlapping data to draw the Power–Freshness Map.")
    else:
        # ----- Axes -----
        # X: PWR400Δ = PWR400 - race median (late power vs field)
        pwr_med = float(np.nanmedian(dfm["PWR400"]))
        dfm["PWR400Δ"] = dfm["PWR400"] - pwr_med

        # Y: Freshness = -FatigueScore  (higher = stronger late / less fatigue)
        dfm["Freshness"] = -dfm["FatigueScore"]

        names = dfm["Horse"].astype(str).to_list()
        xv = dfm["PWR400Δ"].to_numpy()
        yv = dfm["Freshness"].to_numpy()

        # ----- Colour: map Tag → numeric, for a centred colour scale -----
        # late engine → +8, balanced/neutral → 0, front-spent → -8
        tag_map = {
            "late engine": 3.0,
            "balanced": 0.0,
            "neutral": 0.0,
            "front-spent": -3.0,
        }
        dfm["TagVal"] = dfm["Tag"].map(tag_map).fillna(0.0)
        cv = dfm["TagVal"].to_numpy()

        # ----- Dot sizes from PI (same logic as Shape Map) -----
        DOT_MIN, DOT_MAX = 40.0, 140.0
        piv = dfm["PI"].fillna(0).to_numpy()
        pmin, pmax = np.nanmin(piv), np.nanmax(piv)
        if (not np.isfinite(pmin)) or (not np.isfinite(pmax)) or pmin == pmax:
            sizes = np.full_like(xv, DOT_MIN)
        else:
            sizes = DOT_MIN + (piv - pmin) / (pmax - pmin + 1e-9) * (DOT_MAX - DOT_MIN)

        # ----- Limits & quadrant tints -----
        span = max(4.5, float(np.nanmax(np.abs(np.concatenate([xv, yv])))))
        lim = np.ceil(span / 1.5) * 1.5

        fig, ax = plt.subplots(figsize=(7.8, 6.2))

        TINT = 0.12
        # Quadrant meanings:
        #   Q1 (+X, +Y): Big engine & stays on  → "Monsters"
        #   Q2 (-X, +Y): Efficient grinders     → "Strong late, modest power"
        #   Q3 (+X, -Y): Fragile engines        → "Power but fatigued"
        #   Q4 (-X, -Y): Weak & spent          → "Dead zone"
        ax.add_patch(Rectangle((0, 0),       lim,  lim, facecolor="#4daf4a", alpha=TINT, zorder=0))
        ax.add_patch(Rectangle((-lim, 0),    lim,  lim, facecolor="#377eb8", alpha=TINT, zorder=0))
        ax.add_patch(Rectangle((0, -lim),    lim,  lim, facecolor="#ff7f00", alpha=TINT, zorder=0))
        ax.add_patch(Rectangle((-lim, -lim), lim,  lim, facecolor="#984ea3", alpha=TINT, zorder=0))

        ax.axvline(0, color="gray", lw=1.3, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0, color="gray", lw=1.3, ls=(0, (3, 3)), zorder=1)

        # Colour scale centred at 0 (balanced)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

        sc = ax.scatter(
            xv, yv,
            s=sizes,
            c=cv,
            cmap="coolwarm",
            norm=norm,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95,
        )

        # Labels
        label_points_neatly(ax, xv, yv, names)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("PWR400 − field median (late power under load) →")
        ax.set_ylabel("Freshness (−FatigueScore) ↑  — higher = stronger late / less fade")
        ax.set_title("Power–Freshness Map: late power vs fatigue  ·  Size = PI  ·  Colour = Fatigue Tag")

        # Size legend (PI bands)
        s_ex = [DOT_MIN, 0.5 * (DOT_MIN + DOT_MAX), DOT_MAX]
        h_ex = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=np.sqrt(s / np.pi),
                markeredgecolor="black",
            )
            for s in s_ex
        ]
        ax.legend(
            h_ex,
            ["PI low", "PI mid", "PI high"],
            loc="upper left",
            frameon=False,
            fontsize=8,
        )

        # Colourbar for fatigue tag
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fatigue Tag (−3 front-spent · 0 balanced/neutral · +3 late engine)")

        ax.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(fig)

        # PNG download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        powfresh_png = buf.getvalue()
        st.download_button(
            "Download Power–Freshness Map (PNG)",
            powfresh_png,
            file_name="power_freshness_map.png",
            mime="image/png",
        )
        st.caption(
            "X = PWR400 vs field (late power under load); "
            "Y = −FatigueScore (freshness, higher = stronger late); "
            "Colour = fatigue tag; Size = PI."
        )
# ======================= /Power–Freshness Map =======================

# ======================= R&V (CAR) — Context-Aware Reliability =======================
st.markdown("## R&V — Context-Aware Reliability (CAR)")

import numpy as np
import pandas as pd

df = metrics.copy()

# ---------- Small robust helpers ----------
def _nz(x, alt=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else alt
    except Exception:
        return alt

def _winsor(s: pd.Series, q: float = 0.02):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)

def _mad(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def _robust01(s: pd.Series):
    """Percentile rank 0–1 using median/MAD for stability."""
    s = pd.to_numeric(s, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return pd.Series(0.5, index=s.index)
    # rank to 0..1
    r = finite.rank(method="average", pct=True)
    out = pd.Series(np.nan, index=s.index)
    out.loc[finite.index] = r
    return out.fillna(0.5)

# ---------- If MV / RC not present, compute light versions from splits (exclude first) ----------
have_MV = "MV" in df.columns
have_RC = "RC" in df.columns

def _collect_time_cols(_work: pd.DataFrame):
    # Any '*_Time' plus optional 'Finish_Time'
    tcols = [c for c in _work.columns if c.endswith("_Time")]
    # order columns by numeric end marker; keep Finish at the end
    def _key(c):
        if c == "Finish_Time": return (10**9)
        try:
            return int(c.split("_")[0])
        except Exception:
            return 10**8
    tcols = sorted([c for c in tcols if c != "Finish_Time"], key=_key) + (["Finish_Time"] if "Finish_Time" in _work.columns else [])
    return tcols

if not (have_MV and have_RC):
    # Build per-horse segment speeds (m/s) from times.
    step = int(metrics.attrs.get("STEP", 100))
    tcols = _collect_time_cols(work)
    # exclude the first split to avoid start noise
    seg_cols = tcols[1:] if len(tcols) > 1 else tcols[:]
    # guard
    if len(seg_cols) >= 3:
        # Compute speeds per segment length
        seg_lens = []
        for c in seg_cols:
            if c == "Finish_Time":
                seg_lens.append(float(step))
            else:
                # infer length from column gap; safe default = step
                seg_lens.append(float(step))
        seg_lens = np.asarray(seg_lens, dtype=float)

        # Make a small speed matrix (rows=horses, cols=segments)
        # Join by horse where possible
        w = work.copy()
        if "Horse" in w.columns and "Horse" in df.columns:
            w = w.set_index("Horse").reindex(df["Horse"]).reset_index()

        spd = []
        for _, r in w.iterrows():
            times = pd.to_numeric(r[seg_cols], errors="coerce").to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                v = seg_lens / times
            # ignore 0/neg/inf
            v[~np.isfinite(v)] = np.nan
            spd.append(v)
        spd = np.asarray(spd, dtype=float)  # shape (N_horse, N_seg)

        # Trim to segments that have at least some data
        valid_mask = np.isfinite(spd).sum(axis=0) >= 2
        spd = spd[:, valid_mask]
        # After trim, require at least 3 segments
        if spd.shape[1] >= 3:
            # Fill nan per row with row-median to keep variability meaningful
            row_med = np.nanmedian(spd, axis=1, keepdims=True)
            spd_f = np.where(np.isfinite(spd), spd, row_med)

            # MV_raw: variability across segments (lower = steadier).
            # use coefficient of variation via MAD for robustness.
            med = np.nanmedian(spd_f, axis=1)
            mad = np.array([_mad(spd_f[i, :]) for i in range(spd_f.shape[0])])
            with np.errstate(divide="ignore", invalid="ignore"):
                mv_raw = mad / np.where(med > 0, med, np.nan)
            mv_raw = pd.Series(mv_raw, index=df.index)

            # RC_raw: rebound coefficient — late strength vs late trough.
            # Find trough over the last ~half of race, compare to mean of final 2 segments.
            H, S = spd_f.shape
            late_start = max(1, S // 2)
            trough = np.nanargmin(np.nanmedian(spd_f[:, late_start:], axis=0)) + late_start  # index in [late_start..S-1]
            tail = np.nanmean(spd_f[:, max(S-2, 0):S], axis=1)
            base = spd_f[:, trough]
            with np.errstate(divide="ignore", invalid="ignore"):
                rc_raw = tail / np.where(base > 0, base, np.nan)
            rc_raw = pd.Series(rc_raw, index=df.index)

            if not have_MV: df["MV"] = mv_raw
            if not have_RC: df["RC"] = rc_raw

# ---------- Core normalisations ----------
mv = _winsor(df.get("MV", pd.Series(np.nan, index=df.index)), q=0.02)
rc = _winsor(df.get("RC", pd.Series(np.nan, index=df.index)), q=0.02)

# Lower MV is better -> invert after normalising; Higher RC is better.
mv01 = 1.0 - _robust01(mv)     # steadiness (0..1, higher = steadier)
rc01 = _robust01(rc)           # rebound (0..1, higher = more kick late)

# Class lens: use PI if available, else GCI
if "PI" in df.columns and pd.to_numeric(df["PI"], errors="coerce").notna().any():
    cls = _winsor(pd.to_numeric(df["PI"], errors="coerce"), q=0.02)
else:
    cls = _winsor(pd.to_numeric(df.get("GCI", np.nan), errors="coerce"), q=0.02)

cls01 = _robust01(cls)

# ---------- Class-aware blending (CAR) ----------
# More tolerance for variability when class is high
mv_term = np.power(mv01, 1.0 - 0.35*cls01)            # elite horses keep more of their steadiness score
rc_term = rc01 * (1.0 + 0.25*cls01)                   # elite kick counts a bit extra

CAR = 10.0 * (0.45*mv_term + 0.35*rc_term + 0.20*cls01)
CAR = np.clip(CAR, 0.0, 10.0)

# ---------- Risk tag ----------
def _risk_tag(car):
    if not np.isfinite(car): return "None"
    if car >= 8.0:  return "Reliable"
    if car >= 6.0:  return "Mostly reliable"
    if car >= 4.0:  return "Variable"
    return "Fragile"

tags = [ _risk_tag(v) for v in CAR ]

# ---------- Output table ----------
out = pd.DataFrame({
    "Horse": df["Horse"],
    "R&V": np.round(CAR, 2),
    "Risk Tag": tags,
    "MV (↓ steadier)": np.round(pd.to_numeric(df.get("MV", np.nan), errors="coerce"), 3),
    "RC (↑ better rebound)": np.round(pd.to_numeric(df.get("RC", np.nan), errors="coerce"), 3),
})

# Add PI for context if present
if "PI" in df.columns:
    out["PI"] = np.round(pd.to_numeric(df["PI"], errors="coerce"), 2)
elif "GCI" in df.columns:
    out["GCI"] = np.round(pd.to_numeric(df["GCI"], errors="coerce"), 2)

# Sort by most reliable first, then strongest class
sort_cols = ["R&V"]
if "PI" in out.columns: sort_cols += ["PI"]
elif "GCI" in out.columns: sort_cols += ["GCI"]
out = out.sort_values(sort_cols, ascending=[False] + [False]*(len(sort_cols)-1), kind="mergesort").reset_index(drop=True)

st.dataframe(out, use_container_width=True)
st.caption("CAR = class-aware reliability: blends Steadiness (MV), Rebound (RC) and Class (PI/GCI). "
           "High CAR → repeatable pattern you can trust; low CAR → fragile/shape-dependent.")
# ======================= /R&V (CAR) =======================

# ======================= xWin — Probability to Win (100-replay view) =======================
st.markdown("## xWin — Probability to Win")

XW = metrics.copy()
gr_col = metrics.attrs.get("GR_COL", "Grind")
D_m    = float(race_distance_input)
RSI    = float(metrics.attrs.get("RSI", 0.0))        # + slow-early, − fast-early
SCI    = float(metrics.attrs.get("SCI", 0.0))        # 0..1 (shape consensus)
going  = str(metrics.attrs.get("GOING", "Good"))

# ---------- helpers ----------
def _clip(x, lo, hi):
    try:
        x = float(x); 
        return lo if x < lo else (hi if x > hi else x)
    except:
        return lo

def _lerp(a, b, t): 
    t = _clip(t, 0.0, 1.0)
    return a + (b - a) * t

def _winsor(s: pd.Series, p=0.02):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def _robust_z(s: pd.Series):
    """Median / MAD z-score; clipped to ±3 for stability."""
    x  = _winsor(pd.to_numeric(s, errors="coerce"))
    mu = np.nanmedian(x)
    sd = mad_std(x)
    if not np.isfinite(sd) or sd <= 0:
        z = (x - mu) / 1.0
    else:
        z = (x - mu) / sd
    return z.clip(-3.0, 3.0)

def _weights_for_distance(dm):
    """Distance-aware weights for Travel/Kick/Sustain (sum=1 before going tweak)."""
    dm = float(dm)
    knots = [
        (1000, dict(T=0.30, K=0.45, S=0.25)),   # sprints → K heavier
        (1200, dict(T=0.30, K=0.40, S=0.30)),
        (1400, dict(T=0.32, K=0.36, S=0.32)),
        (1600, dict(T=0.34, K=0.32, S=0.34)),
        (1800, dict(T=0.36, K=0.28, S=0.36)),
        (2000, dict(T=0.38, K=0.25, S=0.37)),
        (2400, dict(T=0.40, K=0.22, S=0.38)),   # staying → S heavier
    ]
    if dm <= knots[0][0]: return knots[0][1]
    if dm >= knots[-1][0]: return knots[-1][1]
    for (a_dm, a_w), (b_dm, b_w) in zip(knots, knots[1:]):
        if a_dm <= dm <= b_dm:
            t = (dm - a_dm) / (b_dm - a_dm)
            return {k: _lerp(a_w[k], b_w[k], t) for k in a_w}
    return knots[-1][1]

def _apply_going_nudge(w, going_str, field_n=12):
    """Small surface/going tweak; renormalises to 1."""
    w = w.copy()
    scale = min(1.0, max(1, int(field_n)) / 12.0)
    if going_str == "Firm":
        w["K"] *= (1.00 + 0.04*scale)
        w["T"] *= (1.00 + 0.02*scale)
        w["S"] *= (1.00 - 0.04*scale)
    elif going_str in ("Soft","Heavy"):
        amp = 0.05 if going_str == "Soft" else 0.08
        w["S"] *= (1.00 + amp*scale)
        w["T"] *= (1.00 + 0.02*scale)
        w["K"] *= (1.00 - amp*scale)
    S = sum(w.values()) or 1.0
    for k in w: w[k] /= S
    return w

def _temperature(N, accel, grind, dm, tfs_plus=None):
    """
    Race 'temperature' τ for softmax: lower = sharper probs (decisive ability gaps),
    higher = flatter probs (chaos, bunching, traffic).
    """
    N = max(1, int(N))

    def _mad01(s):
        s = pd.to_numeric(s, errors="coerce")
        d = mad_std(s)
        if not np.isfinite(d): return 0.0
        # ~ 1σ ~ 4–5 idx pts → map near 1.0
        return float(min(1.0, d / 4.5))

    d_ac  = _mad01(accel)
    d_gr  = _mad01(grind)
    base  = 0.95
    size_adj = -0.04*np.log1p(N)                    # bigger fields → lower τ
    disp_adj = -0.16*(0.5*d_ac + 0.5*d_gr)          # clear sectional separation → lower τ
    dist_adj = (0.04 if dm <= 1100 else (0.00 if dm <= 1800 else -0.02))
    tfs_adj  = 0.0
    if tfs_plus is not None:
        # more widespread friction → noisier replays → higher τ
        tp = pd.to_numeric(tfs_plus, errors="coerce").fillna(0.0)
        tfs_adj = 0.08 * float(np.clip(np.nanmean(np.maximum(0.0, (tp - 0.2)/0.4)), 0.0, 1.0))

    tau = base + size_adj + disp_adj + dist_adj + tfs_adj
    return float(_clip(tau, 0.55, 1.15))

def _to_fractional_odds(p):
    """p in [0,1] → 'x.y/1' (fair fractional style)."""
    try:
        p = float(p)
        if p <= 0: return "-"
        dec = 1.0 / p
        frac = dec - 1.0
        return f"{frac:.1f}/1"
    except:
        return "-"

# ---------- ensure inputs ----------
for c in ["tsSPI","Accel",gr_col,"F200_idx","PI"]:
    if c not in XW.columns: XW[c] = np.nan

# ---------- robust sectionals → within-race latent ability (z) ----------
zT = _robust_z(XW["tsSPI"])   # Travel
zK = _robust_z(XW["Accel"])   # Kick
zS = _robust_z(XW[gr_col])    # Sustain

# pace legitimacy guard (if race crawled mid, trim Travel influence a bit)
ts_med = pd.to_numeric(XW["tsSPI"], errors="coerce").median(skipna=True)
trim_T = 0.0
if np.isfinite(ts_med) and ts_med < 100.0:
    trim_T = min(0.20, max(0.0, (100.0 - ts_med) / 10.0))  # up to 20%
zT_eff = zT * (1.0 - trim_T)

# ---------- distance + going weights ----------
W = _weights_for_distance(D_m)               # {'T','K','S'}
W = _apply_going_nudge(W, going, field_n=len(XW))

# ---------- shape de-bias (KSI proxy) ----------
# Positive when horse ran AGAINST prevailing shape; negative when WITH shape
ksi_raw = -np.sign(RSI) * (pd.to_numeric(XW["Accel"], errors="coerce") - pd.to_numeric(XW["tsSPI"], errors="coerce"))
ksi01   = np.tanh((ksi_raw / 6.0).fillna(0.0))        # ~[-1..+1]
shape_boost = 0.15 * np.clip(ksi01, 0, 1) * SCI       # up to +15% (against)
shape_damp  = 0.08 * np.clip(-ksi01, 0, 1) * SCI      # up to −8%  (with)

# ---------- trip friction (from Hidden Horses if present) ----------
tfs_plus = None
try:
    if 'hh' in locals() and "TFS_plus" in hh.columns and "Horse" in XW.columns:
        tmp = hh[["Horse","TFS_plus"]].copy()
        XW = XW.merge(tmp, on="Horse", how="left")
        tfs_plus = pd.to_numeric(XW["TFS_plus"], errors="coerce").fillna(0.0)
except Exception:
    pass
if tfs_plus is None:
    tfs_plus = pd.Series(0.0, index=XW.index)

# harsher in sprints, softer in staying trips
if D_m <= 1400:   tfs_cap = 0.12
elif D_m >= 1800: tfs_cap = 0.08
else:             tfs_cap = _lerp(0.12, 0.08, (D_m-1400)/400.0)
tfs_pen = np.minimum(tfs_cap, np.maximum(0.0, (tfs_plus - 0.2)/0.4))

# ---------- core latent score (no history; pure one-run) ----------
# small optional stability from sectionals dispersion (SOS-like)
sos = (0.45*zT + 0.35*zK + 0.20*zS).fillna(0.0)
sos01 = ((sos - np.nanpercentile(sos, 5)) /
         max(1e-9, (np.nanpercentile(sos,95) - np.nanpercentile(sos,5))))
sos01 = sos01.clip(0, 1)

core = (
    W["T"] * zT_eff.fillna(0.0) +
    W["K"] * zK.fillna(0.0)     +
    W["S"] * zS.fillna(0.0)     +
    0.05   * sos01.fillna(0.0)  # very light stabiliser
)

# multiplicative de-lucking: reward against-shape, damp with-shape, damp friction
mult_adj = (1.0 + shape_boost - shape_damp) * (1.0 - tfs_pen)
power = (core * mult_adj).fillna(0.0)

# ---------- field-size shrink & temperature ----------
N    = int(len(XW.index))
tau  = _temperature(N, XW["Accel"], XW[gr_col], D_m, tfs_plus=tfs_plus)
alpha = N / (N + 6.0)              # small-field shrink (same motif you use elsewhere)
if N <= 6:
    power = 0.90 * power           # reduce overconfidence in tiny fields

# ---------- softmax → probabilities ----------
logits   = power / max(1e-6, tau)
mx       = float(np.nanmax(logits)) if np.isfinite(logits).any() else 0.0
exps     = np.exp((logits - mx).clip(-50, 50))
sum_exps = float(np.nansum(exps)) or 1.0
probs    = (exps / sum_exps) * alpha
probs    = probs / (probs.sum() or 1.0)     # renormalise after shrink

XW["xWin"] = probs

# ---------- tidy drivers ----------
def _driver_line(r):
    bits = []
    # early hint
    f200 = float(r.get("F200_idx", np.nan))
    if np.isfinite(f200):
        if f200 >= 101: bits.append("Quick early")
        elif f200 <= 98: bits.append("Slower away")

    # sectional pillars
    if float(zT_eff.get(r.name, 0)) >= 0.5: bits.append("Travel +")
    if float(zK.get(r.name,     0)) >= 0.5: bits.append("Kick ++")
    if float(zS.get(r.name,     0)) >= 0.5: bits.append("Sustain +")

    # shape cue (only if SCI decent)
    if SCI >= 0.6:
        k = float(ksi01.get(r.name, 0))
        if k > 0.35:   bits.append("Against shape")
        elif k < -0.35: bits.append("With shape")

    # midrace trim note
    if trim_T > 0: bits.append("(slow mid)")

    return " · ".join(bits)

XW["Drivers"] = XW.apply(_driver_line, axis=1)

# ---------- view ----------
view = XW.loc[:, ["Horse","xWin","Drivers"]].copy()
view["xWin"] = (100.0 * view["xWin"]).round(1)
view["Odds (≈fair)"] = XW["xWin"].apply(lambda p: _to_fractional_odds(p))

view = view.sort_values("xWin", ascending=False).reset_index(drop=True)

st.dataframe(
    view.style.format({"xWin": "{:.1f}%"}),
    use_container_width=True
)

with st.expander("xWin settings & notes"):
    w_note = ", ".join([f"{k}:{W[k]:.2f}" for k in ["T","K","S"]])
    st.caption(
        f"xWin = softmax of within-race latent ability (Travel/Kick/Sustain) with distance/going weights ({w_note}), "
        f"shape de-bias via RSI×SCI, trip friction damp, and a race 'temperature' τ={tau:.2f} from field size & dispersion. "
        f"Interpretation: chance to win if this same race were replayed 100 times."
    )
