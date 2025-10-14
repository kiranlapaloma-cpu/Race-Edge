# ======================= Batch 1 — Core + UI + I/O + DB bootstrap =======================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io, math, re, os, sqlite3, hashlib
from datetime import datetime

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
    """Validate only the real columns; no synthetic ‘missing’ list."""
    exp_cols = expected_segments_from_df(df)
    invalid_counts = {}
    for c in exp_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    bads = [f"{k} ({v} rows)" for k, v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs), [], invalid_counts

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
        base = {"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17}
    elif dm < 1100:
        base = _interp_weights(dm,
            1000, {"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17},
            1100, {"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20})
    elif dm < 1200:
        base = _interp_weights(dm,
            1100, {"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20},
            1200, {"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25})
    elif dm == 1200:
        base = {"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25}
    else:
        shift = max(0.0, (dm - 1200.0) / 100.0) * 0.01
        grind = min(0.25 + shift, 0.40)
        F200, ACC = 0.08, 0.30
        ts = max(0.0, 1.0 - F200 - ACC - grind)
        base = {"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":grind}

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

# ======================= Ahead of Handicap (Single-Race, Field-Aware) =======================
st.markdown("## Ahead of Handicap — Single Race Field Context")

AH = metrics.copy()
# Safety: ensure required columns exist
for c in ["PI","Accel",GR_COL,"Finish_Pos","Horse"]:
    if c not in AH.columns:
        AH[c] = np.nan

pi_rs = AH["PI"].clip(lower=0.0, upper=10.0)
med   = float(np.nanmedian(pi_rs))
sigma = mad_std(pi_rs - med)
sigma = 0.90 if (not np.isfinite(sigma) or sigma < 0.90) else sigma

# 2) Sample-size shrink for small/odd fields
N     = int(pi_rs.notna().sum())
alpha = N / (N + 6.0)  # → ~0.63 at N=10; ~0.4 at N=4

# 3) Field-shape influence (devoid of weight/handicap; uses only this race)
#    Reward balance late; softly reduce if very skewed (helps dampen fluky “shape gifts”).
bal          = 100.0 - (pd.to_numeric(AH["Accel"], errors="coerce") - pd.to_numeric(AH[GR_COL], errors="coerce")).abs() / 2.0
AH["_BAL"]   = bal
# Map BAL to a small multiplier in [0.92, 1.05]
AH["_FSI"]   = (0.92 + (AH["_BAL"] - 95.0) / 100.0).clip(0.92, 1.05)

# 4) z’s and final AHS 0–10 scale
z_raw        = (pi_rs - med) / sigma
AH["z_clean"]= alpha * z_raw
AH["z_FA"]   = AH["z_clean"] * AH["_FSI"]
AH["AHS"]    = (5.0 + 2.2 * AH["z_FA"]).clip(0.0, 10.0).round(2)

# 5) Tiers + Confidence (field size)
def ah_tier(v):
    if not np.isfinite(v): return ""
    if v >= 7.20: return "🏆 Dominant"
    if v >= 6.20: return "🔥 Clear Ahead"
    if v >= 5.40: return "🟢 Ahead"
    if v <= 4.60: return "🔻 Behind"
    return "⚪ Neutral"

def ah_conf(n):
    if n >= 12: return "High"
    if n >= 8:  return "Med"
    return "Low"

AH["AH_Tier"]       = AH["AHS"].map(ah_tier)
AH["AH_Confidence"] = ah_conf(N)

# 6) Display table (sorted by AHS then finish)
ah_cols = ["Horse","Finish_Pos","PI","AHS","AH_Tier","AH_Confidence","z_FA","_FSI"]
for c in ah_cols:
    if c not in AH.columns: AH[c] = np.nan

AH_view = AH.sort_values(["AHS","Finish_Pos"], ascending=[False, True])[ah_cols]
st.dataframe(AH_view, use_container_width=True)
st.caption("AH = Ahead-of-Handicap (single race). Centre = trimmed median PI; spread = MAD σ; FSI mildly rewards late balance.")
# ======================= End Ahead of Handicap =======================

# ======================= PI → Lengths Estimator (compact) =======================
st.markdown("---")
st.markdown("### PI → Lengths Estimator")

# Formula: lengths per 1.0 PI at distance D (m)
def lengths_per_pi(distance_m: float) -> float:
    # Baseline 3.5L @1000m, ~4.0L @1200m, grows ~linearly with distance
    L = 3.5 + 0.00125 * (float(distance_m) - 1000.0)
    return float(np.clip(L, 2.5, 8.5))  # soft safety clamp

colA, colB = st.columns([1,1])
with colA:
    D_for_pi = st.number_input("Distance for conversion (m)", min_value=800, max_value=3600,
                               value=int(race_distance_input), step=50)
with colB:
    dPI = st.number_input("ΔPI between horses (winner − rival)", value=1.0, step=0.1, format="%.1f")

L_per_PI = lengths_per_pi(D_for_pi)
est_margin = dPI * L_per_PI

# Friendly readout
lead_word = "ahead" if est_margin >= 0 else "behind"
st.metric(
    label="Estimated margin from PI gap",
    value=f"{abs(est_margin):.2f} lengths",
    delta=f"{L_per_PI:.2f} L per 1.0 PI @ {D_for_pi}m"
)
st.caption("Rule of thumb: ~4.0L/PI at 1200m, ~5.0L/PI at 1600m, ~6.0L/PI at 2000m (scaled linearly).")

# ======================= /PI → Lengths Estimator =======================

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
        ax.set_title("Quadrants: +X=Accel (400→200) · +Y="+("Corrected Grind" if USE_CG else "Grind")+" · Colour=tsSPIΔ")
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

# ======================= Pace Curve — field average (black) + Top 10 finishers =======================
st.markdown("## Pace Curve — field average (black) + Top 10 finishers")
pace_png = None

step = int(metrics.attrs.get("STEP", 100))
D    = float(race_distance_input)

# Build segments from the columns that actually exist.
# Each segment is (label, length_m, src_col). We purposely give every segment a UNIQUE KEY.
marks = _collect_markers(work)  # e.g. [1200,1100,1000,...] for a 1250 race (100m splits)

segs = []
if marks:
    # First leg: D → first marker (e.g., 1250→1200 uses 1200_Time)
    m1 = int(marks[0])
    L0 = max(1.0, D - m1)  # 50 for 1250→1200, 160 for 1160→1000, etc.
    if f"{m1}_Time" in work.columns:
        segs.append((f"{int(D)}→{m1}", float(L0), f"{m1}_Time"))

    # Middle legs: marker_i → marker_{i+1}
    # IMPORTANT: the time column for a→b is **b_Time** (the split ending at b).
    for a, b in zip(marks, marks[1:]):
        src = f"{int(b)}_Time"
        if src in work.columns:
            segs.append((f"{int(a)}→{int(b)}", float(a - b), src))

# Finish leg: always add Finish_Time if present
if "Finish_Time" in work.columns:
    fin_len = 100.0 if step == 100 else 200.0
    segs.append((f"{step}→0 (Finish)", fin_len, "Finish_Time"))

# Abort nicely if we still have nothing
if not segs:
    st.info("Not enough *_Time columns to draw the pace curve.")
else:
    # Compute speeds per segment into columns with UNIQUE KEYS
    seg_keys = [f"seg_{i}" for i in range(len(segs))]
    speed_df = pd.DataFrame(index=work.index, columns=seg_keys, dtype=float)

    for key, (label, L, src_col) in zip(seg_keys, segs):
        t = pd.to_numeric(work[src_col], errors="coerce")
        t = t.mask((t <= 0) | (~np.isfinite(t)))
        speed_df[key] = L / t  # NaN-safe; per-horse speed for this segment

    # Field average over horses for each segment (ignore all-NaN columns)
    field_avg = speed_df.mean(axis=0, skipna=True).to_numpy()
    if not np.isfinite(np.nanmean(field_avg)):
        st.info("Pace curve: all segments missing/invalid.")
    else:
        # Choose which horses to plot
        if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
            top10 = metrics.sort_values("Finish_Pos").head(10)
            top10_rule = "Top-10 by Finish_Pos"
        else:
            top10 = metrics.sort_values("PI", ascending=False).head(10)
            top10_rule = "Top-10 by PI"

        # X axis
        x_idx    = list(range(len(segs)))
        x_labels = [lbl for (lbl, _, _) in segs]

        fig2, ax2 = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
        ax2.plot(x_idx, field_avg, linewidth=2.2, color="black",
                 label="Field average", marker=None)

        palette = color_cycle(len(top10))
        for i, (_, r) in enumerate(top10.iterrows()):
            # find the row in the raw table for this horse (for timing columns)
            if "Horse" in work.columns and "Horse" in metrics.columns:
                row0 = work[work["Horse"] == r.get("Horse")]
                row_times = row0.iloc[0] if not row0.empty else r
            else:
                row_times = r

            y_vals = []
            for (_, L, src_col) in segs:
                t = pd.to_numeric(row_times.get(src_col, np.nan), errors="coerce")
                y = (L / float(t)) if (pd.notna(t) and float(t) > 0.0) else np.nan
                y_vals.append(y)

            if not np.isfinite(np.nanmean(y_vals)):
                continue

            ax2.plot(x_idx, y_vals, linewidth=1.1, marker="o", markersize=2.5,
                     label=str(r.get("Horse", "")), color=palette[i])

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(x_labels, rotation=45, ha="right")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Pace over segments (left = early; right = home straight; includes Finish)")
        ax2.grid(True, linestyle="--", alpha=0.30)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                   ncol=3, frameon=False, fontsize=9)

        st.pyplot(fig2)
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        pace_png = buf.getvalue()
        st.download_button("Download pace curve (PNG)", pace_png,
                           file_name="pace_curve.png", mime="image/png")
        st.caption(f"Top-10 plotted: {top10_rule}. Finish segment included explicitly.")

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

cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel",gr_col,"SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns: hh[c] = np.nan
hh_view = hh.sort_values(["Tier","HiddenScore","PI"], ascending=[True,False,False])[cols_hh]
st.dataframe(hh_view, use_container_width=True)
st.caption("Hidden Horses v2 — RS-aware tiering enabled · 🔥 ≥7.2/6.0 · 🟡 ≥6.2/5.0.")

# ======================= Ability Matrix v2 — Intrinsic vs Hidden Ability (Strict) =======================
st.markdown("---")
st.markdown("## Ability Matrix v2 — Intrinsic vs Hidden Ability (Strict)")

# Merge HiddenScore in
AM = metrics.copy()
if "Horse" not in AM.columns:
    AM["Horse"] = work.get("Horse", "")
AM = AM.merge(hh_view[["Horse","HiddenScore"]], on="Horse", how="left")
AM["HiddenScore"] = AM["HiddenScore"].fillna(0.0)

# Use corrected grind if active
gr_col = metrics.attrs.get("GR_COL", "Grind")

# ----- Core components -----
AM["IAI"]  = 0.35*AM["tsSPI"] + 0.25*AM["Accel"] + 0.25*AM[gr_col] + 0.15*AM["F200_idx"]
AM["BAL"]  = 100.0 - (AM["Accel"] - AM[gr_col]).abs() / 2.0
AM["COMP"] = 100.0 - (AM["tsSPI"] - 100.0).abs()

# ----- Percentiles within this race -----
def pct_rank(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0.0).clip(0.0, 1.0)

AM["IAI_pct"]  = pct_rank(AM["IAI"])
AM["HID_pct"]  = pct_rank(AM["HiddenScore"])
AM["BAL_pct"]  = 1.0 - pct_rank((AM["BAL"]  - 100.0).abs())
AM["COMP_pct"] = 1.0 - pct_rank((AM["COMP"] - 100.0).abs())

# ----- Hidden contribution caps (blocks hidden-only "elites") -----
def hidden_scale(iai):
    iai = float(iai) if pd.notna(iai) else np.nan
    if not np.isfinite(iai): return 0.0
    if iai < 101.0:  return 0.25     # average engines
    if iai < 101.5:  return 0.50     # decent engines
    return 1.00                      # strong engines

AM["_hid_scale"] = AM["IAI"].map(hidden_scale)

# ----- Composite score (for ordering/plot) -----
AM["AbilityScore"] = (
      6.5 * AM["IAI_pct"]
    + 2.5 * (AM["HID_pct"] * AM["_hid_scale"])
    + 0.6 * AM["BAL_pct"]
    + 0.4 * AM["COMP_pct"]
).clip(0.0, 10.0).round(2)

# ----- Confidence by field size -----
field_n = int(len(AM.index))
def conf_band(n):
    if n >= 12: return "High"
    if n >= 8:  return "Med"
    return "Low"
if "Confidence" not in AM.columns:
    AM["Confidence"] = conf_band(field_n)

# ----- Strict tier gates (Section E) -----
small_field = field_n <= 7
elite_iai_floor = 102.0 if small_field else 101.8
elite_pct_floor = 0.90  if small_field else 0.85

def in_range(x, lo, hi):
    x = float(x) if pd.notna(x) else np.nan
    return np.isfinite(x) and (lo <= x <= hi)

def to_float(x, default=np.nan):
    try:
        v = float(x);  return v if np.isfinite(v) else default
    except Exception:
        return default

def tier_for_row(r):
    iai      = to_float(r.get("IAI"))
    pi       = to_float(r.get("PI"))
    gci      = to_float(r.get("GCI"))
    iai_pct  = to_float(r.get("IAI_pct"), 0.0)
    bal      = to_float(r.get("BAL"))
    conf_ok  = str(r.get("Confidence","")).strip() in ("High","Med")

    # 🥇 Elite — all must pass
    if (
        np.isfinite(iai) and iai >= elite_iai_floor and
        np.isfinite(pi)  and pi  >= 7.2 and
        np.isfinite(gci) and gci >= 6.0 and
        iai_pct >= elite_pct_floor and
        in_range(bal, 98.0, 104.0) and
        conf_ok
    ):
        return "🥇 Elite"

    # 🥈 High — all must pass
    if (
        np.isfinite(iai) and iai >= 101.0 and
        np.isfinite(pi)  and pi  >= 6.2 and
        iai_pct >= 0.70 and
        in_range(bal, 97.0, 105.0)
    ):
        return "🥈 High"

    # 🥉 Competitive — any
    if (
        (np.isfinite(iai) and iai >= 100.4) or
        iai_pct >= 0.55 or
        (np.isfinite(pi) and pi >= 5.4)
    ):
        return "🥉 Competitive"

    return "⚪ Ordinary"

AM["AbilityTier"] = AM.apply(tier_for_row, axis=1)

# Near-Elite helper (passes 4/6 elite gates but not all)
def near_elite_row(r):
    iai, pi, gci, iai_pct, bal = map(to_float, (r.get("IAI"), r.get("PI"), r.get("GCI"),
                                                r.get("IAI_pct"), r.get("BAL")))
    conf_ok  = str(r.get("Confidence","")).strip() in ("High","Med")
    hits = 0
    hits += int(np.isfinite(iai) and iai >= elite_iai_floor)
    hits += int(np.isfinite(pi)  and pi  >= 7.2)
    hits += int(np.isfinite(gci) and gci >= 6.0)
    hits += int(iai_pct >= elite_pct_floor)
    hits += int(in_range(bal, 98.0, 104.0))
    hits += int(conf_ok)
    return "⭐ Near-Elite" if (hits >= 4 and r.get("AbilityTier") != "🥇 Elite") else ""

AM["NearEliteFlag"] = AM.apply(near_elite_row, axis=1)

# “Why this tier?” explainer
def why_tier_row(r):
    conf_ok = str(r.get("Confidence","")).strip() in ("High","Med")
    iai = to_float(r['IAI']); pi = to_float(r['PI']); gci = to_float(r['GCI'])
    iai_pct = to_float(r['IAI_pct']); bal = to_float(r['BAL'])
    return " · ".join([
        f"IAI {iai:.2f} {'✅' if np.isfinite(iai) and iai>=elite_iai_floor else '❌'}",
        f"PI {pi:.2f} {'✅' if np.isfinite(pi) and pi>=7.2 else '❌'}",
        f"GCI {gci:.2f} {'✅' if np.isfinite(gci) and gci>=6.0 else '❌'}",
        f"IAI_pct {iai_pct:.2f} {'✅' if iai_pct>=elite_pct_floor else '❌'}",
        f"BAL {bal:.1f} {'✅' if in_range(bal,98,104) else '❌'}",
        f"Conf {r.get('Confidence','')} {'✅' if conf_ok else '❌'}",
    ])

AM["WhyTier"] = AM.apply(why_tier_row, axis=1)

# ---------- Plot (IAI vs Hidden) ----------
try:
    ability_png
except NameError:
    ability_png = None

need_cols_am = {"Horse","IAI","HiddenScore","PI","BAL"}
if not need_cols_am.issubset(AM.columns):
    st.info("Ability Matrix: missing columns to plot.")
else:
    plot_df = AM.dropna(subset=["IAI","HiddenScore","PI","BAL"]).copy()
    if plot_df.empty:
        st.info("Not enough complete data to draw Ability Matrix.")
    else:
        x = plot_df["IAI"] - 100.0          # engine vs par
        y = plot_df["HiddenScore"]          # hidden 0..3
        sizes = 60.0 + (plot_df["PI"].clip(0,10) / 10.0) * 200.0

        # ---- Robust colour scale (BAL centered at 100) ----
        vals = pd.to_numeric(plot_df["BAL"], errors="coerce").to_numpy()
        vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
            vmin, vmax = 95.0, 105.0
        EPS = 1e-3
        if vmin >= 100.0: vmin = 100.0 - EPS
        if vmax <= 100.0: vmax = 100.0 + EPS
        norm = _safe_bal_norm(plot_df["BAL"], center=100.0)

        figA, axA = plt.subplots(figsize=(8.6, 6.0))
        sc = axA.scatter(
            x, y,
            s=sizes,
            c=plot_df["BAL"],
            cmap="coolwarm",
            norm=norm,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95
        )

        # label repel (defined earlier)
        label_points_neatly(axA, x.values, y.values, plot_df["Horse"].astype(str).tolist())

        axA.axvline(0.0, color="gray", lw=1.0, ls="--")
        axA.axhline(1.2, color="gray", lw=0.8, ls=":")
        axA.set_xlabel("Intrinsic Ability (IAI – 100)  →")
        axA.set_ylabel("HiddenScore (0–3)  ↑")
        axA.set_title("Ability Matrix v2 — Size = PI · Colour = BAL (100 = balanced late)")

        for s, lab in [(60, "PI low"), (160, "PI mid"), (260, "PI high")]:
            axA.scatter([], [], s=s, label=lab, color="gray", edgecolor="black")
        axA.legend(loc="upper left", frameon=False, fontsize=8, title="Point size:")

        cbar = figA.colorbar(sc, ax=axA, fraction=0.05, pad=0.04)
        cbar.set_label("BAL (100 = balanced late)")

        axA.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(figA)

        # Download image
        buf = io.BytesIO()
        figA.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        ability_png = buf.getvalue()
        st.download_button("Download Ability Matrix (PNG)", ability_png,
                           file_name="ability_matrix_v2.png", mime="image/png")

# ---------- Final table ----------
if "DirectionHint" not in AM.columns:
    def dir_hint_row(r):
        dv = float(r.get("Accel", np.nan)) - float(r.get(gr_col, np.nan))
        if not np.isfinite(dv): return ""
        if dv >= 2.0:  return "⚡ Sprint-lean (turn of foot)"
        if dv <= -2.0: return "🪨 Stayer-lean (sustained)"
        return "⚖ Balanced"
    AM["DirectionHint"] = AM.apply(dir_hint_row, axis=1)

am_cols = ["Horse","Finish_Pos","IAI","HiddenScore","BAL","COMP",
           "AbilityScore","AbilityTier","NearEliteFlag","WhyTier",
           "DirectionHint","Confidence","PI","GCI"]
for c in am_cols:
    if c not in AM.columns:
        AM[c] = np.nan

AM_view = AM.sort_values(
    ["AbilityTier","AbilityScore","PI","Finish_Pos"],
    ascending=[True, False, False, True]
)[am_cols]
st.dataframe(AM_view, use_container_width=True)

# ... end of Ability Matrix render (plot + AM_view table) ...

# ===================== CeilingCore v1.0 — One-Run Class Ceiling Engine =====================
# Replaces RIE/NRCI. Uses only single-race data: GCI (or GCI RS), RSI/SCI, sectionals, and within-race medians.
# No database. No wind. Shape-aware. Mass-based race class anchor.

import re, unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------- CONFIG (tweak here) ---------------------------
CONFIG = {
    # Class band thresholds (GCI -> band)
    "CLASS_BANDS": [
        (9.30, "🏆 Elite Group 1 calibre", 7),
        (8.50, "🥇 Group 1 calibre",       6),
        (7.80, "🥈 Group 2 calibre",       5),
        (7.00, "🥉 Group 3 calibre",       4),
        (6.30, "⚜️ Listed / Premier Handicap", 3),
        (5.30, "💠 Strong Handicapper",    2),
        (-1.0, "🔹 Low Handicapper",       1),
    ],
    # Shape adjustment weights (AgainstShape rewarded > WithShape penalised)
    "SHAPE_ALPHA": 0.60,
    "SHAPE_BETA":  0.40,
    "SHAPE_WEAKEN_IF_SCI_LT": 0.40,  # if SCI < this, halve ShapeAdj
    # Portability nudges (each capped to ±0.5 class)
    "EFF_GAMMA": 0.40,
    "TEN_DELTA": 0.30,
    "PORT_CAP":  0.50,
    "HALVE_SHAPE_IF_E_AND_T_LOW": 5.5,  # if Eff < and Ten < this, halve positive ShapeAdj
    # Dominance bonus (mass aware)
    "DOM_DELTA_CLASS_MIN": 1.0,
    "DOM_INDEX_MIN": 1.10,
    "DOM_WITHSHAPE_MAX": 0.25,
    "DOM_BONUS": 0.50,
    # Density taper: bonus *= clip(1 - 0.5*(IQR_width / 1.5), 0.6, 1.0)
    "DENSITY_TAPER_BASE": 0.5,
    "DENSITY_TAPER_MIN":  0.6,
    "DENSITY_TAPER_DEN":  1.5,
    # Reliability caps & fragility
    "UPSIDE_CAP_SOFT": 1.5,  # Reliability < 8.0
    "UPSIDE_CAP_HARD": 2.0,  # Reliability >= 8.0
    "FRAGILITY_K":     3.0,  # MAD multiple
    "FRAGILITY_R_GATE": 8.5,
    "FRAGILITY_HIT":   -0.3,
    # Confidence cutoffs
    "CONF_HIGH": 0.75,
    "CONF_MED":  0.60,
}

# --------------------------- Tiny helpers ---------------------------
def _clip01(x): 
    try: 
        return max(0.0, min(1.0, float(x)))
    except: 
        return 0.0

def _nz(x, alt=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else alt
    except Exception:
        return alt

# Normalize string → slug for robust joins (handles spaces, punctuation, unicode)
_ws = re.compile(r"\s+")
_nonword = re.compile(r"[^a-z0-9]+")

def _slug(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = _ws.sub(" ", s.lower().strip())
    return _nonword.sub("", s)

def _ensure_horsekey(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    out = df.copy()
    if "Horse" not in out.columns:
        # try to find a column that looks like 'Horse'
        cand = next((c for c in out.columns if _slug(c) == "horse"), None)
        if cand: out = out.rename(columns={cand: "Horse"})
        else:
            out["Horse"] = ""
    if "HorseKey" not in out.columns:
        out["HorseKey"] = out["Horse"].map(_slug)
    return out

def _find_finish_col(df: pd.DataFrame) -> str | None:
    if df is None: return None
    bad_tokens = ("grid", "gate", "stall", "box", "barrier", "draw", "700")
    def is_bad(name): 
        n = _slug(name)
        return any(tok in n for tok in bad_tokens)
    prefer = ["finish_pos","finishpos","finish","placing","position","pos","finish_posn"]
    nm = { _slug(c): c for c in df.columns if not is_bad(c) }
    for want in prefer:
        if want in nm: return nm[want]
    return None

def _finish_key(series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.round().astype("Int64")

def _gci_to_class(gci_value: float):
    g = _nz(gci_value, 0.0)
    for thr, label, idx in CONFIG["CLASS_BANDS"]:
        if g >= thr:
            return label, idx
    return "🔹 Low Handicapper", 1

def _map_series_to_class(s: pd.Series):
    labels, idxs = zip(*[_gci_to_class(v) for v in s.fillna(np.nan)])
    return list(labels), list(idxs)

# --------------------------- Robust GCI picker (GCI RS → GCI) ---------------------------
def _pick_gci_series(metrics_df: pd.DataFrame, view_df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Aligns by (HorseKey, Finish) if available; else HorseKey. 
    Priority: metrics['GCI RS'] → metrics['GCI'] → view['GCI RS'] → view['GCI'].
    Returns (Series aligned to view_df.index, source_tag)
    """
    rie = _ensure_horsekey(view_df)
    mdf = _ensure_horsekey(metrics_df) if metrics_df is not None else None

    rie_fin_col = _find_finish_col(rie)
    mdf_fin_col = _find_finish_col(mdf) if mdf is not None else None
    rie_has_fin = rie_fin_col is not None
    mdf_has_fin = (mdf is not None) and (mdf_fin_col is not None)

    def _pick(df, tag):
        if df is None: return None, None
        # choose column
        col = None
        if "GCI RS" in df.columns: col = "GCI RS"
        elif "GCI" in df.columns:  col = "GCI"
        else:
            lowmap = {c.lower(): c for c in df.columns}
            col = lowmap.get("gci rs") or lowmap.get("gcirs") or lowmap.get("gci")
        if not col: return None, None
        rhs = df[["HorseKey", col]].copy()
        rhs[col] = pd.to_numeric(rhs[col], errors="coerce")
        if tag == "metrics" and mdf_has_fin and rie_has_fin:
            rhs["FinishKey"] = _finish_key(df[mdf_fin_col])
            lhs = rie[["HorseKey"]].copy()
            lhs["FinishKey"] = _finish_key(rie[rie_fin_col])
            aligned = lhs.merge(rhs, on=["HorseKey","FinishKey"], how="left")[col]
            return aligned, f"{tag}:{col} (Horse+Finish)"
        else:
            aligned = rie[["HorseKey"]].merge(rhs, on="HorseKey", how="left")[col]
            return aligned, f"{tag}:{col} (Horse)"

    # Try metrics first
    s, src = _pick(mdf, "metrics")
    if s is not None and s.notna().any():
        return s, src

    # Then the view itself
    s, src = _pick(rie, "rie")
    if s is not None and s.notna().any():
        return s, src

    # Nothing — return NaNs
    return pd.Series(np.nan, index=view_df.index), "none"

# --------------------------- Race mass anchor (IQR-core median) ---------------------------
def _race_gci_mass(gci_all: pd.Series) -> tuple[float, float, int]:
    """
    Returns (RaceGCI_mass, IQR_width, n_used). 
    Primary: median on IQR core [Q25,Q75]. 
    Fallbacks: trimmed median for n<=5 (drop min & max), else Q20–Q80 trimmed median.
    """
    s = pd.to_numeric(gci_all, errors="coerce").dropna()
    n = len(s)
    if n == 0: return np.nan, 0.0, 0
    s_sorted = s.sort_values()
    q25, q75 = np.percentile(s_sorted, [25, 75])
    core = s_sorted[(s_sorted >= q25) & (s_sorted <= q75)]
    if len(core) >= 3:
        return float(np.median(core)), float(q75 - q25), int(len(core))
    # small-field fallback
    if n <= 5:
        s_trim = s_sorted.iloc[1:-1] if n >= 3 else s_sorted
        return float(np.median(s_trim)), float(q75 - q25), int(len(s_trim))
    # Q20–Q80 trimmed median
    q20, q80 = np.percentile(s_sorted, [20, 80])
    mid = s_sorted[(s_sorted >= q20) & (s_sorted <= q80)]
    return float(np.median(mid)), float(q75 - q25), int(len(mid))

# --------------------------- Reliability proxy (single run) ---------------------------
def _reliability_single(metrics_like: pd.DataFrame) -> pd.Series:
    # Completeness over core columns
    core_cols = ["F200_idx","tsSPI","Accel","Grind","Finish_Time"]
    present = metrics_like[core_cols].notna().sum(axis=1) / len(core_cols)
    # Late stability: small gap Accel vs Grind
    accel = pd.to_numeric(metrics_like.get("Accel"), errors="coerce")
    grind = pd.to_numeric(metrics_like.get("Grind"), errors="coerce")
    late_stab = 1.0 - (abs(accel - grind)/6.0).clip(0.0, 1.0)
    rel01 = (0.65*present + 0.35*late_stab).clip(0.0, 1.0)
    return 10.0*rel01

# --------------------------- CeilingCore main ---------------------------
def build_CeilingCore(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    # Ensure minimal columns
    if "Horse" not in df.columns: df["Horse"] = "(Unnamed)"
    # Sectionals & shape inputs
    tsSPI = pd.to_numeric(df.get("tsSPI"), errors="coerce")
    Accel = pd.to_numeric(df.get("Accel"), errors="coerce")
    Grind = pd.to_numeric(df.get("Grind", df.get("GRIND")), errors="coerce")
    DeltaG = pd.to_numeric(df.get("DeltaG"), errors="coerce")
    RSI   = _nz(df.attrs.get("RSI", df.get("RSI").iloc[0] if "RSI" in df.columns else np.nan), np.nan)
    SCI   = _nz(df.attrs.get("SCI", df.get("SCI").iloc[0] if "SCI" in df.columns else 0.5), 0.5)
    # Align GCI (GCI RS -> GCI)
    gci_series, gci_src = _pick_gci_series(metrics_df=df, view_df=df)

    # Race mass anchor
    race_gci_mass, iqr_width, n_core = _race_gci_mass(gci_series)
    race_label, race_idx = _gci_to_class(race_gci_mass)

    # Current class per horse
    labels_now, idxs_now = _map_series_to_class(gci_series)
    ClassBand_now = pd.Series(labels_now, index=df.index)
    ClassIndex_now = pd.Series(idxs_now, index=df.index)

    # Shape alignment (RSI direction vs late vs mid)
    dLM = (Accel - tsSPI)
    Align = np.tanh((np.sign(RSI) * (dLM/6.0)).fillna(0.0))  # keep in [-1,1] smoothly
    AgainstShape = (_clip01(-Align) * (0.6 + 0.4*_clip01(SCI)))
    WithShape    = (_clip01(+Align) * (0.6 + 0.4*_clip01(SCI)))
    # ShapeAdj (halve if low SCI)
    shape_alpha = CONFIG["SHAPE_ALPHA"]; shape_beta = CONFIG["SHAPE_BETA"]
    ShapeAdj = shape_alpha*AgainstShape - shape_beta*WithShape
    if SCI < CONFIG["SHAPE_WEAKEN_IF_SCI_LT"]:
        ShapeAdj = 0.5*ShapeAdj

    # Portability: Efficiency & Tenacity single-run proxies (0–10 → centered at 6)
    BAL  = 100.0 - (Accel - Grind).abs()/2.0
    COMP = 100.0 - (tsSPI - 100.0).abs()
    # center to medians inside the race
    bal_med  = float(pd.to_numeric(BAL, errors="coerce").median(skipna=True))
    comp_med = float(pd.to_numeric(COMP, errors="coerce").median(skipna=True))
    Eff01 = pd.to_numeric(BAL.map(lambda v: max(0.0, 1.0 - abs(v - bal_med)/12.0)), errors="coerce").fillna(0.0)
    Comp01= pd.to_numeric(COMP.map(lambda v: max(0.0, 1.0 - abs(v - comp_med)/12.0)), errors="coerce").fillna(0.0)
    Eff = 10.0*(0.6*Eff01 + 0.4*Comp01)  # 0..10
    press_core = 0.70*(Grind - 100.0).clip(lower=0.0) + 0.30*(DeltaG - 98.0).clip(lower=0.0)
    # rank-CDF-like within race for Tenacity
    Ten = 10.0 * press_core.rank(pct=True, method="average").fillna(0.0)

    EFF_GAMMA = CONFIG["EFF_GAMMA"]; TEN_DELTA = CONFIG["TEN_DELTA"]; PORT_CAP = CONFIG["PORT_CAP"]
    EffAdj = (EFF_GAMMA * ((Eff - 6.0)/4.0)).clip(-PORT_CAP, PORT_CAP)
    TenAdj = (TEN_DELTA * ((Ten - 6.0)/4.0)).clip(-PORT_CAP, PORT_CAP)
    # If both E and T are low, halve positive ShapeAdj
    mask_low_ET = (Eff < CONFIG["HALVE_SHAPE_IF_E_AND_T_LOW"]) & (Ten < CONFIG["HALVE_SHAPE_IF_E_AND_T_LOW"])
    ShapeAdj = np.where((mask_low_ET) & (ShapeAdj > 0), 0.5*ShapeAdj, ShapeAdj)

    # Dominance vs mass anchor (with density taper)
    DeltaClass_now = ClassIndex_now - race_idx
    DominanceIndex = pd.to_numeric(gci_series, errors="coerce") / _nz(race_gci_mass, np.nan)
    DOM_OK = (DeltaClass_now >= CONFIG["DOM_DELTA_CLASS_MIN"]) & (DominanceIndex >= CONFIG["DOM_INDEX_MIN"]) & (WithShape < CONFIG["DOM_WITHSHAPE_MAX"])
    taper = np.clip(1.0 - CONFIG["DENSITY_TAPER_BASE"] * (iqr_width / CONFIG["DENSITY_TAPER_DEN"]), CONFIG["DENSITY_TAPER_MIN"], 1.0)
    DomAdj = np.where(DOM_OK, CONFIG["DOM_BONUS"] * taper, 0.0)

    # Reliability (single run) & fragility
    tmp = pd.DataFrame({"F200_idx": df.get("F200_idx"), "tsSPI": tsSPI, "Accel": Accel, "Grind": Grind, "Finish_Time": df.get("Finish_Time")})
    Reliability = _reliability_single(tmp)
    is_outlier_high = False
    try:
        m = float(np.nanmedian(gci_series))
        mad = float(np.nanmedian(np.abs(gci_series - m))) or 0.0
        is_outlier_high = (pd.to_numeric(gci_series, errors="coerce") > (m + CONFIG["FRAGILITY_K"] * 1.4826 * mad))
    except Exception:
        is_outlier_high = False
    FragilityAdj = np.where(is_outlier_high & (Reliability < CONFIG["FRAGILITY_R_GATE"]), CONFIG["FRAGILITY_HIT"], 0.0)

    # Upside caps (apply to positive side of total adj)
    total_pos_adj = (ShapeAdj + EffAdj + TenAdj + DomAdj).clip(lower=0.0)
    cap = np.where(Reliability < 8.0, CONFIG["UPSIDE_CAP_SOFT"], CONFIG["UPSIDE_CAP_HARD"])
    capped_pos_adj = np.minimum(total_pos_adj, cap)
    # keep negative parts untouched
    total_neg_adj = (ShapeAdj + EffAdj + TenAdj + DomAdj).clip(upper=0.0)
    TotalAdj = capped_pos_adj + total_neg_adj + FragilityAdj

    # Final projection
    ProjectedIndex = (ClassIndex_now + TotalAdj).clip(1, 7)
    ProjectedBand = [_gci_to_class( idx if isinstance(idx, (int,float)) else float(idx) )[0] for idx in ProjectedIndex]

    # --- rebuild Conf defensively if not present ---
    if "Conf" not in locals():
        # Ability/pressure/efficiency coherence (0.5..1.0)
        Ability01  = pd.to_numeric(gci_series, errors="coerce").rank(pct=True, method="average").fillna(0.0)
        Pressure01 = press_core.rank(pct=True, method="average").fillna(0.0)
        Eff01_coh  = (Eff/10.0).clip(0.0, 1.0)
        core = np.minimum(Ability01, np.maximum(Pressure01, Eff01_coh))
        coherence = 0.5 + 0.5*core  # 0.5..1.0
        Conf = 0.5*coherence + 0.5*(Reliability/10.0)
    # Confidence buckets
    ConfLevel = np.where(Conf >= CONFIG["CONF_HIGH"], "High",
                  np.where(Conf >= CONFIG["CONF_MED"], "Medium", "Low"))

    # --- ensure vectors are Series aligned to df.index (avoid scalar collapse) ---
    def _ensure_series(x):
        if isinstance(x, pd.Series):
            return x.reindex(df.index)
        if isinstance(x, (np.ndarray, list, tuple)):
            return pd.Series(x, index=df.index)
        try:
            v = float(x)
            return pd.Series([v] * len(df), index=df.index)
        except Exception:
            return pd.Series([np.nan] * len(df), index=df.index)

    AgainstShape = _ensure_series(AgainstShape)
    WithShape    = _ensure_series(WithShape)
    Eff          = _ensure_series(Eff)
    Ten          = _ensure_series(Ten)
    DomAdj       = _ensure_series(DomAdj)
    FragilityAdj = _ensure_series(FragilityAdj)
    TotalAdj     = _ensure_series(TotalAdj)
    DOM_OK       = _ensure_series(DOM_OK).astype(bool)

    # Why line (concise, consistent)
    def _why(i):
        bits = []
        if AgainstShape.iloc[i] > 0.35:
            bits.append("ran against race shape")
        elif WithShape.iloc[i] > 0.35:
            bits.append("performance aligned with shape")
        if Eff.iloc[i] >= 6.3 and Ten.iloc[i] >= 6.3:
            bits.append("profile travels")
        elif Eff.iloc[i] < 5.5 or Ten.iloc[i] < 5.5:
            bits.append("set-up sensitive")
        if DOM_OK.iloc[i]:
            bits.append("dominant vs field mass")
        if FragilityAdj.iloc[i] < 0:
            bits.append("single-run spike (fragility cap)")
        if not bits:
            bits = ["solid single-run read"]
        tone = (
            "projects ½–1 up" if TotalAdj.iloc[i] >= 0.6
            else ("right level / ½-step" if TotalAdj.iloc[i] >= 0.2 else "right level")
        )
        return f"{'; '.join(bits)}; {tone}."

    Why = [_why(i) for i in df.index]

    out = pd.DataFrame({
        "Horse": df["Horse"].astype(str),
        "GCI_Value": pd.to_numeric(gci_series, errors="coerce").round(2),
        "GCI_Source": gci_src,
        "CurrentBand": ClassBand_now,
        "ClassIndex": ClassIndex_now.astype(float).round(2),
        "RaceGCI_mass": round(race_gci_mass, 2) if np.isfinite(race_gci_mass) else np.nan,
        "RaceIQR_Width": round(iqr_width, 2),
        "RaceClassIndex": float(race_idx),
        "DeltaClass": (ClassIndex_now - race_idx).astype(float).round(2),

        "AgainstShape": pd.Series(AgainstShape).round(2),
        "WithShape": pd.Series(WithShape).round(2),
        "ShapeAdj": pd.Series(ShapeAdj).round(2),
        "Eff": pd.Series(Eff).round(2),
        "Ten": pd.Series(Ten).round(2),
        "EffAdj": pd.Series(EffAdj).round(2),
        "TenAdj": pd.Series(TenAdj).round(2),
        "DomAdj": pd.Series(DomAdj).round(2),
        "FragilityAdj": pd.Series(FragilityAdj).round(2),

        "Reliability": pd.Series(Reliability).round(2),
        "TotalAdj": pd.Series(TotalAdj).round(2),

        "ProjectedIndex": pd.Series(ProjectedIndex).round(2),
        "ProjectedBand": pd.Series(ProjectedBand),
        "Confidence": pd.Series(ConfLevel),
        "Why": Why,
    })

    # Diagnostics caption
    try:
        matched = int(pd.to_numeric(gci_series, errors="coerce").notna().sum())
        st.caption(f"GCI source: {gci_src} · Matched {matched}/{len(df)} · Anchor: IQR-core median (n={n_core}, IQR={iqr_width:.2f}) → RaceClassIndex {race_idx}")
    except Exception:
        pass

    # Sort by projected ceiling
    return out.sort_values(["ProjectedIndex","GCI_Value"], ascending=[False, False]).reset_index(drop=True)

# --------------------------- Minimal Render ---------------------------
st.markdown("---")
st.markdown("## Race Intelligence — CeilingCore (Single-Run Class Ceiling)")
try:
    ceiling_view = build_CeilingCore(metrics)
    show_cols = [
        "Horse","GCI_Value","CurrentBand",
        "RaceGCI_mass","RaceClassIndex","DeltaClass",
        "ProjectedBand","ProjectedIndex","Confidence","Why"
    ]
    st.dataframe(ceiling_view[show_cols], use_container_width=True, hide_index=True)
    st.caption("CeilingCore projects each horse's next-run class ceiling using GCI, race shape (RSI/SCI), sectional balance, and a mass-based race anchor — no history required.")
except Exception as e:
    st.error("CeilingCore failed.")
    st.exception(e)
# ===================== /CeilingCore v1.0 =====================
# ======== PDF Export v1 (drop-in) ========
import io, math
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import mm
    _e_pdf = None
except Exception as e:
    _e_pdf = e

def _fmt(x, nd=2):
    try:
        v = float(x)
        if not math.isfinite(v): return ""
        return f"{v:.{nd}f}"
    except Exception:
        return str(x) if x is not None else ""

def _make_table(df, cols, max_rows=12):
    """Return a ReportLab Table from a pandas dataframe (safe, short)."""
    if df is None or len(df) == 0:
        return None
    use = [c for c in cols if c in df.columns]
    if not use:
        return None
    sub = df[use].head(max_rows).copy()
    # format numerics lightly
    for c in use:
        if str(c).lower() == "horse": continue
        sub[c] = sub[c].map(lambda v: _fmt(v))
    data = [use] + sub.values.tolist()
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#34495e")),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("FONT", (0,1), (-1,-1), "Helvetica", 8),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#bdc3c7")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.black),
    ]))
    return tbl

def build_race_pdf(metrics,
                   sectional_df=None,
                   hh_view=None,
                   wd_view=None,
                   dna_summaries=None,
                   images=None,
                   race_distance_m=None,
                   split_step=None) -> bytes:
    """
    Build a compact PDF and return bytes.
    Expected:
      - metrics.attrs used for RSI/SCI/SHAPE_TAG/GOING/RQS/RPS/RACE_PROFILE
      - sectional_df: your 'display_df' table (Sectional Metrics)
      - hh_view: Hidden Horses v2 table
      - wd_view: Winning DNA table
      - dna_summaries: list of strings (per-horse one-liners), optional
      - images: dict with {'shape': bytes, 'pace': bytes, 'ability': bytes} optional
    """
    if 'reportlab' in str(_e_pdf).lower():
        raise RuntimeError("reportlab is not installed. Add 'reportlab' to requirements.txt")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm
    )

    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]; H1.fontSize = 16; H1.spaceAfter = 6
    H2 = styles["Heading2"]; H2.fontSize = 13; H2.spaceBefore = 6; H2.spaceAfter = 4
    H3 = styles["Heading3"]; H3.fontSize = 11; H3.spaceBefore = 6; H3.spaceAfter = 2
    P  = styles["BodyText"]; P.fontSize = 9.2; P.leading = 12

    story = []

    # ---------- Header ----------
    Dm   = int(float(race_distance_m or metrics.attrs.get("DIST", 0)) or 0)
    step = int(split_step or metrics.attrs.get("STEP", 100))
    rsi  = float(metrics.attrs.get("RSI", 0.0))
    sci  = float(metrics.attrs.get("SCI", 0.0))
    tag  = str(metrics.attrs.get("SHAPE_TAG", "EVEN"))
    going = str(metrics.attrs.get("GOING", "Good"))
    rqs  = metrics.attrs.get("RQS", None)
    rps  = metrics.attrs.get("RPS", None)
    prof = str(metrics.attrs.get("RACE_PROFILE", ""))

    story.append(Paragraph(f"Race Edge Report", H1))
    hdr_line = f"Distance: <b>{Dm} m</b>  |  Splits: <b>{step}m</b>  |  Shape: <b>{tag}</b>  |  RSI: <b>{rsi:+.2f}</b>  |  SCI: <b>{sci:.2f}</b>  |  Going: <b>{going}</b>"
    story.append(Paragraph(hdr_line, P))
    meta_line = []
    if rqs is not None: meta_line.append(f"RQS <b>{float(rqs):.1f}</b>/100")
    if rps is not None: meta_line.append(f"RPS <b>{float(rps):.1f}</b>/100")
    if prof:            meta_line.append(f"Profile: <b>{prof}</b>")
    if meta_line:
        story.append(Paragraph(" · ".join(meta_line), P))
    story.append(Spacer(1, 6))

    # PI going meta (if available)
    pi_meta = metrics.attrs.get("PI_GOING_META", {})
    if pi_meta:
        moved = []
        mult = pi_meta.get("multipliers", {})
        for k in ["Accel","F200_idx","tsSPI","Grind"]:
            m = mult.get(k, 1.0)
            if abs(m-1.0) >= 0.005:
                moved.append(f"{k}×{m:.3f}")
        if moved:
            story.append(Paragraph("PI going multipliers: " + ", ".join(moved), P))
            story.append(Spacer(1, 6))

    # ---------- Tables ----------
    story.append(Paragraph("Sectional Metrics — snapshot", H2))
    sec_cols = ["Horse","PI","GCI_RS","F200_idx","tsSPI","Accel",metrics.attrs.get("GR_COL","Grind")]
    tbl = _make_table(sectional_df, sec_cols, max_rows=12)
    story.append(tbl if tbl else Paragraph("No data", P))
    story.append(PageBreak())
                              
    story.append(Spacer(1, 8))
    story.append(Paragraph("Hidden Horses v2 — signals", H2))
    hh_cols = ["Horse","Tier","HiddenScore","SOS","ASI2","TFS","UEI"]
    tbl = _make_table(hh_view, hh_cols, max_rows=12)
    story.append(tbl if tbl else Paragraph("No data", P))
    story.append(PageBreak())

    story.append(Spacer(1, 8))
    story.append(Paragraph("Winning DNA — top factors", H2))
    wd_cols = ["Horse","WinningDNA","EZ01","MC01","LP01","LL01","SOS01"]
    tbl = _make_table(wd_view, wd_cols, max_rows=12)
    story.append(tbl if tbl else Paragraph("No data", P))
    
    # ---------- Race Pulse (one-liners) ----------
    if dna_summaries and len(dna_summaries) > 0:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Race Pulse — summaries", H2))
        for line in dna_summaries[:14]:
            story.append(Paragraph("• " + line, P))

    # ---------- Images (optional) ----------
    img_keys = ["shape","pace","ability"]
    have_any = images and any(k in images and images[k] for k in img_keys)
    if have_any:
        story.append(PageBreak())
        story.append(Paragraph("Figures", H2))
        for label, key in [("Sectional Shape Map", "shape"),
                           ("Pace Curve", "pace"),
                           ("Ability Matrix", "ability")]:
            if not images.get(key): 
                continue
            try:
                img = Image(io.BytesIO(images[key]))
                # ✅ scale dynamically to fit A4 height (~250 mm usable)
                max_w, max_h = 170*mm, 120*mm
                img.drawWidth = min(img.imageWidth, max_w)
                ratio = img.drawWidth / float(img.imageWidth)
                img.drawHeight = min(img.imageHeight * ratio, max_h)

                story.append(Paragraph(label, H3))
                story.append(img)
                story.append(PageBreak())
            except Exception:
                # skip bad image
                pass

    doc.build(story)
    return buf.getvalue()

# ---- Streamlit hook (place this block where you want the button) ----
try:
    if st.button("Generate PDF report"):
        if '_e_pdf' in globals() and isinstance(_e_pdf, Exception):
            st.error("reportlab missing. Add it to requirements.txt.")
        else:
            # Collect optional text lines from Winning DNA expander if you created them
            try:
                dna_lines = (WD.sort_values("WinningDNA", ascending=False)["DNA_Summary"]
                               .dropna().astype(str).tolist())
            except Exception:
                dna_lines = []

            # Collect images if present
            imgs = {}
            for k, var in [("shape", "shape_map_png"), ("pace", "pace_png"), ("ability", "ability_png")]:
                try:
                    if var in globals() and globals()[var] is not None:
                        imgs[k] = globals()[var]
                except Exception:
                    pass

            # Build and download
            pdf_bytes = build_race_pdf(
                metrics,
                sectional_df=display_df if 'display_df' in globals() else metrics,
                hh_view=hh_view if 'hh_view' in globals() else None,
                wd_view=WD_view if 'WD_view' in globals() else (WD if 'WD' in globals() else None),
                dna_summaries=dna_lines,
                images=imgs,
                race_distance_m=race_distance_input,
                split_step=split_step
            )
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"race_edge_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
except Exception as e:
    st.error(f"PDF generation failed: {e}")
# ======== /PDF Export v1 ========
