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
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

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
  shape_tag      TEXT,         -- NEW: SED tag e.g., SLOW_EARLY, EVEN, FAST_EARLY
  sci            REAL,         -- NEW: Shape Consistency Index (0..1)
  fra_applied    INTEGER,      -- NEW: flag if False-Run Adjustment was used (0/1)
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

# ----------------------- Integrity helpers (used later) -------------------
def expected_segments(distance_m: float, step:int) -> list[str]:
    want = [f"{m}_Time" for m in range(int(distance_m) - step, step-1, -step)]
    want.append("Finish_Time")  # For 200m, this is the 200→0 split
    return want

def integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    exp_cols = expected_segments(distance_m, step)
    missing = [c for c in exp_cols if c not in df.columns]
    invalid_counts = {}
    for c in exp_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    if missing:
        msgs.append("Missing: " + ", ".join(missing))
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
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
# (Drop-in replacement)

import math
import numpy as np
import pandas as pd

# -------- Stage helpers (100m/200m aware) --------
def collect_markers(df):
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    return sorted(set(marks), reverse=True)

def sum_times(row, cols):
    vals = [as_num(row.get(c)).item() if hasattr(as_num(row.get(c)), "item") else as_num(row.get(c)) for c in cols]
    vals = [v for v in vals if pd.notna(v) and v > 0]
    return np.sum(vals) if len(vals) else np.nan

def make_range_cols(D, start_inclusive, end_inclusive, step):
    """Return columns for D→… decreasing by step (inclusive bounds given in meters)."""
    if start_inclusive < end_inclusive:
        return []
    want = list(range(int(start_inclusive), int(end_inclusive) - 1, -step))
    return [f"{m}_Time" for m in want]

def stage_speed(row, cols, meters_per_split):
    if not cols: return np.nan
    tsum = sum_times(row, cols)
    if pd.isna(tsum) or tsum <= 0: return np.nan
    valid = [c for c in cols if pd.notna(row.get(c)) and as_num(row.get(c)) > 0]
    dist = meters_per_split * len(valid)
    if dist <= 0: return np.nan
    return dist / tsum

def grind_speed(row, step):
    """Grind = last 100 + finish (for 100m data) OR finish split only (for 200m data)."""
    if step == 100:
        t100 = as_num(row.get("100_Time"))
        tfin = as_num(row.get("Finish_Time"))
        parts = []; dist = 0.0
        if pd.notna(t100) and t100 > 0: parts.append(float(t100)); dist += 100.0
        if pd.notna(tfin) and tfin > 0: parts.append(float(tfin)); dist += 100.0
        if len(parts) == 0 or dist <= 0: return np.nan
        return dist / sum(parts)
    else:
        tfin = as_num(row.get("Finish_Time"))
        if pd.isna(tfin) or tfin <= 0:
            return np.nan
        return 200.0 / float(tfin)

# -------- Distance + context weights for PI v3.x --------
def _lerp(a, b, t):
    return a + (b - a) * float(t)

def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    dm = float(distance_m or 1200)
    if dm <= 1000:
        base = {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17}
    elif dm < 1100:
        base = _interpolate_weights(
            dm,
            1000, {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17},
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20}
        )
    elif dm < 1200:
        base = _interpolate_weights(
            dm,
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20},
            1200, {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
        )
    elif dm == 1200:
        base = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
    else:
        shift_units = max(0.0, (dm - 1200.0) / 100.0) * 0.01
        grind = min(0.25 + shift_units, 0.40)
        F200, ACC = 0.08, 0.30
        ts = max(0.0, 1.0 - F200 - ACC - grind)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":grind}

    # Context tweak based on within-race bias (Accel vs Grind medians)
    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)          # sensitivity curve
        max_shift = 0.02 * scale
        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]
        if bias > 0:           # acceleration rewarded → rebalance a touch toward Grind
            delta = min(max_shift, ACC - 0.26)
            ACC -= delta; GR += delta
        elif bias < 0:         # grinding rewarded → swing a hair back to Accel
            delta = min(max_shift, GR - 0.18)
            GR  -= delta; ACC += delta
        GR = min(GR, 0.40)
        ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}
    return base

# -------- Metric builder (handles 100m and 200m) --------
def build_metrics_and_shape(df_in: pd.DataFrame,
                            D_actual_m: float,
                            step: int,
                            use_cg: bool,
                            dampen_cg: bool,
                            use_race_shape: bool,
                            debug: bool):
    w = df_in.copy()

    # Normalize finish pos
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    seg_markers = collect_markers(w)

    # Per-segment speeds
    for m in seg_markers:
        w[f"spd_{m}"] = (step * 1.0) / as_num(w.get(f"{m}_Time"))
    w["spd_Finish"] = (100.0 if step == 100 else 200.0) / as_num(w.get("Finish_Time")) if "Finish_Time" in w.columns else np.nan

    # Race time (sum segments incl. finish)
    if len(seg_markers) > 0:
        wanted = list(range(int(D_actual_m) - step, step-1, -step))
        cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in w.columns]
        if "Finish_Time" in w.columns:
            cols = cols + ["Finish_Time"]
        w["RaceTime_s"] = w[cols].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0, np.nan).sum(axis=1)
    else:
        w["RaceTime_s"] = as_num(w.get("Race Time", np.nan))

    # ---------- Build stage composite speeds ----------
    D = float(D_actual_m)

    # F200
    if step == 100:
        f200_cols = [c for c in [f"{int(D-100)}_Time", f"{int(D-200)}_Time"] if c in w.columns]
        f200_dist = 100.0 * len(f200_cols)
    else:
        c = f"{int(D-200)}_Time"
        f200_cols = [c] if c in w.columns else []
        f200_dist = 200.0 * len(f200_cols)
    w["_F200_spd"] = w.apply(lambda r: (f200_dist / sum_times(r, f200_cols)) if len(f200_cols)>=1 and pd.notna(sum_times(r, f200_cols)) and sum_times(r, f200_cols)>0 else np.nan, axis=1)

    # tsSPI: (D-300)…600 for 100m; (D-400)…600 for 200m
    tssp_start = int(D - (300 if step == 100 else 400))
    tssp_cols = make_range_cols(D, tssp_start, 600, step)
    tssp_cols = [c for c in tssp_cols if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: stage_speed(r, tssp_cols, float(step)), axis=1)

    # Accel: 600→200
    if step == 100:
        accel_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]
    else:
        accel_cols = [c for c in [f"{m}_Time" for m in [600,400]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: stage_speed(r, accel_cols, float(step)), axis=1)

    # Grind (finish)
    w["_GR_spd"] = w.apply(lambda r: grind_speed(r, step), axis=1)

    # ---------- Map speeds → indices (robust, small-field stabilizers) ----------
    def shrink_center(idx_series):
        x = idx_series.dropna().values
        N_eff = len(x)
        if N_eff == 0:
            return 100.0, 0
        med_race = float(np.median(x))
        alpha = N_eff / (N_eff + 6.0)
        return alpha * med_race + (1 - alpha) * 100.0, N_eff

    def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
        gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
        return delta_series * min(gamma, cap)

    def variance_floor(idx_series, floor=1.5, cap=1.25):
        deltas = idx_series - 100.0
        sigma = mad_std(deltas)
        if not np.isfinite(sigma) or sigma <= 0:
            return idx_series
        if sigma < floor:
            factor = min(cap, floor / sigma)
            return 100.0 + deltas * factor
        return idx_series

    def speed_to_index(spd_series):
        med = spd_series.median(skipna=True)
        idx_raw = 100.0 * (spd_series / med)
        center, n_eff = shrink_center(idx_raw)
        idx = 100.0 * (spd_series / (center / 100.0 * med))
        idx = 100.0 + dispersion_equalizer(idx - 100.0, n_eff)
        idx = variance_floor(idx)
        return idx

    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F200_spd"], errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"],  errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"],  errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],   errors="coerce"))

    # ---------- Corrected Grind (CG) ----------
    ACC_field = pd.to_numeric(w["_ACC_spd"], errors="coerce").mean(skipna=True)
    GR_field  = pd.to_numeric(w["_GR_spd"],  errors="coerce").mean(skipna=True)
    FSR = float(GR_field / ACC_field) if (ACC_field and ACC_field > 0 and math.isfinite(ACC_field) and math.isfinite(GR_field)) else np.nan
    if not math.isfinite(FSR):
        FSR = 1.0
    CollapseSeverity = float(min(10.0, max(0.0, (0.95 - FSR) * 100.0)))  # index points

    def delta_g_row(r):
        mid = float(r.get("_MID_spd", np.nan))
        grd = float(r.get("_GR_spd",  np.nan))
        if not (math.isfinite(mid) and math.isfinite(grd) and mid > 0):
            return np.nan
        return 100.0 * (grd / mid)
    w["DeltaG"] = w.apply(delta_g_row, axis=1)

    def finisher_factor_row(dg):
        if not math.isfinite(dg): return 0.0
        return float(clamp((dg - 98.0) / 4.0, 0.0, 1.0))
    w["FinisherFactor"] = w["DeltaG"].map(finisher_factor_row)
    w["GrindAdjPts"] = (CollapseSeverity * (1.0 - w["FinisherFactor"])).round(2)

    # Subtract adj (cap fades for big late fades)
    w["Grind_CG"] = (w["Grind"] - w["GrindAdjPts"]).clip(lower=90.0, upper=110.0)
    def fade_cap(g, dg):
        if not math.isfinite(g) or not math.isfinite(dg): return g
        if dg < 97.0 and g > 100.0:
            return 100.0 + 0.5 * (g - 100.0)
        return g
    w["Grind_CG"] = [fade_cap(g, dg) for g, dg in zip(w["Grind_CG"], w["DeltaG"])]

    # ---------- PI v3.2 (pre Race Shape) ----------
    acc_med = w["Accel"].median(skipna=True)
    grd_med = (w["Grind_CG"] if use_cg else w["Grind"]).median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(D), acc_med, grd_med)

    # Optional dampen on collapse
    if use_cg and dampen_cg and CollapseSeverity >= 3.0:
        d = min(0.02 + 0.01 * (CollapseSeverity - 3.0), 0.08)
        shift = min(d, PI_W["Grind"])
        PI_W["Grind"] -= shift
        PI_W["Accel"] += shift * 0.5
        PI_W["tsSPI"] += shift * 0.5

    GR_COL = "Grind_CG" if use_cg else "Grind"

    def pi_pts_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(GR_COL) if k == "Grind" else row.get(k, np.nan)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        if not weights: return np.nan
        return sum(parts) / sum(weights)

    w["PI_pts"] = w.apply(pi_pts_row, axis=1)
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75:
        sigma = 0.75
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

    # ---------- GCI (pre Race Shape; 0–10) ----------
    acc_med_g = w["Accel"].median(skipna=True)
    grd_med_g = (w["Grind_CG"] if use_cg else w["Grind"]).median(skipna=True)
    Wg = pi_weights_distance_and_context(float(D), acc_med_g, grd_med_g)

    wT   = 0.25
    wPACE= Wg["Accel"] + Wg["Grind"]
    wSS  = Wg["tsSPI"]
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try:
            winner_time = float(w["RaceTime_s"].min())
        except Exception:
            winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

    gci_vals = []
    for _, r in w.iterrows():
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"]) - winner_time
            if d <= 0.30:   T = 1.0
            elif d <= 0.60: T = 0.7
            elif d <= 1.00: T = 0.4
            else:           T = 0.2

        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get(GR_COL))
        SS = map_pct(r.get("tsSPI"))

        acc, grd_eff = r.get("Accel"), r.get(GR_COL)
        if pd.isna(acc) or pd.isna(grd_eff):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd_eff - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)

        score01 = (wT * T) + (wPACE * LQ) + (wSS * SS) + (wEFF * EFF)
        gci_vals.append(round(10.0 * score01, 3))

    w["GCI"] = gci_vals

    # ---------- RACE SHAPE MODULE v2.2 (conservative SLOW/FAST + SCI) ----------
    shape_tag   = "EVEN"
    sci         = 1.0
    fra_applied = 0

    if use_race_shape:
        # Ensure EARLY/LATE indices ALWAYS exist for the table
        if step == 100:
            early_markers = [int(D - 100), int(D - 200)]
        else:
            early_markers = [int(D - 200)]
        early_cols = [f"{m}_Time" for m in early_markers if f"{m}_Time" in w.columns]

        if step == 100:
            late_cols = [c for c in [f"{m}_Time" for m in [500, 400, 300, 200]] if c in w.columns]
        else:
            late_cols = [c for c in [f"{m}_Time" for m in [600, 400]] if c in w.columns]

        w["_EARLY_spd"] = w.apply(lambda r: stage_speed(r, early_cols, float(step)) if early_cols else np.nan, axis=1)
        w["_LATE_spd"]  = w.apply(lambda r: stage_speed(r, late_cols,  float(step)) if late_cols  else np.nan, axis=1)

        w["EARLY_idx"] = speed_to_index(pd.to_numeric(w["_EARLY_spd"], errors="coerce"))
        w["LATE_idx"]  = speed_to_index(pd.to_numeric(w["_LATE_spd"],  errors="coerce"))

        E_med = float(pd.to_numeric(w["EARLY_idx"], errors="coerce").median(skipna=True))
        M_med = float(pd.to_numeric(w["tsSPI"],     errors="coerce").median(skipna=True))
        L_med = float((0.6*pd.to_numeric(w["Accel"], errors="coerce") +
                       0.4*pd.to_numeric(w[GR_COL],   errors="coerce")).median(skipna=True))

        dE, dM, dL = (E_med - 100.0), (M_med - 100.0), (L_med - 100.0)

        def _gate(series, base=2.2):
            z = pd.to_numeric(series, errors="coerce") - 100.0
            s = mad_std(z)
            s = 2.0 if (not np.isfinite(s) or s <= 0) else s
            return max(base, 0.6*s)

        gE = _gate(w["EARLY_idx"])
        gL = _gate(0.6*w["Accel"] + 0.4*w[GR_COL])

        if   D <= 1200: scale = 1.00
        elif D <  1800: scale = 1.10
        else:           scale = 1.20
        gE *= scale; gL *= scale

        delta_EL = (pd.to_numeric(w["LATE_idx"], errors="coerce") -
                    pd.to_numeric(w["EARLY_idx"], errors="coerce"))
        sci_plus  = float((delta_EL >  +1.0).mean()) if delta_EL.notna().any() else np.nan
        sci_minus = float((delta_EL <  -1.0).mean()) if delta_EL.notna().any() else np.nan

        fsr_val = float(FSR) if np.isfinite(FSR) else np.nan
        confirm_slow = (np.isfinite(fsr_val) and fsr_val >= 1.03)
        confirm_fast = (np.isfinite(fsr_val) and fsr_val <= 0.97)

        slow_early = (dE <= -gE) and (dL >= +gL) and ((dL - dE) >= 3.5) \
                     and (sci_plus >= 0.55 if np.isfinite(sci_plus) else True) \
                     and (99.0 <= M_med <= 101.8)
        fast_early = (dE >= +gE) and (dL <= -gL) and ((dE - dL) >= 3.5) \
                     and (sci_minus >= 0.55 if np.isfinite(sci_minus) else True) \
                     and (98.2 <= M_med <= 101.8)

        if slow_early and confirm_slow:
            shape_tag = "SLOW_EARLY"
        elif fast_early and confirm_fast:
            shape_tag = "FAST_EARLY"
        else:
            if slow_early and (sci_plus >= 0.65):
                shape_tag = "SLOW_EARLY"
            elif fast_early and (sci_minus >= 0.65):
                shape_tag = "FAST_EARLY"
            else:
                shape_tag = "EVEN"

        if shape_tag == "SLOW_EARLY":
            sci = float(sci_plus if np.isfinite(sci_plus) else 1.0)
        elif shape_tag == "FAST_EARLY":
            sci = float(sci_minus if np.isfinite(sci_minus) else 1.0)
        else:
            sci = float((delta_EL.abs() <= 1.5).mean()) if delta_EL.notna().any() else 1.0

        # FRA (False-Run Adjustment) — mild, only if SCI is decent
        w["PI_RS"]  = w["PI"].astype(float)
        w["GCI_RS"] = w["GCI"].astype(float)

        if (shape_tag == "SLOW_EARLY") and (sci >= 0.60):
            f = 0.12 + 0.08 * (sci - 0.60) / 0.40
            late_excess = ((pd.to_numeric(w["Accel"], errors="coerce") +
                            pd.to_numeric(w[GR_COL],   errors="coerce"))/2.0 - 100.0
                          ).clip(lower=0.0, upper=8.0).fillna(0.0)
            w["PI_RS"]  = (w["PI"]  - f * (late_excess / 4.0)).clip(0.0, 10.0)
            w["GCI_RS"] = (w["GCI"] - f * (late_excess / 3.0)).clip(0.0, 10.0)
            fra_applied = 1

        elif (shape_tag == "FAST_EARLY") and (sci >= 0.60):
            f2 = 0.10 + 0.05 * (sci - 0.60) / 0.40
            sturdiness = ((pd.to_numeric(w[GR_COL], errors="coerce") - 100.0)
                          - (100.0 - pd.to_numeric(w["Accel"], errors="coerce")).clip(lower=0.0)
                         ).clip(lower=0.0, upper=6.0).fillna(0.0)
            w["PI_RS"]  = (w["PI"]  + f2 * (sturdiness / 4.0)).clip(0.0, 10.0)
            w["GCI_RS"] = (w["GCI"] + f2 * (sturdiness / 3.0)).clip(0.0, 10.0)
            fra_applied = 1

    else:
        # race shape OFF → passthrough
        w["PI_RS"]  = w["PI"].astype(float)
        w["GCI_RS"] = w["GCI"].astype(float)

                                
    # ---------- Final rounding ----------
    for c in ["EARLY_idx","LATE_idx","F200_idx","tsSPI","Accel","Grind","Grind_CG",
              "PI","PI_RS","GCI","GCI_RS","RaceTime_s","DeltaG","FinisherFactor","GrindAdjPts"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce").round(3)

    # ---------- Attach race-level diagnostics ----------
    w.attrs["FSR"] = float(FSR)
    w.attrs["CollapseSeverity"] = float(CollapseSeverity)
    w.attrs["GR_COL"] = GR_COL
    w.attrs["STEP"] = step
    w.attrs["SHAPE_TAG"] = shape_tag
    w.attrs["SCI"] = float(sci)
    w.attrs["FRA_APPLIED"] = int(fra_applied)

    if debug:
        st.write({
            "FSR": w.attrs["FSR"],
            "CollapseSeverity": w.attrs["CollapseSeverity"],
            "PI_W": PI_W,
            "SHAPE_TAG": shape_tag,
            "SCI": sci,
            "FRA_APPLIED": fra_applied
        })

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

# ======================= Data Integrity & Header (post compute) ==========================
integrity_text, missing_cols, invalid_counts = integrity_scan(work, race_distance_input, split_step)

st.markdown(
    f"## Race Distance: **{int(race_distance_input)}m**  |  Split step: **{split_step}m**  "
    f"|  Shape: **{metrics.attrs.get('SHAPE_TAG','EVEN')}**  "
    f"|  SCI: **{metrics.attrs.get('SCI',1.0):.2f}**  "
    f"|  FRA: **{'Yes' if metrics.attrs.get('FRA_APPLIED',0)==1 else 'No'}**"
)
if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    warn = []
    if missing_cols: warn.append("Missing: " + ", ".join(missing_cols))
    if bads: warn.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    if warn: st.markdown(f"*(⚠ {' • '.join(warn)})*")
if split_step == 200:
    st.caption("Finish column assumed to be the 200→0 segment (`Finish_Time`).")

# ======================= Sectional Metrics table (incl. RS & EARLY/LATE) =====================
st.markdown("## Sectional Metrics (PI v3.2 & GCI + CG + Race Shape)")

show_cols = [
    "Horse", "Finish_Pos", "RaceTime_s",
    "F200_idx", "tsSPI", "Accel",
    "Grind", "Grind_CG",
    "EARLY_idx", "LATE_idx",
    "GrindAdjPts", "DeltaG",
    "PI", "PI_RS", "GCI", "GCI_RS"
]
for c in show_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan

display_df = metrics[show_cols].copy()
_finish_sort = display_df["Finish_Pos"].fillna(1e9)
display_df = display_df.assign(_FinishSort=_finish_sort)
display_df = display_df.sort_values(["PI_RS","_FinishSort"], ascending=[False, True]).drop(columns=["_FinishSort"])
st.dataframe(display_df, use_container_width=True)

st.caption(
    f"CG={'ON' if USE_CG else 'OFF'} (FSR={metrics.attrs.get('FSR',1.0):.3f}; Collapse={metrics.attrs.get('CollapseSeverity',0.0):.1f}).  "
    f"Race Shape={metrics.attrs.get('SHAPE_TAG','EVEN')} (SCI={metrics.attrs.get('SCI',1.0):.2f}; FRA={'Yes' if metrics.attrs.get('FRA_APPLIED',0)==1 else 'No'})."
)
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

# ======================= Pace Curve — field average (black) + Top 8 finishers =======================
st.markdown("## Pace Curve — field average (black) + Top 8 finishers")
pace_png = None

if seg_markers or ("Finish_Time" in work.columns):
    step = metrics.attrs.get("STEP", 100)
    wanted = [m for m in range(int(race_distance_input) - step, step-1, -step)]
    segs = []
    for m in wanted:
        c = f"{m}_Time"
        if c in work.columns:
            segs.append((m+step, m, float(step), c))
    if "Finish_Time" in work.columns:
        segs.append((step, 0, float(step), "Finish_Time"))

    # If nothing valid, bail early
    if len(segs) == 0:
        st.info("Not enough *_Time columns to draw the pace curve.")
    else:
        # Build speed table with NaN-safe division
        seg_cols = [c for (_,_,_,c) in segs]
        times_df = work[seg_cols].apply(pd.to_numeric, errors="coerce")
        times_df = times_df.mask((times_df <= 0) | (~np.isfinite(times_df)))
        speed_df = pd.DataFrame(index=work.index)
        for (s, e, L, c) in segs:
            speed_df[c] = L / times_df[c]

        # Field average (ignore all-NaN columns gracefully)
        field_avg = speed_df.mean(axis=0, skipna=True).to_numpy()
        if not np.isfinite(np.nanmean(field_avg)):
            st.info("Pace curve: all segments missing/invalid.")
        else:
            # Choose top-8 lines to draw
            if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
                top8 = metrics.sort_values("Finish_Pos").head(8)
                top8_rule = "Top-8 by Finish_Pos"
            else:
                top8 = metrics.sort_values("PI", ascending=False).head(8)
                top8_rule = "Top-8 by PI"

            x_idx = list(range(len(segs)))
            def seg_label(s, e, c):
                return f"{int(s)}→{int(e)}" if c != "Finish_Time" else f"{step}→0 (Finish)"
            x_labels = [seg_label(s,e,c) for (s,e,_,c) in segs]

            fig2, ax2 = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
            ax2.plot(x_idx, field_avg, linewidth=2.2, color="black", label="Field average", marker=None)

            palette = color_cycle(len(top8))
            for i, (_, r) in enumerate(top8.iterrows()):
                # Pick the row with times for the matching horse
                if "Horse" in work.columns and "Horse" in metrics.columns:
                    row0 = work[work["Horse"] == r.get("Horse")]
                    row_times = row0.iloc[0] if not row0.empty else r
                else:
                    row_times = r

                # Collect speeds for this horse, skipping bad segments
                y_vals = []
                for (_, _, L, c) in segs:
                    t = pd.to_numeric(row_times.get(c, np.nan), errors="coerce")
                    y = (L / float(t)) if (pd.notna(t) and float(t) > 0.0) else np.nan
                    y_vals.append(y)

                # If this horse has no valid segments, skip plotting it
                if not np.isfinite(np.nanmean(y_vals)):
                    continue

                ax2.plot(x_idx, y_vals, linewidth=1.1, marker="o", markersize=2.5,
                         label=str(r.get("Horse", "")), color=palette[i])

            ax2.set_xticks(x_idx)
            ax2.set_xticklabels(x_labels, rotation=45, ha="right")
            ax2.set_ylabel("Speed (m/s)")
            ax2.set_title("Pace over segments (left = early; right = home straight; includes Finish)")
            ax2.grid(True, linestyle="--", alpha=0.30)
            ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)

            st.pyplot(fig2)
            buf = io.BytesIO()
            fig2.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            pace_png = buf.getvalue()
            st.download_button("Download pace curve (PNG)", pace_png,
                               file_name="pace_curve.png", mime="image/png")
            st.caption(f"Top-8 plotted: {top8_rule}. Finish segment included explicitly.")
else:
    st.info("Not enough *_Time columns to draw the pace curve.")
    
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

# --- ASI² (bias awareness) ---
acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get(gr_col), errors="coerce").median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)
B = min(1.0, abs(bias) / 4.0)
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get(gr_col), errors="coerce")
hh["ASI2"] = (B * (-S if bias >= 0 else S).clip(lower=0.0) / 5.0).fillna(0.0)

# --- TFS (trip friction) ---
def tfs_row(r):
    last_cols = [c for c in ["300_Time", "200_Time", "100_Time"] if c in r.index]
    spds = [metrics.attrs.get("STEP",100) / as_num(r.get(c)) for c in last_cols if pd.notna(r.get(c)) and as_num(r.get(c)) > 0]
    if len(spds) < 2: return np.nan
    sigma = np.std(spds, ddof=0)
    mid = as_num(r.get("_MID_spd"))
    return np.nan if not np.isfinite(mid) or mid <= 0 else 100.0 * (sigma / mid)

hh["TFS"] = hh.apply(tfs_row, axis=1)
D_rounded = int(np.ceil(float(race_distance_input)/200.0)*200)
gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
hh["TFS_plus"] = hh["TFS"].apply(lambda x: 0.0 if pd.isna(x) or x < gate else min(0.6, (x-gate)/3.0))

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
    pi_rs, gci_rs = as_num(r.get("PI_RS")), as_num(r.get("GCI_RS"))
    if np.isfinite(pi_rs) and np.isfinite(gci_rs):
        if pi_rs >= 7.2 and gci_rs >= 6.0: return "🔥 Top Hidden"
        if pi_rs >= 6.2 and gci_rs >= 5.0: return "🟡 Notable Hidden"
        return ""
    hs = as_num(r.get("HiddenScore"))
    if not np.isfinite(hs): return ""
    if hs >= 1.8: return "🔥 Top Hidden"
    if hs >= 1.2: return "🟡 Notable Hidden"
    return ""
hh["Tier"] = hh.apply(hh_tier_row, axis=1)

# --- Descriptive note ---
def hh_note(r):
    pi_rs, gci_rs = as_num(r.get("PI_RS")), as_num(r.get("GCI_RS"))
    bits=[]
    if np.isfinite(pi_rs) and np.isfinite(gci_rs):
        bits.append(f"PI_RS {pi_rs:.2f}, GCI_RS {gci_rs:.2f}")
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

# ======================= Batch 4 — Database, Search & PDF Export (DROP-IN) =======================
import sqlite3
import io
from datetime import datetime

# ----------------------- DB helpers -----------------------
def _open_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _ensure_schema(conn: sqlite3.Connection):
    """
    Ensures that the DB schema is valid.
    Drops and recreates tables if mismatched, with consistent TEXT race_id linkage.
    """

    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF;")

    # Drop old tables safely — one per execute() to avoid multi-statement error
    try:
        cur.execute("DROP TABLE IF EXISTS performances;")
    except Exception as e:
        print("Drop performances failed:", e)
    try:
        cur.execute("DROP TABLE IF EXISTS races;")
    except Exception as e:
        print("Drop races failed:", e)

    # Recreate tables with consistent TEXT race_id
    cur.execute("""
    CREATE TABLE races(
        race_id        TEXT PRIMARY KEY,
        date           TEXT,
        track          TEXT,
        race_no        INTEGER,
        distance_m     INTEGER NOT NULL,
        split_step     INTEGER NOT NULL,
        fsr            REAL,
        collapse       REAL,
        rsbi           REAL,
        rsp            REAL,
        use_cg         INTEGER,
        dampen_when_collapsed INTEGER,
        use_shape_module INTEGER,
        notes          TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE performances(
        perf_id        TEXT PRIMARY KEY,
        race_id        TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
        horse          TEXT NOT NULL,
        horse_canon    TEXT NOT NULL,
        finish_pos     INTEGER,
        race_time_s    REAL,
        iai            REAL,
        hidden         REAL,
        ability        REAL,
        tier           TEXT,
        direction      TEXT,
        confidence     TEXT,
        pi             REAL,
        gci            REAL,
        inserted_ts    TEXT DEFAULT (datetime('now'))
    );
    """)

    conn.commit()
    cur.execute("PRAGMA foreign_keys=ON;")

def _table_cols(conn: sqlite3.Connection, tbl: str) -> set[str]:
    cur = conn.cursor()
    return {row[1] for row in cur.execute(f"PRAGMA table_info({tbl})")}

def _insert_or_replace(conn: sqlite3.Connection, tbl: str, row_dict: dict):
    cols_present = _table_cols(conn, tbl)
    payload = {k: v for k, v in row_dict.items() if k in cols_present}
    if not payload:
        return
    keys = ",".join(payload.keys())
    qmarks = ",".join(["?"] * len(payload))
    conn.execute(f"INSERT OR REPLACE INTO {tbl} ({keys}) VALUES ({qmarks})", list(payload.values()))

# ----------------------- PDF Export -----------------------
def _export_pdf_report(*,
                       distance_m:int,
                       metrics_table_df:pd.DataFrame,
                       shape_png:bytes|None,
                       pace_png:bytes|None,
                       ability_png:bytes|None,
                       ability_table_df:pd.DataFrame,
                       hidden_table_df:pd.DataFrame,
                       race_title:str):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import cm
    except Exception:
        st.error("PDF export needs `reportlab`. Install with: `pip install reportlab>=4`")
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph(f"<b>{race_title}</b> — {int(distance_m)} m", styles["Heading1"]))
    fsr = metrics.attrs.get("FSR", None)
    cs  = metrics.attrs.get("CollapseSeverity", None)
    step= metrics.attrs.get("STEP", None)
    cg  = "ON" if USE_CG else "OFF"
    cg_line = f"CG: {cg}"
    if fsr is not None:
        cg_line += f" · FSR={float(fsr):.3f}"
    if cs is not None:
        cg_line += f" · CollapseSeverity={float(cs):.1f} pts"
    if step is not None:
        cg_line += f" · Splits: {int(step)} m"
    story.append(Paragraph(cg_line, styles["Normal"]))
    story.append(Spacer(0, 6))

    # Sectional Metrics table (safe cast to string)
    story.append(Paragraph("Sectional Metrics", styles["Heading3"]))
    tbl_df = metrics_table_df.copy().fillna("")
    data = [list(tbl_df.columns)] + tbl_df.astype(str).values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.25,colors.whitesmoke),
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('ALIGN',(2,1),(-1,-1),'RIGHT')
    ]))
    story.append(t); story.append(Spacer(0, 8))

    # Images
    if shape_png:
        story.append(Paragraph("Sectional Shape Map", styles["Heading3"]))
        story.append(Image(io.BytesIO(shape_png), width=24*cm, height=17*cm, kind="proportional"))
        story.append(Spacer(0, 6))
    if pace_png:
        story.append(Paragraph("Pace Curve", styles["Heading3"]))
        story.append(Image(io.BytesIO(pace_png), width=24*cm, height=15*cm, kind="proportional"))
        story.append(Spacer(0, 6))
    if ability_png:
        story.append(Paragraph("Ability Matrix", styles["Heading3"]))
        story.append(Image(io.BytesIO(ability_png), width=24*cm, height=16*cm, kind="proportional"))
        story.append(Spacer(0, 6))

    # Hidden Horses (flagged only)
    story.append(Paragraph("Hidden Horses (flagged)", styles["Heading3"]))
    flagged = hidden_table_df[hidden_table_df.get("Tier","") != ""].copy()
    if not flagged.empty:
        data_hh = [list(flagged.columns)] + flagged.fillna("").astype(str).values.tolist()
        t2 = Table(data_hh, repeatRows=1)
        t2.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('GRID',(0,0),(-1,-1),0.25,colors.whitesmoke),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('ALIGN',(2,1),(-1,-1),'RIGHT')
        ]))
        story.append(t2)
    else:
        story.append(Paragraph("No horses flagged in this race.", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return buf

# ----------------------- Save Current Race -----------------------
def _save_current_race_to_db(db_path: str,
                             race_date_str: str,
                             race_track: str,
                             race_no_int: int,
                             title_override: str | None = None) -> int:
    """
    Saves one `races` row + N `performances` rows.
    Returns the number of performances saved.
    """
    # Pull context from analysis
    step        = int(metrics.attrs.get("STEP", 100))
    fsr_val     = float(metrics.attrs.get("FSR", 1.0))
    collapse_pt = float(metrics.attrs.get("CollapseSeverity", 0.0))
    rsbi_val    = float(metrics.attrs.get("RSBI", np.nan)) if hasattr(metrics, "attrs") else np.nan
    rsp_val     = float(metrics.attrs.get("RSP",  np.nan)) if hasattr(metrics, "attrs") else np.nan

    use_cg_val  = 1 if USE_CG else 0
    dampen_val  = 1 if DAMPEN_CG else 0
    use_shape_val = 1 if ('USE_SHAPE' in globals() and USE_SHAPE) else 0

    # Robust keys
    _date  = str(race_date_str or "").strip()
    _track = str(race_track or "").strip()
    _rno   = int(race_no_int or 0)

    # Deterministic race_id
    race_id = sha1(f"{_date}|{_track}|{_rno}|{int(race_distance_input)}|{step}")

    # Make sure we can save PI/GCI/IAI/etc.
    _m = metrics.loc[:, ["Horse","RaceTime_s"]].copy() if "RaceTime_s" in metrics.columns else pd.DataFrame(columns=["Horse","RaceTime_s"])
    _am = AM_view.copy() if 'AM_view' in globals() else pd.DataFrame()
    to_save = _am.merge(_m, on="Horse", how="left") if not _am.empty else pd.DataFrame()

    conn = _open_db(db_path)
    _ensure_schema(conn)

    # Races row
    race_row = {
        "race_id": race_id,
        "date": _date,
        "track": _track,
        "race_no": _rno,
        "distance_m": int(race_distance_input),
        "split_step": step,
        "fsr": fsr_val,
        "collapse": collapse_pt,
        # Optional, only written if columns exist in your DB
        "rsbi": rsbi_val,
        "rsp":  rsp_val,
        "use_cg": use_cg_val,
        "dampen_when_collapsed": dampen_val,
        "use_shape_module": use_shape_val,
        "notes": (title_override or "")
    }
    _insert_or_replace(conn, "races", race_row)

    # Performances rows
    n_saved = 0
    if not to_save.empty:
        for _, r in to_save.iterrows():
            horse = str(r.get("Horse","")).strip()
            if not horse:
                continue
            horse_canon = canon_horse(horse)
            perf_id = sha1(f"{race_id}|{horse_canon}")

            perf_row = {
                "perf_id":     perf_id,
                "race_id":     race_id,
                "horse":       horse,
                "horse_canon": horse_canon,
                "finish_pos":  int(r.get("Finish_Pos")) if pd.notna(r.get("Finish_Pos")) else None,
                "race_time_s": float(r.get("RaceTime_s")) if pd.notna(r.get("RaceTime_s")) else None,
                "iai":         float(r.get("IAI")) if pd.notna(r.get("IAI")) else None,
                "hidden":      float(r.get("HiddenScore")) if pd.notna(r.get("HiddenScore")) else None,
                "ability":     float(r.get("AbilityScore")) if pd.notna(r.get("AbilityScore")) else None,
                "tier":        str(r.get("AbilityTier")) if pd.notna(r.get("AbilityTier")) else None,
                "direction":   str(r.get("DirectionHint")) if pd.notna(r.get("DirectionHint")) else None,
                "confidence":  str(r.get("Confidence")) if pd.notna(r.get("Confidence")) else None,
                "pi":          float(r.get("PI")) if pd.notna(r.get("PI")) else None,
                "gci":         float(r.get("GCI")) if pd.notna(r.get("GCI")) else None,
            }
            _insert_or_replace(conn, "performances", perf_row)
            n_saved += 1

    conn.commit()
    conn.close()
    return n_saved

# ----------------------- UI: Horse search (can be used anytime) -----------------------
st.markdown("---")
st.markdown("### 🐎 Horse Database Search")
_search_db_path = db_path if 'db_path' in globals() else "race_edge.db"   # sidebar text_input earlier
search_name = st.text_input("Search horse name (exact or partial):", key="db_search_name").strip()
if search_name:
    try:
        conn = _open_db(_search_db_path)
        df_search = pd.read_sql_query(
            "SELECT * FROM performances WHERE horse LIKE ? ORDER BY rowid DESC",
            conn, params=[f"%{search_name}%"]
        )
        conn.close()
        if df_search.empty:
            st.info("No records found.")
        else:
            st.dataframe(df_search, use_container_width=True)
            st.caption("Most recent first.")
    except Exception as e:
        st.error("Search failed.")
        st.exception(e)
else:
    st.caption("Type at least part of a horse name to search stored performances.")

# ----------------------- UI: Save to DB -----------------------
st.markdown("---")
st.markdown("### 💾 Save current race to database")
colA, colB, colC, colD = st.columns([1.1, 1.1, 0.6, 1.3])
with colA:
    ui_race_date = st.text_input("Race date (YYYY-MM-DD):", value=datetime.utcnow().strftime("%Y-%m-%d"))
with colB:
    ui_track = st.text_input("Track / Meeting:", value="")
with colC:
    ui_rno = st.number_input("Race #", min_value=0, max_value=50, value=1, step=1)
with colD:
    ui_title = st.text_input("Optional title / note:", value="")

if st.button("Save this race to DB"):
    try:
        saved_n = _save_current_race_to_db(
            db_path,
            race_date_str=ui_race_date,
            race_track=ui_track,
            race_no_int=int(ui_rno),
            title_override=ui_title
        )
        st.success(f"Saved race and {saved_n} performances to {db_path}.")
    except Exception as e:
        st.error("Saving failed. See details below.")
        st.exception(e)

# ----------------------- UI: PDF Export -----------------------
st.markdown("---")
st.markdown("### 📥 Export Complete Race Report (PDF)")
pdf_btn = st.button("Generate PDF Report")
if pdf_btn:
    pdf_buf = _export_pdf_report(
        distance_m=int(race_distance_input),
        metrics_table_df=display_df if 'display_df' in globals() else pd.DataFrame(),
        shape_png=shape_map_png if 'shape_map_png' in globals() else None,
        pace_png=pace_png if 'pace_png' in globals() else None,
        ability_png=ability_png if 'ability_png' in globals() else None,
        ability_table_df=AM_view if 'AM_view' in globals() else pd.DataFrame(),
        hidden_table_df=hh_view if 'hh_view' in globals() else pd.DataFrame(),
        race_title=f"{ui_track or 'Race'} · {ui_race_date or ''}"
    )
    if pdf_buf is not None:
        st.download_button(
            "📥 Download PDF",
            data=pdf_buf.getvalue(),
            file_name=f"RaceEdge_{int(race_distance_input)}m_Report.pdf",
            mime="application/pdf"
        )
