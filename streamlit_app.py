# ======================= Batch 1 — Core + UI setup =======================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io, math, re, os, sqlite3, hashlib
from datetime import datetime

# ----------------------- Page config -----------------------
st.set_page_config(
    page_title="Race Edge — PI v3.1 + Hidden v2 + Ability v2 + CG + DB",
    layout="wide"
)

DB_DEFAULT_PATH = "race_edge.db"

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

# ----------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("### Upload race")
    up = st.file_uploader(
        "Upload CSV/XLSX with **100 m** or **200 m** splits (Finish column can be `Finish_Time` or `Finish_Split`).",
        type=["csv","xlsx","xls"]
    )
    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000, step=50, value=1600)
    USE_CG = st.toggle("Use Corrected Grind (CG)", value=True, help="Adjust Grind when field finish collapses; preserves finisher credit.")
    DAMPEN_CG = st.toggle("Dampen Grind weight if collapsed", value=True, help="Shift a little weight Grind→Accel/tsSPI on collapse races.")
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

    st.markdown("---")
    st.markdown("### Database")
    db_path = st.text_input("Database path", value=DB_DEFAULT_PATH)
    init_btn = st.button("Initialise / Check DB")

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
  created_ts     TEXT DEFAULT (datetime('now')),
  src_hash       TEXT
);
CREATE TABLE IF NOT EXISTS performances(
  perf_id        TEXT PRIMARY KEY,
  race_id        TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
  horse          TEXT NOT NULL,
  horse_canon    TEXT NOT NULL,
  finish_pos     INTEGER,
  race_time_s    REAL,
  f200_idx       REAL,
  tsspi          REAL,
  accel          REAL,
  grind          REAL,
  grind_cg       REAL,
  delta_g        REAL,
  finisher_factor REAL,
  grind_adj_pts  REAL,
  pi             REAL,
  gci            REAL,
  hidden         REAL,
  ability        REAL,
  ability_tier   TEXT,
  iai            REAL,
  bal            REAL,
  comp           REAL,
  z_iai          REAL,
  h_sig          REAL,
  z_bal          REAL,
  z_comp         REAL,
  dir_hint       TEXT,
  confidence     TEXT,
  inserted_ts    TEXT DEFAULT (datetime('now'))
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

# Stop until a file is uploaded
if not up:
    st.stop()

def normalize_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize common variants (case-insensitive):
      • '<meters>_time' or '<meters>m_time'  -> '<meters>_Time'
      • '<meters>_split' or '<meters>m_split'-> '<meters>_Time'
      • 'finish_time' / 'finish_split' / 'finish' -> 'Finish_Time'
      • 'finish_pos' -> 'Finish_Pos'
    """
    notes = []
    # Build a lookup of lowercased keys -> original column names
    lmap = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    def alias(src_key, alias_col):
        nonlocal df, notes
        if src_key in lmap and alias_col not in df.columns:
            df[alias_col] = df[lmap[src_key]]
            notes.append(f"Aliased `{lmap[src_key]}` → `{alias_col}`")

    # Finish column variants
    for k in ("finish_time", "finish_split", "finish"):
        alias(k, "Finish_Time")
    alias("finish_pos", "Finish_Pos")

    # Segment columns: accept optional trailing 'm' before the underscore
    # Matches: '1200_time', '1200m_time', '800_split', '800m_split', etc.
    pat = re.compile(r"^(\d{2,4})m?_(time|split)$")
    for lk, orig in lmap.items():
        m = pat.match(lk)
        if m:
            alias_col = f"{m.group(1)}_Time"
            if alias_col not in df.columns:
                df[alias_col] = df[orig]
                notes.append(f"Aliased `{orig}` → `{alias_col}`")

    return df, notes

# ----------------------- Split-step detection ----------------
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

# ----------------------- File load & preview ----------------
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

# --------- Hand-off to Batch 2 (metrics engine will use `work`, `split_step`, `race_distance_input`, and toggles) ---------
# ======================= Batch 2 — Metrics Engine =======================

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
        # 200m splits: Finish_Time is the 200→0 segment
        tfin = as_num(row.get("Finish_Time"))
        if pd.isna(tfin) or tfin <= 0:
            return np.nan
        return 200.0 / float(tfin)

# -------- Distance + context weights for PI v3.1 --------
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

    # Contextual tweak based on within-race bias (Accel vs Grind medians)
    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale
        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]
        if bias > 0:           # race rewarded acceleration → give a little more to Grind for balance
            delta = min(max_shift, ACC - 0.26)
            ACC -= delta; GR += delta
        elif bias < 0:         # race rewarded grinding → swing a hair back to Accel
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
def build_metrics(df_in: pd.DataFrame, D_actual_m: float, step: int, use_cg: bool, dampen_cg: bool, debug: bool):
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

    # tsSPI: (D-300)…600 for 100m; (D-400)…600 for 200m (same philosophy, correct windows)
    tssp_start = int(D - (300 if step == 100 else 400))
    tssp_cols = make_range_cols(D, tssp_start, 600, step)
    tssp_cols = [c for c in tssp_cols if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: stage_speed(r, tssp_cols, float(step)), axis=1)

    # Accel: 600→200 (100m: 500,400,300,200; 200m: 600,400)
    if step == 100:
        accel_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]
    else:
        accel_cols = [c for c in [f"{m}_Time" for m in [600,400]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: stage_speed(r, accel_cols, float(step)), axis=1)

    # Grind
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

    # ---------- PI v3.1 ----------
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

    # ---------- GCI (0–10) ----------
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

    # Round key outputs
    for c in ["F200_idx","tsSPI","Accel","Grind","Grind_CG","PI","GCI","RaceTime_s","DeltaG","FinisherFactor","GrindAdjPts"]:
        if c in w.columns:
            w[c] = w[c].round(3)

    # Attach race-level diagnostics
    w.attrs["FSR"] = FSR
    w.attrs["CollapseSeverity"] = CollapseSeverity
    w.attrs["GR_COL"] = GR_COL
    w.attrs["STEP"] = step

    if debug:
        st.write({"FSR": FSR, "CollapseSeverity": CollapseSeverity, "PI_W": PI_W})

    return w, seg_markers

# ---- Compute metrics now
try:
    metrics, seg_markers = build_metrics(work, float(race_distance_input), split_step, USE_CG, DAMPEN_CG, DEBUG)
except Exception as e:
    st.error("Metric computation failed.")
    st.exception(e)
    st.stop()

# ======================= Data Integrity & Header ==========================
def expected_segments(distance_m: float, step:int) -> list[str]:
    want = [f"{m}_Time" for m in range(int(distance_m) - step, step-1, -step)]
    want.append("Finish_Time")
    return want

exp_cols = expected_segments(race_distance_input, split_step)
missing_cols = [c for c in exp_cols if c not in work.columns]
invalid_counts = {}
for c in exp_cols:
    if c in work.columns:
        s = pd.to_numeric(work[c], errors="coerce")
        invalid_counts[c] = int(((s <= 0) | s.isna()).sum())

def integrity_line():
    msgs = []
    if missing_cols:
        msgs.append("Missing: " + ", ".join(missing_cols))
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs)

st.markdown(f"## Race Distance: **{int(race_distance_input)}m**  |  Split step: **{split_step}m**")
if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    st.markdown(f"*(⚠ {integrity_line()})*")
if split_step == 200:
    st.caption("Finish column assumed to be the 200→0 segment (Finish_Split).")

# ======================= Sectional Metrics table ==========================
st.markdown("## Sectional Metrics (PI v3.1 & GCI + CG diagnostics)")
show_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200_idx", "tsSPI", "Accel",
             "Grind", "Grind_CG", "GrindAdjPts", "DeltaG", "PI", "GCI"]
for c in show_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan

display_df = metrics[show_cols].copy()
_finish_sort = display_df["Finish_Pos"].fillna(1e9)
display_df = display_df.assign(_FinishSort=_finish_sort)
display_df = display_df.sort_values(["PI","_FinishSort"], ascending=[False, True]).drop(columns=["_FinishSort"])
st.dataframe(display_df, use_container_width=True)
st.caption(f"CG on: {USE_CG}. Race FSR={metrics.attrs.get('FSR',1.0):.3f}; CollapseSeverity={metrics.attrs.get('CollapseSeverity',0.0):.1f} index pts.")

# -------- Hand-off to Batch 3 (visuals + Hidden Horses + Ability v2) --------
# ======================= Batch 3 — Visuals + Hidden v2 + Ability v2 =======================
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ----------------------- Label repel (fallback, no external deps) -----------------------
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
        for t,(xi,yi) in zip(texts,xy):
            tx,ty=t.get_position(); pt=trans.transform((tx,ty)); pp=trans.transform((xi,yi))
            d=((pt[0]-pp[0])**2+(pt[1]-pp[1])**2)**0.5; tgt=25.0
            if abs(d-tgt)>1.0:
                v=(pt-pp)/(d+1e-9); pt2=pt+v*(0.6*(tgt-d)); t.set_position(inv.transform(pt2)); moved=True
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
        xv=dfm["AccelΔ"].to_numpy()
        yv=dfm["GrindΔ"].to_numpy()
        cv=dfm["tsSPIΔ"].to_numpy()
        piv=dfm["PI"].fillna(0).to_numpy()

        # axis range
        span = np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]) if np.isfinite(xv).any() and np.isfinite(yv).any() else 1.0
        if not np.isfinite(span) or span <= 0: span = 1.0
        lim = max(4.5, float(np.ceil(span/1.5)*1.5))

        # point sizes by PI
        DOT_MIN, DOT_MAX = 40.0, 140.0
        pmin, pmax = float(np.nanmin(piv)), float(np.nanmax(piv))
        sizes = np.full_like(xv, DOT_MIN) if (not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax-pmin)<1e-9) \
                else DOT_MIN + (piv - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)

        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        TINT = 0.06
        ax.add_patch(Rectangle((0,0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))
        ax.axvline(0, color="gray", lw=1.3, ls=(0,(3,3)))
        ax.axhline(0, color="gray", lw=1.3, ls=(0,(3,3)))

        vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
        vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax: vmin, vmax = -1.0, 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        sc = ax.scatter(xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                        edgecolor="black", linewidth=0.6, alpha=0.95)

        # neat labels
        label_points_neatly(ax, xv, yv, names)

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Acceleration vs field (points)  →")
        ax.set_ylabel(("Corrected " if USE_CG else "") + "Grind vs field (points)  ↑")
        ax.set_title("Quadrants: +X = Accel (400→200 zone); +Y = " + ("Corrected Grind" if USE_CG else "Grind") + ".  Colour = tsSPIΔ")

        # size legend
        s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
        h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
        ax.legend(h_ex, ["PI: low","PI: mid","PI: high"], loc="upper left", frameon=False, fontsize=8)

        # colour bar
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("tsSPI − 100")

        ax.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(fig)

        # download
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        shape_map_png = buf.getvalue()
        st.download_button("Download shape map (PNG)", shape_map_png, file_name="shape_map.png", mime="image/png")
        st.caption(("Y uses Corrected Grind (CG). " if USE_CG else "") + "Size = PI. X: Accel. Colour = tsSPIΔ.")

# ======================= Visual 2: Pace Curve (with key) =======================
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

    if len(segs) > 0:
        times_df = work[[c for (_,_,_,c) in segs]].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0, np.nan)
        speed_df = pd.DataFrame(index=work.index)
        for (s, e, L, c) in segs:
            speed_df[c] = L / times_df[c]

        field_avg = speed_df.mean(axis=0).to_numpy()

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
            if "Horse" in work.columns and "Horse" in metrics.columns:
                row0 = work[work["Horse"] == r.get("Horse")]
                row_times = row0.iloc[0] if not row0.empty else r
            else:
                row_times = r
            y_vals = []
            for (_, _, L, c) in segs:
                t = pd.to_numeric(row_times.get(c, np.nan), errors="coerce")
                y_vals.append(L / t if pd.notna(t) and t > 0 else np.nan)
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

# ======================= Hidden Horses (v2) =======================
st.markdown("## Hidden Horses (v2)")

hh = metrics.copy()
gr_col = GR_COL

# 1) SOS (robust z blends, winsorized)
need_cols = {"tsSPI","Accel",gr_col}
if need_cols.issubset(hh.columns) and len(hh) > 0:
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh[gr_col], errors="coerce"))

    def rz(s: pd.Series) -> pd.Series:
        mu = np.nanmedian(s)
        sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    z_ts = rz(ts_w).clip(-2.5, 3.5)
    z_ac = rz(ac_w).clip(-2.5, 3.5)
    z_gr = rz(gr_w).clip(-2.5, 3.5)

    hh["SOS_raw"] = 0.45 * z_ts + 0.35 * z_ac + 0.20 * z_gr
    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and (q95 > q5)) else 1.0
    hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(lower=0.0, upper=2.0)
else:
    hh["SOS"] = 0.0

# 2) ASI² — bias aware (against-race-shape)
acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get(gr_col), errors="coerce").median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)
B = min(1.0, abs(bias) / 4.0)
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get(gr_col), errors="coerce")
if bias >= 0:
    hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
else:
    hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)

# 3) TFS — late variance vs mid-pace (scaled + gate by distance)
def tfs_row(row):
    # last 200/100 chunks we have: try 300,200,100 if present
    last_cols = [c for c in ["300_Time","200_Time","100_Time"] if c in row.index]
    spds = []
    for c in last_cols:
        t = pd.to_numeric(row.get(c), errors="coerce")
        spds.append((metrics.attrs.get("STEP",100))/ t if pd.notna(t) and t > 0 else np.nan)
    spds = [s for s in spds if pd.notna(s)]
    if len(spds) < 2:
        return np.nan
    sigma = float(np.std(spds, ddof=0))
    mid = float(row.get("_MID_spd", np.nan))
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    return 100.0 * (sigma / mid)
hh["TFS"] = hh.apply(tfs_row, axis=1)

D_rounded = int(np.ceil(float(race_distance_input) / 200.0) * 200)
gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
def tfs_plus(x):
    if pd.isna(x) or x < gate: return 0.0
    return min(0.6, (x - gate) / 3.0)
hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

# 4) UEI — underused engine patterns
def uei_row(r):
    ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac = pd.to_numeric(r.get("Accel"), errors="coerce")
    gr = pd.to_numeric(r.get(gr_col), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr):
        return 0.0
    val = 0.0
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    return round(val, 3)
hh["UEI"] = hh.apply(uei_row, axis=1)

# 5) HiddenScore v2 (0..3)
hidden = (
    0.55 * pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
    0.30 * pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
    0.10 * pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
    0.05 * pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0)
)
if int(hh.shape[0]) <= 6:
    hidden = hidden * 0.90

h_med = float(np.nanmedian(hidden))
h_mad = float(np.nanmedian(np.abs(hidden - h_med)))
h_sigma = max(1e-6, 1.4826 * h_mad)
hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5 * h_sigma)).clip(lower=0.0, upper=3.0)

def hh_tier(s):
    if pd.isna(s): return ""
    if s >= 1.8:   return "🔥 Top Hidden"
    if s >= 1.2:   return "🟡 Notable Hidden"
    return ""
hh["Tier"] = hh["HiddenScore"].apply(hh_tier)

def hh_note(r):
    bits = []
    if r.get("Tier", "") != "":
        if pd.to_numeric(r.get("SOS"), errors="coerce") >= 1.2:
            bits.append("sectionals superior")
        asi2 = pd.to_numeric(r.get("ASI2"), errors="coerce")
        if asi2 >= 0.8:   bits.append("ran against strong bias")
        elif asi2 >= 0.4: bits.append("ran against bias")
        if pd.to_numeric(r.get("TFS_plus"), errors="coerce") > 0:
            bits.append("trip friction late")
        if pd.to_numeric(r.get("UEI"), errors="coerce") >= 0.5:
            bits.append("latent potential if shape flips")
    return ("; ".join(bits).capitalize() + ".") if bits else ""
hh["Note"] = hh.apply(hh_note, axis=1)

cols_hh = [
    "Horse", "Finish_Pos", "PI", "GCI",
    "tsSPI", "Accel", gr_col,
    "SOS", "ASI2", "TFS", "UEI",
    "HiddenScore", "Tier", "Note"
]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

hh_view = hh.sort_values(["Tier", "HiddenScore", "PI"], ascending=[True, False, False])[cols_hh]
st.dataframe(hh_view, use_container_width=True)
st.caption("Hidden Horses v2: SOS (robust outlier), ASI² (against shape), TFS (trip friction), UEI (underused engine). Tier: 🔥 ≥1.8, 🟡 ≥1.2.")

# ======================= Ability Matrix v2 (percentile-based) =======================
st.markdown("---")
st.markdown("## Ability Matrix v2 — Intrinsic vs Hidden Ability")

ability_png = None
AM = metrics.copy()
AM = AM.merge(hh_view[["Horse","HiddenScore"]], on="Horse", how="left")
AM["HiddenScore"] = AM["HiddenScore"].fillna(0.0)

# Intrinsic Ability Index (IAI) uses corrected grind when active
AM["IAI"]  = 0.35*AM["tsSPI"] + 0.25*AM["Accel"] + 0.25*AM[gr_col] + 0.15*AM["F200_idx"]
AM["BAL"]  = 100.0 - (AM["Accel"] - AM[gr_col]).abs() / 2.0
AM["COMP"] = 100.0 - (AM["tsSPI"] - 100.0).abs()

# Robust z + percentile utilities
def robust_z(s):
    s = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(s)
    sig = mad_std(s - med)
    if not np.isfinite(sig) or sig == 0: return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - med) / sig

def pct_rank(s):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, method="average")
    return r.fillna(0.0).clip(0.0,1.0)

AM["IAI_pct"]  = pct_rank(AM["IAI"])            # stronger engine → higher
AM["HID_pct"]  = pct_rank(AM["HiddenScore"])    # hidden upside vs field
AM["BAL_pct"]  = 1.0 - pct_rank((AM["BAL"] - 100.0).abs())   # closer to 100 → higher
AM["COMP_pct"] = 1.0 - pct_rank((AM["COMP"] - 100.0).abs())  # steadier → higher

# Score (softened so fewer “Ordinary”)
# 0..10 scale: heavier on intrinsic, meaningful on hidden, small bonuses on balance/consistency
AM["AbilityScore"] = (
      6.5 * AM["IAI_pct"]
    + 2.5 * AM["HID_pct"]
    + 0.6 * AM["BAL_pct"]
    + 0.4 * AM["COMP_pct"]
).clip(0.0, 10.0).round(2)

def ability_tier(x):
    if pd.isna(x): return ""
    if x >= 7.5:   return "🥇 Elite"
    if x >= 6.2:   return "🥈 High"
    if x >= 4.8:   return "🥉 Competitive"
    return "⚪ Ordinary"
AM["AbilityTier"] = AM["AbilityScore"].apply(ability_tier)

# Directional hint (sprint vs staying lean)
def dir_hint_row(r):
    dv = float(r.get("Accel", np.nan)) - float(r.get(gr_col, np.nan))
    if not np.isfinite(dv): return ""
    if dv >= 2.0:  return "⚡ Sprint-lean (turn of foot)"
    if dv <= -2.0: return "🪨 Stayer-lean (sustained)"
    return "⚖ Balanced"
AM["DirectionHint"] = AM.apply(dir_hint_row, axis=1)

# Confidence band by field size
def conf_band(n):
    if n >= 12: return "High"
    if n >= 8:  return "Med"
    return "Low"
AM["Confidence"] = conf_band(len(AM.index))

# ---------- Plot (IAI vs Hidden) ----------
need_cols_am = {"Horse","IAI","HiddenScore","PI","BAL"}
if not need_cols_am.issubset(AM.columns):
    st.info("Ability Matrix: missing columns.")
else:
    plot_df = AM.dropna(subset=["IAI","HiddenScore","PI","BAL"]).copy()
    if plot_df.empty:
        st.info("Not enough complete data to draw Ability Matrix.")
    else:
        x = plot_df["IAI"] - 100.0
        y = plot_df["HiddenScore"]
        sizes = 60.0 + (plot_df["PI"].clip(0,10) / 10.0) * 200.0
        vmin, vmax = float(plot_df["BAL"].min()), float(plot_df["BAL"].max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            # Replace the vmin/vmax/norm section in Shape Map with:
vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = -1.0, 1.0
EPS = 0.1
if vmax <= 0.0: vmax = 0.0 + EPS
if vmin >= 0.0: vmin = 0.0 - EPS
norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        norm = TwoSlopeNorm(vcenter=100.0, vmin=vmin, vmax=vmax)

        figA, axA = plt.subplots(figsize=(8.6, 6.0))
        sc = axA.scatter(x, y, s=sizes, c=plot_df["BAL"], cmap="coolwarm", norm=norm,
                         edgecolor="black", linewidth=0.6, alpha=0.95)

        label_points_neatly(axA, x.values, y.values, plot_df["Horse"].astype(str).tolist())

        axA.axvline(0.0, color="gray", lw=1.0, ls="--")
        axA.axhline(1.2, color="gray", lw=0.8, ls=":")
        axA.set_xlabel("Intrinsic Ability (IAI – 100)  →")
        axA.set_ylabel("HiddenScore (0–3)  ↑")
        axA.set_title("Ability Matrix v2 — Size = PI · Colour = BAL (balance)")

        for s, lab in [(60, "PI low"), (160, "PI mid"), (260, "PI high")]:
            axA.scatter([], [], s=s, label=lab, color="gray", edgecolor="black")
        axA.legend(loc="upper left", frameon=False, fontsize=8, title="Point size:")
        cbar = figA.colorbar(sc, ax=axA, fraction=0.05, pad=0.04)
        cbar.set_label("BAL (100 = balanced late)")
        axA.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(figA)

        buf = io.BytesIO()
        figA.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        ability_png = buf.getvalue()
        st.download_button("Download Ability Matrix (PNG)", ability_png,
                           file_name="ability_matrix_v2.png", mime="image/png")

# ---------- Table ----------
am_cols = ["Horse","Finish_Pos","IAI","HiddenScore","BAL","COMP","AbilityScore","AbilityTier","DirectionHint","Confidence","PI"]
for c in am_cols:
    if c not in AM.columns:
        AM[c] = np.nan
AM_view = AM.sort_values(["AbilityScore","PI","Finish_Pos"], ascending=[False, False, True])[am_cols]
st.dataframe(AM_view, use_container_width=True)

# ------------------ Hand-off to Batch 4 (DB save/search + PDF) ------------------
