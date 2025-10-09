# ======================= Batch A — Core + UI + I/O + Integrity + Weight UI =======================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io, math, re, os, sqlite3, hashlib
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm

# ----------------------- Page config -----------------------
st.set_page_config(
    page_title="Race Edge — PI v3.2 + RS v2 + CG + Ability v2 + DB",
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

def _safe_bal_norm(series, center=100.0, pad=0.5):
    """Return a TwoSlopeNorm that always satisfies vmin < center < vmax."""
    arr = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        vmin, vmax = center - 5.0, center + 5.0
        return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

    vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
    if vmin == vmax:
        vmin = vmin - max(pad, 0.1); vmax = vmax + max(pad, 0.1)
    if vmax <= center: vmax = center + max(pad, (center - vmin) * 0.05 + 0.1)
    if vmin >= center: vmin = center - max(pad, (vmax - center) * 0.05 + 0.1)
    if not (vmin < center < vmax):
        if vmin >= center: vmin = center - 0.1
        if vmax <= center: vmax = center + 0.1
    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

# ----------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(f"### Race Edge v{APP_VERSION}")
    st.caption("PI v3.2 with Race Shape (SED/FRA/SCI), CG, Hidden v2, Ability v2, DB")

    st.markdown("#### Upload race file")
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

    st.markdown("#### Weight Scenario (baseline 60 kg)")
    USE_WEIGHT = st.toggle("Enable weight adjustment engine", value=False,
                           help="If file has 'Horse Weight', that value is used; else the global Δ below applies to all.")
    weight_delta = st.number_input("Global Δ weight (kg)", min_value=-10.0, max_value=+10.0, value=0.0, step=0.5,
                                   help="Applied only when no 'Horse Weight' column present (or if you later force scenario).")

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

# ----------------------- File load & header normalization -------------------
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

# ----------------------- Load file --------------------------
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

# Quick integrity preview (full integrity summary appears again after metrics)
integrity_text, _miss, _bad = integrity_scan(work, race_distance_input, split_step)
st.caption(f"Integrity: {integrity_text or 'OK'}")
if split_step == 200:
    st.caption("Finish column assumed to be the 200→0 segment (`Finish_Time`).")

# ======================= End of Batch A =======================
# ======================= Batch B — Weight model (helpers + apply) =======================

def _weight_sensitivity_per_kg(distance_m: float) -> float:
    """
    Distance-sensitive time impact per kilogram (fraction per kg).
    Smoothly ramps from ~0.08%/kg (sprints) to ~0.18%/kg (staying trips).
    """
    dm = float(distance_m or 1200)
    lo_dm, hi_dm = 800.0, 3200.0
    lo, hi = 0.0008, 0.0018
    if dm <= lo_dm: return lo
    if dm >= hi_dm: return hi
    t = (dm - lo_dm) / (hi_dm - lo_dm)
    return lo + t * (hi - lo)

def _derive_weight_deltas(df: pd.DataFrame, *, baseline_kg: float, ui_delta_kg: float, use_weight: bool) -> pd.Series:
    """
    Returns kg delta per row (weight - baseline).
    Priority:
      1) If a column named 'Horse Weight' (case-insensitive match) exists → use it (minus baseline).
      2) Else if use_weight is True → apply global scenario delta (ui_delta_kg) uniformly.
      3) Else → 0 for everyone (no effect).
    """
    candidates = [c for c in df.columns if c.strip().lower().replace("_"," ").replace("-", " ") == "horse weight"]
    if candidates:
        wcol = candidates[0]
        w_kg = pd.to_numeric(df[wcol], errors="coerce")
        return (w_kg - float(baseline_kg)).fillna(0.0)
    if use_weight:
        return pd.Series(np.full(len(df), float(ui_delta_kg)), index=df.index, dtype=float)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

def apply_weight_to_times(df: pd.DataFrame, *, distance_m: float, baseline_kg: float, ui_delta_kg: float, use_weight: bool) -> pd.DataFrame:
    """
    Returns a COPY of df where all *_Time columns (including Finish_Time) are
    multiplicatively adjusted by (1 + sens_per_kg * kg_delta).
    Heavier → slower (times increase); lighter → faster (times decrease).
    """
    out = df.copy()
    sens = float(_weight_sensitivity_per_kg(distance_m))
    kg_delta = _derive_weight_deltas(out, baseline_kg=baseline_kg, ui_delta_kg=ui_delta_kg, use_weight=use_weight)

    # Collect time columns
    time_cols = []
    for c in out.columns:
        lc = c.lower().strip()
        if lc.endswith("_time") or lc in ("finish_time","finish_split","finish"):
            time_cols.append(c)

    if not time_cols:
        out.attrs["WEIGHT_APPLIED"] = False
        out.attrs["WEIGHT_SENS_PER_KG"] = sens
        out.attrs["WEIGHT_BASELINE"] = baseline_kg
        out.attrs["WEIGHT_MODE"] = "none"
        return out

    # factor = 1 + sens * kg_delta  (clip to reasonable bounds for safety)
    factor = (1.0 + sens * kg_delta).clip(lower=0.90, upper=1.10)

    # Apply column-wise
    for c in time_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = (s * factor).where(s > 0, np.nan)

    out.attrs["WEIGHT_APPLIED"] = bool(kg_delta.abs().sum() > 0)
    out.attrs["WEIGHT_SENS_PER_KG"] = sens
    out.attrs["WEIGHT_BASELINE"] = baseline_kg
    out.attrs["WEIGHT_MODE"] = "file_column" if (("horse weight" in [x.strip().lower().replace("_"," ").replace("-"," ") for x in out.columns]) \
                          and kg_delta.abs().sum() > 0) else ("scenario" if use_weight else "none")
    out["_WeightΔ_kg"] = kg_delta
    return out

# --- make sure sidebar widgets are visible globally (Streamlit scopes them locally sometimes) ---
if "WEIGHT_BASELINE" not in globals() and "WEIGHT_BASELINE" in locals():
    WEIGHT_BASELINE = locals()["WEIGHT_BASELINE"]
if "WEIGHT_SENS_PER_KG" not in globals() and "WEIGHT_SENS_PER_KG" in locals():
    WEIGHT_SENS_PER_KG = locals()["WEIGHT_SENS_PER_KG"]
if "USE_WEIGHT" not in globals() and "USE_WEIGHT" in locals():
    USE_WEIGHT = locals()["USE_WEIGHT"]

# ---- Apply weight & proceed to metrics (Batch 2 will consume `work_w`) ----
BASE_WEIGHT = 60.0
ui_delta = weight_delta if 'weight_delta' in globals() else 0.0
use_weight_flag = bool(USE_WEIGHT) if 'USE_WEIGHT' in globals() else False

work_w = apply_weight_to_times(
    work,
    distance_m=float(race_distance_input),
    baseline_kg=BASE_WEIGHT,
    ui_delta_kg=float(ui_delta),
    use_weight=use_weight_flag
)

if work_w.attrs.get("WEIGHT_APPLIED", False):
    st.caption(
        f"Weight model active — baseline {work_w.attrs.get('WEIGHT_BASELINE',60):.1f} kg, "
        f"sensitivity {work_w.attrs.get('WEIGHT_SENS_PER_KG',0.0)*100:.3f}%/kg, mode={work_w.attrs.get('WEIGHT_MODE','')}."
    )

# ======================= End of Batch B =======================
# ======================= Batch 2 — Metrics Engine + Race Shape (SED/SCI/FRA) =======================
# (Drop-in: adaptive F-windows, robust RaceTime_s, and weight normalisation)

import math
import numpy as np
import pandas as pd

# -------- Stage helpers (100m/200m aware) --------
def collect_markers(df: pd.DataFrame) -> list[int]:
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    return sorted(set(marks), reverse=True)

def sum_times(row: pd.Series, cols: list[str]) -> float:
    vals = [as_num(row.get(c)) for c in cols]
    vals = [float(v) for v in vals if pd.notna(v) and v > 0]
    return float(np.sum(vals)) if vals else np.nan

def make_range_cols(D: float, start_inclusive: int, end_inclusive: int, step: int) -> list[str]:
    if start_inclusive < end_inclusive:
        return []
    want = list(range(int(start_inclusive), int(end_inclusive) - 1, -int(step)))
    return [f"{m}_Time" for m in want]

def stage_speed(row: pd.Series, cols: list[str], meters_per_split: float) -> float:
    if not cols: return np.nan
    tsum = sum_times(row, cols)
    if pd.isna(tsum) or tsum <= 0: return np.nan
    valid = [c for c in cols if pd.notna(row.get(c)) and as_num(row.get(c)) > 0]
    dist = meters_per_split * len(valid)
    if dist <= 0: return np.nan
    return dist / tsum

def grind_speed(row: pd.Series, step: int) -> float:
    """Grind = last 100 + finish (100m data) OR finish split only (200m data)."""
    if step == 100:
        t100 = as_num(row.get("100_Time"))
        tfin = as_num(row.get("Finish_Time"))
        parts, dist = [], 0.0
        if pd.notna(t100) and t100 > 0: parts.append(float(t100)); dist += 100.0
        if pd.notna(tfin) and tfin > 0: parts.append(float(tfin)); dist += 100.0
        if not parts or dist <= 0: return np.nan
        return dist / sum(parts)
    else:
        tfin = as_num(row.get("Finish_Time"))
        if pd.isna(tfin) or tfin <= 0: return np.nan
        return 200.0 / float(tfin)

# -------- Distance + context weights for PI v3.x --------
def _lerp(a, b, t): return a + (b - a) * float(t)

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

    # within-race bias tweak
    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale
        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]
        if bias > 0:
            delta = min(max_shift, ACC - 0.26)
            ACC -= delta; GR += delta
        elif bias < 0:
            delta = min(max_shift, GR - 0.18)
            GR  -= delta; ACC += delta
        GR = min(GR, 0.40)
        ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}
    return base

# ---- safety defaults in case widgets aren't in globals on this rerun
try:
    WEIGHT_BASELINE
except NameError:
    WEIGHT_BASELINE = 60.0

try:
    WEIGHT_SENS_PER_KG
except NameError:
    WEIGHT_SENS_PER_KG = 0.0011

try:
    USE_WEIGHT
except NameError:
    USE_WEIGHT = False

WEIGHTS_MAP = globals().get("WEIGHTS_MAP", WEIGHTS_MAP if "WEIGHTS_MAP" in globals() else None)
# -------- Metric builder (handles 100m and 200m) --------
def build_metrics_and_shape(
    df_in: pd.DataFrame,
    D_actual_m: float,
    step: int,
    use_cg: bool,
    dampen_cg: bool,
    use_race_shape: bool,
    *,
    use_weight: bool = False,
    baseline_kg: float = 60.0,
    kg_effect_pct: float = 0.60,
    weight_sens_per_kg: float = 0.0011,
    weights_map: dict | None = None,
    debug: bool = False,
):
    w = df_in.copy()

    # Finish_Pos as numeric if present
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    seg_markers = collect_markers(w)

    # Per-segment speeds (once)
    for m in seg_markers:
        w[f"spd_{m}"] = (step * 1.0) / as_num(w.get(f"{m}_Time"))
    w["spd_Finish"] = (
        (100.0 if step == 100 else 200.0) / as_num(w.get("Finish_Time"))
        if "Finish_Time" in w.columns
        else np.nan
    )

    # Robust RaceTime_s = sum of the *_Time columns that actually exist (+ Finish_Time if present)
    if seg_markers:
        wanted = list(range(int(D_actual_m) - int(step), int(step) - 1, -int(step)))
        cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in w.columns]
        if "Finish_Time" in w.columns:
            cols += ["Finish_Time"]
        w["RaceTime_s"] = w[cols].apply(pd.to_numeric, errors="coerce").mask(lambda s: s <= 0).sum(axis=1)
    else:
        # fallback (rare)
        w["RaceTime_s"] = as_num(w.get("RaceTime_s"))

    # ---------- Build stage composite speeds (ADAPTIVE WINDOWS) ----------
    D = float(D_actual_m)
    markers = collect_markers(w)

    def _adaptive_f_window_cols(D, step, markers):
        """
        Returns (f_cols, f_dist) according to rules:
          100 m splits:
             - normal: F200 = [D-100, D-200]
             - if D ends with 50: F150 = [D-50, D-150]
          200 m splits:
             - first span ≈100/160/200/250 → pick first marker and label distance accordingly
        """
        if not markers:
            return [], 0.0
        m1 = int(markers[0])
        first_span = D - m1
        if int(step) == 100:
            if int(D) % 100 == 50:
                wanted = [int(D - 50), int(D - 150)]
                cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in w.columns]
                dist = 150.0 if len(cols) == 2 else 100.0 * len(cols)
                return cols, float(dist)
            else:
                wanted = [int(D - 100), int(D - 200)]
                cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in w.columns]
                dist = 100.0 * len(cols)
                return cols, float(dist)
        # step == 200
        cols = [f"{m1}_Time"] if f"{m1}_Time" in w.columns else []
        if   first_span <= 120: dist = 100.0
        elif first_span <= 180: dist = 160.0
        elif first_span <= 220: dist = 200.0
        else:                   dist = 250.0
        return cols, float(dist)

    def _adaptive_tssp_start(D, step):
        if int(step) == 100:
            return int(D - 150) if (int(D) % 100 == 50) else int(D - 300)
        if not markers:
            return int(D - 400)
        m1 = int(markers[0]); first_span = D - m1
        if   first_span <= 120: return int(D - 100)
        elif first_span <= 180: return int(D - 150)
        elif first_span <= 220: return int(D - 400)
        else:                   return int(D - 250)

    # ---- F-window (early)
    f_cols, f_dist = _adaptive_f_window_cols(D, int(step), markers)
    w["_F_spd"] = w.apply(
        lambda r: (f_dist / sum_times(r, f_cols))
        if (f_cols and pd.notna(sum_times(r, f_cols)) and sum_times(r, f_cols) > 0)
        else np.nan,
        axis=1
    )

    # ---- tsSPI (mid)
    tssp_start = _adaptive_tssp_start(D, int(step))
    tssp_cols = [c for c in make_range_cols(D, tssp_start, 600, int(step)) if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: stage_speed(r, tssp_cols, float(step)), axis=1)

    # ---- Accel (600→200)
    if int(step) == 100:
        accel_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]
    else:
        accel_cols = [c for c in [f"{m}_Time" for m in [600,400]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: stage_speed(r, accel_cols, float(step)), axis=1)

    # ---- Grind (finish)
    w["_GR_spd"] = w.apply(lambda r: grind_speed(r, int(step)), axis=1)

    # ---------- Map speeds → indices ----------
    def mad_std(x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        if x.size == 0: return np.nan
        med = np.median(x); mad = np.median(np.abs(x - med))
        return 1.4826 * mad

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

    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F_spd"],  errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"], errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"], errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],  errors="coerce"))

    # ---------- Weight normalisation of indices (optional) ----------
    if use_weight:
        # vector of kg per row (by Horse)
        w_horse = w.get("Horse", pd.Series([""] * len(w))).astype(str)
        kg_vec = []
        for nm in w_horse:
            kg = None
            if isinstance(weights_map, dict):
                kg = weights_map.get(str(nm).strip())
            if kg is None or not np.isfinite(kg):
                kg = float(baseline_kg)
            kg_vec.append(float(kg))
        kg_vec = np.asarray(kg_vec, dtype=float)

        # per-phase sensitivity
        PHASE_MULT = {"F200_idx":1.25, "tsSPI":1.00, "Accel":1.10, "Grind":0.80}

        def _adj(col):
            base = pd.to_numeric(w[col], errors="coerce").astype(float).to_numpy()
            mult = 1.0 + float(weight_sens_per_kg) * PHASE_MULT[col] * (float(baseline_kg) - kg_vec)
            mult = np.clip(mult, 0.90, 1.10)
            return base * mult

        w["F200_eff"]  = _adj("F200_idx")
        w["tsSPI_eff"] = _adj("tsSPI")
        w["Accel_eff"] = _adj("Accel")
        w["Grind_eff"] = _adj("Grind")
    else:
        w["F200_eff"]  = pd.to_numeric(w["F200_idx"], errors="coerce")
        w["tsSPI_eff"] = pd.to_numeric(w["tsSPI"],    errors="coerce")
        w["Accel_eff"] = pd.to_numeric(w["Accel"],    errors="coerce")
        w["Grind_eff"] = pd.to_numeric(w["Grind"],    errors="coerce")

    # ---------- Corrected Grind (CG) ----------
    ACC_field = pd.to_numeric(w["_ACC_spd"], errors="coerce").mean(skipna=True)
    GR_field  = pd.to_numeric(w["_GR_spd"],  errors="coerce").mean(skipna=True)
    FSR = float(GR_field / ACC_field) if (ACC_field and ACC_field > 0 and math.isfinite(ACC_field) and math.isfinite(GR_field)) else np.nan
    if not math.isfinite(FSR): FSR = 1.0
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

    # Apply CG to the *effective* Grind (post-weight if used)
    GR_BASE_SER = pd.to_numeric(w["Grind_eff"], errors="coerce")
    w["Grind_CG"] = (GR_BASE_SER - w["GrindAdjPts"]).clip(lower=90.0, upper=110.0)

    def _fade_cap(g, dg):
        if not math.isfinite(g) or not math.isfinite(dg): return g
        if dg < 97.0 and g > 100.0:
            return 100.0 + 0.5 * (g - 100.0)
        return g
    w["Grind_CG"] = [_fade_cap(g, dg) for g, dg in zip(w["Grind_CG"], w["DeltaG"])]

    # ---------- PI v3.2 ----------
    acc_med = w["Accel_eff"].median(skipna=True)
    grd_med = (w["Grind_CG"] if use_cg else w["Grind_eff"]).median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(D), acc_med, grd_med)

    # Optional dampen on collapse
    if use_cg and dampen_cg and CollapseSeverity >= 3.0:
        d = min(0.02 + 0.01 * (CollapseSeverity - 3.0), 0.08)
        shift = min(d, PI_W["Grind"])
        PI_W["Grind"] -= shift
        PI_W["Accel"] += shift * 0.5
        PI_W["tsSPI"] += shift * 0.5

    # choose effective columns for PI/GCI
    F200_COL = "F200_eff"
    MID_COL  = "tsSPI_eff"
    ACC_COL  = "Accel_eff"
    GR_COL   = "Grind_CG" if use_cg else "Grind_eff"

    def pi_pts_row(row):
        parts, weights = [], []
        val_map = {
            "F200_idx": row.get(F200_COL),
            "tsSPI":    row.get(MID_COL),
            "Accel":    row.get(ACC_COL),
            "Grind":    row.get(GR_COL)
        }
        for k, wgt in PI_W.items():
            v = val_map.get(k, np.nan)
            if pd.notna(v):
                parts.append(wgt * (float(v) - 100.0))
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

    # ---------- GCI ----------
    acc_med_g = w[ACC_COL].median(skipna=True)
    grd_med_g = (w["Grind_CG"] if use_cg else w["Grind_eff"]).median(skipna=True)
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

        LQ = 0.6 * map_pct(r.get(ACC_COL)) + 0.4 * map_pct(r.get(GR_COL))
        SS = map_pct(r.get(MID_COL))

        acc, grd_eff = r.get(ACC_COL), r.get(GR_COL)
        if pd.isna(acc) or pd.isna(grd_eff):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd_eff - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)

        score01 = (wT * T) + (wPACE * LQ) + (wSS * SS) + (wEFF * EFF)
        gci_vals.append(round(10.0 * score01, 3))

    w["GCI"] = gci_vals

    # ---------- EARLY/LATE composite indices ----------
    w["EARLY_idx"] = 0.6 * pd.to_numeric(w[F200_COL], errors="coerce") + 0.4 * pd.to_numeric(w[MID_COL], errors="coerce")
    w["LATE_idx"]  = 0.5 * pd.to_numeric(w[ACC_COL],  errors="coerce") + 0.5 * pd.to_numeric(w[GR_COL],  errors="coerce")

    # ---------- RACE SHAPE MODULE v2.2 (SED/SCI + FRA) ----------
    shape_tag   = "EVEN"
    sci         = 1.0
    fra_applied = 0

    if use_race_shape:
        def _mad_std(v):
            v = pd.to_numeric(v, errors="coerce") - 100.0
            v = v.dropna().to_numpy()
            if v.size == 0: return np.nan
            return 1.4826 * np.median(np.abs(v - np.median(v)))

        E_med = float(pd.to_numeric(w["EARLY_idx"], errors="coerce").median(skipna=True))
        M_med = float(pd.to_numeric(w[MID_COL],     errors="coerce").median(skipna=True))
        L_med = float(pd.to_numeric(w["LATE_idx"],  errors="coerce").median(skipna=True))

        dE, dM, dL = (E_med - 100.0), (M_med - 100.0), (L_med - 100.0)

        gE = max(2.2, 0.6 * (_mad_std(w["EARLY_idx"]) if np.isfinite(_mad_std(w["EARLY_idx"])) else 2.0))
        gL = max(2.2, 0.6 * (_mad_std(w["LATE_idx"])  if np.isfinite(_mad_std(w["LATE_idx"]))  else 2.0))
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

        # FRA — gentle nudges
        w["PI_RS"]  = w["PI"].astype(float)
        w["GCI_RS"] = w["GCI"].astype(float)

        if (shape_tag == "SLOW_EARLY") and (sci >= 0.60):
            f = 0.12 + 0.08 * (sci - 0.60) / 0.40
            late_excess = (pd.to_numeric(w["LATE_idx"], errors="coerce") - 100.0).clip(lower=0.0, upper=8.0).fillna(0.0)
            w["PI_RS"]  = (w["PI"]  - f * (late_excess / 4.0)).clip(0.0, 10.0)
            w["GCI_RS"] = (w["GCI"] - f * (late_excess / 3.0)).clip(0.0, 10.0)
            fra_applied = 1

        elif (shape_tag == "FAST_EARLY") and (sci >= 0.60):
            f2 = 0.10 + 0.05 * (sci - 0.60) / 0.40
            sturdiness = (pd.to_numeric(w[GR_COL], errors="coerce") - 100.0).clip(lower=0.0, upper=6.0).fillna(0.0)
            w["PI_RS"]  = (w["PI"]  + f2 * (sturdiness / 4.0)).clip(0.0, 10.0)
            w["GCI_RS"] = (w["GCI"] + f2 * (sturdiness / 3.0)).clip(0.0, 10.0)
            fra_applied = 1
    else:
        w["PI_RS"]  = w["PI"].astype(float)
        w["GCI_RS"] = w["GCI"].astype(float)

    # ---------- Final rounding ----------
    for c in ["F200_eff","tsSPI_eff","Accel_eff","Grind_eff",
              "EARLY_idx","LATE_idx","F200_idx","tsSPI","Accel","Grind","Grind_CG",
              "PI","PI_RS","GCI","GCI_RS","RaceTime_s","DeltaG","FinisherFactor","GrindAdjPts"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce").round(3)

    # ---------- Attach race-level diagnostics ----------
    w.attrs["FSR"] = float(FSR)
    w.attrs["CollapseSeverity"] = float(CollapseSeverity)
    w.attrs["GR_COL"] = GR_COL
    w.attrs["STEP"] = int(step)
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

# ---- Compute metrics + race shape now (KEYWORD-ONLY call) ----
# --- Safety: ensure baseline + sensitivity from sidebar exist in global scope ---
if "WEIGHT_BASELINE" not in globals() and "WEIGHT_BASELINE" in locals():
    WEIGHT_BASELINE = locals()["WEIGHT_BASELINE"]
if "WEIGHT_SENS_PER_KG" not in globals() and "WEIGHT_SENS_PER_KG" in locals():
    WEIGHT_SENS_PER_KG = locals()["WEIGHT_SENS_PER_KG"]
try:
    metrics, seg_markers = build_metrics_and_shape(
        df_in=work,
        D_actual_m=float(race_distance_input),
        step=int(split_step),
        use_cg=USE_CG,
        dampen_cg=DAMPEN_CG,
        use_race_shape=USE_RACE_SHAPE,
        # weight engine (from the sidebar + editor)
        use_weight=USE_WEIGHT,
        baseline_kg=float(WEIGHT_BASELINE),
        kg_effect_pct=float(0.60),
        weight_sens_per_kg=float(WEIGHT_SENS_PER_KG),
        weights_map=WEIGHTS_MAP,
        debug=DEBUG,
    )
except Exception as e:
    st.error("Metric computation failed.")
    st.exception(e)
    st.stop()
# ======================= End of Batch 2 =======================
