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
# ======================= Batch 2 — Metrics Engine + Race Shape (SED / SCI / FRA v2.2) =======================
import math
import numpy as np
import pandas as pd

# ----------------------- Stage helpers -----------------------
# ----------------------- Integrity helpers (100/200 m aware; odd distances OK) -------------------
def expected_segments(distance_m: float, step: int, *, from_existing: list[str] | None = None) -> list[str]:
    """
    Build the list of expected *_Time columns for integrity display.

    • If from_existing is provided (list of actual *_Time columns found), we
      just use those (sorted high→low) and append Finish_Time if present.
    • Else, we align the first marker to the FLOOR multiple of step, not D-step.
      e.g. 1250 @ 100m → 1200,1100,...,100 and then Finish_Time.
           1450 @ 200m → 1400,1200,...,200 and then Finish_Time.
    """
    if from_existing:
        marks = []
        for c in from_existing:
            if c.endswith("_Time") and c != "Finish_Time":
                try:
                    marks.append(int(c.split("_")[0]))
                except Exception:
                    pass
        marks = sorted(set(marks), reverse=True)
        want = [f"{m}_Time" for m in marks]
        if "Finish_Time" in from_existing:
            want.append("Finish_Time")
        return want

    D = int(distance_m)
    step = int(step)
    start = (D // step) * step               # floor to clean hundred/two-hundred
    if start == 0: start = step
    want = [f"{m}_Time" for m in range(start, step-1, -step)]
    want.append("Finish_Time")
    return want

def integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    """
    Friendly integrity line:
    • We no longer accuse the file of “missing 1150/1050…” for 1250 races.
    • We only warn about: completely missing all segment columns,
      missing Finish_Time, and any nonpositive / nonnumeric values present.
    """
    # What *_Time columns do we actually have?
    actual_cols = [c for c in df.columns if c.endswith("_Time")]
    # Expected, aligned to the file’s reality
    exp_cols = expected_segments(distance_m, step, from_existing=actual_cols)

    missing = [c for c in exp_cols if c not in df.columns]
    invalid_counts = {}
    for c in exp_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            invalid_counts[c] = int(((s <= 0) | s.isna()).sum())

    msgs = []
    # Only warn “Missing segments” when we have literally none
    have_any_split = any(c.endswith("_Time") and c != "Finish_Time" for c in df.columns)
    if not have_any_split:
        msgs.append("No segment columns found")
    # Finish_Time is important to call out explicitly
    if "Finish_Time" not in df.columns:
        msgs.append("Finish_Time missing")
    # Invalid values message (suppress zeros for columns that don’t exist)
    bads = [f"{k} ({v} rows)" for k, v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))

    return " • ".join(msgs), missing, invalid_counts
def collect_markers(df):
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try: marks.append(int(c.split("_")[0]))
            except Exception: pass
    return sorted(set(marks), reverse=True)

def sum_times(row, cols):
    vals = [as_num(row.get(c)) for c in cols]
    vals = [v for v in vals if pd.notna(v) and v > 0]
    return np.sum(vals) if vals else np.nan

def stage_speed(row, cols, meters_per_split):
    if not cols: return np.nan
    tsum = sum_times(row, cols)
    if pd.isna(tsum) or tsum <= 0: return np.nan
    dist = meters_per_split * len([c for c in cols if pd.notna(row.get(c))])
    return np.nan if dist <= 0 else dist / tsum

def grind_speed(row, step):
    if step == 100:
        t100, tfin = as_num(row.get("100_Time")), as_num(row.get("Finish_Time"))
        parts=[t for t in [t100,tfin] if pd.notna(t) and t>0]
        return np.nan if not parts else (100*len(parts))/sum(parts)
    else:
        tfin = as_num(row.get("Finish_Time"))
        return np.nan if pd.isna(tfin) or tfin<=0 else 200.0/float(tfin)

# ----------------------- Metric builder -----------------------
def build_metrics_and_shape(df_in: pd.DataFrame,
                            D_actual_m: float,
                            step: int,
                            use_cg: bool,
                            dampen_cg: bool,
                            use_race_shape: bool,
                            debug: bool):
    w = df_in.copy()
    D = float(D_actual_m)

    # -------- Inferred finish positions (if missing) --------
    finish_inferred = 0
    if "Finish_Pos" not in w.columns or w["Finish_Pos"].isna().mean() > 0.7:
        tmp = w.reset_index(drop=False).rename(columns={"index":"_row"})
        tmp["_rt"] = pd.to_numeric(tmp.get("RaceTime_s", np.nan), errors="coerce")
        tmp = tmp.sort_values(by=["_rt"], ascending=True).reset_index(drop=True)
        tmp["Finish_Pos"] = np.arange(1, len(tmp)+1)
        w = tmp.sort_values("_row").drop(columns=["_row","_rt"]).reset_index(drop=True)
        finish_inferred = 1
    else:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # -------- Speeds --------
    seg_markers = collect_markers(w)
    for m in seg_markers:
        w[f"spd_{m}"] = (step * 1.0) / as_num(w.get(f"{m}_Time"))
    w["spd_Finish"] = (100.0 if step==100 else 200.0) / as_num(w.get("Finish_Time")) if "Finish_Time" in w.columns else np.nan

        # -------- Stage composites (ADAPTIVE to odd distances) --------
    seg_markers = collect_markers(w)
    step = int(step)

    def _adaptive_f_window_cols(D, step, markers):
        """
        Early window:
          100m splits:
            • D ends with 50  → F150 = [D-50, D-150] if present
            • else            → F200 = [D-100, D-200]
          200m splits (use first panel's true span):
            • first span ~160 → F160 = [first marker]
            • first span ~250 → F250 = [first marker]
            • first span ~100 → F100 = [first marker]
            • normal even     → F200 = [D-200]
        """
        if not markers:
            return [], 0.0
        first_m = int(markers[0])
        first_span = float(D - first_m)

        if step == 100:
            if int(D) % 100 == 50:
                want = [int(D - 50), int(D - 150)]
            else:
                want = [int(D - 100), int(D - 200)]
            cols = [f"{m}_Time" for m in want if f"{m}_Time" in w.columns]
            dist = 50.0 + 100.0 if (len(cols) == 2 and int(D) % 100 == 50) else 100.0 * len(cols)
            return cols, dist

        # step == 200
        col = f"{first_m}_Time"
        if col not in w.columns:
            return [], 0.0
        if first_span <= 120:  dist = 100.0
        elif first_span <= 180: dist = 160.0
        elif first_span <= 220: dist = 200.0
        else:                   dist = 250.0
        return [col], dist

    def _adaptive_tssp_start(D, step, markers):
        """
        MID window (tsSPI) start:
          100m:   D-300 normally, D-150 for D%100==50 (e.g., 1250)
          200m:   depends on first span (≈100/160/200/250) → start at D-100/150/400/250
        """
        if step == 100:
            return int(D - 150) if (int(D) % 100 == 50) else int(D - 300)
        if not markers:
            return int(D - 400)
        first_m = int(markers[0])
        first_span = float(D - first_m)
        if first_span <= 120:   return int(D - 100)
        if first_span <= 180:   return int(D - 150)
        if first_span <= 220:   return int(D - 400)
        return int(D - 250)

    # F-window speed
    f_cols, f_dist = _adaptive_f_window_cols(D, step, seg_markers)
    w["_F_spd"] = w.apply(
        lambda r: (f_dist / sum_times(r, f_cols)) if (f_cols and pd.notna(sum_times(r, f_cols)) and sum_times(r, f_cols) > 0)
        else np.nan, axis=1
    )

    # MID (tsSPI) speed
    tssp_start = _adaptive_tssp_start(D, step, seg_markers)
    mid_cols = [c for c in [f"{m}_Time" for m in range(tssp_start, 600-1, -step)] if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: stage_speed(r, mid_cols, float(step)), axis=1)

    # Accel (unchanged windows)
    if step == 100:
        acc_cols = [c for c in [f"{m}_Time" for m in [500, 400, 300, 200]] if c in w.columns]
    else:
        acc_cols = [c for c in [f"{m}_Time" for m in [600, 400]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: stage_speed(r, acc_cols, float(step)), axis=1)

    # Grind (unchanged)
    w["_GR_spd"] = w.apply(lambda r: grind_speed(r, step), axis=1)

        # -------- Robust total race time (sum of all segment times) --------
    # Works whether the file uses Finish_Time, Finish_Split or Finish (headers are
    # usually normalized earlier, but we guard again here).
    # We also accept any "<meters>_Time" or "<meters>_Split" columns.
    if ("RaceTime_s" not in w.columns) or (pd.to_numeric(w["RaceTime_s"], errors="coerce").isna().all()):
        # Collect candidate time columns safely
        time_cols = []
        for c in w.columns:
            lc = str(c).strip()
            if lc.lower() in ("finish_time", "finish_split", "finish"):
                time_cols.append(c)
            elif lc.endswith("_Time") or lc.endswith("_time") or lc.endswith("_Split") or lc.endswith("_split"):
                # only accept if it looks like a distance prefix, e.g. "1200_Time"
                parts = lc.split("_", 1)
                if parts and parts[0].isdigit():
                    time_cols.append(c)

        # If we somehow missed a Finish* column but have one in the frame, add it
        for cand in ("Finish_Time", "Finish_Split", "Finish"):
            if cand in w.columns and cand not in time_cols:
                time_cols.append(cand)

        # Sum all valid, positive times
        def _sum_race_time(row):
            vals = []
            for c in time_cols:
                v = pd.to_numeric(row.get(c), errors="coerce")
                if pd.notna(v) and float(v) > 0.0:
                    vals.append(float(v))
            return np.sum(vals) if vals else np.nan

        w["RaceTime_s"] = w.apply(_sum_race_time, axis=1)
    else:
        # Ensure numeric and clean non-positive entries
        w["RaceTime_s"] = pd.to_numeric(w["RaceTime_s"], errors="coerce")
        w.loc[(w["RaceTime_s"] <= 0) | (~np.isfinite(w["RaceTime_s"])) , "RaceTime_s"] = np.nan                            

    # -------- Speed→index conversion --------
    def speed_to_index(spd):
        med=spd.median(skipna=True)
        return 100.0*(spd/med)
    w["F200_idx"]=speed_to_index(pd.to_numeric(w["_F_spd"],errors="coerce"))
    w["tsSPI"]=speed_to_index(pd.to_numeric(w["_MID_spd"],errors="coerce"))
    w["Accel"]=speed_to_index(pd.to_numeric(w["_ACC_spd"],errors="coerce"))
    w["Grind"]=speed_to_index(pd.to_numeric(w["_GR_spd"],errors="coerce"))

    # -------- Corrected Grind (CG) --------
    ACC_field=w["_ACC_spd"].mean(skipna=True)
    GR_field=w["_GR_spd"].mean(skipna=True)
    FSR=float(GR_field/ACC_field) if ACC_field and ACC_field>0 else 1.0
    CollapseSeverity=float(min(10.0,max(0.0,(0.95-FSR)*100.0)))
    def delta_g_row(r):
        mid,gr=float(r.get("_MID_spd",np.nan)),float(r.get("_GR_spd",np.nan))
        return np.nan if not (math.isfinite(mid) and math.isfinite(gr) and mid>0) else 100.0*(gr/mid)
    w["DeltaG"]=w.apply(delta_g_row,axis=1)
    w["FinisherFactor"]=w["DeltaG"].apply(lambda dg:float(clamp((dg-98.0)/4.0,0.0,1.0)) if math.isfinite(dg) else 0.0)
    w["GrindAdjPts"]=(CollapseSeverity*(1.0-w["FinisherFactor"])).round(2)
    w["Grind_CG"]=(w["Grind"]-w["GrindAdjPts"]).clip(lower=90.0,upper=110.0)

    # -------- PI v3.2 (pre Race Shape) --------
    acc_med=w["Accel"].median(skipna=True)
    grd_med=(w["Grind_CG"] if use_cg else w["Grind"]).median(skipna=True)
    PI_W={"F200_idx":0.1,"tsSPI":0.37,"Accel":0.31,"Grind":0.22}
    GR_COL="Grind_CG" if use_cg else "Grind"
    def pi_pts_row(r):
        parts=[]
        for k,wgt in PI_W.items():
            v=r.get(GR_COL) if k=="Grind" else r.get(k)
            if pd.notna(v): parts.append(wgt*(v-100.0))
        return np.nan if not parts else sum(parts)
    pts=w.apply(pi_pts_row,axis=1)
    med=float(np.nanmedian(pts))
    sigma=mad_std(pts-med) or 0.75
    w["PI"]=(5.0+2.2*((pts-med)/sigma)).clip(0,10).round(2)

    # -------- GCI --------
    winner_time=w["spd_Finish"].max()
    def map_pct(x,lo=98.0,hi=104.0):
        return clamp((float(x)-lo)/(hi-lo),0.0,1.0) if pd.notna(x) else 0.0
    gci=[]
    for _,r in w.iterrows():
        LQ=0.6*map_pct(r.get("Accel"))+0.4*map_pct(r.get(GR_COL))
        SS=map_pct(r.get("tsSPI"))
        EFF=1.0-((abs(r.get("Accel",100)-100)+abs(r.get(GR_COL,100)-100))/16.0)
        gci.append(round(10*(0.25*LQ+0.35*SS+0.4*EFF),3))
    w["GCI"]=gci

    # -------- Race Shape v2.2 + FRA --------
    shape_tag="EVEN"; sci=1.0; fra_applied=0
    if use_race_shape:
        w["EARLY_idx"]=0.6*w["F200_idx"]+0.4*w["tsSPI"]
        w["LATE_idx"]=0.6*w["Accel"]+0.4*w[GR_COL]
        E_med,M_med,L_med=(w["EARLY_idx"].median(),w["tsSPI"].median(),w["LATE_idx"].median())
        dE,dL=E_med-100.0,L_med-100.0
        gE,gL=mad_std(w["EARLY_idx"]-100.0),mad_std(w["LATE_idx"]-100.0)
        delta_EL=w["LATE_idx"]-w["EARLY_idx"]
        sci_plus=(delta_EL>+1).mean(); sci_minus=(delta_EL<-1).mean()
        if dE<=-gE and dL>=gL and sci_plus>=0.55: shape_tag="SLOW_EARLY"
        elif dE>=gE and dL<=-gL and sci_minus>=0.55: shape_tag="FAST_EARLY"
        sci=float(max(sci_plus,sci_minus,1.0))
        w["PI_RS"]=w["PI"]; w["GCI_RS"]=w["GCI"]
        if shape_tag=="SLOW_EARLY" and sci>=0.6:
            f=0.12+0.08*(sci-0.6)/0.4
            late_ex=((w["Accel"]+w[GR_COL])/2.0-100).clip(lower=0,upper=8)
            w["PI_RS"]=(w["PI"]-f*(late_ex/4)).clip(0,10); w["GCI_RS"]=(w["GCI"]-f*(late_ex/3)).clip(0,10)
            fra_applied=1
        elif shape_tag=="FAST_EARLY" and sci>=0.6:
            f2=0.10+0.05*(sci-0.6)/0.4
            sturd=((w[GR_COL]-100)-(100-w["Accel"]).clip(lower=0)).clip(lower=0,upper=6)
            w["PI_RS"]=(w["PI"]+f2*(sturd/4)).clip(0,10); w["GCI_RS"]=(w["GCI"]+f2*(sturd/3)).clip(0,10)
            fra_applied=1
    else:
        w["PI_RS"]=w["PI"]; w["GCI_RS"]=w["GCI"]

    # -------- Round + attach attrs --------
    for c in ["F200_idx","tsSPI","Accel","Grind","Grind_CG","PI","GCI","PI_RS","GCI_RS","EARLY_idx","LATE_idx"]:
        if c in w.columns: w[c]=pd.to_numeric(w[c],errors="coerce").round(3)
    w.attrs.update({
        "FSR":float(FSR),"CollapseSeverity":float(CollapseSeverity),
        "GR_COL":GR_COL,"STEP":step,
        "SHAPE_TAG":shape_tag,"SCI":float(sci),"FRA_APPLIED":fra_applied,
        "Finish_Pos_Inferred":finish_inferred
    })
    if debug: st.write({"SHAPE":shape_tag,"SCI":sci,"FRA":fra_applied})
    return w, seg_markers

# ----------------------- Run metrics -----------------------
try:
    metrics, seg_markers = build_metrics_and_shape(
        work_w, float(race_distance_input), int(split_step),
        USE_CG, DAMPEN_CG, USE_RACE_SHAPE, DEBUG)
except Exception as e:
    st.error("Metric computation failed."); st.exception(e); st.stop()

# ----------------------- Header summary -----------------------
integrity_text, missing_cols, invalid_counts = integrity_scan(work, race_distance_input, split_step)
st.markdown(
    f"## Race Distance {int(race_distance_input)} m | Splits {split_step} m | "
    f"Shape **{metrics.attrs.get('SHAPE_TAG','EVEN')}** | SCI {metrics.attrs.get('SCI',1.0):.2f} | "
    f"FRA {'Yes' if metrics.attrs.get('FRA_APPLIED',0)==1 else 'No'} | "
    f"Finish Pos {'Inferred' if metrics.attrs.get('Finish_Pos_Inferred',0)==1 else 'From file'}"
)
if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    warn=[]
    if missing_cols: warn.append("Missing: "+", ".join(missing_cols))
    bads=[f"{k} ({v} rows)" for k,v in invalid_counts.items() if v>0]
    if bads: warn.append("Invalid/zero times: "+", ".join(bads))
    if warn: st.markdown(f"*(⚠ {' • '.join(warn)})*")

# ----------------------- Metrics table -----------------------
show_cols=["Horse","Finish_Pos","RaceTime_s",
           "F200_idx","tsSPI","Accel","Grind","Grind_CG",
           "EARLY_idx","LATE_idx","PI","PI_RS","GCI","GCI_RS"]
for c in show_cols:
    if c not in metrics.columns: metrics[c]=np.nan
display_df=metrics[show_cols].copy()
display_df=display_df.sort_values(["PI_RS","Finish_Pos"],ascending=[False,True])
st.dataframe(display_df,use_container_width=True)
st.caption(
    f"CG={'ON' if USE_CG else 'OFF'} (FSR {metrics.attrs.get('FSR',1.0):.3f}; Collapse {metrics.attrs.get('CollapseSeverity',0.0):.1f}). "
    f"Race Shape {metrics.attrs.get('SHAPE_TAG','EVEN')} (SCI {metrics.attrs.get('SCI',1.0):.2f}; FRA {'Yes' if metrics.attrs.get('FRA_APPLIED',0)==1 else 'No'})."
)
# ======================= End of Batch 2 =======================
