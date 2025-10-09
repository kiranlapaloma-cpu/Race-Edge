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
