# Race Edge - PI v3.2  |  streamlit_app.py

# Thin Streamlit orchestrator.  All algorithms live in:

# utils.py, data_io.py, metrics.py, models.py, visuals.py, db.py

import io
import numpy as np
import pandas as pd
import streamlit as st

# – patch Streamlit emitters before any st.* calls –––––––––––––

from utils import sanitize

pd.options.mode.use_inf_as_na = True

_orig = {fn: getattr(st, fn) for fn in
(“write”, “json”, “metric”, “dataframe”, “table”)}

st.write      = lambda *a, **k: _orig[“write”](*[sanitize(x) for x in a],
**{k_: sanitize(v) for k_, v in k.items()})
st.json       = lambda o, *a, **k: _orig[“json”](sanitize(o), *a, **k)
st.dataframe  = lambda d=None, *a, **k: _orig[“dataframe”](
sanitize(d).reset_index(drop=True) if isinstance(sanitize(d), pd.DataFrame)
else sanitize(d), *a, **k)
st.table      = lambda d=None, *a, **k: _orig[“table”](sanitize(d), *a, **k)

def _safe_metric(label, value, delta=None, *a, **k):
v = sanitize(value); d = sanitize(delta)
return _orig[“metric”](label,
“-” if v is None else v,
“-” if (delta is not None and d is None) else d,
*a, **k)
st.metric = _safe_metric

if hasattr(st, “data_editor”):
_oe = st.data_editor
st.data_editor = lambda d=None, *a, **k: _oe(sanitize(d), *a, **k)

_od = st.download_button
def _safe_download(*a, **k):
if “data” in k:
k[“data”] = sanitize(k[“data”])
if isinstance(k[“data”], (pd.DataFrame, pd.Series)):
k[“data”] = k[“data”].to_csv(index=False).encode(“utf-8”)
return _od(*a, **k)
st.download_button = _safe_download

# – imports —————————————————————–

from data_io  import load_file, detect_step, integrity_scan, collect_markers
from metrics  import (build_metrics_and_shape, compute_rqs, compute_rps,
classify_race_profile)
from models   import (ahead_of_handicap, winning_dna, hidden_horses,
pwx_efi, fatigue_gradient, compute_pwr400,
context_aware_reliability, xwin)
from visuals  import shape_map, pace_curve, power_freshness_map, fig_to_png
from db       import init_db, save_race, query_horse, query_recent_races

APP_VERSION    = “3.2”
DB_DEFAULT     = “race_edge.db”
ALL_MODULES    = [
“Sectional Metrics”,
“Race Class Summary”,
“Ahead of Handicap”,
“Shape Map”,
“Pace Curve”,
“Winning DNA”,
“Hidden Horses”,
“PWX + EFI”,
“Fatigue Gradient”,
“PWR400”,
“Power-Freshness Map”,
“R&V (CAR)”,
“xWin”,
]
DEFAULT_MODULES = [
“Sectional Metrics”, “Race Class Summary”,
“Shape Map”, “Hidden Horses”, “xWin”,
]

# ____________________________________________________________________________

# PAGE CONFIG

# ____________________________________________________________________________

st.set_page_config(
page_title=f”Race Edge v{APP_VERSION}”,
page_icon=”[RE]”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# – minimal custom CSS —————————————————––

st.markdown(”””

<style>
  [data-testid="stSidebar"] { min-width: 280px; }
  .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
  .badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 6px; font-weight: 700;
    font-size: 0.88rem; color: white; margin: 2px 0;
  }
  .section-divider { border-top: 1px solid rgba(128,128,128,0.25); margin: 1.2rem 0 0.6rem; }
</style>

“””, unsafe_allow_html=True)

# ____________________________________________________________________________

# SIDEBAR

# ____________________________________________________________________________

with st.sidebar:
st.markdown(f”## [RE] Race Edge `v{APP_VERSION}`”)
st.caption(“PI v3.2 . Race Shape . CG . Hidden v2 . DB”)
st.divider()

```
# -- Upload --------------------------------------------------------------
st.markdown("### Upload Race")
up = st.file_uploader(
    "CSV or XLSX with 100m / 200m splits",
    type=["csv", "xlsx", "xls"],
    help="Accepts `Finish_Time`, `Finish_Split`, or `Finish` as the final column.",
)
race_distance_input = st.number_input(
    "Race Distance (m)", min_value=800, max_value=4000, step=50, value=1600
)

st.divider()
st.markdown("### Model Settings")

USE_CG = st.toggle("Corrected Grind (CG)", value=True,
                   help="Adjust Grind when the field finish collapses.")
DAMPEN_CG = st.toggle("Dampen Grind weight if collapsed", value=True)
USE_RACE_SHAPE = st.toggle("Race Shape module (SED/FRA/SCI)", value=True)

USE_GOING  = st.toggle("Going Adjustment", value=True,
                       help="Adjusts PI weighting only - not indices or GCI.")
GOING_TYPE = st.selectbox(
    "Track Going", ["Good", "Firm", "Soft", "Heavy"], index=0,
    disabled=not USE_GOING,
) if USE_GOING else "Good"

WIND_ON  = st.toggle("Wind affected race?", value=False)
WIND_TAG = st.selectbox(
    "Wind note", ["Headwind", "Tailwind", "Crosswind", "Negligible"],
    disabled=not WIND_ON,
) if WIND_ON else "None"

st.divider()
st.markdown("### Modules to Run")
ACTIVE_MODULES = st.multiselect(
    "Select modules",
    ALL_MODULES,
    default=DEFAULT_MODULES,
    help="Uncheck modules you don't need to speed up the analysis.",
)
if st.button("[*] Enable All"):
    ACTIVE_MODULES = ALL_MODULES

st.divider()
st.markdown("### Database")
db_path    = st.text_input("DB path", value=DB_DEFAULT)
col_init, col_save = st.columns(2)
with col_init:
    if st.button("Init DB"):
        ok, msg = init_db(db_path)
        st.success(msg) if ok else st.error(msg)

SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
DEBUG         = st.toggle("Debug info", value=False)
```

# ____________________________________________________________________________

# STOP UNTIL FILE UPLOADED

# ____________________________________________________________________________

if not up:
st.markdown(”## Welcome to Race Edge [RE]”)
st.info(“Upload a race CSV / XLSX in the sidebar to begin.”)
st.markdown(”””
**Expected columns:**

- `Horse` - horse name
- `<meters>_Time` e.g. `1400_Time`, `1200_Time` … `100_Time`, `Finish_Time`
- `Finish_Pos` (optional - auto-generated if missing)
- `Horse Weight` / `Weight` / `Wt` (optional - used for mass-aware PI and PWR400)

Both **100m** and **200m** split files are accepted.
“””)
st.stop()

# ____________________________________________________________________________

# LOAD FILE

# ____________________________________________________________________________

try:
work, alias_notes = load_file(up)
st.success(f”[OK] **{up.name}** loaded - {len(work)} runners”)
except Exception as e:
st.error(“Failed to read file.”)
if DEBUG: st.exception(e)
st.stop()

split_step = detect_step(work)
st.markdown(f”**Detected split step:** `{split_step} m`”)

if alias_notes and SHOW_WARNINGS:
st.info(“Header aliases: “ + “ . “.join(alias_notes))

with st.expander(”[>] Raw data preview”, expanded=False):
st.dataframe(work.head(15), use_container_width=True)

integrity_text, _miss, _bad = integrity_scan(work, race_distance_input, split_step)
if integrity_text != “OK” and SHOW_WARNINGS:
st.caption(f”[!] Integrity: {integrity_text}”)

# ____________________________________________________________________________

# COMPUTE METRICS

# ____________________________________________________________________________

try:
metrics, seg_markers = build_metrics_and_shape(
work,
float(race_distance_input),
int(split_step),
USE_CG,
DAMPEN_CG,
USE_RACE_SHAPE,
going=GOING_TYPE,
debug=DEBUG,
)
except Exception as e:
st.error(“Metric computation failed.”)
st.exception(e)
st.stop()

# Ensure Horse Weight column exists for downstream modules

if “Horse Weight” not in metrics.columns:
wt_candidates = [“Horse_Weight”, “Wt”, “Weight”, “Weight (kg)”]
src = next((c for c in wt_candidates if c in metrics.columns), None)
if src:
metrics[“Horse Weight”] = pd.to_numeric(metrics[src], errors=“coerce”).fillna(60.0)
else:
metrics[“Horse Weight”] = 60.0

metrics.attrs[“WIND_AFFECTED”] = bool(WIND_ON)
metrics.attrs[“WIND_TAG”]      = str(WIND_TAG)

# Race-level scores

metrics.attrs[“RQS”] = compute_rqs(metrics, metrics.attrs)
metrics.attrs[“RPS”] = compute_rps(metrics)
_profile_label, _profile_color = classify_race_profile(
float(metrics.attrs.get(“RQS”, np.nan)),
float(metrics.attrs.get(“RPS”, np.nan)),
)
metrics.attrs[“RACE_PROFILE”]       = _profile_label
metrics.attrs[“RACE_PROFILE_COLOR”] = _profile_color

# ____________________________________________________________________________

# RACE HEADER

# ____________________________________________________________________________

st.divider()
c1, c2, c3, c4 = st.columns(4)
with c1:
st.metric(“Distance”, f”{int(race_distance_input)}m”)
with c2:
st.metric(“Shape”, metrics.attrs.get(“SHAPE_TAG”, “EVEN”))
with c3:
st.metric(“RSI”, f”{metrics.attrs.get(‘RSI’, 0.0):+.2f} / 10”)
with c4:
st.metric(“SCI”, f”{metrics.attrs.get(‘SCI’, 0.0):.2f}”)

c5, c6, c7, c8 = st.columns(4)
with c5:
rqs_val = float(metrics.attrs.get(“RQS”, 0.0))
st.metric(“RQS (field quality)”, f”{rqs_val:.1f} / 100”)
with c6:
rps_val = float(metrics.attrs.get(“RPS”, 0.0))
st.metric(“RPS (peak strength)”, f”{rps_val:.1f} / 100”)
with c7:
st.metric(“Finish”, metrics.attrs.get(“FINISH_FLAV”, “Balanced”))
with c8:
st.metric(“Going (PI)”, metrics.attrs.get(“GOING”, “Good”))

# Profile badge + RQS tier

col_badge, col_rqs = st.columns([2, 3])
with col_badge:
st.markdown(
f’<span class="badge" style="background:{_profile_color}">’
f’{_profile_label}</span>’,
unsafe_allow_html=True,
)
st.caption(“RQS = depth/consistency . RPS = peak . Badge = depth vs dominance”)
with col_rqs:
rqs_color = (
“#27AE60” if rqs_val >= 80 else
“#F39C12” if rqs_val >= 65 else
“#E67E22” if rqs_val >= 45 else “#C0392B”
)
rqs_label = (
“Elite Class” if rqs_val >= 80 else
“Competitive Field” if rqs_val >= 65 else
“Moderate Class” if rqs_val >= 45 else “Weak Field”
)
st.markdown(
f’<span class="badge" style="background:{rqs_color}">’
f’RQS {rqs_val:.1f} / 100 - {rqs_label}</span>’,
unsafe_allow_html=True,
)

if WIND_ON:
st.info(f”[!] Wind note: **{WIND_TAG}** - informational only, not modelled.”)

st.divider()

# ____________________________________________________________________________

# HELPER: module active check

# ____________________________________________________________________________

def active(name: str) -> bool:
return name in ACTIVE_MODULES

GR_COL = metrics.attrs.get(“GR_COL”, “Grind”)

# ____________________________________________________________________________

# MODULE 1 - Sectional Metrics

# ____________________________________________________________________________

if active(“Sectional Metrics”):
st.markdown(”## Sectional Metrics”)

```
show_cols = [
    "Horse", "Finish_Pos", "RaceTime_s",
    "F200_idx", "tsSPI", "Accel", "Grind", "Grind_CG",
    "EARLY_idx", "LATE_idx", "GrindAdjPts", "DeltaG",
    "PI", "GCI", "GCI_RS", "RS_Component", "RSI_Cue",
]
tmp = metrics.copy()
for c in show_cols:
    if c not in tmp.columns: tmp[c] = np.nan

display_df = tmp[show_cols].copy()
_fs = pd.to_numeric(display_df["Finish_Pos"], errors="coerce").fillna(1e9)
display_df = (display_df.assign(_s=_fs)
              .sort_values(["PI", "_s"], ascending=[False, True])
              .drop(columns=["_s"]))

st.dataframe(display_df, use_container_width=True)

# Going note
pi_meta = metrics.attrs.get("PI_GOING_META", {})
if pi_meta:
    mult  = pi_meta.get("multipliers", {})
    moved = [f"{k}x{mult[k]:.3f}" for k in ["Accel", "F200_idx", "tsSPI", "Grind"]
             if abs(mult.get(k, 1.0) - 1.0) >= 0.005]
    if moved:
        st.caption(f"Going: {pi_meta.get('going','Good')} - PI multipliers: {', '.join(moved)}")
st.caption("RSI: + slow-early (late favoured) . - fast-early . [BLU] with shape . [RED] against shape")
```

# ____________________________________________________________________________

# MODULE 2 - Race Class Summary

# ____________________________________________________________________________

if active(“Race Class Summary”):
st.markdown(”## Race Class Summary”)
gci_col = (“GCI_RS” if (“GCI_RS” in metrics.columns and
pd.to_numeric(metrics[“GCI_RS”], errors=“coerce”).notna().any())
else “GCI”)
s = pd.to_numeric(metrics.get(gci_col, pd.Series(dtype=float)),
errors=“coerce”).dropna()
if s.empty:
st.info(“No valid GCI values.”)
else:
med_v    = float(s.median())
mad_raw  = float(np.nanmedian(np.abs(s - med_v)))
q75, q25 = np.nanpercentile(s, 75), np.nanpercentile(s, 25)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric(“Mean GCI”,    f”{float(s.mean()):.2f}”)
with c2: st.metric(“Median GCI”,  f”{med_v:.2f}”)
with c3: st.metric(“Spread (MAD)”,f”{mad_raw:.2f}”)
with c4: st.metric(“IQR”,         f”{float(q75-q25):.2f}”)
st.caption(f”Source: **{gci_col}**”)

# ____________________________________________________________________________

# MODULE 3 - Ahead of Handicap

# ____________________________________________________________________________

if active(“Ahead of Handicap”):
st.markdown(”## Ahead of the Handicap”)
result = ahead_of_handicap(metrics, float(race_distance_input),
weight_col=“Horse Weight”)
if result is None:
st.info(“Insufficient data (need Horse, PI, and a weight column).”)
else:
st.dataframe(result, use_container_width=True)
ca, cb, cc = st.columns(3)
with ca: st.metric(“Field median PI”, f”{result.attrs[‘PI_med’]:.2f}”)
with cb:
corr = result.attrs[“corr”]
st.metric(“Weight<->PI corr”, “n/a” if not np.isfinite(corr) else f”{corr:+.2f}”)
with cc: st.metric(“beta_eff”, f”{result.attrs[‘beta_eff’]:.2f} PI/kg”)
st.caption(“RanAbove (kg) = how many kg above/below field median a horse effectively ran.”)
csv = result.to_csv(index=False).encode()
st.download_button(“Download CSV”, csv,
file_name=“ahead_of_handicap.csv”, mime=“text/csv”)

# ____________________________________________________________________________

# MODULE 4 - Shape Map

# ____________________________________________________________________________

if active(“Shape Map”):
st.markdown(”## Sectional Shape Map”)
if not {“Horse”, “Accel”, GR_COL, “tsSPI”, “PI”}.issubset(metrics.columns):
st.warning(“Shape Map: required columns missing.”)
else:
fig = shape_map(metrics, use_cg=USE_CG)
st.pyplot(fig)
png = fig_to_png(fig)
st.download_button(“Download PNG”, png,
file_name=“shape_map.png”, mime=“image/png”)

# ____________________________________________________________________________

# MODULE 5 - Pace Curve

# ____________________________________________________________________________

if active(“Pace Curve”):
st.markdown(”## Pace Curve”)
fig = pace_curve(work, metrics, float(race_distance_input))
st.pyplot(fig)
if st.toggle(“Save Pace Curve as PNG”, value=False):
png = fig_to_png(fig)
st.download_button(“Download PNG”, png,
file_name=“pace_curve.png”, mime=“image/png”)

# ____________________________________________________________________________

# MODULE 6 - Winning DNA

# ____________________________________________________________________________

if active(“Winning DNA”):
st.markdown(”## Winning DNA Matrix”)
WD = winning_dna(metrics, float(race_distance_input))
W_dna = WD.attrs.get(“DNA_WEIGHTS”, {})

```
def _tier_badge(sc):
    if not np.isfinite(sc): return ("", "")
    if sc >= 8.0: return ("[HOT] Prime", "A")
    if sc >= 7.0: return ("[GRN] Live", "B")
    if sc >= 6.0: return ("[WHT] Comp", "C")
    return ("[WHT] Setup", "D")

cards = WD.sort_values("WinningDNA", ascending=False).head(6)
if len(cards) > 0:
    st.markdown("**Top profiles**")
    cols_ = st.columns(3)
    for i, (_, r) in enumerate(cards.iterrows()):
        with cols_[i % 3]:
            sc = float(r["WinningDNA"])
            badge, grade = _tier_badge(sc)
            st.markdown(
                f"""<div style="border:1px solid rgba(128,128,128,0.2);border-radius:10px;
                    padding:10px;margin-bottom:8px;">
                  <b style="font-size:1.02rem">{r['Horse']}</b><br/>
                  {badge} . Grade <b>{grade}</b> . DNA <b>{sc:.2f}</b>/10<br/>
                  <span style="font-size:.88rem;opacity:.85">{r['DNA_TopTraits']}</span><br/>
                  <span style="font-size:.82rem;opacity:.75">
                    EZ {float(r['EZ01']):.2f} . MC {float(r['MC01']):.2f} .
                    LP {float(r['LP01']):.2f} . LL {float(r['LL01']):.2f} .
                    SOS {float(r['SOS01']):.2f}
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )

show = ["Horse", "WinningDNA", "EZ01", "MC01", "LP01", "LL01", "SOS01", "DNA_TopTraits"]
st.dataframe(WD.sort_values("WinningDNA", ascending=False)[show],
             use_container_width=True)
if W_dna:
    w_str = ", ".join(f"{k} {W_dna[k]:.2f}" for k in ["EZ","MC","LP","LL","SOS"])
    st.caption(f"Weights: {w_str}")
```

# ____________________________________________________________________________

# MODULE 7 - Hidden Horses

# ____________________________________________________________________________

if active(“Hidden Horses”):
st.markdown(”## Hidden Horses v2”)
hh = hidden_horses(metrics, float(race_distance_input))
cols_hh = [“Horse”, “Finish_Pos”, “PI”, “GCI”, “tsSPI”, “Accel”, GR_COL,
“SOS”, “ASI2”, “TFS”, “UEI”, “HiddenScore”, “Tier”, “Note”]
for c in cols_hh:
if c not in hh.columns: hh[c] = np.nan
view = hh[cols_hh].copy()
for c in [“PI”,“GCI”,“ASI2”,“SOS”,“TFS”,“UEI”,“Accel”,GR_COL,“tsSPI”,“HiddenScore”]:
if c in view.columns:
s_ = pd.Series(np.ravel(view[c].values), index=view.index)
view[c] = pd.to_numeric(s_, errors=“coerce”).round(2)
st.dataframe(view, use_container_width=True)
st.caption(“Sorted: Tier -> HiddenScore -> PI. [HOT] Top Hidden / [YLW] Notable Hidden.”)

# ____________________________________________________________________________

# MODULE 8 - PWX + EFI

# ____________________________________________________________________________

if active(“PWX + EFI”):
st.markdown(”## PWX + EFI - Burst vs Efficiency”)
result = pwx_efi(metrics)
if result is None:
st.warning(“PWX/EFI: missing required columns.”)
else:
st.dataframe(result, use_container_width=True)

# ____________________________________________________________________________

# MODULE 9 - Fatigue Gradient

# ____________________________________________________________________________

if active(“Fatigue Gradient”):
st.markdown(”## Fatigue Gradient”)
ftab = fatigue_gradient(metrics, work)
st.dataframe(ftab, use_container_width=True)
else:
ftab = pd.DataFrame()   # used by Power-Freshness Map below

# ____________________________________________________________________________

# MODULE 10 - PWR400

# ____________________________________________________________________________

if active(“PWR400”):
st.markdown(”## Late Power Index - PWR400”)
metrics = compute_pwr400(metrics, float(race_distance_input),
int(split_step), weight_col=“Horse Weight”)
note = metrics.attrs.get(“PWR400_NOTE”, {})
if “error” in note:
st.warning(f”PWR400: {note[‘error’]}”)
else:
alpha_ = note.get(“alpha”)
st.caption(
f”Trip: **{note.get(‘distance_m’)}m** . “
f”alpha ~= **{alpha_:.3f}** . “
f”Window: **{note.get(‘v400_source’)}**”
)
disp_cols = [“Horse”, “Finish_Pos”, “Horse Weight”, “PWR400_v400”, “PWR400”]
disp_cols = [c for c in disp_cols if c in metrics.columns]
show_ = metrics.sort_values(“PWR400”, ascending=False)[disp_cols].copy()
st.dataframe(show_, use_container_width=True)
st.caption(“PWR400 ~= 100 = field typical . 110+ = strong late engine under load”)

# ____________________________________________________________________________

# MODULE 11 - Power-Freshness Map

# ____________________________________________________________________________

if active(“Power-Freshness Map”):
st.markdown(”## Power-Freshness Map”)
if ftab.empty:
st.info(“Run **Fatigue Gradient** module to enable this chart.”)
elif “PWR400” not in metrics.columns:
st.info(“Run **PWR400** module to enable this chart.”)
else:
fig = power_freshness_map(metrics, ftab)
if fig is None:
st.info(“Not enough overlapping data.”)
else:
st.pyplot(fig)
png = fig_to_png(fig)
st.download_button(“Download PNG”, png,
file_name=“power_freshness_map.png”, mime=“image/png”)

# ____________________________________________________________________________

# MODULE 12 - R&V (CAR)

# ____________________________________________________________________________

if active(“R&V (CAR)”):
st.markdown(”## R&V - Context-Aware Reliability”)
car_df = context_aware_reliability(metrics, work)
st.dataframe(car_df, use_container_width=True)
st.caption(“High CAR -> repeatable . Low CAR -> fragile / shape-dependent”)

# ____________________________________________________________________________

# MODULE 13 - xWin

# ____________________________________________________________________________

if active(“xWin”):
st.markdown(”## xWin - Probability to Win”)
tfs_ser = None
if “hh” in dir() and isinstance(hh, pd.DataFrame) and “TFS_plus” in hh.columns:
tfs_ser = pd.to_numeric(hh.set_index(“Horse”)[“TFS_plus”], errors=“coerce”)

```
xw_df = xwin(metrics, float(race_distance_input),
             going=GOING_TYPE, tfs_series=tfs_ser)
tau_   = xw_df.attrs.get("tau", "-")
W_xw   = xw_df.attrs.get("weights", {})

st.dataframe(
    xw_df.style.format({"xWin": "{:.1f}%"}),
    use_container_width=True,
)
with st.expander("xWin methodology"):
    w_note = ", ".join(f"{k}:{W_xw[k]:.2f}" for k in ["T","K","S"]) if W_xw else "-"
    st.caption(
        f"Softmax of within-race latent ability (Travel/Kick/Sustain). "
        f"Weights: {w_note}. "
        f"Shape de-bias via RSIxSCI. Trip friction damp. tau = {tau_:.2f}. "
        f"Interpretation: % chance to win across 100 identical replays."
    )
```

# ____________________________________________________________________________

# SAVE TO DB

# ____________________________________________________________________________

st.divider()
st.markdown(”### [DB] Save Race to Database”)

with st.form(“save_form”):
col_d, col_t, col_n = st.columns(3)
with col_d: save_date  = st.text_input(“Date (YYYY-MM-DD)”, value=””)
with col_t: save_track = st.text_input(“Track name”, value=””)
with col_n: save_rno   = st.number_input(“Race number”, min_value=1, max_value=20,
value=1, step=1)
submitted = st.form_submit_button(“Save Race”)

if submitted:
extra = {}
for _, r in metrics.iterrows():
nm = str(r.get(“Horse”, “”))
extra[nm] = {
“WinningDNA”:   float(r[“WinningDNA”])   if “WinningDNA”   in metrics.columns else None,
“PWR400”:       float(r[“PWR400”])        if “PWR400”       in metrics.columns else None,
}
if not ftab.empty and “Horse” in ftab.columns:
for _, r in ftab.iterrows():
nm = str(r.get(“Horse”,””))
extra.setdefault(nm, {})[“FatigueScore”] = float(r.get(“FatigueScore”, np.nan))
if “xWin” in dir() and isinstance(xw_df, pd.DataFrame):
for _, r in xw_df.iterrows():
nm = str(r.get(“Horse”,””))
extra.setdefault(nm, {})[“xWin”] = float(r.get(“xWin”, np.nan))

```
from utils import sha1 as _sha1
src_hash = _sha1(up.name + str(race_distance_input) + str(split_step))
ok, msg = save_race(
    db_path, metrics,
    distance_m=float(race_distance_input),
    split_step=int(split_step),
    date=save_date,
    track=save_track,
    race_no=int(save_rno),
    app_version=APP_VERSION,
    src_hash=src_hash,
    extra_cols=extra,
)
st.success(msg) if ok else st.error(msg)
```

# ____________________________________________________________________________

# HORSE HISTORY LOOKUP

# ____________________________________________________________________________

st.divider()
st.markdown(”### [?] Horse History (from DB)”)
horse_query = st.text_input(“Search horse name”, value=””,
placeholder=“e.g. Captain Marvel”)
if horse_query.strip():
hist = query_horse(db_path, horse_query.strip())
if hist.empty:
st.info(“No records found.”)
else:
cols_show = [“date”, “track”, “distance_m”, “finish_pos”, “pi”,
“gci_rs”, “xwin”, “pwr400”, “fatigue_score”, “shape_tag”, “going”]
cols_show = [c for c in cols_show if c in hist.columns]
st.dataframe(hist[cols_show], use_container_width=True)
st.caption(f”{len(hist)} run(s) found for ‘{horse_query}’”)

# ____________________________________________________________________________

# DEBUG PANEL

# ____________________________________________________________________________

if DEBUG:
with st.expander(”[dbg] Debug - attrs”):
import json
from utils import sanitize_jsonable
st.json(sanitize_jsonable(dict(metrics.attrs)))
with st.expander(”[dbg] Debug - metrics columns”):
st.write(list(metrics.columns))
