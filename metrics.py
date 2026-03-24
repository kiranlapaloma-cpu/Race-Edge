“””
metrics.py — core metric engine for Race Edge.

All PI / GCI / RSI / CG algorithms are UNCHANGED from the original.
This file only reorganises them into importable functions and removes
duplicated helpers (which now live in utils.py).
“””
import math
import re
import numpy as np
import pandas as pd

from utils import (
as_num, clamp, mad_std, winsorize, lerp,
mass_series, parse_mass_to_kg, pick_mass_column,
pct_at_or_above,
)
from data_io import collect_markers

# ──────────────────────────────────────────────

# Stage / speed helpers  (unchanged logic)

# ──────────────────────────────────────────────

def _sum_times(row, cols):
vals = [pd.to_numeric(row.get(c), errors=“coerce”) for c in cols]
vals = [float(v) for v in vals if pd.notna(v) and v > 0]
return np.sum(vals) if vals else np.nan

def _make_range_cols(D, start_incl, end_incl, step):
if start_incl < end_incl:
return []
want = list(range(int(start_incl), int(end_incl) - 1, -int(step)))
return [f”{m}_Time” for m in want]

def _stage_speed(row, cols, meters_per_split):
if not cols:
return np.nan
tsum = _sum_times(row, cols)
if not (pd.notna(tsum) and tsum > 0):
return np.nan
valid = [c for c in cols
if pd.notna(row.get(c))
and pd.to_numeric(row.get(c), errors=“coerce”) > 0]
dist = meters_per_split * len(valid)
return np.nan if dist <= 0 else dist / tsum

def _grind_speed(row, step):
if step == 100:
t100 = pd.to_numeric(row.get(“100_Time”), errors=“coerce”)
tfin = pd.to_numeric(row.get(“Finish_Time”), errors=“coerce”)
parts, dist = [], 0.0
if pd.notna(t100) and t100 > 0:
parts.append(float(t100)); dist += 100.0
if pd.notna(tfin) and tfin > 0:
parts.append(float(tfin)); dist += 100.0
return np.nan if not parts or dist <= 0 else dist / sum(parts)
else:
tfin = pd.to_numeric(row.get(“Finish_Time”), errors=“coerce”)
return np.nan if (pd.isna(tfin) or tfin <= 0) else 200.0 / float(tfin)

# ──────────────────────────────────────────────

# Adaptive window helpers  (unchanged)

# ──────────────────────────────────────────────

def _adaptive_f_cols_and_dist(D, step, markers, frame_cols):
if not markers:
return [], 0.0
D = float(D); step = int(step)
if step == 100:
if int(D) % 100 == 50:
wanted = [int(D - 50), int(D - 150)]
cols = [f”{m}_Time” for m in wanted if f”{m}_Time” in frame_cols]
dist = 150.0 if len(cols) == 2 else 100.0 * len(cols)
else:
wanted = [int(D - 100), int(D - 200)]
cols = [f”{m}_Time” for m in wanted if f”{m}_Time” in frame_cols]
dist = 100.0 * len(cols)
return cols, float(dist)
m1 = int(markers[0])
first_span = max(1.0, D - m1)
c = f”{m1}_Time”
cols = [c] if c in frame_cols else []
if first_span <= 120:    dist = 100.0
elif first_span <= 180:  dist = 160.0
elif first_span <= 220:  dist = 200.0
else:                    dist = 250.0
return cols, float(dist)

def _adaptive_tssp_start(D, step, markers):
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

# ──────────────────────────────────────────────

# Speed → Index  (unchanged)

# ──────────────────────────────────────────────

def _shrink_center(idx_series):
x = pd.to_numeric(idx_series, errors=“coerce”).dropna().values
if x.size == 0:
return 100.0, 0
med_race = float(np.median(x))
alpha = x.size / (x.size + 6.0)
return alpha * med_race + (1 - alpha) * 100.0, x.size

def _dispersion_equalizer(delta_series, n_eff, N_ref=10, beta=0.20, cap=1.20):
gamma = 1.0 + beta * max(0, N_ref - n_eff) / N_ref
return delta_series * min(gamma, cap)

def _variance_floor(idx_series, floor=1.5, cap=1.25):
deltas = idx_series - 100.0
sigma = mad_std(deltas)
if not np.isfinite(sigma) or sigma <= 0:
return idx_series
if sigma < floor:
factor = min(cap, floor / sigma)
return 100.0 + deltas * factor
return idx_series

def speed_to_idx(spd_series):
s = pd.to_numeric(spd_series, errors=“coerce”)
med = s.median(skipna=True)
idx_raw = 100.0 * (s / med)
center, n_eff = _shrink_center(idx_raw)
idx = 100.0 * (s / (center / 100.0 * med))
idx = 100.0 + _dispersion_equalizer(idx - 100.0, n_eff)
return _variance_floor(idx)

# ──────────────────────────────────────────────

# PI weights  (unchanged)

# ──────────────────────────────────────────────

def pi_weights_distance_and_context(
distance_m: float,
acc_med,
grd_med,
going: str = “Good”,
field_n: int | None = None,
return_meta: bool = False,
):
dm = float(distance_m or 1200)

```
if dm <= 1000:   F200, ts, ACC, GR = 0.10, 0.20, 0.45, 0.25
elif dm <= 1200: F200, ts, ACC, GR = 0.10, 0.25, 0.40, 0.25
elif dm <= 1400: F200, ts, ACC, GR = 0.10, 0.30, 0.35, 0.25
elif dm <= 2200: F200, ts, ACC, GR = 0.08, 0.35, 0.35, 0.22
else:            F200, ts, ACC, GR = 0.05, 0.40, 0.35, 0.20

base = {"F200_idx": F200, "tsSPI": ts, "Accel": ACC, "Grind": GR}

if acc_med is not None and grd_med is not None \
        and math.isfinite(acc_med) and math.isfinite(grd_med):
    bias = acc_med - grd_med
    scale = math.tanh(abs(bias) / 6.0)
    max_shift = 0.02 * scale
    F200, ts, ACC, GR = base["F200_idx"], base["tsSPI"], base["Accel"], base["Grind"]
    if bias > 0:
        delta = min(max_shift, ACC - 0.26); ACC -= delta; GR += delta
    elif bias < 0:
        delta = min(max_shift, GR - 0.18); GR -= delta; ACC += delta
    GR = min(GR, 0.40)
    ts = max(0.0, 1.0 - F200 - ACC - GR)
    base = {"F200_idx": F200, "tsSPI": ts, "Accel": ACC, "Grind": GR}

n = max(1, int(field_n or 12))
field_scale = min(1.0, n / 12.0)

if going == "Firm":
    amp = 0.06 * field_scale
    mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0 + amp, "Grind": 1.0}
elif going == "Soft":
    amp = 0.06 * field_scale
    mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0 + amp}
elif going == "Heavy":
    amp = 0.10 * field_scale
    mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0 + amp}
else:
    mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0}

weighted = {k: base[k] * mult[k] for k in base}
s = sum(weighted.values()) or 1.0
out = {k: v / s for k, v in weighted.items()}

if not return_meta:
    return out
return out, {"going": going, "field_n": n, "multipliers": mult,
             "base": base.copy(), "final": out.copy()}
```

# ──────────────────────────────────────────────

# Distance-aware per-kg penalty

# ──────────────────────────────────────────────

def perkg_pts(dm: float) -> float:
knots = [(1000, 0.10), (1200, 0.12), (1400, 0.14),
(1600, 0.16), (2000, 0.20), (2400, 0.24)]
dm = float(dm)
if dm <= knots[0][0]: return knots[0][1]
if dm >= knots[-1][0]: return knots[-1][1]
for (a, va), (b, vb) in zip(knots, knots[1:]):
if a <= dm <= b:
return va + (vb - va) * (dm - a) / (b - a)
return 0.16

# ──────────────────────────────────────────────

# Main builder  (algorithms 100 % unchanged)

# ──────────────────────────────────────────────

def build_metrics_and_shape(
df_in: pd.DataFrame,
D_actual_m: float,
step: int,
use_cg: bool,
dampen_cg: bool,
use_race_shape: bool,
going: str = “Good”,
debug: bool = False,
) -> tuple[pd.DataFrame, list[int]]:
“””
Compute all sectional indices, PI, GCI, RSI/SCI and race-shape attrs.
Returns (metrics_df, seg_markers).

```
NOTE: going is now passed as a parameter rather than read from globals.
"""
w = df_in.copy()
seg_markers = collect_markers(w)
D = float(D_actual_m)
step = int(step)

# ── per-segment raw speeds ──
for m in seg_markers:
    w[f"spd_{m}"] = (step * 1.0) / pd.to_numeric(
        w.get(f"{m}_Time"), errors="coerce"
    )
if "Finish_Time" in w.columns:
    finish_len = 100.0 if step == 100 else 200.0
    w["spd_Finish"] = finish_len / pd.to_numeric(
        w.get("Finish_Time"), errors="coerce"
    )

# ── RaceTime ──
if seg_markers:
    wanted = [f"{m}_Time" for m in range(int(D) - step, step - 1, -step)
              if f"{m}_Time" in w.columns]
    if "Finish_Time" in w.columns:
        wanted += ["Finish_Time"]
    w["RaceTime_s"] = (
        w[wanted].apply(pd.to_numeric, errors="coerce")
        .clip(lower=0).replace(0, np.nan).sum(axis=1)
    )
else:
    w["RaceTime_s"] = pd.to_numeric(w.get("Race Time"), errors="coerce")

# ── composite speeds ──
f_cols, f_dist = _adaptive_f_cols_and_dist(D, step, seg_markers, w.columns)
w["_F_spd"] = w.apply(
    lambda r: (f_dist / _sum_times(r, f_cols))
    if (f_cols and pd.notna(_sum_times(r, f_cols)) and _sum_times(r, f_cols) > 0)
    else np.nan,
    axis=1,
)

tssp_start = _adaptive_tssp_start(D, step, seg_markers)
mid_cols = [c for c in _make_range_cols(D, tssp_start, 600, step) if c in w.columns]
w["_MID_spd"] = w.apply(lambda r: _stage_speed(r, mid_cols, float(step)), axis=1)

if step == 100:
    acc_cols = [c for c in [f"{m}_Time" for m in [500, 400, 300, 200]] if c in w.columns]
else:
    acc_cols = [c for c in [f"{m}_Time" for m in [600, 400]] if c in w.columns]
w["_ACC_spd"] = w.apply(lambda r: _stage_speed(r, acc_cols, float(step)), axis=1)
w["_GR_spd"]  = w.apply(lambda r: _grind_speed(r, step), axis=1)

# ── speed → indices ──
w["F200_idx"] = speed_to_idx(w["_F_spd"])
w["tsSPI"]    = speed_to_idx(w["_MID_spd"])
w["Accel"]    = speed_to_idx(w["_ACC_spd"])
w["Grind"]    = speed_to_idx(w["_GR_spd"])

# ── Corrected Grind ──
ACC_field = pd.to_numeric(w["_ACC_spd"], errors="coerce").mean(skipna=True)
GR_field  = pd.to_numeric(w["_GR_spd"],  errors="coerce").mean(skipna=True)
FSR = float(GR_field / ACC_field) \
    if (math.isfinite(ACC_field) and ACC_field > 0 and math.isfinite(GR_field)) \
    else 1.0
CollapseSeverity = float(min(10.0, max(0.0, (0.95 - FSR) * 100.0)))

def _delta_g_row(r):
    mid = float(r.get("_MID_spd", np.nan))
    grd = float(r.get("_GR_spd",  np.nan))
    if not (math.isfinite(mid) and math.isfinite(grd) and mid > 0):
        return np.nan
    return 100.0 * (grd / mid)

w["DeltaG"] = w.apply(_delta_g_row, axis=1)
w["FinisherFactor"] = w["DeltaG"].map(
    lambda dg: 0.0 if not math.isfinite(dg)
    else float(clamp((dg - 98.0) / 4.0, 0.0, 1.0))
)
w["GrindAdjPts"] = (CollapseSeverity * (1.0 - w["FinisherFactor"])).round(2)
w["Grind_CG"] = (w["Grind"] - w["GrindAdjPts"]).clip(lower=90.0, upper=110.0)

def _fade_cap(g, dg):
    if not (math.isfinite(g) and math.isfinite(dg)):
        return g
    return 100.0 + 0.5 * (g - 100.0) if (dg < 97.0 and g > 100.0) else g

w["Grind_CG"] = [
    _fade_cap(g, dg) for g, dg in zip(w["Grind_CG"], w["DeltaG"])
]

# ── PI v3.2 ──
GR_COL = "Grind_CG" if use_cg else "Grind"
acc_med = pd.to_numeric(w["Accel"], errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(w[GR_COL],  errors="coerce").median(skipna=True)

PI_W, PI_META = pi_weights_distance_and_context(
    D, acc_med, grd_med,
    going=going, field_n=len(w), return_meta=True,
)

if use_cg and dampen_cg and CollapseSeverity >= 3.0:
    d = min(0.02 + 0.01 * (CollapseSeverity - 3.0), 0.08)
    shift = min(d, PI_W["Grind"])
    PI_W["Grind"] -= shift
    PI_W["Accel"] += shift * 0.5
    PI_W["tsSPI"] += shift * 0.5

# ── mass-aware PI ──
mass_kg, mass_src = mass_series(w)
w.attrs["MASS_SRC"] = mass_src
mass_ref = float(np.nanmedian(mass_kg)) if np.isfinite(np.nanmedian(mass_kg)) else np.nan
mass_delta = (
    (mass_kg - mass_ref)
    if np.isfinite(mass_ref)
    else pd.Series(0.0, index=w.index, dtype=float)
)
PER_KG_PTS = 0.14

def _pi_pts_row(r):
    acc = r.get("Accel"); mid = r.get("tsSPI")
    f   = r.get("F200_idx"); gr  = r.get(GR_COL)
    parts = []
    if pd.notna(f):   parts.append(PI_W["F200_idx"] * (f   - 100.0))
    if pd.notna(mid): parts.append(PI_W["tsSPI"]    * (mid - 100.0))
    if pd.notna(acc): parts.append(PI_W["Accel"]    * (acc - 100.0))
    if pd.notna(gr):  parts.append(PI_W["Grind"]    * (gr  - 100.0))
    if not parts:
        return np.nan
    base = sum(parts) / (sum(PI_W.values()) or 1.0)
    idx = r.name
    md = float(mass_delta.loc[idx]) if idx in mass_delta.index else 0.0
    return base - (PER_KG_PTS * md)

w["PI_pts"] = w.apply(_pi_pts_row, axis=1)
pts = pd.to_numeric(w["PI_pts"], errors="coerce")
med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
centered = pts - med
sigma = mad_std(centered)
sigma = 0.75 if (not np.isfinite(sigma) or sigma < 0.75) else sigma
w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

# ── GCI ──
winner_time = None
if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
    try:
        winner_time = float(w["RaceTime_s"].min())
    except Exception:
        pass

Wg = pi_weights_distance_and_context(
    D,
    pd.to_numeric(w["Accel"],  errors="coerce").median(skipna=True),
    pd.to_numeric(w[GR_COL],   errors="coerce").median(skipna=True),
)
wT    = 0.25
wPACE = Wg["Accel"] + Wg["Grind"]
wSS   = Wg["tsSPI"]
wEFF  = max(0.0, 1.0 - (wT + wPACE + wSS))

def _map_pct(x, lo=98.0, hi=104.0):
    x = float(x) if pd.notna(x) else np.nan
    return 0.0 if not np.isfinite(x) else clamp((x - lo) / (hi - lo), 0.0, 1.0)

gci_vals = []
for _, r in w.iterrows():
    T = 0.0
    if winner_time is not None and pd.notna(r.get("RaceTime_s")):
        d = float(r["RaceTime_s"]) - winner_time
        T = 1.0 if d <= 0.30 else (0.7 if d <= 0.60 else (0.4 if d <= 1.00 else 0.2))
    LQ  = 0.6 * _map_pct(r.get("Accel")) + 0.4 * _map_pct(r.get(GR_COL))
    SS  = _map_pct(r.get("tsSPI"))
    acc_v, grd_v = r.get("Accel"), r.get(GR_COL)
    EFF = 0.0 if (pd.isna(acc_v) or pd.isna(grd_v)) else \
        clamp(1.0 - (abs(acc_v - 100.0) + abs(grd_v - 100.0)) / 2.0 / 8.0, 0.0, 1.0)
    gci_vals.append(round(10.0 * (wT * T + wPACE * LQ + wSS * SS + wEFF * EFF), 3))
w["GCI"] = gci_vals

# ── GCI_RS ──
RSI_val = float(w.attrs.get("RSI", 0.0))
SCI_val = float(w.attrs.get("SCI", 0.0))
dLM = (pd.to_numeric(w["Accel"], errors="coerce") -
       pd.to_numeric(w["tsSPI"], errors="coerce"))
expo = np.tanh((dLM / 6.0)) * np.sign(RSI_val)
consensus = 0.60 + 0.40 * max(0.0, min(1.0, SCI_val))
expo *= consensus
with_shape = np.clip(expo, 0.0, 1.0)
against    = np.clip(-expo, 0.0, 1.0)
adj = 0.35 * against - 0.22 * with_shape
dist_gain = 1.00 + (0.06 if 1400 <= D_actual_m <= 1800 else 0.00)
adj *= dist_gain
if SCI_val < 0.40:
    adj *= 0.5
w["GCI_RS"] = (pd.to_numeric(w["GCI"], errors="coerce") + adj).clip(0.0, 10.0).round(3)

# ── EARLY / LATE ──
w["EARLY_idx"] = (0.65 * pd.to_numeric(w["F200_idx"], errors="coerce") +
                  0.35 * pd.to_numeric(w["tsSPI"],    errors="coerce"))
w["LATE_idx"]  = (0.60 * pd.to_numeric(w["Accel"],   errors="coerce") +
                  0.40 * pd.to_numeric(w[GR_COL],    errors="coerce"))

# ── attrs ──
w.attrs.update({
    "GR_COL": GR_COL,
    "STEP":   step,
    "FSR":    FSR,
    "CollapseSeverity": CollapseSeverity,
    "GOING":  going,
    "PI_GOING_META": PI_META,
    "PI_MASS_NOTE": {
        "mass_col":   mass_src,
        "ref_kg":     60.0,
        "perkg_pts":  round(PER_KG_PTS, 3),
        "distance_m": int(D),
    },
})

# ── Race Shape (RSI / SCI) ──
acc = pd.to_numeric(w["Accel"], errors="coerce")
mid = pd.to_numeric(w["tsSPI"], errors="coerce")
grd = pd.to_numeric(w[GR_COL],  errors="coerce")
dLM = acc - mid
dLG = grd - acc

def _madv(s):
    v = mad_std(pd.to_numeric(s, errors="coerce"))
    return 0.0 if not np.isfinite(v) else float(v)

sgn = np.sign(dLM.dropna().to_numpy())
if sgn.size:
    sgn_med = int(np.sign(np.median(dLM.dropna())))
    sci = float((sgn == sgn_med).mean()) if sgn_med != 0 else 0.0
else:
    sgn_med = 0; sci = 0.0

med_dLM = float(np.nanmedian(dLM))
mad_dLM = _madv(dLM)
if not np.isfinite(mad_dLM) or mad_dLM <= 0:
    mad_dLM = 1.0

if   D <= 1100: dist_gain = 0.95
elif D <= 1400: dist_gain = 1.05
elif D <= 1800: dist_gain = 1.12
elif D <= 2000: dist_gain = 1.05
else:           dist_gain = 0.98

mad_dLG = _madv(dLG)
fin_strength = 0.0 if mad_dLG == 0 else clamp(
    abs(float(np.nanmedian(dLG))) / max(mad_dLG, 1e-6), 0.0, 2.0
)
fin_bonus = 0.30 * fin_strength

rsi_signed = (med_dLM / mad_dLM) * 3.2
rsi_signed *= (0.60 + 0.40 * sci) * dist_gain
rsi = float(np.round(
    np.sign(rsi_signed) * min(10.0, abs(rsi_signed) * (1.0 + fin_bonus)), 2
))

if   abs(rsi) < 1.2: shape_tag = "EVEN"
elif rsi > 0:         shape_tag = "SLOW_EARLY"
else:                 shape_tag = "FAST_EARLY"

w["RS_Component"] = (acc - mid).round(3)

def _align_row(val, rsi_val, eps=0.25):
    if not (np.isfinite(val) and np.isfinite(rsi_val)) or abs(rsi_val) < 1.2:
        return 0
    if val > +eps and rsi_val > 0: return +1
    if val < -eps and rsi_val < 0: return +1
    if val > +eps and rsi_val < 0: return -1
    if val < -eps and rsi_val > 0: return -1
    return 0

w["RSI_Align"] = [_align_row(v, rsi) for v in w["RS_Component"]]
w["RSI_Cue"]   = [
    "🔵 ➜ with shape" if a > 0 else ("🔴 ⇦ against shape" if a < 0 else "⚪ neutral")
    for a in w["RSI_Align"]
]

fin_flav  = "Balanced Finish"
med_dLG_v = float(np.nanmedian(dLG))
gLG_gate  = max(1.40, 0.50 * _madv(dLG))
if   med_dLG_v >= +gLG_gate: fin_flav = "Attritional Finish"
elif med_dLG_v <= -gLG_gate: fin_flav = "Sprint Finish"

w.attrs.update({
    "RSI":          rsi,
    "RSI_STRENGTH": float(min(10.0, abs(rsi))),
    "SCI":          sci,
    "SHAPE_TAG":    shape_tag,
    "FINISH_FLAV":  fin_flav,
})

return w, seg_markers
```

# ──────────────────────────────────────────────

# RQS / RPS  (unchanged algorithms)

# ──────────────────────────────────────────────

def compute_rqs(df: pd.DataFrame, attrs: dict) -> float:
if df is None or len(df) == 0:
return 0.0
pi = pd.to_numeric(df.get(“PI_RS”, df.get(“PI”)), errors=“coerce”)
pi = pi[pi.notna()]
if pi.empty:
return 0.0
p90  = float(np.nanpercentile(pi.values, 90))
S1   = 10.0 * p90
S2   = 100.0 * pct_at_or_above(pi, 6.8)
center  = float(np.nanmedian(pi))
mad_raw = float(np.nanmedian(np.abs(pi - center)))
if not np.isfinite(mad_raw): mad_raw = 1.2
S3   = 100.0 * max(0.0, 1.0 - abs(mad_raw - 1.2) / 1.2)
n    = int(len(df.index))
trust = float(min(1.15, max(0.85, 0.85 + 0.30 * max(0.0, min(1.0, (n - 6) / 8.0)))))
fra_applied = int(attrs.get(“FRA_APPLIED”, 0) or 0)
sci         = float(attrs.get(“SCI”, 1.0))
penalty = 0.0
if fra_applied == 1 and sci >= 0.60:
penalty = float(min(10.0, max(0.0, 10.0 * (sci - 0.60) / 0.40)))
rqs = (0.55 * S1 + 0.25 * S2 + 0.20 * S3) * trust - penalty
return float(np.clip(round(rqs, 1), 0.0, 100.0))

def compute_rps(df: pd.DataFrame) -> float:
“”“Star-aware peak strength — uses PI_RS if present, falls back to PI.”””
if df is None or len(df) == 0:
return 0.0
pi = pd.to_numeric(df.get(“PI_RS”, df.get(“PI”)), errors=“coerce”).dropna()
n  = int(pi.size)
if n == 0:
return 0.0
p95  = float(np.nanpercentile(pi, 95)) if n >= 5 else float(np.nanmax(pi))
p90  = float(np.nanpercentile(pi, 90)) if n >= 4 else p95
pmax = float(np.nanmax(pi))
top2 = float(np.partition(pi.values, -2)[-2]) if n >= 2 else p90
gap_top = max(0.0, pmax - p90)
gap_2   = max(0.0, pmax - top2)
trust   = max(0.0, min(1.0, (n - 6) / 6.0))

```
def _sat(x, mid=1.8, span=1.2):
    return max(0.0, min(1.0, x / (mid + span)))

dom    = max(_sat(gap_top), _sat(gap_2))
w_star = min(0.95, max(0.0, 0.30 + 0.40 * dom * trust))
rps_pi = (1.0 - w_star) * p95 + w_star * pmax
return float(np.clip(round(10.0 * rps_pi, 1), 0.0, 100.0))
```

def classify_race_profile(rqs: float, rps: float) -> tuple[str, str]:
if not (math.isfinite(rqs) and math.isfinite(rps)):
return “Unknown”, “#7f8c8d”
delta = rps - rqs
if delta >= 18.0:
return “🔴 Top-Heavy”,      “#e74c3c”
elif rqs >= (rps - 10.0):
return “🟢 Deep Field”,     “#2ecc71”
else:
return “⚪ Average Profile”, “#95a5a6”
