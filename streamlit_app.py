import io
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Edge — Analyst v4.0 (Canon Metrics)", layout="wide")

# ------------------ Helpers ------------------
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def mad_std(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if x.size == 0: return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

# ------------------ Phase window utilities ------------------
def sum_times(row, cols):
    vals = [as_num(row.get(c)) for c in cols]
    vals = [v for v in vals if pd.notna(v) and v > 0]
    return np.sum(vals) if len(vals) else np.nan

def stage_block_cols(D, start_m, end_m_inclusive):
    # Inclusive descending 100 m steps: e.g., D-300..600 includes 600
    if start_m < end_m_inclusive:
        return []
    want = list(range(int(start_m), int(end_m_inclusive) - 1, -100))
    return [f"{m}_Time" for m in want]

def stage_speed(row, cols, meters_per_split=100.0):
    if not cols: return np.nan
    tsum = sum_times(row, cols)
    if pd.isna(tsum) or tsum <= 0: return np.nan
    valid = sum(1 for c in cols if pd.notna(row.get(c)) and as_num(row.get(c)) > 0)
    dist = meters_per_split * valid
    if dist <= 0: return np.nan
    return dist / tsum

def grind_speed(row):
    t100 = as_num(row.get("100_Time"))
    tfin = as_num(row.get("Finish_Time"))
    parts, dist = [], 0.0
    if pd.notna(t100) and t100 > 0: parts.append(float(t100)); dist += 100.0
    if pd.notna(tfin) and tfin > 0: parts.append(float(tfin)); dist += 100.0
    if not parts or dist <= 0: return np.nan
    return dist / sum(parts)

# ------------------ Index stabilizers (exact canon) ------------------
def shrink_center(idx_series: pd.Series) -> Tuple[float, int]:
    x = idx_series.dropna().values
    N_eff = len(x)
    if N_eff == 0:
        return 100.0, 0
    med_race = float(np.median(x))
    alpha = N_eff / (N_eff + 6.0)
    return alpha * med_race + (1 - alpha) * 100.0, N_eff

def dispersion_equalizer(delta_series: pd.Series, N_eff: int, N_ref=10, beta=0.20, cap=1.20):
    gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
    return delta_series * min(gamma, cap)

def variance_floor(idx_series: pd.Series, floor=1.5, cap=1.25):
    deltas = idx_series - 100.0
    sigma = mad_std(deltas)
    if not np.isfinite(sigma) or sigma <= 0:
        return idx_series
    if sigma < floor:
        factor = min(cap, floor / sigma)
        return 100.0 + deltas * factor
    return idx_series

def speed_to_index(spd_series: pd.Series) -> pd.Series:
    med = spd_series.median(skipna=True)
    idx_raw = 100.0 * (spd_series / med)
    center, n_eff = shrink_center(idx_raw)
    idx = 100.0 * (spd_series / (center / 100.0 * med))
    idx = 100.0 + dispersion_equalizer(idx - 100.0, n_eff)
    idx = variance_floor(idx)
    return idx

# ------------------ PI v3.1 (distance + context) ------------------
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

def pi_weights_distance_and_context(distance_m: float, acc_median: float | None, grd_median: float | None) -> dict:
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

# ------------------ Core metric build (exact canon) ------------------
def build_metrics(df_in: pd.DataFrame, D_actual_m: float):
    w = df_in.copy()

    # RaceTime_s from splits (if available)
    # Build expected split list: D-100 .. 100 then Finish
    want = [f"{m}_Time" for m in range(int(D_actual_m) - 100, 99, -100)]
    if "Finish_Time" in w.columns:
        want = want + ["Finish_Time"]
    present = [c for c in want if c in w.columns]
    if present:
        w["RaceTime_s"] = w[present].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0, np.nan).sum(axis=1)
    elif "Race Time" in w.columns:
        w["RaceTime_s"] = as_num(w.get("Race Time"))
    # Fallback if neither present: leave missing

    # Stage columns
    D = float(D_actual_m)
    f200_cols  = [c for c in [f"{int(D-100)}_Time", f"{int(D-200)}_Time"] if c in w.columns]
    tssp_cols  = [c for c in stage_block_cols(D, int(D-300), 600) if c in w.columns]
    accel_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]

    # Stage speeds
    w["_F200_spd"] = w.apply(lambda r: (200.0 / sum_times(r, f200_cols)) if len(f200_cols)>=1 and pd.notna(sum_times(r, f200_cols)) and sum_times(r, f200_cols)>0 else np.nan, axis=1)
    w["_MID_spd"]  = w.apply(lambda r: stage_speed(r, tssp_cols, meters_per_split=100.0), axis=1)
    w["_ACC_spd"]  = w.apply(lambda r: stage_speed(r, accel_cols, meters_per_split=100.0), axis=1)
    w["_GR_spd"]   = w.apply(grind_speed, axis=1)

    # Indices (100-based with stabilizers)
    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F200_spd"], errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"],  errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"],  errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],   errors="coerce"))

    # PI v3.1 points and standardized 0–10
    acc_med = w["Accel"].median(skipna=True)
    grd_med = w["Grind"].median(skipna=True)
    W = pi_weights_distance_and_context(float(D), acc_med, grd_med)

    def pi_pts_row(r):
        parts = []; weights = []
        for k in ["F200_idx","tsSPI","Accel","Grind"]:
            v = pd.to_numeric(r.get(k), errors="coerce")
            if pd.notna(v):
                parts.append(float(v)); weights.append(float(W[k]))
        if not weights: return np.nan
        return sum(p*w for p, w in zip(parts, weights)) / sum(weights)

    w["PI_pts"] = w.apply(pi_pts_row, axis=1)
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75:
        sigma = 0.75
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(3)

    # GCI (0–10) — original mapping (98..104)
    Wg = W
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
        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))
        acc, grd = r.get("Accel"), r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)
        score01 = (wT * T) + (wPACE * LQ) + (wSS * SS) + (wEFF * EFF)
        gci_vals.append(round(10.0 * score01, 3))
    w["GCI"] = gci_vals

    return w, {"f200_cols":f200_cols, "tssp_cols":tssp_cols, "accel_cols":accel_cols}

# ------------------ Hidden Abilities (IA/LP/HAI) ------------------
def compute_hidden_abilities(indices: pd.DataFrame) -> pd.DataFrame:
    IA = 0.20*indices["F200_idx"] + 0.30*indices["tsSPI"] + 0.25*indices["Accel"] + 0.25*indices["Grind"]
    late = 0.45*indices["Accel"] + 0.55*indices["Grind"]
    mid  = 0.40*indices["tsSPI"] + 0.20*indices["F200_idx"]
    gap  = late - mid
    LP_base = 0.35*indices["tsSPI"] + 0.65*late
    LP = LP_base + 0.15*gap
    HAI = 0.70*IA + 0.30*LP
    return pd.DataFrame({"IA":IA, "LP":LP, "HAI":HAI})

# ------------------ Hidden Horses v2 (mirror canon) ------------------
def winsorize(s, p_lo=0.10, p_hi=0.90):
    lo = s.quantile(p_lo); hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

def compute_hidden_horses(metrics: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    hh = metrics.copy()

    # SOS
    if {"tsSPI","Accel","Grind"}.issubset(hh.columns) and len(hh) > 0:
        ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
        ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
        gr_w = winsorize(pd.to_numeric(hh["Grind"], errors="coerce"))
        def rz(s):
            mu = float(s.median(skipna=True))
            sd = float(s.std(ddof=0))
            if not math.isfinite(sd) or sd <= 1e-9: return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - mu) / sd
        z_ts = rz(ts_w).clip(-2.5, 3.5)
        z_ac = rz(ac_w).clip(-2.5, 3.5)
        z_gr = rz(gr_w).clip(-2.5, 3.5)
        hh["SOS_raw"] = 0.45*z_ts + 0.35*z_ac + 0.20*z_gr
        q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
        denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and (q95 > q5)) else 1.0
        hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(lower=0.0, upper=2.0)
    else:
        hh["SOS"] = 0.0

    # ASI²
    acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
    grd_med = pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
    bias = (acc_med - 100.0) - (grd_med - 100.0)
    B = min(1.0, abs(bias) / 4.0)
    S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get("Grind"), errors="coerce")
    if bias >= 0:
        hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
    else:
        hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)

    # TFS
    def tfs_row(row):
        last3_cols = [c for c in ["300_Time","200_Time","100_Time"] if c in row.index]
        spds = []
        for c in last3_cols:
            t = pd.to_numeric(row.get(c), errors="coerce")
            spds.append(100.0 / t if pd.notna(t) and t > 0 else np.nan)
        spds = [s for s in spds if pd.notna(s)]
        if len(spds) < 2: return np.nan
        sigma = float(np.std(spds, ddof=0))
        mid = float(row.get("_MID_spd", np.nan))
        if not np.isfinite(mid) or mid <= 0: return np.nan
        return 100.0 * (sigma / mid)
    hh["TFS"] = hh.apply(tfs_row, axis=1)

    # TFS_plus gated by distance
    D_rounded = int(np.ceil(float(distance_m) / 200.0) * 200)
    if D_rounded <= 1200: gate = 4.0
    elif D_rounded < 1800: gate = 3.5
    else: gate = 3.0
    def tfs_plus(x):
        if pd.isna(x) or x < gate: return 0.0
        return min(0.6, (x - gate) / 3.0)
    hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

    # UEI
    def uei_row(row):
        ts = pd.to_numeric(row.get("tsSPI"), errors="coerce")
        ac = pd.to_numeric(row.get("Accel"), errors="coerce")
        gr = pd.to_numeric(row.get("Grind"), errors="coerce")
        if pd.isna(ts) or pd.isna(ac) or pd.isna(gr): return 0.0
        val = 0.0
        if ts >= 102 and ac <= 98 and gr <= 98:
            gap = min((ts - 102) / 3.0, 1.0)
            val = max(val, 0.3 + 0.3 * gap)
        if ts >= 102 and gr >= 102 and ac <= 100:
            gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0)
            val = max(val, 0.3 + 0.3 * gap)
        return round(val, 3)
    hh["UEI"] = hh.apply(uei_row, axis=1)

    # HiddenScore (0..3) with median/MAD scaling (exact)
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

    return hh[["SOS","ASI2","TFS","TFS_plus","UEI","HiddenScore","Tier"]]

# ------------------ Visuals ------------------
def plot_shape_map(df_show: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    x = df_show["Accel"] - 100.0
    y = df_show["Grind"] - 100.0
    sizes = (df_show["HAI"] - df_show["HAI"].min()) / (df_show["HAI"].max() - df_show["HAI"].min() + 1e-9) * 400 + 50
    sc = ax.scatter(x, y, s=sizes, c=df_show["tsSPI"], alpha=0.9)
    for _, r in df_show.iterrows():
        ax.text(r["Accel"]-100.0, r["Grind"]-100.0, str(r["Horse"]), fontsize=8, ha="center", va="center")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Acceleration (600–200) vs field (Δ index)")
    ax.set_ylabel("Grind (200–Finish) vs field (Δ index)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("tsSPI deviation (index)")
    ax.set_title("Sectional Shape Map")
    st.pyplot(fig)

def plot_pace_curve(df_times: pd.DataFrame, distance_m: int):
    split_cols = [f"{x}_Time" for x in range(distance_m - 100, 0, -100)] + ["Finish_Time"]
    have_cols = [c for c in split_cols if c in df_times.columns]
    if not have_cols:
        st.info("No per-100m splits available to draw pace curve.")
        return

    speeds = df_times[have_cols].apply(lambda col: 100.0 / as_num(col), axis=0)
    mean_speed = speeds.mean(axis=0)

    N = min(8, len(df_times))
    sel = df_times.sort_values("Finish_Pos").head(N).index if "Finish_Pos" in df_times.columns else df_times.head(N).index

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_ticks = list(range(len(have_cols)))

    # Field mean
    ax.plot(x_ticks, mean_speed.values, color="black", linewidth=2.5, label="Field Mean")

    # Selected horses
    for i in sel:
        label = str(df_times.loc[i].get("Horse", f"H{i+1}"))
        ax.plot(x_ticks, speeds.loc[i].values, linewidth=1.3, label=label)

    ax.set_xticks(x_ticks)
    xlabs = [c.replace("_Time","") if c!="Finish_Time" else "FIN" for c in have_cols]
    ax.set_xticklabels(xlabs, rotation=0, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("Segment (m from finish)", fontsize=10)
    ax.set_ylabel("Speed (m/s)", fontsize=10)
    ax.set_title(f"Pace Curve — 100 m Segment Speeds ({distance_m} m Race)", fontsize=11, pad=8)

    # Legend beneath chart (neat)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    st.pyplot(fig)

# ------------------ UI ------------------
st.title("Race Edge — Analyst v4.0 (Canon Metrics)")
st.caption("F200 · tsSPI · Accel(600–200) · Grind(200–Finish) — PI v3.1 & GCI as per original app.")

with st.sidebar:
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=3600, value=1600, step=100)
    show_tables = st.checkbox("Show metrics tables", value=True)
    show_narratives = st.checkbox("Show horse narratives", value=True)
    show_visuals = st.checkbox("Show visuals", value=True)

uploaded = st.file_uploader("Upload race CSV (per-100m splits: 1600_Time … 100_Time, Finish_Time)", type=["csv"])

if not uploaded:
    st.stop()

raw = pd.read_csv(uploaded)
df = raw.copy()
# Normalize minimal columns
name_col = None
for cand in ["Horse","Horse_Name","Name","Runner"]:
    if cand in df.columns:
        name_col = cand; break
if not name_col:
    df["Horse"] = [f"H{i+1}" for i in range(len(df))]
    name_col = "Horse"
if "Finish_Pos" not in df.columns:
    df["Finish_Pos"] = np.arange(1, len(df)+1)

# Build metrics
metrics, phase_cols = build_metrics(df, float(distance_m))

# Abilities from indices
indices = metrics[["F200_idx","tsSPI","Accel","Grind"]]
abilities = compute_hidden_abilities(indices)
metrics = pd.concat([metrics, abilities], axis=1)

# Hidden Horses
hh = compute_hidden_horses(metrics.join(indices), float(distance_m))
metrics = pd.concat([metrics, hh], axis=1)

# ------------------ Output ------------------
st.subheader("Race Synopsis")
st.write(f"Distance: **{int(distance_m)} m** · Field: **{len(df)}**")
st.write(f"Phase columns: F200={phase_cols['f200_cols']}, tsSPI={phase_cols['tssp_cols']}, Accel={phase_cols['accel_cols']}")

if show_visuals:
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("##### Pace Curve")
        plot_pace_curve(df, distance_m)
    with c2:
        st.markdown("##### Sectional Shape Map")
        show_cols = [name_col,"tsSPI","Accel","Grind","HAI"]
        df_show = metrics[show_cols].rename(columns={name_col:"Horse"}).copy()
        plot_shape_map(df_show)

if show_tables:
    st.subheader("Metrics Table (Canon)")
    cols = [name_col, "Finish_Pos"]
    if "RaceTime_s" in metrics.columns:
        cols += ["RaceTime_s"]
    cols += ["F200_idx","tsSPI","Accel","Grind","PI","GCI","Tier","HiddenScore"]
    st.dataframe(metrics[cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]), use_container_width=True)

    st.subheader("Ability Matrix")
    A = metrics[[name_col,"IA","LP","HAI","Tier"]].rename(columns={name_col:"Horse"})
    st.dataframe(A.sort_values("HAI", ascending=False), use_container_width=True)

if show_narratives:
    st.subheader("Horse Narratives")
    # z-scores vs field medians
    sig = {k: mad_std(metrics[k].values) or 1.0 for k in ["F200_idx","tsSPI","Accel","Grind"]}
    med = {k: float(np.nanmedian(metrics[k])) for k in ["F200_idx","tsSPI","Accel","Grind"]}
    for _, r in metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).iterrows():
        zF = (r["F200_idx"] - med["F200_idx"]) / (sig["F200_idx"] if sig["F200_idx"]>1e-9 else 1.0)
        zM = (r["tsSPI"]     - med["tsSPI"])     / (sig["tsSPI"]     if sig["tsSPI"]>1e-9 else 1.0)
        zA = (r["Accel"]     - med["Accel"])     / (sig["Accel"]     if sig["Accel"]>1e-9 else 1.0)
        zG = (r["Grind"]     - med["Grind"])     / (sig["Grind"]     if sig["Grind"]>1e-9 else 1.0)
        st.markdown(f"**{r[name_col]} ({int(r['Finish_Pos'])})** — F200 {zF:+.1f}σ • tsSPI {zM:+.1f}σ • Accel {zA:+.1f}σ • Grind {zG:+.1f}σ.")
        st.write(f"IA {r['IA']:.0f} / LP {r['LP']:.0f} → HAI {r['HAI']:.0f} ({r['Tier'] or '—'}).")
        st.write("---")
