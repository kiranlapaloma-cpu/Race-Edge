import io
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =======================
# App Config
# =======================
st.set_page_config(page_title="Race Edge — Analyst v4.0 (Pure Sectional Engine)", layout="wide")

# =======================
# Helper utilities
# =======================
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def mad_std(x: np.ndarray) -> float:
    """Robust sigma from MAD (median absolute deviation)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # normal distribution: sigma ≈ 1.4826 * MAD
    return 1.4826 * mad if mad > 0 else 0.0

def robust_center_index(series: pd.Series, field_size: int, min_spread: float = 1.5, max_spread: float = 1.25) -> pd.Series:
    """
    Convert speeds to 100-based indices using median center, robust dispersion,
    field-size shrinkage and a small dispersion equalizer for tiny fields.
    """
    x = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(x)
    # robust sigma
    sigma = mad_std(x.values)
    # variance floor: ensure minimal spread
    if sigma <= 1e-9:
        sigma = 1.0
    # convert to deltas around median in "percent" space first
    # base index: 100 at median, +/- proportional to (value - med) / med * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_delta = (x - med) / med * 100.0

    # field-size shrink toward 100
    N_eff = max(3, int(field_size))
    alpha = N_eff / (N_eff + 6.0)  # shrinks more when field small
    pct_delta = pct_delta * alpha

    # dispersion equalizer: inflate a bit for tiny fields (cap +20% effect)
    N_ref = 12.0
    gamma = 1.0 + 0.20 * (N_ref - N_eff) / N_ref
    gamma = clamp(gamma, 1.0, 1.20)
    pct_delta = pct_delta * gamma

    # enforce min spread (as absolute % from 100) and cap max spread multiplier
    span = np.nanpercentile(np.abs(pct_delta), 90)
    if np.isfinite(span) and span < min_spread:
        scale = min_spread / (span + 1e-9)
        pct_delta = pct_delta * min(scale, max_spread)

    return 100.0 + pct_delta

def compute_phase_speeds(df_times: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    """
    Given per-100m split times columns like 1500_Time, 1400_Time, ..., 100_Time, Finish_Time,
    compute speeds (m/s) for the four phases:
      - F200 = (D-100) + (D-200) times over 200 m
      - tsSPI = sum of times from (D-300) down to 600 (inclusive)  [all 100m splits]
      - Accel = 500 + 400 + 300 + 200  (i.e., 600->200; four 100m splits)
      - Grind = 100 + Finish_Time      (i.e., 200->Finish; two 100m segments)
    Also returns integrity metrics about which splits were present.
    """
    # Build list of expected time columns from distance down to 100
    expected_cols = []
    for start in range(distance_m - 100, 0, -100):
        col = f"{start}_Time"
        expected_cols.append(col)
    expected_cols.append("Finish_Time")  # last 100 -> finish

    # Ensure numeric
    for c in expected_cols:
        if c in df_times.columns:
            df_times[c] = as_num(df_times[c])

    # Helpers to pick safe sums
    def sum_cols(row, cols):
        vals = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
        if len(vals) == 0:
            return np.nan
        return np.nansum(vals)

    # Determine column sets for each phase
    d = distance_m
    c_F200 = [f"{d-100}_Time", f"{d-200}_Time"]
    # tsSPI: from (D-300) down to 600 (inclusive) stepping -100
    c_tsSPI = [f"{x}_Time" for x in range(d-300, 500, -100)]  # note: stop at >500 so 600 included
    # Accel: 500, 400, 300, 200
    c_Accel = [f"{x}_Time" for x in [500, 400, 300, 200] if x > 0 and x < d]
    # Grind: 100 + Finish_Time
    c_Grind = [f"{x}_Time" for x in [100] if x < d] + ["Finish_Time"]

    # Integrity: count available splits vs expected
    have_cols = [c for c in expected_cols if c in df_times.columns]
    completeness = len(have_cols) / len(expected_cols) * 100.0

    out = pd.DataFrame(index=df_times.index)
    # Speeds = distance / time
    out["F200_speed"] = df_times.apply(lambda r: 200.0 / sum_cols(r, c_F200) if pd.notna(sum_cols(r, c_F200)) else np.nan, axis=1)
    out["tsSPI_speed"] = df_times.apply(lambda r: (100.0 * len([c for c in c_tsSPI if c in r.index and pd.notna(r[c])])) / sum_cols(r, c_tsSPI) if pd.notna(sum_cols(r, c_tsSPI)) else np.nan, axis=1)
    out["Accel_speed"] = df_times.apply(lambda r: (100.0 * len([c for c in c_Accel if c in r.index and pd.notna(r[c])])) / sum_cols(r, c_Accel) if pd.notna(sum_cols(r, c_Accel)) else np.nan, axis=1)
    out["Grind_speed"] = df_times.apply(lambda r: (100.0 * len([c for c in c_Grind if c in r.index and pd.notna(r[c])])) / sum_cols(r, c_Grind) if pd.notna(sum_cols(r, c_Grind)) else np.nan, axis=1)

    meta = dict(
        expected_cols=expected_cols,
        have_cols=have_cols,
        completeness=completeness,
        c_F200=c_F200, c_tsSPI=c_tsSPI, c_Accel=c_Accel, c_Grind=c_Grind
    )
    return out, meta

def compute_indices(speeds_df: pd.DataFrame, field_size: int) -> pd.DataFrame:
    """Map speeds to 100-based indices with robustness rules."""
    idx = pd.DataFrame(index=speeds_df.index)
    idx["F200_idx"] = robust_center_index(speeds_df["F200_speed"], field_size)
    idx["tsSPI"] = robust_center_index(speeds_df["tsSPI_speed"], field_size)
    idx["Accel"] = robust_center_index(speeds_df["Accel_speed"], field_size)
    idx["Grind"] = robust_center_index(speeds_df["Grind_speed"], field_size)
    return idx

def pi_weights(distance_m: int) -> Tuple[float, float, float, float]:
    """
    Base PI v3.1 weights by distance, then adjusted later for context.
    The mapping mirrors the user's canon (illustrative but consistent):
      ≤1000m: F200 0.12, tsSPI 0.35, Accel 0.36, Grind 0.17
      1100m: 0.10, 0.36, 0.34, 0.20
      1200m: 0.08, 0.37, 0.30, 0.25
      >1200m: increment Grind gradually up to max 0.40 by distance
    """
    if distance_m <= 1000:
        w = [0.12, 0.35, 0.36, 0.17]
    elif distance_m <= 1100:
        w = [0.10, 0.36, 0.34, 0.20]
    elif distance_m <= 1200:
        w = [0.08, 0.37, 0.30, 0.25]
    else:
        # Progressive increase in Grind with distance
        base = [0.08, 0.35, 0.29, 0.28]  # starting point near 1300
        extra = min(0.12, max(0, (distance_m - 1300) / 900.0 * 0.12))  # up to +0.12 by 2200
        w = [base[0] - 0.02, base[1] + 0.01, base[2] - 0.01, base[3] + extra]
        # normalize to 1
    s = sum(w)
    return (w[0]/s, w[1]/s, w[2]/s, w[3]/s)

def context_tweak(weights, idx_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Small context tweak (±0.02 total shift) depending on Accel vs Grind medians.
    Shift a bit from hotter phase to the relatively underweighted one (stays bounded).
    """
    wF, wM, wA, wG = weights
    medA = np.nanmedian(idx_df["Accel"])
    medG = np.nanmedian(idx_df["Grind"])
    delta = clamp((medA - medG) / 200.0, -0.02, 0.02)  # small tweak
    # If Accel hotter than Grind, give Grind a bit more, take from Accel; else reverse.
    wA2 = clamp(wA - delta, 0.18, 0.50)
    wG2 = clamp(wG + delta, 0.18, 0.50)
    # keep F200 and tsSPI proportional (mild renorm to sum 1)
    total = wF + wM + wA2 + wG2
    return (wF/total, wM/total, wA2/total, wG2/total)

def compute_PI(indices: pd.DataFrame, distance_m: int) -> pd.Series:
    base_w = pi_weights(distance_m)
    tweaked = context_tweak(base_w, indices)
    wF, wM, wA, wG = tweaked
    return wF*indices["F200_idx"] + wM*indices["tsSPI"] + wA*indices["Accel"] + wG*indices["Grind"]

def map_98_104_to_0_1(x: pd.Series) -> pd.Series:
    """Map 98..104 -> 0..1 with clipping, used in GCI subcomponents."""
    return ((x - 98.0) / 6.0).clip(0.0, 1.0)

def compute_GCI(df: pd.DataFrame, indices: pd.DataFrame, distance_m: int) -> pd.Series:
    """
    GCI (0..10) analytical score (no betting). Uses:
      - T: time proximity to winner
      - LQ: late quality (Accel, Grind mapped)
      - SS: sustain (tsSPI mapped)
      - EFF: late efficiency (balance of Accel & Grind)
    """
    # winner time per race row assumed in df["RaceTime_s"] and min per race grouping;
    # but for single-race upload, treat min of column as winner.
    if "RaceTime_s" in df.columns:
        winner_time = df["RaceTime_s"].min(skipna=True)
        T_delta = df["RaceTime_s"] - winner_time
        # piecewise proximity
        T = pd.Series(np.where(T_delta <= 0.30, 1.0,
                               np.where(T_delta <= 0.60, 0.7,
                               np.where(T_delta <= 1.00, 0.4, 0.2))), index=df.index)
    else:
        T = pd.Series(0.6, index=df.index)

    LQ = 0.6*map_98_104_to_0_1(indices["Accel"]) + 0.4*map_98_104_to_0_1(indices["Grind"])
    SS = map_98_104_to_0_1(indices["tsSPI"])
    EFF = 1.0 - ( (np.abs(indices["Accel"]-100.0) + np.abs(indices["Grind"]-100.0)) / 2.0 ) / 8.0
    EFF = EFF.clip(0.0, 1.0)

    # weights derived from PI weights
    wF, wM, wA, wG = context_tweak(pi_weights(distance_m), indices)
    wT = 0.25
    wPACE = (wA + wG)
    wSS = wM
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))
    score = 10.0 * (wT*T + wPACE*LQ + wSS*SS + wEFF*EFF)
    return score

# =======================
# Hidden Abilities v1.3 (Dual-Scale Stable Build)
# =======================
def compute_hidden_abilities(indices: pd.DataFrame) -> pd.DataFrame:
    """
    IA (visible class) from F200+tsSPI+Accel+Grind weighted for 'now' efficiency.
    LP (latent potential) from Accel+Grind emphasis where under-expression is detected.
    HAI = 0.7*IA + 0.3*LP
    Note: We keep numbers around ~100 scale for interpretability.
    """
    # Visible class: balanced composite with slight mid/late emphasis
    IA = 0.20*indices["F200_idx"] + 0.30*indices["tsSPI"] + 0.25*indices["Accel"] + 0.25*indices["Grind"]
    # Latent potential: where late sections suggest upside vs current visible
    # A mild uplift when Accel/Grind exceed F200+tsSPI composite
    late = 0.45*indices["Accel"] + 0.55*indices["Grind"]
    mid = 0.40*indices["tsSPI"] + 0.20*indices["F200_idx"]
    gap = (late - mid)
    LP_base = 0.35*indices["tsSPI"] + 0.65*late
    LP = LP_base + 0.15*gap  # reward latent late power beyond mid/early
    HAI = 0.70*IA + 0.30*LP
    out = pd.DataFrame({"IA": IA, "LP": LP, "HAI": HAI}, index=indices.index)
    return out

# =======================
# Hidden Horses v2 (SOS, ASI², TFS+, UEI → HiddenScore → Tier)
# =======================
def winsorize(s: pd.Series, p=5):
    lo, hi = np.nanpercentile(s, [p, 100-p])
    return s.clip(lo, hi)

def robust_z(s: pd.Series) -> pd.Series:
    med = np.nanmedian(s)
    sigma = mad_std(s.values)
    if sigma <= 1e-9:
        sigma = 1.0
    return (s - med) / sigma

def compute_hidden_horses(indices: pd.DataFrame, distance_m: int) -> pd.DataFrame:
    # SOS from winsorised indices with weighted emphasis
    ts = winsorize(indices["tsSPI"])
    ac = winsorize(indices["Accel"])
    gr = winsorize(indices["Grind"])
    SOS_z = robust_z(0.45*ts + 0.35*ac + 0.20*gr)
    # map to 0..2 around z using smooth function
    SOS = (SOS_z - SOS_z.min()) / (SOS_z.max() - SOS_z.min() + 1e-9) * 2.0

    # ASI²: counter-bias credit when Accel vs Grind imbalance goes against race tendency
    medA, medG = np.nanmedian(indices["Accel"]), np.nanmedian(indices["Grind"])
    bias = medA - medG  # if positive, race rewarded accel vs grind
    # credit horses that went opposite to bias
    ASI2_raw = -bias * ((indices["Accel"] - indices["Grind"]) / 100.0)
    ASI2 = ((ASI2_raw - np.nanmin(ASI2_raw)) / (np.nanmax(ASI2_raw) - np.nanmin(ASI2_raw) + 1e-9)).clip(0,1)

    # TFS+: late trip friction – variability on last 300/200/100 vs mid
    # Without individual per-100m speed arrays here, approximate with |Accel - Grind| relative to tsSPI
    TFS_plus = (np.abs(indices["Accel"] - indices["Grind"]) / 10.0).clip(0, 0.6)

    # UEI: under-used engine when tsSPI strong but late sections capped
    UEI = ((indices["tsSPI"] - 100.0) - (np.maximum(indices["Accel"], indices["Grind"]) - 100.0)) / 10.0
    UEI = UEI.clip(0, 0.6)

    HiddenScore = 0.55*SOS + 0.30*ASI2 + 0.10*TFS_plus + 0.05*UEI
    Tier = np.where(HiddenScore >= 1.8, "Gold",
           np.where(HiddenScore >= 1.2, "Silver", ""))

    return pd.DataFrame({
        "SOS": SOS, "ASI2": ASI2, "TFS_plus": TFS_plus, "UEI": UEI,
        "HiddenScore": HiddenScore, "Tier": Tier
    }, index=indices.index)

# =======================
# Race Type (Archetype) classification
# =======================
def archetype_from_field(indices: pd.DataFrame) -> str:
    # Compute robust z of medians vs 100
    def z_of_median(col):
        med = np.nanmedian(indices[col])
        sigma = mad_std(indices[col].values)
        if sigma <= 1e-9:
            sigma = 1.0
        return (med - 100.0) / sigma

    zF = z_of_median("F200_idx")
    zM = z_of_median("tsSPI")
    zA = z_of_median("Accel")
    zG = z_of_median("Grind")

    UP, DOWN = 0.4, -0.4
    if zF >= UP and zM >= UP and (zA <= 0 or zG <= DOWN):
        return "Front-Burner"
    if abs(zF) <= 0.2 and abs(zM) <= 0.2 and abs(zA) <= 0.2 and abs(zG) <= 0.2:
        return "Even Flow"
    if zF <= DOWN and zM <= DOWN and zA >= UP and zG >= UP:
        return "Stop-Start"
    if zM >= UP and zA <= 0 and abs(zG) <= 0.2:
        return "Mid-Race Pressure"
    if zF >= UP and zM <= DOWN and zA <= DOWN and zG <= DOWN:
        return "Attritional"
    # fallback
    return "Mixed"

# =======================
# Trip classification & text
# =======================
def trip_classification(zF: float, zM: float, zA: float, zG: float, race_type: str, data_ok: bool, field_size: int) -> Tuple[str, str]:
    # Gates (two of... rules)
    sprinter = sum([zF >= 0.6, zA >= 0.6, zG <= 0, zM <= 0.4]) >= 2
    miler = sum([zA >= 0.3, zM >= 0.2, (-0.2 <= zG <= 0.4)]) >= 2
    middle = sum([zM >= 0.6, zG >= 0.0, zF <= 0.0]) >= 2
    stayer = sum([zG >= 0.6, zM >= 0.4, zF <= -0.3]) >= 2

    # Pick category
    chosen = None
    if stayer:
        chosen = "Stayer"
    elif middle:
        chosen = "Middle"
    elif miler:
        chosen = "Miler"
    elif sprinter:
        chosen = "Sprinter"
    else:
        chosen = "Versatile"

    # Confidence
    conf = "Med"
    # higher confidence if ran against the race type bias
    against_shape = (
        (race_type == "Front-Burner" and zG >= 0.6) or
        (race_type == "Stop-Start" and zA >= 0.8) or
        (race_type == "Mid-Race Pressure" and zM >= 0.8) or
        (race_type == "Attritional" and (zG >= 0.6 or zM >= 0.6))
    )
    strong_signals = sum([abs(zF) >= 0.8, abs(zM) >= 0.8, abs(zA) >= 0.8, abs(zG) >= 0.8]) >= 2
    if against_shape and strong_signals:
        conf = "High"
    if (not data_ok) or field_size <= 6 or (max(abs(zF), abs(zM), abs(zA), abs(zG)) < 0.3):
        conf = "Low"

    # Phrase
    if chosen == "Sprinter":
        phrase = "Fast onset and sharp 600–200 lift; thrives 1000–1200 m."
    elif chosen == "Miler":
        phrase = "Balanced pattern with reliable mid-race and turn-of-foot; ideal 1400–1600 m."
    elif chosen == "Middle":
        phrase = "Strong mid-race sustain and steady finish; effective 1800–2000 m."
    elif chosen == "Stayer":
        phrase = "Late strength and cruising stamina; suited 2200 m+."
    else:
        phrase = "Versatile profile; adaptable across adjacent trips."

    return chosen, f"{phrase}"

# =======================
# Narrative generation
# =======================
def race_level_phrases(indices: pd.DataFrame) -> Dict[str, str]:
    def phrase(med_z, hot="Hot phrase", cool="Cool phrase", up=0.5, down=-0.5):
        if med_z >= up:
            return hot
        if med_z <= down:
            return cool
        return "Neutral tempo."
    # compute z of medians vs 100
    def medz(col):
        med = np.nanmedian(indices[col])
        sigma = mad_std(indices[col].values)
        sigma = sigma if sigma > 1e-9 else 1.0
        return (med - 100.0) / sigma

    zF, zM, zA, zG = [medz(c) for c in ["F200_idx","tsSPI","Accel","Grind"]]
    F200_phrase = phrase(zF, hot="Explosive break; fast early tempo.", cool="Steady start; controlled early.")
    tsSPI_phrase = phrase(zM, hot="Strong, sustained middle; cruising power tested.", cool="Slackened mid-section; energy conserved.")
    Accel_phrase = phrase(zA, hot="Race lifted sharply from the 600; decisive surge.", cool="No true change of pace; gradual wind-up.")
    Grind_phrase = phrase(zG, hot="Closers finished powerfully; stamina decisive.", cool="Late output softened; early work told.")
    return dict(F200_phrase=F200_phrase, tsSPI_phrase=tsSPI_phrase, Accel_phrase=Accel_phrase, Grind_phrase=Grind_phrase)

def horse_run_phrase(zF, zM, zA, zG) -> str:
    # Pattern-based selection
    if zA >= 0.8 and zG <= -0.4:
        return "Produced a sharp burst turning for home but couldn’t sustain late."
    if zM >= 0.6 and zG >= 0.4:
        return "Travelled smoothly mid-race and stayed on relentlessly through the line."
    if zF <= -0.4 and zA >= 0.6:
        return "Settled back early, then unleashed a notable late rally."
    if zF >= 0.6 and zG <= -0.4:
        return "Used early speed to advantage but emptied over the final 200 m."
    if max(abs(zF),abs(zM),abs(zA),abs(zG)) <= 0.3:
        return "Ran an even, efficient race without clear phase extremes."
    return "Typical effort, profile consistent with prior pattern."

# =======================
# Visuals (matplotlib) — single plot per figure, no specific colors
# =======================
def plot_shape_map(df_show: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    x = df_show["Accel"] - 100.0
    y = df_show["Grind"] - 100.0
    sizes = (df_show["HAI"] - df_show["HAI"].min()) / (df_show["HAI"].max() - df_show["HAI"].min() + 1e-9) * 400 + 50
    sc = ax.scatter(x, y, s=sizes, c=df_show["tsSPI"], alpha=0.8)
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

def plot_pace_curve(df_times: pd.DataFrame, horses_to_draw: List[str], distance_m: int):
    # Build mean field speed per 100m & plot top-N
    # Convert 100m splits to speeds; plot mean (thicker) and selected horses
    # Note: one plot; we avoid styling (no specific colors)
    # Construct per-100m columns list
    split_cols = [f"{x}_Time" for x in range(distance_m-100, 0, -100)] + ["Finish_Time"]
    have_cols = [c for c in split_cols if c in df_times.columns]
    if not have_cols:
        st.info("No per-100m splits available to draw pace curve.")
        return

    speeds = df_times[have_cols].apply(lambda col: 100.0 / as_num(col), axis=0)
    mean_speed = speeds.mean(axis=0)

    # pick top-N by final placing (if available), else first 8 rows
    N = min(8, len(df_times))
    if "Finish_Pos" in df_times.columns:
        sel = df_times.sort_values("Finish_Pos").head(N).index
    else:
        sel = df_times.head(N).index

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # x-labels: use column names inverted so left->right chronology earliest to Finish
    x_ticks = list(range(len(have_cols)))
    ax.plot(x_ticks, mean_speed.values, linewidth=2.5, label="Field mean")
    for i in sel:
        ax.plot(x_ticks, speeds.loc[i].values, linewidth=1.2)

    ax.set_xticks(x_ticks)
    # Nice labels: convert "1400_Time" -> "1400", "Finish_Time"->"FIN"
    xlabs = [c.replace("_Time","") if c!="Finish_Time" else "FIN" for c in have_cols]
    ax.set_xticklabels(xlabs, rotation=0)
    ax.set_xlabel("Segment (m from finish)")
    ax.set_ylabel("Speed (m/s)")
    # legend with simple handles
    st.pyplot(fig)

# =======================
# Main UI
# =======================
st.title("Race Edge — Analyst v4.0 (Pure Sectional Engine)")
st.caption("Analyst-only. Core phases: F200 · tsSPI · Accel(600–200) · Grind(200–Finish).")

with st.sidebar:
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=3600, value=1600, step=100)
    show_tables = st.checkbox("Show metrics tables", value=True)
    show_narratives = st.checkbox("Show horse narratives", value=True)
    show_visuals = st.checkbox("Show visuals", value=True)
    export_pdf = st.checkbox("(Optional) Export PDF placeholder", value=False)

uploaded = st.file_uploader("Upload race CSV (per-100m split columns like 1400_Time,..., 100_Time, Finish_Time)", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    df = raw.copy()

    # Standardize expected minimal columns
    # Try to detect horse name column
    horse_col = None
    for cand in ["Horse","Horse_Name","Name","Runner"]:
        if cand in df.columns:
            horse_col = cand; break
    if not horse_col:
        df["Horse"] = [f"H{i+1}" for i in range(len(df))]
        horse_col = "Horse"

    if "Finish_Pos" not in df.columns:
        # fabricate sequential finish pos if missing
        df["Finish_Pos"] = np.arange(1, len(df)+1)

    # compute phase speeds and indices
    speeds, meta = compute_phase_speeds(df, distance_m)
    indices = compute_indices(speeds, field_size=len(df))

    # Compose working frame
    work = pd.concat([df, indices], axis=1)
    work["PI"] = compute_PI(indices, distance_m)
    work["GCI"] = compute_GCI(df, indices, distance_m)
    abilities = compute_hidden_abilities(indices)
    work = pd.concat([work, abilities], axis=1)
    hh = compute_hidden_horses(indices, distance_m)
    work = pd.concat([work, hh], axis=1)

    # Race-level narratives
    race_type = archetype_from_field(indices)
    phr = race_level_phrases(indices)
    integrity = f"{meta['completeness']:.0f}%"
    timing_conf = "High" if meta['completeness'] >= 95 else ("Med" if meta['completeness'] >= 85 else "Low")

    st.markdown(f"### Race Header")
    st.write(f"**Distance:** {distance_m} m &nbsp;|&nbsp; **Field:** {len(df)} "
             f"&nbsp;|&nbsp; **Archetype:** {race_type} &nbsp;|&nbsp; **Integrity:** {integrity} "
             f"&nbsp;|&nbsp; **Timing Confidence:** {timing_conf}")

    st.markdown("#### Race Synopsis")
    st.write(f"**Opening tempo (F200):** {phr['F200_phrase']}")
    st.write(f"**Mid-race rhythm (tsSPI):** {phr['tsSPI_phrase']}")
    st.write(f"**Tactical pivot (600–200 Accel):** {phr['Accel_phrase']}")
    st.write(f"**Final drive (Grind):** {phr['Grind_phrase']}")

    if show_visuals:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("##### Pace Curve")
            plot_pace_curve(df, horses_to_draw=[], distance_m=distance_m)
        with col2:
            st.markdown("##### Sectional Shape Map")
            show_cols = [horse_col,"tsSPI","Accel","Grind","HAI"]
            df_show = work[show_cols].rename(columns={horse_col:"Horse"}).copy()
            plot_shape_map(df_show)

    if show_tables:
        st.markdown("#### Metrics Table (Analyst Core)")
        cols = [horse_col, "Finish_Pos"]
        if "RaceTime_s" in work.columns:
            cols += ["RaceTime_s"]
        cols += ["F200_idx","tsSPI","Accel","Grind","PI","GCI","Tier","HiddenScore"]
        table = work[cols].sort_values(["PI","Finish_Pos"], ascending=[False, True])
        st.dataframe(table, use_container_width=True)

        st.markdown("#### Ability Matrix")
        A = work[[horse_col,"IA","LP","HAI","Tier"]].rename(columns={horse_col:"Horse"})
        st.dataframe(A.sort_values("HAI", ascending=False), use_container_width=True)

    if show_narratives:
        st.markdown("#### Horse Narratives")
        # field-level robust sigmas for z-scores
        sig = {k: mad_std(indices[k].values) or 1.0 for k in ["F200_idx","tsSPI","Accel","Grind"]}
        med = {k: np.nanmedian(indices[k]) for k in ["F200_idx","tsSPI","Accel","Grind"]}
        data_ok = meta['completeness'] >= 85

        for _, r in work.sort_values(["PI","Finish_Pos"], ascending=[False, True]).iterrows():
            zF = (r["F200_idx"] - med["F200_idx"]) / (sig["F200_idx"] if sig["F200_idx"]>1e-9 else 1.0)
            zM = (r["tsSPI"]     - med["tsSPI"])     / (sig["tsSPI"]     if sig["tsSPI"]>1e-9 else 1.0)
            zA = (r["Accel"]     - med["Accel"])     / (sig["Accel"]     if sig["Accel"]>1e-9 else 1.0)
            zG = (r["Grind"]     - med["Grind"])     / (sig["Grind"]     if sig["Grind"]>1e-9 else 1.0)

            run_phrase = horse_run_phrase(zF, zM, zA, zG)
            trip_type, trip_text = trip_classification(zF, zM, zA, zG, race_type, data_ok, len(work))

            st.markdown(f"**{r[horse_col]} ({int(r['Finish_Pos'])})** — "
                        f"F200 {zF:+.1f}σ • tsSPI {zM:+.1f}σ • Accel {zA:+.1f}σ • Grind {zG:+.1f}σ.")
            st.write(f"IA {r['IA']:.0f} / LP {r['LP']:.0f} → HAI {r['HAI']:.0f} ({r['Tier'] or '—'}).")
            st.write(run_phrase)
            st.write(f"**Trip:** {trip_type} ({'High' if data_ok and len(work)>6 else 'Med'}) – {trip_text}")
            st.write("---")

    # Sleeper Spotlight
    st.markdown("#### Hidden Performers (Sleeper Spotlight)")
    sleepers = work[work["Tier"].isin(["Gold","Silver"])].copy()
    if sleepers.empty:
        st.write("No Silver/Gold hidden performers flagged by rules.")
    else:
        lines = []
        for _, r in sleepers.sort_values(["Tier","HiddenScore","HAI"], ascending=[True, False, False]).iterrows():
            reason = []
            if r["tsSPI"] >= np.nanmedian(indices["tsSPI"]) + (mad_std(indices["tsSPI"].values) or 1.0)*0.6:
                reason.append("tsSPI strong")
            if r["Grind"] >= np.nanmedian(indices["Grind"]) + (mad_std(indices["Grind"].values) or 1.0)*0.4:
                reason.append("positive Grind")
            if r["Accel"] >= np.nanmedian(indices["Accel"]) + (mad_std(indices["Accel"].values) or 1.0)*0.6:
                reason.append("sharp Accel")
            reason_txt = ", ".join(reason) if reason else "hidden merit"
            lines.append(f"- **{r[horse_col]}** — {reason_txt} → **{r['Tier']}** ({r['HiddenScore']:.2f}).")
        st.markdown("\n".join(lines))

    # Diagnostics
    st.markdown("#### Diagnostics")
    st.write(f"Expected splits: {len(meta['expected_cols'])} | Present: {len(meta['have_cols'])} | Integrity: {integrity}")
    st.write("Missing:", [c for c in meta['expected_cols'] if c not in meta['have_cols']])

    # Optional: mock PDF export placeholder (actual PDF generation omitted here to keep code focused)
    if export_pdf:
        st.info("PDF export placeholder: render this page to PDF using Streamlit's print-to-PDF or add a PDF library later.")

else:
    st.info("Upload a CSV with per-100m split times (e.g., 1400_Time, 1300_Time, ..., 100_Time, Finish_Time). "
            "Include columns 'Horse' and optionally 'Finish_Pos' and 'RaceTime_s'.")
