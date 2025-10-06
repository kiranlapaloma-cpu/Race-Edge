import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Race Edge — Analyst v4.0 (Canon + Upgraded Visuals)", layout="wide")

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

# ------------------ Phase windows & speeds (STRICT) ------------------
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
    """STRICT: all listed splits must exist and be > 0; else NaN."""
    if not cols: return np.nan
    vals = []
    for c in cols:
        v = as_num(row.get(c))
        if pd.isna(v) or v <= 0:
            return np.nan
        vals.append(float(v))
    dist = meters_per_split * len(cols)
    tsum = float(np.sum(vals))
    return dist / tsum if tsum > 0 else np.nan

def grind_speed(row):
    """STRICT: needs both 100_Time and Finish_Time > 0; else NaN."""
    t100 = as_num(row.get("100_Time"))
    tfin = as_num(row.get("Finish_Time"))
    if pd.isna(t100) or t100 <= 0 or pd.isna(tfin) or tfin <= 0:
        return np.nan
    return 200.0 / float(t100 + tfin)

# ------------------ Index stabilizers (canon) ------------------
def shrink_center(idx_series: pd.Series):
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

def pi_weights_distance_and_context(distance_m: float, acc_median: float | None, grd_median: float | None) -> dict:
    dm = float(distance_m or 1200)

    if dm <= 1000:
        base = {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17}
    elif dm < 1100:
        base = _interpolate_weights(dm,
            1000, {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17},
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20}
        )
    elif dm < 1200:
        base = _interpolate_weights(dm,
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
    if acc_med is not None and grd_med is not None and np.isfinite(acc_med) and np.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = np.tanh(abs(bias) / 6.0)
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

# ------------------ Metric build (canon) ------------------
def build_metrics(df_in: pd.DataFrame, D_actual_m: float):
    w = df_in.copy()

    # Rebuild RaceTime_s from available splits if not present
    want = [f"{m}_Time" for m in range(int(D_actual_m) - 100, 99, -100)] + ["Finish_Time"]
    present = [c for c in want if c in w.columns]
    if present:
        w["RaceTime_s"] = w[present].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0, np.nan).sum(axis=1)

    # Stage columns
    D = float(D_actual_m)
    f200_cols  = [c for c in [f"{int(D-100)}_Time", f"{int(D-200)}_Time"] if c in w.columns]
    tssp_cols  = [c for c in stage_block_cols(D, int(D-300), 600) if c in w.columns]
    accel_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]

    # Stage speeds (STRICT)
    w["_F200_spd"] = w.apply(lambda r: (200.0 / sum_times(r, f200_cols)) if len(f200_cols)==2 and pd.notna(sum_times(r, f200_cols)) and sum_times(r, f200_cols)>0 else np.nan, axis=1)
    w["_MID_spd"]  = w.apply(lambda r: stage_speed(r, tssp_cols, meters_per_split=100.0), axis=1)
    w["_ACC_spd"]  = w.apply(lambda r: stage_speed(r, accel_cols, meters_per_split=100.0), axis=1)
    w["_GR_spd"]   = w.apply(grind_speed, axis=1)

    # Indices (100-based with stabilizers)
    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F200_spd"], errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"],  errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"],  errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],   errors="coerce"))

    # PI v3.1: weighted points → standardized to 0..10
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
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

    # GCI (0–10) with original 98..104 map
    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        winner_time = float(pd.to_numeric(w["RaceTime_s"], errors="coerce").min())

    wT   = 0.25
    wPACE= W["Accel"] + W["Grind"]
    wSS  = W["tsSPI"]
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))

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

    return w, {"F200": f200_cols, "tsSPI": tssp_cols, "Accel": accel_cols, "Grind": ["100_Time","Finish_Time"]}

# ------------------ Hidden Abilities & Horses (as before) ------------------
def compute_hidden_abilities(indices: pd.DataFrame) -> pd.DataFrame:
    IA = 0.20*indices["F200_idx"] + 0.30*indices["tsSPI"] + 0.25*indices["Accel"] + 0.25*indices["Grind"]
    late = 0.45*indices["Accel"] + 0.55*indices["Grind"]
    mid  = 0.40*indices["tsSPI"] + 0.20*indices["F200_idx"]
    gap  = (late - mid)
    LP_base = 0.35*indices["tsSPI"] + 0.65*late
    LP = LP_base + 0.15*gap
    HAI = 0.70*IA + 0.30*LP
    return pd.DataFrame({"IA":IA, "LP":LP, "HAI":HAI})

def compute_hidden_horses(df: pd.DataFrame) -> pd.DataFrame:
    hh = df.copy()
    # SOS
    ts = pd.to_numeric(hh["tsSPI"], errors="coerce")
    ac = pd.to_numeric(hh["Accel"], errors="coerce")
    gr = pd.to_numeric(hh["Grind"], errors="coerce")
    def rz(s):
        mu = float(np.nanmedian(s)); sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd
    z_ts = rz(ts).clip(-2.5, 3.5)
    z_ac = rz(ac).clip(-2.5, 3.5)
    z_gr = rz(gr).clip(-2.5, 3.5)
    hh["SOS_raw"] = 0.45*z_ts + 0.35*z_ac + 0.20*z_gr
    q5, q95 = np.nanpercentile(hh["SOS_raw"], [5,95])
    denom = (q95 - q5) if (np.isfinite(q5) and np.isfinite(q95) and q95>q5) else 1.0
    hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(0.0, 2.0)
    # ASI² (counter-bias late)
    medA, medG = np.nanmedian(ac), np.nanmedian(gr)
    bias = (medA - 100.0) - (medG - 100.0)
    B = min(1.0, abs(bias) / 4.0)
    S = ac - gr
    if bias >= 0:
        hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
    else:
        hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)
    # TFS_plus approximate
    hh["TFS_plus"] = (np.abs(ac - gr) / 10.0).clip(0, 0.6)
    # UEI
    hh["UEI"] = ((ts - 100.0) - (np.maximum(ac, gr) - 100.0)) / 10.0
    hh["UEI"] = hh["UEI"].clip(0, 0.6)
    hidden = 0.55*hh["SOS"] + 0.30*hh["ASI2"] + 0.10*hh["TFS_plus"] + 0.05*hh["UEI"]
    hh["HiddenScore"] = hidden.clip(0, 3.0)
    hh["Tier"] = np.where(hh["HiddenScore"]>=1.8, "🔥 Top Hidden",
                   np.where(hh["HiddenScore"]>=1.2, "🟡 Notable Hidden",""))
    return hh[["SOS","ASI2","TFS_plus","UEI","HiddenScore","Tier"]]

# ------------------ Visuals (upgraded) ------------------
def plot_pace_curve(df_times: pd.DataFrame, distance_m: int):
    # Build column order D-100..100 + Finish
    split_cols = [f"{x}_Time" for x in range(distance_m - 100, 0, -100)] + ["Finish_Time"]
    have_cols = [c for c in split_cols if c in df_times.columns]
    if not have_cols:
        st.info("No per-100m splits available to draw pace curve.")
        return

    # speeds per 100m
    speeds = df_times[have_cols].apply(lambda col: 100.0 / as_num(col), axis=0)
    mean_speed = speeds.mean(axis=0)

    # choose up to 8 to label (top by finish)
    N = min(8, len(df_times))
    if "Finish_Pos" in df_times.columns:
        sel = df_times.sort_values("Finish_Pos").head(N).index
    else:
        sel = df_times.head(N).index

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_ticks = list(range(len(have_cols)))

    # field mean (thicker)
    ax.plot(x_ticks, mean_speed.values, linewidth=2.5, label="Field Mean")

    # selected horses
    for i in sel:
        label = str(df_times.loc[i].get("Horse", f"H{i+1}"))
        ax.plot(x_ticks, speeds.loc[i].values, linewidth=1.3, label=label)

    # clean axes
    xlabs = [c.replace("_Time", "") if c != "Finish_Time" else "FIN" for c in have_cols]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xlabs, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("Segment (m from finish)", fontsize=10)
    ax.set_ylabel("Speed (m/s)", fontsize=10)
    ax.set_title(f"Pace Curve — 100 m Segment Speeds ({distance_m} m Race)", fontsize=11, pad=8)

    # neat legend (key) below the plot
    lg = ax.legend(loc="upper center",
                   bbox_to_anchor=(0.5, -0.20),
                   ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    st.pyplot(fig)

def plot_shape_map(df_show: pd.DataFrame):
    # expects columns: Horse, tsSPI, Accel, Grind, HAI
    fig, ax = plt.subplots(figsize=(6.6, 5.8))

    x = df_show["Accel"] - 100.0   # +X = stronger Accel (600→200)
    y = df_show["Grind"] - 100.0   # +Y = stronger Grind (200→Finish)

    # dot size scaled by HAI
    size = (df_show["HAI"] - df_show["HAI"].min()) / (df_show["HAI"].max() - df_show["HAI"].min() + 1e-9)
    sizes = (size * 400) + 60

    # color = tsSPI deviation
    sc = ax.scatter(x, y, s=sizes, c=df_show["tsSPI"], alpha=0.9)

    # names
    for _, r in df_show.iterrows():
        ax.text(r["Accel"] - 100.0, r["Grind"] - 100.0, str(r["Horse"]),
                fontsize=8, ha="center", va="center")

    # axes & origin
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlabel("Acceleration (600→200) vs field (Δ index)")
    ax.set_ylabel("Grind (200→Finish) vs field (Δ index)")
    ax.set_title("Sectional Shape Map")

    # colorbar label
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("tsSPI deviation (index)")

    # quadrant annotations (neat, with arrows)
    ax.annotate("Sustainers\n(Accel ↑ • Grind ↑)",
                xy=(2.0, 2.0), xytext=(4.5, 6.0),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=9, ha="left")
    ax.annotate("Stayers / Grinders\n(Accel ↓ • Grind ↑)",
                xy=(-2.0, 2.0), xytext=(-8.0, 6.0),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=9, ha="right")
    ax.annotate("Burst Sprinters\n(Accel ↑ • Grind ↓)",
                xy=(2.0, -2.0), xytext=(6.5, -6.0),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=9, ha="left")
    ax.annotate("Spent early / Faded\n(Accel ↓ • Grind ↓)",
                xy=(-2.0, -2.0), xytext=(-8.0, -6.0),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=9, ha="right")

    # mini size legend for HAI
    for s, lab in [(60, "lower HAI"), (260, "mid HAI"), (460, "higher HAI")]:
        ax.scatter([], [], s=s, label=lab)
    size_leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                         frameon=False, fontsize=8, title="Dot size:")
    ax.add_artist(size_leg)

    st.pyplot(fig)

# ------------------ UI ------------------
st.title("Race Edge — Analyst v4.0 (Canon + Upgraded Visuals)")
st.caption("Strict canon metrics • Clear key on pace curve • Annotated shape map.")

with st.sidebar:
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=3600, value=1600, step=100)
    show_tables = st.checkbox("Show metrics tables", value=True)
    show_visuals = st.checkbox("Show visuals", value=True)

uploaded = st.file_uploader("Upload race CSV (… 1600_Time, …, 100_Time, Finish_Time + Horse, Finish_Pos, RaceTime_s)", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
if "Horse" not in df.columns:
    for cand in ["Horse_Name","Name","Runner"]:
        if cand in df.columns:
            df = df.rename(columns={cand:"Horse"}); break
if "Finish_Pos" not in df.columns:
    df["Finish_Pos"] = np.arange(1, len(df)+1)

metrics, phase_cols = build_metrics(df, float(distance_m))

indices = metrics[["F200_idx","tsSPI","Accel","Grind"]]
abilities = compute_hidden_abilities(indices)
metrics = pd.concat([metrics, abilities], axis=1)
hidden = compute_hidden_horses(metrics)
metrics = pd.concat([metrics, hidden], axis=1)

st.subheader("Race Header")
st.write(f"**Distance:** {int(distance_m)} m | **Field:** {len(df)}")
st.write("**Phase columns:**", phase_cols)

if show_visuals:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pace Curve")
        plot_pace_curve(df, int(distance_m))
    with c2:
        st.markdown("#### Sectional Shape Map")
        show_cols = ["Horse","tsSPI","Accel","Grind","HAI"]
        plot_shape_map(metrics[show_cols])

if show_tables:
    st.markdown("### Metrics Table (Analyst Core)")
    cols = ["Horse","Finish_Pos"]
    if "RaceTime_s" in metrics.columns: cols += ["RaceTime_s"]
    cols += ["F200_idx","tsSPI","Accel","Grind","PI","GCI","Tier","HiddenScore"]
    st.dataframe(metrics[cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]), use_container_width=True)

    st.markdown("### Ability Matrix")
    st.dataframe(metrics[["Horse","IA","LP","HAI","Tier"]].sort_values("HAI", ascending=False), use_container_width=True)
