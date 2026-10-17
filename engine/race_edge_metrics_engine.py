"""Race Edge core sectional metrics engine.

Extracted from the working Streamlit app without changing formulas.
"""
import math
import numpy as np
import pandas as pd
# Identical robust dispersion helper from the original performance.py.
# Inlined to avoid importing Streamlit and Matplotlib on the API server.
def mad_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def _collect_markers(df: pd.DataFrame):
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    return sorted(set(marks), reverse=True)

def _sum_times(row, cols):
    vals = [pd.to_numeric(row.get(c), errors="coerce") for c in cols]
    vals = [float(v) for v in vals if pd.notna(v) and v > 0]
    return np.sum(vals) if vals else np.nan

def _make_range_cols(D, start_incl, end_incl, step):
    if start_incl < end_incl:  # defensive
        return []
    want = list(range(int(start_incl), int(end_incl)-1, -int(step)))
    return [f"{m}_Time" for m in want]

def _stage_speed(row, cols, meters_per_split):
    if not cols: return np.nan
    tsum = _sum_times(row, cols)
    if not (pd.notna(tsum) and tsum > 0): return np.nan
    valid = [c for c in cols if pd.notna(row.get(c)) and pd.to_numeric(row.get(c), errors="coerce") > 0]
    dist = meters_per_split * len(valid)
    return np.nan if dist <= 0 else dist / tsum

def _grind_speed(row, step):
    # 100m: last 100 + Finish (100); 200m: Finish (200)
    if step == 100:
        t100 = pd.to_numeric(row.get("100_Time"), errors="coerce")
        tfin = pd.to_numeric(row.get("Finish_Time"), errors="coerce")
        parts, dist = [], 0.0
        if pd.notna(t100) and t100 > 0: parts.append(float(t100)); dist += 100.0
        if pd.notna(tfin) and tfin > 0: parts.append(float(tfin)); dist += 100.0
        return np.nan if not parts or dist <= 0 else dist / sum(parts)
    else:
        tfin = pd.to_numeric(row.get("Finish_Time"), errors="coerce")
        return np.nan if (pd.isna(tfin) or tfin <= 0) else 200.0 / float(tfin)

def _pct_at_or_above(s, thr):
    s = pd.to_numeric(s, errors="coerce")
    s = s[s.notna()]
    return 0.0 if s.empty else float((s >= thr).mean())

# -------- Adaptive F-window + tsSPI start --------
def _adaptive_f_cols_and_dist(D, step, markers, frame_cols):
    """
    Returns (f_cols, f_dist_m) for the 'early panel'.
    Rules (your spec):
      100m:  D ending with ..50  → F150 = [D-50, D-150]
             else                → F200 = [D-100, D-200]
      200m:  infer first-span from D - first_marker
             ~100 → F100 ; ~160 → F160 ; ~200 → F200 ; ~250 → F250
             (implemented by bucketing first-span)
    """
    if not markers:
        return [], 0.0

    D = float(D); step = int(step)
    if step == 100:
        if int(D) % 100 == 50:
            wanted = [int(D-50), int(D-150)]
            cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in frame_cols]
            dist = 150.0 if len(cols) == 2 else 100.0 * len(cols)
        else:
            wanted = [int(D-100), int(D-200)]
            cols = [f"{m}_Time" for m in wanted if f"{m}_Time" in frame_cols]
            dist = 100.0 * len(cols)
        return cols, float(dist)

    # 200m data
    m1 = int(markers[0])
    first_span = max(1.0, D - m1)  # m
    c = f"{m1}_Time"
    cols = [c] if c in frame_cols else []
    # bucket by span (loose thresholds are robust to rounding)
    if first_span <= 120:    dist = 100.0
    elif first_span <= 180:  dist = 160.0
    elif first_span <= 220:  dist = 200.0
    else:                    dist = 250.0
    return cols, float(dist)

def _adaptive_tssp_start(D, step, markers):
    """Return the first sectional marker after the opening block.

    For odd-distance 200m races (for example 1160m), the markers are aligned to
    the course rather than to ``distance - 200``.  Using arithmetic such as
    1160 - 150 = 1010 creates column names that do not exist.  The actual next
    marker in the uploaded file is therefore used.
    """
    D = float(D); step = int(step)
    if step == 100:
        # F150 ends at D-150; the next 100m split is labelled D-250.
        return int(D - (250 if int(D) % 100 == 50 else 300))
    if not markers:
        return int(D - 400)
    ordered = sorted({int(m) for m in markers}, reverse=True)
    if len(ordered) >= 2:
        return ordered[1]
    return ordered[0]

# -------- Speed→Index mapping (robust to small fields) --------
def _shrink_center(idx_series):
    x = pd.to_numeric(idx_series, errors="coerce").dropna().values
    if x.size == 0: return 100.0, 0
    med_race = float(np.median(x))
    alpha = x.size / (x.size + 6.0)
    return alpha * med_race + (1 - alpha) * 100.0, x.size

def _dispersion_equalizer(delta_series, n_eff, N_ref=10, beta=0.20, cap=1.20):
    gamma = 1.0 + beta * max(0, N_ref - n_eff) / N_ref
    return delta_series * min(gamma, cap)

def _variance_floor(idx_series, floor=1.5, cap=1.25):
    deltas = idx_series - 100.0
    sigma = mad_std(deltas)
    if not np.isfinite(sigma) or sigma <= 0: return idx_series
    if sigma < floor:
        factor = min(cap, floor / sigma)
        return 100.0 + deltas * factor
    return idx_series

def _speed_to_idx(spd_series):
    s = pd.to_numeric(spd_series, errors="coerce")
    med = s.median(skipna=True)
    idx_raw = 100.0 * (s / med)
    center, n_eff = _shrink_center(idx_raw)
    idx = 100.0 * (s / (center/100.0 * med))
    idx = 100.0 + _dispersion_equalizer(idx - 100.0, n_eff)
    idx = _variance_floor(idx)
    return idx

# -------- Distance/context weights for PI --------
def _lerp(a, b, t): return a + (b - a) * float(t)

def _interp_weights(dm, a_dm, a_w, b_dm, b_w):
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

def pi_weights_distance_and_context(
    distance_m: float,
    acc_med: float|None,
    grd_med: float|None,
    going: str = "Good",
    field_n: int | None = None,
    return_meta: bool = False
) -> dict | tuple[dict, dict]:
    """
    Returns PI weights (sum=1). If return_meta=True, also returns a meta dict
    describing the going multipliers actually applied.
    Going affects PI weighting only; the underlying sectional indices remain unchanged.
    """

    dm = float(distance_m or 1200)

    # ---- Continuous base weights by distance ----
    # Refined to reward complete sprint performances and genuine staying strength.
    # Values between anchors are linearly interpolated, avoiding abrupt distance bands.
    anchors = [
        (1000.0, {"F200_idx":0.12, "tsSPI":0.23, "Accel":0.30, "Grind":0.35}),
        (1200.0, {"F200_idx":0.10, "tsSPI":0.25, "Accel":0.32, "Grind":0.33}),
        (1400.0, {"F200_idx":0.10, "tsSPI":0.30, "Accel":0.35, "Grind":0.25}),
        (1600.0, {"F200_idx":0.08, "tsSPI":0.34, "Accel":0.34, "Grind":0.24}),
        (2000.0, {"F200_idx":0.07, "tsSPI":0.37, "Accel":0.31, "Grind":0.25}),
        (2400.0, {"F200_idx":0.05, "tsSPI":0.40, "Accel":0.28, "Grind":0.27}),
        (3000.0, {"F200_idx":0.03, "tsSPI":0.42, "Accel":0.23, "Grind":0.32}),
    ]
    if dm <= anchors[0][0]:
        base = anchors[0][1].copy()
    elif dm >= anchors[-1][0]:
        base = anchors[-1][1].copy()
    else:
        base = anchors[-1][1].copy()
        for (d0, w0), (d1, w1) in zip(anchors, anchors[1:]):
            if d0 <= dm <= d1:
                base = _interp_weights(dm, d0, w0, d1, w1)
                break
    F200, ts, ACC, GR = (base["F200_idx"], base["tsSPI"], base["Accel"], base["Grind"])

    # ---- Mild context nudge (your bias step) ----
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale
        F200, ts, ACC, GR = base["F200_idx"], base["tsSPI"], base["Accel"], base["Grind"]
        if bias > 0:
            delta = min(max_shift, ACC - 0.26); ACC -= delta; GR += delta
        elif bias < 0:
            delta = min(max_shift, GR - 0.18); GR -= delta; ACC += delta
        GR = min(GR, 0.40); ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":GR}

    # ---- Going modulation (affects PI weighting ONLY) ----
    # Field-size damper: full effect by 12+ runners; scale down in small fields
    n = max(1, int(field_n or 12))
    field_scale = min(1.0, n / 12.0)

    # Going multipliers (before renormalization)
    # Good = neutral (all 1.0)
    # Firm: reward Accel & F200; soften Grind & tsSPI
    # Soft: reward Grind & tsSPI; soften Accel & F200
    # Heavy: stronger version of Soft
    if going == "Firm":
        # Firm: boost Accel ONLY
        amp = 0.03 * field_scale
        mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0 + amp, "Grind": 1.0}

    elif going == "Soft":
        # Soft: boost Grind ONLY
        amp = 0.03 * field_scale
        mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0 + amp}

    elif going == "Heavy":
        # Heavy: boost Grind ONLY (stronger)
        amp = 0.05 * field_scale
        mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0 + amp}

    else:  # "Good" or unknown
        mult = {"F200_idx": 1.0, "tsSPI": 1.0, "Accel": 1.0, "Grind": 1.0}

    weighted = {k: base[k] * mult[k] for k in base.keys()}
    s = sum(weighted.values()) or 1.0
    out = {k: v / s for k, v in weighted.items()}

    if not return_meta:
        return out
    meta = {
        "going": going,
        "field_n": n,
        "multipliers": mult,
        "base": base.copy(),
        "final": out.copy()
    }
    return out, meta

# -------- Core builder --------
def build_metrics_and_shape(df_in: pd.DataFrame,
                            D_actual_m: float,
                            step: int,
                            use_cg: bool,
                            dampen_cg: bool,
                            use_race_shape: bool,
                            debug: bool,
                            going_type: str = "Good"):
    """Build Race Edge metrics and race-shape outputs.

    ``going_type`` is passed explicitly by the Streamlit controller so the
    modular engine no longer relies on a global variable from streamlit_app.py.
    """
    w = df_in.copy()
    seg_markers = _collect_markers(w)
    D = float(D_actual_m); step = int(step)

    # per-segment speeds (raw)
    for m in seg_markers:
        w[f"spd_{m}"] = (step * 1.0) / pd.to_numeric(w.get(f"{m}_Time"), errors="coerce")
    w["spd_Finish"] = ((100.0 if step == 100 else 200.0) /
                       pd.to_numeric(w.get("Finish_Time"), errors="coerce")) if "Finish_Time" in w.columns else np.nan

    # ----- SRI: Speed Retention Index -----
    # Peak_Speed = fastest sectional speed
    # Peak_Location = section where the peak happened (remaining metres marker, FIN for finish split)
    # SRI = average speed from the peak section through the finish ÷ peak speed × 100
    # This works for both 100m and 200m files, including odd first panels.
    def _sri_profile(val):
        if pd.isna(val): return "-"
        try:
            x = float(val)
        except Exception:
            return "-"
        if x >= 96.0: return "Elite sustainer"
        if x >= 94.0: return "Strong sustainer"
        if x >= 92.0: return "Balanced"
        if x >= 90.0: return "Tactical"
        return "Peak-and-fade"

    sri_segments = []  # race order: early -> finish, each as (location_label, length_m, time_col)
    if seg_markers:
        first_m = int(seg_markers[0])
        first_len = max(1.0, D - first_m)
        first_col = f"{first_m}_Time"
        if first_col in w.columns:
            sri_segments.append((str(first_m), float(first_len), first_col))
        for a, b in zip(seg_markers, seg_markers[1:]):
            col = f"{int(b)}_Time"
            if col in w.columns:
                sri_segments.append((str(int(b)), float(a - b), col))
    if "Finish_Time" in w.columns:
        sri_segments.append(("FIN", float(100.0 if step == 100 else 200.0), "Finish_Time"))

    def _sri_row(r):
        speeds, labels = [], []
        for lab, dist_m, col in sri_segments:
            t = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(t) and t > 0 and dist_m > 0:
                speeds.append(float(dist_m) / float(t))
                labels.append(lab)
            else:
                speeds.append(np.nan)
                labels.append(lab)
        arr = np.asarray(speeds, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            return pd.Series({"Peak_Speed": np.nan, "Peak_Location": "-", "SRI": np.nan, "SRI_Profile": "-"})
        peak_i = int(np.nanargmax(arr))
        peak = float(arr[peak_i])
        tail = arr[peak_i:]
        tail = tail[np.isfinite(tail)]
        if peak <= 0 or tail.size == 0:
            sri = np.nan
        else:
            sri = float(np.nanmean(tail) / peak * 100.0)
        return pd.Series({
            "Peak_Speed": round(peak, 3),
            "Peak_Location": labels[peak_i],
            "SRI": round(sri, 2) if np.isfinite(sri) else np.nan,
            "SRI_Profile": _sri_profile(sri)
        })

    if sri_segments:
        sri_out = w.apply(_sri_row, axis=1)
        for c in ["Peak_Speed", "Peak_Location", "SRI", "SRI_Profile"]:
            w[c] = sri_out[c]
    else:
        w["Peak_Speed"] = np.nan
        w["Peak_Location"] = "-"
        w["SRI"] = np.nan
        w["SRI_Profile"] = "-"

    # RaceTime = sum the actual sectional columns present (incl Finish).
    # This is essential for odd trips such as 1160m, whose markers are commonly
    # 1000, 800, 600, 400 and 200 rather than 960, 760, 560, etc.
    if seg_markers:
        wanted = [f"{int(m)}_Time" for m in seg_markers if f"{int(m)}_Time" in w.columns]
        if "Finish_Time" in w.columns:
            wanted.append("Finish_Time")
        w["RaceTime_s"] = w[wanted].apply(pd.to_numeric, errors="coerce").clip(lower=0).replace(0,np.nan).sum(axis=1)
    else:
        w["RaceTime_s"] = pd.to_numeric(w.get("Race Time"), errors="coerce")

    # ----- Composite speeds -----
    f_cols, f_dist = _adaptive_f_cols_and_dist(D, step, seg_markers, w.columns)
    w["_F_spd"]   = w.apply(lambda r: (f_dist / _sum_times(r, f_cols)) if (f_cols and pd.notna(_sum_times(r,f_cols)) and _sum_times(r,f_cols)>0) else np.nan, axis=1)

    tssp_start    = _adaptive_tssp_start(D, step, seg_markers)
    mid_cols      = [c for c in _make_range_cols(D, tssp_start, 600, step) if c in w.columns]
    w["_MID_spd"] = w.apply(lambda r: _stage_speed(r, mid_cols, float(step)), axis=1)

    if step == 100:
        # Accel always represents 600m remaining to 200m remaining.
        # With 100m data this is four consecutive 100m panels.
        acc_cols = [c for c in [f"{m}_Time" for m in [500,400,300,200]] if c in w.columns]
    else:
        # With 200m data the same physical 400m phase is represented by
        # the 600->400 and 400->200 panels, stored as 400_Time and 200_Time.
        acc_cols = [c for c in [f"{m}_Time" for m in [400,200]] if c in w.columns]
    w["_ACC_spd"] = w.apply(lambda r: _stage_speed(r, acc_cols, float(step)), axis=1)

    w["_GR_spd"]  = w.apply(lambda r: _grind_speed(r, step), axis=1)

    # ----- Speed → Indices -----
    w["F200_idx"] = _speed_to_idx(w["_F_spd"])
    w["tsSPI"]    = _speed_to_idx(w["_MID_spd"])
    w["Accel"]    = _speed_to_idx(w["_ACC_spd"])
    w["Grind"]    = _speed_to_idx(w["_GR_spd"])

    # ----- TOF: Turn of Foot Differential -----
    # tsSPI and Accel both sit around 100, so division adds little signal.
    # TOF = Accel - tsSPI isolates how much sharper the horse became in the acceleration phase
    # compared with its sustained/travel phase.
    w["TOF"] = (pd.to_numeric(w["Accel"], errors="coerce") -
                pd.to_numeric(w["tsSPI"], errors="coerce")).round(2)

    def _tof_profile(val):
        if pd.isna(val): return "-"
        try:
            x = float(val)
        except Exception:
            return "-"
        if x >= 3.0: return "Explosive turn of foot"
        if x >= 1.5: return "Sharp accelerator"
        if x >= 0.5: return "Tactical kick"
        if x > -0.5: return "Balanced"
        if x > -1.5: return "Sustainer"
        return "Builder / grinder"

    w["TOF_Profile"] = w["TOF"].map(_tof_profile)

    # ----- Corrected Grind (CG) -----
    ACC_field = pd.to_numeric(w["_ACC_spd"], errors="coerce").mean(skipna=True)
    GR_field  = pd.to_numeric(w["_GR_spd"],  errors="coerce").mean(skipna=True)
    FSR = float(GR_field / ACC_field) if (math.isfinite(ACC_field) and ACC_field > 0 and math.isfinite(GR_field)) else np.nan
    if not np.isfinite(FSR): FSR = 1.0
    CollapseSeverity = float(min(10.0, max(0.0, (0.95 - FSR) * 100.0)))

    def _delta_g_row(r):
        mid, grd = float(r.get("_MID_spd", np.nan)), float(r.get("_GR_spd", np.nan))
        if not (math.isfinite(mid) and math.isfinite(grd) and mid > 0): return np.nan
        return 100.0 * (grd / mid)
    w["DeltaG"] = w.apply(_delta_g_row, axis=1)

    w["FinisherFactor"] = w["DeltaG"].map(lambda dg: 0.0 if not math.isfinite(dg) else float(clamp((dg-98.0)/4.0, 0.0, 1.0)))
    w["GrindAdjPts"]    = (CollapseSeverity * (1.0 - w["FinisherFactor"])).round(2)

    w["Grind_CG"] = (w["Grind"] - w["GrindAdjPts"]).clip(lower=90.0, upper=110.0)
    def _fade_cap(g, dg):
        if not (math.isfinite(g) and math.isfinite(dg)): return g
        return 100.0 + 0.5*(g-100.0) if (dg < 97.0 and g > 100.0) else g
    w["Grind_CG"] = [ _fade_cap(g, dg) for g, dg in zip(w["Grind_CG"], w["DeltaG"]) ]

    # ----- PI v4.4: actual-opening-block four-core model -----
    # PI is based only on the opening-block index, tsSPI, Accel and raw Grind.
    # The opening index receives the exact percentage of the race represented
    # by the detected opening block (for example F100, F150, F160, F200 or F250).
    # The remaining influence is divided using:
    # tsSPI : Accel : Grind = 1 : (1 + 200 / race distance) : 1.
    # The Accel premium is therefore tied to the percentage that an additional
    # 200 m represents of the complete race, while the opening weight remains
    # faithful to the actual sectional block used.
    GR_COL = "Grind"

    race_distance = max(float(D), 1.0)
    opening_block_m = float(np.clip(f_dist, 0.0, race_distance))
    w_f200 = opening_block_m / race_distance
    remaining = max(0.0, 1.0 - w_f200)
    accel_ratio = 1.0 + (200.0 / race_distance)
    ratio_total = 1.0 + accel_ratio + 1.0

    PI_W = {
        "F200_idx": w_f200,
        "tsSPI": remaining * (1.0 / ratio_total),
        "Accel": remaining * (accel_ratio / ratio_total),
        "Grind": remaining * (1.0 / ratio_total),
    }

    # Keep metadata available to the dashboard, exports and reports.
    w.attrs["GOING"] = going_type
    w.attrs["PI_GOING_META"] = {
        "method": "actual_opening_block_ratio_1_1_plus_200_over_distance_1",
        "distance_m": int(round(race_distance)),
        "opening_block_m": round(opening_block_m, 2),
        "opening_label": f"F{int(round(opening_block_m))}" if opening_block_m > 0 else "Opening",
        "weights": PI_W.copy(),
    }
    w.attrs["PI_MASS_NOTE"] = {
        "mass_col": "(not used in PI)",
        "ref_kg": None,
        "perkg_pts": 0.0,
        "distance_m": int(round(race_distance)),
    }

    def _pi_pts_row(r):
        values = {
            "F200_idx": pd.to_numeric(pd.Series([r.get("F200_idx")]), errors="coerce").iloc[0],
            "tsSPI": pd.to_numeric(pd.Series([r.get("tsSPI")]), errors="coerce").iloc[0],
            "Accel": pd.to_numeric(pd.Series([r.get("Accel")]), errors="coerce").iloc[0],
            "Grind": pd.to_numeric(pd.Series([r.get("Grind")]), errors="coerce").iloc[0],
        }
        valid = {k: v for k, v in values.items() if np.isfinite(v)}
        if not valid:
            return np.nan

        valid_weight = sum(PI_W[k] for k in valid) or 1.0
        return sum(PI_W[k] * (v - 100.0) for k, v in valid.items()) / valid_weight

    w["Sprint_Conversion_Penalty"] = 0.0
    w["PI_pts"] = w.apply(_pi_pts_row, axis=1)

    w.attrs["PI_REFINED_META"] = {
        "distance_m": int(round(race_distance)),
        "weights": PI_W.copy(),
        "method": "f200_distance_proportional_remaining_ratio_1_1_plus_200_over_distance_1",
        "inputs": ["F200_idx", "tsSPI", "Accel", "Grind"],
        "uses_race_time": False,
        "uses_corrected_grind": False,
        "uses_weight_adjustment": False,
    }

    # Race-relative robust linear scale. The field median centres on 5,
    # while genuine above- and below-field separation is preserved.
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    finite_pts = pts[np.isfinite(pts)]
    med = float(np.nanmedian(finite_pts)) if len(finite_pts) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    sigma = 0.75 if (not np.isfinite(sigma) or sigma < 0.75) else float(sigma)

    z_score = centered / sigma
    w["PI"] = (
        5.0 + 1.5 * z_score
    ).clip(0.5, 9.5).round(2)

    w.attrs["PI_REFINED_META"].update({
        "scale_method": "robust_linear_median_mad",
        "scale_slope": 1.5,
        "scale_floor": 0.5,
        "scale_ceiling": 9.5,
        "median_center": 5.0,
        "mad_floor": 0.75,
    })

    # ----- Phase PI decomposition -----
    # Diagnostic weighted contributions showing each phase's raw influence.
    total_pi_w = sum(PI_W.values()) or 1.0
    pi_scale = 1.0
    pi_intercept = 5.0

    def _phase_component(series, weight):
        vals = pd.to_numeric(series, errors="coerce")
        return (weight * (vals - 100.0)) / total_pi_w

    w["PI_Base"] = 5.0
    w["F200_PIc"] = _phase_component(w.get("F200_idx"), PI_W["F200_idx"]).round(3)
    w["tsSPI_PIc"] = _phase_component(w.get("tsSPI"), PI_W["tsSPI"]).round(3)
    w["Accel_PIc"] = _phase_component(w.get("Accel"), PI_W["Accel"]).round(3)
    w["Grind_PIc"] = _phase_component(w.get("Grind"), PI_W["Grind"]).round(3)
    w["Mass_PIc"] = 0.0
    w["Conversion_PIc"] = 0.0

    phase_cols_calc = ["F200_PIc", "tsSPI_PIc", "Accel_PIc", "Grind_PIc"]
    w["PI_Phase_Total"] = (
        5.0
        + w[phase_cols_calc].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    ).clip(0.0, 10.0).round(2)

    def _main_phase_from_row(r):
        vals = {
            "F200": r.get("F200_PIc"),
            "tsSPI": r.get("tsSPI_PIc"),
            "Accel": r.get("Accel_PIc"),
            "Grind": r.get("Grind_PIc"),
        }
        vals = {
            k: float(v) for k, v in vals.items()
            if pd.notna(v) and np.isfinite(float(v))
        }
        return max(vals, key=vals.get) if vals else "-"

    w["Main_PI_Phase"] = w.apply(_main_phase_from_row, axis=1)
    w.attrs["PI_PHASE_WEIGHTS"] = PI_W.copy()
    w.attrs["PI_PHASE_SCALE"] = float(pi_scale)
    w.attrs["PI_PHASE_BASE"] = float(pi_intercept)
# ---------- /Mass-aware PI points ----------

    # GCI removed: PI is now the sole overall performance rating.
        # ----- EARLY/LATE (blended, for display only) -----
    w["EARLY_idx"] = (0.65*pd.to_numeric(w["F200_idx"], errors="coerce") +
                      0.35*pd.to_numeric(w["tsSPI"],    errors="coerce"))
    w["LATE_idx"]  = (0.60*pd.to_numeric(w["Accel"],    errors="coerce") +
                      0.40*pd.to_numeric(w[GR_COL],     errors="coerce"))

         # --- put this right BEFORE the "Race Shape gates (UNIVERSAL, eased)" block ---

    # Which grind column are we using?
    w.attrs["GR_COL"] = GR_COL
    w.attrs["STEP"]   = step
    w.attrs["FSR"]    = FSR
    w.attrs["CollapseSeverity"] = CollapseSeverity

    # Series for shape calculations
    acc = pd.to_numeric(w["Accel"],  errors="coerce")
    mid = pd.to_numeric(w["tsSPI"],  errors="coerce")
    grd = pd.to_numeric(w[GR_COL],   errors="coerce")

    # Deltas: late vs mid, and finish vs late
    dLM = acc - mid   # +ve = late stronger than mid  → SLOW_EARLY candidate
    dLG = grd - acc   # +ve = grind tougher than late → Attritional finish

        # ========================= Race Shape (Eureka RSI) =========================
    # Uses the SAME primitives you already calculated: tsSPI, Accel, Grind[_CG]
    # Sign convention:
    #   RSI > 0  → Slow-Early / Sprint-home bias (late favoured)
    #   RSI < 0  → Fast-Early / Attritional bias (early favoured)
    # Scale: approximately -10..+10 for human sense-making.

    acc = pd.to_numeric(w["Accel"], errors="coerce")
    mid = pd.to_numeric(w["tsSPI"], errors="coerce")
    grd = pd.to_numeric(w[GR_COL], errors="coerce")

    # Core axis (late minus mid): positive = slow-early; negative = fast-early
    dLM = (acc - mid)

    # Finish flavour axis (grind minus accel): positive = attritional finish; negative = sprint finish
    dLG = (grd - acc)

    def _madv(s):
        v = mad_std(pd.to_numeric(s, errors="coerce"))
        return 0.0 if (not np.isfinite(v)) else float(v)

    # 1) Consensus (SCI) on the shape direction using dLM signs
    sgn = np.sign(dLM.dropna().to_numpy())
    if sgn.size:
        sgn_med = int(np.sign(np.median(dLM.dropna())))
        sci = float((sgn == sgn_med).mean()) if sgn_med != 0 else 0.0
    else:
        sgn_med = 0
        sci = 0.0

    # 2) Directional centre and robust scale
    med_dLM = float(np.nanmedian(dLM))
    mad_dLM = _madv(dLM)
    if not np.isfinite(mad_dLM) or mad_dLM <= 0:
        mad_dLM = 1.0  # safety

    # 3) Distance sensitivity (mile & 7f get a touch more lift)
    D = float(D_actual_m)
    if   D <= 1100: dist_gain = 0.95
    elif D <= 1400: dist_gain = 1.05
    elif D <= 1800: dist_gain = 1.12
    elif D <= 2000: dist_gain = 1.05
    else:           dist_gain = 0.98

    # 4) Finish flavour adds gentle seasoning to magnitude only
    mad_dLG = _madv(dLG)
    fin_strength = 0.0 if mad_dLG == 0 else clamp(abs(np.nanmedian(dLG)) / max(mad_dLG, 1e-6), 0.0, 2.0)
    # mapped ~0..+0.6
    fin_bonus = 0.30 * fin_strength

    # 5) RSI raw → scaled to ≈ [-10, +10], with SCI gating
    # Base scale ~3.2 chosen to make |RSI|~6–8 for notably biased races.
    base_scale = 3.2
    rsi_signed = (med_dLM / mad_dLM) * base_scale
    rsi_signed *= (0.60 + 0.40 * sci)   # respect consensus
    rsi_signed *= dist_gain
    # flavour magnifies magnitude only
    rsi = np.sign(rsi_signed) * min(10.0, abs(rsi_signed) * (1.0 + fin_bonus))
    rsi = float(np.round(rsi, 2))

    # 6) Tag (human label) from RSI only
    if abs(rsi) < 1.2:
        shape_tag = "EVEN"
    elif rsi > 0:
        shape_tag = "SLOW_EARLY"
    else:
        shape_tag = "FAST_EARLY"

    # 7) RSI strength index (0..10) = |RSI|, capped
    rsi_strength = float(min(10.0, abs(rsi)))

    # 8) Per-horse exposure along the same axis (late-minus-mid)
    # Positive RS_Component = ran like late-favoured type; negative = early-favoured type
    w["RS_Component"] = (acc - mid).round(3)

    # Alignment cue: +1 with shape, -1 against shape, 0 neutral
    def _align_row(val, rsi_val, eps=0.25):
        if not (np.isfinite(val) and np.isfinite(rsi_val)) or abs(rsi_val) < 1.2:
            return 0
        if val > +eps and rsi_val > 0: return +1
        if val < -eps and rsi_val < 0: return +1
        if val > +eps and rsi_val < 0: return -1
        if val < -eps and rsi_val > 0: return -1
        return 0

    w["RSI_Align"] = [ _align_row(v, rsi) for v in w["RS_Component"] ]

    # Pretty cue for tables
    def _align_icon(a):
        if a > 0:  return "🔵 ➜ with shape"
        if a < 0:  return "🔴 ⇦ against shape"
        return "⚪ neutral"

    w["RSI_Cue"] = [ _align_icon(a) for a in w["RSI_Align"] ]

    # Save attrs you already expose / use elsewhere
    w.attrs["RSI"]         = float(rsi)
    w.attrs["RSI_STRENGTH"]= float(rsi_strength)
    w.attrs["SCI"]         = float(sci)
    w.attrs["SHAPE_TAG"]   = shape_tag
    # Informational finish flavour (kept from your previous UX)
    fin_flav = "Balanced Finish"
    med_dLG = float(np.nanmedian(dLG))
    gLG_gate = max(1.40, 0.50 * _madv(dLG))  # keep your eased threshold
    if   med_dLG >= +gLG_gate: fin_flav = "Attritional Finish"
    elif med_dLG <= -gLG_gate: fin_flav = "Sprint Finish"
    w.attrs["FINISH_FLAV"] = fin_flav
    return w, seg_markers
