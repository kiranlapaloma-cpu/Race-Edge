"""Original Race Edge RPSS calculation, extracted from performance.py.
No Streamlit or plotting dependency.
"""
import numpy as np
import pandas as pd

DISTANCE_STD_100M = {
    1000: 5.13,
    1160: 5.28,
    1200: 5.35,
    1400: 5.65,
    1450: 5.75,
    1600: 5.90,
    1750: 5.92,
    1800: 5.93,
    1900: 5.94,
    1950: 5.95,
    2000: 5.95,
    2200: 6.02,
    2400: 6.10,
    2450: 6.12,
    2600: 6.20,
    2800: 6.30,
    3000: 6.40,
    3200: 6.50,
}

def _make_range_cols(D, start_incl, end_incl, step):
    """Return sectional time-column names from start marker down to end marker."""
    if start_incl < end_incl:
        return []
    want = list(range(int(start_incl), int(end_incl) - 1, -int(step)))
    return [f"{m}_Time" for m in want]

def _adaptive_tssp_start(D, step, markers):
    """Return the first sectional marker after the opening block.

    Mirrors the metrics engine logic, including odd-distance 200 m races such as
    1160 m where the actual uploaded markers must be used.
    """
    D = float(D)
    step = int(step)
    if step == 100:
        # F150 ends at D-150; the next 100m split is labelled D-250.
        return int(D - (250 if int(D) % 100 == 50 else 300))
    if not markers:
        return int(D - 400)
    ordered = sorted({int(m) for m in markers}, reverse=True)
    if len(ordered) >= 2:
        return ordered[1]
    return ordered[0]

def winsorize(s, p_lo=0.10, p_hi=0.90):
    try:
        lo = s.quantile(p_lo); hi = s.quantile(p_hi)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s

def _benchmark_std_100m(distance_m: float) -> float:
    """Practical trip benchmark ladder anchored at 5.50s per 100m for 1000m races."""
    knots = sorted((float(k), float(v)) for k, v in DISTANCE_STD_100M.items())
    d = float(distance_m)
    if d <= knots[0][0]:
        return float(knots[0][1])
    if d >= knots[-1][0]:
        return float(knots[-1][1])
    for (a, va), (b, vb) in zip(knots, knots[1:]):
        if a <= d <= b:
            if a == b:
                return float(vb)
            t = (d - a) / (b - a)
            return float(va + (vb - va) * t)
    return 5.70

def _benchmark_split_time(distance_m: float, split_step: int) -> float:
    std100 = _benchmark_std_100m(distance_m)
    step = int(split_step)
    if step == 100:
        return float(std100)
    if step == 200:
        return float(std100 * 2.0)
    return float(std100 * (step / 100.0))

def _robust_center(series: pd.Series, p_lo: float = 0.10, p_hi: float = 0.90) -> float:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return np.nan
    try:
        return float(winsorize(s, p_lo=p_lo, p_hi=p_hi).mean())
    except Exception:
        return float(s.median())

def _phase_avg_split(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    vals = df[cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan) if cols else pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_avg_split_time"] = vals.mean(axis=1, skipna=True) if cols else np.nan
    out[f"{prefix}_valid_splits"] = vals.notna().sum(axis=1) if cols else 0
    return out

def compute_rpss(metrics_df: pd.DataFrame, distance_m: float, split_step: int, seg_markers: list[int]):
    """
    RPSS = Race Pace Strength Score.
    Uses the tsSPI section only for the headline score, benchmarked to a distance standard.
    Also reports race-level Acceleration and Grind phase split averages to describe shape change.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return None
    step = int(split_step)
    D = float(distance_m)
    tssp_start = _adaptive_tssp_start(D, step, seg_markers)
    mid_cols = [c for c in _make_range_cols(D, tssp_start, 600, step) if c in metrics_df.columns]
    if not mid_cols:
        return None

    # Pace-shape companion windows using the user's segment-time definitions.
    # 200m files: 400_Time = 600→400, 200_Time = 400→200, Finish_Time = 200→FIN
    # 100m files: 400_Time = 500→400, 300_Time = 400→300, 200_Time = 300→200,
    #             100_Time = 200→100, Finish_Time = 100→FIN
    if step == 100:
        accel_cols = [c for c in ["500_Time", "400_Time", "300_Time", "200_Time"] if c in metrics_df.columns]
        grind_cols = [c for c in ["100_Time", "Finish_Time"] if c in metrics_df.columns]
    else:
        accel_cols = [c for c in ["400_Time", "200_Time"] if c in metrics_df.columns]
        grind_cols = [c for c in ["Finish_Time"] if c in metrics_df.columns]

    df = metrics_df.copy()
    mid_stats = _phase_avg_split(df, mid_cols, "_RPSS")
    acc_stats = _phase_avg_split(df, accel_cols, "_ACC")
    grd_stats = _phase_avg_split(df, grind_cols, "_GRD")
    for part in [mid_stats, acc_stats, grd_stats]:
        for c in part.columns:
            df[c] = part[c]

    usable = df[(pd.to_numeric(df["_RPSS_avg_split_time"], errors="coerce").notna()) & (df["_RPSS_valid_splits"] > 0)].copy()
    if usable.empty:
        return None

    benchmark_std_100m = _benchmark_std_100m(D)
    benchmark_split_time = _benchmark_split_time(D, step)

    usable["_RPSS_vs_std_pct"] = 100.0 * (benchmark_split_time / pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce"))
    usable["_RPSS_margin"] = benchmark_split_time - pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce")

    finish_col = "Finish_Pos" if "Finish_Pos" in usable.columns else None
    if finish_col:
        usable["_FinishSort"] = pd.to_numeric(usable[finish_col], errors="coerce").fillna(1e9)
        usable = usable.sort_values(["_FinishSort", "Horse" if "Horse" in usable.columns else usable.columns[0]])
    top_n = max(1, int(np.ceil(len(usable) / 2.0)))
    top_half = usable.head(top_n).copy() if finish_col else usable.nsmallest(top_n, columns="_RPSS_avg_split_time")

    race_avg = _robust_center(usable["_RPSS_avg_split_time"])
    top_half_avg = _robust_center(top_half["_RPSS_avg_split_time"])
    rpss = float(100.0 * (benchmark_split_time / race_avg)) if np.isfinite(race_avg) and race_avg > 0 else np.nan
    top_half_rpss = float(100.0 * (benchmark_split_time / top_half_avg)) if np.isfinite(top_half_avg) and top_half_avg > 0 else np.nan
    beaters_pct = float((pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce") <= benchmark_split_time).mean() * 100.0)
    top_half_beaters_pct = float((pd.to_numeric(top_half["_RPSS_avg_split_time"], errors="coerce") <= benchmark_split_time).mean() * 100.0)

    accel_avg = _robust_center(usable["_ACC_avg_split_time"])
    grind_avg = _robust_center(usable["_GRD_avg_split_time"])
    accel_vs_std = float(100.0 * (benchmark_split_time / accel_avg)) if np.isfinite(accel_avg) and accel_avg > 0 else np.nan
    grind_vs_std = float(100.0 * (benchmark_split_time / grind_avg)) if np.isfinite(grind_avg) and grind_avg > 0 else np.nan
    accel_lift_pct = float((race_avg / accel_avg) * 100.0) if np.isfinite(race_avg) and np.isfinite(accel_avg) and accel_avg > 0 else np.nan
    grind_hold_pct = float((accel_avg / grind_avg) * 100.0) if np.isfinite(accel_avg) and np.isfinite(grind_avg) and grind_avg > 0 else np.nan

    if not np.isfinite(rpss):
        verdict = "Unavailable"
    elif rpss < 97.0:
        verdict = "Weak sustained pace"
    elif rpss < 98.5:
        verdict = "Fair sustained pace"
    elif rpss < 99.5:
        verdict = "Slightly below strong standard"
    elif rpss <= 100.5:
        verdict = "Genuine / strong"
    elif rpss <= 102.0:
        verdict = "Very strong"
    else:
        verdict = "Exceptional"

    out_cols = [c for c in ["Horse", "Finish_Pos"] if c in usable.columns]
    out = usable[out_cols + ["_RPSS_avg_split_time", "_RPSS_vs_std_pct", "_RPSS_margin", "_RPSS_valid_splits"]].copy()
    out = out.rename(columns={
        "_RPSS_avg_split_time": "Avg tsSPI split time",
        "_RPSS_vs_std_pct": "tsSPI vs std (%)",
        "_RPSS_margin": "Margin vs std (s)",
        "_RPSS_valid_splits": "tsSPI splits used",
    })

    winner_row = None
    if "Finish_Pos" in usable.columns:
        _fp = pd.to_numeric(usable["Finish_Pos"], errors="coerce")
        _w = usable.loc[_fp == _fp.min()] if _fp.notna().any() else usable.iloc[:1]
        if len(_w):
            winner_row = _w.iloc[0]
    elif len(usable):
        winner_row = usable.iloc[0]

    def _phase_best_name_and_vals(df, avg_col):
        if avg_col not in df.columns or "Horse" not in df.columns:
            return None, np.nan, np.nan
        _vals = pd.to_numeric(df[avg_col], errors="coerce")
        _tmp = df.loc[_vals.notna()].copy()
        if _tmp.empty:
            return None, np.nan, np.nan
        _tmp[avg_col] = pd.to_numeric(_tmp[avg_col], errors="coerce")
        _best = _tmp.loc[_tmp[avg_col].idxmin()]
        _best_avg = float(_best[avg_col])
        _best_vs = float(100.0 * (benchmark_split_time / _best_avg)) if np.isfinite(_best_avg) and _best_avg > 0 else np.nan
        return _best.get("Horse"), _best_avg, _best_vs

    win_tsspi = float(winner_row["_RPSS_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_RPSS_avg_split_time")) else np.nan
    win_tsspi_vs = float(100.0 * (benchmark_split_time / win_tsspi)) if np.isfinite(win_tsspi) and win_tsspi > 0 else np.nan
    win_accel = float(winner_row["_ACC_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_ACC_avg_split_time")) else np.nan
    win_accel_vs = float(100.0 * (benchmark_split_time / win_accel)) if np.isfinite(win_accel) and win_accel > 0 else np.nan
    win_grind = float(winner_row["_GRD_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_GRD_avg_split_time")) else np.nan
    win_grind_vs = float(100.0 * (benchmark_split_time / win_grind)) if np.isfinite(win_grind) and win_grind > 0 else np.nan
    winner_name = winner_row.get("Horse") if winner_row is not None and "Horse" in winner_row.index else None

    best_tsspi_name, best_tsspi, best_tsspi_vs = _phase_best_name_and_vals(usable, "_RPSS_avg_split_time")
    best_acc_name, best_acc, best_acc_vs = _phase_best_name_and_vals(usable, "_ACC_avg_split_time")
    best_grd_name, best_grd, best_grd_vs = _phase_best_name_and_vals(usable, "_GRD_avg_split_time")

    phase_table = pd.DataFrame([
        {
            "Phase": "tsSPI",
            "Avg split (s)": race_avg,
            "Vs std (%)": rpss,
            "Change vs prior (%)": np.nan,
            "Winner": winner_name,
            "Winner split (s)": win_tsspi,
            "Winner vs std (%)": win_tsspi_vs,
            "Best horse": best_tsspi_name,
            "Best split (s)": best_tsspi,
            "Best vs std (%)": best_tsspi_vs,
        },
        {
            "Phase": "Acceleration",
            "Avg split (s)": accel_avg,
            "Vs std (%)": accel_vs_std,
            "Change vs prior (%)": accel_lift_pct,
            "Winner": winner_name,
            "Winner split (s)": win_accel,
            "Winner vs std (%)": win_accel_vs,
            "Best horse": best_acc_name,
            "Best split (s)": best_acc,
            "Best vs std (%)": best_acc_vs,
        },
        {
            "Phase": "Grind",
            "Avg split (s)": grind_avg,
            "Vs std (%)": grind_vs_std,
            "Change vs prior (%)": grind_hold_pct,
            "Winner": winner_name,
            "Winner split (s)": win_grind,
            "Winner vs std (%)": win_grind_vs,
            "Best horse": best_grd_name,
            "Best split (s)": best_grd,
            "Best vs std (%)": best_grd_vs,
        },
    ])

    return {
        "distance_m": D,
        "split_step": step,
        "mid_cols": mid_cols,
        "accel_cols": accel_cols,
        "grind_cols": grind_cols,
        "benchmark_std_100m": benchmark_std_100m,
        "benchmark_split_time": benchmark_split_time,
        "race_avg_split_time": race_avg,
        "top_half_avg_split_time": top_half_avg,
        "accel_avg_split_time": accel_avg,
        "grind_avg_split_time": grind_avg,
        "accel_vs_std": accel_vs_std,
        "grind_vs_std": grind_vs_std,
        "accel_lift_pct": accel_lift_pct,
        "grind_hold_pct": grind_hold_pct,
        "rpss": rpss,
        "top_half_rpss": top_half_rpss,
        "beaters_pct": beaters_pct,
        "top_half_beaters_pct": top_half_beaters_pct,
        "verdict": verdict,
        "phase_table": phase_table,
        "table": out.reset_index(drop=True),
    }

