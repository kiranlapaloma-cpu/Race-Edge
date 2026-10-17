from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _num(value: Any):
    try:
        if value is None or pd.isna(value):
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _round_mr(value: Any):
    v = _num(value)
    if v is None:
        return None
    return int(math.floor(v + 0.5))


def _fmt_mr(value: Any) -> str:
    v = _round_mr(value)
    return "-" if v is None else str(v)


def _fmt_edge(value: Any, decimals: int = 1) -> str:
    v = _num(value)
    if v is None:
        return "-"
    if decimals == 0:
        n = int(round(v))
        return f"+{n}" if n > 0 else str(n)
    return f"{v:+.{decimals}f}"


def evidence_label(runs: int) -> str:
    if runs >= 3:
        return "Established"
    if runs == 2:
        return "Emerging"
    if runs == 1:
        return "Limited"
    return "No evidence"


def pace_label(rpss: Any = None, race_test: Any = None) -> str | None:
    v = _num(rpss)
    if v is not None:
        if v >= 98.0:
            return "Fast"
        if v >= 94.0:
            return "Even"
        return "Slow"

    text = str(race_test or "").strip().lower()
    if "fast" in text:
        return "Fast"
    if "even" in text:
        return "Even"
    if "slow" in text:
        return "Slow"
    return None


def distance_band(distance: Any) -> str | None:
    d = _num(distance)
    if d is None:
        return None
    d = int(round(d))
    if d <= 1200:
        return "1000-1200m"
    if d <= 1400:
        return "1250-1400m"
    if d <= 1600:
        return "1450-1600m"
    if d <= 1800:
        return "1700-1800m"
    if d <= 2200:
        return "1900-2200m"
    return "2300m+"


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or len(history) == 0:
        return pd.DataFrame()

    h = history.copy()
    if "race_date" not in h.columns:
        h["race_date"] = pd.NaT
    h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce")

    for col in ["race_number", "distance", "rpss", "official_mr", "mr_achieved"]:
        if col not in h.columns:
            h[col] = np.nan
        h[col] = pd.to_numeric(h[col], errors="coerce")

    if "race_test" not in h.columns:
        h["race_test"] = ""
    if "track" not in h.columns:
        h["track"] = ""
    if "course" not in h.columns:
        h["course"] = ""

    h["MR Edge"] = h["mr_achieved"] - h["official_mr"]
    h["Pace"] = [pace_label(r, t) for r, t in zip(h["rpss"], h["race_test"])]
    h["Distance Band"] = h["distance"].map(distance_band)
    h["Track/Course"] = (
        h["track"].fillna("").astype(str).str.strip()
        + " | "
        + h["course"].fillna("").astype(str).str.strip()
    ).str.strip(" |")
    h.loc[h["Track/Course"].eq(""), "Track/Course"] = None

    return h.sort_values(
        ["race_date", "race_number"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _best_group(h: pd.DataFrame, group_col: str) -> dict | None:
    if h.empty or group_col not in h.columns:
        return None

    valid = h.loc[
        h[group_col].notna()
        & h["mr_achieved"].notna()
        & h["official_mr"].notna()
    ].copy()
    if valid.empty:
        return None

    grouped = (
        valid.groupby(group_col, dropna=True)
        .agg(
            runs=("mr_achieved", "size"),
            avg_mr=("mr_achieved", "mean"),
            avg_edge=("MR Edge", "mean"),
            peak_mr=("mr_achieved", "max"),
        )
        .reset_index()
    )
    if grouped.empty:
        return None

    eligible = grouped[grouped["runs"] >= 2].copy()
    pool = eligible if not eligible.empty else grouped
    pool = pool.sort_values(
        ["avg_edge", "runs", "peak_mr"],
        ascending=[False, False, False],
        na_position="last",
    )
    row = pool.iloc[0]
    runs = int(row["runs"])
    return {
        "label": str(row[group_col]),
        "runs": runs,
        "avg_mr": float(row["avg_mr"]),
        "avg_edge": float(row["avg_edge"]),
        "peak_mr": float(row["peak_mr"]),
        "evidence": evidence_label(runs),
    }


def build_performance_profile(
    history: pd.DataFrame,
    current_official_mr: Any = None,
) -> dict:
    h = _prepare_history(history)
    if h.empty:
        return {
            "runs": 0,
            "current_official_mr": _num(current_official_mr),
            "last_recorded_official_mr": None,
            "latest_mr_achieved": None,
            "highest_mr_achieved": None,
            "recent_best_mr": None,
            "recent_best_evidence": "No evidence",
            # Backward-compatible aliases for any older callers.
            "established_mr": None,
            "established_evidence": "No evidence",
            "average_mr_edge": None,
            "trend": "No evidence",
            "peak": None,
            "best_track_course": None,
            "best_distance": None,
            "best_pace": None,
            "history": h,
        }

    official = h["official_mr"].dropna()
    valid_mr = h.loc[h["mr_achieved"].notna(), ["race_date", "race_number", "mr_achieved"]]
    mr_series = h["mr_achieved"].dropna()
    edge_series = h["MR Edge"].dropna()

    # Recent Best = the strongest MR Achieved from up to the horse's
    # three most recent valid runs. A horse does not need three runs:
    # 1 run -> that run, 2 runs -> best of 2, 3+ runs -> best of latest 3.
    recent_values = mr_series.head(3)
    recent_best = float(recent_values.max()) if not recent_values.empty else None
    recent_best_evidence = evidence_label(len(recent_values))

    trend = "Limited evidence"
    trend_values = mr_series.head(4)
    if len(trend_values) >= 3:
        # History is newest first. Compare newest with the oldest value in the recent window.
        movement = float(trend_values.iloc[0] - trend_values.iloc[-1])
        if movement >= 3.0:
            trend = "Improving"
        elif movement <= -3.0:
            trend = "Declining"
        else:
            trend = "Stable"

    peak = None
    if not valid_mr.empty:
        highest = float(valid_mr["mr_achieved"].max())
        # h is newest first, so the first tied maximum is the most recent peak.
        peak_row = h.loc[h["mr_achieved"].eq(highest)].iloc[0]
        peak = {
            "date": peak_row.get("race_date"),
            "track": str(peak_row.get("track") or "").strip() or None,
            "course": str(peak_row.get("course") or "").strip() or None,
            "distance": _num(peak_row.get("distance")),
            "rpss": _num(peak_row.get("rpss")),
            "pace": peak_row.get("Pace"),
            "mr_achieved": highest,
            "official_mr_then": _num(peak_row.get("official_mr")),
            "mr_edge": _num(peak_row.get("MR Edge")),
        }

    return {
        "runs": int(len(h)),
        "current_official_mr": _num(current_official_mr),
        "last_recorded_official_mr": float(official.iloc[0]) if not official.empty else None,
        "latest_mr_achieved": float(mr_series.iloc[0]) if not mr_series.empty else None,
        "highest_mr_achieved": float(mr_series.max()) if not mr_series.empty else None,
        "recent_best_mr": recent_best,
        "recent_best_evidence": recent_best_evidence,
        # Backward-compatible aliases for any older callers.
        "established_mr": recent_best,
        "established_evidence": recent_best_evidence,
        "average_mr_edge": float(edge_series.mean()) if not edge_series.empty else None,
        "trend": trend,
        "peak": peak,
        "best_track_course": _best_group(h, "Track/Course"),
        "best_distance": _best_group(h, "Distance Band"),
        "best_pace": _best_group(h, "Pace"),
        "history": h,
    }


def _peak_context_line(peak: dict | None) -> str:
    if not peak:
        return "No valid MR Achieved performance is available yet."
    bits = []
    track = peak.get("track")
    course = peak.get("course")
    if track and course:
        bits.append(f"{track} | {course}")
    elif track or course:
        bits.append(str(track or course))
    d = _num(peak.get("distance"))
    if d is not None:
        bits.append(f"{int(round(d))}m")
    pace = peak.get("pace")
    rpss = _num(peak.get("rpss"))
    if pace and rpss is not None:
        bits.append(f"{pace} pace (RPSS {rpss:.1f})")
    elif pace:
        bits.append(f"{pace} pace")
    elif rpss is not None:
        bits.append(f"RPSS {rpss:.1f}")
    return " | ".join(bits) if bits else "Context unavailable"


def render_performance_profile(
    profile: dict,
    *,
    show_current_official: bool,
    compact: bool = False,
):
    import streamlit as st  # Only needed by the legacy UI renderer.
    st.markdown("#### Performance Profile")

    if profile.get("runs", 0) <= 0:
        st.caption("No saved Race Edge runs are available for a performance profile yet.")
        return

    if show_current_official:
        official_label = "Current Official MR"
        official_value = profile.get("current_official_mr")
    else:
        official_label = "Last Recorded Official MR"
        official_value = profile.get("last_recorded_official_mr")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(official_label, _fmt_mr(official_value))
    c2.metric(
        "Recent Best MR",
        _fmt_mr(profile.get("recent_best_mr", profile.get("established_mr"))),
    )
    c3.metric("Latest MR Achieved", _fmt_mr(profile.get("latest_mr_achieved")))
    c4.metric("Highest MR Achieved", _fmt_mr(profile.get("highest_mr_achieved")))

    c5, c6, c7 = st.columns(3)
    c5.metric("Average MR +/-", _fmt_edge(profile.get("average_mr_edge"), 1))
    c6.metric("Trend", str(profile.get("trend") or "-"))
    c7.metric(
        "Profile Evidence",
        str(profile.get("recent_best_evidence", profile.get("established_evidence")) or "-"),
    )

    peak = profile.get("peak")
    st.markdown("##### Peak Performance")
    if not peak:
        st.caption("No valid peak performance is available yet.")
    else:
        st.markdown(
            f"**MR {_fmt_mr(peak.get('mr_achieved'))}** | {_peak_context_line(peak)}"
        )
        details = [
            f"Official MR then: {_fmt_mr(peak.get('official_mr_then'))}",
            f"MR +/-: {_fmt_edge(peak.get('mr_edge'), 0)}",
        ]
        dt = peak.get("date")
        if pd.notna(dt):
            try:
                details.append(pd.Timestamp(dt).strftime("%Y-%m-%d"))
            except Exception:
                pass
        st.caption(" | ".join(details))

    st.markdown("##### Best Conditions")
    rows = []
    for label, key in [
        ("Track/Course", "best_track_course"),
        ("Distance", "best_distance"),
        ("Pace", "best_pace"),
    ]:
        item = profile.get(key)
        if item:
            rows.append({
                "Factor": label,
                "Best": item.get("label"),
                "Runs": item.get("runs"),
                "Avg MR +/-": item.get("avg_edge"),
                "Peak MR": item.get("peak_mr"),
                "Evidence": item.get("evidence"),
            })
        else:
            rows.append({
                "Factor": label,
                "Best": "Insufficient data",
                "Runs": 0,
                "Avg MR +/-": np.nan,
                "Peak MR": np.nan,
                "Evidence": "No evidence",
            })

    best = pd.DataFrame(rows)
    best["Runs"] = pd.to_numeric(best["Runs"], errors="coerce").astype("Int64")
    best["Avg MR +/-"] = pd.to_numeric(best["Avg MR +/-"], errors="coerce").round(1)
    best["Peak MR"] = pd.to_numeric(best["Peak MR"], errors="coerce").round().astype("Int64")
    st.dataframe(best, width="stretch", hide_index=True)

    if compact:
        st.caption(
            "Preferences are ranked by average historical MR +/- using the Official MR saved for each specific run."
        )
    else:
        st.caption(
            "Historical MR +/- always uses the Official MR saved for that exact run. "
            "Current race-card MR is never substituted into historical performances."
        )
