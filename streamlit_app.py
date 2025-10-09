# ======================= Batch A — Sidebar & Input + Weight UI =======================
with st.sidebar:
    st.markdown(f"### Race Edge v{APP_VERSION}")
    st.caption("PI v3.2 with Race Shape (SED/FRA/SCI), CG, Hidden v2, Ability v2, DB, and Weight Sensitivity")

    st.markdown("#### Upload race file")
    up = st.file_uploader(
        "Upload CSV/XLSX with 100 m or 200 m splits.\n"
        "Optional columns: `Horse Weight`, `Finish_Pos`, `Finish_Time`.",
        type=["csv","xlsx","xls"]
    )

    # Distance input (50 m steps, 800–4000 m)
    race_distance_input = st.number_input(
        "Race Distance (m)",
        min_value=800, max_value=4000, step=50, value=1600
    )

    st.markdown("#### Core toggles")
    USE_CG = st.toggle("Use Corrected Grind (CG)", value=True,
                       help="Adjust Grind for field collapse to credit finishers.")
    DAMPEN_CG = st.toggle("Dampen Grind weight if collapsed", value=True,
                          help="Shift a little weight Grind → Accel/tsSPI on collapse races.")
    USE_RACE_SHAPE = st.toggle("Use Race Shape module (SED/FRA/SCI)", value=True,
                               help="Detect slow-early / fast-early patterns and apply False-Run Adjustment.")
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

    st.markdown("---")
    st.markdown("#### Weight adjustment (BETA)")

    USE_WEIGHT = st.toggle("Enable weight adjustment engine", value=False,
                            help="If `Horse Weight` column missing, use this to simulate weight scenarios.")
    if USE_WEIGHT:
        BASE_WEIGHT = 60.0
        st.caption(f"Baseline = **{BASE_WEIGHT:.1f} kg** · Changes apply ± in **0.5 kg** steps")

        st.markdown("##### Scenario controls")
        colW1, colW2, colW3 = st.columns(3)
        with colW1:
            weight_delta = st.number_input("Δ weight (kg)", -10.0, +10.0, 0.0, 0.5)
        with colW2:
            st.write("") # layout spacer
            st.caption("Positive = heavier → slower")
        with colW3:
            st.write("") # layout spacer
            st.caption("Negative = lighter → faster")

        st.markdown(
            "You can later fine-tune per-horse weights in the main UI grid once data loads. "
            "When the CSV includes `Horse Weight`, this toggle has no effect unless you manually override."
        )
    else:
        BASE_WEIGHT, weight_delta = 60.0, 0.0

    st.markdown("---")
    st.markdown("#### Database path & init")
    db_path = st.text_input("Database path", value=DB_DEFAULT_PATH)
    init_btn = st.button("Initialise / Check DB")

# [WEIGHT UI END]
# ======================= Batch B — Weight model (helpers) =======================

def _weight_sensitivity_per_kg(distance_m: float) -> float:
    """
    Distance-sensitive time impact per kilogram (fraction per kg).
    Calibrated to be small for sprints, larger for routes; smooth & capped.

    Examples:
      800m  → ~0.08% / kg
      1000m → ~0.10% / kg
      1600m → ~0.135% / kg
      2000m → ~0.15% / kg
      2400m → ~0.165% / kg
      3200m → ~0.18% / kg
    """
    dm = float(distance_m or 1200)
    # base ramps linearly from 0.0008 at 800m to 0.0018 at 3200m
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
      1) If a column named 'Horse Weight' (case-insensitive matched) exists → use it (minus baseline).
      2) Else if USE_WEIGHT is True → use global scenario delta (ui_delta_kg) uniformly.
      3) Else → 0 for everyone (no effect).
    """
    # find 'Horse Weight' with tolerant casing / spacing
    candidates = [c for c in df.columns if c.strip().lower().replace("  "," ").replace("_"," ").replace("-", " ") == "horse weight"]
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
        # nothing to do
        out.attrs["WEIGHT_APPLIED"] = False
        out.attrs["WEIGHT_SENS_PER_KG"] = sens
        out.attrs["WEIGHT_BASELINE"] = baseline_kg
        out.attrs["WEIGHT_MODE"] = "none"
        return out

    # Build per-row multiplicative factor
    # factor = 1 + sens * kg_delta  (clip to reasonable bounds)
    factor = (1.0 + sens * kg_delta).clip(lower=0.90, upper=1.10)

    # Apply column-wise (vectorized)
    for c in time_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = (s * factor).where(s > 0, np.nan)

    # Annotate
    out.attrs["WEIGHT_APPLIED"] = bool(kg_delta.abs().sum() > 0)
    out.attrs["WEIGHT_SENS_PER_KG"] = sens
    out.attrs["WEIGHT_BASELINE"] = baseline_kg
    out.attrs["WEIGHT_MODE"] = ("file_column" if any(k != 0.0 for k in kg_delta.fillna(0.0)) and ("Horse Weight" in [c.title() for c in out.columns]) else
                                ("scenario" if use_weight else "none"))

    # Keep the effective deltas for later debugging / display
    out["_WeightΔ_kg"] = kg_delta
    return out
# ======================= /Batch B — helpers =======================
# ======================= Batch B — Apply weight & then compute =======================
# Get the UI settings from Batch A (already defined in your sidebar)
BASE_WEIGHT = 60.0
ui_delta = weight_delta if 'weight_delta' in globals() else 0.0
use_weight_flag = bool(USE_WEIGHT) if 'USE_WEIGHT' in globals() else False

# 1) Apply weight model to a copy of `work`
work_w = apply_weight_to_times(
    work,
    distance_m=float(race_distance_input),
    baseline_kg=BASE_WEIGHT,
    ui_delta_kg=float(ui_delta),
    use_weight=use_weight_flag
)

# 2) Compute metrics off the adjusted times
try:
    metrics, seg_markers = build_metrics_and_shape(
        work_w,                                 # <<< use weight-adjusted table
        float(race_distance_input),
        int(split_step),
        USE_CG,
        DAMPEN_CG,
        USE_RACE_SHAPE,
        DEBUG
    )
except Exception as e:
    st.error("Metric computation failed.")
    st.exception(e)
    st.stop()
# ======================= /Batch B =======================
