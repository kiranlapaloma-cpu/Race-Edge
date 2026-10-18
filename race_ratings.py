"""Read-only Race Edge ratings engine. Original handicap conversion retained verbatim."""
import math
from datetime import date
import numpy as np
import pandas as pd
from wfa import get_wfa_lb

def canon_horse(value):
    import re
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()

def _horse_metadata_frame(df):
    out = pd.DataFrame({"Horse": df["Horse"].astype(str)})
    aliases = {"Age": ["Age", "A"], "Horse Weight": ["Horse Weight", "Horse_Weight", "Wt", "Weight", "Weight (kg)", "Wgh"], "Official MR": ["Official MR", "Official_MR", "MR", "Merit Rating"]}
    cmap = {str(c).strip().lower(): c for c in df.columns}
    for target, candidates in aliases.items():
        source = next((cmap.get(c.lower()) for c in candidates if cmap.get(c.lower()) is not None), None)
        out[target] = df[source] if source is not None else pd.NA
    out["Horse Key"] = out["Horse"].map(canon_horse)
    return out

def _db_round_mr(value):
    if value is None or not np.isfinite(float(value)):
        return None
    return int(math.floor(float(value) + 0.5))

def build_database_handicap(metrics_df: pd.DataFrame, distance_m: float, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create the line-horse rating frame using the existing PI conversion and 1 kg = 2 MR points."""
    if metrics_df is None or "Horse" not in metrics_df.columns or "PI" not in metrics_df.columns:
        return pd.DataFrame()

    weight_candidates = ["Horse Weight", "Horse_Weight", "Wt", "Weight", "Weight (kg)"]
    weight_col = next((c for c in weight_candidates if c in metrics_df.columns), None)

    out = metrics_df[["Horse", "PI"]].copy()
    if weight_col is not None:
        out["Weight (kg)"] = pd.to_numeric(metrics_df[weight_col], errors="coerce")
    elif source_df is not None and isinstance(source_df, pd.DataFrame) and "Horse" in source_df.columns:
        _source_meta = _horse_metadata_frame(source_df)
        _weights = _source_meta[["Horse Key", "Horse Weight"]].drop_duplicates("Horse Key")
        out["Horse Key"] = out["Horse"].map(canon_horse)
        out = out.merge(_weights, on="Horse Key", how="left")
        out["Weight (kg)"] = pd.to_numeric(out["Horse Weight"], errors="coerce")
        out = out.drop(columns=[c for c in ["Horse Key", "Horse Weight"] if c in out.columns])
    else:
        out["Weight (kg)"] = 60.0
    out["Weight (kg)"] = pd.to_numeric(out["Weight (kg)"], errors="coerce").fillna(60.0)
    out["PI"] = pd.to_numeric(out["PI"], errors="coerce")
    out = out.dropna(subset=["Horse", "PI"]).reset_index(drop=True)
    if out.empty:
        return out

    d = float(distance_m)
    if d <= 1200:
        beta0 = 0.30
    elif d <= 1600:
        beta0 = 0.35
    elif d <= 2000:
        beta0 = 0.40
    elif d <= 2400:
        beta0 = 0.45
    else:
        beta0 = 0.50

    corr_df = out[["Weight (kg)", "PI"]].dropna()
    corr = corr_df["Weight (kg)"].corr(corr_df["PI"]) if len(corr_df) >= 6 else np.nan
    field_size = int(out["PI"].notna().sum())
    tiny_dampen = 0.0 if field_size < 6 else min(1.0, (field_size - 5) / 7.0)
    corr_mag = 0.0 if not np.isfinite(corr) else abs(float(corr))
    beta_eff = beta0 * (1.0 + 0.40 * corr_mag * tiny_dampen)

    rsi = float(metrics_df.attrs.get("RSI", np.nan))
    sci = float(metrics_df.attrs.get("SCI", np.nan))
    if np.isfinite(rsi) and np.isfinite(sci) and sci >= 0.5:
        if rsi < -0.6:
            beta_eff *= 1.10
        elif rsi > 0.6:
            beta_eff *= 0.90
    beta_eff = float(np.clip(beta_eff, 0.22, 0.70))

    conversion_factor = {
        7: 0.68, 8: 0.76, 9: 0.84, 10: 0.90, 11: 0.95,
    }.get(field_size, 1.00 if field_size >= 12 else 0.60)

    pi_median = float(np.nanmedian(out["PI"]))
    out["Performance MR"] = ((out["PI"] - pi_median) / beta_eff) * conversion_factor * 2.0
    out["Horse"] = out["Horse"].astype(str)
    return out[["Horse", "Weight (kg)", "PI", "Performance MR"]]

def calculate_handicap_ratings(metrics_rows, source_rows, distance_m, race_date, line_horse=None, line_mr=None, missing_ages=None, diagnostics=None):
    metrics = pd.DataFrame(metrics_rows)
    source = pd.DataFrame(source_rows)
    if "Horse" not in source or "Horse" not in metrics:
        raise ValueError("Horse column is required in source and metrics")
    if not isinstance(race_date, date):
        race_date = date.fromisoformat(str(race_date))
    metrics.attrs.update(diagnostics or {})
    handicap = build_database_handicap(metrics, distance_m, source)
    if handicap.empty:
        raise ValueError("Unable to calculate Performance MR: PI data is missing")
    meta = _horse_metadata_frame(source).drop_duplicates("Horse Key")
    meta = meta.set_index("Horse Key")
    overrides = {canon_horse(k):v for k,v in (missing_ages or {}).items()}
    rows = []
    absent = []
    for _,r in handicap.iterrows():
        name = str(r["Horse"])
        key = canon_horse(name)
        source_age = pd.to_numeric(meta.at[key,"Age"],errors="coerce") if key in meta.index else np.nan
        if pd.notna(source_age):
            if key in overrides:
                raise ValueError(f"Age for {name} is supplied in the CSV and cannot be edited")
            age = int(source_age)
            age_origin = "CSV"
        elif key in overrides:
            age = int(overrides[key])
            age_origin = "manual_missing_only"
        else:
            age = None
            age_origin = "missing"
            absent.append(name)
        if age is not None and not 2 <= age <= 15:
            raise ValueError(f"Age for {name} must be between 2 and 15")
        mr = pd.to_numeric(meta.at[key,"Official MR"],errors="coerce") if key in meta.index else np.nan
        weight = float(r["Weight (kg)"])
        wfa_lb = get_wfa_lb(race_date,distance_m,age) if age is not None else None
        rows.append({"Horse":name,"PI":float(r["PI"]),"Performance MR":float(r["Performance MR"]),"Age":age,"Age Source":age_origin,"Official MR":None if pd.isna(mr) else _db_round_mr(mr),"Weight (kg)":weight,"WFA (lb)":wfa_lb,"WFA (kg)":None if wfa_lb is None else wfa_lb*.5,"Effective Weight":None if wfa_lb is None else weight+wfa_lb*.5})
    suggested = None
    if "Sustain_Residual" in metrics:
        residuals = {canon_horse(r["Horse"]):pd.to_numeric(r["Sustain_Residual"],errors="coerce") for _,r in metrics.iterrows()}
        candidates = [r for r in rows if 4.9 <= r["PI"] <= 5.9 and pd.notna(residuals.get(canon_horse(r["Horse"]),np.nan))]
        if candidates:
            suggested = min(candidates,key=lambda r:(abs(residuals[canon_horse(r["Horse"])]),abs(r["PI"]-5.0)))["Horse"]
    selected = line_horse or suggested or rows[0]["Horse"]
    anchor = next((r for r in rows if canon_horse(r["Horse"])==canon_horse(selected)),None)
    if anchor is None:
        raise ValueError("Selected line horse was not found")
    assigned = int(line_mr) if line_mr is not None else (anchor["Official MR"] if anchor["Official MR"] is not None else 100)
    if not 0 <= assigned <= 200:
        raise ValueError("Line MR must be between 0 and 200")
    if not absent:
        for r in rows:
            diff = r["Performance MR"] - anchor["Performance MR"]
            adjustment = 2.0*(r["Effective Weight"]-anchor["Effective Weight"])
            raw = assigned+diff+adjustment
            r.update({"Performance Difference":diff,"Weight + WFA Adjustment":adjustment,"MR Achieved Raw":raw,"MR Achieved":_db_round_mr(raw),"MR Edge":None if r["Official MR"] is None else _db_round_mr(raw)-r["Official MR"]})
    return {"rows":rows,"suggested_line_horse":suggested,"line_horse":anchor["Horse"],"line_mr":assigned,"missing_ages":absent,"ready":not absent,"saved":False}
