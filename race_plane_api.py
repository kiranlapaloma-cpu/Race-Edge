"""Race Edge 2.0 experimental Race Plane API.

Standalone router. Uses Normal Grind only. Does not persist data or alter
Core Metrics, PI, ratings, or the existing analysis endpoints.
Register in app.py with:
    from race_plane_api import race_plane_router
    app.include_router(race_plane_router)
"""
from __future__ import annotations

import math
import os
import secrets
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field


race_plane_router = APIRouter()


def _auth(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get('RACE_EDGE_API_KEY', '')
    if not expected:
        raise HTTPException(status_code=503, detail='API access is not configured')
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail='Unauthorized')


class RacePlaneRequest(BaseModel):
    metrics_rows: list[dict[str, Any]] = Field(min_length=4)
    distance_m: float = Field(gt=0, le=3600)
    centred: bool = True
    rpss: float | None = None
    pi_phase_weights: dict[str, float] | None = None


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None:
        return None
    return value


def _robust_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float)
    med = float(np.nanmedian(s)) if np.isfinite(s).any() else 0.0
    mad = float(np.nanmedian(np.abs(s - med))) if np.isfinite(s).any() else 0.0
    if np.isfinite(mad) and mad > 1e-12:
        out = (s - med) / (1.4826 * mad)
    else:
        sd = float(np.nanstd(s, ddof=0)) if np.isfinite(s).any() else 0.0
        mu = float(np.nanmean(s)) if np.isfinite(s).any() else 0.0
        out = (s - mu) / sd if np.isfinite(sd) and sd > 1e-12 else pd.Series(0.0, index=s.index)
    return pd.Series(out, index=s.index, dtype=float).clip(-3.5, 3.5)


def _sustain_profile(value: float) -> str:
    if value >= 3.0: return '🔥 Major above-plane'
    if value >= 1.5: return '🟢 Above expectation'
    if value > -1.5: return '⚪ Around expectation'
    if value > -3.0: return '🟠 Below expectation'
    return '🔴 Emptied / weak sustain'


@race_plane_router.post('/api/race-plane/calculate', dependencies=[Depends(_auth)])
def calculate_race_plane(payload: RacePlaneRequest) -> dict[str, Any]:
    frame = pd.DataFrame(payload.metrics_rows)
    required = ['Horse', 'tsSPI', 'Accel', 'Grind']
    if any(c not in frame.columns for c in required):
        raise HTTPException(status_code=422, detail='Race Plane requires Horse, tsSPI, Accel and Normal Grind.')
    extra = [c for c in ['PI', 'Finish_Pos', 'TOF', 'SRI', 'Peak_Location'] if c in frame.columns]
    df = frame[required + extra].copy()
    for c in ['tsSPI', 'Accel', 'Grind', 'PI', 'Finish_Pos', 'TOF', 'SRI']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['tsSPI', 'Accel', 'Grind']).reset_index(drop=True)
    if len(df) < 4:
        raise HTTPException(status_code=422, detail='At least four complete runners are needed to fit Race Plane.')

    offset = 100.0 if payload.centred else 0.0
    x = df['tsSPI'].to_numpy(dtype=float) - offset
    y = df['Accel'].to_numpy(dtype=float) - offset
    z = df['Grind'].to_numpy(dtype=float) - offset
    matrix = np.column_stack([np.ones(len(df)), x, y])
    coef, _, rank, _ = np.linalg.lstsq(matrix, z, rcond=None)
    intercept, travel_coef, accel_coef = map(float, coef)
    expected = matrix @ coef
    df['Expected_Grind'] = expected + offset
    df['Sustain_Residual'] = df['Grind'] - df['Expected_Grind']
    ss_res = float(np.nansum((z - expected) ** 2))
    ss_tot = float(np.nansum((z - float(np.nanmean(z))) ** 2))
    r2 = float('nan') if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    # Exact PPS robust scaling and transform from the original Race Plane.
    for name in ['tsSPI', 'Accel', 'Grind']:
        df[f'PPS_z_{name}'] = _robust_z(df[name])
    supplied = payload.pi_phase_weights or {}
    default_accel = 1.0 + 200.0 / max(payload.distance_m, 1.0)
    raw_weights = {
        'tsSPI': float(supplied.get('tsSPI', 1.0)),
        'Accel': float(supplied.get('Accel', default_accel)),
        'Grind': float(supplied.get('Grind', 1.0)),
    }
    if not all(math.isfinite(v) for v in raw_weights.values()):
        raise HTTPException(status_code=422, detail='Invalid phase weights.')
    weight_sum = sum(max(0.0, v) for v in raw_weights.values()) or 1.0
    weights = {k: max(0.0, v) / weight_sum for k, v in raw_weights.items()}
    df['PPS_Core'] = sum(weights[k] * df[f'PPS_z_{k}'] for k in weights)
    df['PPS'] = np.clip(5.0 + 2.75 * np.tanh(df['PPS_Core'] / 1.35), 0.0, 10.0).round(2)
    df['PPS_Rank'] = df['PPS'].rank(method='min', ascending=False).astype(int)

    # Original module rounds residuals before determining architecture cutoffs.
    df['Expected_Grind'] = df['Expected_Grind'].round(2)
    df['Sustain_Residual'] = df['Sustain_Residual'].round(2)
    df['Sustain_Profile'] = df['Sustain_Residual'].map(_sustain_profile)
    pps_hi = float(np.nanquantile(df['PPS'], 0.67))
    pps_lo = float(np.nanquantile(df['PPS'], 0.33))
    sr_spread = float(np.nanstd(df['Sustain_Residual'], ddof=1)) if len(df) > 1 else 0.0
    sr_cut = max(1.0, 0.65 * sr_spread) if np.isfinite(sr_spread) else 1.0

    def architecture(row: pd.Series) -> str:
        pps, sr = float(row['PPS']), float(row['Sustain_Residual'])
        if not (np.isfinite(pps) and np.isfinite(sr)): return 'Unclear profile'
        if pps >= pps_hi and sr >= sr_cut: return 'Complete performance'
        if pps >= pps_hi and sr <= -sr_cut: return 'Strong but incomplete'
        if pps >= pps_hi: return 'Strong balanced run'
        if pps <= pps_lo and sr >= sr_cut: return 'Honest sustain, limited level'
        if pps <= pps_lo and sr <= -sr_cut: return 'Below benchmark'
        if sr >= sr_cut: return 'Hidden finishing strength'
        if sr <= -sr_cut: return 'Acceleration-led, weak sustain'
        return 'Balanced / expected'

    df['Performance_Architecture'] = df.apply(architecture, axis=1)

    # Race Test Profile: original performance.py always fits centred values.
    cx = df['tsSPI'].to_numpy(dtype=float) - 100.0
    cy = df['Accel'].to_numpy(dtype=float) - 100.0
    cz = df['Grind'].to_numpy(dtype=float) - 100.0
    test_matrix = np.column_stack([np.ones(len(df)), cx, cy])
    test_coef, _, test_rank, _ = np.linalg.lstsq(test_matrix, cz, rcond=None)
    test_expected = test_matrix @ test_coef
    test_tot = float(np.nansum((cz - float(np.nanmean(cz))) ** 2))
    test_r2 = float('nan') if test_tot <= 1e-12 else 1.0 - float(np.nansum((cz-test_expected)**2)) / test_tot
    b, c = float(test_coef[1]), float(test_coef[2])
    pos_travel, pos_accel = max(b, 0.0), max(c, 0.0)
    positive_total = pos_travel + pos_accel
    travel_share = pos_travel / positive_total if positive_total > 1e-12 else float('nan')
    accel_share = pos_accel / positive_total if positive_total > 1e-12 else float('nan')
    rpss = payload.rpss
    tempo = ('Slow' if rpss < 94 else 'Even' if rpss < 98 else 'Fast') if rpss is not None and math.isfinite(rpss) else 'Unknown'
    if not np.isfinite(test_r2) or test_r2 < .30 or test_rank < 3 or positive_total <= 1e-12:
        mode = 'Inconclusive'
    elif accel_share >= .65:
        mode = 'Acceleration'
    elif travel_share >= .65:
        mode = 'Sustained Pressure'
    else:
        mode = 'Balanced'
    if not np.isfinite(test_r2) or test_r2 < .30 or test_rank < 3:
        confidence = 'Low'
    elif test_r2 < .50:
        confidence = 'Tentative'
    elif test_r2 < .70:
        confidence = 'Meaningful'
    else:
        confidence = 'Strong'
    if mode == 'Inconclusive':
        label = 'Inconclusive race test'
        summary = ('The fitted plane did not explain enough of the field to identify one dependable race requirement. '
                   'Horse-level residuals can still be useful, but the slope should not drive a strong race-shape conclusion.')
    elif mode == 'Acceleration':
        label = 'Tactical sprint test' if tempo == 'Slow' else 'Acceleration-led test'
        summary = (f'This {tempo.lower() if tempo != "Unknown" else "race"} contest linked finishing strength more closely to acceleration than sustained travelling speed. '
                   'A decisive change of speed was the clearest positive requirement produced by the field.')
    elif mode == 'Sustained Pressure':
        label = 'Sustained-pressure test'
        summary = (f'This {tempo.lower() if tempo != "Unknown" else "race"} contest linked finishing strength most strongly to sustained travelling speed. '
                   'Horses able to carry pressure before the finish were positively rewarded.')
    else:
        label = 'Balanced all-round test'
        summary = (f'This {tempo.lower() if tempo != "Unknown" else "race"} contest required a combination of sustained travelling speed and acceleration. '
                   'No single earlier phase dominated the positive relationship with finishing strength.')

    return _clean({
        'experimental': True, 'saved': False, 'grind_target': 'Grind',
        'centred': payload.centred, 'runners_used': len(df),
        'formula': {'intercept': intercept, 'tsSPI_coefficient': travel_coef, 'Accel_coefficient': accel_coef, 'offset': offset, 'rank': int(rank)},
        'r2': r2, 'residual_std': sr_spread,
        'pps_weights': weights,
        'race_test_profile': {
            'valid': True, 'label': label, 'mode': mode, 'confidence': confidence,
            'summary': summary, 'r2': test_r2, 'intercept': float(test_coef[0]),
            'travel_coef': b, 'accel_coef': c, 'travel_reward_share': travel_share,
            'accel_reward_share': accel_share, 'rpss': rpss, 'tempo': tempo,
            'runners': len(df), 'rank': int(test_rank),
        },
        'rows': df.sort_values(['PPS', 'PI'] if 'PI' in df else ['PPS'], ascending=False).to_dict(orient='records'),
    })
