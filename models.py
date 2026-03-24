“””
models.py - analytical sub-modules for Race Edge.

All algorithms are UNCHANGED.  Only structural changes:

- each module is a standalone function
- shared helpers imported from utils.py (no redefinitions)
- no Streamlit imports - purely computational
  “””
  import math
  import re
  import numpy as np
  import pandas as pd

from utils import as_num, clamp, mad_std, winsorize, lerp, norm_str
from metrics import pi_weights_distance_and_context, speed_to_idx

# ______________________________________________

# Ahead of the Handicap

# ______________________________________________

def ahead_of_handicap(
metrics: pd.DataFrame,
distance_m: float,
weight_col: str = “Horse Weight”,
) -> pd.DataFrame | None:
“”“Returns table or None if insufficient data.”””
RSI = float(metrics.attrs.get(“RSI”, np.nan))
SCI = float(metrics.attrs.get(“SCI”, np.nan))

```
def _field_corr(x, y):
    df = pd.DataFrame({"x": as_num(x), "y": as_num(y)}).dropna()
    if len(df) < 6:
        return np.nan
    c = df["x"].corr(df["y"])
    return float(c) if np.isfinite(c) else np.nan

def _beta_base(d):
    d = float(d)
    if d <= 1200: return 0.30
    if d <= 1600: return 0.35
    if d <= 2000: return 0.40
    if d <= 2400: return 0.45
    return 0.50

def _shape_adjust(beta, rsi, sci):
    if np.isfinite(rsi) and np.isfinite(sci) and sci >= 0.5:
        if rsi < -0.6:  beta *= 1.10
        elif rsi > 0.6: beta *= 0.90
    return beta

cols_needed = {"Horse", "PI", weight_col}
if not cols_needed.issubset(metrics.columns):
    return None

df = metrics[["Horse", "PI", weight_col]].copy()
df["PI"]       = as_num(df["PI"])
df[weight_col] = as_num(df[weight_col])
df = df.dropna(subset=["PI", weight_col])
if df.empty:
    return None

beta0     = _beta_base(distance_m)
corr      = _field_corr(df[weight_col], df["PI"])
n         = int(df["PI"].notna().sum())
tiny_damp = 0.0 if n < 6 else min(1.0, (n - 5) / 7.0)
corr_mag  = 0.0 if not np.isfinite(corr) else abs(corr)
beta_eff  = float(np.clip(
    _shape_adjust(beta0 * (1.0 + 0.40 * corr_mag * tiny_damp), RSI, SCI),
    0.22, 0.70,
))

PI_med = float(np.nanmedian(df["PI"]))
df["ΔPI_vs_med"]  = df["PI"] - PI_med
df["RanAbove_kg"] = df["ΔPI_vs_med"] / beta_eff
df["RanAbove_MR"] = df["RanAbove_kg"] * 2
df["beta_eff (PI/kg)"] = beta_eff

view = df.rename(columns={weight_col: "Wt (kg)"})
view = view[["Horse", "Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg",
             "RanAbove_MR", "beta_eff (PI/kg)"]].sort_values(
    "RanAbove_kg", ascending=False
)
for c in ["Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg", "RanAbove_MR", "beta_eff (PI/kg)"]:
    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)

view.attrs["PI_med"]  = PI_med
view.attrs["corr"]    = corr
view.attrs["beta_eff"] = beta_eff
return view
```

# ______________________________________________

# Winning DNA Matrix

# ______________________________________________

def winning_dna(metrics: pd.DataFrame, distance_m: float) -> pd.DataFrame:
“”“Returns enriched df with WinningDNA, EZ01_LL01, SOS01, DNA_TopTraits, DNA_Summary.”””
WD     = metrics.copy()
gr_col = metrics.attrs.get(“GR_COL”, “Grind”)
RSI    = float(metrics.attrs.get(“RSI”, 0.0))
SCI    = float(metrics.attrs.get(“SCI”, 0.0))
D_m    = float(distance_m)

```
for c in ["Horse", "F200_idx", "tsSPI", "Accel", gr_col]:
    if c not in WD.columns:
        WD[c] = np.nan

def _band_knots(metric):
    if metric == "EZ":
        return [(1000,(96.0,106.0)),(1100,(96.5,105.5)),(1200,(97.0,105.0)),
                (1400,(98.0,104.0)),(1600,(98.5,103.5)),(1800,(99.0,103.0)),(2000,(99.2,102.8))]
    if metric == "MC":
        return [(1000,(98.0,102.0)),(1100,(98.0,102.0)),(1200,(98.0,102.0)),
                (1400,(98.0,102.2)),(1600,(97.8,102.4)),(1800,(97.6,102.6)),(2000,(97.5,102.7))]
    if metric == "LP":
        return [(1000,(96.0,104.0)),(1100,(96.5,103.8)),(1200,(97.0,103.5)),
                (1400,(97.5,103.0)),(1600,(98.0,102.5)),(1800,(98.3,102.3)),(2000,(98.5,102.0))]
    if metric == "LL":
        return [(1000,(98.5,101.5)),(1100,(98.0,102.0)),(1200,(98.0,102.0)),
                (1400,(97.5,102.5)),(1600,(97.0,103.0)),(1800,(96.5,103.5)),(2000,(96.0,104.0))]
    return [(1200,(98.0,102.0)),(2000,(98.0,102.0))]

def _prior_band(dm, metric):
    knots = _band_knots(metric)
    dm = float(dm)
    if dm <= knots[0][0]:  return knots[0][1]
    if dm >= knots[-1][0]: return knots[-1][1]
    for (ad,(alo,ahi)),(bd,(blo,bhi)) in zip(knots, knots[1:]):
        if ad <= dm <= bd:
            t = (dm-ad)/(bd-ad)
            return (lerp(alo,blo,t), lerp(ahi,bhi,t))
    return knots[-1][1]

def _shape_shift(lo, hi, metric, rsi, sci):
    if not np.isfinite(sci) or sci <= 0: return lo, hi
    shift  = 0.2 * float(min(1.0, max(0.0, sci)))
    center = 0.5*(lo+hi); half = 0.5*(hi-lo)
    if rsi < -1e-9:
        if metric in ("EZ","LP"): center += shift
        if metric in ("LL","MC"): center -= shift
    elif rsi > 1e-9:
        if metric in ("LL","MC"): center += shift
        if metric in ("EZ","LP"): center -= shift
    return (center-half, center+half)

def _blend_with_field(lo, hi, series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        q10, q90 = np.nanpercentile(s.dropna(), [10, 90])
        lo = 0.7*lo + 0.3*float(q10)
        hi = 0.7*hi + 0.3*float(q90)
    return lo, max(hi, lo+1.0)

def _s01(x, lo, hi):
    try:
        xv = float(x)
        return float(max(0.0, min(1.0, (xv-lo)/(hi-lo)))) if np.isfinite(xv) else 0.0
    except Exception:
        return 0.0

bands = {}
for mk, col in [("EZ","F200_idx"),("MC","tsSPI"),("LP","Accel"),("LL",gr_col)]:
    lo0,hi0 = _prior_band(D_m, mk)
    lo1,hi1 = _shape_shift(lo0,hi0,mk,RSI,SCI)
    bands[mk] = _blend_with_field(lo1,hi1, WD[col])

WD["EZ01"] = WD["F200_idx"].map(lambda v: _s01(v, *bands["EZ"]))
WD["MC01"] = WD["tsSPI"].map(   lambda v: _s01(v, *bands["MC"]))
WD["LP01"] = WD["Accel"].map(   lambda v: _s01(v, *bands["LP"]))
WD["LL01"] = WD[gr_col].map(    lambda v: _s01(v, *bands["LL"]))

# SOS
def _rz(s):
    mu, sd = np.nanmedian(s), mad_std(s)
    sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
    return (s - mu) / sd

ts = winsorize(pd.to_numeric(WD["tsSPI"], errors="coerce"))
ac = winsorize(pd.to_numeric(WD["Accel"],  errors="coerce"))
gr = winsorize(pd.to_numeric(WD[gr_col],   errors="coerce"))
raw_sos = 0.45*_rz(ts) + 0.35*_rz(ac) + 0.20*_rz(gr)
if raw_sos.notna().any():
    q5, q95 = np.nanpercentile(raw_sos.dropna(), [5, 95])
    WD["SOS01"] = ((raw_sos - q5) / max(q95-q5, 1.0)).clip(0.0, 1.0)
else:
    WD["SOS01"] = 0.0

# Weights
def _base_w(dm):
    knots = [(1000,{"EZ":0.25,"MC":0.21,"LP":0.28,"LL":0.11}),
             (1100,{"EZ":0.22,"MC":0.22,"LP":0.27,"LL":0.14}),
             (1200,{"EZ":0.20,"MC":0.22,"LP":0.25,"LL":0.18}),
             (1400,{"EZ":0.15,"MC":0.24,"LP":0.24,"LL":0.22}),
             (1600,{"EZ":0.10,"MC":0.26,"LP":0.23,"LL":0.26}),
             (1800,{"EZ":0.06,"MC":0.28,"LP":0.22,"LL":0.29}),
             (2000,{"EZ":0.03,"MC":0.30,"LP":0.20,"LL":0.32})]
    dm = float(dm)
    if dm <= knots[0][0]: return knots[0][1]
    if dm >= knots[-1][0]: return knots[-1][1]
    for (ad,aw),(bd,bw) in zip(knots,knots[1:]):
        if ad <= dm <= bd:
            t = (dm-ad)/(bd-ad)
            return {k: lerp(aw[k],bw[k],t) for k in aw}
    return knots[-1][1]

w = _base_w(D_m).copy()
mag = 0.01 * max(0.0, min(1.0, SCI))
if RSI < -1e-9:
    w["EZ"]+=mag; w["LP"]+=mag
    w["LL"]=max(0.0,w["LL"]-mag/2); w["MC"]=max(0.0,w["MC"]-mag/2)
elif RSI > 1e-9:
    w["LL"]+=mag; w["MC"]+=mag
    w["EZ"]=max(0.0,w["EZ"]-mag/2); w["LP"]=max(0.0,w["LP"]-mag/2)
s = sum(w.values()) or 1.0
w = {k: v/s*0.85 for k,v in w.items()}
w["SOS"] = 0.15
S = sum(w.values()) or 1.0
W = {k: v/S for k,v in w.items()}

comp01 = (W["EZ"]*WD["EZ01"].fillna(0) + W["MC"]*WD["MC01"].fillna(0) +
          W["LP"]*WD["LP01"].fillna(0) + W["LL"]*WD["LL01"].fillna(0) +
          W["SOS"]*WD["SOS01"].fillna(0))
med_ = float(np.nanmedian(comp01)) if comp01.notna().any() else 0.5
q10,q90 = (np.nanpercentile(comp01.dropna(),[10,90])
           if comp01.notna().any() else (0.3,0.7))
WD["WinningDNA"] = (5.0 + 5.0*((comp01-med_)/max(q90-q10,1e-6))).clip(0,10).round(2)

def _top_traits(r):
    pairs = [("Early Zip",r.get("EZ01",0)),("Mid Control",r.get("MC01",0)),
             ("Late Punch",r.get("LP01",0)),("Lasting Lift",r.get("LL01",0))]
    pairs = [(n,float(v if np.isfinite(v) else 0)) for n,v in pairs]
    pairs.sort(key=lambda x:x[1], reverse=True)
    keep = [n for n,v in pairs[:2] if v >= 0.55]
    ez,ll = float(r.get("EZ01",0)), float(r.get("LL01",0))
    if ez>=0.70 and ll<=0.45: keep.append("Sprinter-leaning")
    if ll>=0.70 and ez<=0.45: keep.append("Stayer-leaning")
    return " . ".join(keep)

WD["DNA_TopTraits"] = WD.apply(_top_traits, axis=1)
WD.attrs["DNA_WEIGHTS"] = W
return WD
```

# ______________________________________________

# Hidden Horses v2

# ______________________________________________

def hidden_horses(metrics: pd.DataFrame, distance_m: float) -> pd.DataFrame:
hh     = metrics.copy()
gr_col = metrics.attrs.get(“GR_COL”, “Grind”)
RSI    = metrics.attrs.get(“RSI”, np.nan)
SCI    = metrics.attrs.get(“SCI”, np.nan)
collapse = float(metrics.attrs.get(“CollapseSeverity”, 0.0) or 0.0)

```
# SOS
need = {"tsSPI","Accel",gr_col}
if need.issubset(hh.columns) and len(hh) > 0:
    def _rz(s):
        mu,sd = np.nanmedian(s), mad_std(s)
        return (s-mu)/(sd if np.isfinite(sd) and sd>0 else 1.0)
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"],errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh[gr_col],  errors="coerce"))
    hh["SOS_raw"] = 0.45*_rz(ts_w)+0.35*_rz(ac_w)+0.20*_rz(gr_w)
    q5,q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    hh["SOS"] = (2.0*(hh["SOS_raw"]-q5)/max(q95-q5,1.0)).clip(0,2)
else:
    hh["SOS"] = 0.0

# TFS
step = int(metrics.attrs.get("STEP",100))
def _tfs_row(r):
    last_cols=[c for c in ["300_Time","200_Time","100_Time"] if c in r.index]
    spds=[step/as_num(r.get(c)) for c in last_cols
          if pd.notna(r.get(c)) and as_num(r.get(c))>0]
    if len(spds)<2: return np.nan
    sigma=np.std(spds,ddof=0)
    mid=as_num(r.get("_MID_spd"))
    return np.nan if not np.isfinite(mid) or mid<=0 else 100.0*(sigma/mid)
hh["TFS"]=hh.apply(_tfs_row, axis=1)
D_r = int(np.ceil(float(distance_m)/200.0)*200)
gate=4.0 if D_r<=1200 else (3.5 if D_r<1800 else 3.0)
hh["TFS_plus"]=hh["TFS"].apply(
    lambda x: 0.0 if pd.isna(x) or x<gate else min(0.6,(x-gate)/3.0))

# ASI
def _rz2(s):
    s=winsorize(pd.to_numeric(s,errors="coerce"))
    mu=np.nanmedian(s); mad=np.nanmedian(np.abs(s-mu))
    sd=1.4826*mad if mad>0 else np.nanstd(s)
    if not np.isfinite(sd) or sd<=0: sd=1.0
    return (s-mu)/sd

if not np.isfinite(RSI) or not np.isfinite(SCI):
    zE = _rz2(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0,index=hh.index)
    zL = _rz2(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0,index=hh.index)
    RSI = float(np.nanmedian(zE)-np.nanmedian(zL))
    SCI = 0.50

_dir=0 if (not np.isfinite(RSI) or abs(RSI)<1e-6) else (1 if RSI>0 else -1)
FS=0.0 if _dir==0 else (0.6+0.4*max(0.0,min(1.0,float(SCI))))*min(1.0,abs(float(RSI))/2.0)
if collapse>=3.0: FS*=0.75

zA,zG=_rz2(hh.get("Accel")),_rz2(hh.get(gr_col))
SO=(zG-zA).clip(lower=0) if _dir==1 else ((zA-zG).clip(lower=0) if _dir==-1 else pd.Series(0.0,index=hh.index))

zE=_rz2(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0,index=hh.index)
zL=_rz2(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0,index=hh.index)
XO=(zL-zE).clip(lower=0) if _dir==1 else ((zE-zL).clip(lower=0) if _dir==-1 else pd.Series(0.0,index=hh.index))

tfs_plus=pd.to_numeric(hh.get("TFS_plus"),errors="coerce").fillna(0.0)
gr_adj  =pd.to_numeric(hh.get("GrindAdjPts"),errors="coerce").fillna(1.0)
D1=1.0-np.minimum(0.35,tfs_plus.clip(lower=0))
D2=1.0-np.minimum(0.25,((gr_adj-1.0).clip(lower=0)/3.0))
D_=D1*D2
Opp=0.6*SO+0.4*XO
hh["ASI2"]=(0.2*10.0*FS*Opp*D_).clip(0.0,2.0).fillna(0.0)

# UEI
def _uei(r):
    ts,ac,gr=[as_num(r.get(k)) for k in ("tsSPI","Accel",gr_col)]
    if any(pd.isna([ts,ac,gr])): return 0.0
    val=0.0
    if ts>=102 and ac<=98 and gr<=98:
        val=0.3+0.3*min((ts-102)/3.0,1.0)
    if ts>=102 and gr>=102 and ac<=100:
        val=max(val,0.3+0.3*min(((ts-102)+(gr-102))/6.0,1.0))
    return round(val,3)
hh["UEI"]=hh.apply(_uei,axis=1)

hidden=(0.55*hh["SOS"]+0.30*hh["ASI2"]+0.10*hh["TFS_plus"]+0.05*hh["UEI"]).fillna(0)
if len(hh)<=6: hidden*=0.9
h_med=float(np.nanmedian(hidden))
h_mad=float(np.nanmedian(np.abs(hidden-h_med)))
h_sigma=max(1e-6,1.4826*h_mad)
hh["HiddenScore"]=(1.2+(hidden-h_med)/(2.5*h_sigma)).clip(0.0,3.0)

def _tier(r):
    hs=as_num(r.get("HiddenScore"))
    if not np.isfinite(hs): return ""
    pi_v  =as_num(r.get("PI"))
    gci_rs=as_num(r.get("GCI_RS")) if pd.notna(r.get("GCI_RS")) else as_num(r.get("GCI"))
    def ok(top):
        if top: return (np.isfinite(pi_v) and pi_v>=5.4) or (np.isfinite(gci_rs) and gci_rs>=4.8)
        return (np.isfinite(pi_v) and pi_v>=4.8) or (np.isfinite(gci_rs) and gci_rs>=4.2)
    if hs>=1.8 and ok(True):  return "🔥 Top Hidden"
    if hs>=1.2 and ok(False): return "🟡 Notable Hidden"
    return ""
hh["Tier"]=hh.apply(_tier,axis=1)

def _note(r):
    pi,gci_rs=as_num(r.get("PI")),as_num(r.get("GCI_RS"))
    bits=[]
    if np.isfinite(pi) and np.isfinite(gci_rs):
        bits.append(f"PI {pi:.2f}, GCI_RS {gci_rs:.2f}")
    else:
        if as_num(r.get("SOS"))>=1.2: bits.append("sectionals superior")
        asi2=as_num(r.get("ASI2"))
        if asi2>=0.8: bits.append("ran against strong bias")
        elif asi2>=0.4: bits.append("ran against bias")
        if as_num(r.get("TFS_plus"))>0: bits.append("trip friction late")
        if as_num(r.get("UEI"))>=0.5: bits.append("latent potential if shape flips")
    return ("; ".join(bits)).capitalize()+"."
hh["Note"]=hh.apply(_note,axis=1)

tier_order=pd.CategoricalDtype(["🔥 Top Hidden","🟡 Notable Hidden",""],ordered=True)
hh["Tier"]=hh["Tier"].astype(tier_order)
return hh.sort_values(["Tier","HiddenScore","PI"],ascending=[True,False,False])
```

# ______________________________________________

# PWX + EFI

# ______________________________________________

def pwx_efi(metrics: pd.DataFrame) -> pd.DataFrame | None:
GR_COL = metrics.attrs.get(“GR_COL”,“Grind”)
if not {“Horse”,“Accel”,GR_COL,“tsSPI”}.issubset(metrics.columns):
return None

```
def _nz(s): return pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan)
def _mad_(x): x=np.asarray(x,float); m=np.nanmedian(x); return 1.4826*np.nanmedian(np.abs(x-m))
def _rz(s):
    v=_nz(s); m=np.nanmedian(v); d=_mad_(v)
    d=d if(np.isfinite(d) and d>1e-9) else 1.0
    return (v-m)/d
def _pct_rank(s):
    v=_nz(s).to_numpy(); order=np.argsort(v,kind="mergesort")
    ranks=np.empty_like(order,float); ranks[order]=np.arange(1,len(v)+1)
    pr=(ranks-0.5)/max(len(v),1); pr[~np.isfinite(v)]=0.5
    return pd.Series(pr,index=s.index)
def _cal(raw):
    pr=_pct_rank(raw); z=_rz(raw); sig=1/(1+np.exp(-0.85*z)); q=0.6*pr+0.4*sig
    q_src=[0.01,0.05,0.25,0.5,0.75,0.9,0.985,0.995]
    y_tgt=[0.6,1.6,3.6,5,6.6,8,9.5,9.9]
    s=np.interp(q,q_src,y_tgt).clip(0,10)
    n=len(raw)
    if n<10: s=(1-min(0.5,(10-n)/10))*s+min(0.5,(10-n)/10)*5
    return pd.Series(s,index=raw.index)

df=metrics[["Horse","Accel",GR_COL,"tsSPI"]].copy()
for c in ["Accel",GR_COL,"tsSPI"]: df[c]=_nz(df[c])

z_acc=_rz(df["Accel"]); z_tsi=_rz(df["tsSPI"])
PWX_raw=0.70*z_acc+0.25*z_tsi
sm=1-(df["Accel"]-df[GR_COL]).abs()/((df["Accel"]+df[GR_COL]).replace(0,np.nan)/2)
EFI_raw=0.45*_rz(df["tsSPI"])+0.35*_rz(df[GR_COL])+0.20*_rz(sm)
df["PWX"]=_cal(PWX_raw); df["EFI"]=_cal(EFI_raw)
return df[["Horse","PWX","EFI"]].sort_values(["PWX","EFI"],ascending=[False,False]).reset_index(drop=True)
```

# ______________________________________________

# Fatigue Gradient

# ______________________________________________

def fatigue_gradient(metrics: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
“”“Refined one-run fatigue table.”””

```
def _to_num(s): return pd.to_numeric(s,errors="coerce").astype(np.float32)
def _nz_(x,alt=0.0):
    try: v=float(x); return v if np.isfinite(v) else float(alt)
    except: return float(alt)

def _late_markers(cols):
    want100=["500_Time","400_Time","300_Time","200_Time","100_Time","Finish_Time"]
    want200=["400_Time","200_Time","Finish_Time"]; have=set(cols)
    if all(c in have for c in ["200_Time","Finish_Time"]) and \
            any(c in have for c in ["300_Time","100_Time","500_Time","400_Time"]):
        got=[c for c in want100 if c in have]
        if len(got)>=4:
            return [(c.replace("_Time",""),100.0,c) if c!="Finish_Time"
                    else ("Finish",np.nan,c) for c in got]
    got=[c for c in want200 if c in have]
    if len(got)>=3:
        return [(c.replace("_Time",""),200.0,c) if c!="Finish_Time"
                else ("Finish",np.nan,c) for c in got]
    if "200_Time" in have and "Finish_Time" in have:
        return [("200",200.0,"200_Time"),("Finish",np.nan,"Finish_Time")]
    return []

def _late_speeds(row,segs,step_def=100.0):
    spd=[]
    for (_,L,src) in segs:
        Lm=float(L) if np.isfinite(L) else float(step_def)
        t=_nz_(row.get(src,np.nan),np.nan)
        spd.append(Lm/t if (np.isfinite(t) and t>0) else np.nan)
    return np.asarray(spd,dtype=np.float32)

def _ts_slope(x,y):
    m=np.isfinite(x)&np.isfinite(y); xx,yy=x[m],y[m]
    if len(xx)<2: return np.nan
    sl=[(yy[j]-yy[i])/((xx[j]-xx[i])*200.0)
        for i in range(len(xx)) for j in range(i+1,len(xx)) if xx[j]!=xx[i]]
    return float(np.median(sl)) if sl else np.nan

def _ols_slope(x,y):
    m=np.isfinite(x)&np.isfinite(y); xx,yy=x[m],y[m]
    if len(xx)<2: return np.nan
    xm,ym=float(np.mean(xx)),float(np.mean(yy))
    num=float(np.sum((xx-xm)*(yy-ym))); den=float(np.sum((xx-xm)**2))
    return float(num/den/200.0) if den>0 else np.nan

def _mono(v):
    w=v[np.isfinite(v)]
    if len(w)<=2: return 0.5
    sgn=np.sign(np.diff(w))
    flips=np.sum(np.abs(np.diff(sgn))>0)
    return float(1.0-flips/max(1,len(sgn)-1))

df   = metrics.copy()
step = _nz_(metrics.attrs.get("STEP",100),100.0)
RSI  = metrics.attrs.get("RSI",np.nan)
SCI  = metrics.attrs.get("SCI",np.nan)
bend = bool(metrics.attrs.get("LATE_BEND",True))

segs0=_late_markers(work.columns)
if not segs0:
    view=pd.DataFrame({"Horse":df.get("Horse",pd.Series(range(len(df))))})
    view["Note"]="Not enough *_Time columns to compute fatigue."
    return view

late_spds=[_late_speeds(r,segs0,step) for _,r in work.iterrows()]
x_idx=np.arange(len(segs0),dtype=np.float32)
slopes=np.full(len(df),np.nan,dtype=np.float32)
monos =np.full(len(df),np.nan,dtype=np.float32)

for i in range(len(df)):
    vi=np.asarray(late_spds[i],dtype=np.float32)
    if vi.size==0 or not np.isfinite(vi).any(): continue
    v=vi.copy()
    if bend and np.isfinite(v[0]) and v.size>=2 and np.isfinite(v[1]):
        v[0]=0.8*v[0]+0.2*v[1]
    vf=v[np.isfinite(v)]
    if vf.size>=3:
        med_=float(np.median(vf)); dev=np.abs(v-med_)/(med_ if med_>0 else 1.0)
        j_=int(np.nanargmax(dev))
        if np.isfinite(dev[j_]) and dev[j_]>0.12: v[j_]=np.nan
    msk=np.isfinite(v); nfin=int(np.sum(msk))
    if nfin>=2:
        slopes[i]=(_ts_slope(x_idx[msk],v[msk]) if nfin>=5
                   else _ols_slope(x_idx[msk],v[msk]))
        monos[i]=_mono(v[msk])
    else:
        monos[i]=0.5

med_sl=np.nanmedian(slopes)
FG_vs_field=-(slopes-med_sl)
if len(df)<=7: FG_vs_field*=0.8

if "Line_vs_field" in df.columns:
    line_raw=_to_num(df["Line_vs_field"])
else:
    zE=_to_num(df.get("EARLY_idx",np.nan)); zL=_to_num(df.get("LATE_idx",np.nan))
    line_raw=(zL-zE)-np.nanmedian(zL-zE)
LineEff=(-line_raw).astype(np.float32)

EARLY_idx=_to_num(df.get("EARLY_idx",np.nan))
EHeat=np.maximum(0.0,EARLY_idx-104.0)
FS_shape=(1.0+np.sign(RSI)*min(0.12,0.08*(abs(float(RSI))**0.8))
          if (np.isfinite(RSI) and np.isfinite(SCI) and SCI>=0.6) else 1.0)
tsSPI_=_to_num(df.get("tsSPI",np.nan))
ClassAdj=np.clip((tsSPI_-100.0)/8.0,-0.2,0.2)
FG_eff=(LineEff-0.50*FG_vs_field-0.03*EHeat).astype(np.float32)
LCS=(0.60*LineEff+0.30*(-FG_vs_field)).astype(np.float32)
FCS=(0.60*(-LineEff)+0.30*FG_vs_field).astype(np.float32)
FatigueScore=(FS_shape*(LCS-FCS)-0.10*ClassAdj).astype(np.float32)

need_c=["Accel",metrics.attrs.get("GR_COL","Grind"),"EARLY_idx","LATE_idx"]
bits=[np.isfinite(_to_num(df[c])) for c in need_c if c in df.columns]
comp=(np.mean(np.column_stack(bits),axis=1).astype(np.float32)
      if bits else np.full(len(df),0.5,dtype=np.float32))
Mono=np.nan_to_num(monos,nan=0.5).astype(np.float32)
Reliability=(0.5*comp+0.5*np.clip(Mono,0,1)).astype(np.float32)

FS_arr=np.asarray(FatigueScore,float)
Tag=np.full(len(FS_arr),"balanced",dtype=object)
Tag[FS_arr< -3.0]="late engine"
Tag[FS_arr>  3.0]="front-spent"
Cue=np.full(len(FS_arr),"repeats",dtype=object)
Cue[FS_arr< -3.0]="↑ trip / stronger pace"
Cue[FS_arr>  3.0]="↓ trip / easier early"

view=pd.DataFrame({
    "Horse":      df.get("Horse",pd.Series(np.arange(len(df)),dtype=object)),
    "FG_rate/200m": np.round(slopes,4),
    "FG_vs_field":  np.round(FG_vs_field,3),
    "Line_vs_field":np.round(line_raw,3),
    "FG_eff(adjusted)":np.round(FG_eff,3),
    "Reliability":  np.round(Reliability,3),
    "Tag": Tag,"Cue": Cue,
    "FatigueScore": np.round(FatigueScore,3),
})
ord_g=np.where(view["Tag"]=="late engine",0,np.where(view["Tag"]=="balanced",1,2)).astype(np.int16)
return (view.assign(_g=ord_g,_s=-view["FatigueScore"].astype(float))
        .sort_values(["_g","_s","Reliability"],ascending=[True,True,False])
        .drop(columns=["_g","_s"]).reset_index(drop=True))
```

# ______________________________________________

# PWR400

# ______________________________________________

def compute_pwr400(
df: pd.DataFrame,
distance_m: float,
step: int,
weight_col: str = “Horse Weight”,
) -> pd.DataFrame:
w=df.copy(); step=int(step); D=float(distance_m)
fin=pd.to_numeric(w.get(“Finish_Time”),errors=“coerce”)
if fin.isna().all():
for c in [“PWR400_v400”,“PWR400_SIdx”,“PWR400_WIdx”,“PWR400_raw”,“PWR400”]:
w[c]=np.nan
w.attrs[“PWR400_NOTE”]={“error”:“Missing Finish_Time”}
return w

```
if step==200:
    if "200_Time" in w.columns:
        t_last400=pd.to_numeric(w["200_Time"],errors="coerce")+fin; src="200_Time + Finish_Time"
    else:
        t_last400=fin.copy(); src="Finish_Time only (fallback)"
else:
    segs=[]; used=[]
    for col in ["300_Time","200_Time","100_Time","Finish_Time"]:
        if col in w.columns:
            segs.append(pd.to_numeric(w[col],errors="coerce")); used.append(col)
    t_last400=sum(segs) if segs else fin.copy()
    src=" + ".join(used) if used else "Finish_Time only"

t_last400=t_last400.where(t_last400>0,np.nan)
v400_kmh=0.4/(t_last400/3600.0); v400_kmh=v400_kmh.replace([np.inf,-np.inf],np.nan)
w["PWR400_v400"]=v400_kmh
v_med=float(np.nanmedian(v400_kmh))
w["PWR400_SIdx"]=(100.0*(v400_kmh/v_med) if np.isfinite(v_med) and v_med>0 else np.nan)

W_=pd.to_numeric(w.get(weight_col),errors="coerce").where(lambda x:x>0,np.nan)
if W_.notna().any():
    Wm=float(np.nanmedian(W_))
    w["PWR400_WIdx"]=100.0*(W_/Wm) if (np.isfinite(Wm) and Wm>0) else np.nan
else:
    Wm=np.nan; w["PWR400_WIdx"]=np.nan

alpha=max(0.20,min(0.70,0.20+0.00025*(D-1000.0)))
S400=pd.to_numeric(w["PWR400_SIdx"],errors="coerce")
Widx=pd.to_numeric(w["PWR400_WIdx"],errors="coerce")
mask=S400.notna()&Widx.notna()
PWR_raw=pd.Series(np.nan,index=w.index,dtype=float)
PWR_raw[mask]=S400[mask]+alpha*(Widx[mask]-100.0)
w["PWR400_raw"]=PWR_raw
raw_med=float(np.nanmedian(PWR_raw))
w["PWR400"]=(100.0*(PWR_raw/raw_med) if np.isfinite(raw_med) and raw_med>0 else np.nan)
w.attrs["PWR400_NOTE"]={
    "distance_m":int(D),"alpha":round(alpha,4),"weight_col":weight_col,
    "v400_source":src,
    "v400_med":round(v_med,5) if np.isfinite(v_med) else None,
    "W_med":round(Wm,3) if np.isfinite(Wm) else None,
}
return w
```

# ______________________________________________

# R&V (CAR)

# ______________________________________________

def context_aware_reliability(metrics: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
df=metrics.copy()

```
def _winsor2(s,q=0.02):
    s=pd.to_numeric(s,errors="coerce"); lo,hi=s.quantile(q),s.quantile(1-q)
    return s.clip(lo,hi)
def _mad2(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if x.size==0: return np.nan
    return 1.4826*np.median(np.abs(x-np.median(x)))
def _r01(s):
    s=pd.to_numeric(s,errors="coerce"); finite=s[np.isfinite(s)]
    if finite.empty: return pd.Series(0.5,index=s.index)
    r=finite.rank(method="average",pct=True)
    out=pd.Series(np.nan,index=s.index); out.loc[finite.index]=r
    return out.fillna(0.5)

have_MV="MV" in df.columns; have_RC="RC" in df.columns
if not (have_MV and have_RC):
    step=int(metrics.attrs.get("STEP",100))
    tcols=sorted([c for c in work.columns if c.endswith("_Time") and c!="Finish_Time"],
                 key=lambda c: int(c.split("_")[0]) if c.split("_")[0].isdigit() else 10**8)
    if "Finish_Time" in work.columns: tcols+= ["Finish_Time"]
    seg_cols=tcols[1:] if len(tcols)>1 else tcols[:]
    if len(seg_cols)>=3:
        seg_lens=np.array([float(step)]*len(seg_cols))
        w_=work.copy()
        if "Horse" in w_.columns and "Horse" in df.columns:
            w_=w_.set_index("Horse").reindex(df["Horse"]).reset_index()
        spd=[]
        for _,r in w_.iterrows():
            times=pd.to_numeric(r[seg_cols],errors="coerce").to_numpy()
            with np.errstate(divide="ignore",invalid="ignore"): v=seg_lens/times
            v[~np.isfinite(v)]=np.nan; spd.append(v)
        spd=np.asarray(spd,dtype=float)
        valid=np.isfinite(spd).sum(axis=0)>=2; spd=spd[:,valid]
        if spd.shape[1]>=3:
            row_med=np.nanmedian(spd,axis=1,keepdims=True)
            spd_f=np.where(np.isfinite(spd),spd,row_med)
            med_=np.nanmedian(spd_f,axis=1)
            mad_=np.array([_mad2(spd_f[i,:]) for i in range(spd_f.shape[0])])
            with np.errstate(divide="ignore",invalid="ignore"):
                mv_raw=mad_/np.where(med_>0,med_,np.nan)
            if not have_MV: df["MV"]=pd.Series(mv_raw,index=df.index)
            H,S=spd_f.shape; late_start=max(1,S//2)
            trough=np.nanargmin(np.nanmedian(spd_f[:,late_start:],axis=0))+late_start
            tail=np.nanmean(spd_f[:,max(S-2,0):S],axis=1); base_=spd_f[:,trough]
            with np.errstate(divide="ignore",invalid="ignore"):
                rc_raw=tail/np.where(base_>0,base_,np.nan)
            if not have_RC: df["RC"]=pd.Series(rc_raw,index=df.index)

mv=_winsor2(df.get("MV",pd.Series(np.nan,index=df.index)))
rc=_winsor2(df.get("RC",pd.Series(np.nan,index=df.index)))
mv01=1.0-_r01(mv); rc01=_r01(rc)
if "PI" in df.columns and pd.to_numeric(df["PI"],errors="coerce").notna().any():
    cls=_winsor2(pd.to_numeric(df["PI"],errors="coerce"))
else:
    cls=_winsor2(pd.to_numeric(df.get("GCI",np.nan),errors="coerce"))
cls01=_r01(cls)
mv_term=np.power(mv01,1.0-0.35*cls01); rc_term=rc01*(1.0+0.25*cls01)
CAR=np.clip(10.0*(0.45*mv_term+0.35*rc_term+0.20*cls01),0.0,10.0)

def _tag(v):
    if not np.isfinite(v): return "None"
    if v>=8: return "Reliable"
    if v>=6: return "Mostly reliable"
    if v>=4: return "Variable"
    return "Fragile"

out=pd.DataFrame({
    "Horse":df["Horse"],"R&V":np.round(CAR,2),
    "Risk Tag":[_tag(v) for v in CAR],
    "MV (↓ steadier)":np.round(pd.to_numeric(df.get("MV",np.nan),errors="coerce"),3),
    "RC (↑ better rebound)":np.round(pd.to_numeric(df.get("RC",np.nan),errors="coerce"),3),
})
if "PI" in df.columns:
    out["PI"]=np.round(pd.to_numeric(df["PI"],errors="coerce"),2)
sort_c=["R&V"]+( ["PI"] if "PI" in out.columns else [])
return out.sort_values(sort_c,ascending=[False]*len(sort_c),kind="mergesort").reset_index(drop=True)
```

# ______________________________________________

# xWin

# ______________________________________________

def xwin(
metrics: pd.DataFrame,
distance_m: float,
going: str = “Good”,
tfs_series: pd.Series | None = None,
) -> pd.DataFrame:
XW=metrics.copy(); gr_col=metrics.attrs.get(“GR_COL”,“Grind”)
RSI=float(metrics.attrs.get(“RSI”,0.0)); SCI=float(metrics.attrs.get(“SCI”,0.0))
D_m=float(distance_m)

```
def _rz(s):
    s2=pd.to_numeric(s,errors="coerce")
    lo,hi=s2.quantile(0.02),s2.quantile(0.98); s2=s2.clip(lo,hi)
    mu=np.nanmedian(s2); sd=mad_std(s2)
    sd=sd if (np.isfinite(sd) and sd>0) else 1.0
    return (s2-mu)/sd

def _dist_weights(dm):
    dm=float(dm)
    knots=[(1000,dict(T=0.30,K=0.45,S=0.25)),(1200,dict(T=0.30,K=0.40,S=0.30)),
           (1400,dict(T=0.32,K=0.36,S=0.32)),(1600,dict(T=0.34,K=0.32,S=0.34)),
           (1800,dict(T=0.36,K=0.28,S=0.36)),(2000,dict(T=0.38,K=0.25,S=0.37)),
           (2400,dict(T=0.40,K=0.22,S=0.38))]
    if dm<=knots[0][0]: return knots[0][1]
    if dm>=knots[-1][0]: return knots[-1][1]
    for (a,aw),(b,bw) in zip(knots,knots[1:]):
        if a<=dm<=b:
            t=(dm-a)/(b-a); return {k:lerp(aw[k],bw[k],t) for k in aw}
    return knots[-1][1]

def _going_nudge(w,g,n=12):
    w=w.copy(); sc=min(1.0,max(1,int(n))/12.0)
    if g=="Firm":    w["K"]*=(1+0.04*sc); w["T"]*=(1+0.02*sc); w["S"]*=(1-0.04*sc)
    elif g in ("Soft","Heavy"):
        amp=0.05 if g=="Soft" else 0.08
        w["S"]*=(1+amp*sc); w["T"]*=(1+0.02*sc); w["K"]*=(1-amp*sc)
    S=sum(w.values()) or 1.0
    for k in w: w[k]/=S
    return w

for c in ["tsSPI","Accel",gr_col,"F200_idx","PI"]:
    if c not in XW.columns: XW[c]=np.nan

zT=_rz(XW["tsSPI"]); zK=_rz(XW["Accel"]); zS=_rz(XW[gr_col])
ts_med=pd.to_numeric(XW["tsSPI"],errors="coerce").median(skipna=True)
trim_T=min(0.20,max(0.0,(100.0-ts_med)/10.0)) if (np.isfinite(ts_med) and ts_med<100) else 0.0
zT_eff=zT*(1.0-trim_T)

W_=_going_nudge(_dist_weights(D_m), going, len(XW))

ksi_raw=-np.sign(RSI)*(pd.to_numeric(XW["Accel"],errors="coerce")-pd.to_numeric(XW["tsSPI"],errors="coerce"))
ksi01=np.tanh((ksi_raw/6.0).fillna(0.0))
shape_boost=0.15*np.clip(ksi01,0,1)*SCI
shape_damp =0.08*np.clip(-ksi01,0,1)*SCI

if tfs_series is not None:
    tfs_plus=tfs_series.reindex(XW.index).fillna(0.0)
else:
    tfs_plus=pd.Series(0.0,index=XW.index)
tfs_cap=(0.12 if D_m<=1400 else (0.08 if D_m>=1800
         else lerp(0.12,0.08,(D_m-1400)/400.0)))
tfs_pen=np.minimum(tfs_cap,np.maximum(0.0,(tfs_plus-0.2)/0.4))

sos=(0.45*zT+0.35*zK+0.20*zS).fillna(0.0)
sos01=((sos-np.nanpercentile(sos,5))/max(1e-9,np.nanpercentile(sos,95)-np.nanpercentile(sos,5))).clip(0,1)
core=(W_["T"]*zT_eff.fillna(0)+W_["K"]*zK.fillna(0)+W_["S"]*zS.fillna(0)+0.05*sos01.fillna(0))
power=(core*(1.0+shape_boost-shape_damp)*(1.0-tfs_pen)).fillna(0.0)

N=int(len(XW)); alpha_=N/(N+6.0)
if N<=6: power*=0.90

def _tau():
    def _m01(s):
        d=mad_std(pd.to_numeric(s,errors="coerce"))
        return float(min(1.0,d/4.5)) if np.isfinite(d) else 0.0
    base=0.95-0.04*np.log1p(N)-0.16*(0.5*_m01(XW["Accel"])+0.5*_m01(XW[gr_col]))
    base+=(0.04 if D_m<=1100 else (0.00 if D_m<=1800 else -0.02))
    if tfs_plus is not None:
        tp=pd.to_numeric(tfs_plus,errors="coerce").fillna(0.0)
        base+=0.08*float(np.clip(np.nanmean(np.maximum(0.0,(tp-0.2)/0.4)),0.0,1.0))
    return float(clamp(base,0.55,1.15))

tau=_tau()
logits=power/max(1e-6,tau); mx=float(np.nanmax(logits)) if np.isfinite(logits).any() else 0.0
exps=np.exp((logits-mx).clip(-50,50)); sum_e=float(np.nansum(exps)) or 1.0
probs=(exps/sum_e)*alpha_; probs/=(probs.sum() or 1.0)
XW["xWin"]=probs

def _driver(r):
    bits=[]; f200=float(r.get("F200_idx",np.nan))
    if np.isfinite(f200):
        if f200>=101: bits.append("Quick early")
        elif f200<=98: bits.append("Slower away")
    if float(zT_eff.get(r.name,0))>=0.5: bits.append("Travel +")
    if float(zK.get(r.name,0))>=0.5:     bits.append("Kick ++")
    if float(zS.get(r.name,0))>=0.5:     bits.append("Sustain +")
    if SCI>=0.6:
        k=float(ksi01.get(r.name,0))
        if k>0.35: bits.append("Against shape")
        elif k<-0.35: bits.append("With shape")
    if trim_T>0: bits.append("(slow mid)")
    return " . ".join(bits)

XW["Drivers"]=XW.apply(_driver,axis=1)
view=XW[["Horse","xWin","Drivers"]].copy()
view["xWin"]=(100.0*view["xWin"]).round(1)

def _odds(p):
    try:
        p=float(p); return f"{1/p-1:.1f}/1" if p>0 else "-"
    except: return "-"
view["Odds (≈fair)"]=XW["xWin"].apply(_odds)
view.attrs["tau"]=tau; view.attrs["weights"]=W_
return view.sort_values("xWin",ascending=False).reset_index(drop=True)
```
