“””
visuals.py - chart builders for Race Edge.

All functions return matplotlib Figure objects.
No Streamlit imports - purely visual.
“””
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from utils import color_cycle, safe_bal_norm
from data_io import collect_markers

# –––––––––––––––––––––––

# Label repel (built-in fallback)

# –––––––––––––––––––––––

def *repel_labels(ax, x, y, labels, *, init_shift=0.18, k_repel=0.012, max_iter=250):
trans = ax.transData
renderer = ax.figure.canvas.get_renderer()
xy   = np.column_stack([x, y]).astype(float)
offs = np.zeros_like(xy)
for i, (xi, yi) in enumerate(xy):
offs[i] = [init_shift if xi >= 0 else -init_shift,
init_shift if yi >= 0 else -init_shift]
texts, lines = [], []
for (xi, yi), (dx, dy), lab in zip(xy, offs, labels):
t = ax.text(xi + dx, yi + dy, lab, fontsize=8.4, va=“center”, ha=“left”,
bbox=dict(boxstyle=“round,pad=0.18”, fc=“white”, ec=“none”, alpha=0.75))
texts.append(t)
ln = Line2D([xi, xi + dx], [yi, yi + dy], lw=0.75, color=“black”, alpha=0.9)
ax.add_line(ln); lines.append(ln)
inv = ax.transData.inverted()
for _ in range(max_iter):
moved = False
bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.15) for t in texts]
for i in range(len(texts)):
for j in range(i + 1, len(texts)):
if not bbs[i].overlaps(bbs[j]): continue
ci = ((bbs[i].x0 + bbs[i].x1) / 2, (bbs[i].y0 + bbs[i].y1) / 2)
cj = ((bbs[j].x0 + bbs[j].x1) / 2, (bbs[j].y0 + bbs[j].y1) / 2)
vx, vy = ci[0] - cj[0], ci[1] - cj[1]
if vx == 0 and vy == 0: vx = 1.0
n* = (vx**2 + vy**2)**0.5
dx_, dy_ = (vx / n_) * k_repel * 72, (vy / n_) * k_repel * 72
for t, s in ((texts[i], +1), (texts[j], -1)):
tx, ty = t.get_position()
px = trans.transform((tx, ty)) + s * np.array([dx_, dy_])
t.set_position(inv.transform(px)); moved = True
if not moved: break
for t, ln, (xi, yi) in zip(texts, lines, xy):
tx, ty = t.get_position(); ln.set_data([xi, tx], [yi, ty])

def label_points_neatly(ax, x, y, names):
try:
from adjustText import adjust_text
texts = [ax.text(xi, yi, nm, fontsize=8.4,
bbox=dict(boxstyle=“round,pad=0.18”, fc=“white”, ec=“none”, alpha=0.75))
for xi, yi, nm in zip(x, y, names)]
adjust_text(texts, x=x, y=y, ax=ax,
only_move={“points”: “y”, “text”: “xy”},
force_points=0.6, force_text=0.7,
expand_text=(1.05, 1.15), expand_points=(1.05, 1.15),
arrowprops=dict(arrowstyle=”->”, lw=0.75, color=“black”,
alpha=0.9, shrinkA=0, shrinkB=3))
except Exception:
_repel_labels(ax, x, y, names)

# –––––––––––––––––––––––

# Shared quadrant background helper

# –––––––––––––––––––––––

_QUAD_COLORS = [”#4daf4a”, “#377eb8”, “#ff7f00”, “#984ea3”]

def _draw_quadrants(ax, lim, tint=0.12):
ax.add_patch(Rectangle((0,  0),    lim,  lim, facecolor=_QUAD_COLORS[0], alpha=tint, zorder=0))
ax.add_patch(Rectangle((-lim, 0),  lim,  lim, facecolor=_QUAD_COLORS[1], alpha=tint, zorder=0))
ax.add_patch(Rectangle((0,  -lim), lim,  lim, facecolor=_QUAD_COLORS[2], alpha=tint, zorder=0))
ax.add_patch(Rectangle((-lim,-lim),lim,  lim, facecolor=_QUAD_COLORS[3], alpha=tint, zorder=0))
ax.axvline(0, color=“gray”, lw=1.3, ls=(0, (3, 3)), zorder=1)
ax.axhline(0, color=“gray”, lw=1.3, ls=(0, (3, 3)), zorder=1)

def _pi_sizes(piv, dot_min=40.0, dot_max=140.0):
pmin, pmax = np.nanmin(piv), np.nanmax(piv)
if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin == pmax:
return np.full_like(piv, dot_min)
return dot_min + (piv - pmin) / (pmax - pmin + 1e-9) * (dot_max - dot_min)

def _pi_legend_handles(dot_min=40.0, dot_max=140.0):
s_ex = [dot_min, 0.5 * (dot_min + dot_max), dot_max]
return [Line2D([0], [0], marker=“o”, color=“w”, markerfacecolor=“gray”,
markersize=np.sqrt(s / np.pi), markeredgecolor=“black”)
for s in s_ex]

# –––––––––––––––––––––––

# Visual 1 - Sectional Shape Map

# –––––––––––––––––––––––

def shape_map(metrics: pd.DataFrame, use_cg: bool = True) -> plt.Figure:
gr_col = metrics.attrs.get(“GR_COL”, “Grind”)
dfm = metrics[[“Horse”, “Accel”, gr_col, “tsSPI”, “PI”]].copy()
for c in [“Accel”, gr_col, “tsSPI”, “PI”]:
dfm[c] = pd.to_numeric(dfm[c], errors=“coerce”)
dfm = dfm.dropna(subset=[“Accel”, gr_col, “tsSPI”])

```
xv  = (dfm["Accel"]  - 100.0).to_numpy()
yv  = (dfm[gr_col]   - 100.0).to_numpy()
cv  = (dfm["tsSPI"]  - 100.0).to_numpy()
piv = dfm["PI"].fillna(0).to_numpy()
names = dfm["Horse"].astype(str).tolist()

span = max(4.5, float(np.nanmax(np.abs(np.concatenate([xv, yv])))))
lim  = np.ceil(span / 1.5) * 1.5

fig, ax = plt.subplots(figsize=(7.8, 6.2))
_draw_quadrants(ax, lim)

vmin, vmax = np.nanmin(cv), np.nanmax(cv)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = -1.0, 1.0
norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

sc = ax.scatter(xv, yv, s=_pi_sizes(piv), c=cv, cmap="coolwarm", norm=norm,
                edgecolor="black", linewidth=0.6, alpha=0.95)
label_points_neatly(ax, xv, yv, names)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Acceleration vs field (points) →")
ax.set_ylabel(("Corrected " if use_cg else "") + "Grind vs field (points) ↑")
ax.set_title("Shape Map . +X=Accel . +Y=" + ("CG" if use_cg else "Grind") + " . Colour=tsSPIΔ")
ax.legend(_pi_legend_handles(), ["PI low", "PI mid", "PI high"],
          loc="upper left", frameon=False, fontsize=8)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("tsSPI − 100")
ax.grid(True, linestyle=":", alpha=0.25)
fig.tight_layout()
return fig
```

# –––––––––––––––––––––––

# Visual 2 - Pace Curve

# –––––––––––––––––––––––

def pace_curve(
work: pd.DataFrame,
metrics: pd.DataFrame,
race_distance: float,
) -> plt.Figure:
step   = int(metrics.attrs.get(“STEP”, 100))
marks  = collect_markers(work)

```
segs = []
if marks:
    m1 = int(marks[0])
    L0 = max(1.0, race_distance - m1)
    if f"{m1}_Time" in work.columns:
        segs.append((f"{int(race_distance)}→{m1}", float(L0), f"{m1}_Time"))
    for a, b in zip(marks, marks[1:]):
        src = f"{int(b)}_Time"
        if src in work.columns:
            segs.append((f"{int(a)}→{int(b)}", float(a - b), src))
if "Finish_Time" in work.columns:
    segs.append((f"{step}→0 (Finish)", float(step), "Finish_Time"))

if not segs:
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No segment data", ha="center"); return fig

arr = np.full((len(work), len(segs)), np.nan, dtype="float32")
for j, (_, L, col) in enumerate(segs):
    if col in work.columns:
        t = pd.to_numeric(work[col], errors="coerce").astype("float32")
        t = np.where((t > 0) & np.isfinite(t), t, np.nan)
        arr[:, j] = L / t
field_avg = np.nanmean(arr, axis=0)

if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
    top10 = metrics.nsmallest(10, "Finish_Pos")
else:
    top10 = metrics.nlargest(10, "PI")

x_idx    = np.arange(len(segs))
x_labels = [lbl for (lbl, _, _) in segs]
palette  = color_cycle(len(top10))

fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.plot(x_idx, field_avg, color="black", lw=2.0, label="Field average")
for i, (_, r) in enumerate(top10.iterrows()):
    speeds = np.full(len(segs), np.nan, dtype="float32")
    for j, (_, L, col) in enumerate(segs):
        t = pd.to_numeric(r.get(col, np.nan), errors="coerce")
        if np.isfinite(t) and t > 0:
            speeds[j] = L / float(t)
    if np.any(np.isfinite(speeds)):
        ax.plot(x_idx, speeds, lw=1.0, marker="o", ms=2.5,
                color=palette[i], label=str(r.get("Horse", "")))

ax.set_xticks(x_idx)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Speed (m/s)")
ax.set_title("Pace over segments (left=early, right=home straight)")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
return fig
```

# –––––––––––––––––––––––

# Visual 3 - Power-Freshness Map

# –––––––––––––––––––––––

def power_freshness_map(metrics: pd.DataFrame, ftab: pd.DataFrame) -> plt.Figure | None:
need_p  = {“Horse”, “PI”, “PWR400”}
need_fg = {“Horse”, “FatigueScore”, “Tag”}
if not (need_p.issubset(metrics.columns) and need_fg.issubset(ftab.columns)):
return None

```
dfm = metrics[["Horse", "PI", "PWR400"]].merge(
    ftab[["Horse", "FatigueScore", "Tag"]], on="Horse", how="inner"
)
for c in ["PI", "PWR400", "FatigueScore"]:
    dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
dfm = dfm.dropna(subset=["PWR400", "FatigueScore"])
if dfm.empty:
    return None

pwr_med = float(np.nanmedian(dfm["PWR400"]))
dfm["PWR400Δ"]  = dfm["PWR400"] - pwr_med
dfm["Freshness"] = -dfm["FatigueScore"]

tag_map = {"late engine": 3.0, "balanced": 0.0, "neutral": 0.0, "front-spent": -3.0}
dfm["TagVal"] = dfm["Tag"].map(tag_map).fillna(0.0)

xv = dfm["PWR400Δ"].to_numpy()
yv = dfm["Freshness"].to_numpy()
cv = dfm["TagVal"].to_numpy()
piv = dfm["PI"].fillna(0).to_numpy()
names = dfm["Horse"].astype(str).tolist()

span = max(4.5, float(np.nanmax(np.abs(np.concatenate([xv, yv])))))
lim  = np.ceil(span / 1.5) * 1.5

fig, ax = plt.subplots(figsize=(7.8, 6.2))
_draw_quadrants(ax, lim)

norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
sc   = ax.scatter(xv, yv, s=_pi_sizes(piv), c=cv, cmap="coolwarm", norm=norm,
                  edgecolor="black", linewidth=0.6, alpha=0.95)
label_points_neatly(ax, xv, yv, names)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("PWR400 − field median (late power under load) →")
ax.set_ylabel("Freshness (−FatigueScore) ↑")
ax.set_title("Power-Freshness Map  .  Size=PI  .  Colour=Fatigue Tag")
ax.legend(_pi_legend_handles(), ["PI low", "PI mid", "PI high"],
          loc="upper left", frameon=False, fontsize=8)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(
    "Fatigue Tag (−3 front-spent . 0 balanced . +3 late engine)"
)
ax.grid(True, linestyle=":", alpha=0.25)
fig.tight_layout()
return fig
```

# –––––––––––––––––––––––

# PNG export helper

# –––––––––––––––––––––––

def fig_to_png(fig: plt.Figure, dpi: int = 200) -> bytes:
buf = io.BytesIO()
fig.savefig(buf, format=“png”, dpi=dpi, bbox_inches=“tight”, facecolor=“white”)
return buf.getvalue()
