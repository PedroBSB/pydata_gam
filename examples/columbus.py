"""
Bayesian GAM on the COLUMBUS dataset (CRIME → HOVAL) with wiggliness regularization.
Two stacked panels:
  (1) Smooth effect f(x) with 95% credible band
  (2) Marginal effect df/dx with 95% credible band

NEW: The HTML includes a slider to change the *smoothing penalty scale* (ρ) interactively.
We pre-fit models for a small grid of penalty scales and switch the displayed curves via the slider.

Install:
  pip install numpy pandas plotly patsy pyro-ppl torch libpysal
Optional (for PNG export):
  pip install -U kaleido
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.chdir("..")

import numpy as np
import pandas as pd

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Spline basis
from patsy import dmatrix

# Data
from libpysal.examples import load_example
import libpysal

# PyTorch / Pyro
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

# ----------------------------
# Config
# ----------------------------
RANDOM_SEED = 7
SPLINE_DF   = 12        # effective degrees of freedom for the B-spline basis
SPLINE_DEG  = 3         # cubic

# SVI + slider grid
NUM_STEPS_PER_RHO = 2000
LR                = 0.02
N_DRAWS           = 300            # posterior draws per rho for bands
PENALTY_SCALES    = [0.05, 0.1, 0.2, 0.4, 0.8]   # ρ (Normal(0, ρ) on D2 w); smaller → stronger smoothing

PLOT_WIDTH  = 980
PLOT_HEIGHT = 720  # taller for two rows
SAVE_DIR    = "./resources"
os.makedirs(SAVE_DIR, exist_ok=True)

# Use double throughout for Pyro/PyTorch numerics
torch.set_default_dtype(torch.double)
rng = np.random.default_rng(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ----------------------------
# Load COLUMBUS data (CRIME & HOVAL)
# ----------------------------
ex = load_example("Columbus")
dbf_path = ex.get_path("columbus.dbf")

# libpysal DBF is *not* a context manager → open/close explicitly
db = libpysal.io.open(dbf_path)
df = pd.DataFrame({col: db.by_col(col) for col in db.header})
db.close()

# Use CRIME as predictor, HOVAL (median housing value) as response
df = df[["CRIME", "HOVAL"]].dropna().copy()

# Scale predictor to [0,1] for numerics
x_raw = df["CRIME"].values.astype(float)
x_min, x_max = x_raw.min(), x_raw.max()
x = (x_raw - x_min) / (x_max - x_min + 1e-12)

y = df["HOVAL"].values.astype(float)
y_mean = y.mean()
y_centered = y - y_mean   # center response; we'll add mean back for plotting

N = len(y)

# ----------------------------
# Build cubic B-spline basis in patsy
# ----------------------------
# Omit intercept here; model has its own intercept.
B = dmatrix(
    f"bs(x, df={SPLINE_DF}, degree={SPLINE_DEG}, include_intercept=False)",
    {"x": x},
    return_type="dataframe"
).to_numpy()

# Drop any numerically-constant columns, if present
var_cols = B.var(axis=0)
keep = ~np.isclose(var_cols, 0.0, atol=1e-12)
B = B[:, keep]
K = B.shape[1]

# Second-difference operator D2 for wiggliness penalty
def second_difference_matrix(k: int) -> np.ndarray:
    if k < 3:
        return np.zeros((0, k))
    D = np.zeros((k - 2, k))
    for i in range(k - 2):
        D[i, i]     = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D

D2 = second_difference_matrix(K)

# Torch tensors (double)
X_t = torch.tensor(B)
y_t = torch.tensor(y_centered)
D2_t = torch.tensor(D2)

# Grid for plotting
x_grid = np.linspace(0.0, 1.0, 400)
B_grid = dmatrix(
    f"bs(x, df={SPLINE_DF}, degree={SPLINE_DEG}, include_intercept=False)",
    {"x": x_grid},
    return_type="dataframe"
).to_numpy()[:, keep]  # apply same column mask

# ----------------------------
# Pyro model factory with FIXED rho (penalty scale)
# ----------------------------
def make_model_with_fixed_rho(rho_fixed: float):
    rho_fixed_t = torch.tensor(float(rho_fixed))

    def _model(X, y, D2):
        N, K = X.shape
        # Priors (scalars)
        b0 = pyro.sample("b0", dist.Normal(0.0, 10.0))
        sigma = pyro.sample("sigma", dist.HalfCauchy(2.0))

        # Vector prior on weights: K is event dimension
        w = pyro.sample("w",
                        dist.Normal(torch.zeros(K), 10.0*torch.ones(K)).to_event(1))

        # Likelihood
        mean = b0 + (X @ w)
        with pyro.plate("obs", N):
            pyro.sample("y_obs", dist.Normal(mean, sigma), obs=y)

        # Wiggliness penalty: (D2 w) ~ Normal(0, rho_fixed)
        if D2.shape[0] > 0:
            d2w = D2 @ w
            zero = torch.zeros(D2.shape[0])
            pyro.sample("d2w_pen",
                        dist.Normal(zero, rho_fixed_t).to_event(1),
                        obs=d2w)
    return _model

def fit_for_rho(rho_scale: float, num_steps: int = NUM_STEPS_PER_RHO, n_draws: int = N_DRAWS):
    pyro.clear_param_store()
    model = make_model_with_fixed_rho(rho_scale)
    guide = AutoDiagonalNormal(model)
    optim = ClippedAdam({"lr": LR, "clip_norm": 10.0})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    losses = []
    for step in range(1, num_steps + 1):
        loss = svi.step(X_t, y_t, D2_t)
        losses.append(loss / N)
        if step % 500 == 0:
            print(f"[rho={rho_scale:.3g}] step {step:4d}  loss/N={losses[-1]:.4f}")

    # Posterior means
    post = guide()
    b0 = float(post["b0"].detach())
    w  = post["w"].detach().numpy()

    # Smooth and band on grid
    f_mean_centered = B_grid @ w + b0
    f_mean = f_mean_centered + y_mean

    # Draws for bands
    f_samps = []
    for _ in range(n_draws):
        s = guide()
        b0_s = float(s["b0"].detach())
        w_s  = s["w"].detach().numpy()
        f_samps.append(B_grid @ w_s + b0_s + y_mean)
    f_samps = np.vstack(f_samps)
    f_lo, f_hi = np.percentile(f_samps, [2.5, 97.5], axis=0)

    # Derivative (marginal effect) + band
    d_mean = np.gradient(f_mean, x_grid, edge_order=2)
    d_samps = np.gradient(f_samps, x_grid, axis=1, edge_order=2)
    d_lo, d_hi = np.percentile(d_samps, [2.5, 97.5], axis=0)

    return {
        "rho": rho_scale,
        "losses": losses,
        "f_mean": f_mean,
        "f_lo": f_lo,
        "f_hi": f_hi,
        "d_mean": d_mean,
        "d_lo": d_lo,
        "d_hi": d_hi,
    }

# ----------------------------
# Fit models across penalty scales
# ----------------------------
results = [fit_for_rho(r) for r in PENALTY_SCALES]

# ----------------------------
# Build TWO-ROW Plotly with SLIDER over penalty (rho)
# ----------------------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
    row_heights=[0.6, 0.4],
    subplot_titles=("Smooth effect f(x)", "Marginal effect df/dx")
)

# Static elements (data points & zero baseline)
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers",
    name="HOVAL (data)",
    opacity=0.5,
    hovertemplate="CRIME (scaled)=%{x:.3f}<br>HOVAL=%{y:.2f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[x_grid.min(), x_grid.max()],
    y=[0, 0],
    mode="lines",
    line=dict(width=1, dash="dash"),
    name="zero baseline",
    hoverinfo="skip",
    showlegend=False
), row=2, col=1)

# For each rho, add 4 traces: f-band, f-line, d-band, d-line
trace_groups = []
for i, res in enumerate(results):
    # f-band
    t1 = go.Scatter(
        x=np.r_[x_grid, x_grid[::-1]],
        y=np.r_[res["f_hi"], res["f_lo"][::-1]],
        fill="toself",
        line=dict(width=0),
        name=f"95% band (f) — ρ={res['rho']}",
        hoverinfo="skip",
        opacity=0.28,
        visible=(i == 0),
        showlegend=False
    )
    # f-line
    t2 = go.Scatter(
        x=x_grid, y=res["f_mean"],
        mode="lines",
        name=f"Smooth f(x) — ρ={res['rho']}",
        line=dict(width=3),
        visible=(i == 0),
        hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.2f}<extra></extra>"
    )
    # d-band
    t3 = go.Scatter(
        x=np.r_[x_grid, x_grid[::-1]],
        y=np.r_[res["d_hi"], res["d_lo"][::-1]],
        fill="toself",
        line=dict(width=0),
        name=f"95% band (df/dx) — ρ={res['rho']}",
        hoverinfo="skip",
        opacity=0.28,
        visible=(i == 0),
        showlegend=False
    )
    # d-line
    t4 = go.Scatter(
        x=x_grid, y=res["d_mean"],
        mode="lines",
        name=f"df/dx — ρ={res['rho']}",
        line=dict(width=3),
        visible=(i == 0),
        hovertemplate="x=%{x:.3f}<br>df/dx=%{y:.3f}<extra></extra>"
    )

    fig.add_trace(t1, row=1, col=1)
    fig.add_trace(t2, row=1, col=1)
    fig.add_trace(t3, row=2, col=1)
    fig.add_trace(t4, row=2, col=1)
    trace_groups.append((t1, t2, t3, t4))

# Slider steps toggle visibility for each rho's 4 traces (plus keep static traces on)
n_groups = len(trace_groups)
n_dynamic_traces = 4 * n_groups
# Dynamic traces start after 2 static traces (data points, zero baseline)
static_count = 2

steps = []
for i, res in enumerate(results):
    vis = [True, True] + [False] * n_dynamic_traces
    # Turn on this rho's group
    base = static_count + 4 * i
    for j in range(4):
        vis[base + j] = True
    steps.append({
        "label": f"ρ={res['rho']}",
        "method": "update",
        "args": [{"visible": vis},
                 {"title": f"Bayesian GAM (Pyro) — COLUMBUS (CRIME → HOVAL) — ρ={res['rho']}"}]
    })

fig.update_layout(
    width=PLOT_WIDTH, height=PLOT_HEIGHT,
    template="simple_white",
    title=f"Bayesian GAM (Pyro): Smooth & Marginal Effect — COLUMBUS — ρ={results[0]['rho']}",
    xaxis_title="CRIME (scaled to [0,1])",
    yaxis_title="HOVAL",
    yaxis2_title="df/dx (per unit of scaled CRIME)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    margin=dict(l=50, r=20, t=70, b=60),
    sliders=[{
        "active": 0,
        "x": 0.1, "y": -0.06,
        "len": 0.8,
        "currentvalue": {"prefix": "Penalty scale ρ: "},
        "pad": {"t": 30, "b": 10},
        "steps": steps
    }]
)

# Save interactive HTML
html_path = os.path.join(SAVE_DIR, "columbus_gam_pyro_smooth_and_marginal_slider.html")
pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
print(f"Saved Plotly HTML → {html_path}")

