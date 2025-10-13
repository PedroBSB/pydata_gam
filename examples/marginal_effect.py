# Smooth effect + Marginal Effect (derivative) for a Bayesian GAM-style fit
# using PyGAM basis (design matrix) + Pyro Bayesian linear regression on basis
# + Plotly animation. Real data: Engel curve (food expenditure vs income).
#
# Assumes a companion `setup.py` provides:
# N, K, SPL_ORDER, N_STEPS, FRAME_DURATION_MS, TRANSITION_MS, ANIM_WIDTH, ANIM_HEIGHT, TOTAL_GIF_SECONDS
#
# Notes:
# - Basis built on scaled x in [0,1] for numerical stability.
# - Derivative df/dx reported w.r.t. RAW (original) income units.

import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

os.chdir("..")

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

import statsmodels.api as sm
from pygam import LinearGAM, s
from setup import *

# ----------------------------
# Pyro / Torch
# ----------------------------
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal

# For reproducibility
np.random.seed(1)
pyro.set_rng_seed(1)
torch.set_default_dtype(torch.float64)

# ----------------------------
# Load real micro data: Engel curve (food exp vs income)
# ----------------------------
engel = sm.datasets.engel.load_pandas().data.rename(
    columns={"income": "income", "foodexp": "food"}
)

# Use raw x (income) and y (food expenditure) in original currency units
x_raw = engel["income"].to_numpy(dtype=float)
y_raw = engel["food"].to_numpy(dtype=float)

# Optional outlier clipping (commented out):
# q1, q99 = np.quantile(x_raw, [0.01, 0.99])
# keep = (x_raw >= q1) & (x_raw <= q99)
# x_raw = x_raw[keep]; y_raw = y_raw[keep]

# We *scale* income to [0,1] only to build the pyGAM spline basis robustly.
xmin, xmax = float(x_raw.min()), float(x_raw.max())
x_scaled = (x_raw - xmin) / (xmax - xmin + 1e-12)

# We'll work with a demeaned target for visualization parity with your script,
# but fit the Bayesian model on the (possibly demeaned) target.
ycent = y_raw - y_raw.mean()
data_df = pd.DataFrame({"x_scaled": x_scaled, "x_raw": x_raw, "y": y_raw, "ycent": ycent})

# ----------------------------
# Build cubic regression spline basis via pyGAM on *scaled* x
# ----------------------------
X_in = data_df["x_scaled"].values.reshape(-1, 1)
# Fit a GAM only to extract the smooth basis; use near-zero y to avoid overfitting
y_dummy = 1e-6 * np.random.randn(len(data_df))
gam = LinearGAM(s(0, n_splines=K, spline_order=SPL_ORDER), lam=1.0).fit(X_in, y_dummy)

X_design = gam._modelmat(X_in)  # scipy sparse
if hasattr(X_design, "toarray"):
    X_design = X_design.toarray()

# Drop any constant (intercept) column (we'll include mean via ycent handling as needed)
col_vars = np.var(X_design, axis=0)
const_cols = np.where(np.isclose(col_vars, 0.0, atol=1e-12))[0]
sm2 = np.delete(X_design, const_cols, axis=1) if const_cols.size > 0 else X_design  # (N, K_eff)

K_eff   = sm2.shape[1]
bf_names = [f"F{i+1}" for i in range(K_eff)]

# Common sorted index for smooth lines & derivative
sort_idx     = np.argsort(data_df["x_scaled"].values)
x_sorted_scl = data_df["x_scaled"].values[sort_idx]
x_sorted_raw = data_df["x_raw"].values[sort_idx]

# ----------------------------
# Bayesian linear model on the spline basis using Pyro
# Model: ycent = sm2 @ beta + eps, beta ~ N(0, tau^2), sigma ~ HalfCauchy(1)
# ----------------------------
X_np = sm2
y_np = data_df["ycent"].values

X_t = torch.from_numpy(X_np)                    # (N, K_eff)
y_t = torch.from_numpy(y_np)                    # (N,)

def model(X, y):
    N, K = X.shape
    # Priors
    tau = pyro.sample("tau", dist.LogNormal(0.0, 0.5))        # sd for weights
    sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))        # noise sd
    w = pyro.sample("w", dist.Normal(torch.zeros(K), tau*torch.ones(K)).to_event(1))
    mu = X @ w
    with pyro.plate("data", N):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

guide = AutoNormal(model)

optim = ClippedAdam({"lr": 0.02, "clip_norm": 5.0})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_steps = 5000
for step in range(num_steps):
    loss = svi.step(X_t, y_t)
    if step % 500 == 0:
        pass  # you can print(loss) if you like

# Extract posterior means (you could also draw samples if you want bands)
posterior_median = {k: v.detach().cpu().numpy() for k, v in pyro.get_param_store().items()}
# AutoNormal stores loc/scale params; use guide.median for a clean point estimate
beta_map = guide.median({"X": X_t, "y": y_t})["w"].detach().cpu().numpy()  # (K_eff,)

# ----------------------------
# Tween setup: from a neutral vector to posterior beta
# We'll tween from "all ones" (neutral) to beta_map like your previous script
# ----------------------------
beta_target = beta_map
beta_start  = np.ones_like(beta_target)

# ----------------------------
# Helpers
# ----------------------------
def _fitted_and_derivative(alpha: float):
    """
    Returns:
      fitted_sorted: f(x_raw) at sorted x (on original y scale after adding back the mean)
      dfdx_sorted  : df/dx_raw at sorted x (np.gradient over raw income)
      basis_alpha_sorted: (N, K_eff) individual basis*weight curves at sorted x (top panel)
    """
    # tween weights: w_j(alpha) = (1-alpha)*beta_start + alpha*beta_target_j
    w = (1.0 - alpha) * beta_start + alpha * beta_target  # (K_eff,)
    basis_alpha = sm2 * w[np.newaxis, :]                  # (N, K_eff)
    fitted_centered = basis_alpha.sum(axis=1)             # centered fit
    fitted = fitted_centered + y_raw.mean()               # back to original y scale

    basis_alpha_sorted = basis_alpha[sort_idx, :]
    fitted_sorted = fitted[sort_idx]

    # Derivative via numerical gradient wrt RAW income (handles uneven spacing)
    dfdx_sorted = np.gradient(fitted_sorted, x_sorted_raw, edge_order=2)

    return fitted_sorted, dfdx_sorted, basis_alpha_sorted

def traces_for_alpha(alpha: float):
    """
    Build a FLAT list of traces in a fixed order for animation frames.
    Subplot mapping:
      - First panel (row=1): points, K_eff basis lines, black fitted line
      - Second panel (row=2): derivative line and a horizontal 0 baseline
    """
    # Observed points (top panel)
    traces = [go.Scatter(
        x=data_df["x_raw"].values,
        y=data_df["y"].values,
        mode="markers",
        marker=dict(size=6),
        opacity=0.25,
        name="observed",
        showlegend=False
    )]

    fitted_sorted, dfdx_sorted, basis_alpha_sorted = _fitted_and_derivative(alpha)

    # Basis lines (top panel)
    for j in range(K_eff):
        traces.append(
            go.Scatter(
                x=x_sorted_raw,
                y=basis_alpha_sorted[:, j] + y_raw.mean(),  # plot on original y scale
                mode="lines",
                line=dict(width=1),
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
                name=bf_names[j] if j < len(bf_names) else "bf"
            )
        )

    # Fitted line (top panel)
    traces.append(
        go.Scatter(
            x=x_sorted_raw,
            y=fitted_sorted,
            mode="lines",
            line=dict(width=2, color="black"),
            name="fitted",
            hovertemplate="Income=%{x:.2f}<br>Food exp=%{y:.2f}<extra></extra>"
        )
    )

    # Derivative df/dx (bottom panel) in original units: $food per $income
    traces.append(
        go.Scatter(
            x=x_sorted_raw,
            y=dfdx_sorted,
            mode="lines",
            line=dict(width=2),
            name="df/d(income)",
            hovertemplate="Income=%{x:.2f}<br>Marginal budget share=%{y:.4f}<extra></extra>"
        )
    )

    # Zero baseline (bottom panel)
    traces.append(
        go.Scatter(
            x=[x_sorted_raw.min(), x_sorted_raw.max()],
            y=[0, 0],
            mode="lines",
            line=dict(width=1, dash="dash"),
            name="zero",
            hoverinfo="skip",
            showlegend=False
        )
    )
    return traces

# ----------------------------
# Axis ranges for stable view
# ----------------------------
# sample a few alphas to bound value & derivative ranges
all_vals_min = []
all_vals_max = []
all_dmins = []
all_dmaxs = []

for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    f_sorted, d_sorted, basis_alpha_sorted = _fitted_and_derivative(a)
    all_vals_min.append(min(f_sorted.min(), (basis_alpha_sorted + y_raw.mean()).min(), data_df["y"].min()))
    all_vals_max.append(max(f_sorted.max(), (basis_alpha_sorted + y_raw.mean()).max(), data_df["y"].max()))
    all_dmins.append(d_sorted.min())
    all_dmaxs.append(d_sorted.max())

ymin = float(min(all_vals_min))
ymax = float(max(all_vals_max))
dymin = float(min(all_dmins))
dymax = float(max(all_dmaxs))
# pad derivative range a bit for the zero baseline
pad = 0.05 * (dymax - dymin + 1e-9)
dymin -= pad
dymax += pad

# ----------------------------
# Build figure with 2 rows (shared x)
# ----------------------------
fig_anim = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
    row_heights=[0.62, 0.38],
    subplot_titles=("Engel curve: Food expenditure vs Income", "Marginal budget share: d(Food)/d(Income)")
)

# Initial traces for alpha=0.0 in a fixed order
initial_traces = traces_for_alpha(0.0)
# Map: [0] points (row=1), [1..K_eff] basis (row=1), [K_eff+1] fitted (row=1),
#      [K_eff+2] derivative (row=2), [K_eff+3] zero baseline (row=2)
fig_anim.add_trace(initial_traces[0], row=1, col=1)
for j in range(1, 1 + K_eff):
    fig_anim.add_trace(initial_traces[j], row=1, col=1)
fig_anim.add_trace(initial_traces[1 + K_eff], row=1, col=1)
fig_anim.add_trace(initial_traces[2 + K_eff], row=2, col=1)
fig_anim.add_trace(initial_traces[3 + K_eff], row=2, col=1)

# Build tween frames (each with the same number/order of traces)
alphas = np.linspace(0.0, 1.0, N_STEPS)
frames = []
for i, a in enumerate(alphas):
    frames.append(go.Frame(name=f"{i:03d}", data=traces_for_alpha(float(a))))
fig_anim.frames = frames

# Layout & controls
fig_anim.update_layout(
    width=ANIM_WIDTH,
    height=int(ANIM_HEIGHT * 1.4),
    template="simple_white",
    margin=dict(l=50, r=20, t=80, b=60),
    xaxis_title="Household income",
    yaxis_title="Food expenditure",
    yaxis2_title="Marginal budget share  d(Food)/d(Income)",
    font=dict(family="Fira Sans", size=16),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.02, "y": 1.14,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": FRAME_DURATION_MS, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": TRANSITION_MS, "easing": "cubic-in-out"}}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"mode": "immediate",
                                  "frame": {"duration": 0, "redraw": False}}]
            }
        ]
    }],
    sliders=[{
        "active": 0,
        "x": 0.1, "y": -0.07,
        "len": 0.8,
        "currentvalue": {"prefix": "alpha: "},
        "pad": {"t": 30, "b": 10},
        "steps": [
            {"label": f"{a:.2f}",
             "method": "animate",
             "args": [[f"{i:03d}"], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}]}
            for i, a in enumerate(alphas)
        ]
    }]
)

# Axis ranges
fig_anim.update_xaxes(range=[float(x_raw.min()), float(x_raw.max())], row=1, col=1)
fig_anim.update_xaxes(range=[float(x_raw.min()), float(x_raw.max())], row=2, col=1)
fig_anim.update_yaxes(range=[ymin, ymax], row=1, col=1)
fig_anim.update_yaxes(range=[dymin, dymax], row=2, col=1)

# ----------------------------
# Save outputs (HTML + optional GIF)
# ----------------------------
os.makedirs("./resources", exist_ok=True)

anim_html_path = "./resources/engel-gam-pyro-tween-with-derivative.html"
pio.write_html(fig_anim, file=anim_html_path, auto_open=False, include_plotlyjs="cdn")

# GIF: render each tween frame and stitch
if not HAVE_IMAGEIO:
    print("imageio not installed → skipping GIF export. Install with: pip install imageio")
else:
    gif_path = "./resources/engel-gam-pyro-tween-with-derivative.gif"
    png_frames = []
    for i, a in enumerate(alphas):
        # rebuild a per-frame subplot figure so the GIF shows both panels
        f = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            row_heights=[0.62, 0.38]
        )
        tr = traces_for_alpha(float(a))
        f.add_trace(tr[0], row=1, col=1)
        for j in range(1, 1 + K_eff):
            f.add_trace(tr[j], row=1, col=1)
        f.add_trace(tr[1 + K_eff], row=1, col=1)
        f.add_trace(tr[2 + K_eff], row=2, col=1)
        f.add_trace(tr[3 + K_eff], row=2, col=1)

        f.update_layout(
            width=ANIM_WIDTH,
            height=int(ANIM_HEIGHT * 1.4),
            template="simple_white",
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis_title="Household income",
            yaxis_title="Food expenditure",
            yaxis2_title="Marginal budget share  d(Food)/d(Income)",
            font=dict(family="Fira Sans", size=16)
        )
        f.update_xaxes(range=[float(x_raw.min()), float(x_raw.max())], row=1, col=1)
        f.update_xaxes(range=[float(x_raw.min()), float(x_raw.max())], row=2, col=1)
        f.update_yaxes(range=[ymin, ymax], row=1, col=1)
        f.update_yaxes(range=[dymin, dymax], row=2, col=1)

        try:
            # Requires kaleido: pip install -U kaleido
            png_bytes = pio.to_image(f, format="png", scale=1)
        except Exception as e:
            print("Kaleido not available for static export → skipping GIF.\n"
                  "Install with: pip install kaleido\n", e)
            png_frames = None
            break

        # imageio expects a file, URI, or file-like object; wrap bytes in BytesIO
        png_frames.append(imageio.imread(io.BytesIO(png_bytes)))

    if png_frames:
        n_frames = len(png_frames)
        min_frame = 0.02  # some viewers quantize to centiseconds
        per_frame = max(TOTAL_GIF_SECONDS / max(n_frames, 1), min_frame)
        imageio.mimsave(gif_path, png_frames, duration=per_frame)
        print(f"Saved GIF (~{TOTAL_GIF_SECONDS:.1f}s): {gif_path}")

print("Saved HTML:")
print(f" - {anim_html_path}")
