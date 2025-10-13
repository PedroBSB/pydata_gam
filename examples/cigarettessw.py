# cigs_pyro_gam_1d.py
# ---------------------------------------------------------------
# Pyro "GAM" with ONLY:
#   y = log(packs), x = log(price)
# Smooth f(x) via cubic regression spline (patsy)
# Outputs (saved to ../resources/):
#   (A) Elasticity vs price:        d log(packs) / d log(price)
#   (B) Marginal effect vs price:   d packs / d price
#   (C) 2D "%Δ price vs %Δ packs" around the median price
# ---------------------------------------------------------------

import os
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix

import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------
# 0) Reproducibility
# -----------------------------
pyro.set_rng_seed(123)
np.random.seed(123)
torch.manual_seed(123)

# -----------------------------
# 1) File setup: save in ../resources/
# -----------------------------
try:
    os.chdir("..")  # move up from examples/ if needed
except Exception:
    pass
RES_DIR = Path("resources")
RES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2) Data: AER::CigarettesSW (Rdatasets)
# -----------------------------
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CigarettesSW.csv"
df = pd.read_csv(URL)

# Keep only what we need and form logs
df = df[["packs", "price"]].dropna().reset_index(drop=True)
df["lpack"] = np.log(df["packs"].astype(float))
df["lprice"] = np.log(df["price"].astype(float))

# Targets and predictor (log scale)
y_np = df["lpack"].to_numpy()
x = df["lprice"].to_numpy()

# -----------------------------
# 3) Univariate spline basis for x = log(price)
# -----------------------------
def spline_basis(x1d, df_spline=16, degree=3):
    """Cubic regression spline basis for 1D array x1d. No intercept column."""
    mat = dmatrix(
        f"bs(x, df={df_spline}, degree={degree}, include_intercept=False) - 1",
        {"x": x1d},
        return_type="dataframe",
    )
    return np.asarray(mat, dtype=np.float64)

# Build basis at observed x
B = spline_basis(x, df_spline=16, degree=3)  # (n, K)

# Standardize spline columns for stability (intercept handled in model)
X_np = B.copy()
X_mean = X_np.mean(axis=0)
X_std  = X_np.std(axis=0) + 1e-8
X_stdzd = (X_np - X_mean) / X_std

# Torch tensors
X = torch.tensor(X_stdzd, dtype=torch.float32)        # only spline terms
y = torch.tensor(y_np, dtype=torch.float32)
n, k = X.shape  # k = number of spline columns

# -----------------------------
# 4) Pyro model: intercept unpenalized, spline weights mildly shrunk
# -----------------------------
def model(X, y=None):
    """
    y_i ~ Normal(beta0 + X_i dot beta_s, sigma)
    beta0 ~ Normal(0, 10)            # unpenalized intercept (wide prior)
    beta_s_j ~ Normal(0, tau)        # spline weights share a scale tau
    tau ~ HalfCauchy(8.0)
    sigma ~ HalfCauchy(5.0)
    """
    beta0 = pyro.sample("beta0", dist.Normal(0.0, 10.0))
    tau   = pyro.sample("tau",   dist.HalfCauchy(8.0))
    beta_s = pyro.sample("beta_s", dist.Normal(0.0, tau).expand([k]).to_event(1))
    sigma = pyro.sample("sigma", dist.HalfCauchy(5.0))

    mu = beta0 + X.matmul(beta_s)
    with pyro.plate("obs", X.shape[0]):
        pyro.sample("y", dist.Normal(mu, sigma), obs=y)

guide = AutoDiagonalNormal(model)

optimizer = Adam({"lr": 0.02})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# -----------------------------
# 5) Train
# -----------------------------
num_steps = 6000
for t in range(num_steps):
    loss = svi.step(X, y)
    if (t + 1) % 500 == 0:
        print(f"SVI step {t+1:4d} | ELBO loss = {loss:.2f}")

# Posterior sampling for averaging predictions (more stable than a single draw)
predictive = Predictive(model, guide=guide, num_samples=300)
samples = predictive(X, y)
beta0_s = samples["beta0"].detach().numpy().reshape(-1, 1)   # (S, 1)
beta_s  = samples["beta_s"].detach().numpy()                  # (S, K)

# -----------------------------
# 6) Helpers for prediction / derivatives
# -----------------------------
def design_for_x(new_x):
    """Build standardized spline design for new_x (1D array of lprice)."""
    B_new = spline_basis(new_x, df_spline=16, degree=3)  # (m, K)
    B_new = (B_new - X_mean) / X_std
    return B_new  # numpy (m, K)

def predict_logpacks(new_x):
    """Posterior-mean prediction of f(x) = E[log(packs) | lprice=x]."""
    B_new = design_for_x(new_x)                           # (m, K)
    mu_s = beta0_s + beta_s @ B_new.T                     # (S, m)
    return mu_s.mean(axis=0)                              # (m,)

def elasticity(new_x, h=5e-3):
    """
    Elasticity: d log(packs) / d log(price) at x = log(price),
    via centered finite differences on the posterior-mean f(x).
    Uses h ≈ 0.5% in log-price for numerical stability.
    """
    new_x = np.asarray(new_x)
    mu_p = predict_logpacks(new_x + h)
    mu_m = predict_logpacks(new_x - h)
    return (mu_p - mu_m) / (2.0 * h)

def marginal_effect(new_x, mu_log=None, elas=None):
    """
    Marginal effect on packs: d packs / d price
    = elasticity * (1/price) * packs_hat
    where price = exp(x), packs_hat = exp(mu_log).
    """
    x_grid = np.asarray(new_x)
    price_lvl = np.exp(x_grid)
    if mu_log is None:
        mu_log = predict_logpacks(x_grid)
    if elas is None:
        elas = elasticity(x_grid)
    packs_hat = np.exp(mu_log)
    return (elas / price_lvl) * packs_hat

# -----------------------------
# 7) Evaluate on grid, build plots
# -----------------------------
# Grid in log(price)
x_grid = np.linspace(x.min(), x.max(), 250)
mu_log = predict_logpacks(x_grid)
elas_grid = elasticity(x_grid)
me_grid = marginal_effect(x_grid, mu_log=mu_log, elas=elas_grid)

# (A) Elasticity vs price (level)
fig_el = go.Figure()
fig_el.add_trace(go.Scatter(
    x=np.exp(x_grid),
    y=elas_grid,
    mode="lines",
    name="Elasticity d log(packs) / d log(price)"
))
fig_el.add_hline(y=-1.0, line_dash="dot", annotation_text="Elasticity = -1 (unit-elastic)")
fig_el.update_layout(
    title="Cigarette Demand: Price Elasticity (Pyro 1D GAM)",
    xaxis_title="Price (level)",
    yaxis_title="Elasticity",
    template="plotly_white",
    width=900, height=500
)
pio.write_html(fig_el, file=str(RES_DIR / "cigs_1d_price_elasticity.html"),
               auto_open=False, include_plotlyjs="cdn")

# (B) Marginal effect vs price (level)
fig_me = go.Figure()
fig_me.add_trace(go.Scatter(
    x=np.exp(x_grid),
    y=me_grid,
    mode="lines",
    name="Marginal effect ∂packs/∂price"
))
fig_me.update_layout(
    title="Cigarette Demand: Marginal Effect (Pyro 1D GAM)",
    xaxis_title="Price (level)",
    yaxis_title="∂packs / ∂price",
    template="plotly_white",
    width=900, height=500
)
pio.write_html(fig_me, file=str(RES_DIR / "cigs_1d_marginal_effect.html"),
               auto_open=False, include_plotlyjs="cdn")

# (C) 2D "%Δ price vs %Δ packs" around the median price
p0_log = np.median(x)           # baseline log price
p0 = np.exp(p0_log)             # baseline price (level)
mu0 = predict_logpacks(np.array([p0_log]))[0]
packs0 = np.exp(mu0)

pct_changes_price = np.linspace(-0.30, 0.30, 121)   # -30% to +30%
price_new = p0 * (1.0 + pct_changes_price)
logprice_new = np.log(price_new)

mu_new = predict_logpacks(logprice_new)
packs_new = np.exp(mu_new)
pct_changes_packs = (packs_new - packs0) / packs0

fig_pct = go.Figure()
fig_pct.add_trace(go.Scatter(
    x=100*pct_changes_price,         # X axis in %
    y=100*pct_changes_packs,         # Y axis in %
    mode="lines",
    name="Model-implied %Δ packs vs %Δ price (around median price)"
))
elas_at_mid = elasticity(np.array([p0_log]))[0]
elas_at_mid = float(np.asarray(elas_at_mid))  # <-- cast to scalar for formatting/math
xline = np.array([-5.0, 5.0])
yline = elas_at_mid * xline
fig_pct.add_trace(go.Scatter(
    x=xline, y=yline, mode="lines", line=dict(dash="dot"),
    name=f"Tangent at 0 (slope ≈ elasticity @ median price = {elas_at_mid:.2f})"
))
fig_pct.add_hline(y=0, line_color="gray", line_dash="dot")
fig_pct.add_vline(x=0, line_color="gray", line_dash="dot")
fig_pct.update_layout(
    title="Cigarette Demand: %Δ Packs vs %Δ Price (Pyro 1D GAM, around median price)",
    xaxis_title="Δ Price (%)",
    yaxis_title="Δ Packs (%)",
    template="plotly_white",
    width=900, height=550
)
pio.write_html(fig_pct, file=str(RES_DIR / "cigs_1d_pct_change_response.html"),
               auto_open=False, include_plotlyjs="cdn")

print("Saved:", (RES_DIR / "cigs_1d_price_elasticity.html").resolve())
print("Saved:", (RES_DIR / "cigs_1d_marginal_effect.html").resolve())
print("Saved:", (RES_DIR / "cigs_1d_pct_change_response.html").resolve())

# -----------------------------
# 8) Interpretation
# -----------------------------
print("\nINTERPRETATION NOTES")
print("• Elasticity is negative and typically between 0 and -1 → inelastic demand (especially short run).")
print("• Addiction/habits lower responsiveness, but elasticity is not zero; it varies with price level.")
print("• Marginal effect (∂packs/∂price) is in units; magnitude tends to be larger where fitted packs are higher.")
print("• The %Δ plot shows nonlinearity for larger changes; the dashed tangent’s slope at 0 equals local elasticity.")
