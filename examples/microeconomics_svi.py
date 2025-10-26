#!/usr/bin/env python3
"""
Engel Curve (heteroscedastic Bayesian GAM via Pyro SVI)
- Learns smoothing strengths λ in Pyro (independent of pyGAM's lam).
- ENFORCES monotone-increasing mean via nonnegative first differences.
- Fits on standardized y for numerical stability, then back-transforms.
- Plots scatter + 95% posterior predictive band + posterior mean.

Requirements:
  pip install numpy statsmodels pygam torch pyro-ppl matplotlib
"""

import os
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from pygam import LinearGAM, s


# ---------------------- Data & utils ----------------------
def load_engel():
    engel = sm.datasets.engel.load_pandas().data.rename(
        columns={"income": "income", "foodexp": "food"}
    )
    x = engel["income"].to_numpy(dtype=float)
    y = engel["food"].to_numpy(dtype=float)
    return x, y


def standardize(arr: np.ndarray):
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd == 0.0:
        return np.zeros_like(arr, dtype=float), mu, 1.0
    return (arr - mu) / sd, mu, sd


def build_pygam_basis_and_terms(xz: np.ndarray, n_splines: int):
    """
    Build B-spline design matrix B and base penalty S using pyGAM WITHOUT fitting.
    We set lam=1.0 so S is unit-scaled; λ is learned in Pyro.
    """
    X = xz.reshape(-1, 1)
    term = s(0, n_splines=n_splines, constraints=None, lam=1.0)
    gam = LinearGAM(term, fit_intercept=False)
    gam.terms.compile(X)

    B = gam.terms.build_columns(X).toarray().astype(np.float32)   # (n, m)
    S = gam.terms.build_penalties().toarray().astype(np.float32)  # (m, m)
    return B, S, gam.terms


def make_L_firstdiff_matrix(m: int, dtype=torch.float32):
    """
    Construct L (m x (m-1)) so that theta = L @ delta has first differences delta:
        theta_0 = 0
        theta_i = sum_{k=1}^i delta_k, i=1..m-1
    If delta >= 0, then theta has nonnegative first differences ⇒ monotone μ(x).
    """
    L = torch.zeros((m, m - 1), dtype=dtype)
    if m > 1:
        # fill strict lower-triangular with ones
        L[1:, :] = torch.tril(torch.ones((m - 1, m - 1), dtype=dtype))
    return L


# ---------------------- Pyro model ----------------------
def make_model(B_mu_t, S_mu_t, B_sig_t, S_sig_t, L_t, y_z_t=None):
    """
    Standardized target y_z. Mean and noise:
      θ = L @ softplus(η)  (monotone-increasing)
      μ_z(x) = β0 + B_mu θ
      σ_z(x) = softplus(ρ0 + B_sig γ) + 1e-6

    Priors (matched to standardized scale):
      β0 ~ N(0, 5),  η ~ N(0,1)^(m-1),  γ ~ N(0,1)^(m_sig)
      λμ, λσ ~ Gamma(2,1)
    Quadratic smoothness penalties via pyro.factor:
      -0.5 * λμ * θᵀ Sμ θ,  -0.5 * λσ * γᵀ Sσ γ
    """
    n, m_mu = B_mu_t.shape
    m_sig = B_sig_t.shape[1]
    assert L_t.shape == (m_mu, m_mu - 1)

    def model():
        beta0 = pyro.sample("beta0", dist.Normal(0.0, 5.0))

        # Monotone θ via positive first differences
        eta = pyro.sample("eta", dist.Normal(torch.zeros(m_mu - 1, dtype=B_mu_t.dtype),
                                             torch.ones(m_mu - 1, dtype=B_mu_t.dtype)).to_event(1))
        delta = F.softplus(eta)                           # (m-1,)  >= 0
        theta = L_t @ delta                               # (m,)

        lam_mu = pyro.sample("lam_mu", dist.Gamma(2.0, 1.0))
        quad_mu = 0.5 * lam_mu * torch.dot(theta, S_mu_t @ theta)
        pyro.factor("smooth_penalty_mu", -quad_mu)

        rho0 = pyro.sample("rho0", dist.Normal(0.0, 1.0))
        gamma = pyro.sample("gamma", dist.Normal(torch.zeros(m_sig, dtype=B_sig_t.dtype),
                                                 torch.ones(m_sig, dtype=B_sig_t.dtype)).to_event(1))
        lam_sig = pyro.sample("lam_sig", dist.Gamma(2.0, 1.0))
        quad_sig = 0.5 * lam_sig * torch.dot(gamma, S_sig_t @ gamma)
        pyro.factor("smooth_penalty_sig", -quad_sig)

        mu = beta0 + B_mu_t @ theta
        log_sigma = rho0 + B_sig_t @ gamma
        sigma = F.softplus(log_sigma) + 1e-6

        with pyro.plate("data", n):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y_z_t)

    return model


# ---------------------- Posterior sampling ----------------------
def draw_latent_samples(guide, num_samples=500):
    """
    Draw samples from the AutoNormal guide and return a dict of tensors.
    """
    out = {k: [] for k in ["beta0", "eta", "rho0", "gamma"]}
    with torch.no_grad():
        for _ in range(num_samples):
            tr = pyro.poutine.trace(guide).get_trace()
            for k in out:
                out[k].append(tr.nodes[k]["value"].detach())
    for k in out:
        out[k] = torch.stack(out[k], dim=0)
    return out


# ---------------------- Train, predict, plot ----------------------
def main():
    pyro.set_rng_seed(42)
    torch.set_default_dtype(torch.float32)

    # Data
    x_raw, y_raw = load_engel()
    x_z, x_mu, x_sd = standardize(x_raw)
    y_z, y_mu, y_sd = standardize(y_raw)

    # Bases (unit penalty scale from pyGAM)
    n_splines_mu = 10
    n_splines_sig = 6
    B_mu, S_mu, terms_mu = build_pygam_basis_and_terms(x_z, n_splines_mu)
    B_sig, S_sig, terms_sig = build_pygam_basis_and_terms(x_z, n_splines_sig)

    # Tensors
    y_z_t = torch.from_numpy(y_z.astype(np.float32))
    B_mu_t = torch.from_numpy(B_mu)
    S_mu_t = torch.from_numpy(S_mu + np.eye(S_mu.shape[0], dtype=np.float32) * 1e-6)
    B_sig_t = torch.from_numpy(B_sig)
    S_sig_t = torch.from_numpy(S_sig + np.eye(S_sig.shape[0], dtype=np.float32) * 1e-6)

    # Monotone transform matrix (θ = L @ softplus(η))
    L_t = make_L_firstdiff_matrix(B_mu_t.shape[1], dtype=B_mu_t.dtype)

    # Model / guide / SVI
    model = make_model(B_mu_t, S_mu_t, B_sig_t, S_sig_t, L_t, y_z_t)
    guide = AutoNormal(model)
    svi = SVI(model, guide, ClippedAdam({"lr": 0.02, "clip_norm": 10.0}), loss=Trace_ELBO())

    for step in range(1, 2501):
        loss = svi.step()
        if step % 250 == 0 or step <= 5:
            print(f"[step {step:4d}] ELBO loss: {loss:.3f}")

    # Grid for plotting
    x_grid = np.linspace(x_raw.min(), x_raw.max(), 400).astype(np.float32)
    xg_z = (x_grid - x_mu) / x_sd
    Xg = xg_z.reshape(-1, 1)
    B_mu_g = terms_mu.build_columns(Xg).toarray().astype(np.float32)
    B_sig_g = terms_sig.build_columns(Xg).toarray().astype(np.float32)
    B_mu_g_t = torch.from_numpy(B_mu_g)
    B_sig_g_t = torch.from_numpy(B_sig_g)

    # Posterior draws
    S = 800
    lat = draw_latent_samples(guide, num_samples=S)

    # Build θ draws from η via monotone transform
    # delta = softplus(eta): (S, m-1) -> θ: (S, m)
    delta_draws = F.softplus(lat["eta"])
    theta_draws = delta_draws @ L_t.T                             # (S, m)

    # μ_z(x) and σ_z(x) on grid, then back-transform to original y-scale
    mu_z_draws = (B_mu_g_t @ theta_draws.T) + lat["beta0"]        # (G, S)
    log_sigma_z_draws = (B_sig_g_t @ lat["gamma"].T) + lat["rho0"]# (G, S)
    sigma_z_draws = F.softplus(log_sigma_z_draws) + 1e-6

    with torch.no_grad():
        y_pred_z_draws = mu_z_draws + torch.randn_like(mu_z_draws) * sigma_z_draws
        y_pred_draws = y_pred_z_draws * y_sd + y_mu
        mu_draws = mu_z_draws * y_sd + y_mu

        lower = torch.quantile(y_pred_draws, 0.025, dim=1)
        upper = torch.quantile(y_pred_draws, 0.975, dim=1)
        mu_mean = mu_draws.mean(dim=1)

    # Plot
    os.makedirs("./resources", exist_ok=True)
    out_path = "./resources/engel_pyro_monotone_posterior_predictive.png"
    plt.figure(figsize=(8, 5.5))
    plt.scatter(x_raw, y_raw, s=16, alpha=0.55, label="Data")
    plt.fill_between(x_grid, lower.numpy(), upper.numpy(), alpha=0.25, label="95% posterior predictive")
    plt.plot(x_grid, mu_mean.numpy(), lw=2.5, label="Posterior mean μ(x) (monotone ↑)")
    plt.xlabel("Household income")
    plt.ylabel("Food expenditure")
    plt.title("Engel Curve • Heteroscedastic Bayesian GAM (Pyro, monotone ↑)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()
    print(f"Saved figure: {out_path}")

    # In-sample diagnostics (back-transformed)
    mu_train_z_draws = (B_mu_t @ theta_draws.T) + lat["beta0"]       # (n, S)
    mu_train_mean = mu_train_z_draws.mean(dim=1) * y_sd + y_mu
    y_t = torch.from_numpy(y_raw.astype(np.float32))
    mse = torch.mean((mu_train_mean - y_t) ** 2).item()
    print(f"In-sample MSE (μ back-transformed): {mse:.4f}")

    # Coverage
    log_sigma_train_z_draws = (B_sig_t @ lat["gamma"].T) + lat["rho0"]
    sigma_train_z_draws = F.softplus(log_sigma_train_z_draws) + 1e-6
    y_pred_train_draws = (mu_train_z_draws + torch.randn_like(mu_train_z_draws) * sigma_train_z_draws) * y_sd + y_mu
    lower_train = torch.quantile(y_pred_train_draws, 0.025, dim=1)
    upper_train = torch.quantile(y_pred_train_draws, 0.975, dim=1)
    coverage = ((y_t >= lower_train) & (y_t <= upper_train)).float().mean().item()
    print(f"In-sample 95% predictive coverage: {coverage:.3f}")


if __name__ == "__main__":
    main()
