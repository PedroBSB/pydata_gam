# gam_pdp_california.py
# ----------------------
# An end-to-end example using pyGAM + partial dependence
# to understand non-linear effects in a tabular regression problem.
# This version plots using a pandas DataFrame and column NAMES (not numpy indices).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prefer the modern RMSE metric; fall back for older sklearn
try:
    from sklearn.metrics import root_mean_squared_error as rmse_metric
except Exception:
    rmse_metric = None

from pygam import LinearGAM, s

# ----------------------
# 1) Load & prep data
# ----------------------
data = fetch_california_housing(as_frame=True)

# Keep features as a DataFrame (for plotting by column name)
X_df = data.data.copy()
y = data.target  # pandas Series; median house value in $100k

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)
n_features = X_df.shape[1]
terms = s(0)
for i in range(1, n_features):
    terms += s(i)

gam = LinearGAM(terms).gridsearch(X_train_df.values, y_train.values)
gam.summary()

# ----------------------
# 3) Performance on holdout
# ----------------------
y_pred = gam.predict(X_test_df.values)

# Robust RMSE across sklearn versions
if rmse_metric is not None:
    rmse = rmse_metric(y_test.values, y_pred)
else:
    try:
        rmse = mean_squared_error(y_test.values, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))

r2 = r2_score(y_test.values, y_pred)

print(f"\nHoldout RMSE: {rmse:.4f}")
print(f"Holdout R^2 : {r2:.4f}\n")

# ----------------------
# 4) Partial dependence plots (with 95% CIs)
#    Helper to draw 1D PDP for a single feature by NAME
#    X-axis spans ONLY the dataset min/max for that feature.
# ----------------------
def plot_pdp(gam, X_ref_df: pd.DataFrame, feature: str, n_grid: int = 120):
    """
    gam: fitted pyGAM model
    X_ref_df: reference DataFrame (e.g., training data) used for min/max and medians
    feature: column name of the feature you want to plot
    """
    if feature not in X_ref_df.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(X_ref_df.columns)}")


    x_min = float(X_ref_df[feature].quantile(0.01))
    x_max = float(X_ref_df[feature].quantile(0.90))
    xg = np.linspace(x_min, x_max, n_grid)

    # Keep other features fixed at their median values (by column name)
    med = X_ref_df.median(numeric_only=True)
    XX_df = pd.DataFrame(np.tile(med.values, (n_grid, 1)), columns=med.index)

    XX_df[feature] = xg

    # Map column name
    term_idx = X_ref_df.columns.get_loc(feature)

    # Partial dependence and confidence band
    pdp, confi = gam.partial_dependence(term=term_idx, X=XX_df.values, width=0.95)

    # Draw`
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(xg, pdp, linewidth=2)
    ax.fill_between(xg, confi[:, 0], confi[:, 1], alpha=0.25, edgecolor="none")
    ax.set_title(f"Partial dependence: {feature}", fontsize=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Effect on predicted price ($100k)")  # Identity link → same units as target
    ax.set_xlim(x_min, x_max)  # clamp to dataset min/max

    # # Optional guidelines at the 5th/50th/95th percentiles
    # q5, q50, q95 = X_ref_df[feature].quantile([0.05, 0.50, 0.95])
    # for q, ls in zip([q5, q50, q95], ["--", ":", "--"]):
    #     ax.axvline(q, color="k", alpha=0.25, linestyle=ls)

    plt.tight_layout()
    return fig, ax

# Pick a few interpretable features by NAME
to_show = [
    "AveRooms",
    "MedInc",
    "HouseAge",
    "AveOccup",
]

for feat in to_show:
    plot_pdp(gam, X_train_df, feat)

plt.show()

# ----------------------
# (Optional) Save figures
# ----------------------
# for feat in to_show:
#     fig, _ = plot_pdp(gam, X_train_df, feat)
#     fig.savefig(f"pdp_{feat}.png", dpi=160, bbox_inches="tight")
