import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pygam import LinearGAM, s
os.chdir("..")
# ----------------------------
# Parameters
# ----------------------------
np.random.seed(42)
N     = 200
x     = np.linspace(0, 1, N)
X_in  = x.reshape(-1, 1)

K     = 10   # number of basis functions (n_splines)
order = 3    # cubic

# ----------------------------
# Build spline basis using pyGAM
# ----------------------------
# Fit dummy model just to generate the design matrix
y_dummy = 1e-6 * np.random.randn(N)
gam = LinearGAM(s(0, n_splines=K, spline_order=order), lam=1.0).fit(X_in, y_dummy)

# Extract basis (dense matrix)
X_design = gam._modelmat(X_in)
if hasattr(X_design, "toarray"):
    X_design = X_design.toarray()

# Drop intercept if present (constant column)
col_vars = np.var(X_design, axis=0)
const_cols = np.where(np.isclose(col_vars, 0.0, atol=1e-12))[0]
if const_cols.size > 0:
    X_basis = np.delete(X_design, const_cols, axis=1)
else:
    X_basis = X_design

nbasis = X_basis.shape[1]
basis_cols = [f"f{i+1}" for i in range(nbasis)]

# Reshape to long format for plotting
basis_df = pd.DataFrame(X_basis, columns=basis_cols)
basis_df["x"] = x
basis_long = basis_df.melt(id_vars="x", var_name="bf", value_name="value")

# ----------------------------
# Plot with Plotly
# ----------------------------
fig = go.Figure()

for bf_name, df_b in basis_long.groupby("bf", sort=False):
    fig.add_trace(
        go.Scatter(
            x=df_b["x"], y=df_b["value"],
            mode="lines",
            line=dict(width=2),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip"  # no clutter on hover
        )
    )

fig.update_layout(
    width=1000,
    height=int(1000 / 1.77777777),
    template="simple_white",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="x",
    yaxis_title="b(x)",
    font=dict(family="Fira Sans", size=20)
)

# ----------------------------
# Save outputs
# ----------------------------
os.makedirs("./resources", exist_ok=True)
fig.write_html("./resources/basis-functions.html", include_plotlyjs="cdn", auto_open=False)

print("Saved static basis plot → ./resources/basis-functions.html")
