import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
os.chdir("..")

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

from pygam import LinearGAM, s
from setup import *
np.random.seed(1)

# ----------------------------
# Helper truth function
# ----------------------------
def f_true(x):
    return x**11 * (10 * (1 - x))**6 + ((10 * (10 * x)**3) * (1 - x)**10)

# ----------------------------
# Generate data (x, ytrue, ycent, yobs)
# ----------------------------
x = np.random.rand(N)
ytrue = f_true(x)
ycent = ytrue - ytrue.mean()
yobs  = ycent + np.random.normal(scale=0.5, size=N)

data_df = pd.DataFrame({"x": x, "ytrue": ytrue, "ycent": ycent, "yobs": yobs})

# ----------------------------
# Build cubic regression spline basis via pyGAM
# ----------------------------
X_in    = data_df["x"].values.reshape(-1, 1)
y_dummy = 1e-6 * np.random.randn(N)
gam     = LinearGAM(s(0, n_splines=K, spline_order=SPL_ORDER), lam=1.0).fit(X_in, y_dummy)

X_design = gam._modelmat(X_in)  # scipy sparse
if hasattr(X_design, "toarray"):
    X_design = X_design.toarray()

# Drop any constant (intercept) column
col_vars = np.var(X_design, axis=0)
const_cols = np.where(np.isclose(col_vars, 0.0, atol=1e-12))[0]
sm2 = np.delete(X_design, const_cols, axis=1) if const_cols.size > 0 else X_design  # (N, K_eff)

K_eff   = sm2.shape[1]
bf_names = [f"F{i+1}" for i in range(K_eff)]

# Solve for weights β in ycent ~ sm2 - 1
beta, *_ = np.linalg.lstsq(sm2, data_df["ycent"].values, rcond=None)  # (K_eff,)

# ----------------------------
# Helpers
# ----------------------------
def traces_for_alpha(alpha: float):
    """
    Build traces for a tween step alpha in [0,1].
    w_j(alpha) = (1-alpha)*1 + alpha*beta_j
    """
    # Points in background
    traces = [go.Scatter(
        x=data_df["x"].values,
        y=data_df["yobs"].values,
        mode="markers",
        marker=dict(size=6),
        opacity=0.2,
        name="yobs",
        showlegend=False
    )]

    # Column-wise multiply basis by tweened weights
    w = (1.0 - alpha) + alpha * beta  # shape (K_eff,)
    basis_alpha = sm2 * w[np.newaxis, :]      # (N, K_eff)
    fitted = basis_alpha.sum(axis=1)

    # Basis lines (one per column)
    # Sort x for smooth lines
    sort_idx = np.argsort(data_df["x"].values)
    x_sorted = data_df["x"].values[sort_idx]
    basis_alpha_sorted = basis_alpha[sort_idx, :]

    for j in range(K_eff):
        traces.append(
            go.Scatter(
                x=x_sorted,
                y=basis_alpha_sorted[:, j],
                mode="lines",
                line=dict(width=1),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
                name="bf"
            )
        )

    # Black fitted line
    traces.append(
        go.Scatter(
            x=x_sorted,
            y=fitted[sort_idx],
            mode="lines",
            line=dict(width=2, color="black"),
            name="fitted",
            hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.3f}<extra></extra>"
        )
    )

    return traces

# ----------------------------
# HTML Animation with tween frames
# ----------------------------
# Precompute y-limits for stable axes
all_basis_vals_min = []
all_basis_vals_max = []
# sample a few alphas to bound y-range reasonably without computing all upfront
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    w = (1.0 - a) + a * beta
    vals = (sm2 * w[np.newaxis, :])
    all_basis_vals_min.append(vals.min())
    all_basis_vals_max.append(vals.max())
ymin = float(min(min(all_basis_vals_min), data_df["yobs"].min()))
ymax = float(max(max(all_basis_vals_max), data_df["yobs"].max()))

# Initial frame (alpha=0 -> unweighted)
fig_anim = go.Figure(data=traces_for_alpha(0.0))

# Build tween frames
alphas = np.linspace(0.0, 1.0, N_STEPS)
frames = []
for i, a in enumerate(alphas):
    frames.append(go.Frame(name=f"{i:03d}", data=traces_for_alpha(float(a))))
fig_anim.frames = frames

# Layout & controls
fig_anim.update_layout(
    width=ANIM_WIDTH,
    height=ANIM_HEIGHT,
    template="simple_white",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="x",
    yaxis_title="f(x)",
    font=dict(family="Fira Sans", size=16),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.02, "y": 1.08,
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
fig_anim.update_xaxes(range=[0, 1])
fig_anim.update_yaxes(range=[ymin, ymax])

# ----------------------------
# Save outputs (HTML + optional GIF)
# ----------------------------
os.makedirs("./resources", exist_ok=True)

anim_html_path = "./resources/gam-crs-tween-animation.html"
pio.write_html(fig_anim, file=anim_html_path, auto_open=False, include_plotlyjs="cdn")

# GIF: render each tween frame and stitch
if not HAVE_IMAGEIO:
    print("imageio not installed → skipping GIF export. Install with: pip install imageio")
else:
    gif_path = "./resources/gam-crs-tween-animation.gif"
    png_frames = []
    for i, a in enumerate(alphas):
        f = go.Figure(data=traces_for_alpha(float(a)))
        f.update_layout(
            width=ANIM_WIDTH,
            height=ANIM_HEIGHT,
            template="simple_white",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="x",
            yaxis_title="f(x)",
            font=dict(family="Fira Sans", size=16)
        )
        f.update_xaxes(range=[0, 1])
        f.update_yaxes(range=[ymin, ymax])

        try:
            # Requires kaleido: pip install -U kaleido
            png_bytes = pio.to_image(f, format="png", scale=1)
        except Exception as e:
            print("Kaleido not available for static export → skipping GIF.\n"
                  "Install with: pip install kaleido\n", e)
            png_frames = None
            break

        png_frames.append(imageio.imread(png_bytes))

    if png_frames:
        n_frames = len(png_frames)
        min_frame = 0.02  # some viewers quantize to centiseconds
        per_frame = max(TOTAL_GIF_SECONDS / max(n_frames, 1), min_frame)
        imageio.mimsave(gif_path, png_frames, duration=per_frame)
        print(f"Saved GIF (~{TOTAL_GIF_SECONDS:.1f}s): {gif_path}")

print("Saved HTML:")
print(f" - {anim_html_path}")
