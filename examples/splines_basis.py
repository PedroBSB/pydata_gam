import os
import io  # <-- NEW: for BytesIO when building the GIF
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
os.chdir("..")

# Optional (for GIF export) — handled gracefully if missing
try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

from pygam import LinearGAM, s
from setup import *

np.random.seed(2)

# ----------------------------
# Build data + basis functions via pyGAM
# ----------------------------
N     = 500
x     = np.sort(np.random.rand(N))
X_in  = x.reshape(-1, 1)
K     = 10                                   # number of basis functions (n_splines)
order = 3                                    # cubic
draws = 20                                   # number of animation states

# Fit a tiny, stable model just to initialize the spline design
y_dummy = 1e-6 * np.random.randn(N)
gam = LinearGAM(s(0, n_splines=K, spline_order=order), lam=1.0)
gam.fit(X_in, y_dummy)

# Extract the model matrix on our x-grid (sparse -> dense)
X_design = gam._modelmat(X_in)
if hasattr(X_design, "toarray"):
    X_design = X_design.toarray()

# Drop any constant (intercept) column if present
col_vars = np.var(X_design, axis=0)
const_cols = np.where(np.isclose(col_vars, 0.0, atol=1e-12))[0]
if const_cols.size > 0:
    X_basis = np.delete(X_design, const_cols, axis=1)
else:
    X_basis = X_design

nbasis = X_basis.shape[1]
basis_cols = [f"f{i+1}" for i in range(nbasis)]

# ----------------------------
# Helpers to build animation frames
# ----------------------------
def make_draw_frame(draw_idx: int, mu=1.0, sigma=1.0):
    """Random weights per basis -> long df of basis lines + their sum (spline)."""
    beta = np.random.normal(loc=mu, scale=sigma, size=nbasis)   # ~N(1,1)
    weighted = X_basis * beta[np.newaxis, :]                    # (N, nbasis)
    spline = weighted.sum(axis=1)                               # (N,)

    df_long = pd.DataFrame({
        "x": np.tile(x, nbasis),
        "y": weighted.T.reshape(-1),
        "bf": np.repeat(basis_cols, N),
        "draw": draw_idx
    })
    df_spline = pd.DataFrame({"x": x, "spline": spline, "draw": draw_idx})
    return df_long, df_spline

# Precompute all frames
all_basis = []
all_spline = []
for d in range(1, draws + 1):
    b, s = make_draw_frame(d)
    all_basis.append(b)
    all_spline.append(s)

all_basis  = pd.concat(all_basis, ignore_index=True)
all_spline = pd.concat(all_spline, ignore_index=True)

# ----------------------------
# Build Plotly figure + frames
# ----------------------------
def frame_traces(df_b_long: pd.DataFrame, df_s_cur: pd.DataFrame):
    # One thin line per basis function (no legend, like guides(colour = FALSE))
    basis_traces = []
    for _, df_b in df_b_long.groupby("bf", sort=False):
        basis_traces.append(
            go.Scatter(
                x=df_b["x"], y=df_b["y"],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
                name="bf"
            )
        )
    # The summed spline (thicker, black-ish)
    spline_trace = go.Scatter(
        x=df_s_cur["x"], y=df_s_cur["spline"],
        mode="lines",
        line=dict(width=3, color="black"),
        name="spline",
        hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.3f}<extra></extra>"
    )
    return basis_traces + [spline_trace]

# Initial frame (draw == 1)
df_b0 = all_basis[all_basis["draw"] == 1]
df_s0 = all_spline[all_spline["draw"] == 1]
init_traces = frame_traces(df_b0, df_s0)
fig = go.Figure(data=init_traces)

# Animation frames
frames = []
for d in range(1, draws + 1):
    df_bd = all_basis[all_basis["draw"] == d]
    df_sd = all_spline[all_spline["draw"] == d]
    frames.append(
        go.Frame(
            name=str(d),
            data=frame_traces(df_bd, df_sd),
            traces=list(range(nbasis + 1))   # tell Plotly which traces update
        )
    )
fig.frames = frames

# Layout & animation controls
fig.update_layout(
    width=anim_width,
    height=anim_height,
    template="simple_white",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="x",
    yaxis_title="f(x)",
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.02, "y": 1.08,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 200}}]
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
        "currentvalue": {"prefix": "draw: "},
        "pad": {"t": 30, "b": 10},
        "steps": [
            {"label": str(d),
             "method": "animate",
             "args": [[str(d)], {"mode": "immediate",
                                 "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}]}
            for d in range(1, draws + 1)
        ]
    }]
)

# Axes ranges locked across frames (like gganimate does visually)
ymin = min(all_basis["y"].min(), all_spline["spline"].min())
ymax = max(all_basis["y"].max(), all_spline["spline"].max())
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[ymin, ymax])

# --------------------------------
# Save outputs (HTML + optional GIF)
# --------------------------------
os.makedirs("./resources", exist_ok=True)

# 1) Interactive HTML (recommended)
pio.write_html(fig, file="./resources/basis-fun-anim.html", auto_open=False, include_plotlyjs="cdn")

# 2) Static GIF (optional, requires: pip install imageio kaleido)
if not HAVE_IMAGEIO:
    print("imageio not installed → skipping GIF export. Install with: pip install imageio")
else:
    gif_path = "./resources/basis-fun-anim.gif"
    png_frames = []

    for d in range(1, draws + 1):
        df_bd = all_basis[all_basis["draw"] == d]
        df_sd = all_spline[all_spline["draw"] == d]
        f_d = go.Figure(data=frame_traces(df_bd, df_sd))
        f_d.update_layout(
            width=anim_width,
            height=anim_height,
            template="simple_white",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="x",
            yaxis_title="f(x)"
        )
        f_d.update_xaxes(range=[0, 1])
        f_d.update_yaxes(range=[ymin, ymax])

        try:
            # Needs kaleido under the hood
            png_bytes = pio.to_image(f_d, format="png", scale=2)
        except Exception as e:
            print("Kaleido not available for static export → skipping GIF.\n"
                  "Install with: pip install kaleido\n", e)
            png_frames = None
            break

        # Read PNG bytes into an array via BytesIO (no temp files)
        png_frames.append(imageio.imread(io.BytesIO(png_bytes)))

    if png_frames:
        # ~12.5 fps (0.08s per frame). Adjust as you like.
        imageio.mimsave(gif_path, png_frames, duration=0.24, loop=0)
        print("Saved GIF:", gif_path)

print("Saved HTML: ./resources/basis-fun-anim.html")
