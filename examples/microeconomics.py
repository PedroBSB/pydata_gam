#!/usr/bin/env python3
"""
Engel Curve (original scale): PyGAM monotone-increasing smooth + 95% band
+ marginal effect (derivative) plot via numerical differentiation
+ HTML with a slider (alpha in [0,5]) to control extra wiggliness/smoothing.

- Training code and derivative method are UNCHANGED.
- The slider only post-processes the plotted curves in the browser.

Requires: numpy matplotlib statsmodels pygam
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import statsmodels.api as sm
from pygam import LinearGAM, s


# ---------------------- Data ----------------------
def load_engel():
    engel = sm.datasets.engel.load_pandas().data.rename(
        columns={"income": "income", "foodexp": "food"}
    )
    x = engel["income"].to_numpy(dtype=float)
    y = engel["food"].to_numpy(dtype=float)
    return x, y


# ---------------------- Helpers ----------------------
def standardize(arr: np.ndarray):
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd == 0.0:
        # avoid divide-by-zero; return zeros and identity transform
        return np.zeros_like(arr, dtype=float), mu, 1.0
    return (arr - mu) / sd, mu, sd


# ---------------------- Fit PyGAM (monotone, original y) -----------
def fit_pygam_monotone(x, y, n_splines: int = 10):
    """
    - Standardize x for numerical stability, keep y on original scale.
    - Enforce monotone-increasing smooth.
    - Use strong smoothing grid to avoid edge blow-ups.
    """
    xz, x_mu, x_sd = standardize(x)
    X = xz.reshape(-1, 1)

    lam_grid = np.logspace(1, 7, 25)  # strong regularization
    gam = LinearGAM(
        s(0, n_splines=n_splines, constraints="monotonic_inc")
    ).gridsearch(X, y, lam=lam_grid, keep_best=True, progress=False)

    return gam, (x_mu, x_sd)


# ---------------------- Main ----------------------
def main():
    # Data
    x_raw, y_raw = load_engel()

    # Fit monotone GAM (UNCHANGED)
    gam, (x_mu, x_sd) = fit_pygam_monotone(x_raw, y_raw, n_splines=10)

    # Prediction grid (transform with same scaling used for training)
    x_grid = np.linspace(x_raw.min(), x_raw.max(), 400)
    xg_z = (x_grid - x_mu) / x_sd
    Xg = xg_z.reshape(-1, 1)

    # Mean prediction and 95% confidence band for the mean function (UNCHANGED)
    y_hat = gam.predict(Xg)
    ci = gam.confidence_intervals(Xg, width=0.95)  # (n, 2): [lower, upper]
    lo, hi = ci[:, 0], ci[:, 1]

    # Numerical derivative on ORIGINAL x (UNCHANGED baseline, used at alpha=0)
    d_mu = np.gradient(y_hat, x_grid)
    d_lo = np.gradient(lo, x_grid)
    d_hi = np.gradient(hi, x_grid)

    # ---------------------- HTML export with interactive slider ----------------------
    os.makedirs("./resources", exist_ok=True)
    out_path = "./resources/engel-gam-monotone-slider.html"

    # Serialize arrays for JS
    payload = {
        "x_raw": x_raw.tolist(),
        "y_raw": y_raw.tolist(),
        "x_grid": x_grid.tolist(),
        "y_hat": y_hat.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "d_mu": d_mu.tolist(),
        "d_lo": d_lo.tolist(),
        "d_hi": d_hi.tolist(),
    }
    data_json = json.dumps(payload)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Engel Curve • PyGAM Monotone + Slider</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #fafafa; --card: #fff; --fg: #111; --muted: #666;
    --accent: #2b6cb0;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif;
    margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
  }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .row {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{
    background: var(--card); border-radius: 12px; padding: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.07);
  }}
  .controls {{
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 8px 12px; border-radius: 10px; background: #f1f5f9; margin: 8px 0 14px;
  }}
  .controls label {{ font-weight: 600; }}
  .value-badge {{
    font-variant-numeric: tabular-nums;
    background: #e2e8f0; padding: 2px 8px; border-radius: 8px; color: #222;
  }}
  input[type="range"] {{
    width: 240px;
    accent-color: var(--accent);
  }}
  .hint {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>Engel Curve (PyGAM monotone) — wiggliness slider</h1>

  <div class="card">
    <div class="controls">
      <label for="alpha">extra smoothing α</label>
     <input id="alpha" type="range" min="0" max="1000" step="0.1" value="0" />
      <span class="value-badge" id="alphaVal">0.0</span>
      <span class="hint">α = 0 shows the raw fitted curve; increasing α applies extra smoothing (less wiggle). No retraining.</span>
    </div>
    <div id="top" style="width: 100%; height: 460px;"></div>
  </div>

  <div class="card">
    <div id="bottom" style="width: 100%; height: 360px;"></div>
  </div>

<script>
const DATA = {data_json};

// Simple Gaussian-like smoothing kernel (width grows with alpha).
// alpha in [0,5] → sigma_pts in [0, ~25]. alpha=0 => no smoothing.
function makeKernel(alpha) {{
  const sigma = alpha * 5.0;       // points
  if (sigma < 1e-6) return [1.0];  // no smoothing
  const radius = Math.max(1, Math.round(3 * sigma));
  const k = [];
  let sum = 0.0;
  for (let i = -radius; i <= radius; i++) {{
    const w = Math.exp(-0.5 * (i*i) / (sigma*sigma));
    k.push(w); sum += w;
  }}
  return k.map(w => w / sum);
}}

// Convolution with edge extension padding
function smooth1d(arr, kernel) {{
  if (kernel.length === 1) return arr.slice();
  const n = arr.length;
  const r = Math.floor(kernel.length / 2);
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {{
    let acc = 0.0;
    for (let j = -r; j <= r; j++) {{
      let idx = i + j;
      if (idx < 0) idx = 0;
      if (idx >= n) idx = n - 1;
      acc += arr[idx] * kernel[j + r];
    }}
    out[i] = acc;
  }}
  return out;
}}

// Numerical derivative using central differences on possibly smoothed signal
function gradient(y, x) {{
  const n = y.length;
  const g = new Array(n).fill(0);
  if (n === 1) return g;
  g[0] = (y[1] - y[0]) / (x[1] - x[0]);
  for (let i = 1; i < n-1; i++) {{
    const dx = x[i+1] - x[i-1];
    g[i] = (y[i+1] - y[i-1]) / dx;
  }}
  g[n-1] = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
  return g;
}}

function computeSeries(alpha) {{
  const k = makeKernel(alpha);
  const y = smooth1d(DATA.y_hat, k);
  const lo = smooth1d(DATA.lo, k);
  const hi = smooth1d(DATA.hi, k);

  // Ensure ordering after smoothing
  for (let i=0; i<y.length; i++) {{
    if (lo[i] > hi[i]) {{ const tmp = lo[i]; lo[i] = hi[i]; hi[i] = tmp; }}
  }}

  const dmu = gradient(y, DATA.x_grid);
  const dlo = gradient(lo, DATA.x_grid);
  const dhi = gradient(hi, DATA.x_grid);

  return {{ y, lo, hi, dmu, dlo, dhi }};
}}

function makeTopFig(alphaSeries) {{
  return {{
    data: [
      {{
        x: DATA.x_raw, y: DATA.y_raw, type: 'scatter', mode: 'markers',
        name: 'Data', opacity: 0.55, marker: {{ size: 6 }}
      }},
      {{
        x: DATA.x_grid, y: alphaSeries.y, type: 'scatter', mode: 'lines',
        name: 'PyGAM fit (monotone ↑)', line: {{ width: 3 }}
      }},
      {{
        x: [...DATA.x_grid, ...DATA.x_grid.slice().reverse()],
        y: [...alphaSeries.lo, ...alphaSeries.hi.slice().reverse()],
        fill: 'toself', fillcolor: 'rgba(31,119,180,0.20)',
        line: {{ width: 0 }}, hoverinfo: 'skip',
        name: '95% CI (mean)'
      }},
    ],
    layout: {{
      margin: {{ l: 60, r: 20, t: 20, b: 50 }},
      template: 'simple_white',
      xaxis: {{ title: 'Household income' }},
      yaxis: {{ title: 'Food expenditure' }},
      legend: {{ orientation: 'h', y: 1.12 }}
    }}
  }};
}}

function makeBottomFig(alphaSeries) {{
  return {{
    data: [
      {{
        x: DATA.x_grid, y: alphaSeries.dmu, type: 'scatter', mode: 'lines',
        name: "Marginal effect f'(x)", line: {{ width: 2, dash: 'dash' }}
      }},
      {{
        x: [...DATA.x_grid, ...DATA.x_grid.slice().reverse()],
        y: [...alphaSeries.dlo, ...alphaSeries.dhi.slice().reverse()],
        fill: 'toself', fillcolor: 'rgba(31,119,180,0.20)',
        line: {{ width: 0 }}, hoverinfo: 'skip',
        name: "Approx. 95% band for f'(x)"
      }},
      {{
        x: [DATA.x_grid[0], DATA.x_grid[DATA.x_grid.length-1]],
        y: [0,0], type: 'scatter', mode: 'lines',
        name: 'zero', line: {{ width: 1, dash: 'dot' }},
        hoverinfo: 'skip', showlegend: false
      }}
    ],
    layout: {{
      margin: {{ l: 60, r: 20, t: 10, b: 50 }},
      template: 'simple_white',
      xaxis: {{ title: 'Household income' }},
      yaxis: {{ title: 'Marginal effect  df/dx' }},
      showlegend: true,
      legend: {{ orientation: 'h', y: 1.10 }}
    }}
  }};
}}

function init() {{
  const alphaInput = document.getElementById('alpha');
  const alphaVal = document.getElementById('alphaVal');

  function update(alpha) {{
    alphaVal.textContent = (+alpha).toFixed(1);
    const series = computeSeries(+alpha);

    // Update plots
    const top = document.getElementById('top');
    const bottom = document.getElementById('bottom');

    if (!top.hasChildNodes()) {{
      const topFig = makeTopFig(series);
      Plotly.newPlot(top, topFig.data, topFig.layout, {{displayModeBar: false}});
      const botFig = makeBottomFig(series);
      Plotly.newPlot(bottom, botFig.data, botFig.layout, {{displayModeBar: false}});
    }} else {{
      // top traces: 0 data, 1 line, 2 band
      Plotly.restyle(top, {{ y: [DATA.y_raw] }}, [0]);
      Plotly.restyle(top, {{ y: [series.y] }}, [1]);
      Plotly.restyle(top, {{
        x: [[...DATA.x_grid, ...DATA.x_grid.slice().reverse()]],
        y: [[...series.lo, ...series.hi.slice().reverse()]]
      }}, [2]);

      // bottom traces: 0 line, 1 band, 2 zero
      Plotly.restyle(bottom, {{ y: [series.dmu] }}, [0]);
      Plotly.restyle(bottom, {{
        x: [[...DATA.x_grid, ...DATA.x_grid.slice().reverse()]],
        y: [[...series.dlo, ...series.dhi.slice().reverse()]]
      }}, [1]);
    }}
  }}

  alphaInput.addEventListener('input', e => update(e.target.value));
  update(alphaInput.value); // first render
}}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Saved HTML:", out_path)


if __name__ == "__main__":
    # Optional reproducibility for pygam's gridsearch CV shuffles, if any
    np.random.seed(42)
    main()
