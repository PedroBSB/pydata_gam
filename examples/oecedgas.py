# oecedgas.py
# ---------------------------------------------------------------
# - Reads OECD gasoline dataset from URL
# - Fits a GAM with pyGAM using P-splines ('ps')
# - Provides .update() (append + refit) and .predict()
# - Plots data + fitted surface with Plotly and saves to ../resources/
# - Performs os.chdir("..") per your request
# ---------------------------------------------------------------

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pygam import LinearGAM, s, te

# Reproducibility
np.random.seed(123)

# Go up one directory (from examples/ to project root)
try:
    os.chdir("..")
except Exception:
    pass

# Ensure resources/ exists
RES_DIR = Path("resources")
RES_DIR.mkdir(parents=True, exist_ok=True)


class GASGAM:
    """
    Simple wrapper around pyGAM LinearGAM:

        gas ~ te(price, income, basis='ps') + s(cars, basis='ps')

    Notes:
      * pyGAM supports 'ps' (P-spline) and 'cp' (cubic) bases.
      * update(new_df) appends and refits (pyGAM has no partial_fit).
    """
    def __init__(self, lam=0.6, n_splines_2d=25, n_splines_1d=20, basis='ps'):
        self.lam = lam
        self.n_splines_2d = n_splines_2d
        self.n_splines_1d = n_splines_1d
        if basis not in ('ps', 'cp'):
            raise ValueError("basis must be 'ps' or 'cp' for pyGAM.")
        self.basis = basis
        self.gam = None
        self._train_df = None

    def _xy(self, df):
        X = df[['price', 'income', 'cars']].to_numpy()
        y = df['gas'].to_numpy()
        return X, y

    def fit(self, df):
        self._train_df = df[['gas', 'price', 'income', 'cars']].copy()
        X, y = self._xy(self._train_df)

        # Pass lam to the CONSTRUCTOR (not to .fit)
        self.gam = LinearGAM(
            te(0, 1, basis=self.basis, n_splines=self.n_splines_2d) +
            s(2, basis=self.basis, n_splines=self.n_splines_1d),
            lam=self.lam
        )
        self.gam.fit(X, y)
        return self

    def update(self, new_df):
        """Append new rows and refit."""
        add = new_df[['gas', 'price', 'income', 'cars']].copy()
        self._train_df = pd.concat([self._train_df, add], ignore_index=True)
        X, y = self._xy(self._train_df)

        # Reuse same model structure; ensure lam is set on the model
        self.gam.lam = self.lam
        self.gam.fit(X, y)
        return self

    def predict(self, df):
        """Predict E[gas] for rows in df."""
        X = df[['price', 'income', 'cars']].to_numpy()
        return self.gam.predict(X)


def main():
    # 1) Load data
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/OECDGas.csv"
    df = pd.read_csv(url)

    # Keep relevant columns and clean
    keep = ['gas', 'price', 'income', 'cars', 'country', 'year']
    df = df[keep].dropna().reset_index(drop=True)

    # 2) Fit GAM with P-splines
    model = GASGAM(
        lam=0.6,            # smoother -> increase; wigglier -> decrease
        n_splines_2d=25,    # flexibility of the 2D surface
        n_splines_1d=20,    # flexibility of the 1D smooth
        basis='ps'          # 'ps' (P-splines) or 'cp'
    ).fit(df)

    # 3) Prediction grid for price × income (cars held at mean)
    price_grid = np.linspace(df['price'].min(), df['price'].max(), 50)
    income_grid = np.linspace(df['income'].min(), df['income'].max(), 50)
    price_mesh, income_mesh = np.meshgrid(price_grid, income_grid)
    cars_mean = df['cars'].mean()
    cars_mesh = np.full_like(price_mesh, cars_mean)

    grid_df = pd.DataFrame({
        'price': price_mesh.ravel(),
        'income': income_mesh.ravel(),
        'cars': cars_mesh.ravel()
    })
    zhat = model.predict(grid_df).reshape(price_mesh.shape)

    # 4) Plotly figure: surface (fit) + scatter (data)
    scatter = go.Scatter3d(
        x=df['price'],
        y=df['income'],
        z=df['gas'],
        mode='markers',
        marker=dict(size=3, opacity=0.6),
        name='Observed'
    )
    surface = go.Surface(
        x=price_mesh,
        y=income_mesh,
        z=zhat,
        opacity=0.85,
        showscale=True,
        name='GAM fit (cars at mean)'
    )
    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title="OECD Gasoline: GAM (pyGAM, P-splines)",
        scene=dict(
            xaxis_title="log Price",
            yaxis_title="log Income",
            zaxis_title="log Gasoline per Car"
        ),
        template="plotly_white",
        width=950,
        height=700
    )

    # 5) Save HTML to ../resources/
    out_html = RES_DIR / "oecd_gasoline_gam.html"
    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs='cdn')
    print(f"Saved interactive figure to: {out_html.resolve()}")

    # 6) Tiny demo: update() + predict()
    new_rows = df.sample(2, random_state=42).copy()
    model.update(new_rows)
    preds = model.predict(new_rows)
    print("Demo predictions after update():")
    for i, p in enumerate(preds):
        print(f"  row {i}: pred log(gas) = {p:.3f}")


if __name__ == "__main__":
    main()
