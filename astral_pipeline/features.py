"""
features.py
-----------
Feature engineering and preprocessing.

Three feature sets are constructed and compared (per project plan):

  Set A — Raw bands + redshift
      Columns: u, g, r, i, z, redshift
      Rationale: Preserves absolute brightness information.

  Set B — Color indices + redshift
      Columns: u-g, g-r, r-i, i-z, redshift
      Rationale: Color indices subtract out brightness and isolate spectral
      shape — the physically motivated representation. Adjacent-band
      differences are standard in photometric classification literature.

  Set C — Combined (raw + color indices + redshift)
      Columns: u, g, r, i, z, u-g, g-r, r-i, i-z, redshift
      Rationale: Gives the clustering algorithm the full picture; the
      algorithm can decide which dimensions carry signal.

Scaling: RobustScaler is preferred over StandardScaler because it uses
the interquartile range and is therefore less sensitive to the residual
extreme values that can persist even after sentinel removal.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


# ── Color index definitions ────────────────────────────────────────────────
COLOR_INDEX_PAIRS = [
    ("u", "g"),   # u-g
    ("g", "r"),   # g-r
    ("r", "i"),   # r-i
    ("i", "z"),   # i-z
]


def add_color_indices(X: pd.DataFrame) -> pd.DataFrame:
    """
    Append adjacent-band color indices to a dataframe that already contains
    the five magnitude columns (u, g, r, i, z).

    Color index = brighter_band - redder_band (e.g. u - g).

    Parameters
    ----------
    X : pd.DataFrame  — must contain columns u, g, r, i, z

    Returns
    -------
    pd.DataFrame with four additional columns: u-g, g-r, r-i, i-z
    """
    X = X.copy()
    for blue, red in COLOR_INDEX_PAIRS:
        col_name = f"{blue}-{red}"
        X[col_name] = X[blue] - X[red]
    return X


def add_custom_differences(X: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """
    Add custom difference features for arbitrary (left, right) column pairs.
    """
    X = X.copy()
    for left, right in pairs:
        X[f"{left}-{right}"] = X[left] - X[right]
    return X


def build_feature_sets(X_raw: pd.DataFrame) -> dict:
    """
    Construct all three feature sets from the raw feature matrix.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Output of loader.get_features_and_labels — columns: u,g,r,i,z,redshift

    Returns
    -------
    dict with keys 'A', 'B', 'C', each mapping to a pd.DataFrame (unscaled).
    """
    X_with_colors = add_color_indices(X_raw)

    band_cols    = ["u", "g", "r", "i", "z"]
    color_cols   = [f"{b}-{r}" for b, r in COLOR_INDEX_PAIRS]
    redshift_col = ["redshift"]

    sets = {
        "A": X_raw[band_cols + redshift_col].copy(),
        "B": X_with_colors[color_cols + redshift_col].copy(),
        "C": X_with_colors[band_cols + color_cols + redshift_col].copy(),
    }

    print("[features] Feature sets constructed:")
    for name, df in sets.items():
        print(f"  Set {name}: {list(df.columns)}")

    return sets


def make_feature_set(X_raw: pd.DataFrame, include_bands: bool = True,
                     include_colors: bool = True, include_redshift: bool = True,
                     extra_cols: list = None) -> pd.DataFrame:
    """
    Build a custom feature set from toggles.
    """
    X = add_color_indices(X_raw) if include_colors else X_raw.copy()
    cols = []
    if include_bands:
        cols += [c for c in ["u", "g", "r", "i", "z"] if c in X.columns]
    if include_colors:
        cols += [f"{b}-{r}" for b, r in COLOR_INDEX_PAIRS if f"{b}-{r}" in X.columns]
    if include_redshift and "redshift" in X.columns:
        cols.append("redshift")
    if extra_cols:
        cols += [c for c in extra_cols if c in X.columns and c not in cols]
    return X[cols].copy()


def get_scaler(kind: str = "robust"):
    """
    Return a scaler instance by name.
    Supported: 'robust', 'standard', 'minmax'.
    """
    kind = kind.lower()
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler kind: {kind}")


def scale_features(X: pd.DataFrame, scaler=None, scaler_kind: str = "robust") -> tuple:
    """
    Apply RobustScaler to a feature dataframe.

    Parameters
    ----------
    X : pd.DataFrame (unscaled)

    Returns
    -------
    X_scaled : np.ndarray  — scaled feature matrix
    scaler   : fitted RobustScaler instance (for later inverse transforms)
    """
    if scaler is None:
        scaler = get_scaler(scaler_kind)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def prepare_all_sets(feature_sets: dict, scaler_kind: str = "robust") -> dict:
    """
    Scale all feature sets and return a dict of
    { 'A': (X_scaled, scaler), 'B': ..., 'C': ... }
    """
    prepared = {}
    for name, X in feature_sets.items():
        X_scaled, scaler = scale_features(X, scaler_kind=scaler_kind)
        prepared[name] = {
            "X_scaled": X_scaled,
            "scaler": scaler,
            "columns": list(X.columns),
            "X_df": X,
        }
        print(f"[features] Set {name} scaled — shape: {X_scaled.shape}")
    return prepared
