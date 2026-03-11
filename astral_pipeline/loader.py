"""
loader.py
---------
Handles loading the raw CSV, validation, outlier removal, and feature/label splitting.

Design decisions (mirrored in notebook Phase 0):
  - 'class' is NEVER passed to any clustering function.
  - 'ra' and 'dec' are dropped: sky position carries no information about
    intrinsic object type in a general photometric survey.
  - Sentinel magnitudes (values >= 50 or <= -50) are treated as invalid
    and affected rows are removed.
  - 'redshift' is KEPT: it directly encodes distance/recession velocity and
    occupies distinct regimes for stars (~0), galaxies (0-1), quasars (>1).
    The README only forbids use of 'class'.
"""

import pandas as pd
import numpy as np

COORD_COLS = ["ra", "dec"]
LABEL_COL  = "class"
BAND_COLS  = ["u", "g", "r", "i", "z"]
MAG_UPPER  = 50.0
MAG_LOWER  = -50.0


def load_raw(filepath: str) -> pd.DataFrame:
    """Load raw CSV, skipping comment lines starting with '#'."""
    df = pd.read_csv(filepath, comment="#")
    return df


def validate(df: pd.DataFrame) -> None:
    """Assert required columns are present and numeric."""
    required = BAND_COLS + ["redshift", LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in BAND_COLS + ["redshift"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' should be numeric, got {df[col].dtype}")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with sentinel/physically impossible values.
    SDSS uses ~999 for saturated/unobserved pixels. Negative redshift
    is physically impossible.
    """
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in BAND_COLS:
        mask &= (df[col] > MAG_LOWER) & (df[col] < MAG_UPPER)
    mask &= df["redshift"] >= 0.0
    df_clean = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        print(f"[loader] Removed {n_removed} rows with sentinel/invalid values "
              f"({n_removed / n_before * 100:.2f}% of data).")
    else:
        print("[loader] No sentinel/invalid rows found.")
    return df_clean


def get_features_and_labels(df: pd.DataFrame):
    """
    Split cleaned dataframe into feature matrix and label series.

    Returns
    -------
    X_raw : pd.DataFrame  — columns: u, g, r, i, z, redshift
    y     : pd.Series     — class labels (for post-hoc evaluation ONLY)
    """
    drop_cols = COORD_COLS + [LABEL_COL]
    if "objid" in df.columns:
        drop_cols.append("objid")
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), df[LABEL_COL].copy()


def load_pipeline(filepath: str):
    """
    Convenience: load → validate → remove outliers → split.

    Returns
    -------
    X_raw    : pd.DataFrame   (u, g, r, i, z, redshift)
    y        : pd.Series      (class labels, for evaluation only)
    df_clean : pd.DataFrame   (full cleaned dataframe)
    """
    df = load_raw(filepath)
    validate(df)
    df_clean = remove_outliers(df)
    X_raw, y = get_features_and_labels(df_clean)
    print(f"[loader] Final dataset: {len(df_clean)} rows, {X_raw.shape[1]} features.")
    print(f"[loader] Class distribution:\n{y.value_counts().to_string()}")
    return X_raw, y, df_clean
