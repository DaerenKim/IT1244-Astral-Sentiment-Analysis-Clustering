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


def validate_schema(df: pd.DataFrame, required: list = None, numeric: list = None) -> dict:
    """
    Validate dataframe schema and return a report dict.

    Parameters
    ----------
    required : list
        Required columns. Defaults to standard photometric fields + label.
    numeric : list
        Columns expected to be numeric. Defaults to bands + redshift.
    """
    required = required or (BAND_COLS + ["redshift", LABEL_COL])
    numeric = numeric or (BAND_COLS + ["redshift"])

    missing = [c for c in required if c not in df.columns]
    non_numeric = [c for c in numeric if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    is_valid = (len(missing) == 0) and (len(non_numeric) == 0)
    return {
        "is_valid": is_valid,
        "missing_columns": missing,
        "non_numeric_columns": non_numeric,
        "n_rows": len(df),
        "n_cols": df.shape[1],
    }


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


def report_quality(df: pd.DataFrame) -> dict:
    """
    Return high-level data-quality diagnostics for the expected schema.
    """
    report = {"n_rows": len(df), "n_cols": df.shape[1]}
    schema = validate_schema(df)
    report["schema"] = schema

    if schema["is_valid"]:
        miss = {c: int(df[c].isna().sum()) for c in BAND_COLS + ["redshift", LABEL_COL]}
        report["missing_by_column"] = miss
        sentinel_counts = {
            c: int(((df[c] <= MAG_LOWER) | (df[c] >= MAG_UPPER)).sum())
            for c in BAND_COLS
        }
        report["sentinel_by_band"] = sentinel_counts
        report["negative_redshift_count"] = int((df["redshift"] < 0).sum())
    return report


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


def split_features_labels(df: pd.DataFrame, label_col: str = LABEL_COL,
                          drop_cols: list = None):
    """
    Generalized splitter with configurable label/drop columns.
    """
    drop_cols = list(drop_cols) if drop_cols is not None else list(COORD_COLS)
    drop_cols = drop_cols + [label_col]
    if "objid" in df.columns and "objid" not in drop_cols:
        drop_cols.append("objid")
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), df[label_col].copy()


def load_pipeline(filepath: str, drop_coords: bool = True, keep_redshift: bool = True,
                  remove_invalid: bool = True):
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
    df_clean = remove_outliers(df) if remove_invalid else df.copy()

    if drop_coords:
        work = df_clean.drop(columns=[c for c in COORD_COLS if c in df_clean.columns])
    else:
        work = df_clean.copy()

    if (not keep_redshift) and ("redshift" in work.columns):
        work = work.drop(columns=["redshift"])

    X_raw, y = split_features_labels(work, label_col=LABEL_COL, drop_cols=[])
    print(f"[loader] Final dataset: {len(df_clean)} rows, {X_raw.shape[1]} features.")
    print(f"[loader] Class distribution:\n{y.value_counts().to_string()}")
    return X_raw, y, df_clean
