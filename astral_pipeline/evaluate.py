"""
evaluate.py
-----------
Clustering evaluation metrics — internal and external.

IMPORTANT design principle:
  Internal metrics  — use only X and cluster labels. ONLY these are used during
                      model selection. No label information enters this process.
  External metrics  — compare cluster labels to true class labels (y). Computed
                      AFTER all clustering decisions are finalised, for reporting only.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)


def internal_metrics(X: np.ndarray, labels: np.ndarray,
                     sample_size: int = 3000, random_state: int = 42) -> dict:
    """
    Compute internal clustering quality metrics (no labels used).
    Silhouette: higher is better. Davies-Bouldin: lower is better. CH: higher is better.
    Noise points (label == -1) are excluded.
    """
    mask = labels != -1
    X_v, l_v = X[mask], labels[mask]
    if len(set(l_v)) < 2:
        print("[evaluate] WARNING: fewer than 2 clusters — metrics undefined.")
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    sil = silhouette_score(X_v, l_v, sample_size=min(sample_size, len(X_v)), random_state=random_state)
    return {
        "silhouette":        round(sil, 4),
        "davies_bouldin":    round(davies_bouldin_score(X_v, l_v), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_v, l_v), 2),
    }


def cluster_purity(labels: np.ndarray, y) -> float:
    """Purity = fraction of points assigned to their cluster's majority class."""
    mask = labels != -1
    l_v, y_v = labels[mask], np.asarray(y)[mask]
    total_correct = sum(
        np.bincount(pd.factorize(y_v[l_v == cid])[0]).max()
        for cid in np.unique(l_v)
    )
    return float(round(total_correct / len(l_v), 4))


def external_metrics_strict(labels: np.ndarray, y) -> dict:
    """
    Strict external-metrics API with explicit argument order.
    """
    mask = labels != -1
    l_v, y_v = labels[mask], np.asarray(y)[mask]
    return {
        "ari":    round(adjusted_rand_score(y_v, l_v), 4),
        "nmi":    round(normalized_mutual_info_score(y_v, l_v), 4),
        "purity": cluster_purity(labels, y),
    }


def external_metrics(a, b) -> dict:
    """
    Compute external metrics using ground-truth labels.
    *** CALL ONLY AFTER ALL CLUSTERING DECISIONS ARE FINALISED ***
    """
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if np.issubdtype(a_arr.dtype, np.number) and not np.issubdtype(b_arr.dtype, np.number):
        labels, y = a, b
    elif np.issubdtype(b_arr.dtype, np.number) and not np.issubdtype(a_arr.dtype, np.number):
        labels, y = b, a
    else:
        labels, y = a, b

    return external_metrics_strict(labels, y)


def cluster_class_mapping(labels: np.ndarray, y) -> pd.DataFrame:
    """
    For each cluster, show the count of each true class.
    Reveals which cluster captures which object type.
    """
    mask = labels != -1
    df = pd.DataFrame({"cluster": labels[mask], "true_class": np.asarray(y)[mask]})
    mapping = df.groupby(["cluster", "true_class"]).size().unstack(fill_value=0)
    mapping["dominant_class"] = mapping.idxmax(axis=1)
    totals = mapping.drop(columns="dominant_class").sum(axis=1)
    maxima = mapping.drop(columns="dominant_class").max(axis=1)
    mapping["purity"] = (maxima / totals).round(4)
    return mapping


def cluster_to_class_mapping(y, labels) -> pd.DataFrame:
    """
    Backward-compatible alias with argument order used in older notebooks.
    """
    return cluster_class_mapping(labels, y)


def build_summary_table(results: list) -> pd.DataFrame:
    """
    Build a comparison table across all algorithm × feature-set combinations.
    Each item in results must have: algorithm, feature_set, labels, X_scaled, y, n_clusters.
    """
    rows = []
    for r in results:
        # Newer package path: raw outputs + data, compute metrics here.
        if {"labels", "X_scaled", "y"}.issubset(r.keys()):
            int_m = internal_metrics(r["X_scaled"], r["labels"])
            ext_m = external_metrics(r["labels"], r["y"])
            rows.append({
                "algorithm": r["algorithm"],
                "feature_set": r["feature_set"],
                "k": r.get("n_clusters", len(set(r["labels"])) - (1 if -1 in r["labels"] else 0)),
                "silhouette": int_m["silhouette"],
                "davies_bouldin": int_m["davies_bouldin"],
                "calinski_harabasz": int_m["calinski_harabasz"],
                "ari": ext_m["ari"],
                "nmi": ext_m["nmi"],
                "purity": ext_m["purity"],
            })
        # Notebook path: metrics already pre-computed in each row.
        else:
            rows.append({
                "algorithm": r["algorithm"],
                "feature_set": r["feature_set"],
                "k": r.get("k", r.get("n_clusters", np.nan)),
                "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"],
                "calinski_harabasz": r["calinski_harabasz"],
                "ari": r["ari"],
                "nmi": r["nmi"],
                "purity": r["purity"],
            })
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)


def to_display_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert canonical lowercase summary columns to display-friendly labels.
    """
    rename_map = {
        "algorithm": "Algorithm",
        "feature_set": "Feature Set",
        "k": "N Clusters",
        "silhouette": "Silhouette ↑",
        "davies_bouldin": "Davies-Bouldin ↓",
        "calinski_harabasz": "Calinski-Harabasz ↑",
        "ari": "ARI ↑ (post-hoc)",
        "nmi": "NMI ↑ (post-hoc)",
        "purity": "Purity ↑ (post-hoc)",
    }
    cols = [c for c in rename_map if c in summary_df.columns]
    return summary_df[cols].rename(columns=rename_map).copy()
