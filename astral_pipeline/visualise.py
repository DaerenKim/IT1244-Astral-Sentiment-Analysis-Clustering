"""
visualise.py
------------
Plotting utilities for the clustering pipeline.

All functions return matplotlib Figure objects so they can be displayed
inline in the notebook or saved to disk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA

# Consistent colour palette across all plots
CLASS_PALETTE   = {"STAR": "#f4c542", "GALAXY": "#4a90d9", "QSO": "#e85d4a"}
CLUSTER_PALETTE = sns.color_palette("tab10")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[visualise] umap-learn not installed — UMAP plots will be skipped.")


# ── EDA plots ─────────────────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series) -> plt.Figure:
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[CLASS_PALETTE.get(c, "grey") for c in counts.index])
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}\n({val/len(y)*100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig


def plot_feature_distributions(X_raw: pd.DataFrame, y: pd.Series) -> plt.Figure:
    cols = list(X_raw.columns)
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        for cls in y.unique():
            mask = y == cls
            axes[i].hist(X_raw.loc[mask, col], bins=60, alpha=0.5,
                         label=cls, color=CLASS_PALETTE.get(cls, "grey"),
                         density=True)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
    axes[0].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by True Class\n(for EDA only — not used in clustering)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_redshift_distribution(X_raw: pd.DataFrame, y: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in ["STAR", "GALAXY", "QSO"]:
        mask = y == cls
        ax.hist(X_raw.loc[mask, "redshift"], bins=80, alpha=0.6,
                label=cls, color=CLASS_PALETTE[cls], density=True)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Density")
    ax.set_title("Redshift Distribution by True Class\n"
                 "Stars ≈ 0 | Galaxies 0–1 | Quasars > 1", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_redshift_boxplot_by_class(X_raw: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """
    One subplot per class so each gets its own y-scale.
    """
    df = pd.DataFrame({
        "redshift": X_raw["redshift"].values,
        "class": y.values,
    })

    classes = [c for c in ["STAR", "GALAXY", "QSO"] if c in df["class"].unique()]
    palette = {"STAR": "#f4c542", "GALAXY": "#4a90d9", "QSO": "#e85d4a"}

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = df[df["class"] == cls].copy()
        sub["group"] = cls  # single category for seaborn boxplot axis

        sns.boxplot(
            data=sub,
            x="group",
            y="redshift",
            color=palette.get(cls, "grey"),
            showfliers=False,
            ax=ax,
        )

        # Optional sparse points for distribution feel
        sample_n = min(len(sub), 1000)
        if sample_n > 0:
            samp = sub.sample(sample_n, random_state=42)
            sns.stripplot(
                data=samp,
                x="group",
                y="redshift",
                color="black",
                alpha=0.2,
                size=2,
                ax=ax,
            )

        ax.set_title(f"{cls} Redshift", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Redshift")

    fig.suptitle("Redshift Boxplots by Class (Independent Scales)", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_color_index_scatter(X_with_colors: pd.DataFrame, y: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [("u-g", "g-r"), ("g-r", "r-i"), ("r-i", "i-z")]
    for ax, (cx, cy) in zip(axes, pairs):
        for cls in ["STAR", "GALAXY", "QSO"]:
            mask = y == cls
            ax.scatter(X_with_colors.loc[mask, cx], X_with_colors.loc[mask, cy],
                       s=2, alpha=0.3, label=cls, color=CLASS_PALETTE[cls])
        ax.set_xlabel(cx); ax.set_ylabel(cy)
        ax.set_title(f"{cx} vs {cy}")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.suptitle("Color-Color Diagrams (true class coloring — EDA only)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(X: pd.DataFrame) -> plt.Figure:
    """Plot a feature correlation matrix for EDA."""
    corr = X.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    return fig


# ── K-selection plots ──────────────────────────────────────────────────────

def plot_k_selection(k_results: dict, gmm_results: dict = None,
                     chosen_k: int = None) -> plt.Figure:
    """
    Plot k-selection diagnostics.

    Supports both:
      - Newer API: plot_k_selection(k_results) where k_results already includes
        K-Means and GMM metrics.
      - Notebook API: plot_k_selection(kmeans_sweep, gmm_sweep, chosen_k=...).
    """
    if gmm_results is not None:
        merged = dict(k_results)
        merged["gmm_bic"] = gmm_results["bic"]
        merged["gmm_aic"] = gmm_results["aic"]
        k_results = merged

    k = k_results["k_values"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    metrics = [
        ("inertia",           "Inertia (Elbow)",          "lower ↓", False),
        ("silhouette",        "Silhouette Score",         "higher ↑", True),
        ("davies_bouldin",    "Davies-Bouldin Index",     "lower ↓",  False),
        ("calinski_harabasz", "Calinski-Harabasz Index",  "higher ↑", True),
        ("gmm_bic",           "GMM BIC",                  "lower ↓",  False),
        ("gmm_aic",           "GMM AIC",                  "lower ↓",  False),
    ]

    for ax, (key, title, direction, higher_better) in zip(axes, metrics):
        vals = k_results[key]
        ax.plot(k, vals, "o-", color="#4a90d9", linewidth=2, markersize=6)
        best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
        ax.axvline(k[best_idx], color="#e85d4a", linestyle="--", alpha=0.7,
                   label=f"best k={k[best_idx]}")
        if chosen_k is not None:
            ax.axvline(chosen_k, color="#222222", linestyle=":", alpha=0.9,
                       label=f"theoretical k={chosen_k}")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_title(f"{title}\n({direction})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("K-Selection Metrics — Treated as Genuinely Unknown",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Projection & cluster plots ────────────────────────────────────────────

def _pca_2d(X: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca


def compute_embedding(X: np.ndarray, method: str = "pca", random_state: int = 42,
                      **kwargs) -> np.ndarray:
    """
    Compute a 2D embedding for visualization.
    Supported methods: 'pca', 'umap'
    """
    m = method.lower()
    if m == "pca":
        coords, _ = _pca_2d(X)
        return coords
    if m == "umap":
        if not UMAP_AVAILABLE:
            raise ValueError("UMAP requested but umap-learn is not installed.")
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=kwargs.get("n_neighbors", 30),
            min_dist=kwargs.get("min_dist", 0.1),
        )
        return reducer.fit_transform(X)
    raise ValueError(f"Unknown embedding method: {method}")


def plot_embedding(X: np.ndarray, labels: np.ndarray, method: str = "pca",
                   title: str = "") -> plt.Figure:
    """
    Plot a single embedding view colored by cluster labels.
    """
    coords = compute_embedding(X, method=method)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cid in sorted(set(labels)):
        mask = labels == cid
        color = "lightgrey" if cid == -1 else CLUSTER_PALETTE[cid % 10]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.4, color=color, label=label)
    ax.set_title(f"{method.upper()} — {title}".strip(" —"))
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, y: pd.Series,
                      title: str = "") -> plt.Figure:
    coords, pca = _pca_2d(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cluster assignments
    n_clusters = len(set(labels) - {-1})
    for cid in sorted(set(labels)):
        mask = labels == cid
        color = "lightgrey" if cid == -1 else CLUSTER_PALETTE[cid % 10]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        s=2, alpha=0.4, color=color, label=label)
    axes[0].set_title(f"Cluster Assignments\n{title}")
    axes[0].legend(markerscale=4, fontsize=8)

    # True class (post-hoc reveal)
    for cls in ["STAR", "GALAXY", "QSO"]:
        mask = np.asarray(y) == cls
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=2, alpha=0.4, color=CLASS_PALETTE[cls], label=cls)
    axes[1].set_title("True Classes (post-hoc)")
    axes[1].legend(markerscale=4, fontsize=8)

    for ax in axes:
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle(f"PCA Projection — {title}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_umap_clusters(X: np.ndarray, labels: np.ndarray, y: pd.Series,
                       title: str = "") -> plt.Figure:
    if not UMAP_AVAILABLE:
        print("[visualise] UMAP not available — skipping.")
        return None

    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30)
    coords = reducer.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for cid in sorted(set(labels)):
        mask = labels == cid
        color = "lightgrey" if cid == -1 else CLUSTER_PALETTE[cid % 10]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        s=2, alpha=0.4, color=color, label=label)
    axes[0].set_title(f"Cluster Assignments\n{title}")
    axes[0].legend(markerscale=4, fontsize=8)

    for cls in ["STAR", "GALAXY", "QSO"]:
        mask = np.asarray(y) == cls
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=2, alpha=0.4, color=CLASS_PALETTE[cls], label=cls)
    axes[1].set_title("True Classes (post-hoc)")
    axes[1].legend(markerscale=4, fontsize=8)

    for ax in axes:
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    fig.suptitle(f"UMAP Projection — {title}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_embedding_comparison(X: np.ndarray, labels: np.ndarray, y: pd.Series,
                              title: str = "") -> plt.Figure:
    """
    Backward-compatible notebook helper: PCA and UMAP side-by-side.
    """
    pca_coords, _ = _pca_2d(X)
    if UMAP_AVAILABLE:
        import umap as umap_lib
        umap_coords = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30).fit_transform(X)
        right_title = "UMAP"
        right_coords = umap_coords
        right_xlabel, right_ylabel = "UMAP-1", "UMAP-2"
    else:
        right_title = "PCA"
        right_coords = pca_coords
        right_xlabel, right_ylabel = "PC1", "PC2"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cid in sorted(set(labels)):
        mask = labels == cid
        color = "lightgrey" if cid == -1 else CLUSTER_PALETTE[cid % 10]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                        s=2, alpha=0.4, color=color, label=label)
        axes[1].scatter(right_coords[mask, 0], right_coords[mask, 1],
                        s=2, alpha=0.4, color=color, label=label)

    axes[0].set_title("PCA")
    axes[1].set_title(right_title)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel(right_xlabel); axes[1].set_ylabel(right_ylabel)
    axes[0].legend(markerscale=4, fontsize=8)

    fig.suptitle(f"Embedding Comparison — {title}", fontweight="bold")
    fig.tight_layout()
    return fig


# ── Evaluation plots ───────────────────────────────────────────────────────

def plot_confusion_heatmap(mapping_df: pd.DataFrame, title: str = "") -> plt.Figure:
    class_cols = [c for c in mapping_df.columns
                  if c not in ("dominant_class", "purity")]
    data = mapping_df[class_cols].astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.5, ax=ax)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Cluster → Class Mapping\n{title}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_summary_table(summary_df: pd.DataFrame) -> plt.Figure:
    # Accept canonical lowercase schema and map it to display labels on the fly.
    if "algorithm" in summary_df.columns:
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
        present = {k: v for k, v in rename_map.items() if k in summary_df.columns}
        summary_df = summary_df.rename(columns=present)

    int_cols = ["Algorithm", "Feature Set", "N Clusters",
                "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
    ext_cols = ["Algorithm", "Feature Set",
                "ARI ↑ (post-hoc)", "NMI ↑ (post-hoc)", "Purity ↑ (post-hoc)"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 5))

    for ax, cols, title in zip(axes,
                                [int_cols, ext_cols],
                                ["Internal Metrics (used for model selection)",
                                 "External Metrics (post-hoc evaluation only)"]):
        sub = summary_df[cols].copy()
        ax.axis("off")
        tbl = ax.table(cellText=sub.values, colLabels=sub.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(sub.columns))))
        ax.set_title(title, fontweight="bold", pad=12)

    fig.suptitle("Full Model Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
