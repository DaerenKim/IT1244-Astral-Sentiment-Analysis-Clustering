"""
cluster.py
----------
Clustering algorithm wrappers: K-Means, GMM, DBSCAN.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def find_optimal_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42) -> dict:
    """Compute cluster-quality metrics across a range of k values."""
    results = {"k_values": list(k_range), "inertia": [], "silhouette": [],
               "davies_bouldin": [], "calinski_harabasz": [], "gmm_bic": [], "gmm_aic": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X, lbl, sample_size=2000, random_state=random_state))
        results["davies_bouldin"].append(davies_bouldin_score(X, lbl))
        results["calinski_harabasz"].append(calinski_harabasz_score(X, lbl))
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
        gmm.fit(X)
        results["gmm_bic"].append(gmm.bic(X))
        results["gmm_aic"].append(gmm.aic(X))
        print(f"  k={k}: silhouette={results['silhouette'][-1]:.4f}, "
              f"DB={results['davies_bouldin'][-1]:.4f}, BIC={results['gmm_bic'][-1]:.1f}")
    return results


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> dict:
    """Fit K-Means with k clusters."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    print(f"[cluster] K-Means (k={k}): inertia={model.inertia_:.2f}, sizes={np.bincount(labels)}")
    return {"labels": labels, "model": model, "n_clusters": k,
            "inertia": model.inertia_, "algorithm": "KMeans"}


def run_gmm(X: np.ndarray, k: int, random_state: int = 42) -> dict:
    """Fit Gaussian Mixture Model with k components."""
    model = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
    model.fit(X)
    labels = model.predict(X)
    proba = model.predict_proba(X)
    bic, aic = model.bic(X), model.aic(X)
    print(f"[cluster] GMM (k={k}): BIC={bic:.1f}, AIC={aic:.1f}, sizes={np.bincount(labels)}")
    return {"labels": labels, "proba": proba, "model": model, "n_clusters": k,
            "bic": bic, "aic": aic, "algorithm": "GMM"}


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> dict:
    """Fit DBSCAN. Noise points are labelled -1."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels) - {-1})
    print(f"[cluster] DBSCAN (eps={eps}, min_samples={min_samples}): "
          f"n_clusters={n_clusters}, noise={n_noise} ({n_noise/len(labels)*100:.1f}%)")
    return {"labels": labels, "model": model, "n_clusters": n_clusters,
            "n_noise": n_noise, "algorithm": "DBSCAN"}


def tune_dbscan(X: np.ndarray, eps_values: list, min_samples_values: list) -> list:
    """Grid search over DBSCAN hyperparameters. Returns results sorted by silhouette."""
    results = []
    for eps in eps_values:
        for ms in min_samples_values:
            res = run_dbscan(X, eps=eps, min_samples=ms)
            valid_mask = res["labels"] != -1
            n_valid, n_clust = valid_mask.sum(), res["n_clusters"]
            if n_clust >= 2 and n_valid > n_clust:
                sil = silhouette_score(X[valid_mask], res["labels"][valid_mask],
                                       sample_size=min(2000, n_valid))
                res["silhouette"] = sil
            else:
                res["silhouette"] = -1.0
            res["eps"] = eps
            res["min_samples"] = ms
            results.append(res)
    results.sort(key=lambda r: r["silhouette"], reverse=True)
    return results


# ── Notebook-compatibility wrappers ───────────────────────────────────────

def sweep_kmeans_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42) -> dict:
    """
    Sweep K-Means over k and return K-Means-focused metrics.

    Kept for backward compatibility with earlier notebook API.
    """
    base = find_optimal_k(X, k_range=k_range, random_state=random_state)
    return {
        "k_values": base["k_values"],
        "inertia": base["inertia"],
        "silhouette": base["silhouette"],
        "davies_bouldin": base["davies_bouldin"],
        "calinski_harabasz": base["calinski_harabasz"],
    }


def sweep_gmm_k(X: np.ndarray, k_range=range(2, 11), random_state: int = 42) -> dict:
    """
    Sweep GMM over k and return information-criterion metrics.

    Kept for backward compatibility with earlier notebook API.
    """
    k_values = list(k_range)
    bic_vals, aic_vals = [], []
    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
        gmm.fit(X)
        bic, aic = gmm.bic(X), gmm.aic(X)
        bic_vals.append(bic)
        aic_vals.append(aic)
        print(f"  k={k}: GMM BIC={bic:.1f}, AIC={aic:.1f}")
    return {"k_values": k_values, "bic": bic_vals, "aic": aic_vals}


def recommend_k(kmeans_sweep: dict, gmm_sweep: dict) -> int:
    """
    Recommend k by simple majority vote across standard selection criteria.
    """
    ks = kmeans_sweep["k_values"]
    votes = []
    votes.append(ks[int(np.argmax(kmeans_sweep["silhouette"]))])         # higher better
    votes.append(ks[int(np.argmin(kmeans_sweep["davies_bouldin"]))])     # lower better
    votes.append(ks[int(np.argmax(kmeans_sweep["calinski_harabasz"]))])  # higher better
    votes.append(gmm_sweep["k_values"][int(np.argmin(gmm_sweep["bic"]))])  # lower better
    votes.append(gmm_sweep["k_values"][int(np.argmin(gmm_sweep["aic"]))])  # lower better

    unique, counts = np.unique(votes, return_counts=True)
    k_star = int(unique[np.argmax(counts)])
    print(f"[cluster] Recommended k={k_star} from votes={votes}")
    return k_star


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42):
    """
    Backward-compatible alias returning (labels, model).
    """
    res = run_kmeans(X, k=k, random_state=random_state)
    return res["labels"], res["model"]


def fit_gmm(X: np.ndarray, k: int, random_state: int = 42):
    """
    Backward-compatible alias returning (labels, model).
    """
    res = run_gmm(X, k=k, random_state=random_state)
    return res["labels"], res["model"]


def fit_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10):
    """
    Backward-compatible alias returning (labels, model).
    """
    res = run_dbscan(X, eps=eps, min_samples=min_samples)
    return res["labels"], res["model"]
