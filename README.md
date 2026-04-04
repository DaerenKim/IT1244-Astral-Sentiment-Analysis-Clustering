# Astronomical Object Photometric Dataset

Unsupervised clustering of stars, galaxies, and quasars from large-scale sky survey photometric data — without using class labels during training.

---

## Overview

The universe contains a diverse array of astronomical objects that emit or reflect light across the electromagnetic spectrum. Automatically distinguishing between **stars**, **galaxies**, and **quasars** from photometric survey data is a fundamental challenge in modern astronomy, with applications in large-scale structure mapping, cosmological modeling, and the identification of rare or transient phenomena.

The labels provided in the dataset are used **solely for post-hoc evaluation** of clustering results — not as training features.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

**Files:** `Dataset/star-galaxy-quasar.csv` (raw) · `Dataset/astral_data.csv` (preprocessed)

Spectroscopically confirmed observations collected from a large-scale sky survey.

### Spatial Coordinates

| Column | Description |
|--------|-------------|
| `ra` | Right Ascension — celestial longitude, measured in degrees (0–360) along the celestial equator from a fixed reference point |
| `dec` | Declination — celestial latitude, measured in degrees (−90 to +90) from the celestial equator |

### Photometric Magnitudes

Magnitude is a logarithmic measure of brightness — lower values mean brighter objects. A difference of 5 magnitudes equals a factor of 100 in brightness. Together, the five bands capture each object's spectral energy distribution from ultraviolet to infrared.

| Column | Band | Wavelength |
|--------|------|------------|
| `u` | u-band | Ultraviolet (~355 nm) |
| `g` | g-band | Green/Blue (~469 nm) |
| `r` | r-band | Red (~617 nm) |
| `i` | i-band | Near-infrared (~748 nm) |
| `z` | z-band | Infrared (~893 nm) |

> **Note:** Differences between adjacent bands (e.g. `u − g`, `g − r`) are called *color indices* and are often more informative than raw magnitudes, as they describe the shape of an object's spectrum rather than its absolute brightness.

### Redshift

| Column | Description |
|--------|-------------|
| `redshift` | Spectroscopic redshift — quantifies how much an object's light is stretched to longer wavelengths due to recessional velocity. Near 0 for stars; 0–1 for galaxies; ≥1 for quasars |

### Class Label

| Column | Description |
|--------|-------------|
| `class` | Ground-truth object type from spectroscopic follow-up. **Do not use as a feature during clustering.** |

---

## Object Classes

| Class | Description |
|-------|-------------|
| `STAR` | Point sources within the Milky Way galaxy |
| `GALAXY` | Vast extragalactic collections of stars |
| `QSO` | Quasi-stellar objects (quasars) |

---

## Notebooks

All notebooks operate on the **preprocessed feature set**: color indices (`u-g`, `g-r`, `r-i`, `i-z`) and log-transformed redshift (`redshift_log`).

| Notebook | Description |
|----------|-------------|
| [`0_eda.ipynb`](Notebooks/0_eda.ipynb) | Exploratory data analysis — distributions, correlations, and feature engineering |
| [`1_kmeans.ipynb`](Notebooks/1_kmeans.ipynb) | K-Means clustering with elbow/silhouette selection, STACO stability analysis, and PCA visualisation |
| [`2_dbscan_hdbscan.ipynb`](Notebooks/2_dbscan_hdbscan.ipynb) | Density-based clustering via DBSCAN and HDBSCAN with grid search and noise analysis |
| [`3_gmm.ipynb`](Notebooks/3_gmm.ipynb) | Gaussian Mixture Model clustering with BIC/AIC component selection, covariance structure tuning, and membership-probability analysis |

---

## Methods

### K-Means (`1_kmeans.ipynb`)

K-Means is used as the baseline centroid-based method. Key steps:

- **K selection** via elbow plot (inertia) and silhouette score
- **Stability analysis** using STACO (Cramér's V across 100 random initialisations)
- **Evaluation** for K = 2, 3, 4, 5 using silhouette, Davies–Bouldin, Calinski–Harabasz, ARI, and NMI
- **PCA visualisation** against ground-truth labels

Key finding: K = 2 achieves the best internal metrics (silhouette ≈ 0.69) by separating stars from non-stars, but fails to distinguish galaxies from quasars. K = 3 aligns with the physical prior but suffers from the algorithm's spherical cluster assumption. K = 4 best recovers QSOs as a distinct group.

### DBSCAN & HDBSCAN (`2_dbscan_hdbscan.ipynb`)

Density-based methods that identify clusters of arbitrary shape and treat low-density regions as noise:

- **DBSCAN** grid search over `eps` and `min_samples` (targeting 3 clusters)
- **HDBSCAN** grid search with soft clustering and noise handling
- Per-class F1 scores and majority-vote cluster-to-class mapping

### Gaussian Mixture Model (`3_gmm.ipynb`)

GMM extends K-Means by modelling clusters as elliptical Gaussian distributions:

- **Component selection** via BIC and AIC
- **Covariance structure** comparison: full, tied, diagonal, spherical
- **Membership probabilities** as a value-add over hard assignment methods
- Final models evaluated with ARI, NMI, and per-class purity

---

## Evaluation Metrics

| Metric | Type | What it measures |
|--------|------|-----------------|
| Silhouette score | Internal | Cluster cohesion and separation |
| Davies–Bouldin index | Internal | Average ratio of intra- to inter-cluster distances (lower = better) |
| Calinski–Harabasz index | Internal | Ratio of between- to within-cluster dispersion (higher = better) |
| ARI | External | Agreement with ground-truth labels, chance-corrected |
| NMI | External | Normalised mutual information with ground-truth labels |

> Internal metrics are computed without labels. External metrics (ARI, NMI) use the withheld `class` column only for post-hoc validation.

---

## Project Structure

```
.
├── Dataset/
│   ├── star-galaxy-quasar.csv   # Raw survey data
│   └── astral_data.csv          # Preprocessed feature matrix (color indices + redshift_log)
├── Notebooks/
│   ├── 0_eda.ipynb
│   ├── 1_kmeans.ipynb
│   ├── 2_dbscan_hdbscan.ipynb
│   └── 3_gmm.ipynb
├── requirements.txt
└── README.md
```

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

- `scikit-learn` — K-Means, DBSCAN, GMM, metrics
- `hdbscan` — HDBSCAN clustering
- `joblib` — parallelised STACO stability runs
- `matplotlib` / `seaborn` — visualisation
