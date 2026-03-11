# Astronomical Object Photometric Dataset Documentation

## Overview and Problem Statement
The universe contains a diverse array of astronomical objects that emit or reflect light across the electromagnetic spectrum. Automatically distinguishing between stars, galaxies, and quasars from photometric survey data is a fundamental challenge in modern astronomy with applications in large-scale structure mapping, cosmological modeling, and the identification of rare or transient phenomena.


You are tasked with applying unsupervised clustering methods to group astronomical objects by their photometric properties, without relying on the provided class labels during training. The labels are provided solely for post-hoc evaluation of your clusters.

## Dataset File

The dataset is provided as a single CSV file: `star-galaxy-quasar.csv`

It contains spectroscopically confirmed observations of astronomical objects collected from a large-scale sky survey.

## Column Descriptions

### Spatial Coordinates

| Column | Description |
|--------|-------------|
| `ra` | **Right Ascension** — the celestial equivalent of longitude. Measured in degrees from 0 to 360, it specifies how far east an object is along the celestial equator, starting from a fixed reference point in the sky. |
| `dec` | **Declination** — the celestial equivalent of latitude. Measured in degrees from −90 (south celestial pole) to +90 (north celestial pole), it specifies how far north or south of the celestial equator an object lies. |

### Photometric Magnitudes

Magnitude is a logarithmic measure of an object's brightness — a **lower value means a brighter object**. A difference of 5 magnitudes corresponds to a factor of 100 in brightness. The five columns below each measure brightness through a different wavelength filter, together capturing the object's spectral energy distribution from ultraviolet to infrared light.

| Column | Band | Wavelength Region |
|--------|------|-------------------|
| `u` | u-band | Ultraviolet (~355 nm) 
| `g` | g-band | Green/Blue (~469 nm) | 
| `r` | r-band | Red (~617 nm) |
| `i` | i-band | Near-infrared (~748 nm) | 
| `z` | z-band | Infrared (~893 nm) | 

Differences between adjacent bands (e.g. `u − g`, `g − r`) are called **color indices** and are often more informative than the raw magnitudes themselves, as they describe the shape of an object's spectrum rather than its absolute brightness.

### Redshift

| Column | Description |
|--------|-------------|
| `redshift` | **Spectroscopic redshift** — when an object is moving away from us, its light is stretched to longer (redder) wavelengths. Redshift quantifies this stretching. A value near 0 indicates the object is close and roughly stationary relative to Earth (typical of stars within our galaxy). Higher values indicate greater distances and recessional velocities due to the expansion of the universe. Quasars can have redshifts of 1.0 or more; galaxies typically range from 0 to ~1; stars are near 0. |

### Class Label

| Column | Description |
|--------|-------------|
| `class` | **Ground-truth object type**, determined from spectroscopic follow-up observations. **Do not use this column as a feature during clustering** — it is provided solely for evaluating your results afterward. |

### Object Classes (3 categories)
- `STAR` : Point sources within the Milky Way galaxy; 
- `GALAXY` : Vast extragalactic collections of stars; 
- `QSO` : Quasi-stellar objects (quasars); 

## Class Distribution

Here we shall provide some directions to help kickstart your project. You are not required to answer these questions and are encouraged to explore other directions that interest you.

## Suggestions & Comments

The following prompts are intended to guide your exploration. You are not required to address all of them and are encouraged to pursue other questions that interest you.

### Feature Selection and Engineering
Not all columns may be equally useful for clustering. Consider which features are most physically meaningful. It was previously stated that color indices are often more discriminative than raw magnitudes? What other features could you engineer? Reading about the properties of these astral objects can be a good starting point. 
- The five photometric magnitudes (`u`, `g`, `r`, `i`, `z`) encode the spectral energy distribution of each object
- Color indices (differences between adjacent bands, e.g., `u - g`, `g - r`) are often more discriminative than raw magnitudes

### Choosing and Evaluating Clustering Algorithms
Suppose you do not know that there are only 3 types of objects. Would you be able to determine the number of clusters or do they not seperate nicely?
Is this because these astronomical objects have similar properties or are your features not discriminative?
