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
 
**File:** `star-galaxy-quasar.csv`
 
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