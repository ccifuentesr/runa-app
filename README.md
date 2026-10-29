<div align="center">

<p align="center">
  <img src="assets/runa-logo-transparent.png" alt="RUNA" width="620"/>
</p>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-active-6ee7b7)](https://runa-ccifuentes.streamlit.app)

**Stellar characterisation from catalogue data**

[Open RUNA](https://runa-ccifuentes.streamlit.app) · [Portfolio](https://ccifuentesr.github.io/runa.html)

</div>

RUNA is a Python application that brings together astronomy utilities developed during my research on nearby cool stars. A selected object is the common input across the application, so catalogue data, visualisations and empirical estimates remain traceable to their source.

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal. Catalogue and image queries require an internet connection.

## Application areas

### Identification

Resolves a stellar name with SIMBAD and retrieves Gaia DR3, 2MASS and UCAC4 values when available. It displays catalogue identifiers, astrometry, radial velocity, a direct parallax distance, an SDSS DR17 cutout and a polar position view around the Sun.

### Photometry

Shows Gaia colour–magnitude context, retrieves GALEX, SDSS9, UCAC4, 2MASS, AllWISE and Gaia photometry, and converts available measurements into an observed SED. The results can be downloaded in a VOSA-compatible input format. It also provides an approximate G−J photometric distance using the Cifuentes et al. (2020) calibration, with its applicability warning.

#### Spectral-type estimation

Estimates an M-dwarf subtype from Gaia G and 2MASS J photometry with a quadratic colour model trained on selected M0–M5.5 stars. `train_model.py` reserves a stratified 20% test set and compares this two-band model with a random forest using G−J and J−W2. The deployed estimator keeps the two-band model because its inputs are inspectable and it works when W2 is unavailable. The result is a photometric estimate, not a replacement for spectroscopy.

### Models

Documents the offline evaluation of the spectral-type estimator and its training-data selection. It also evaluates a Gaia-based classifier trained on confirmed close unresolved components, identified by the Cifuentes et al. (2025) nomenclature, and on `Single` category 1–2 controls. Resolved systems, `Single` category 3 rows and Gaia candidates are outside the supervised target. Components receive separate labels but their systems remain together during validation. Multiplicity applies the selected classifier to the loaded Gaia DR3 diagnostics and shows an uncalibrated score plus a decision at the validation threshold. Internal metrics do not establish external performance or confirm a companion.

### Parameters

**Status: In preparation.** Calculates absolute magnitudes and displays matched published bolometric luminosities from [Cifuentes et al. (2020)](https://doi.org/10.1051/0004-6361/202038295) and [Cifuentes et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...693A.228C/abstract). A separate panel lists bibliographic SIMBAD measurements of [Fe/H], projected rotation, rotational period and spectral type, with links to their original references. When applicable, RUNA estimates mass from the main-sequence mass–luminosity relation of [Eker et al. (2018)](https://doi.org/10.1093/mnras/sty1834), without extrapolating outside its published intervals. It also includes an experimental Gaia/WISE photometric metallicity estimate based on [Duque-Arribas et al. (2023)](https://doi.org/10.3847/1538-4357/acacf6).

### Kinematics

**Status: Experimental.** Propagates Gaia coordinates to a requested epoch and prepares the inputs needed for BANYAN Σ young-association classification. Interpret membership probabilities alongside data quality and external evidence.

### Multiplicity

Applies the seven Gaia DR3 statistical indicators from Cifuentes et al. (2025, Table 3), scores the loaded source for close unresolved multiplicity, visualises RVS score against RUWE, and retrieves nearby Gaia sources for visual inspection. The model score is not a calibrated probability or binary confirmation.

An on-demand archive search reports positionally matched ESO raw science spectra, CARMENES DR1 spectra and the public NOT/FIES target lists. It shows instruments, observing dates, exposure times, programme metadata and public-release status when supplied by the source. Official HARPS-N/TNG, WHT/ING and full Calar Alto archive interfaces are linked for manual review where RUNA has no stable documented query. Gaia positions are propagated to the observing epoch before the final match. Archive coverage is presented as observational context, not as evidence that a source is single or that its radial velocities have been analysed.

Automatically retrieves Gaia DR3 epoch photometry from Gaia@AIP below the nearby-source view. G, BP and RP are plotted on an inverted magnitude axis, with per-observation variability rejection flags and a CSV download. A successful query with no published series is distinguished from a service error. The implementation follows `kobe.query_gaia_lightcurve.py`.

### Planets

Searches the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) by host aliases and Gaia DR3 ID, and the community-maintained [Open Exoplanet Catalogue](https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue) by SIMBAD aliases. The table shows status, orbital period, radius, mass, semimajor axis and eccentricity; a face-on sketch uses the listed orbital parameters. NASA values take precedence when an object appears in both catalogues. A local PSCompPars snapshot, `data/exoplanet_population_20261028.csv`, provides the confirmed-planet mass--period background without an additional query at app runtime.

### Data

**Status: Reference.** Includes the published K5 V–L2.0 average parameters from Cifuentes et al. (2020, Table 6) as a browsable reference table.

## Project structure

```text
app.py                    Streamlit interface
identify_star.py          SIMBAD, Gaia, 2MASS, UCAC4 and SDSS retrieval
simbad_measurements.py    Bibliographic SIMBAD stellar measurements
photometry.py             Catalogue photometry, CMD and SED tools
predict_star.py           Two-band spectral-type estimator
train_model.py            Reproducible model comparison
multiplicity_model.py     Multiplicity data preparation and model definitions
evaluate_multiplicity_models.py  Grouped multiplicity-model evaluation
parameters.py             Magnitudes, luminosity, metallicity and mass relations
mamajek.py                Mean dwarf sequence from the bundled reference table
multiplicity.py           Gaia DR3 multiplicity indicators
eso_archive.py            On-demand ESO archival spectroscopy search
spectroscopy_archives.py  CARMENES, NOT/FIES and multi-archive coordination
kinematics.py             Coordinate propagation and BANYAN Σ inputs
light_curves.py           TESS light curves
planets.py                NASA and Open Exoplanet Catalogue queries and planet matching
data/                     Published catalogues and filter data
```

## Sources

- [Cifuentes et al. (2020)](https://doi.org/10.1051/0004-6361/202038295) — nearby K and M dwarfs, photometric relations and Table 6 reference data.
- [Cifuentes et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...693A.228C/abstract) — updated CARMENES input catalogue and Gaia multiplicity criteria.
- [Eker et al. (2018)](https://doi.org/10.1093/mnras/sty1834) — main-sequence mass–luminosity relation.
- [Duque-Arribas et al. (2023)](https://doi.org/10.3847/1538-4357/acacf6) — Gaia/WISE photometric metallicity calibration.
- [SIMBAD](https://simbad.cds.unistra.fr/), [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3), [VizieR](https://vizier.cds.unistra.fr/), [ESO Science Archive](https://archive.eso.org/), [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), [Open Exoplanet Catalogue](https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue) and [MAST](https://mast.stsci.edu/) — remote data services used by RUNA.

## Data and limits

The repository contains the runtime catalogues and filter data required by the application. Remote services can be unavailable, incomplete or return no counterpart. Missing values are presented as unavailable rather than imputed. RUNA is a research tool: inspect measurement quality, calibration range and method notes before interpreting an individual result.
