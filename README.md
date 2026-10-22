# RUNA

RUNA is an app for exploring stellar properties. Identification loads one star for the other tools. The web interface places it on a Gaia colour–magnitude diagram, estimates the spectral subtype of an M dwarf from its magnitudes, calculates absolute magnitudes and luminosity from Cifuentes et al. (2020) relations, and checks Gaia DR3 multiplicity indicators. A separate manual spectral-type mode does not change the loaded star. Photometry can also be converted to an observed spectral energy distribution (SED).

## Web interface

With Python 3.11 or newer, install the dependencies and start the web app:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open the address shown in the terminal. **Identification** is the first step: enter a name to load an object for the whole session. It retrieves SIMBAD data, Gaia photometry by source ID, and 2MASS J by the object's exact 2MASS identifier; converts a positive SIMBAD parallax to distance; and displays an SDSS DR17 colour cutout where SDSS has coverage. **Photometry** uses Gaia DR3 G, BP, RP and parallax to place the star on a colour–magnitude diagram. It also offers an approximate photometric distance from G−J when the colour is in the Cifuentes et al. (2020) calibration range, with a warning that the typical accuracy is about 36% and that solar-like metallicity and good photometry are recommended. Its light-grey reference sample contains Gaia stars with reliable basic astrometry and photometry. **Spectral type** automatically uses the loaded star's Gaia G and 2MASS J magnitudes with the same model as the command-line tool. Its **Manual** mode accepts two magnitudes without changing the loaded object. **Parameters** calculates absolute magnitudes and luminosities. **Multiplicity** checks the seven Gaia DR3 indicators in Cifuentes et al. (2025), Table 3, for the loaded star. **Planets** searches the NASA Exoplanet Archive for planets associated with the loaded star name and its catalogue identifiers. Missing catalogue matches are shown clearly. An indicator is evidence to investigate, not confirmation of a companion. The catalogue and image queries require internet access.

Each working tool has its own Python file: `predict_star.py`, `identify_star.py`, `parameters.py`, `multiplicity.py`, `light_curves.py`, and `planets.py`. `app.py` displays their results.

## Estimate one star

From this directory, install the dependencies and start the prompt:

```bash
python3 -m pip install -r requirements.txt
python3 predict_star.py
```

Enter the **Gaia G** and **2MASS J** magnitudes of a star. For example:

```text
Gaia G magnitude: 10.5
2MASS J magnitude: 7.7
G-J color: 2.80 mag
Estimated spectral type: M3.5
Photometric estimate for an ordinary M0-M5.5 dwarf; confirm with spectroscopy.
```

`predict_star.py` calculates `G-J` and uses a quadratic model to estimate a subtype between M0 and M5.5, rounded to the nearest 0.5. It is trained on all 1,510 selected catalogue stars after model evaluation. It rejects colors outside the current training range (about 1.94 to 3.73 mag). Use reliable magnitudes: the prompt does not check their measurement quality.

## Reproduce the evaluation

```bash
python3 train_model.py
```

This script filters the catalogue, reserves 20% of the selected stars for testing, and compares two models on the same test set:

- A quadratic reference model using `G-J` (test MAE: 0.258 spectral subtypes).
- A random forest using `G-J` and `J-W2` (test MAE: 0.249 spectral subtypes).

The difference is small on this one split. The single-star prompt uses the quadratic model because it needs only G and J; the random forest also needs a W2 magnitude. The evaluation writes metrics, individual test predictions, and two plots to `results/`. The source CSV is never changed.

## Data and limits

The data come from [Cifuentes et al. (2020)](https://doi.org/10.1051/0004-6361/202038295). The CSV contains broadband photometry, not full spectra. Training excludes stars flagged as multiple or young and requires usable G, J, and W2 measurements. The sample is selected and has relatively few late-M stars. Test errors describe this catalogue, not performance on every M dwarf. Spectroscopy is needed to confirm a spectral type.

Runtime data are kept under `data/`: `cifuentes20_dataset.csv` is the spectral-type training catalogue, `cifuentes25_dataset.csv` supports the published-parameter lookup, and `data/filters/` contains the VOSA filter metadata, effective-wavelength metadata, and local photometric transmission curves used by the SED display. The application no longer depends on those files remaining in another directory under `CODE`.
