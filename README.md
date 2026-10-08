# RUNA

RUNA is an app for exploring stellar properties such as spectral type, mass, radius, and multiplicity. The current command-line feature estimates the spectral subtype of an M dwarf from two photometric measurements. It does not yet estimate the other properties or provide a graphical interface.

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
