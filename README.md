# RUNA

RUNA is a planned app for exploring stellar properties such as spectral type, mass, radius, and multiplicity. Its first small scikit-learn example estimates M-dwarf subtypes from M0 to M5.5 using colors calculated from Gaia, 2MASS, and WISE magnitudes. The data come from the catalogue in [Cifuentes et al. (2020)](https://doi.org/10.1051/0004-6361/202038295). The CSV contains broadband photometry, not full spectra.

## Run

From this directory, with Python installed:

```bash
python3 -m pip install -r requirements.txt
python3 train_model.py
```

The training script reads `cifuentes20_dataset.csv` without changing it. It writes metrics, test-set predictions, and two plots to `results/`.

## Estimate one star

Run the terminal prompt:

```bash
python3 predict_star.py
```

Enter a Gaia `G` magnitude and a 2MASS `J` magnitude. The script calculates `G-J` and returns an estimated M subtype, rounded to the nearest 0.5. It uses the simple one-color model, trained on all selected catalogue stars after the train/test evaluation. The random forest requires a third magnitude, `W2`, so it is not used here. Colors outside the training range are rejected.

## How it works

1. Keep M0-M5.5 stars that are not flagged as multiple or young. Require usable `G`, `J`, and `W2` measurements.
2. Calculate two colors: `G-J` and `J-W2`. Luminosity, temperature, mass, radius, and identifiers are never used as model inputs.
3. Reserve 20% of the stars for testing, with a similar mix of subtypes in the training and test sets.
4. Fit a quadratic curve using only `G-J` as a reference. Fit a `RandomForestRegressor` using both colors.
5. Compare mean absolute error (MAE) in spectral subtypes and the share of predictions within one subtype. Report MAE separately for early, middle, and late subtypes.

Both models are evaluated on the **same held-out stars**. `RUWE` is included in the prediction file for investigating difficult cases, but it is not used for training.

## Interpretation

The reference model has an MAE of 0.258 subtypes; the random forest has an MAE of 0.249. This small difference on one train/test split does not establish a clear advantage for the more complex model. The largest test error is for a star with a `RUWE` flag. It remains in the test results rather than being removed after seeing the prediction.

The catalogue is a selected sample with relatively few late-M stars. These are internal test results, not a validation for arbitrary stars. The model can provide an initial photometric estimate; spectroscopy is still needed to confirm a spectral type.

The transferable method is estimating a costly-to-measure property from accessible measurements, with data-quality checks, a simple reference model, and error analysis across groups.
