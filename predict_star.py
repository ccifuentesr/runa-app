"""Estimate one M-dwarf subtype from Gaia G and 2MASS J magnitudes."""

import math

import pandas as pd

from train_model import load_stars, make_reference_model


def main():
    try:
        g_mag = float(input("Gaia G magnitude: "))
        j_mag = float(input("2MASS J magnitude: "))
    except ValueError:
        raise SystemExit("Please enter numeric magnitudes.")

    if not math.isfinite(g_mag) or not math.isfinite(j_mag):
        raise SystemExit("Magnitudes must be finite numbers.")

    color = g_mag - j_mag
    stars = load_stars()
    low, high = stars["G-J"].min(), stars["G-J"].max()
    if not low <= color <= high:
        raise SystemExit(
            f"G-J = {color:.2f} is outside the training range "
            f"({low:.2f} to {high:.2f} mag). No estimate was made."
        )

    model = make_reference_model()
    model.fit(stars[["G-J"]], stars["SpTnum"])
    prediction = model.predict(pd.DataFrame({"G-J": [color]}))[0]
    subtype = round(max(0.0, min(5.5, prediction)) * 2) / 2

    print(f"G-J color: {color:.2f} mag")
    print(f"Estimated spectral type: M{subtype:.1f}")
    print("Photometric estimate for an ordinary M0-M5.5 dwarf; confirm with spectroscopy.")


if __name__ == "__main__":
    main()
