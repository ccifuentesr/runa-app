"""Estimate one M-dwarf subtype from Gaia G and 2MASS J magnitudes."""

import math

import pandas as pd

from train_model import load_stars, make_reference_model


def estimate_subtype(g_mag, j_mag):
    """Estimate an M-dwarf subtype from Gaia G and 2MASS J."""
    if not math.isfinite(g_mag) or not math.isfinite(j_mag):
        raise ValueError("Magnitudes must be finite numbers.")

    color = g_mag - j_mag
    stars = load_stars()
    low = stars["G-J"].min()
    high = stars["G-J"].max()
    if not low <= color <= high:
        raise ValueError(
            f"G-J = {color:.2f} is outside the training range "
            f"({low:.2f} to {high:.2f} mag). No estimate was made."
        )

    model = make_reference_model()
    model.fit(stars[["G-J"]], stars["SpTnum"])
    new_star = pd.DataFrame({"G-J": [color]})
    prediction = model.predict(new_star)[0]

    if prediction < 0:
        prediction = 0.0
    elif prediction > 5.5:
        prediction = 5.5

    # Round to the nearest half subtype, for example M3.0 or M3.5.
    subtype = round(prediction * 2) / 2
    return color, subtype


def main():
    try:
        g_mag = float(input("Gaia G magnitude: "))
        j_mag = float(input("2MASS J magnitude: "))
    except ValueError:
        raise SystemExit("Please enter numeric magnitudes.")

    try:
        color, subtype = estimate_subtype(g_mag, j_mag)
    except ValueError as error:
        raise SystemExit(str(error))

    print(f"G-J color: {color:.2f} mag")
    print(f"Estimated spectral type: M{subtype:.1f}")
    print("Photometric estimate for an ordinary M0-M5.5 dwarf; confirm with spectroscopy.")


if __name__ == "__main__":
    main()
