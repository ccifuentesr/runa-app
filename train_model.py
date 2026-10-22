"""Estimate M-dwarf spectral subtype from broad-band photometry.

Run with: python train_model.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a window.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "cifuentes20_dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RANDOM_STATE = 42
FEATURES = ["G-J", "J-W2"]


def load_stars():
    """Keep M0-M5.5 stars with usable G, J, and W2 measurements."""
    stars = pd.read_csv(DATA_FILE)
    required_columns = [
        "Karmn", "SpTnum", "Multiple", "Young", "RUWE",
        "Gmag", "Jmag", "W2mag", "q_Gmag", "q_Jmag", "q_W2mag",
    ]
    for column in required_columns:
        if column not in stars.columns:
            raise ValueError(f"Missing CSV column: {column}")

    for column in ("SpTnum", "Gmag", "Jmag", "W2mag"):
        stars[column] = pd.to_numeric(stars[column], errors="coerce")

    keep = stars["SpTnum"].between(0, 5.5)
    stars = stars.loc[keep].copy()

    # A false flag means the star or measurement is not marked as problematic.
    quality_flags = ["Multiple", "Young", "q_Gmag", "q_Jmag", "q_W2mag"]
    for column in quality_flags:
        flag = stars[column].astype(str).str.lower()
        stars = stars.loc[flag == "false"].copy()

    columns = ["Karmn", "SpTnum", "Gmag", "Jmag", "W2mag", "RUWE"]
    stars = stars[columns].copy()
    stars["G-J"] = stars["Gmag"] - stars["Jmag"]
    stars["J-W2"] = stars["Jmag"] - stars["W2mag"]
    stars = stars.replace([np.inf, -np.inf], np.nan)
    stars = stars.dropna(subset=FEATURES)
    if stars["Karmn"].duplicated().any():
        raise ValueError("The same star appears more than once")
    return stars


def make_reference_model():
    """A smooth mapping from G-J color to spectral subtype."""
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    regression = LinearRegression()
    return make_pipeline(polynomial, regression)


def model_summary(name, actual, predicted):
    """Return errors in spectral subtypes, including the sparse late range."""
    errors = np.abs(actual - predicted)
    mae = mean_absolute_error(actual, predicted)
    within_one = np.mean(errors <= 1)
    lines = [
        name,
        f"  MAE: {mae:.3f} subtypes",
        f"  Within 1 subtype: {within_one:.1%}",
    ]
    for label, lower, upper in (
        ("M0-M2", 0, 2),
        ("M2.5-M4", 2.5, 4),
        ("M4.5-M5.5", 4.5, 5.5),
    ):
        group = (actual >= lower) & (actual <= upper)
        count = group.sum()
        group_mae = mean_absolute_error(actual[group], predicted[group])
        lines.append(f"  {label}: n={count}, MAE={group_mae:.3f}")
    return "\n".join(lines)


def save_figures(stars, actual, baseline_pred, forest_pred):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(stars["G-J"], stars["SpTnum"], s=12, alpha=0.35)
    ax.set(xlabel="G-J (mag)", ylabel="Spectral subtype", title="Color versus spectral subtype")
    ax.set_yticks(range(6), ["M0", "M1", "M2", "M3", "M4", "M5"])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "color_vs_subtype.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    axes[0].scatter(actual, baseline_pred, s=14, alpha=0.45)
    axes[0].set_title("Reference: G-J")
    axes[1].scatter(actual, forest_pred, s=14, alpha=0.45)
    axes[1].set_title("Forest: G-J and J-W2")
    for ax in axes:
        ax.plot([0, 5.5], [0, 5.5], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="True subtype", xlim=(-0.2, 5.7), ylim=(-0.2, 5.7))
    axes[0].set_ylabel("Predicted subtype")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predictions.png", dpi=160)
    plt.close(fig)


def main():
    stars = load_stars()
    train, test = train_test_split(
        stars, test_size=0.2, random_state=RANDOM_STATE, stratify=stars["SpTnum"]
    )

    # A smooth, one-color reference model. It is trained only on the training stars.
    baseline = make_reference_model()
    baseline.fit(train[["G-J"]], train["SpTnum"])
    baseline_pred = baseline.predict(test[["G-J"]])

    # A small nonlinear model using two measured colors.
    forest = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    forest.fit(train[FEATURES], train["SpTnum"])
    forest_pred = forest.predict(test[FEATURES])

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = "\n\n".join(
        [
            f"Stars: {len(stars)}; training: {len(train)}; test: {len(test)}",
            "Both models use the same held-out stars. Lower MAE is better.",
            model_summary("Quadratic reference (G-J)", test["SpTnum"], baseline_pred),
            model_summary("Random forest (G-J, J-W2)", test["SpTnum"], forest_pred),
        ]
    )
    (RESULTS_DIR / "metrics.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    predictions = test[["Karmn", "SpTnum", "G-J", "J-W2", "RUWE"]].copy()
    predictions["quadratic_prediction"] = baseline_pred
    predictions["forest_prediction"] = forest_pred
    predictions["forest_absolute_error"] = np.abs(test["SpTnum"] - forest_pred)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_figures(stars, test["SpTnum"], baseline_pred, forest_pred)


if __name__ == "__main__":
    main()
