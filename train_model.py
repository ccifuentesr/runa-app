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
DATA_FILE = PROJECT_DIR / "cifuentes20_dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RANDOM_STATE = 42
FEATURES = ["G-J", "J-W2"]


def load_stars():
    """Keep M0-M5.5 stars with usable G, J, and W2 measurements."""
    stars = pd.read_csv(DATA_FILE)
    required = {"Karmn", "SpTnum", "Multiple", "Young", "RUWE"}
    required.update({f"{band}mag" for band in ("G", "J", "W2")})
    required.update({f"q_{band}mag" for band in ("G", "J", "W2")})
    missing = required - set(stars.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    for column in ("SpTnum", "Gmag", "Jmag", "W2mag"):
        stars[column] = pd.to_numeric(stars[column], errors="coerce")

    keep = stars["SpTnum"].between(0, 5.5)
    for flag in ("Multiple", "Young"):
        keep &= stars[flag].astype(str).str.lower().eq("false")
    for band in ("G", "J", "W2"):
        keep &= stars[f"q_{band}mag"].astype(str).str.lower().eq("false")

    stars = stars.loc[keep, ["Karmn", "SpTnum", "Gmag", "Jmag", "W2mag", "RUWE"]].copy()
    stars["G-J"] = stars["Gmag"] - stars["Jmag"]
    stars["J-W2"] = stars["Jmag"] - stars["W2mag"]
    stars = stars.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    if stars["Karmn"].duplicated().any():
        raise ValueError("The same star appears more than once")
    return stars


def make_reference_model():
    """A smooth mapping from G-J color to spectral subtype."""
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
    )


def model_summary(name, actual, predicted):
    """Return errors in spectral subtypes, including the sparse late range."""
    lines = [
        f"{name}",
        f"  MAE: {mean_absolute_error(actual, predicted):.3f} subtypes",
        f"  Within 1 subtype: {np.mean(np.abs(actual - predicted) <= 1):.1%}",
    ]
    for label, lower, upper in (
        ("M0-M2", 0, 2),
        ("M2.5-M4", 2.5, 4),
        ("M4.5-M5.5", 4.5, 5.5),
    ):
        group = (actual >= lower) & (actual <= upper)
        lines.append(
            f"  {label}: n={group.sum()}, MAE={mean_absolute_error(actual[group], predicted[group]):.3f}"
        )
    return "\n".join(lines)


def save_figures(stars, actual, baseline_pred, forest_pred):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(stars["G-J"], stars["SpTnum"], s=12, alpha=0.35)
    ax.set(xlabel="G-J (mag)", ylabel="Spectral subtype", title="Color versus spectral subtype")
    ax.set_yticks(range(6), [f"M{i}" for i in range(6)])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "color_vs_subtype.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, predicted, title in zip(
        axes,
        (baseline_pred, forest_pred),
        ("Reference: G-J", "Forest: G-J and J-W2"),
    ):
        ax.scatter(actual, predicted, s=14, alpha=0.45)
        ax.plot([0, 5.5], [0, 5.5], color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="True subtype", xlim=(-0.2, 5.7), ylim=(-0.2, 5.7))
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
