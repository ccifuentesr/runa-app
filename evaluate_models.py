"""Build the offline spectral-model report: python evaluate_models.py."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_model import (
    DATA_FILE, EXCLUSIONS_FILE, RANDOM_STATE, load_stars, make_reference_model,
)


REPORT_DIR = Path(__file__).resolve().parent / "data" / "models"


def evaluate_models():
    """Compare fixed models on identical splits without tuning on the test set."""
    stars = load_stars()
    train, test = train_test_split(
        stars, test_size=0.2, random_state=RANDOM_STATE,
        stratify=stars["SpTnum"],
    )
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE,
    ).split(train, train["SpTnum"].astype(str)))
    models = [
        ("Linear", LinearRegression(), ["G-J"]),
        ("Quadratic", make_reference_model(), ["G-J"]),
        ("Random forest", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=RANDOM_STATE,
        ), ["G-J", "J-W2"]),
    ]
    predictions = test[["Karmn", "SpTnum", "G-J", "J-W2"]].copy()
    metrics = []
    for name, model, features in models:
        fold_mae = []
        for fit_indices, valid_indices in folds:
            fit, valid = train.iloc[fit_indices], train.iloc[valid_indices]
            model.fit(fit[features], fit["SpTnum"])
            fold_mae.append(mean_absolute_error(
                valid["SpTnum"], model.predict(valid[features]),
            ))
        model.fit(train[features], train["SpTnum"])
        predicted = model.predict(test[features])
        predictions[name] = predicted
        metrics.append({
            "model": name, "features": ", ".join(features),
            "cv_mae": float(np.mean(fold_mae)),
            "cv_std": float(np.std(fold_mae)),
            "test_mae": float(mean_absolute_error(test["SpTnum"], predicted)),
            "within_one": float(np.mean(np.abs(test["SpTnum"] - predicted) <= 1)),
        })
    report = {
        "sample_size": len(stars), "train_size": len(train), "test_size": len(test),
        "seed": RANDOM_STATE, "folds": 5, "sklearn_version": sklearn_version,
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "exclusions_sha256": hashlib.sha256(EXCLUSIONS_FILE.read_bytes()).hexdigest(),
        "exclusions": pd.read_csv(EXCLUSIONS_FILE).to_dict(orient="records"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(
            Path(__file__).with_name("train_model.py").read_bytes()
        ).hexdigest(),
        "colour_range": [float(stars["G-J"].min()), float(stars["G-J"].max())],
        "counts": {str(k): int(v) for k, v in stars.groupby("SpTnum").size().items()},
        "metrics": metrics,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "spectral_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    predictions.to_csv(REPORT_DIR / "spectral_predictions.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Sample: {len(stars)}; train: {len(train)}; test: {len(test)}")


if __name__ == "__main__":
    evaluate_models()
