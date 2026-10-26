"""Evaluate the RUNA catalogue-multiplicity ML draft without app-side training.

Run with: python evaluate_multiplicity_models.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from multiplicity_model import RANDOM_STATE, model_definitions, prepare_training_data


REPORT_DIR = Path(__file__).resolve().parent / "data" / "models"
DECISION_THRESHOLD = 0.5


def binary_metrics(target, probability):
    """Calculate thresholded and ranking metrics for the positive class."""
    prediction = (probability >= DECISION_THRESHOLD).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
    }


def grouped_cross_validation(models, features, target, groups):
    """Compare models using group-separated folds inside the training partition."""
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    results = {name: [] for name in models}
    for train_index, validation_index in splitter.split(features, target, groups):
        for name, model in models.items():
            fitted = clone(model).fit(features.iloc[train_index], target.iloc[train_index])
            probability = fitted.predict_proba(features.iloc[validation_index])[:, 1]
            results[name].append(binary_metrics(target.iloc[validation_index], probability))
    summary = []
    for name, folds in results.items():
        line = {"model": name, "folds": len(folds)}
        for metric in folds[0]:
            values = [fold[metric] for fold in folds]
            line[f"{metric}_mean"] = float(np.mean(values))
            line[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(line)
    return summary


def run_evaluation():
    """Select by grouped CV F1, then evaluate once on a separate system split."""
    features, target, groups, metadata, preparation, label_audit = prepare_training_data()
    models = model_definitions()
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_index, test_index = next(outer_splitter.split(features, target, groups))
    train_features, test_features = features.iloc[train_index], features.iloc[test_index]
    train_target, test_target = target.iloc[train_index], target.iloc[test_index]
    train_groups, test_groups = groups.iloc[train_index], groups.iloc[test_index]

    comparison = grouped_cross_validation(models, train_features, train_target, train_groups)
    comparison.sort(key=lambda row: (row["f1_mean"], row["roc_auc_mean"]), reverse=True)
    selected_name = comparison[0]["model"]
    selected_model = clone(models[selected_name]).fit(train_features, train_target)
    test_probability = selected_model.predict_proba(test_features)[:, 1]
    test_metrics = binary_metrics(test_target, test_probability)

    predictions = metadata.iloc[test_index][[
        "IDstar", "Karmn", "Name", "IDsystem", "Type", "Category", "Comp",
        "system_architecture", "cohort",
    ]].copy()
    predictions["close_unresolved_label"] = test_target.to_numpy()
    predictions["predicted_probability"] = test_probability
    predictions["predicted_label"] = (test_probability >= DECISION_THRESHOLD).astype(int)
    predictions["validation_group"] = test_groups.to_numpy()
    # Keep the untouched catalogue measurements with each held-out prediction so
    # the app can inspect which Gaia diagnostics separate the labelled cohorts.
    for feature_name in features.columns:
        predictions[feature_name] = features.iloc[test_index][feature_name].to_numpy()

    importances = None
    classifier = selected_model.named_steps.get("classifier") if hasattr(selected_model, "named_steps") else None
    if classifier is not None and hasattr(classifier, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": features.columns,
            "importance": classifier.feature_importances_[:len(features.columns)],
        }).sort_values("importance", ascending=False)

    report = {
        "purpose": (
            "Internal grouped validation for distinguishing confirmed close unresolved "
            "components from spectroscopically confirmed singles. This is not external "
            "validation or binary confirmation for a new object."
        ),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "preparation": preparation,
        "feature_selection": {
            "rule": "Keep raw Gaia features with at least 5% non-missing values.",
            "excluded": ["Gaia non-single-star flag: unavailable as a complete raw field in the local catalogue."],
        },
        "validation": {
            "outer_split": "One 80/20 StratifiedGroupKFold split. Each IDsystem>0 system stays together; each IDsystem=0 star is an independent group.",
            "model_comparison": "Four-fold StratifiedGroupKFold on the training partition.",
            "selection_metric": "Mean grouped CV F1, then mean ROC AUC.",
            "test_rows": int(len(test_index)),
            "test_groups": int(test_groups.nunique()),
            "test_multiple_fraction": float(test_target.mean()),
        },
        "comparison": comparison,
        "selected_model": selected_name,
        "decision_threshold": DECISION_THRESHOLD,
        "held_out_test": test_metrics,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "multiplicity_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    predictions.to_csv(REPORT_DIR / "multiplicity_test_predictions.csv", index=False)
    label_audit.to_csv(REPORT_DIR / "multiplicity_label_audit.csv", index=False)
    if importances is not None:
        importances.to_csv(REPORT_DIR / "multiplicity_feature_importance.csv", index=False)
    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(f"Selected model: {report['selected_model']}")
    print("Held-out test:", report["held_out_test"])
