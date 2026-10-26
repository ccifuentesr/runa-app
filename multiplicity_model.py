"""Preparation and model definitions for the RUNA multiplicity ML draft.

The supervised target is a *component-level* label: confirmed close unresolved
multiple components versus spectroscopically confirmed singles. It is not a
general catalogue-multiplicity label or physical proof for a new object.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "cifuentes25_dataset.csv"
RANDOM_STATE = 42
MINIMUM_FEATURE_COMPLETENESS = 0.05

# These fields match the Gaia DR3 diagnostics queried in multiplicity.py.
FEATURE_COLUMNS = {
    "RUWE": "ruwe",
    "IPDgofha": "ipd_harmonic_amplitude",
    "IPDfmp": "ipd_multi_peak_fraction",
    "RVchi2": "rv_chisq_pvalue",
    "RVrnorm": "rv_renormalised_gof",
    "o_RV": "rv_transits",
    "e_RV": "radial_velocity_error",
    "Dup": "duplicated_source",
}

GAIA_VALUE_COLUMNS = {
    "ruwe": "ruwe",
    "ipd_gof_harmonic_amplitude": "ipd_harmonic_amplitude",
    "ipd_frac_multi_peak": "ipd_multi_peak_fraction",
    "rv_chisq_pvalue": "rv_chisq_pvalue",
    "rv_renormalised_gof": "rv_renormalised_gof",
    "rv_nb_transits": "rv_transits",
    "radial_velocity_error": "radial_velocity_error",
    "duplicated_source": "duplicated_source",
}


def gaia_feature_row(values, feature_names):
    """Build one model-input row from the Gaia DR3 response fields."""
    row = {}
    for gaia_name, model_name in GAIA_VALUE_COLUMNS.items():
        value = values.get(gaia_name)
        if model_name == "duplicated_source":
            value = _duplicate_source(value)
        else:
            try:
                value = float(value) if value is not None else np.nan
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                value = np.nan
        row[model_name] = value
    return pd.DataFrame([{name: row.get(name, np.nan) for name in feature_names}], columns=feature_names)


def _duplicate_source(value):
    """Keep Gaia's duplicated-source flag numeric while preserving missingness."""
    if pd.isna(value):
        return np.nan
    normalised = str(value).strip().lower()
    if normalised in {"t", "true", "1"}:
        return 1.0
    if normalised in {"f", "false", "0"}:
        return 0.0
    return np.nan


def _system_architectures(catalogue):
    """Map each system identifier to its one non-empty architecture string."""
    return catalogue.groupby("IDsystem")["System"].first()


def _is_close_unresolved_component(component, architecture):
    """Read the Cifuentes et al. component nomenclature at star level.

    Lowercase companion letters mark unresolved hierarchical components (for
    example Aab and Bab). Parenthesised architecture groups mark an unresolved
    Gaia source (for example (AB)); only the component matching that group is
    labelled positive, never another component in the same system.
    """
    component = str(component).strip()
    architecture = "" if pd.isna(architecture) else str(architecture).strip()
    parenthesised = re.findall(r"\(([^()]+)\)", architecture)
    return bool(re.search(r"[a-z]", component)) or component in parenthesised


def label_cohorts(catalogue):
    """Assign training, test-only and unlabelled roles from catalogue metadata."""
    type_value = catalogue["Type"].fillna("").astype(str).str.strip()
    architecture = catalogue["IDsystem"].map(_system_architectures(catalogue))
    unresolved = pd.Series(
        [_is_close_unresolved_component(component, system)
         for component, system in zip(catalogue["Comp"], architecture)],
        index=catalogue.index,
        name="close_unresolved_component",
    )
    confirmed_positive = type_value.eq("Multiple") & unresolved
    confirmed_single = type_value.eq("Single") & catalogue["Category"].isin((1, 2))
    resolved_multiple = type_value.eq("Multiple") & ~unresolved
    likely_single = type_value.eq("Single") & catalogue["Category"].eq(3)

    cohort = pd.Series("excluded", index=catalogue.index, name="cohort")
    cohort.loc[resolved_multiple] = "resolved_multiple_context"
    cohort.loc[likely_single] = "unconfirmed_single_inference"
    cohort.loc[type_value.isin(("Candidate", "Multiple+"))] = "candidate_inference"
    cohort.loc[confirmed_single] = "confirmed_single_negative"
    cohort.loc[confirmed_positive] = "close_unresolved_positive"
    return cohort, unresolved, architecture


def _validation_groups(frame):
    """Keep whole systems apart, while each IDsystem=0 star is independent."""
    groups = []
    for index, row in frame.iterrows():
        system_id = int(row["IDsystem"])
        groups.append(f"system-{system_id}" if system_id else f"single-{int(row['IDstar'])}")
    return pd.Series(groups, index=frame.index, name="validation_group")


def prepare_training_data():
    """Build component-level labels, features and system-separated groups."""
    catalogue = pd.read_csv(DATA_FILE)
    cohort, unresolved, architecture = label_cohorts(catalogue)
    labelled = cohort.isin(("close_unresolved_positive", "confirmed_single_negative"))
    frame = catalogue.loc[labelled].copy()
    frame["cohort"] = cohort.loc[labelled]
    frame["system_architecture"] = architecture.loc[labelled]
    target = cohort.loc[labelled].eq("close_unresolved_positive").astype(int)
    target = target.rename("is_close_unresolved_component")
    positive_before_feature_filter = int((target == 1).sum())
    negative_before_feature_filter = int((target == 0).sum())

    features = pd.DataFrame(index=frame.index)
    for source_column, model_column in FEATURE_COLUMNS.items():
        if source_column == "Dup":
            features[model_column] = frame[source_column].map(_duplicate_source)
        else:
            features[model_column] = pd.to_numeric(frame[source_column], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    completeness = features.notna().mean()
    selected_features = completeness[completeness >= MINIMUM_FEATURE_COMPLETENESS].index.tolist()
    features = features[selected_features].copy()
    has_any_feature = features.notna().any(axis=1)

    frame = frame.loc[has_any_feature].copy()
    target = target.loc[has_any_feature].copy()
    features = features.loc[has_any_feature].copy()
    groups = _validation_groups(frame)

    label_audit = catalogue[[
        "IDstar", "IDsystem", "Name", "Karmn", "Type", "Category", "Comp", "System",
    ]].copy()
    label_audit["system_architecture"] = architecture
    label_audit["close_unresolved_component"] = unresolved
    label_audit["cohort"] = cohort

    preparation = {
        "catalogue_rows": int(len(catalogue)),
        "labelled_rows_before_feature_filter": int(labelled.sum()),
        "confirmed_close_positive_before_feature_filter": positive_before_feature_filter,
        "confirmed_single_negative_before_feature_filter": negative_before_feature_filter,
        "resolved_multiple_context_rows": int((cohort == "resolved_multiple_context").sum()),
        "unconfirmed_single_inference_rows": int((cohort == "unconfirmed_single_inference").sum()),
        "candidate_inference_rows": int((cohort == "candidate_inference").sum()),
        "excluded_all_features_missing": int((~has_any_feature).sum()),
        "training_rows": int(len(frame)),
        "confirmed_close_positive_rows": int((target == 1).sum()),
        "confirmed_single_negative_rows": int((target == 0).sum()),
        "validation_groups": int(groups.nunique()),
        "feature_completeness": {name: float(value) for name, value in completeness.items()},
        "selected_features": selected_features,
    }
    return features, target, groups, frame, preparation, label_audit


def model_definitions():
    """Return transparent baseline, linear and nonlinear classifiers."""
    return {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
            )),
        ]),
        "Random forest": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
            )),
        ]),
    }
