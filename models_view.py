"""Display saved spectral-model evaluation without training during page visits."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from evaluate_models import REPORT_DIR
from train_model import DATA_FILE, EXCLUSIONS_FILE


MULTIPLICITY_REPORT_FILE = REPORT_DIR / "multiplicity_evaluation.json"
MULTIPLICITY_PREDICTIONS_FILE = REPORT_DIR / "multiplicity_test_predictions.csv"


def report_table(frame, formats=None):
    """Render a compact static table in the app's existing visual style."""
    st.markdown(
        '<div class="model-report-table">'
        + frame.to_html(index=False, border=0, formatters=formats)
        + '</div>', unsafe_allow_html=True,
    )


def _show_spectral_type_estimation():
    """Present model provenance, held-out performance and inspectable diagnostics."""
    st.markdown("""<style>
    .model-report-table {overflow-x:auto;}
    .model-report-table table {width:100%;border-collapse:collapse;font-size:1.02rem;}
    .model-report-table th {color:#92929d;font-weight:500;text-align:left;}
    .model-report-table td,.model-report-table th {padding:.65rem .7rem;border:0;border-bottom:1px solid #29292d;}
    .model-subtitle {color:#e4e4e7;font-size:1.2rem;line-height:1.4;margin:.1rem 0 1.15rem;}
    .model-flow {display:grid;grid-template-columns:minmax(0,1fr) 28px minmax(0,1.35fr) 28px minmax(0,1.5fr);align-items:stretch;gap:.5rem;margin:.8rem 0 1.1rem;}
    .model-flow-step {min-width:0;border:1px solid #333338;border-radius:10px;padding:.9rem 1rem;background:#18181b;}
    .model-flow-step small {display:block;color:#92929d;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.45rem;}
    .model-flow-step strong {display:block;color:#e4e4e7;font-size:1.15rem;font-weight:550;}
    .model-flow-step span {display:block;color:#92929d;font-size:.87rem;margin-top:.35rem;}
    .model-flow-step.model-selection {border-color:#595443;}
    .model-flow-step.model-selection strong {color:#f4cf75;}
    .model-flow-arrow {align-self:center;justify-self:center;color:#72727c;}
    .model-split {display:grid;grid-template-columns:1fr 1fr;gap:.55rem;}
    .model-split-card {border:1px solid #333338;border-radius:10px;padding:.8rem .85rem;background:#18181b;}
    .model-split-card small {display:block;color:#92929d;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.4rem;}
    .model-split-card strong {display:block;color:#e4e4e7;font-size:1.08rem;font-weight:570;}
    .model-split-card span {display:block;color:#92929d;font-size:.82rem;line-height:1.35;margin-top:.3rem;}
    .model-split-card.model-train {border-color:#426b5b;}
    .model-split-card.model-train strong {color:#64dfb4;}
    .model-split-card.model-test {border-color:#5d5968;}
    .model-split-card.model-test strong {color:#c9c4e8;}
    .model-flow-outcome {margin-top:.55rem;padding-top:.65rem;border-top:1px solid #29292d;color:#a1a1aa;font-size:.84rem;line-height:1.4;}
    .model-flow-outcome strong {color:#e4e4e7;font-weight:550;}
    .model-context-card {border:1px solid #595443;border-radius:10px;
                         background:#18181b;padding:1rem 1.1rem;margin:.75rem 0;}
    .model-context-card small {display:block;color:#92929d;font-size:.76rem;
                               letter-spacing:.06em;text-transform:uppercase;margin-bottom:.42rem;}
    .model-context-card strong {display:block;color:#f4cf75;font-size:1.18rem;font-weight:550;}
    .model-context-card p {color:#b7b7c0;font-size:.92rem;line-height:1.45;margin:.45rem 0 0;}
    @media(max-width:700px) {
        .model-flow {grid-template-columns:1fr;gap:.4rem;}
        .model-flow-arrow {transform:rotate(90deg);}
        .model-split {grid-template-columns:1fr;}
    }
    </style>""", unsafe_allow_html=True)
    st.markdown(
        '<p class="model-subtitle">A supervised regression from photometric colours '
        'to M-dwarf subtype.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-context-card"><small>Context</small>'
        '<strong>Colours offer a first map</strong><p>Light is our main source of '
        'information about stars. Photometric filters measure that light across different '
        'wavelength bands; comparing their brightness gives us a star\'s colours. This '
        'model uses those colours to estimate spectral type when spectroscopy is unavailable '
        'or would require costly telescope time. Reddening, metallicity, unresolved '
        'multiplicity and poor photometry can shift colours, so a photometric estimate '
        'supports initial characterisation but does not replace a spectrum.</p></div>',
        unsafe_allow_html=True,
    )
    report_file = REPORT_DIR / "spectral_evaluation.json"
    if not report_file.exists():
        st.info("The offline evaluation report has not been generated yet.")
        return
    report = json.loads(report_file.read_text(encoding="utf-8"))
    predictions = pd.read_csv(REPORT_DIR / "spectral_predictions.csv")
    if hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() != report["data_sha256"]:
        st.warning("The training catalogue has changed. Regenerate the evaluation report.")
    if hashlib.sha256(EXCLUSIONS_FILE.read_bytes()).hexdigest() != report.get("exclusions_sha256"):
        st.warning("The exclusions have changed. Regenerate the evaluation report.")

    with st.container(border=True):
        st.markdown("#### From observations to prediction")
        st.markdown(f'''
        <div class="model-flow" aria-label="Prediction workflow">
          <div class="model-flow-step"><small>01 · Observations</small><strong>Published subtype<br>and colours</strong><span>G, J and W2 catalogue measurements</span></div>
          <div class="model-flow-arrow" aria-hidden="true">→</div>
          <div class="model-flow-step model-selection"><small>02 · Quality selection</small><strong>{report['sample_size']} selected stars</strong><span>Usable photometry only. Excludes flagged young stars, known or candidate multiples, and problematic G, J or W2 measurements.</span></div>
          <div class="model-flow-arrow" aria-hidden="true">→</div>
          <div>
            <div class="model-split" aria-label="Training and test split">
              <div class="model-split-card model-train"><small>03 · Split · Train · 80%</small><strong>{report['train_size']} stars</strong><span>Model fitting and five-fold cross-validation.</span></div>
              <div class="model-split-card model-test"><small>03 · Split · Test · 20%</small><strong>{report['test_size']} stars</strong><span>Held out from fitting to measure prediction errors.</span></div>
            </div>
            <div class="model-flow-outcome"><strong>Deploy:</strong> after evaluation, the quadratic G−J model is fitted to the full quality-controlled catalogue to predict M0–M5.5 subtypes.</div>
          </div>
        </div>
        ''', unsafe_allow_html=True)
        with st.expander("Training data and selection"):
            for exclusion in report.get("exclusions", []):
                st.caption(f"Excluded: {exclusion['Karmn']} — {exclusion['reason']}")
            st.markdown(
                "Source: [Cifuentes et al. (2020)](https://doi.org/10.1051/0004-6361/202038295). "
                "M0–M5.5; exclude flagged multiples, young stars and flagged G, J or W2 "
                "measurements. Both colours must be finite; duplicate Karmn IDs are rejected."
            )
            low, high = report["colour_range"]
            st.write(f"Accepted G−J range: {low:.3f}–{high:.3f} mag. No extrapolation in Photometry.")
            st.caption(
                "The common comparison sample requires W2 even for the one-colour models. "
                "No RUWE cut is applied. Selection favours well-behaved catalogue stars."
            )
            report_table(pd.DataFrame({
                "Spectral type": [f"M{float(k):.1f}" for k in report["counts"]],
                "Stars": list(report["counts"].values()),
            }))

    with st.container(border=True):
        st.markdown("#### Model comparison")
        st.write(
            "**MAE · Mean absolute error** is the average absolute difference between "
            "predicted and published spectral subtype. An MAE of 0.25 means an average "
            "error of a quarter of a subtype. Lower is better."
        )
        metrics = pd.DataFrame(report["metrics"])
        table = metrics.rename(columns={
            "model": "Model", "features": "Colours", "cv_mae": "CV MAE",
            "cv_std": "CV spread", "test_mae": "Test MAE", "within_one": "Within 1 subtype",
        })
        report_table(table, {
            "CV MAE": "{:.3f}".format, "CV spread": "{:.3f}".format,
            "Test MAE": "{:.3f}".format, "Within 1 subtype": "{:.1%}".format,
        })
        with st.expander("Evaluation protocol and model settings"):
            st.write(
                "One fixed 80/20 split, stratified by published subtype. Five-fold "
                "stratified cross-validation uses only the training partition. All models "
                "share the same folds and held-out test stars; seed 42. CV spread is the "
                "standard deviation across folds, not a prediction uncertainty."
            )
            st.write(
                "Linear: ordinary least squares. Quadratic: degree-two polynomial features "
                "and ordinary least squares. Random forest: 100 trees, minimum 5 samples "
                "per leaf, with G−J and J−W2. No hyperparameter search."
            )
            st.caption(
                "Metrics use continuous predictions before clipping or rounding. The forest "
                "also uses an extra colour, so this comparison cannot isolate the effect of "
                "algorithm choice. These are internal catalogue tests, not external validation."
            )
            st.caption(
                "The sample was curated after inspecting test residuals and checking revised "
                "catalogue labels. This regenerated test is not a new independent validation."
            )

    with st.container(border=True):
        st.markdown("#### Prediction diagnostics")
        model = st.selectbox("Model", metrics["model"].tolist(), index=1, key="models_diagnostic")
        actual = predictions["SpTnum"].to_numpy()
        predicted = predictions[model].to_numpy()
        residual = predicted - actual
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.patch.set_facecolor("#151517")
        colour = "#64dfb4"
        axes[0].scatter(actual, predicted, s=22, color=colour, alpha=0.6, edgecolors="none")
        axes[0].plot([0, 5.5], [0, 5.5], "--", color="#92929d", linewidth=1)
        axes[0].set(xlabel="Published subtype", ylabel="Predicted subtype")
        axes[1].scatter(actual, residual, s=22, color=colour, alpha=0.6, edgecolors="none")
        axes[1].axhline(0, color="#92929d", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Published subtype", ylabel="Predicted − published")
        for ax in axes:
            ax.set_facecolor("#151517")
            for spine in ax.spines.values():
                spine.set_color("#414147")
            ax.tick_params(colors="#c9c9d0")
            ax.xaxis.label.set_color("#dedee3")
            ax.yaxis.label.set_color("#dedee3")
            ax.grid(False)
            ax.set_xticks(range(6), [f"M{i}" for i in range(6)])
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Each point is a held-out star. Overlapping points can hide sample density.")
        with st.expander("Errors by spectral range · individual predictions"):
            rows = []
            for label, low, high in [("M0–M2", 0, 2), ("M2.5–M4", 2.5, 4), ("M4.5–M5.5", 4.5, 5.5)]:
                mask = (actual >= low) & (actual <= high)
                rows.append({"Range": label, "Test stars": int(mask.sum()),
                             "MAE": float(np.abs(residual[mask]).mean())})
            report_table(pd.DataFrame(rows), {"MAE": "{:.3f}".format})
            st.download_button("Download test predictions", predictions.to_csv(index=False, sep=";"),
                               file_name="runa_spectral_predictions.csv", mime="text/csv")
    st.caption(
        "Scope: ordinary M dwarfs. Sparse spectral bins and uncertain reference labels "
        "limit interpretation. A test MAE is not an uncertainty for an individual star."
    )
    with st.expander("Reproducibility"):
        st.write(f"scikit-learn {report['sklearn_version']} · seed {report['seed']}")
        st.caption("Generated offline with evaluate_models.py; opening Models does not retrain models.")
        st.code(f"Catalogue SHA-256: {report['data_sha256']}", language=None)


def _show_unresolved_binarity_analysis():
    """Document the grouped validation of the catalogue-multiplicity ML draft."""
    st.markdown("""<style>
    .model-context-card {border:1px solid #595443;border-radius:10px;
                         background:#18181b;padding:1rem 1.1rem;margin:.75rem 0;}
    .model-context-card small {display:block;color:#92929d;font-size:.76rem;
                               letter-spacing:.06em;text-transform:uppercase;margin-bottom:.42rem;}
    .model-context-card strong {display:block;color:#f4cf75;font-size:1.18rem;font-weight:550;}
    .model-context-card p {color:#b7b7c0;font-size:.92rem;line-height:1.45;margin:.45rem 0 0;}
    .binarity-model-flow {display:grid;grid-template-columns:minmax(0,1fr) 28px minmax(0,1.2fr) 28px minmax(0,1.45fr);align-items:stretch;gap:.5rem;margin:.8rem 0 1.1rem;}
    .binarity-model-step {min-width:0;border:1px solid #333338;border-radius:10px;padding:.9rem 1rem;background:#18181b;}
    .binarity-model-step small {display:block;color:#92929d;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.45rem;}
    .binarity-model-step strong {display:block;color:#e4e4e7;font-size:1.08rem;font-weight:550;}
    .binarity-model-step span {display:block;color:#92929d;font-size:.87rem;line-height:1.35;margin-top:.35rem;}
    .binarity-model-labels {border-color:#595443;}
    .binarity-model-labels strong {color:#f4cf75;}
    .binarity-model-arrow {align-self:center;justify-self:center;color:#72727c;}
    .binarity-model-split {display:grid;grid-template-columns:1fr 1fr;gap:.55rem;}
    .binarity-model-split-card {border:1px solid #333338;border-radius:10px;padding:.8rem .85rem;background:#18181b;}
    .binarity-model-split-card small {display:block;color:#92929d;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.4rem;}
    .binarity-model-split-card strong {display:block;color:#e4e4e7;font-size:1.03rem;font-weight:570;}
    .binarity-model-split-card span {display:block;color:#92929d;font-size:.82rem;line-height:1.35;margin-top:.3rem;}
    .binarity-model-train {border-color:#426b5b;}
    .binarity-model-train strong {color:#64dfb4;}
    .binarity-model-test {border-color:#5d5968;}
    .binarity-model-test strong {color:#c9c4e8;}
    .binarity-model-outcome {margin-top:.55rem;padding-top:.65rem;border-top:1px solid #29292d;color:#a1a1aa;font-size:.84rem;line-height:1.4;}
    .binarity-model-outcome strong {color:#e4e4e7;font-weight:550;}
    .binarity-sample-flow {display:grid;grid-template-columns:minmax(0,.85fr) 28px minmax(0,2.15fr) 28px minmax(0,1.2fr);align-items:stretch;gap:.5rem;margin:.8rem 0 .6rem;}
    .binarity-sample-source,.binarity-sample-ready {border:1px solid #333338;border-radius:10px;padding:.9rem 1rem;background:#18181b;}
    .binarity-sample-source small,.binarity-sample-ready small,.binarity-sample-decision > small {display:block;color:#92929d;font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.45rem;}
    .binarity-sample-source strong,.binarity-sample-ready strong {display:block;color:#e4e4e7;font-size:1.12rem;font-weight:550;line-height:1.18;}
    .binarity-sample-source span,.binarity-sample-ready span {display:block;color:#92929d;font-size:.83rem;line-height:1.35;margin-top:.38rem;}
    .binarity-sample-arrow {align-self:center;justify-self:center;color:#72727c;}
    .binarity-sample-decision {min-width:0;}
    .binarity-sample-branches {display:grid;grid-template-columns:1fr 1fr;gap:.5rem;}
    .binarity-sample-branch {border:1px solid #333338;border-radius:10px;padding:.78rem .85rem;background:#18181b;min-width:0;}
    .binarity-sample-branch small {display:block;color:#92929d;font-size:.7rem;letter-spacing:.05em;text-transform:uppercase;margin-bottom:.34rem;}
    .binarity-sample-branch strong {display:block;color:#e4e4e7;font-size:1.18rem;font-weight:570;line-height:1.05;}
    .binarity-sample-branch strong span {font-size:.82rem;font-weight:500;}
    .binarity-sample-branch p {color:#a1a1aa;font-size:.78rem;line-height:1.3;margin:.34rem 0 .48rem;}
    .binarity-sample-positive {border-color:#426b5b;}
    .binarity-sample-positive strong {color:#64dfb4;}
    .binarity-sample-negative {border-color:#5d5968;}
    .binarity-sample-negative strong {color:#c9c4e8;}
    .binarity-sample-excluded {grid-column:1 / -1;border-color:#3d3d44;padding:.62rem .85rem;}
    .binarity-sample-excluded strong {font-size:1rem;color:#c3c3ca;}
    .binarity-sample-excluded p {margin:.25rem 0 .42rem;}
    .binarity-sample-bar {height:4px;background:#29292d;border-radius:999px;overflow:hidden;}
    .binarity-sample-bar i {display:block;height:100%;border-radius:inherit;background:#64dfb4;}
    .binarity-sample-negative .binarity-sample-bar i {background:#c9c4e8;}
    .binarity-sample-excluded .binarity-sample-bar i {background:#72727c;}
    .binarity-feature-groups {display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.58rem;}
    .binarity-feature-groups span {border:1px solid #333338;border-radius:999px;padding:.25rem .5rem;color:#b7b7c0;font-size:.74rem;line-height:1.2;}
    .binarity-feature-groups b {color:#e4e4e7;font-weight:550;}
    .binarity-sample-note {color:#92929d;font-size:.78rem;line-height:1.35;margin:0;}
    @media(max-width:700px) {
        .binarity-model-flow {grid-template-columns:1fr;gap:.4rem;}
        .binarity-model-arrow {transform:rotate(90deg);}
        .binarity-model-split {grid-template-columns:1fr;}
        .binarity-sample-flow {grid-template-columns:1fr;gap:.4rem;}
        .binarity-sample-arrow {transform:rotate(90deg);}
    }
    </style>""", unsafe_allow_html=True)
    st.markdown(
        '<p class="model-subtitle">A supervised classifier for confirmed close unresolved '
        'components versus spectroscopically confirmed single stars.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-context-card"><small>Context</small>'
        '<strong>Single is an observational label</strong><p>Without detailed '
        'spectroscopic observations, a star can look alone while a companion remains '
        'hidden in its light or motion. This model uses Gaia DR3 astrometric, image-fitting '
        'and radial-velocity diagnostics to recognise patterns associated with confirmed '
        'close unresolved components. It supports follow-up prioritisation; it does not '
        'turn an absence of evidence into a physical verdict.</p></div>',
        unsafe_allow_html=True,
    )

    if not MULTIPLICITY_REPORT_FILE.exists():
        st.info("The offline multiplicity evaluation has not been generated yet.")
        return
    report = json.loads(MULTIPLICITY_REPORT_FILE.read_text(encoding="utf-8"))
    preparation = report["preparation"]

    with st.container(border=True):
        st.markdown("#### From catalogue evidence to a model score")
        train_rows = preparation["training_rows"] - report["validation"]["test_rows"]
        st.markdown(f'''
        <div class="binarity-model-flow" aria-label="Multiplicity model workflow">
          <div class="binarity-model-step"><small>01 · Observations</small><strong>Gaia DR3 diagnostics</strong><span>Astrometry, image fitting and radial-velocity measurements for each star.</span></div>
          <div class="binarity-model-arrow" aria-hidden="true">→</div>
          <div class="binarity-model-step binarity-model-labels"><small>02 · Component labels</small><strong>{preparation['confirmed_close_positive_rows']} close unresolved<br>{preparation['confirmed_single_negative_rows']} confirmed single</strong><span>Cifuentes nomenclature identifies the positive component; category 1–2 singles provide controls.</span></div>
          <div class="binarity-model-arrow" aria-hidden="true">→</div>
          <div>
            <div class="binarity-model-split" aria-label="Training and test split">
              <div class="binarity-model-split-card binarity-model-train"><small>03 · Train · 80%</small><strong>{train_rows} stars</strong><span>Four-fold grouped comparison of baseline, logistic regression and random forest.</span></div>
              <div class="binarity-model-split-card binarity-model-test"><small>03 · Test · 20%</small><strong>{report['validation']['test_rows']} stars</strong><span>{report['validation']['test_groups']} held-out groups. Components of one system stay together.</span></div>
            </div>
            <div class="binarity-model-outcome"><strong>In Multiplicity:</strong> RUNA shows this internal score and a 0.50 decision cut. External labels and calibration are needed before interpreting it as a probability.</div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Data preparation and feature selection")
        catalogue_rows = preparation["catalogue_rows"]
        positive_rows = preparation["confirmed_close_positive_rows"]
        negative_rows = preparation["confirmed_single_negative_rows"]
        excluded_rows = catalogue_rows - positive_rows - negative_rows
        st.markdown(f'''
        <div class="binarity-sample-flow" aria-label="Cifuentes 2025 component-level sampling">
          <div class="binarity-sample-source"><small>01 · Catalogue</small><strong>{catalogue_rows:,} components</strong><span>One row is one star, not one system.</span></div>
          <div class="binarity-sample-arrow" aria-hidden="true">→</div>
          <div class="binarity-sample-decision">
            <small>02 · Label decision in Cifuentes et al. (2025)</small>
            <div class="binarity-sample-branches">
              <div class="binarity-sample-branch binarity-sample-positive"><small>Positive · y = 1</small><strong>{positive_rows:,} <span>close unresolved</span></strong><p>Confirmed multiple; Aab, Aabc, (AB) …</p><div class="binarity-sample-bar"><i style="width:{positive_rows / catalogue_rows:.1%}"></i></div></div>
              <div class="binarity-sample-branch binarity-sample-negative"><small>Negative · y = 0</small><strong>{negative_rows:,} <span>confirmed single</span></strong><p>Single; detailed spectroscopy, category 1–2</p><div class="binarity-sample-bar"><i style="width:{negative_rows / catalogue_rows:.1%}"></i></div></div>
              <div class="binarity-sample-branch binarity-sample-excluded"><small>Not a training label · {excluded_rows:,} components</small><strong>Kept outside the score</strong><p>{preparation['resolved_multiple_context_rows']:,} resolved multiples · {preparation['unconfirmed_single_inference_rows']:,} Single category 3 · {preparation['candidate_inference_rows']:,} candidates / Multiple+</p><div class="binarity-sample-bar"><i style="width:{excluded_rows / catalogue_rows:.1%}"></i></div></div>
            </div>
          </div>
          <div class="binarity-sample-arrow" aria-hidden="true">→</div>
          <div class="binarity-sample-ready"><small>03 · Model input</small><strong>{preparation['training_rows']:,} labelled components</strong><span>System-aware split; imputation happens inside each training fold.</span><div class="binarity-feature-groups"><span><b>Astrometry</b> · RUWE</span><span><b>Image fit</b> · IPD</span><span><b>Radial velocity</b> · 4 fields</span><span><b>Source flag</b> · duplicate</span></div></div>
        </div>
        <p class="binarity-sample-note">Bars show each cohort's share of the 2,634-component catalogue. The Gaia non-single-star field is not used: it is incomplete in the local raw catalogue.</p>
        ''', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Model training and comparison")
        comparison = pd.DataFrame(report["comparison"]).rename(columns={
            "model": "Model", "f1_mean": "CV F1", "f1_std": "F1 spread",
            "roc_auc_mean": "CV ROC AUC", "balanced_accuracy_mean": "CV balanced accuracy",
            "precision_mean": "CV precision", "recall_mean": "CV recall",
        })
        report_table(
            comparison[["Model", "CV F1", "F1 spread", "CV ROC AUC", "CV balanced accuracy"]],
            {"CV F1": "{:.3f}".format, "F1 spread": "{:.3f}".format,
             "CV ROC AUC": "{:.3f}".format, "CV balanced accuracy": "{:.3f}".format},
        )
        st.caption(
            f"Selected model: {report['selected_model']}. Selection uses mean grouped "
            "cross-validation F1, then ROC AUC; no hyperparameter search was used."
        )

    if MULTIPLICITY_PREDICTIONS_FILE.exists():
        predictions = pd.read_csv(MULTIPLICITY_PREDICTIONS_FILE)
        actual = predictions["close_unresolved_label"].to_numpy(dtype=float)
        predicted = predictions["predicted_probability"].to_numpy(dtype=float)
        diagnostic_features = {
            "ruwe": ("RUWE", "astrometry", "log"),
            "ipd_harmonic_amplitude": ("IPD harmonic amplitude", "image fit", "log"),
            "ipd_multi_peak_fraction": ("IPD multi-peak fraction", "image fit", "log"),
            "rv_chisq_pvalue": ("RV chi-square p-value", "radial velocity", "linear"),
            "rv_renormalised_gof": ("RV renormalised goodness of fit", "radial velocity", "linear"),
            "rv_transits": ("RV transits", "radial velocity", "linear"),
            "radial_velocity_error": ("RV uncertainty", "radial velocity", "log"),
            "duplicated_source": ("Duplicated source", "source flag", "linear"),
        }
        feature_scores = []
        for key, (label, group, transform) in diagnostic_features.items():
            if key not in predictions:
                continue
            values = pd.to_numeric(predictions[key], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.sum() < 5 or np.unique(actual[valid]).size != 2:
                continue
            auc = float(roc_auc_score(actual[valid], values[valid]))
            direction = "higher → positive" if auc >= 0.5 else "lower → positive"
            feature_scores.append({
                "key": key, "label": label, "group": group, "transform": transform,
                "auc": max(auc, 1 - auc), "direction": direction,
                "n": int(valid.sum()), "values": values, "valid": valid,
            })
        feature_scores.sort(key=lambda item: item["auc"], reverse=True)
        with st.container(border=True):
            st.markdown("#### Prediction diagnostics")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.18]})
            fig.patch.set_facecolor("#151517")
            colour_positive = "#64dfb4"
            colour_negative = "#c9c4e8"
            class_names = ["Confirmed single", "Close unresolved"]
            class_scores = [predicted[actual == 0], predicted[actual == 1]]
            positions = [0, 1]
            boxes = axes[0].boxplot(
                class_scores, positions=positions, widths=.46, patch_artist=True,
                showfliers=False, medianprops={"color": "#f4cf75", "linewidth": 1.8},
                whiskerprops={"color": "#777780"}, capprops={"color": "#777780"},
            )
            for box, color in zip(boxes["boxes"], [colour_negative, colour_positive]):
                box.set_facecolor(color)
                box.set_alpha(.24)
                box.set_edgecolor(color)
            rng = np.random.default_rng(42)
            for index, (scores, color) in enumerate(zip(class_scores, [colour_negative, colour_positive])):
                x = rng.uniform(index - .13, index + .13, size=len(scores))
                axes[0].scatter(x, scores, s=25, color=color, alpha=.78,
                                edgecolors="#151517", linewidths=.35, zorder=3)
                axes[0].text(index, 1.04, f"n={len(scores)}", ha="center", va="bottom",
                             color="#a1a1aa", fontsize=9)
            axes[0].axhline(.5, color="#f4cf75", linestyle=(0, (3, 3)), linewidth=1)
            axes[0].set(
                title="Model score by reference label",
                ylabel="Close unresolved score", xlim=(-.55, 1.55), ylim=(-.04, 1.16),
            )
            axes[0].set_xticks(positions, class_names)
            if feature_scores:
                ranked = feature_scores[:8]
                yloc = np.arange(len(ranked))
                vals = [item["auc"] for item in ranked]
                axes[1].barh(yloc, np.array(vals) - .5, left=.5, height=.62,
                             color=[colour_positive if item["auc"] >= .65 else "#9ca3af" for item in ranked],
                             alpha=.82)
                axes[1].scatter(vals, yloc, s=32, color="#e4e4e7", zorder=3)
                axes[1].axvline(.5, color="#92929d", linestyle="--", linewidth=1)
                axes[1].set_yticks(
                    yloc,
                    [f"{item['label']}  ·  n={item['n']}" for item in ranked],
                )
                axes[1].invert_yaxis()
                axes[1].set_xlim(.45, 1.02)
                axes[1].set_xlabel("Single-feature AUC · direction chosen for display")
                axes[1].set_title("Which Gaia diagnostics separate the labels?")
                for y, item in zip(yloc, ranked):
                    axes[1].text(min(item["auc"] + .012, .985), y,
                                 f"{item['auc']:.2f}  {item['direction']}",
                                 va="center", ha="left", color="#d4d4d8", fontsize=8)
            else:
                axes[1].text(.5, .5, "Feature measurements are not available in the saved test file.",
                             ha="center", va="center", color="#a1a1aa", transform=axes[1].transAxes)
            for axis in axes:
                axis.set_facecolor("#151517")
                for spine in axis.spines.values():
                    spine.set_color("#414147")
                axis.tick_params(colors="#c9c9d0")
                axis.xaxis.label.set_color("#dedee3")
                axis.yaxis.label.set_color("#dedee3")
                axis.title.set_color("#e4e4e7")
                axis.title.set_fontsize(10)
                axis.tick_params(labelsize=8.5)
                axis.grid(axis="y" if axis is axes[0] else "x", color="#29292d", linewidth=.7)
                axis.set_axisbelow(True)
            axes[0].set_xticklabels(class_names, rotation=0, ha="center")
            axes[0].set_xlabel("")
            axes[1].tick_params(axis="y", labelsize=8.5, pad=4)
            axes[1].set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Left: each dot is one held-out star; amber marks the 0.5 decision threshold. "
                "Right: exploratory single-feature AUC on the held-out sample, using available "
                "measurements only (n shown). Values above 0.5 are oriented toward the more "
                "positive direction. Useful signals here include RUWE, RV goodness of fit and "
                "RV uncertainty; this small internal sample is not external validation."
            )

    with st.container(border=True):
        st.markdown("#### Validation")
        test = report["held_out_test"]
        test_metrics = st.columns(4)
        test_metrics[0].metric("Test F1", f"{test['f1']:.3f}")
        test_metrics[1].metric("Test ROC AUC", f"{test['roc_auc']:.3f}")
        test_metrics[2].metric("Test precision", f"{test['precision']:.3f}")
        test_metrics[3].metric("Test recall", f"{test['recall']:.3f}")
        st.write(
            f"The final check uses {report['validation']['test_rows']:,} rows from "
            f"{report['validation']['test_groups']:,} separate groups. Components of a "
            "known system cannot appear in both training and test partitions."
        )
        st.warning(
            "This is internal catalogue validation. Resolved systems, unconfirmed singles "
            "and Gaia candidates were not used for this metric. External labels and score "
            "calibration are still required before showing an individual probability."
        )
    st.caption(
        "Generated offline with evaluate_multiplicity_models.py; opening Models does not "
        "retrain the classifier. Source: Cifuentes et al. (2025)."
    )


def show_models():
    """Choose an inspectable RUNA machine-learning model."""
    choice = st.radio(
        "Choose a machine-learning application",
        ["Spectral type estimation", "Unresolved binarity analysis"],
        horizontal=True,
        key="models_model_choice",
    )
    if choice == "Spectral type estimation":
        _show_spectral_type_estimation()
    else:
        _show_unresolved_binarity_analysis()
