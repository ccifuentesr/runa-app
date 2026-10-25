"""Display saved spectral-model evaluation without training during page visits."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from evaluate_models import REPORT_DIR
from train_model import DATA_FILE, EXCLUSIONS_FILE


def report_table(frame, formats=None):
    """Render a compact static table in the app's existing visual style."""
    st.markdown(
        '<div class="model-report-table">'
        + frame.to_html(index=False, border=0, formatters=formats)
        + '</div>', unsafe_allow_html=True,
    )


def show_models():
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
    @media(max-width:700px) {
        .model-flow {grid-template-columns:1fr;gap:.4rem;}
        .model-flow-arrow {transform:rotate(90deg);}
        .model-split {grid-template-columns:1fr;}
    }
    </style>""", unsafe_allow_html=True)
    st.subheader("Spectral type estimation")
    st.markdown(
        '<p class="model-subtitle">A supervised regression from photometric colours '
        'to M-dwarf subtype.</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("#### Context")
        st.write(
            "Light is our main source of information about stars. Photometric filters "
            "measure that light across different wavelength bands; comparing their "
            "brightness gives us a star's colours. This model uses those colours to "
            "estimate spectral type, offering a fast alternative when spectroscopy "
            "is unavailable or would require costly telescope time. Spectral type "
            "also helps estimate fundamental properties such as temperature, mass "
            "and radius through calibrated relations. But, reddening, metallicity, "
            "unresolved multiplicity and poor photometry can shift colours. "
            "For this reason, a photometric estimate supports initial characterisation, "
            "but it does not replace the detail or diagnostic power of a spectrum."
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
