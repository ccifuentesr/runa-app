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
    .model-flow {display:flex;align-items:stretch;gap:.65rem;margin:.8rem 0 1.1rem;}
    .model-flow-step {flex:1;min-width:0;border:1px solid #333338;border-radius:10px;padding:.9rem 1rem;background:#18181b;}
    .model-flow-step small {display:block;color:#92929d;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.45rem;}
    .model-flow-step strong {display:block;color:#e4e4e7;font-size:1.15rem;font-weight:550;}
    .model-flow-step span {display:block;color:#92929d;font-size:.87rem;margin-top:.35rem;}
    .model-flow-step:last-child {border-color:#426b5b;}
    .model-flow-step:last-child strong {color:#64dfb4;}
    .model-flow-arrow {align-self:center;color:#72727c;}
    @media(max-width:700px) {
        .model-flow {flex-direction:column;gap:.4rem;}
        .model-flow-arrow {transform:rotate(90deg);}
    }
    </style>""", unsafe_allow_html=True)
    st.subheader("Spectral type estimation")
    st.caption("A supervised regression from photometric colours to M-dwarf subtype.")
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
        st.markdown('''
        <div class="model-flow" aria-label="Prediction workflow">
          <div class="model-flow-step"><small>01 · Observations</small><strong>G and J</strong><span>Measured magnitudes</span></div>
          <div class="model-flow-arrow" aria-hidden="true">→</div>
          <div class="model-flow-step"><small>02 · Feature</small><strong>G−J colour</strong><span>Difference in magnitudes</span></div>
          <div class="model-flow-arrow" aria-hidden="true">→</div>
          <div class="model-flow-step"><small>03 · Model</small><strong>Quadratic regression</strong><span>Learned from labelled stars</span></div>
          <div class="model-flow-arrow" aria-hidden="true">→</div>
          <div class="model-flow-step"><small>04 · Prediction</small><strong>Spectral subtype</strong><span>M0–M5.5 · half-subtype steps</span></div>
        </div>
        ''', unsafe_allow_html=True)
        st.caption(
            "Photometry currently uses the quadratic model, fitted to the full filtered "
            "catalogue. Predictions are bounded to M0–M5.5 and rounded to half a subtype. "
            "The evaluation below uses separate training and test stars."
        )
        columns = st.columns(3)
        for column, label, value in zip(columns, ["Filtered stars", "Training", "Held-out test"],
                                        [report["sample_size"], report["train_size"], report["test_size"]]):
            column.metric(label, value)
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
        st.caption("Lower MAE is better. Errors are measured in spectral subtypes.")
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
            st.download_button("Download test predictions", predictions.to_csv(index=False),
                               file_name="runa_spectral_predictions.csv", mime="text/csv")
    st.caption(
        "Scope: ordinary M dwarfs. Sparse spectral bins and uncertain reference labels "
        "limit interpretation. A test MAE is not an uncertainty for an individual star."
    )
    with st.expander("Reproducibility"):
        st.write(f"scikit-learn {report['sklearn_version']} · seed {report['seed']}")
        st.caption("Generated offline with evaluate_models.py; opening Models does not retrain models.")
        st.code(f"Catalogue SHA-256: {report['data_sha256']}", language=None)
