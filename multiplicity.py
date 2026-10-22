"""Gaia DR3 multiplicity indicators from Cifuentes et al. (2025)."""

import numpy as np
from astroquery.gaia import Gaia

GAIA_COLUMNS = (
    "ruwe", "ipd_gof_harmonic_amplitude", "ipd_frac_multi_peak",
    "rv_chisq_pvalue", "rv_renormalised_gof", "rv_nb_transits",
    "radial_velocity_error", "duplicated_source", "non_single_star",
)


def _number(value):
    """Return a Gaia value as float, or None for missing measurements."""
    try:
        if np.ma.is_masked(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def rvs_score(values):
    """Calculate the continuous RV variability score used in RUNA.

    Each sigmoid is centred on one part of the Gaia DR3 RV variability
    criterion: p-value 0.01, renormalised GOF 4, and ten transits.
    """
    p_value = values.get("rv_chisq_pvalue")
    gof = values.get("rv_renormalised_gof")
    transits = values.get("rv_nb_transits")
    if p_value is None or gof is None or transits is None:
        return None

    p_value = float(np.clip(float(p_value), 1e-300, 1.0))
    gof = float(gof)
    transits = float(transits)

    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))

    p_term = sigmoid(2.0 * (-np.log10(p_value) - 2.0))
    gof_term = sigmoid(gof - 4.0)
    transit_term = sigmoid(0.5 * (transits - 10.0))
    return float(p_term + gof_term + transit_term)


def query_rvs_reference(limit=20000):
    """Retrieve Gaia DR3 sources inside ten parsecs for comparison."""
    query = f"""
        SELECT TOP {int(limit)}
            source_id, parallax, ruwe,
            rv_chisq_pvalue, rv_renormalised_gof, rv_nb_transits
        FROM gaiadr3.gaia_source
        WHERE parallax > 100
          AND ruwe IS NOT NULL
          AND rv_chisq_pvalue IS NOT NULL
          AND rv_renormalised_gof IS NOT NULL
          AND rv_nb_transits IS NOT NULL
    """
    result = Gaia.launch_job_async(query).get_results()
    sources = []
    for row in result:
        values = {
            "ruwe": _number(row["ruwe"]),
            "rv_chisq_pvalue": _number(row["rv_chisq_pvalue"]),
            "rv_renormalised_gof": _number(row["rv_renormalised_gof"]),
            "rv_nb_transits": _number(row["rv_nb_transits"]),
        }
        score = rvs_score(values)
        if values["ruwe"] is None or score is None:
            continue
        sources.append({
            "source_id": str(row["source_id"]),
            "ruwe": values["ruwe"],
            "rvs": score,
        })
    return sources


def rvs_ruwe_plot(reference, target_values, target_name):
    """Plot the target against the ten-parsec Gaia RVS reference sample."""
    import matplotlib.pyplot as plt

    target_ruwe = _number(target_values.get("ruwe"))
    target_rvs = rvs_score(target_values)
    if target_ruwe is None or target_rvs is None:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")
    if reference:
        ax.scatter(
            [source["ruwe"] for source in reference],
            [source["rvs"] for source in reference],
            s=16, color="#d4d4d8", alpha=0.28, linewidths=0,
            label="Gaia DR3 < 10 pc",
        )
    ax.scatter(
        target_ruwe, target_rvs, s=105, color="#6ee7b7",
        edgecolor="#111113", linewidth=1.1, zorder=3,
        label=str(target_name),
    )
    ax.axvline(2.0, color="#d8a099", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.axhline(2.0, color="#d8a099", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(3.0, target_ruwe * 1.4))
    ax.set_ylim(0.0, 3.1)
    ax.set_xlabel("RUWE", fontsize=13, color="#d4d4d8")
    ax.set_ylabel("RV variability score (RVS)", fontsize=13, color="#d4d4d8")
    ax.tick_params(colors="#a1a1aa", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.legend(
        facecolor="#19191e", edgecolor="#3f3f46", labelcolor="#d4d4d8",
        frameon=True, fontsize=9, loc="upper left",
    )
    fig.tight_layout()
    return fig


def query_gaia(source_id):
    """Retrieve the Gaia fields used by the seven published criteria."""
    if not str(source_id).isdigit():
        raise ValueError("A valid Gaia DR3 source ID is required.")

    columns = ", ".join(GAIA_COLUMNS)
    query = f"SELECT {columns} FROM gaiadr3.gaia_source WHERE source_id = {source_id}"
    result = Gaia.launch_job(query).get_results()
    if len(result) == 0:
        return None

    row = result[0]
    values = {}
    for column in GAIA_COLUMNS:
        value = row[column]
        if np.ma.is_masked(value) or value is None:
            values[column] = None
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            values[column] = None
        else:
            values[column] = value.item() if hasattr(value, "item") else value
    return values


def evaluate_criteria(values):
    """Apply the seven cuts in Table 3, keeping missing measurements unknown."""
    ruwe = values["ruwe"]
    ipd_amplitude = values["ipd_gof_harmonic_amplitude"]
    multi_peak = values["ipd_frac_multi_peak"]
    rv_pvalue = values["rv_chisq_pvalue"]
    rv_gof = values["rv_renormalised_gof"]
    rv_transits = values["rv_nb_transits"]
    rv_error = values["radial_velocity_error"]
    duplicated = values["duplicated_source"]
    non_single = values["non_single_star"]

    criteria = {}
    criteria[1] = ruwe > 2 if ruwe is not None else None
    if (ipd_amplitude is not None and ipd_amplitude <= 0.1) or (
        ruwe is not None and ruwe <= 1.4
    ):
        criteria[2] = False
    elif ipd_amplitude is None or ruwe is None:
        criteria[2] = None
    else:
        criteria[2] = True
    criteria[3] = multi_peak > 30 if multi_peak is not None else None
    if (rv_pvalue is not None and rv_pvalue >= 0.01) or (
        rv_gof is not None and rv_gof <= 4
    ) or (rv_transits is not None and rv_transits < 10):
        criteria[4] = False
    elif rv_pvalue is None or rv_gof is None or rv_transits is None:
        criteria[4] = None
    else:
        criteria[4] = True
    criteria[5] = rv_error >= 10 if rv_error is not None else None
    criteria[6] = bool(duplicated) if duplicated is not None else None
    criteria[7] = bool(non_single) if non_single is not None else None
    return criteria


def check_multiplicity(gaia_id):
    """Check the loaded star without resolving its name a second time."""
    values = query_gaia(gaia_id)
    if values is None:
        raise ValueError("No Gaia DR3 source was found for this identifier.")
    return values, evaluate_criteria(values)
