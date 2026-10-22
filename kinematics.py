"""Kinematic tools used by the RUNA Kinematics section.

The BANYAN Sigma call follows the small-star version already used in
``cafe.analysis_banyan_sigma.py`` and ``kobe.analysis_banyan_sigma.py``.
"""

import numpy as np

try:
    from banyan_sigma import membership_probability
except ImportError:  # Keep the rest of RUNA usable without this optional tool.
    membership_probability = None


YA_AGES = {
    "TWA": 10, "BPMG": 24, "THOR": 24, "COL": 42, "CAR": 45,
    "ARG": 40, "ABDMG": 149, "CARN": 200, "THA": 45, "UCL": 16,
    "LCC": 15, "USCO": 11, "CRA": 10, "IC2391": 50, "IC2602": 46,
    "EPSC": 3.7, "ETAC": 11, "OCT": 35, "PLE": 112, "HYA": 625,
    "PRAE": 590, "UMA": 414, "CBER": 562, "118TAU": 10,
    "PL8": 60, "ROPH": 2, "TAU": 1, "MUTAU": 60,
    "PERI": 60, "UCRA": 10, "VCA": 11, "XFOR": 500,
}


def _number(value):
    """Return a finite float, or ``None`` for a missing catalogue value."""
    if value is None or np.ma.is_masked(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def banyan_inputs(star):
    """Prepare the Gaia/SIMBAD values accepted by BANYAN Sigma."""
    gaia = star.get("gaia_photometry") or {}
    values = {
        "ra": _number(gaia.get("ra", star.get("ra"))),
        "dec": _number(gaia.get("dec", star.get("dec"))),
        "pmra": _number(gaia.get("pmra", star.get("pmra"))),
        "pmdec": _number(gaia.get("pmdec", star.get("pmdec"))),
        "epmra": _number(gaia.get("pmra_error", star.get("pm_err_maj"))),
        "epmdec": _number(
            gaia.get("pmdec_error", star.get("pm_err_min", star.get("pm_err_maj")))
        ),
        "plx": _number(gaia.get("parallax", star.get("plx_value"))),
        "eplx": _number(gaia.get("parallax_error", star.get("plx_err"))),
        "rv": _number(star.get("rvz_radvel")),
        "erv": _number(star.get("rvz_err")),
    }
    return values


def _normalise_probabilities(result):
    """Convert BANYAN's association log-probabilities to linear values."""
    young = {}
    for column in result.columns:
        association, field = column
        if field == "LN_P" and association not in (
            "ALL", "FIELD", "BEST_HYP", "BEST_YA", "LIST_PROB_YAS", "YA_PROB"
        ):
            young[association] = float(result[column].iloc[0])

    if not young:
        return {}, 1.0

    finite_values = [value for value in young.values() if np.isfinite(value)]
    if not finite_values:
        return {}, 1.0
    ln_max = np.max(finite_values)
    probabilities = {key: np.exp(value - ln_max) for key, value in young.items()}
    field_log_probability = float(result[("FIELD", "LN_P")].iloc[0])
    total = sum(probabilities.values()) + np.exp(field_log_probability - ln_max)
    probabilities = {key: value / total for key, value in probabilities.items()}
    field_probability = np.exp(field_log_probability - ln_max) / total
    return probabilities, field_probability


def run_banyan(star):
    """Run BANYAN Sigma for the selected star and return a compact result."""
    global membership_probability
    if membership_probability is None:
        try:
            from banyan_sigma import membership_probability as banyan_call
        except ImportError:
            banyan_call = None
        membership_probability = banyan_call
    if membership_probability is None:
        raise RuntimeError(
            "BANYAN Sigma is not installed. Add banyan_sigma to the RUNA environment."
        )

    values = banyan_inputs(star)
    required = ("ra", "dec", "pmra", "pmdec", "epmra", "epmdec")
    if any(values[key] is None for key in required):
        raise ValueError(
            "Gaia coordinates, proper motion and their uncertainties are needed."
        )

    kwargs = {key: values[key] for key in required}
    if values["plx"] is not None and values["eplx"] is not None and values["plx"] > 0:
        kwargs.update(plx=values["plx"], eplx=values["eplx"], use_plx=True)
    if values["rv"] is not None and values["erv"] is not None and values["erv"] > 0:
        kwargs.update(rv=values["rv"], erv=values["erv"], use_rv=True)

    result = membership_probability(**kwargs)
    probabilities, field_probability = _normalise_probabilities(result)
    best_hypothesis = str(result[("BEST_HYP", "Global")].iloc[0])
    best_association = str(result[("BEST_YA", "Global")].iloc[0])
    hypothesis_key = (
        best_hypothesis if best_hypothesis != "FIELD" else best_association
    )

    output = {
        "inputs": values,
        "best_hypothesis": best_hypothesis,
        "best_association": best_association,
        "association_probability": probabilities.get(best_association, 0.0),
        "field_probability": field_probability,
        "probabilities": sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        ),
        "values": result,
    }
    for name in ("U", "V", "W", "EU", "EV", "EW", "X", "Y", "Z"):
        try:
            output[name] = float(result[(hypothesis_key, name)].iloc[0])
        except (KeyError, TypeError, ValueError):
            output[name] = None

    output["age_myr"] = YA_AGES.get(best_hypothesis, YA_AGES.get(best_association))
    return output
