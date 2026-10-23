"""Build a VOSA photometry input file from RUNA catalogue records."""

import math
import re
from catalogue_selection import preferred_photometry


# Mapping reused from cafe_tess/cafe.prepare_vosa.py.
VOSA_FILTERS = {
    ("APASS DR9", "B"): "Generic/Johnson.B",
    ("APASS DR9", "V"): "Generic/Johnson.V",
    ("APASS DR9", "g"): "Misc/APASS.sdss_g",
    ("APASS DR9", "r"): "Misc/APASS.sdss_r",
    ("APASS DR9", "i"): "Misc/APASS.sdss_i",
    ("Gaia DR3", "BP"): "GAIA/GAIA3.Gbp",
    ("Gaia DR3", "G"): "GAIA/GAIA3.G",
    ("Gaia DR3", "RP"): "GAIA/GAIA3.Grp",
    ("UCAC4", "B"): "Generic/Johnson.B",
    ("UCAC4", "V"): "Generic/Johnson.V",
    ("UCAC4", "g"): "Misc/UCAC.sdss_g",
    ("UCAC4", "r"): "Misc/UCAC.sdss_r",
    ("UCAC4", "i"): "Misc/UCAC.sdss_i",
    ("SDSS9", "u"): "SLOAN/SDSS.u",
    ("SDSS9", "g"): "SLOAN/SDSS.g",
    ("SDSS9", "r"): "SLOAN/SDSS.r",
    ("SDSS9", "i"): "SLOAN/SDSS.i",
    ("SDSS9", "z"): "SLOAN/SDSS.z",
    ("2MASS", "J"): "2MASS/2MASS.J",
    ("2MASS", "H"): "2MASS/2MASS.H",
    ("2MASS", "K"): "2MASS/2MASS.Ks",
    ("AllWISE", "W1"): "WISE/WISE.W1",
    ("AllWISE", "W2"): "WISE/WISE.W2",
    ("AllWISE", "W3"): "WISE/WISE.W3",
    ("AllWISE", "W4"): "WISE/WISE.W4",
}

GAIA_ERROR_FLOORS = {
    "BP": 0.0027901700,
    "G": 0.0027553202,
    "RP": 0.0037793818,
}


def _number(value):
    """Return a finite float, or None for unavailable catalogue values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_vosa_input(star, records):
    """Return a VOSA-compatible text file and its small provenance summary."""
    gaia = star.get("gaia_photometry", {})
    parallax = _number(gaia.get("parallax", star.get("plx_value")))
    parallax_error = _number(gaia.get("parallax_error", star.get("plx_err")))
    if parallax is None or parallax <= 0:
        return None

    distance = 1000.0 / parallax
    distance_error = (
        distance * parallax_error / parallax
        if parallax_error is not None else None
    )
    distance_text = (
        f"{distance:.3f}+-{distance_error:.3f}"
        if distance_error is not None else f"{distance:.3f}"
    )
    object_name = re.sub(r"\s+", "_", str(star["main_id"]).strip())
    lines = []
    for record in preferred_photometry(records, allowed_filters=VOSA_FILTERS):
        catalogue = record.get("catalogue")
        for band in record.get("bands", []):
            vosa_filter = VOSA_FILTERS.get((catalogue, band.get("label")))
            magnitude = _number(band.get("value"))
            if vosa_filter is None or magnitude is None:
                continue

            error = _number(band.get("error"))
            if catalogue == "Gaia DR3" and error is None:
                error = GAIA_ERROR_FLOORS[band["label"]]
            if error is None or error <= 0:
                continue

            lines.append(
                f"{object_name} {float(star['ra']):.6f} {float(star['dec']):.6f} "
                f"{distance_text} --- {vosa_filter} {magnitude:.3f} "
                f"{error:.3f} --- ---"
            )

    if not lines:
        return None
    return {
        "content": "\n".join(lines) + "\n",
        "filename": f"{object_name}_vosa.txt",
        "bands": len(lines),
        "distance": distance,
        "distance_error": distance_error,
    }
