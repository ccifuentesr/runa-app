"""Published stellar parameters included with RUNA."""

import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
CATALOGUES = (
    {
        "filename": "cifuentes25_dataset.csv",
        "reference": "Cifuentes et al. (2025)",
        "gaia_column": "GaiaDR3",
        "columns": {
            "luminosity": ("LBol", "e_LBol"),
            "radius": ("Rad", "e_Rad"),
            "mass": ("Mass", "e_Mass"),
            "effective_temperature": ("Teff", "e_Teff"),
            "metallicity": ("[Fe/H]", None),
        },
    },
    {
        "filename": "cifuentes20_dataset.csv",
        "reference": "Cifuentes et al. (2020)",
        "gaia_column": None,
        "columns": {
            "luminosity": ("Lbol", "e_Lbol"),
            "radius": ("Radius", "e_Radius"),
            "mass": ("Mass", "e_Mass"),
            "effective_temperature": ("Teff", None),
        },
    },
)


def _normalise_identifier(value):
    """Normalise common stellar names while retaining numeric groups."""
    text = str(value or "").lower().replace("−", "-")
    parts = re.findall(r"[a-z]+|\d+", text)
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def _float_or_none(value, positive=False):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if positive and number <= 0:
        return None
    return number


def _catalogue_rows(filename):
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _matches(row, star, gaia_column):
    gaia_id = str(star.get("gaia_id") or "").strip()
    if gaia_column and gaia_id and row.get(gaia_column, "").strip() == gaia_id:
        return True

    names = [star.get("main_id"), star.get("query_name"), *star.get("identifiers", [])]
    normalised_names = {_normalise_identifier(name) for name in names if name}
    candidates = [row.get("Name"), row.get("SimbadName"), row.get("Karmn"), row.get("GJ")]
    return any(
        _normalise_identifier(candidate) in normalised_names
        for candidate in candidates if candidate
    )


def published_multiplicity(star):
    """Return the published multiplicity description for a matched star."""
    if not star:
        return None
    for row in _catalogue_rows("cifuentes25_dataset.csv"):
        if _matches(row, star, "GaiaDR3"):
            return {
                "type": row.get("Type", "").strip(),
                "sb": row.get("SB", "").strip(),
                "candidate": row.get("Candidate", "").strip() == "T",
                "reference": "Cifuentes et al. (2025)",
            }
    return None


def published_stellar_parameters(star):
    """Return matching published values grouped by stellar parameter."""
    if not star:
        return {}

    results = {
        "luminosity": [],
        "radius": [],
        "mass": [],
        "effective_temperature": [],
        "metallicity": [],
    }
    for catalogue in CATALOGUES:
        for row in _catalogue_rows(catalogue["filename"]):
            if not _matches(row, star, catalogue["gaia_column"]):
                continue
            for parameter, columns in catalogue["columns"].items():
                value_column, error_column = columns
                value = _float_or_none(
                    row.get(value_column),
                    positive=parameter != "metallicity",
                )
                if value is None:
                    continue
                error = None
                if error_column:
                    error = _float_or_none(row.get(error_column), positive=True)
                results[parameter].append({
                    "reference": catalogue["reference"],
                    "value": value,
                    "error": error,
                    "name": (
                        row.get("Name") or row.get("SimbadName") or row.get("Karmn")
                    ),
                })
            break
    return results
