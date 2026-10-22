"""Published bolometric luminosities included with RUNA."""

import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
CATALOGUES = (
    ("cifuentes25_dataset.csv", "Cifuentes et al. (2025)", "GaiaDR3", "LBol", "e_LBol"),
    ("cifuentes20_dataset.csv", "Cifuentes et al. (2020)", None, "Lbol", "e_Lbol"),
)


def _normalise_identifier(value):
    """Normalise common stellar names while retaining numeric groups."""
    text = str(value or "").lower().replace("−", "-")
    parts = re.findall(r"[a-z]+|\d+", text)
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def _float_or_none(value):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


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


def published_luminosities(star):
    """Return matching, published Cifuentes luminosities in solar units."""
    if not star:
        return []

    results = []
    for filename, reference, gaia_column, value_column, error_column in CATALOGUES:
        for row in _catalogue_rows(filename):
            if not _matches(row, star, gaia_column):
                continue
            value = _float_or_none(row.get(value_column))
            if value is None:
                continue
            results.append({
                "reference": reference,
                "value": value,
                "error": _float_or_none(row.get(error_column)),
                "name": row.get("Name") or row.get("SimbadName") or row.get("Karmn"),
            })
            break
    return results
