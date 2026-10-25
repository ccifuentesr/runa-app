"""Reference data from Cifuentes et al. (2020), Table 6."""

import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
TABLE6_FILE = DATA_DIR / "cifuentes20_table6.csv"
TABLE7_FILE = DATA_DIR / "cifuentes20_table7.csv"


def load_table6():
    """Load the published average parameters for K5 V to L2 dwarfs."""
    with TABLE6_FILE.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_table7():
    """Load the published average absolute magnitudes for K5 V to L2 dwarfs."""
    with TABLE7_FILE.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_reference_data():
    """Combine Tables 6 and 7 by spectral type for the Data view."""
    magnitudes = {row["sp_type"]: row for row in load_table7()}
    combined = []
    for parameters in load_table6():
        row = dict(parameters)
        row.update({
            key: value
            for key, value in magnitudes.get(parameters["sp_type"], {}).items()
            if key != "sp_type"
        })
        combined.append(row)
    return combined


def temperature_from_spectral_type(spectral_type):
    """Return the Table 6 temperature for one tabulated spectral subtype."""
    match = re.search(r"([KML])\s*(\d+(?:\.\d+)?)", str(spectral_type or "").upper())
    if match is None:
        return None

    spectral_class = match.group(1)
    subtype = float(match.group(2))
    for row in load_table6():
        row_match = re.search(r"([KML])\s*(\d+(?:\.\d+)?)", row["sp_type"])
        if row_match is None:
            continue
        if row_match.group(1) != spectral_class:
            continue
        if float(row_match.group(2)) != subtype:
            continue
        return {
            "value": float(row["teff"]),
            "error": float(row["teff_error"]),
            "spectral_type": row["sp_type"],
        }
    return None
