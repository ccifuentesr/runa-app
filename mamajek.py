"""Local effective-temperature lookup from the Mamajek dwarf sequence."""

import csv
import re
from functools import lru_cache
from pathlib import Path


DATA_FILE = Path(__file__).with_name("data") / "mamajek_dwarf_sequence.csv"
SPECTRAL_TYPE_PATTERN = re.compile(
    r"([OBAFGKMLTY])\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)


def spectral_type_key(spectral_type):
    """Normalise a spectral type to its dwarf-sequence lookup key."""
    match = SPECTRAL_TYPE_PATTERN.search(str(spectral_type or ""))
    if match is None:
        return None
    spectral_class = match.group(1).upper()
    subtype = float(match.group(2))
    return spectral_class, subtype, f"{spectral_class}{subtype:g}V"


@lru_cache(maxsize=1)
def load_mamajek_sequence():
    """Load the full adapted Mamajek dwarf sequence from local data."""
    with DATA_FILE.open(newline="", encoding="utf-8") as source:
        return tuple(dict(row) for row in csv.DictReader(source))


@lru_cache(maxsize=1)
def _sequence():
    return {
        row["spectral_type"]: row
        for row in load_mamajek_sequence()
    }


def temperature_from_mamajek(spectral_type):
    """Return an exact dwarf-sequence temperature for a spectral subtype.

    The full local CSV retains the source's stellar, photometric and colour
    columns. The source recommends citing Pecaut & Mamajek (2013) until an
    updated paper exists.
    """
    key = spectral_type_key(spectral_type)
    if key is None:
        return None
    spectral_class, _, lookup_key = key
    row = _sequence().get(lookup_key)
    if row is None:
        return None
    return {
        "value": float(row["teff_k"]),
        "error": None,
        "spectral_type": row["spectral_type"],
        "spectral_class": spectral_class,
        "reference": f'{row["reference"]} · Mamajek dwarf sequence',
    }
