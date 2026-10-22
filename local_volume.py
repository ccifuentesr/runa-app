"""Checks for stars in the published 10 pc samples."""

import csv
import math
import re
from functools import lru_cache
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
TEN_PC_FILE = "gonzalez-payo25_dataset-1.csv"
MULTIPLE_FILE = "gonzalez-payo25_dataset-2.csv"


def _normalise_identifier(value):
    """Return a simple form that can be compared across catalogues."""
    text = str(value or "").lower().replace("−", "-")
    parts = re.findall(r"[a-z]+|\d+", text)
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def _star_names(star):
    names = [star.get("main_id"), star.get("query_name")]
    names.extend(star.get("identifiers", []))
    return {_normalise_identifier(name) for name in names if name}


def _row_names(row):
    names = [row.get("Name"), row.get("AName"), row.get("SimbadName")]
    if row.get("GJ"):
        names.append("GJ " + row["GJ"].strip())
    return {_normalise_identifier(name) for name in names if name}


@lru_cache(maxsize=None)
def _catalogue_rows(filename):
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _coordinate_degrees(ra_text, dec_text):
    """Convert catalogue sexagesimal coordinates to decimal degrees."""
    try:
        ra_parts = [float(value) for value in str(ra_text).split()]
        dec_parts = str(dec_text).split()
        if len(ra_parts) != 3 or len(dec_parts) != 3:
            return None
        sign = -1 if dec_parts[0].startswith("-") else 1
        dec_values = [abs(float(value)) for value in dec_parts]
    except (TypeError, ValueError):
        return None

    ra = 15 * (ra_parts[0] + ra_parts[1] / 60 + ra_parts[2] / 3600)
    dec = sign * (dec_values[0] + dec_values[1] / 60 + dec_values[2] / 3600)
    return ra, dec


def _separation_arcsec(ra1, dec1, ra2, dec2):
    """Return the angular distance between two positions."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    return math.degrees(angle) * 3600


def _matching_row(star, filename):
    star_names = _star_names(star)
    for row in _catalogue_rows(filename):
        if star_names.intersection(_row_names(row)):
            return row

    if star.get("ra") is None or star.get("dec") is None:
        return None
    for row in _catalogue_rows(filename):
        position = _coordinate_degrees(row.get("RAJ2000"), row.get("DEJ2000"))
        if position is None:
            continue
        separation = _separation_arcsec(
            float(star["ra"]), float(star["dec"]), position[0], position[1]
        )
        if separation < 5:
            return row
    return None


def local_volume_status(star):
    """Describe the star's place in the published 10 pc catalogues."""
    distance = star.get("distance_pc")
    if distance is None:
        return {
            "distance_pc": None,
            "inside_10pc": None,
            "sample_match": None,
            "multiple_match": None,
        }

    distance = float(distance)
    if distance >= 10:
        return {
            "distance_pc": distance,
            "inside_10pc": False,
            "sample_match": None,
            "multiple_match": None,
        }

    return {
        "distance_pc": distance,
        "inside_10pc": True,
        "sample_match": _matching_row(star, TEN_PC_FILE),
        "multiple_match": _matching_row(star, MULTIPLE_FILE),
    }
