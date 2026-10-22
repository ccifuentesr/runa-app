"""Look up confirmed planets in the NASA Exoplanet Archive."""

import json
from urllib.parse import urlencode
from urllib.request import urlopen


NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def _clean_candidate(value):
    """Remove the small amount of SIMBAD formatting used in object names."""
    if value is None:
        return None
    value = str(value).strip()
    if value.startswith("NAME "):
        value = value[5:]
    return value or None


def _candidate_names(query_name, main_id, identifiers):
    """Return likely archive host names without querying a name resolver."""
    candidates = []
    for value in (query_name, main_id):
        value = _clean_candidate(value)
        if value and value not in candidates:
            candidates.append(value)

    useful_prefixes = (
        "HD ", "HIP ", "GJ ", "GL ", "LHS ", "LP ", "WOLF ",
        "ROSS ", "TOI-", "KEPLER-", "K2-", "WASP-", "HAT-", "TIC ",
        "KIC ", "HR ",
    )
    for identifier in identifiers or []:
        identifier = _clean_candidate(identifier)
        if not identifier:
            continue
        if identifier.upper().startswith(useful_prefixes):
            if identifier not in candidates:
                candidates.append(identifier)
    return candidates[:12]


def _tap_query(names):
    """Build one small TAP query for the candidate host names."""
    quoted_names = [name.upper().replace("'", "''") for name in names]
    conditions = [f"UPPER(hostname) = '{name}'" for name in quoted_names]
    where = " OR ".join(conditions)
    return (
        "SELECT TOP 100 hostname, pl_name, discoverymethod, disc_year, "
        "pl_orbper, pl_rade, pl_masse, sy_pnum, sy_dist "
        "FROM pscomppars WHERE " + where + " ORDER BY pl_name"
    )


def lookup_planets(query_name, main_id, identifiers=None):
    """Return confirmed planets associated with the loaded star."""
    names = _candidate_names(query_name, main_id, identifiers)
    if not names:
        return []

    params = urlencode({"query": _tap_query(names), "format": "json"})
    with urlopen(f"{NASA_TAP_URL}?{params}", timeout=20) as response:
        rows = json.load(response)
    return rows
