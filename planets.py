"""Look up planets in NASA's archive and the exoplanet.eu catalogue."""

import csv
from io import StringIO
import json
import math
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOPLANET_EU_CSV_URL = "https://exoplanet.eu/catalog/csv/"
JUPITER_TO_EARTH_RADII = 11.209
JUPITER_TO_EARTH_MASSES = 317.828
PROXIMA_GAIA_DR3_ID = "5853498713190525696"
PROXIMA_C_SOURCE = "https://www.exoplanet.eu/catalog/proxima_centauri_c--7082/"


def _clean_candidate(value):
    """Remove the small amount of SIMBAD formatting used in object names."""
    if value is None:
        return None
    value = str(value).strip()
    if value.startswith("NAME "):
        value = value[5:]
    return value or None


def _candidate_names(query_name, main_id, identifiers):
    """Return likely catalogue host names without querying a name resolver."""
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
        raw = str(identifier).strip()
        value = _clean_candidate(raw)
        if not value:
            continue
        if raw.upper().startswith("NAME ") or value.upper().startswith(useful_prefixes):
            if value not in candidates:
                candidates.append(value)
    return candidates[:16]


def _tap_query(names, gaia_id=None):
    """Build a small TAP query using host aliases and an exact Gaia DR3 ID."""
    conditions = []
    for name in names:
        quoted = name.upper().replace("'", "''")
        conditions.append(f"UPPER(hostname) = '{quoted}'")
    if gaia_id is not None and str(gaia_id).isdigit():
        conditions.append(f"gaia_dr3_id = 'Gaia DR3 {gaia_id}'")
    return (
        "SELECT TOP 100 hostname, pl_name, discoverymethod, disc_year, "
        "pl_orbper, pl_orbsmax, pl_orbeccen, pl_rade, pl_masse, "
        "pl_bmasse, pl_bmassprov, pl_bmasselim, sy_pnum, sy_dist FROM pscomppars WHERE "
        + " OR ".join(conditions)
        + " ORDER BY pl_name"
    )


def lookup_planets(query_name, main_id, identifiers=None, gaia_id=None):
    """Return confirmed planets from NASA's PSCompPars table."""
    names = _candidate_names(query_name, main_id, identifiers)
    if not names and not (gaia_id and str(gaia_id).isdigit()):
        return []

    params = urlencode({"query": _tap_query(names, gaia_id), "format": "json"})
    with urlopen(f"{NASA_TAP_URL}?{params}", timeout=20) as response:
        rows = json.load(response)
    for row in rows:
        row["status"] = "Confirmed"
        row["source"] = "NASA Exoplanet Archive"
    return rows


def _number(value):
    """Convert a catalogue number, preserving unavailable values as None."""
    try:
        number = float(value) if value not in (None, "", "—") else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def lookup_exoplanet_eu(query_name, main_id, identifiers=None):
    """Return planet records from exoplanet.eu's filtered CSV export."""
    names = _candidate_names(query_name, main_id, identifiers)[:8]
    if not names:
        return []
    filters = []
    for name in names:
        safe_name = name.replace('"', "")
        filters.append(f'star_name = "{safe_name}"')
    url = EXOPLANET_EU_CSV_URL + "?" + urlencode({"query_f": " OR ".join(filters)})
    request = Request(url, headers={"User-Agent": "RUNA/1.0 (academic catalogue lookup)"})
    with urlopen(request, timeout=15) as response:
        text = response.read(2_000_001).decode("utf-8-sig")
    if len(text) > 2_000_000:
        raise ValueError("exoplanet.eu returned an unexpectedly large catalogue")
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames or not {"name", "star_name", "planet_status"}.issubset(reader.fieldnames):
        raise ValueError("exoplanet.eu did not return its expected CSV columns")

    allowed_hosts = {name.casefold() for name in names}
    rows = []
    for record in reader:
        if (record.get("star_name") or "").strip().casefold() not in allowed_hosts:
            continue
        status = (record.get("planet_status") or "").strip().capitalize()
        if status not in {"Confirmed", "Candidate", "Unconfirmed", "Controversial"}:
            continue
        radius = _number(record.get("radius"))
        mass = _number(record.get("mass"))
        minimum_mass = _number(record.get("mass_sini"))
        rows.append({
            "hostname": (record.get("star_name") or "").strip(),
            "pl_name": (record.get("name") or "").strip(),
            "discoverymethod": (record.get("detection_type") or "").strip(),
            "disc_year": record.get("discovered"),
            "pl_orbper": _number(record.get("orbital_period")),
            "pl_orbsmax": _number(record.get("semi_major_axis")),
            "pl_orbeccen": _number(record.get("eccentricity")),
            "pl_rade": radius * JUPITER_TO_EARTH_RADII if radius is not None else None,
            "pl_masse": mass * JUPITER_TO_EARTH_MASSES if mass is not None else None,
            "pl_bmasse": (mass if mass is not None else minimum_mass) * JUPITER_TO_EARTH_MASSES
            if mass is not None or minimum_mass is not None else None,
            "pl_bmassprov": "Mass" if mass is not None else "M*sini" if minimum_mass is not None else None,
            "pl_bmasselim": 0,
            "status": status,
            "source": "exoplanet.eu",
        })
    return rows


def saved_proxima_candidate(gaia_id):
    """Use a dated exoplanet.eu record if its live export is unreachable."""
    if str(gaia_id) != PROXIMA_GAIA_DR3_ID:
        return []
    return [{
        "hostname": "Proxima Centauri",
        "pl_name": "Proxima Centauri c",
        "discoverymethod": "Radial Velocity / Astrometry",
        "disc_year": 2019,
        "pl_orbper": 1928.0,
        "pl_orbsmax": 1.48,
        "pl_orbeccen": 0.0,
        "pl_rade": None,
        "pl_masse": None,
        "status": "Candidate",
        "source": "exoplanet.eu snapshot · 2026-10-27",
        "source_url": PROXIMA_C_SOURCE,
    }]


def _planet_key(planet):
    """Match a planet letter across host aliases such as Proxima Cen/Centauri."""
    name = str(planet.get("pl_name") or "").strip().casefold()
    host = str(planet.get("hostname") or "").strip().casefold()
    prefix = host + " "
    return name[len(prefix):] if host and name.startswith(prefix) else name


def merge_planets(nasa_planets, eu_planets):
    """Keep NASA parameters for confirmed planets, adding unique EU records."""
    merged = {}
    for planet in [*nasa_planets, *eu_planets]:
        key = _planet_key(planet)
        if not key:
            continue
        if key not in merged:
            merged[key] = dict(planet)
            continue
        existing = merged[key]
        if existing.get("pl_bmasse") is None and planet.get("pl_bmasse") is not None:
            for field in ("pl_bmasse", "pl_bmassprov", "pl_bmasselim"):
                existing[field] = planet.get(field)
        for field in (
            "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_rade", "pl_masse",
            "discoverymethod", "disc_year",
        ):
            if existing.get(field) in (None, "") and planet.get(field) not in (None, ""):
                existing[field] = planet[field]
        if planet.get("source") not in str(existing.get("source") or ""):
            existing["source"] += " + " + planet["source"]
    return sorted(
        merged.values(),
        key=lambda item: (
            item.get("pl_orbsmax") is None,
            item.get("pl_orbsmax") or float("inf"),
            item.get("pl_name") or "",
        ),
    )


def orbital_mass(planet):
    """Return a positive mass and its kind, excluding limits and inferred masses."""
    mass = _number(planet.get("pl_bmasse"))
    # NASA returns "Msini"; other catalogues also use "M*sini".
    provenance = str(planet.get("pl_bmassprov") or "").lower()
    for character in " *()":
        provenance = provenance.replace(character, "")
    if mass is None or mass <= 0 or planet.get("pl_bmasselim") not in (None, 0):
        return None, "unknown"
    if provenance in {"mass", "msini/sini"}:
        return mass, "mass"
    if provenance == "msini":
        return mass, "minimum"
    return None, "unknown"
