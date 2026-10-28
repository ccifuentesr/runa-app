"""Look up planets in NASA's archive and Open Exoplanet Catalogue."""

import gzip
import json
import math
import csv
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OEC_GZIP_URL = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"
OEC_REPOSITORY_URL = "https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue"
JUPITER_TO_EARTH_RADII = 11.209
JUPITER_TO_EARTH_MASSES = 317.828
PLANET_POPULATION_PATH = (
    Path(__file__).resolve().parent / "data" / "exoplanet_population_20261028.csv"
)


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


@lru_cache(maxsize=1)
def load_planet_population():
    """Load the bundled NASA confirmed-planet snapshot for the context plot.

    The snapshot is deliberately local so redrawing the plot does not make a
    catalogue request. It contains only sources with a finite period and a
    measured mass or minimum mass.
    """
    if not PLANET_POPULATION_PATH.exists():
        return []

    planets = []
    with PLANET_POPULATION_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            period = _number(row.get("pl_orbper"))
            mass = _number(row.get("pl_bmasse")) or _number(row.get("pl_masse"))
            mass_limit = _number(row.get("pl_bmasselim"))
            if (
                period is None
                or mass is None
                or period <= 0
                or mass <= 0
                or mass_limit not in (None, 0)
            ):
                continue
            planets.append({
                "pl_name": row.get("pl_name") or "",
                "pl_orbper": period,
                "pl_bmasse": mass,
                "pl_bmassprov": row.get("pl_bmassprov") or "",
            })
    return planets


def _normalise_name(value):
    """Normalise catalogue aliases for exact, whitespace-tolerant matching."""
    return " ".join(str(value or "").casefold().split())


@lru_cache(maxsize=1)
def _open_exoplanet_catalogue_root():
    """Download and parse the official, compressed OEC XML once per process."""
    request = Request(
        OEC_GZIP_URL,
        headers={"User-Agent": "RUNA/1.0 (catalogue lookup)"},
    )
    with urlopen(request, timeout=30) as response:
        compressed = response.read(50_000_001)
    if len(compressed) > 50_000_000:
        raise ValueError("Open Exoplanet Catalogue download is unexpectedly large")
    return ET.fromstring(gzip.decompress(compressed))


def _oec_status(list_name):
    """Map the OEC collection label to RUNA's compact display states."""
    label = str(list_name or "").casefold()
    if "candidate" in label:
        return "Candidate"
    if "unconfirmed" in label or "controversial" in label:
        return "Unconfirmed"
    return "Confirmed"


def lookup_open_exoplanet_catalogue(query_name, main_id, identifiers=None):
    """Return OEC planets whose host aliases exactly match the SIMBAD aliases."""
    candidates = _candidate_names(query_name, main_id, identifiers)
    allowed_names = {_normalise_name(name) for name in candidates}
    if not allowed_names:
        return []

    rows = []
    seen = set()
    for star in _open_exoplanet_catalogue_root().iter("star"):
        host_names = [name.text.strip() for name in star.findall("name") if name.text]
        if not any(_normalise_name(name) in allowed_names for name in host_names):
            continue
        host_name = host_names[0] if host_names else main_id
        for planet in star.findall("planet"):
            planet_name = planet.findtext("name") or ""
            key = (_normalise_name(host_name), _normalise_name(planet_name))
            if not planet_name or key in seen:
                continue
            seen.add(key)
            mass_element = planet.find("mass")
            mass = _number(mass_element.text if mass_element is not None else None)
            mass_type = str(mass_element.get("type") or "").casefold() if mass_element is not None else ""
            is_minimum_mass = mass_type in {"msini", "m sini", "m*sin(i)"}
            radius = _number(planet.findtext("radius"))
            rows.append({
                "hostname": host_name,
                "pl_name": planet_name.strip(),
                "discoverymethod": (planet.findtext("discoverymethod") or "").strip(),
                "disc_year": _number(planet.findtext("discoveryyear")),
                "pl_orbper": _number(planet.findtext("period")),
                "pl_orbsmax": _number(planet.findtext("semimajoraxis")),
                "pl_orbeccen": _number(planet.findtext("eccentricity")),
                "pl_rade": radius * JUPITER_TO_EARTH_RADII if radius is not None else None,
                "pl_masse": mass * JUPITER_TO_EARTH_MASSES if mass is not None and not is_minimum_mass else None,
                "pl_bmasse": mass * JUPITER_TO_EARTH_MASSES if mass is not None else None,
                "pl_bmassprov": "Msini" if is_minimum_mass else "Mass" if mass is not None else None,
                "pl_bmasselim": 0,
                "status": _oec_status(planet.findtext("list")),
                "source": "Open Exoplanet Catalogue",
                "source_url": OEC_REPOSITORY_URL,
            })
    return rows


def _planet_key(planet):
    """Return a host-independent key for one planet in the loaded system.

    Planet names are queried for one host at a time, but catalogue aliases can
    differ in punctuation or whether the final letter has a separating space.
    The terminal letter is therefore the stable key for cross-catalogue merge.
    """
    name = str(planet.get("pl_name") or "").strip().casefold()
    letter = re.search(r"([b-z])$", name)
    if letter is not None:
        return letter.group(1)
    return re.sub(r"[^a-z0-9]+", "", name)


def merge_planets(nasa_planets, oec_planets):
    """Keep NASA records intact and add only OEC planets absent from NASA."""
    merged = {}
    for planet in nasa_planets:
        key = _planet_key(planet)
        if not key:
            continue
        merged.setdefault(key, dict(planet))

    for planet in oec_planets:
        key = _planet_key(planet)
        if key and key not in merged:
            merged[key] = dict(planet)
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
