"""Washington Double Star catalogue lookups for known multiple systems."""

import math
import re

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


def closest_primary_component(ra, dec, identifiers=()):
    """Return the closest WDS sep2 entry involving component A.

    Prefer the WDS identifier in SIMBAD, including component-suffixed aliases.
    Only use a positional match when no WDS identifier is supplied.
    Historical entries still require manual review.
    """
    wds_ids = sorted({match.group(1) for identifier in identifiers
                      if (match := re.match(r"^WDS\s*J?\s*(\d{5}[+-]\d{4})",
                                            str(identifier).strip(), re.IGNORECASE))})
    vizier = Vizier(columns=["WDS", "Disc", "Comp", "sep2", "Obs2"], row_limit=-1)
    if wds_ids:
        tables = []
        for wds_id in wds_ids:
            result = vizier.query_constraints(catalog="B/wds/wds", WDS="=" + wds_id)
            tables.extend(result.values())
    else:
        if ra is None or dec is None:
            return None
        position = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
        tables = vizier.query_region(position, radius=10 * u.arcsec, catalog="B/wds/wds")
    if not tables:
        return None

    candidates = []
    for row in (row for table in tables for row in table):
        if wds_ids and str(row["WDS"]).strip() not in wds_ids:
            continue
        component = str(row["Comp"]).strip().upper() if row["Comp"] is not None else ""
        if "A" not in component:
            continue
        try:
            separation = float(row["sep2"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(separation) or separation <= 0:
            continue
        discoverer = str(row["Disc"]).strip() if row["Disc"] is not None else ""
        try:
            year = float(row["Obs2"])
            year = int(year) if math.isfinite(year) else None
        except (KeyError, TypeError, ValueError):
            year = None
        candidates.append({
            "last_observation_year": year,
            "separation_arcsec": separation,
            "discoverer": discoverer,
            "component": component,
            "wds": str(row["WDS"]).strip() if row["WDS"] is not None else "",
        })

    return min(candidates, key=lambda item: item["separation_arcsec"]) if candidates else None
