"""Washington Double Star catalogue lookups for known multiple systems."""

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier


def closest_primary_component(ra, dec):
    """Return the closest WDS sep2 entry involving component A.

    The WDS position is matched to the SIMBAD primary position in a small
    cone. Historical catalogue entries can be mismatched, so callers should
    always present the result as requiring manual review.
    """
    if not ra or not dec:
        return None

    position = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    vizier = Vizier(columns=["WDS", "Disc", "Comp", "sep2"], row_limit=100)
    tables = vizier.query_region(
        position,
        radius=10 * u.arcsec,
        catalog="B/wds/wds",
    )
    if not tables:
        return None

    candidates = []
    table = tables[0]
    for row in table:
        component = str(row["Comp"]).strip().upper() if row["Comp"] is not None else ""
        if "A" not in component:
            continue
        try:
            separation = float(row["sep2"])
        except (TypeError, ValueError):
            continue
        if not separation > 0:
            continue
        discoverer = str(row["Disc"]).strip() if row["Disc"] is not None else ""
        candidates.append({
            "separation_arcsec": separation,
            "discoverer": discoverer,
            "component": component,
            "wds": str(row["WDS"]).strip() if row["WDS"] is not None else "",
        })

    return min(candidates, key=lambda item: item["separation_arcsec"]) if candidates else None
