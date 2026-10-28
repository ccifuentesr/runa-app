"""Look up and plot Gaia sources in a small cone around a target."""

from math import cos, log10, radians

import matplotlib.pyplot as plt
import numpy as np
from astroquery.gaia import Gaia


def _number(value):
    """Convert a Gaia value, including masked values, to float or None."""
    try:
        if np.ma.is_masked(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def query_gaia_neighbours(ra, dec, radius_arcsec=5, limit=500):
    """Return every Gaia DR3 source in a small sky cone.

    No parallax or proper-motion filter is applied here. Those measurements
    are returned when Gaia has them so they can be inspected later.
    """
    ra = float(ra)
    dec = float(dec)
    radius_deg = float(radius_arcsec) / 3600.0
    dec_delta = radius_deg
    ra_delta = radius_deg / max(cos(radians(dec)), 0.1)
    ra_min = ra - ra_delta
    ra_max = ra + ra_delta
    if ra_min < 0:
        ra_condition = f"(ra > {ra_min + 360.0:.8f} OR ra < {ra_max:.8f})"
    elif ra_max >= 360:
        ra_condition = f"(ra > {ra_min:.8f} OR ra < {ra_max - 360.0:.8f})"
    else:
        ra_condition = f"ra BETWEEN {ra_min:.8f} AND {ra_max:.8f}"
    query = f"""
        SELECT TOP {int(limit)}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, phot_g_mean_mag, ruwe
        FROM gaiadr3.gaia_source
        WHERE {ra_condition}
          AND dec BETWEEN {dec - dec_delta:.8f} AND {dec + dec_delta:.8f}
    """
    rows = Gaia.launch_job(query).get_results()
    sources = []
    for row in rows:
        source = {
            "source_id": str(row["source_id"]),
            "ra": _number(row["ra"]),
            "dec": _number(row["dec"]),
            "parallax": _number(row["parallax"]),
            "parallax_error": _number(row["parallax_error"]),
            "pmra": _number(row["pmra"]),
            "pmdec": _number(row["pmdec"]),
            "g_mag": _number(row["phot_g_mean_mag"]),
            "ruwe": _number(row["ruwe"]),
        }
        if source["ra"] is not None and source["dec"] is not None:
            delta_ra = (source["ra"] - ra + 180.0) % 360.0 - 180.0
            separation = np.hypot(
                delta_ra * cos(radians(dec)) * 3600.0,
                (source["dec"] - dec) * 3600.0,
            )
            if separation > float(radius_arcsec):
                continue
            sources.append(source)
    return sources


def prepare_neighbours(sources, target_ra, target_dec):
    """Add tangent-plane offsets and absolute G where a parallax exists."""
    target_ra = float(target_ra)
    target_dec = float(target_dec)
    cos_dec = cos(radians(target_dec))
    prepared = []
    for source in sources:
        delta_ra = (source["ra"] - target_ra + 180.0) % 360.0 - 180.0
        x_arcsec = delta_ra * cos_dec * 3600.0
        y_arcsec = (source["dec"] - target_dec) * 3600.0
        source = dict(source)
        source["x_arcsec"] = x_arcsec
        source["y_arcsec"] = y_arcsec
        source["separation_arcsec"] = float(np.hypot(x_arcsec, y_arcsec))
        parallax = source["parallax"]
        g_mag = source["g_mag"]
        if parallax and parallax > 0 and g_mag is not None:
            source["absolute_g"] = g_mag + 5.0 * log10(parallax) - 10.0
        else:
            source["absolute_g"] = None
        prepared.append(source)
    return sorted(prepared, key=lambda source: source["separation_arcsec"])


def companion_diagnostics(source, target):
    """Cifuentes et al. (2025), Sect. 3.1: distance and common-motion cuts.

    Gaia pmra already includes cos(dec). Use the shortest angle between
    vectors, avoiding the quadrant ambiguity of a one-argument arctangent.
    None means unavailable, not a failed physical-companion classification.
    """
    if target is None or source["source_id"] == target["source_id"]:
        return None
    values = [_number(row.get(key)) for row in (target, source)
              for key in ("parallax", "pmra", "pmdec")]
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    p1, ra1, dec1, p2, ra2, dec2 = values
    mu1, mu2 = np.hypot(ra1, dec1), np.hypot(ra2, dec2)
    if p1 <= 0 or p2 <= 0 or mu1 == 0 or mu2 == 0:
        return None
    distance_ratio = abs(1 - p1 / p2)
    motion_ratio = np.hypot(ra1 - ra2, dec1 - dec2) / mu1
    pa1, pa2 = np.degrees(np.arctan2([ra1, ra2], [dec1, dec2]))
    delta_pa = abs((pa2 - pa1 + 180) % 360 - 180)
    return dict(distance_ratio=float(distance_ratio), motion_ratio=float(motion_ratio),
                delta_pa_deg=float(delta_pa),
                check=bool(distance_ratio < .10 and motion_ratio < .15 and delta_pa < 15))


def local_proper_motion_plot(sources, radius_arcsec, target_source_id=None):
    """Plot nearby sources and their proper-motion vectors in a sky square."""
    if not sources:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")

    # Use the observed G magnitude for the visual scale.  A lower G means a
    # brighter point in this local field.  We still require a valid parallax
    # before applying the scale, as requested for the nearby-source view.
    valid_magnitudes = [
        source["g_mag"] for source in sources
        if source["absolute_g"] is not None and source["g_mag"] is not None
    ]
    faintest_mag = max(valid_magnitudes) if valid_magnitudes else 20.0
    brightest_mag = min(valid_magnitudes) if valid_magnitudes else 10.0
    magnitude_range = max(faintest_mag - brightest_mag, 1.0)
    x_values = []
    y_values = []
    sizes = []
    colours = []
    for source in sources:
        x_values.append(source["x_arcsec"])
        y_values.append(source["y_arcsec"])
        is_target = target_source_id and source["source_id"] == str(target_source_id)
        absolute_g = source["absolute_g"]
        g_mag = source["g_mag"]
        if absolute_g is None or g_mag is None:
            sizes.append(28.0)
            colours.append("#ef4444" if is_target else "#62626b")
        else:
            brightness_fraction = (faintest_mag - g_mag) / magnitude_range
            sizes.append(float(np.clip(
                28.0 + 150.0 * brightness_fraction, 28.0, 178.0
            )))
            colours.append("#ef4444" if is_target else "#d4d4d8")

    ax.scatter(x_values, y_values, s=sizes, c=colours, alpha=0.9,
               edgecolor="#111113", linewidth=0.7, zorder=3)

    motions = [
        np.hypot(source["pmra"], source["pmdec"])
        for source in sources
        if source["pmra"] is not None and source["pmdec"] is not None
    ]
    if motions:
        arrow_scale = 0.35 * float(radius_arcsec) / max(motions)
        for source in sources:
            if source["pmra"] is None or source["pmdec"] is None:
                continue
            ax.arrow(
                source["x_arcsec"], source["y_arcsec"],
                source["pmra"] * arrow_scale,
                source["pmdec"] * arrow_scale,
                color="#d8a099", width=0.012,
                head_width=0.18, length_includes_head=True,
                alpha=0.85, zorder=2,
            )

    view = float(radius_arcsec)
    ax.set_xlim(view, -view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#303036", lw=0.7)
    ax.axvline(0, color="#303036", lw=0.7)
    ax.grid(False)
    ax.set_xlabel("ΔRA [arcsec]", fontsize=13, color="#a1a1aa")
    ax.set_ylabel("ΔDec [arcsec]", fontsize=13, color="#a1a1aa")
    ax.tick_params(colors="#a1a1aa", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#4a4a52")
    ax.set_title("Nearby Gaia sources", color="#f4f4f5", fontsize=13, pad=12)
    fig.tight_layout()
    return fig
