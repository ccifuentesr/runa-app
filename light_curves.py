"""Small TESS light-curve helper for the Photometry section."""

import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import fits

try:
    from astroquery.mast import Observations
except ImportError:
    Observations = None


def tess_identifier(star):
    """Return the TIC or TOI identifier supplied by SIMBAD."""
    identifiers = star.get("identifiers", []) if star else []
    for identifier in identifiers:
        text = str(identifier).strip()
        tic = re.search(r"\bTIC[\s-]*(\d+)\b", text, flags=re.IGNORECASE)
        if tic:
            return {"label": f"TIC {tic.group(1)}", "query": f"TIC{tic.group(1)}"}

    for identifier in identifiers:
        text = str(identifier).strip()
        toi = re.search(r"\bTOI[\s-]*(\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
        if toi:
            return {"label": f"TOI-{toi.group(1)}", "query": f"TOI-{toi.group(1)}"}
    return None


def _same_target(row, tic_number):
    """Keep the MAST row for the requested TIC, not a nearby source."""
    target_name = str(row["target_name"]).strip()
    return target_name.replace("TIC", "").replace(" ", "") == tic_number


def _light_curve_product(observations, tic_number):
    """Find one SPOC light-curve product, preferring the latest sector."""
    candidates = []
    for row in observations:
        if str(row["dataproduct_type"]) != "timeseries":
            continue
        if str(row["provenance_name"]).upper() != "SPOC":
            continue
        if tic_number and not _same_target(row, tic_number):
            continue
        candidates.append(row)

    def sector_number(row):
        try:
            return int(row["sequence_number"])
        except (TypeError, ValueError):
            return -1

    def target_distance(row):
        try:
            return float(row["distance"])
        except (TypeError, ValueError):
            return float("inf")

    candidates.sort(key=lambda row: (target_distance(row), -sector_number(row)))
    for row in candidates:
        products = Observations.get_product_list(row)
        for product_index, product in enumerate(products):
            if str(product["productSubGroupDescription"]) == "LC":
                return products[product_index:product_index + 1], sector_number(row)
    return None, None


def _read_light_curve(path):
    """Read and lightly clean one SPOC light-curve FITS file."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        if not np.isfinite(flux).any():
            flux = np.asarray(data["SAP_FLUX"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=int)

    valid = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
    if valid.sum() < 10:
        raise ValueError("The TESS file does not contain enough valid measurements.")

    time = time[valid]
    flux = flux[valid]
    median_flux = float(np.nanmedian(flux))
    if not np.isfinite(median_flux) or median_flux == 0:
        raise ValueError("The TESS flux could not be normalised.")

    return {
        "time": time,
        "relative_flux": 100.0 * (flux / median_flux - 1.0),
    }


def fetch_tess_light_curve(star):
    """Download one recent SPOC TESS light curve for a SIMBAD TIC/TOI."""
    identifier = tess_identifier(star)
    if identifier is None:
        return {"error": "SIMBAD did not provide a TIC or TOI identifier."}
    if Observations is None:
        return {"error": "The MAST query package is not installed."}

    try:
        observations = Observations.query_object(
            identifier["query"], radius=0.02 * u.deg
        )
        tic_number = re.sub(r"\D", "", identifier["label"]) if "TIC" in identifier["label"] else None
        product, sector = _light_curve_product(observations, tic_number)
        if product is None:
            return {"error": "No SPOC TESS light curve was found for this identifier."}

        with tempfile.TemporaryDirectory(prefix="runa-tess-") as temporary_dir:
            manifest = Observations.download_products(
                product,
                productType="SCIENCE",
                productSubGroupDescription="LC",
                mrp_only=False,
                download_dir=temporary_dir,
            )
            if manifest is None or len(manifest) == 0:
                return {"error": "MAST did not return a downloadable light curve."}
            local_path = Path(str(manifest["Local Path"][0]))
            curve = _read_light_curve(local_path)

        curve.update({
            "identifier": identifier["label"],
            "sector": sector,
            "source": "TESS SPOC via MAST",
        })
        return curve
    except Exception as error:
        return {"error": f"The TESS light curve could not be loaded: {error}"}


def light_curve_figure(curve, object_name):
    """Plot relative TESS flux in the same restrained RUNA style."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")
    relative_flux = curve["relative_flux"]
    ax.scatter(
        curve["time"], relative_flux,
        s=3, color="#6ee7b7", alpha=0.68, linewidths=0,
    )

    median_flux = float(np.nanmedian(relative_flux))
    mad = float(np.nanmedian(np.abs(relative_flux - median_flux)))
    robust_scatter = 1.4826 * mad
    if not np.isfinite(robust_scatter) or robust_scatter == 0:
        lower, upper = np.nanpercentile(relative_flux, [5, 95])
    else:
        lower = median_flux - 5 * robust_scatter
        upper = median_flux + 5 * robust_scatter
    padding = 0.08 * (upper - lower)
    ax.set_ylim(lower - padding, upper + padding)

    ax.set_xlabel("TESS time (BTJD − 2457000)", fontsize=15, color="#d4d4d8")
    ax.set_ylabel("Relative flux (%)", fontsize=15, color="#d4d4d8")
    ax.tick_params(colors="#a1a1aa", labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.set_title(
        f"{object_name} · {curve['identifier']} · Sector {curve['sector']}",
        color="#ededf0", fontsize=13, pad=10,
    )
    fig.tight_layout()
    return fig
