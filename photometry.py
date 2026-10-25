"""Photometry utilities used by the Photometry section."""

import re
from pathlib import Path

import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from catalogue_selection import OPTICAL_PRIORITY, preferred_photometry


DATA_DIR = Path(__file__).resolve().parent / "data"
FILTER_DIR = DATA_DIR / "filters"
VOSA_FILTER_FILE = FILTER_DIR / "VOSA_filters.csv"
BAND_METADATA_FILE = FILTER_DIR / "catalogue_bands.csv"
FILTER_CURVES_FILE = FILTER_DIR / "filter_curves.csv"


def _load_band_metadata():
    """Read catalogue-specific effective wavelengths from the data folder."""
    metadata = {}
    with BAND_METADATA_FILE.open(encoding="utf-8") as data_file:
        next(data_file, None)
        for line in data_file:
            catalogue, label, wavelength = line.strip().split(",")
            metadata[(catalogue, label)] = float(wavelength)
    return metadata


def _load_vosa_filters():
    """Read effective widths and Vega zero points from VOSA's filter table."""
    filters = {}
    with VOSA_FILTER_FILE.open(encoding="utf-8") as data_file:
        for line in data_file:
            fields = [field.strip() for field in re.split(r"[,\t]+", line)]
            if not fields or fields[0] in {"", "filter"} or len(fields) < 4:
                continue
            label = fields[0]
            try:
                filters[label] = {
                    "wavelength": float(fields[1]),
                    "width": float(fields[2]),
                    "zero_point": float(fields[3]),
                }
            except ValueError:
                continue
    return filters


def _load_filter_curve_names():
    """Read the local transmission-curve filenames for each band."""
    curves = {}
    with FILTER_CURVES_FILE.open(encoding="utf-8") as data_file:
        next(data_file, None)
        for line in data_file:
            label, filename = line.strip().split(",")
            curves[label] = FILTER_DIR / filename
    return curves


EFFECTIVE_WAVELENGTHS = _load_band_metadata()
FILTER_DATA = _load_vosa_filters()
FILTER_CURVES = _load_filter_curve_names()
EFFECTIVE_WIDTHS = {
    label: values["width"] for label, values in FILTER_DATA.items()
}
VEGA_ZERO_POINTS = {
    label: values["zero_point"] for label, values in FILTER_DATA.items()
}

ERG_CM2_TO_W_M2 = 1.0e-3


def load_filter_curve(label):
    """Return wavelength and transmission arrays for one local filter curve."""
    path = FILTER_CURVES.get(label)
    if path is None or not path.exists():
        return None
    try:
        curve = np.loadtxt(path, usecols=(0, 1))
    except (OSError, ValueError):
        return None
    if curve.ndim != 2 or curve.shape[1] != 2:
        return None
    return curve[:, 0], curve[:, 1]


PHOTOMETRIC_CATALOGS = {
    "AllWISE": {
        "table": "II/328/allwise",
        "id_column": "AllWISE",
        "prefixes": ("WISEA J", "WISEA "),
        "columns": [
            "AllWISE", "W1mag", "e_W1mag", "W2mag", "e_W2mag",
            "W3mag", "e_W3mag", "W4mag", "e_W4mag", "qph",
        ],
        "bands": [
            ("W1", "W1mag", "e_W1mag", "#ff824f", "mid-IR"),
            ("W2", "W2mag", "e_W2mag", "#ff684a", "mid-IR"),
            ("W3", "W3mag", "e_W3mag", "#f0443e", "mid-IR"),
            ("W4", "W4mag", "e_W4mag", "#dc2626", "mid-IR"),
        ],
    },
    "2MASS": {
        "table": "II/246/out",
        "id_column": "2MASS",
        "prefixes": ("2MASS J", "2MASS "),
        "columns": [
            "2MASS", "Jmag", "e_Jmag", "Hmag", "e_Hmag",
            "Kmag", "e_Kmag", "Qflg",
        ],
        "bands": [
            ("J", "Jmag", "e_Jmag", "#d7d85a", "near-IR"),
            ("H", "Hmag", "e_Hmag", "#e6bb62", "near-IR"),
            ("Ks", "Kmag", "e_Kmag", "#f59e0b", "near-IR"),
        ],
    },
    "SDSS9": {
        "table": "V/139/sdss9",
        "id_column": None,
        "prefixes": ("SDSS J",),
        "columns": [
            "objID", "umag", "e_umag", "gmag", "e_gmag", "rmag",
            "e_rmag", "imag", "e_imag", "zmag", "e_zmag",
        ],
        "bands": [
            ("u", "umag", "e_umag", "#6d5dfc", "near-UV"),
            ("g", "gmag", "e_gmag", "#2e9bff", "optical"),
            ("r", "rmag", "e_rmag", "#4ccf91", "optical"),
            ("i", "imag", "e_imag", "#c7d65a", "optical"),
            ("z", "zmag", "e_zmag", "#e6a94e", "optical"),
        ],
    },
    "UCAC4": {
        "table": "I/322A/out",
        "id_column": "UCAC4",
        "prefixes": ("UCAC4",),
        "columns": [
            "UCAC4", "Bmag", "e_Bmag", "Vmag", "e_Vmag",
            "gmag", "e_gmag", "rmag", "e_rmag", "imag", "e_imag",
        ],
        "bands": [
            ("B", "Bmag", "e_Bmag", "#4d7dff", "blue-optical"),
            ("V", "Vmag", "e_Vmag", "#45c99a", "optical"),
            ("g", "gmag", "e_gmag", "#78d98b", "optical"),
            ("r", "rmag", "e_rmag", "#e77c4f", "optical"),
            ("i", "imag", "e_imag", "#c7d65a", "red-optical"),
        ],
    },
    "GALEX": {
        "table": "II/312",
        "id_column": None,
        "prefixes": ("GALEXASC J", "GALEX J"),
        "columns": ["GALEX", "FUVmag", "e_FUVmag", "NUVmag", "e_NUVmag"],
        "bands": [
            ("FUV", "FUVmag", "e_FUVmag", "#7c3aed", "far-UV"),
            ("NUV", "NUVmag", "e_NUVmag", "#4f46e5", "near-UV"),
        ],
    },
    "APASS DR9": {
        "table": "II/336/apass9",
        "id_column": None,
        "prefixes": ("APASS",),
        "columns": ["Bmag", "e_Bmag", "Vmag", "e_Vmag",
                    "g'mag", "e_g'mag", "r'mag", "e_r'mag", "i'mag", "e_i'mag"],
        "bands": [
            ("B", "Bmag", "e_Bmag", "#4d7dff", "blue-optical"),
            ("V", "Vmag", "e_Vmag", "#45c99a", "optical"),
            ("g", "g'mag", "e_g'mag", "#2e9bff", "optical"),
            ("r", "r'mag", "e_r'mag", "#e77c4f", "optical"),
            ("i", "i'mag", "e_i'mag", "#c7d65a", "red-optical"),
        ],
    },
    "SkyMapper DR4": {
        "table": "II/379/smssdr4",
        "id_column": None,
        "prefixes": ("SMSS J",),
        "columns": ["ObjectId", "Gaia3Id1", "uPSF", "e_uPSF", "vPSF", "e_vPSF",
                    "gPSF", "e_gPSF", "rPSF", "e_rPSF", "iPSF", "e_iPSF",
                    "zPSF", "e_zPSF"],
        "bands": [
            ("u", "uPSF", "e_uPSF", "#6d5dfc", "near-UV"),
            ("v", "vPSF", "e_vPSF", "#6264ed", "violet"),
            ("g", "gPSF", "e_gPSF", "#2e9bff", "optical"),
            ("r", "rPSF", "e_rPSF", "#e77c4f", "optical"),
            ("i", "iPSF", "e_iPSF", "#c7d65a", "red-optical"),
            ("z", "zPSF", "e_zPSF", "#e6a94e", "red-optical"),
        ],
    },
}


def absolute_magnitude(apparent_magnitude, parallax):
    """Convert apparent magnitude and parallax in mas to absolute magnitude."""
    if apparent_magnitude is None or parallax is None or parallax <= 0:
        return None
    return float(apparent_magnitude) + 5 * np.log10(float(parallax)) - 10


def _number(value):
    """Return a finite value as float, or None for missing catalogue data."""
    try:
        if np.ma.is_masked(value):
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _normalise_identifier(value):
    """Put SIMBAD and VizieR catalogue IDs in the same simple form."""
    if value is None or np.ma.is_masked(value):
        return None
    text = " ".join(str(value).strip().upper().split())
    for prefix in ("WISEA J", "WISEA ", "2MASS J", "2MASS ", "UCAC4 "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("J") and len(text) > 1 and text[1].isdigit():
        text = text[1:]
    return text


def _identifier_for(star, prefixes):
    """Return the first matching identifier from the SIMBAD list."""
    for identifier in star.get("identifiers", []):
        identifier = str(identifier).strip()
        if identifier.startswith(prefixes):
            return identifier
    return None


def _query_catalog(star, config):
    """Query one VizieR catalogue, preferring an exact SIMBAD identifier."""
    identifier = _identifier_for(star, config["prefixes"])
    query_name = identifier or star.get("main_id")
    if not query_name:
        return None, None

    vizier = Vizier(
        columns=["+_r", *config["columns"]],
        catalog=config["table"],
        row_limit=10,
    )
    try:
        if identifier and config.get("id_column"):
            catalogue_id = _normalise_identifier(identifier)
            if config["id_column"] == "AllWISE":
                catalogue_id = "J" + catalogue_id
            results = vizier.query_constraints(**{
                config["id_column"]: "=" + catalogue_id
            })
        else:
            results = vizier.query_object(query_name, radius=5 * u.arcsec)
    except Exception:
        results = None
    if not results:
        return None, query_name

    table = results[0]
    if len(table) == 0:
        return None, query_name

    row = None
    requested_id = _normalise_identifier(identifier)
    id_column = config.get("id_column")
    if requested_id and id_column in table.colnames:
        for candidate in table:
            if _normalise_identifier(candidate[id_column]) == requested_id:
                row = candidate
                break
        if row is None:
            return None, query_name
    else:
        row = table[0]
    if config["table"] == "II/379/smssdr4" and star.get("gaia_id"):
        matches = [candidate for candidate in table
                   if str(candidate["Gaia3Id1"]) == str(star["gaia_id"])]
        if not matches:
            return None, query_name
        row = matches[0]
    return row, query_name


def _catalogue_record(name, config, star):
    """Return one compact catalogue record for the Photometry panel."""
    row, query_name = _query_catalog(star, config)
    bands = []
    for label, value_column, error_column, colour, spectral_range in config["bands"]:
        value = _number(row[value_column]) if row is not None else None
        error = _number(row[error_column]) if row is not None else None
        if name == "UCAC4" and error is not None:
            error /= 100.0
        bands.append({
            "label": label,
            "value": value,
            "error": error,
            "colour": colour,
            "range": spectral_range,
            "wavelength": EFFECTIVE_WAVELENGTHS.get((name, label)),
        })
    return {
        "catalogue": name,
        "query": query_name,
        "matched": row is not None,
        "bands": bands,
    }


def fetch_catalogue_photometry(star):
    """Fetch additional bands and include the already loaded Gaia photometry."""
    records = []
    gaia = star.get("gaia_photometry", {})
    gaia_bands = []
    for label, column, error_column, colour, spectral_range in (
        ("G", "phot_g_mean_mag", "phot_g_mean_mag_error", "#48d597", "optical"),
        ("BP", "phot_bp_mean_mag", "phot_bp_mean_mag_error", "#39b8d3", "blue-optical"),
        ("RP", "phot_rp_mean_mag", "phot_rp_mean_mag_error", "#c9d45f", "red-optical"),
    ):
        gaia_bands.append({
            "label": label,
            "value": _number(gaia.get(column)),
            "error": _number(gaia.get(error_column)),
            "colour": colour,
            "range": spectral_range,
            "wavelength": EFFECTIVE_WAVELENGTHS.get(("Gaia DR3", label)),
        })
    records.append({
        "catalogue": "Gaia DR3",
        "query": "Gaia DR3 " + str(star.get("gaia_id")),
        "matched": any(band["value"] is not None for band in gaia_bands),
        "bands": gaia_bands,
    })

    catalogue_order = ["AllWISE", "2MASS", *OPTICAL_PRIORITY, "GALEX"]
    for name in catalogue_order:
        config = PHOTOMETRIC_CATALOGS[name]
        try:
            records.append(_catalogue_record(name, config, star))
        except Exception:
            records.append({
                "catalogue": name,
                "query": _identifier_for(star, config["prefixes"]),
                "matched": False,
                "bands": [
                    {
                        "label": label, "value": None, "error": None,
                        "colour": colour, "range": spectral_range,
                        "wavelength": EFFECTIVE_WAVELENGTHS.get((name, label)),
                    }
                    for label, _, _, colour, spectral_range in config["bands"]
                ],
            })
    return records


def build_sed_points(records):
    """Convert the available Vega magnitudes into SED points.

    The returned flux is F_lambda in W m-2 A-1.  The plotted SED uses
    lambda F_lambda, following the approach in the TFM code.
    """
    points = []
    for record in preferred_photometry(records):
        for band in record.get("bands", []):
            magnitude = _number(band.get("value"))
            wavelength = _number(band.get("wavelength"))
            label = band.get("label")
            is_ab = (record["catalogue"] == "SkyMapper DR4" or
                     (record["catalogue"] == "APASS DR9" and label in ("g", "r", "i")))
            zero_point = VEGA_ZERO_POINTS.get(label)
            if is_ab and wavelength is not None:
                # AB zero point: 3631 Jy; c is in Angstrom per second.
                zero_point = 3631e-26 * 2.99792458e18 / wavelength ** 2 / ERG_CM2_TO_W_M2
            if magnitude is None or wavelength is None or zero_point is None:
                continue

            flux = (
                zero_point * ERG_CM2_TO_W_M2
                * 10 ** (-magnitude / 2.5)
            )
            error = _number(band.get("error"))
            flux_error = None
            if error is not None:
                flux_error = flux * np.log(10.0) / 2.5 * error
            points.append({
                "catalogue": record["catalogue"],
                "label": band["label"],
                "wavelength": wavelength,
                "width": (None if record["catalogue"] == "SkyMapper DR4"
                          else EFFECTIVE_WIDTHS.get(label)),
                "flux": flux,
                "flux_error": flux_error,
                "lambda_flux": wavelength * flux,
                "lambda_flux_error": (
                    wavelength * flux_error if flux_error is not None else None
                ),
                "colour": band["colour"],
            })
    return sorted(points, key=lambda point: point["wavelength"])


def star_cmd_position(star):
    """Return BP-RP and absolute G for the loaded star."""
    gaia = star.get("gaia_photometry", {})
    g_mag = gaia.get("phot_g_mean_mag")
    bp_mag = gaia.get("phot_bp_mean_mag")
    rp_mag = gaia.get("phot_rp_mean_mag")
    parallax = gaia.get("parallax", star.get("plx_value"))
    absolute_g = absolute_magnitude(g_mag, parallax)

    if bp_mag is None or rp_mag is None or absolute_g is None:
        return None
    return float(bp_mag) - float(rp_mag), absolute_g


def load_gaia_reference_sample():
    """Download a small Gaia DR3 sample with reliable basic measurements."""
    query = """
        SELECT TOP 8000
            parallax,
            phot_g_mean_mag,
            phot_bp_mean_mag,
            phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE random_index < 10000000
          AND parallax > 0
          AND parallax_over_error > 20
          AND ruwe < 1.4
          AND visibility_periods_used > 8
          AND phot_g_mean_flux_over_error > 50
          AND phot_bp_mean_flux_over_error > 50
          AND phot_rp_mean_flux_over_error > 50
          AND phot_g_mean_mag IS NOT NULL
          AND phot_bp_mean_mag IS NOT NULL
          AND phot_rp_mean_mag IS NOT NULL
    """
    rows = Gaia.launch_job_async(query).get_results()

    parallax = np.asarray(rows["parallax"], dtype=float)
    g_mag = np.asarray(rows["phot_g_mean_mag"], dtype=float)
    bp_mag = np.asarray(rows["phot_bp_mean_mag"], dtype=float)
    rp_mag = np.asarray(rows["phot_rp_mean_mag"], dtype=float)

    return {
        "colour": bp_mag - rp_mag,
        "absolute_g": g_mag + 5 * np.log10(parallax) - 10,
    }
