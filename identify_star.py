"""Identify one star and fetch a survey image."""

import io
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.image as mpimg
import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.skyview import SkyView

from service_mirrors import SIMBAD_SERVERS, configured_simbad, configured_viziers


SDSS_CUTOUT_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"


def lookup_star(name):
    name = name.strip()
    if not name:
        raise ValueError("Enter an object name.")

    fields = (
        "ids", "otype", "otype_txt", "sp_type", "sp_bibcode",
        "coo_err_maj", "coo_err_min", "coo_err_angle", "coo_bibcode",
        "plx_value", "plx_err", "plx_bibcode",
        "pmra", "pmdec", "pm_err_maj", "pm_err_min", "pm_err_angle",
        "pm_bibcode",
        "rvz_radvel", "rvz_err", "rvz_bibcode", "fe_h",
    )
    # Adding all fields together performs one metadata request. Adding them one
    # by one makes astroquery retrieve the SIMBAD field list repeatedly.
    last_error = None
    for server in SIMBAD_SERVERS:
        try:
            simbad = configured_simbad(timeout=20, server=server)
            simbad.add_votable_fields(*fields)
            result = simbad.query_object(name)
            break
        except Exception as error:
            last_error = error
    else:
        raise ConnectionError("The SIMBAD query failed on every mirror.") from last_error
    if result is None or len(result) == 0:
        return None

    row = result[0]
    data = {}
    data["simbad_server"] = simbad.server
    columns = (
        "main_id", "otype", "otype_txt", "ra", "dec",
        "coo_err_maj", "coo_err_min", "coo_err_angle", "coo_bibcode",
        "sp_type", "sp_bibcode", "plx_value", "plx_err", "plx_bibcode",
        "pmra", "pmdec", "pm_err_maj", "pm_err_min", "pm_err_angle",
        "pm_bibcode",
        "rvz_radvel", "rvz_err", "rvz_bibcode", "fe_h",
    )
    for column in columns:
        value = row[column] if column in result.colnames else None
        data[column] = None if np.ma.is_masked(value) else value

    data["otype_name"] = None
    if data["otype"]:
        safe_type = str(data["otype"]).replace("'", "''")
        try:
            type_result = simbad.query_tap(
                "SELECT otype_longname FROM otypedef "
                f"WHERE otype = '{safe_type}'"
            )
            if len(type_result):
                data["otype_name"] = str(type_result[0]["otype_longname"])
        except Exception:
            pass

    # Direct parallax conversion used in the catalogue scripts.
    parallax = data["plx_value"]
    data["distance_pc"] = 1000.0 / parallax if parallax is not None and parallax > 0 else None
    parallax_error = data["plx_err"]
    if data["distance_pc"] is not None and parallax_error is not None:
        data["distance_error_pc"] = 1000.0 * parallax_error / parallax ** 2
    else:
        data["distance_error_pc"] = None

    data["gaia_id"] = None
    data["twomass_id"] = None
    data["identifiers"] = []
    if "ids" in result.colnames and not np.ma.is_masked(row["ids"]):
        identifiers = str(row["ids"]).split("|")
        for identifier in identifiers:
            identifier = identifier.strip()
            if identifier:
                data["identifiers"].append(identifier)
            if identifier.startswith("Gaia DR3 "):
                data["gaia_id"] = identifier.replace("Gaia DR3 ", "", 1)
            elif identifier.startswith("2MASS J"):
                data["twomass_id"] = identifier
    if data["gaia_id"] is None or data["twomass_id"] is None:
        try:
            identifiers = simbad.query_objectids(name)
            if identifiers is not None:
                for identifier in identifiers["ID"]:
                    identifier = str(identifier).strip()
                    if identifier and identifier not in data["identifiers"]:
                        data["identifiers"].append(identifier)
                    if identifier.startswith("Gaia DR3 "):
                        data["gaia_id"] = identifier.replace("Gaia DR3 ", "", 1)
                    elif identifier.startswith("2MASS J"):
                        data["twomass_id"] = identifier
        except Exception:
            pass

    return data


def lookup_gaia_photometry(gaia_id):
    """Get Gaia DR3 data by exact source ID, with VizieR as fallback."""
    photometry = {}
    if gaia_id and str(gaia_id).isdigit():
        query = (
            "SELECT ra, dec, ref_epoch, pmra, pmdec, pmra_error, pmdec_error, "
            "parallax, parallax_error, "
            "astrometric_n_obs_al, astrometric_matched_transits, "
            "phot_g_n_obs, phot_bp_n_obs, "
            "rv_nb_transits, "
            "phot_rp_n_obs, "
            "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, "
            "phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, "
            "phot_rp_mean_flux_over_error "
            "FROM gaiadr3.gaia_source WHERE source_id = " + str(gaia_id)
        )
        try:
            rows = Gaia.launch_job(query).get_results()
            if len(rows):
                for column in (
                    "ra", "dec", "ref_epoch", "pmra", "pmdec",
                    "pmra_error", "pmdec_error",
                    "parallax", "parallax_error", "phot_g_mean_mag",
                    "phot_bp_mean_mag", "phot_rp_mean_mag", "astrometric_n_obs_al",
                    "astrometric_matched_transits", "phot_g_n_obs",
                    "phot_bp_n_obs", "phot_rp_n_obs", "rv_nb_transits",
                ):
                    value = rows[0][column]
                    photometry[column] = None if np.ma.is_masked(value) else value
                for band in ("g", "bp", "rp"):
                    column = f"phot_{band}_mean_flux_over_error"
                    value = rows[0][column]
                    if not np.ma.is_masked(value) and value > 0:
                        photometry[f"phot_{band}_mean_mag_error"] = (
                            2.5 / np.log(10) / float(value)
                        )
        except Exception:
            pass

        if not photometry:
            columns = [
                "Source", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx",
                "pmRA", "e_pmRA", "pmDE", "e_pmDE",
                "FG", "e_FG", "Gmag", "FBP", "e_FBP", "BPmag",
                "FRP", "e_FRP", "RPmag",
                "NAL", "MatchObsA", "o_Gmag", "o_BPmag", "o_RPmag", "o_RV",
            ]
            field_map = {
                "ra": "RA_ICRS",
                "dec": "DE_ICRS",
                "parallax": "Plx",
                "parallax_error": "e_Plx",
                "pmra": "pmRA",
                "pmra_error": "e_pmRA",
                "pmdec": "pmDE",
                "pmdec_error": "e_pmDE",
                "phot_g_mean_mag": "Gmag",
                "phot_bp_mean_mag": "BPmag",
                "phot_rp_mean_mag": "RPmag",
                "astrometric_n_obs_al": "NAL",
                "astrometric_matched_transits": "MatchObsA",
                "phot_g_n_obs": "o_Gmag",
                "phot_bp_n_obs": "o_BPmag",
                "phot_rp_n_obs": "o_RPmag",
                "rv_nb_transits": "o_RV",
            }
            for vizier in configured_viziers(
                columns=columns,
                catalog="I/355/gaiadr3",
                row_limit=2,
            ):
                try:
                    results = vizier.query_constraints(
                        Source="=" + str(gaia_id),
                    )
                except Exception:
                    continue
                if not results or not len(results[0]):
                    continue
                matching_row = None
                for row in results[0]:
                    source = row["Source"]
                    if not np.ma.is_masked(source) and str(source) == str(gaia_id):
                        matching_row = row
                        break
                if matching_row is None:
                    continue

                for target, source in field_map.items():
                    value = matching_row[source]
                    photometry[target] = (
                        None if np.ma.is_masked(value) else float(value)
                    )
                photometry["ref_epoch"] = 2016.0
                for band, flux_column, error_column in (
                    ("g", "FG", "e_FG"),
                    ("bp", "FBP", "e_FBP"),
                    ("rp", "FRP", "e_FRP"),
                ):
                    flux = matching_row[flux_column]
                    flux_error = matching_row[error_column]
                    if (
                        not np.ma.is_masked(flux)
                        and not np.ma.is_masked(flux_error)
                        and float(flux) > 0
                        and float(flux_error) > 0
                    ):
                        signal_to_noise = float(flux) / float(flux_error)
                        photometry[f"phot_{band}_mean_flux_over_error"] = signal_to_noise
                        photometry[f"phot_{band}_mean_mag_error"] = (
                            2.5 / np.log(10) / signal_to_noise
                        )
                photometry["gaia_service"] = (
                    f"VizieR ({vizier.VIZIER_SERVER})"
                )
                break

    return photometry


def coordinates_at_epoch(
    ra,
    dec,
    pmra,
    pmdec,
    reference_epoch,
    target_epoch,
):
    """Propagate Gaia coordinates to a requested Julian epoch.

    Gaia's ``pmra`` is mu_alpha*cos(dec), in milliarcseconds per year.
    The calculation is a small-angle linear propagation, suitable for the
    epoch display in RUNA.
    """
    values = (ra, dec, pmra, pmdec, reference_epoch, target_epoch)
    if any(value is None or np.ma.is_masked(value) for value in values):
        return None

    ra = float(ra)
    dec = float(dec)
    pmra = float(pmra)
    pmdec = float(pmdec)
    reference_epoch = float(reference_epoch)
    target_epoch = float(target_epoch)
    years = target_epoch - reference_epoch
    cos_dec = np.cos(np.deg2rad(dec))
    if abs(cos_dec) < 1e-8:
        return None

    ra += pmra * years / (3.6e6 * cos_dec)
    dec += pmdec * years / 3.6e6
    return {"ra": ra % 360.0, "dec": float(np.clip(dec, -90.0, 90.0))}


def lookup_2mass_j(twomass_id):
    """Find J, H and Ks using the exact SIMBAD 2MASS identifier."""
    if not twomass_id or not twomass_id.startswith("2MASS J"):
        return {}

    source_id = twomass_id.replace("2MASS J", "", 1).strip()
    clients = configured_viziers(
        columns=["2MASS", "Jmag", "e_Jmag", "Hmag", "e_Hmag",
                 "Kmag", "e_Kmag", "Qflg"],
        catalog="II/246/out",
        row_limit=10,
    )
    for vizier in clients:
        try:
            results = vizier.query_constraints(**{"2MASS": "=" + source_id})
        except Exception:
            continue
        if not results:
            continue
        for row in results[0]:
            if str(row["2MASS"]).strip() != source_id:
                continue
            quality = row["Qflg"]
            photometry = {}
            for index, (band, column) in enumerate((("j", "Jmag"), ("h", "Hmag"), ("ks", "Kmag"))):
                value, error = row[column], row["e_" + column]
                photometry[band + "_mag"] = None if np.ma.is_masked(value) else float(value)
                photometry[band + "_error"] = None if np.ma.is_masked(error) else float(error)
                photometry[band + "_quality"] = (
                    None if np.ma.is_masked(quality) else str(quality)[index]
                )
            return photometry
    return {}


def lookup_johnson_v(name):
    """Find a Johnson V magnitude from the UCAC4 catalogue."""
    if not name:
        return {}

    clients = configured_viziers(
        columns=["UCAC4", "Vmag", "e_Vmag"],
        catalog="I/322A/out",
        row_limit=1,
    )
    for vizier in clients:
        try:
            results = vizier.query_object(name, radius=5 * u.arcsec)
        except Exception:
            continue
        if not results or not len(results[0]):
            continue
        row = results[0][0]
        v_mag = row["Vmag"]
        if np.ma.is_masked(v_mag):
            continue
        v_error = row["e_Vmag"]
        return {
            "v_mag": float(v_mag),
            # UCAC4 uncertainty columns are given in hundredths of a magnitude.
            "v_error": (
                None if np.ma.is_masked(v_error) else float(v_error) / 100.0
            ),
        }
    return {}


def fetch_sdss_image(ra, dec):
    """Return the SDSS DR17 colour cutout, or None outside its coverage."""
    options = {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}",
               "scale": 0.4, "width": 512, "height": 512}
    url = SDSS_CUTOUT_URL + "?" + urlencode(options)
    with urlopen(url, timeout=20) as response:
        image = response.read()

    # SkyServer can return a nearly uniform image outside the SDSS footprint.
    pixels = mpimg.imread(io.BytesIO(image), format="jpg")
    if pixels.size == 0 or float(pixels.std()) < 1.5:
        return None
    return image


def fetch_dss2_image(ra, dec):
    """Return a DSS2 Red photographic image from NASA SkyView."""
    images = SkyView.get_images(
        position=f"{ra:.6f} {dec:.6f}",
        survey=["DSS2 Red"],
        pixels=512,
    )
    if not images:
        return None

    pixels = np.asarray(images[0][0].data, dtype=float)
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        return None

    lower, upper = np.percentile(finite, [1, 99.5])
    if upper <= lower:
        return None

    scaled = np.clip((pixels - lower) / (upper - lower), 0, 1)
    output = io.BytesIO()
    mpimg.imsave(
        output, scaled, cmap="gray", format="png", vmin=0, vmax=1,
        origin="lower",
    )
    return output.getvalue()


def fetch_survey_image(ra, dec):
    """Try SDSS first and use DSS2 Red as an all-sky backup."""
    try:
        image = fetch_sdss_image(ra, dec)
    except Exception:
        image = None

    if image is not None:
        return {
            "data": image,
            "catalogue": "SDSS DR17",
            "caption": (
                "SDSS DR17 colour cutout · visible (ugriz) · "
                "observations 1998–2009 · 0.4 arcsec/pixel"
            ),
        }

    try:
        image = fetch_dss2_image(ra, dec)
    except Exception:
        image = None

    if image is not None:
        return {
            "data": image,
            "catalogue": "DSS2 Red",
            "caption": (
                "DSS2 Red photographic plate · visible (red) · "
                "plate epoch 1984–1999"
            ),
        }
    return None
