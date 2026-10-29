"""Queries for public high-resolution stellar-spectroscopy archives."""

from datetime import datetime, timedelta, timezone
import math
import re
from urllib.request import Request, urlopen

import numpy as np
from astroquery.utils.tap.core import TapPlus

from eso_archive import query_eso_spectroscopy
from identify_star import coordinates_at_epoch


GAVO_TAP_URL = "https://dc.g-vo.org/tap"
CARMENES_ARCHIVE_URL = "https://carmenes.cab.inta-csic.es/gto/jsp/dr1Public.jsp"
CAHA_ARCHIVE_URL = "https://caha.sdc.cab.inta-csic.es/calto/"
NOT_ARCHIVE_URL = "https://www.not.iac.es/archive/"
NOT_FIES_LISTS = (
    "https://www.not.iac.es/archive/FIES-FITSarchive.2018.list",
    "https://www.not.iac.es/archive/FIES-FITSarchive.2018-2024.list",
)
TNG_ARCHIVE_URL = "https://archives.ia2.inaf.it/tng/"
ING_ARCHIVE_URL = "https://www.ing.iac.es/Astronomy/telescopes/wht/archive.html"
MATCH_RADIUS_ARCSEC = 10.0

FIES_ROW = re.compile(
    r"^(?P<dataset>\S+\.fits)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2}T\S+)\s+"
    r"(?P<exposure>\d+(?:\.\d+)?)\s+"
    r"(?P<mode>\S+)\s+\S+\s+\S+\s+"
    r"(?P<ra>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<dec>[+-]?\d+(?:\.\d+)?)\s+"
)


def _plain(value):
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _separation_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(math.radians, map(float, (ra1, dec1, ra2, dec2)))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _expected_position(ra, dec, pmra, pmdec, reference_epoch, target_epoch):
    propagated = coordinates_at_epoch(
        ra, dec, pmra, pmdec, reference_epoch, target_epoch
    )
    return propagated or {"ra": float(ra), "dec": float(dec)}


def _iso_from_mjd(mjd):
    if mjd is None:
        return None
    timestamp = datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=float(mjd))
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def query_carmenes_dr1(ra, dec, pmra=None, pmdec=None, reference_epoch=None):
    """Find public CARMENES DR1 spectra through the GAVO TAP service."""
    # carmenes.objects publishes SIMBAD coordinates at J2000. Propagating the
    # loaded Gaia position to the observing years can miss nearby high-proper-
    # motion stars by many arcseconds (GX And is displaced by roughly 50 arcsec).
    position = _expected_position(ra, dec, pmra, pmdec, reference_epoch, 2000.0)
    tap = TapPlus(url=GAVO_TAP_URL)
    object_query = f"""
        SELECT carmenes_id, name, nobs, nspect, ra, dec, survey
        FROM carmenes.objects
        WHERE 1=CONTAINS(
            POINT('ICRS',ra,dec),
            CIRCLE('ICRS',{position['ra']:.9f},{position['dec']:.9f},
                   {MATCH_RADIUS_ARCSEC / 3600.0:.10f})
        )
    """
    objects = tap.launch_job(
        object_query, output_format="votable", verbose=False
    ).get_results()
    if not len(objects):
        return {"rows": [], "objects": [], "coverage": "GTO DR1 · 2016–2020"}

    object_rows = [
        {column: _plain(row[column]) for column in objects.colnames}
        for row in objects
    ]
    identifiers = [row["carmenes_id"] for row in object_rows if row.get("carmenes_id")]
    quoted = ",".join("'" + identifier.replace("'", "''") + "'" for identifier in identifiers)
    spectrum_query = f"""
        SELECT ssa_targname, name_hdr, ssa_dateobs, ssa_instrument,
               ssa_snr, ssa_reference, ssa_specstart, ssa_specend,
               ssa_specres, ssa_timeext, accref, datalink
        FROM carmenes.data
        WHERE ssa_targname IN ({quoted})
        ORDER BY ssa_dateobs
    """
    spectra = tap.launch_job(
        spectrum_query, output_format="votable", verbose=False
    ).get_results()
    rows = []
    for spectrum in spectra:
        values = {column: _plain(spectrum[column]) for column in spectra.colnames}
        exposure_days = values.get("ssa_timeext")
        rows.append({
            "archive": "CARMENES DR1",
            "dp_id": values.get("accref"),
            "instrument": values.get("ssa_instrument") or "CARMENES",
            "dp_tech": "ECHELLE",
            "ins_mode": "VIS + NIR",
            "exp_start": _iso_from_mjd(values.get("ssa_dateobs")),
            "exposure": float(exposure_days) * 86400.0 if exposure_days is not None else None,
            "prog_id": "CARMENES GTO DR1",
            "prog_title": "The CARMENES search for exoplanets around M dwarfs",
            "pi_coi": None,
            "public": True,
            "rv_relevance": "High-resolution RV",
            "separation_arcsec": None,
            "datalink_url": values.get("datalink"),
            "access_url": values.get("accref"),
            "snr": values.get("ssa_snr"),
            "reference": values.get("ssa_reference"),
            "target_name": values.get("ssa_targname"),
        })
    return {
        "rows": rows,
        "objects": object_rows,
        "coverage": "GTO DR1 · 2016–2020",
    }


def _download_text(url):
    request = Request(url, headers={"User-Agent": "RUNA/1.0 archival-spectroscopy"})
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def query_not_fies(ra, dec, pmra=None, pmdec=None, reference_epoch=None):
    """Match the official public NOT/FIES target-spectrum lists by position."""
    rows = []
    loaded_lists = 0
    for url in NOT_FIES_LISTS:
        text = _download_text(url)
        loaded_lists += 1
        for line in text.splitlines():
            match = FIES_ROW.match(line.strip())
            if not match:
                continue
            values = match.groupdict()
            try:
                observed = datetime.fromisoformat(values["date"].replace("Z", "+00:00"))
                start = datetime(observed.year, 1, 1, tzinfo=observed.tzinfo)
                end = datetime(observed.year + 1, 1, 1, tzinfo=observed.tzinfo)
                epoch = observed.year + (observed - start).total_seconds() / (end - start).total_seconds()
                expected = _expected_position(
                    ra, dec, pmra, pmdec, reference_epoch, epoch
                )
                separation = _separation_arcsec(
                    expected["ra"], expected["dec"], values["ra"], values["dec"]
                )
            except (TypeError, ValueError):
                continue
            if separation > MATCH_RADIUS_ARCSEC:
                continue
            rows.append({
                "archive": "NOT/FIES",
                "dp_id": values["dataset"],
                "instrument": "FIES",
                "dp_tech": "ECHELLE",
                "ins_mode": values["mode"],
                "exp_start": values["date"],
                "exposure": float(values["exposure"]),
                "prog_id": None,
                "prog_title": None,
                "pi_coi": None,
                "public": True,
                "rv_relevance": "High-resolution RV",
                "separation_arcsec": separation,
                "datalink_url": NOT_ARCHIVE_URL,
                "access_url": NOT_ARCHIVE_URL,
                "snr": None,
                "reference": None,
                "target_name": None,
            })
    return {
        "rows": rows,
        "coverage": "Official target lists · 2006–2024-05",
        "loaded_lists": loaded_lists,
    }


def search_spectroscopy_archives(ra, dec, pmra=None, pmdec=None, reference_epoch=None):
    """Search independent archives without allowing one failure to hide the others."""
    sources = []
    all_rows = []
    automatic = (
        ("ESO", "Raw science spectra", "https://archive.eso.org/", query_eso_spectroscopy),
        ("CARMENES DR1", "GTO spectra · 2016–2020", CARMENES_ARCHIVE_URL, query_carmenes_dr1),
        ("NOT/FIES", "Target lists · 2006–2024-05", NOT_ARCHIVE_URL, query_not_fies),
    )
    for name, coverage, url, query_function in automatic:
        try:
            result = query_function(ra, dec, pmra, pmdec, reference_epoch)
        except Exception as exc:
            sources.append({
                "Archive": name,
                "Coverage": coverage,
                "Matches": None,
                "Status": f"Unavailable · {type(exc).__name__}",
                "Link": url,
            })
            continue
        rows = result.get("rows", [])
        for row in rows:
            row.setdefault("archive", name)
        all_rows.extend(rows)
        sources.append({
            "Archive": name,
            "Coverage": result.get("coverage", coverage),
            "Matches": len(rows),
            "Status": "Matched" if rows else "No positional match",
            "Link": url,
        })

    manual = [
        {
            "Archive": "TNG / HARPS-N",
            "Coverage": "Raw and reduced HARPS-N data",
            "Access": "Manual archive search",
            "Link": TNG_ARCHIVE_URL,
        },
        {
            "Archive": "WHT / ING",
            "Coverage": "UES, ISIS and other WHT spectroscopy",
            "Access": "Manual archive and observing-log search",
            "Link": ING_ARCHIVE_URL,
        },
        {
            "Archive": "Calar Alto",
            "Coverage": "CARMENES, CAFE, FOCES and other public CAHA data",
            "Access": "Manual archive search beyond CARMENES DR1",
            "Link": CAHA_ARCHIVE_URL,
        },
    ]
    sources.extend(
        {
            "Archive": item["Archive"],
            "Coverage": item["Coverage"],
            "Matches": "—",
            "Status": item["Access"],
            "Link": item["Link"],
        }
        for item in manual
    )
    return {"rows": all_rows, "sources": sources, "manual": manual}
