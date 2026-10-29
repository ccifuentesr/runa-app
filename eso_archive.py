"""On-demand search for stellar spectroscopy in the ESO Science Archive."""

from collections import defaultdict
from datetime import date, datetime, timezone
import math

import numpy as np
from astroquery.utils.tap.core import TapPlus

from identify_star import coordinates_at_epoch


ESO_TAP_URL = "https://archive.eso.org/tap_obs"
ESO_TELBIB_URL = "https://telbib.eso.org/"
MATCH_RADIUS_ARCSEC = 10.0
MAX_ROWS = 2000

HIGH_RV_INSTRUMENTS = {
    "ESPRESSO", "HARPS", "NIRPS", "UVES", "FEROS", "CRIRES", "CRIRES+",
}
CONTEXT_INSTRUMENTS = {
    "XSHOOTER", "FLAMES", "GIRAFFE", "FORS1", "FORS2", "EFOSC2",
    "EMMI", "ISAAC", "SINFONI", "KMOS", "MUSE", "SOFI", "VISIR", "CES",
}
SPECTROGRAPH_INSTRUMENTS = sorted(HIGH_RV_INSTRUMENTS | CONTEXT_INSTRUMENTS)


def _plain(value):
    """Convert an Astropy value into a JSON/cache-friendly Python value."""
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


def _year_from_timestamp(timestamp):
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(str(timestamp)[:10], "%Y-%m-%d")
        except ValueError:
            return None
    start = datetime(parsed.year, 1, 1, tzinfo=parsed.tzinfo)
    end = datetime(parsed.year + 1, 1, 1, tzinfo=parsed.tzinfo)
    return parsed.year + (parsed - start).total_seconds() / (end - start).total_seconds()


def _angular_separation_arcsec(ra1, dec1, ra2, dec2):
    values = (ra1, dec1, ra2, dec2)
    if any(value is None for value in values):
        return None
    ra1, dec1, ra2, dec2 = map(math.radians, map(float, values))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _instrument_relevance(instrument):
    name = str(instrument or "").upper()
    if name in HIGH_RV_INSTRUMENTS or any(
        token in name for token in ("HARPS", "ESPRESSO", "NIRPS", "UVES", "FEROS", "CRIRES")
    ):
        return "High-resolution RV"
    if name in CONTEXT_INSTRUMENTS or any(token in name for token in CONTEXT_INSTRUMENTS):
        return "Contextual spectroscopy"
    return "Spectroscopy"


def _search_region(ra, dec, pmra, pmdec, reference_epoch):
    """Cover the proper-motion track with one compact archive cone."""
    current_year = datetime.now(timezone.utc).year
    if any(value is None for value in (pmra, pmdec, reference_epoch)):
        return float(ra), float(dec), MATCH_RADIUS_ARCSEC

    first_year = 1990.0
    last_year = float(current_year + 1)
    middle_year = (first_year + last_year) / 2.0
    centre = coordinates_at_epoch(
        ra, dec, pmra, pmdec, reference_epoch, middle_year
    )
    if centre is None:
        return float(ra), float(dec), MATCH_RADIUS_ARCSEC

    proper_motion_arcsec_year = math.hypot(float(pmra), float(pmdec)) / 1000.0
    track_half_length = proper_motion_arcsec_year * (last_year - first_year) / 2.0
    radius_arcsec = MATCH_RADIUS_ARCSEC + track_half_length
    return centre["ra"], centre["dec"], radius_arcsec


def _archive_query(ra, dec, radius_arcsec):
    circle = (
        "INTERSECTS(s_region,CIRCLE('ICRS',"
        f"{ra:.9f},{dec:.9f},{radius_arcsec / 3600.0:.10f}))=1"
    )
    instruments = ",".join(f"'{name}'" for name in SPECTROGRAPH_INSTRUMENTS)
    return f"""
        SELECT TOP {MAX_ROWS}
            dp_id, instrument, dp_tech, dp_type, exp_start, exposure,
            prog_id, prog_title, pi_coi, release_date, ra, dec,
            telescope, ins_mode, slit_path, grat_path, gris_path,
            access_url, datalink_url
        FROM dbo.raw
        WHERE dp_cat='SCIENCE'
          AND {circle}
          AND (
              dp_tech LIKE '%SPECTR%'
              OR dp_tech LIKE '%ECHELLE%'
              OR instrument IN ({instruments})
          )
        ORDER BY exp_start
    """


def _is_public(release_date):
    if not release_date:
        return None
    try:
        return date.fromisoformat(str(release_date)[:10]) <= date.today()
    except ValueError:
        return None


def query_eso_spectroscopy(ra, dec, pmra=None, pmdec=None, reference_epoch=None):
    """Return ESO raw science spectra positionally matched to one star."""
    if ra is None or dec is None:
        raise ValueError("Coordinates are required for an ESO archive search.")

    search_ra, search_dec, search_radius = _search_region(
        ra, dec, pmra, pmdec, reference_epoch
    )
    job = TapPlus(url=ESO_TAP_URL).launch_job(
        _archive_query(search_ra, search_dec, search_radius),
        output_format="votable",
        verbose=False,
    )
    table = job.get_results()
    rows = []
    for archive_row in table:
        row = {column: _plain(archive_row[column]) for column in table.colnames}
        epoch = _year_from_timestamp(row.get("exp_start"))
        expected = coordinates_at_epoch(
            ra, dec, pmra, pmdec, reference_epoch, epoch
        ) if epoch is not None else None
        expected_ra = expected["ra"] if expected else ra
        expected_dec = expected["dec"] if expected else dec
        separation = _angular_separation_arcsec(
            expected_ra, expected_dec, row.get("ra"), row.get("dec")
        )
        if separation is not None and separation > MATCH_RADIUS_ARCSEC:
            continue
        row["separation_arcsec"] = separation
        row["public"] = _is_public(row.get("release_date"))
        row["rv_relevance"] = _instrument_relevance(row.get("instrument"))
        row["archive"] = "ESO"
        rows.append(row)

    return {
        "rows": rows,
        "truncated": len(table) >= MAX_ROWS,
        "search_radius_arcsec": search_radius,
        "match_radius_arcsec": MATCH_RADIUS_ARCSEC,
    }


def summarise_eso_spectroscopy(rows):
    """Aggregate matched files by instrument and by observing programme."""
    instrument_groups = defaultdict(list)
    programme_groups = defaultdict(list)
    for row in rows:
        archive = row.get("archive") or "ESO"
        instrument_groups[(archive, row.get("instrument") or "Unknown")].append(row)
        programme_groups[(archive, row.get("prog_id") or "Unknown")].append(row)

    instruments = []
    for (archive, instrument), group in instrument_groups.items():
        dates = sorted({str(row["exp_start"])[:10] for row in group if row.get("exp_start")})
        public_count = sum(row.get("public") is True for row in group)
        exposure_hours = sum(
            float(row["exposure"]) for row in group if row.get("exposure") is not None
        ) / 3600.0
        instruments.append({
            "Archive": archive,
            "Instrument": instrument,
            "RV relevance": group[0]["rv_relevance"],
            "Files": len(group),
            "UTC dates": len(dates),
            "Exposure h": round(exposure_hours, 2),
            "First": dates[0] if dates else "—",
            "Last": dates[-1] if dates else "—",
            "Public": public_count,
        })
    relevance_order = {"High-resolution RV": 0, "Contextual spectroscopy": 1, "Spectroscopy": 2}
    instruments.sort(key=lambda item: (relevance_order[item["RV relevance"]], item["Instrument"]))

    programmes = []
    for (archive, programme_id), group in programme_groups.items():
        dates = sorted({str(row["exp_start"])[:10] for row in group if row.get("exp_start")})
        instruments_used = sorted({row.get("instrument") or "Unknown" for row in group})
        title = next((row.get("prog_title") for row in group if row.get("prog_title")), "—")
        investigators = next((row.get("pi_coi") for row in group if row.get("pi_coi")), "—")
        programmes.append({
            "Archive": archive,
            "Programme": programme_id,
            "Instrument": ", ".join(instruments_used),
            "Files": len(group),
            "UTC dates": len(dates),
            "Title": title,
            "PI / Co-I": investigators,
        })
    programmes.sort(key=lambda item: item["Programme"])

    dates = sorted({str(row["exp_start"])[:10] for row in rows if row.get("exp_start")})
    public_count = sum(row.get("public") is True for row in rows)
    proprietary_count = sum(row.get("public") is False for row in rows)
    return {
        "instruments": instruments,
        "programmes": programmes,
        "files": len(rows),
        "epochs": len(dates),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
        "public": public_count,
        "proprietary": proprietary_count,
    }


def eso_file_details(rows):
    """Prepare the matched raw-file metadata for a compact Streamlit table."""
    details = []
    for row in rows:
        separation = row.get("separation_arcsec")
        details.append({
            "Archive source": row.get("archive") or "ESO",
            "Date": str(row.get("exp_start") or "")[:19] or "—",
            "Instrument": row.get("instrument") or "—",
            "Technique": row.get("dp_tech") or "—",
            "Mode": row.get("ins_mode") or "—",
            "Exposure s": row.get("exposure"),
            "Programme": row.get("prog_id") or "—",
            "Public": row.get("public"),
            "S/N": row.get("snr"),
            "Offset arcsec": round(float(separation), 2) if separation is not None else None,
            "Dataset": row.get("dp_id") or "—",
            "Archive": row.get("datalink_url") or row.get("access_url"),
        })
    return details
