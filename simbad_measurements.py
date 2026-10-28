"""Bibliographic stellar measurements from the SIMBAD TAP collections."""

import numpy as np
from astroquery.simbad import Simbad

from identify_star import available_simbad_server


GAIA_CATALOGUE_BIBCODES = {
    "2018yCat.1345....0G",  # Gaia DR2
    "2020yCat.1350....0G",  # Gaia EDR3
    "2022yCat.1355....0G",  # Gaia DR3
}


def _value(row, column):
    value = row[column]
    if np.ma.is_masked(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _is_literature(bibcode):
    return bool(bibcode) and bibcode not in GAIA_CATALOGUE_BIBCODES


def _measurement(value, error, bibcode, qualifier=None):
    return {
        "value": value,
        "error": error,
        "bibcode": bibcode,
        "qualifier": qualifier,
    }


def lookup_simbad_measurements(main_id):
    """Return literature [Fe/H], v sin i, rotation periods and spectral types.

    Each collection is queried separately to avoid multiplying rows through
    a four-way join of SIMBAD's one-to-many measurement tables.
    """
    measurements = {
        "metallicity": [],
        "vsini": [],
        "rotation_period": [],
        "spectral_type": [],
        "unavailable": [],
    }
    if not main_id:
        return measurements

    simbad = Simbad()
    simbad.server = available_simbad_server()
    simbad.timeout = 20
    safe_id = str(main_id).replace("'", "''")
    specs = (
        ("metallicity", "mesFe_h", "m.fe_h, m.flag"),
        ("vsini", "mesRot", "m.vsini, m.vsini_err, m.upvsini"),
        ("rotation_period", "mesVar", "m.period, m.vartyp, m.upperiod"),
        ("spectral_type", "mesSpT", "m.sptype"),
    )
    for kind, table, fields in specs:
        query = (
            f"SELECT {fields}, m.bibcode FROM {table} AS m "
            "JOIN basic AS b ON m.oidref = b.oid "
            f"WHERE b.main_id = '{safe_id}'"
        )
        try:
            rows = simbad.query_tap(query)
        except Exception:
            measurements["unavailable"].append(kind)
            continue
        for row in rows:
            bibcode = _value(row, "bibcode")
            if not _is_literature(bibcode):
                continue
            if kind == "metallicity":
                value = _value(row, "fe_h")
                if value is not None:
                    measurements[kind].append(
                        _measurement(float(value), None, bibcode, _value(row, "flag"))
                    )
            elif kind == "vsini":
                value = _value(row, "vsini")
                if value is not None:
                    error = _value(row, "vsini_err")
                    measurements[kind].append(
                        _measurement(
                            float(value), float(error) if error is not None else None,
                            bibcode, _value(row, "upvsini"),
                        )
                    )
            elif kind == "rotation_period":
                value = _value(row, "period")
                vartyp = str(_value(row, "vartyp") or "").upper()
                if value is not None and float(value) > 0 and vartyp.startswith("ROT"):
                    measurements[kind].append(
                        _measurement(float(value), None, bibcode, _value(row, "upperiod"))
                    )
            else:
                value = _value(row, "sptype")
                if value is not None:
                    measurements[kind].append(_measurement(str(value), None, bibcode))
        measurements[kind].sort(key=lambda item: item["bibcode"], reverse=True)
    return measurements
