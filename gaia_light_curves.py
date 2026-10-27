"""Gaia DR3 epoch photometry, adapted from kobe.query_gaia_lightcurve.py.

Query the same Gaia@AIP array table as Carlos Cifuentes's KOBE script.
Keep time, magnitude and variability rejection flags aligned per transit.
"""

from io import BytesIO
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io.votable import parse


BANDS = {"G": ("g_transit_time", "g_transit_mag", "#6ee7b7"),
         "BP": ("bp_obs_time", "bp_mag", "#79aaff"),
         "RP": ("rp_obs_time", "rp_mag", "#f28482")}


def unpack_epoch_photometry(table):
    """Flatten the array columns without losing masked-value/flag alignment."""
    records = []
    for row in table:
        for band, (time_column, mag_column, _) in BANDS.items():
            times = np.ma.asarray(row[time_column], dtype=float).reshape(-1)
            mags = np.ma.asarray(row[mag_column], dtype=float).reshape(-1)
            flags = np.ma.asarray(row[f"variability_flag_{band.lower()}_reject"]).reshape(-1)
            if not len(times) == len(mags) == len(flags):
                raise ValueError(f"Mismatched epoch arrays in {band}")
            for time, mag, flag in zip(times, mags, flags):
                if np.ma.is_masked(time) or np.ma.is_masked(mag):
                    continue
                if not np.isfinite(time) or not np.isfinite(mag):
                    continue
                records.append({
                    "source_id": str(row["source_id"]), "band": band,
                    "time_bjd_minus_2455197_5": float(time), "mag": float(mag),
                    "variability_rejected": None if np.ma.is_masked(flag) else bool(flag),
                })
    return pd.DataFrame(records, columns=[
        "source_id", "band", "time_bjd_minus_2455197_5", "mag", "variability_rejected",
    ]).sort_values(["band", "time_bjd_minus_2455197_5"]).reset_index(drop=True)


def query_gaia_epoch_photometry(source_id):
    """Retrieve published epoch photometry for the exact Gaia DR3 source."""
    if not str(source_id).isdigit():
        raise ValueError("A numeric Gaia DR3 source ID is required")
    columns = ["source_id"]
    for band, (time_column, mag_column, _) in BANDS.items():
        columns.extend([time_column, mag_column, f"variability_flag_{band.lower()}_reject"])
    query = ("SELECT " + ", ".join(columns)
             + f" FROM gaiadr3.epoch_photometry WHERE source_id = {source_id}")
    params = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query})
    with urlopen("https://gaia.aip.de/tap/sync?" + params, timeout=25) as response:
        document = parse(BytesIO(response.read()))
    for resource in document.resources:
        for info in resource.infos:
            if info.name == "QUERY_STATUS" and info.value == "ERROR":
                raise RuntimeError(info.content or "Gaia@AIP query failed")
    table = document.get_first_table().to_table(use_names_over_ids=True)
    return unpack_epoch_photometry(table)


def gaia_epoch_figure(data):
    """Plot all three passbands on the same inverted magnitude axis."""
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=140)
    fig.patch.set_facecolor("#18181b")
    ax.set_facecolor("#18181b")
    for band, (_, _, color) in BANDS.items():
        points = data[data["band"] == band]
        if points.empty:
            continue
        ax.scatter([], [], edgecolors=color, facecolors="none", label=band)
        for flag, marker in ((False, "o"), (True, "x")):
            selected = points[points["variability_rejected"].eq(flag)]
            style = {"color": color} if flag else {"edgecolors": color, "facecolors": "none"}
            ax.scatter(selected["time_bjd_minus_2455197_5"], selected["mag"],
                       marker=marker, s=27, linewidths=1.1, **style)
        unknown = points[points["variability_rejected"].isna()]
        ax.scatter(unknown["time_bjd_minus_2455197_5"], unknown["mag"],
                   marker=".", s=18, color=color)
    ax.invert_yaxis()
    ax.set_xlabel("BJD − 2455197.5 [days]", color="#d4d4d8")
    ax.set_ylabel("Magnitude [mag]", color="#d4d4d8")
    ax.tick_params(colors="#a1a1aa", direction="in", top=True, right=True)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for spine in ax.spines.values():
        spine.set_color("#53535c")
    ax.legend(frameon=False, labelcolor="#d4d4d8", ncol=3, loc="best")
    ax.margins(x=.05, y=.15)
    fig.tight_layout()
    return fig
