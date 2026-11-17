"""Web interface for RUNA."""

from html import escape
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from identify_star import (
    coordinates_at_epoch,
    fetch_survey_image,
    lookup_2mass_j,
    lookup_gaia_photometry,
    lookup_johnson_v,
    lookup_star,
)
from kinematics import banyan_inputs, run_banyan
from light_curves import fetch_tess_light_curve, light_curve_figure, tess_identifier
from gaia_light_curves import query_gaia_epoch_photometry, gaia_epoch_figure
from gaia_neighbours import (
    companion_diagnostics,
    local_proper_motion_plot,
    prepare_neighbours,
    query_gaia_neighbours,
)
from multiplicity import (
    check_multiplicity,
    query_rvs_reference,
    rvs_ruwe_plot,
)
from parameter_table import stellar_parameter_table, panel_heading
from simbad_measurements import lookup_simbad_measurements
from survey_view import survey_image_view
from parameters import (
    calculate_parameters,
    mass_from_luminosity_eker,
    metallicity_from_gaia_wise,
    radius_from_luminosity_temperature,
)
from photometry import (
    build_sed_points,
    catalogue_photometry_csv,
    fetch_catalogue_photometry,
    flag_sed_anomalies,
    load_filter_curve,
    load_gaia_reference_sample,
    star_cmd_position,
)
from planets import (
    discovery_references, load_planet_population, lookup_open_exoplanet_catalogue, lookup_planets,
    merge_planets, orbital_mass,
)
from position_plots import polar_position_plot
from predict_star import estimate_subtype
from models_view import show_models
from train_model import load_stars
from multiplicity_model import gaia_feature_row, model_definitions, prepare_training_data
from vosa_input import build_vosa_input
from cifuentes20 import load_reference_data, temperature_from_spectral_type
from mamajek import spectral_type_key, temperature_from_mamajek
from local_volume import local_volume_status
from published_parameters import published_stellar_parameters, published_multiplicity
from wds import closest_primary_component
from eso_archive import (
    ESO_TELBIB_URL,
    eso_file_details,
    summarise_eso_spectroscopy,
)
from spectroscopy_archives import search_spectroscopy_archives


LOGGER = logging.getLogger(__name__)


def source_label_html(source):
    """Abbreviate ADS bibcodes in tables while preserving the full value on hover."""
    if not source:
        return ""
    text = str(source)
    label = text.split(".", 1)[0] if text[:4].isdigit() and "." in text else text
    if label != text:
        return f'<span title="{escape(text, quote=True)}">{escape(label)}</span>'
    return escape(text)


def data_row(label, value, source=None, error=None, unit=None, trailing_html=""):
    """Build one compact row for the identification panel."""
    label_html = escape(str(label)).replace("M_G", "M<sub>G</sub>")
    label_html = label_html.replace("M_J", "M<sub>J</sub>")
    label_html = label_html.replace("T_eff", "T<sub>eff</sub>")
    error_html = (
        f'<span class="runa-error">({escape(error)})</span>'
        if error else ""
    )
    unit_html = escape(unit) if unit else ""
    source_html = source_label_html(source)
    row_class = "runa-data-row" if unit else "runa-data-row runa-data-row-no-unit"
    return (
        f'<div class="{row_class}">'
        f'<span class="runa-data-label">{label_html}</span>'
        '<span class="runa-data-value">'
        f'{escape(str(value))}{error_html}{trailing_html}</span>'
        f'<span class="runa-data-unit">{unit_html}</span>'
        f'<span class="runa-data-source">{source_html}</span></div>'
    )


def data_section(title, rows, class_name=""):
    """Show a titled group of identification values."""
    content = "".join(data_row(*row) for row in rows if row[1] is not None)
    if content:
        title_html = escape(str(title)).replace("M_G", "M<sub>G</sub>")
        title_html = title_html.replace("M_J", "M<sub>J</sub>")
        section_classes = "runa-data-section"
        if class_name:
            section_classes += f" {class_name}"
        st.markdown(
            f'<section class="{section_classes}"><div class="runa-section-title">'
            f'{title_html}</div>{content}</section>',
            unsafe_allow_html=True,
        )


def parameter_section(title, rows, empty_message=None):
    """Show published or estimated values inside a compact parameter card."""
    kind = "estimated" if title.lower() == "estimated" else "published"
    content = []
    for label, value, source, error, unit in rows:
        if value is None:
            continue
        label_html = escape(str(label)).replace("M_G", "M<sub>G</sub>")
        label_html = label_html.replace("M_J", "M<sub>J</sub>")
        label_html = label_html.replace("T_eff", "T<sub>eff</sub>")
        error_html = (
            f'<span class="runa-parameter-error">({escape(error)})</span>'
            if error else ""
        )
        unit_html = (
            f'<span class="runa-parameter-unit">{escape(unit)}</span>'
            if unit else ""
        )
        source_html = source_label_html(source) if source else "Source unavailable"
        content.append(
            '<div class="runa-parameter-row">'
            f'<div class="runa-parameter-label">{label_html}</div>'
            '<div class="runa-parameter-value">'
            f'{escape(str(value))}{error_html}{unit_html}</div>'
            f'<div class="runa-parameter-source">{source_html}</div></div>'
        )

    if not content and empty_message:
        content.append(
            f'<div class="runa-parameter-empty">{escape(empty_message)}</div>'
        )
    if content:
        st.markdown(
            f'<section class="runa-parameter-section runa-parameter-{kind}">'
            f'<div class="runa-parameter-kind">{escape(title)}</div>'
            f'{"".join(content)}</section>',
            unsafe_allow_html=True,
        )


def format_value_and_error(value, error, significant_digits=6):
    """Format a positive value and its uncertainty with matching decimals."""
    value = float(value)
    decimals = (
        max(0, significant_digits - 1 - math.floor(math.log10(abs(value))))
        if value else significant_digits
    )
    value_text = f"{value:.{decimals}f}"
    error_text = f"{float(error):.{decimals}f}" if error is not None else None
    return value_text, error_text


def cifuentes20_table(rows):
    """Render Cifuentes et al. (2020), Tables 6 and 7 as one table."""
    magnitude_columns = (
        ("mb", "mb_error", "M<sub>B</sub>"),
        ("mg_prime", "mg_prime_error", "M<sub>g′</sub>"),
        ("mgbp", "mgbp_error", "M<sub>G<span>BP</span></sub>"),
        ("mv", "mv_error", "M<sub>V</sub>"),
        ("mr_prime", "mr_prime_error", "M<sub>r′</sub>"),
        ("mg", "mg_error", "M<sub>G</sub>"),
        ("mi_prime", "mi_prime_error", "M<sub>i′</sub>"),
        ("mgrp", "mgrp_error", "M<sub>G<span>RP</span></sub>"),
        ("mj", "mj_error", "M<sub>J</sub>"),
        ("mh", "mh_error", "M<sub>H</sub>"),
        ("mks", "mks_error", "M<sub>K<span>s</span></sub>"),
        ("mw1", "mw1_error", "M<sub>W1</sub>"),
        ("mw2", "mw2_error", "M<sub>W2</sub>"),
        ("mw3", "mw3_error", "M<sub>W3</sub>"),
    )

    def measurement_cell(row, value_key, error_key):
        value = row.get(value_key)
        if value in (None, ""):
            return '<td class="runa-cifuentes-missing">—</td>'
        error = row.get(error_key)
        error_html = (
            f' <span class="runa-cifuentes-error">({escape(str(error))})</span>'
            if error not in (None, "") else ""
        )
        return f'<td>{escape(str(value))}{error_html}</td>'

    magnitude_headers = "".join(
        f'<th class="runa-cifuentes-band">{label}<small>mag</small></th>'
        for _, _, label in magnitude_columns
    )
    header = (
        '<div class="runa-cifuentes-scroll"><table class="runa-cifuentes-table">'
        '<thead><tr>'
        '<th class="runa-cifuentes-type">Spectral type</th>'
        '<th>Luminosity<small>10<sup>−4</sup> L<sub>sol</sub></small></th>'
        '<th>T<sub>eff</sub><small>K</small></th>'
        '<th>Radius<small>R<sub>sol</sub></small></th>'
        '<th>Mass<small>M<sub>sol</sub></small></th>'
        f'{magnitude_headers}'
        '<th>BC<sub>G</sub><small>mag</small></th>'
        '<th>BC<sub>J</sub><small>mag</small></th>'
        '<th>N</th></tr></thead><tbody>'
    )
    body = []
    for row in rows:
        cells = [
            f'<td class="runa-cifuentes-type">{escape(str(row["sp_type"]))}</td>',
            measurement_cell(row, "lbol", "lbol_error"),
            measurement_cell(row, "teff", "teff_error"),
            measurement_cell(row, "radius", "radius_error"),
            measurement_cell(row, "mass", "mass_error"),
        ]
        cells.extend(
            measurement_cell(row, value_key, error_key)
            for value_key, error_key, _ in magnitude_columns
        )
        cells.extend((
            measurement_cell(row, "bc_g", "bc_g_error"),
            measurement_cell(row, "bc_j", "bc_j_error"),
            f'<td>{escape(str(row["n"]))}</td>',
        ))
        body.append(f'<tr>{"".join(cells)}</tr>')
    html = header + "".join(body) + '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)


def catalogue_photometry_table(records, sort_by_wavelength=False):
    """Show extra catalogue bands, optionally ordered by effective wavelength."""
    bands = []
    for record in records:
        for band in record["bands"]:
            row = dict(band)
            row["catalogue"] = record["catalogue"]
            row["matched"] = record["matched"]
            bands.append(row)

    if sort_by_wavelength:
        bands.sort(
            key=lambda band: (
                band.get("wavelength") is None,
                band.get("wavelength") or 0,
            )
        )

    rows = [
        '<section class="runa-photometry-table">'
        '<div class="runa-photometry-row runa-photometry-head">'
        '<span>Catalogue</span><span>Band</span><span>λ<sub>eff</sub><small>Å</small></span>'
        '<span>Magnitude</span><span>Error</span><span>Range</span></div>'
    ]
    for band in bands:
        value = band["value"]
        error = band["error"]
        wavelength = band.get("wavelength")
        value_text = f"{value:.3f}" if value is not None else "—"
        error_text = f"({error:.3f})" if error is not None else "—"
        wavelength_text = f"{wavelength:.1f}" if wavelength is not None else "—"
        catalogue_class = " runa-photometry-missing" if not band["matched"] else ""
        colour = escape(str(band["colour"]))
        warning = band.get("sed_warning")
        value_html = escape(value_text)
        if warning:
            value_html += (
                '<span class="runa-photometry-warning" tabindex="0" '
                f'aria-label="Suspicious magnitude: {escape(warning, quote=True)}">'
                '<span aria-hidden="true">!</span>'
                f'<span class="runa-photometry-warning-text">{escape(warning)}</span></span>'
            )
        rows.append(
            f'<div class="runa-photometry-row{catalogue_class}">'
            f'<span class="runa-photometry-catalogue">{escape(band["catalogue"])}</span>'
            f'<span class="runa-band-name" style="--band-colour: {colour}">'
            f'{escape(band["label"])}</span>'
            f'<span class="runa-photometry-wavelength">{escape(wavelength_text)}</span>'
            f'<span>{value_html}</span>'
            f'<span class="runa-photometry-error">{escape(error_text)}</span>'
            f'<span class="runa-photometry-range">{escape(band["range"])}</span>'
            '</div>'
        )
    rows.append('</section>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def catalogue_band(records, catalogue, label):
    """Find one band in the records loaded by the Photometry search."""
    for record in records or []:
        if record.get("catalogue") != catalogue:
            continue
        for band in record.get("bands", []):
            if band.get("label") == label:
                return band
    return None


def sed_figure(points, object_name, show_bands=False):
    """Build the observed lambda-F-lambda plot for catalogue photometry."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")

    labelled_wavelengths = set()
    last_label_wavelength = None
    last_label_height = 1.025
    for point in points:
        y_error = point.get("lambda_flux_error")
        ax.errorbar(
            point["wavelength"],
            point["lambda_flux"],
            yerr=y_error,
            fmt="o",
            markersize=6,
            color=point["colour"],
            ecolor=point["colour"],
            capsize=2 if y_error is not None else 0,
            linewidth=0.9,
            zorder=2,
        )
        if point.get("sed_warning"):
            ax.scatter(
                [point["wavelength"]], [point["lambda_flux"]],
                s=150, marker="o", facecolors="none", edgecolors="#f4cf75",
                linewidths=1.3, zorder=4,
            )
        wavelength_key = round(point["wavelength"], 1)
        if wavelength_key not in labelled_wavelengths:
            label_height = 1.025
            if (
                last_label_wavelength is not None
                and point["wavelength"] / last_label_wavelength < 1.25
            ):
                label_height = 1.10 if last_label_height == 1.025 else 1.025
            ax.text(
                point["wavelength"],
                label_height,
                point["label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="600",
                color=point["colour"],
                clip_on=False,
            )
            labelled_wavelengths.add(wavelength_key)
            last_label_wavelength = point["wavelength"]
            last_label_height = label_height

    ax.set_xscale("log")
    ax.set_yscale("log")
    if show_bands:
        y_min, y_max = ax.get_ylim()
        log_min, log_max = np.log10(y_min), np.log10(y_max)
        plotted_curves = set()
        for point in points:
            label = point["label"]
            if point["catalogue"] == "SkyMapper DR4":
                continue
            if label in plotted_curves:
                continue
            curve = load_filter_curve(label)
            if curve is not None:
                curve_wavelength, transmission = curve
                transmission = np.clip(transmission, 0.0, None)
                maximum = transmission.max()
                if maximum > 0:
                    curve_height = 10 ** (
                        log_min + (transmission / maximum) * (log_max - log_min)
                    )
                    ax.fill_between(
                        curve_wavelength,
                        y_min,
                        curve_height,
                        color=point["colour"],
                        alpha=0.10,
                        linewidth=0,
                        zorder=0,
                    )
                    ax.plot(
                        curve_wavelength,
                        curve_height,
                        color=point["colour"],
                        alpha=0.28,
                        linewidth=0.7,
                        zorder=0.5,
                    )
                    plotted_curves.add(label)
                    continue

            width = point.get("width")
            if width is not None:
                half_width = width / 2.0
                ax.axvspan(
                    point["wavelength"] - half_width,
                    point["wavelength"] + half_width,
                    color=point["colour"],
                    alpha=0.11,
                    linewidth=0,
                    zorder=0,
                )
            plotted_curves.add(label)

    ax.set_xlabel("Effective wavelength λ (Å)", fontsize=16, labelpad=8)
    ax.set_ylabel("λFλ (W m⁻²)", fontsize=16, labelpad=8)
    ax.tick_params(colors="#a1a1aa", labelsize=12)
    ax.xaxis.label.set_color("#d4d4d8")
    ax.yaxis.label.set_color("#d4d4d8")
    ax.title.set_color("#f4f4f5")
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
    ax.grid(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return fig


def large_metric(label, value, error=None):
    """Show a compact large value while keeping simple catalogue notation."""
    label_html = escape(str(label)).replace("M_G", "M<sub>G</sub>")
    label_html = label_html.replace("M_J", "M<sub>J</sub>")
    error_html = (
        f'<div class="runa-large-metric-error">({escape(str(error))})</div>'
        if error is not None else ""
    )
    st.markdown(
        '<div class="runa-large-metric">'
        f'<div class="runa-large-metric-label">{label_html}</div>'
        f'<div class="runa-large-metric-value">{escape(str(value))}</div>'
        f'{error_html}</div>',
        unsafe_allow_html=True,
    )


def neighbour_table(sources, target_source_id=None):
    """Show the Gaia sources in the same compact style as identification."""
    target = next((s for s in sources if s["source_id"] == str(target_source_id)), None)
    rows = [
        '<section class="runa-neighbour-table">'
        '<div class="runa-neighbour-row runa-neighbour-head">'
        '<span>Gaia DR3 ID</span>'
        '<span class="runa-neighbour-heading runa-radius-head">'
        'r<small>arcsec</small></span>'
        '<span class="runa-neighbour-heading">parallax<small>mas</small></span>'
        '<span class="runa-neighbour-heading">G<small>mag</small></span></div>'
    ]
    for source in sources:
        is_target = (
            target_source_id
            and source["source_id"] == str(target_source_id)
        )
        row_class = " runa-neighbour-target" if is_target else ""
        target_badge = (
            '<span class="runa-neighbour-badge">Target</span>'
            if is_target else ""
        )
        diagnostics = companion_diagnostics(source, target)
        if diagnostics and diagnostics["check"]:
            row_class = " runa-neighbour-check"
            detail = (f"Distance difference: {100 * diagnostics['distance_ratio']:.2f}% · "
                      f"Proper-motion ratio: {100 * diagnostics['motion_ratio']:.2f}% · "
                      f"Direction difference: {diagnostics['delta_pa_deg']:.2f} deg")
            target_badge = (f'<span class="runa-neighbour-badge runa-companion-check" '
                            f'title="{escape(detail)}">Check</span>')
        separation = f"{source['separation_arcsec']:.2f}"
        parallax = (
            f"{source['parallax']:.3f}"
            if source["parallax"] is not None else "—"
        )
        g_mag = (
            f"{source['g_mag']:.3f}"
            if source["g_mag"] is not None else "—"
        )
        rows.append(
            f'<div class="runa-neighbour-row{row_class}">'
            f'<span>{escape(source["source_id"])} {target_badge}</span>'
            f'<span>{escape(separation)}</span>'
            f'<span>{escape(parallax)}</span>'
            f'<span>{escape(g_mag)}</span>'
            '</div>'
        )
    rows.append("</section>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption(
        "Check: distance and proper motion are compatible with the target under the "
        "[Cifuentes et al. (2025)](https://doi.org/10.1051/0004-6361/202452527) cuts. "
        "A possible physical companion, not a confirmed one. "
        "Missing astrometry is not assessed; close pairs may have distorted Gaia measurements."
    )
    if target is None:
        st.caption("Target absent from this table: companion checks are unavailable.")


def planet_table(planets):
    """Show planet status, parameters and source in a compact table."""
    rows = [
        '<section class="runa-planet-table">'
        '<div class="runa-planet-row runa-planet-head">'
        '<span>Planet</span><span>Status</span>'
        '<span class="runa-orbital-symbol">P<sub>orb</sub><small>days</small></span>'
        '<span class="runa-orbital-symbol">R<small>R⊕</small></span>'
        '<span>M<small>M⊕</small></span>'
        '<span class="runa-orbital-symbol">a<small>AU</small></span>'
        '<span class="runa-orbital-symbol">e</span></div>'
    ]

    def display_number(value, decimals):
        try:
            number = float(value)
            return f"{number:,.{decimals}f}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    for planet in planets:
        status = str(planet.get("status") or "Unknown")
        status_class = "confirmed" if status == "Confirmed" else "candidate"
        mass_value, mass_kind = orbital_mass(planet)
        mass_note = "<small>M sin i</small>" if mass_kind == "minimum" else ""
        source_detail = " + ".join(
            source for source in str(planet.get("source") or "").split(" + ")
            if source != "NASA Exoplanet Archive"
        )
        details = " · ".join(
            str(value) for value in (
                planet.get("discoverymethod"), planet.get("disc_year"),
                source_detail,
            ) if value
        )
        rows.append(
            '<div class="runa-planet-row">'
            f'<span>{escape(str(planet.get("pl_name") or "—"))}'
            f'<small>{escape(details)}</small></span>'
            f'<span class="runa-planet-status-{status_class}">{escape(status)}</span>'
            f'<span>{display_number(planet.get("pl_orbper"), 2)}</span>'
            f'<span>{display_number(planet.get("pl_rade"), 2)}</span>'
            f'<span>{display_number(mass_value, 2)}{mass_note}</span>'
            f'<span>{display_number(planet.get("pl_orbsmax"), 3)}</span>'
            f'<span>{display_number(planet.get("pl_orbeccen"), 2)}</span></div>'
        )
    rows.append("</section>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def planet_orbit_figure(planets):
    """Draw face-on Kepler ellipses with the host star at one focus."""
    orbits = []
    for planet in planets:
        try:
            axis = float(planet.get("pl_orbsmax"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(axis) or axis <= 0:
            continue
        try:
            eccentricity = float(planet.get("pl_orbeccen"))
        except (TypeError, ValueError):
            eccentricity = None
        if eccentricity is None or not 0 <= eccentricity < 1:
            eccentricity = None
        orbits.append((axis, eccentricity, planet))
    if not orbits:
        return None
    orbits.sort(key=lambda item: item[0])

    # A second, physically scaled panel keeps close orbits visible when a
    # distant candidate would otherwise compress them into the central dot.
    split = len(orbits) > 1 and orbits[-1][0] / orbits[0][0] > 15
    if split:
        gaps = [orbits[i + 1][0] / orbits[i][0] for i in range(len(orbits) - 1)]
        inner = orbits[:gaps.index(max(gaps)) + 1]
        panels = [("Inner orbits", inner), ("Full system", orbits)]
    else:
        panels = [("Orbital architecture", orbits)]

    figure, axes = plt.subplots(1, len(panels), figsize=(8.8, 3.65), dpi=120)
    axes = np.atleast_1d(axes)
    figure.patch.set_facecolor("#18181b")
    masses = [orbital_mass(planet)[0] for _, _, planet in orbits]
    largest_mass = max((mass for mass in masses if mass is not None), default=1.0)
    def planet_radius(planet):
        try:
            radius = float(planet.get("pl_rade"))
        except (TypeError, ValueError):
            return None
        return radius if math.isfinite(radius) and radius > 0 else None

    radii = [planet_radius(planet) for _, _, planet in orbits
             if orbital_mass(planet)[0] is None]
    largest_radius = max((radius for radius in radii if radius is not None), default=1.0)
    angles = np.linspace(0, 2 * np.pi, 361)
    for axis_plot, (title, orbit_set) in zip(axes, panels):
        axis_plot.set_facecolor("#18181b")
        for index, (axis, eccentricity, planet) in enumerate(orbit_set):
            e = eccentricity if eccentricity is not None else 0.0
            x = axis * (np.cos(angles) - e)
            y = axis * np.sqrt(1 - e * e) * np.sin(angles)
            orbit_color = "#86cfa5" if planet.get("status") == "Confirmed" else "#f4cf75"
            axis_plot.plot(
                x, y, color=orbit_color, linewidth=.7,
                linestyle="-" if eccentricity is not None else "--", alpha=.9,
            )
            label_angle = .65 + index * 1.6
            label_x = axis * (np.cos(label_angle) - e)
            label_y = axis * np.sqrt(1 - e * e) * np.sin(label_angle)
            mass, mass_kind = orbital_mass(planet)
            if mass is not None:
                # Matplotlib's scatter size is an area; keep area proportional
                # to mass with one shared scale across both orbital panels.
                marker_area = 150 * mass / largest_mass
            else:
                radius = planet_radius(planet)
                # Radius scales the marker radius; mass and radius use separate scales.
                marker_area = 150 * (radius / largest_radius) ** 2 if radius else 36
            axis_plot.scatter(
                [label_x], [label_y], s=marker_area, marker="o",
                facecolors="#ef4444" if mass_kind == "mass" else "#18181b",
                edgecolors="#ef4444", linewidths=1.25, zorder=4,
            )
            label = str(planet.get("pl_name") or "").split()[-1]
            axis_plot.annotate(
                label, (label_x, label_y),
                xytext=(14 * np.cos(label_angle), 14 * np.sin(label_angle)),
                textcoords="offset points", ha="center", va="center",
                color="#ef4444", fontsize=8.5, fontweight="medium",
                bbox=dict(facecolor="#18181b", edgecolor="none", pad=1.8), zorder=6,
            )
        axis_plot.scatter([0], [0], s=130, marker="*", c="#f4cf75", zorder=5)
        limit = max(axis * (1 + (e or 0)) for axis, e, _ in orbit_set) * 1.19
        axis_plot.set(xlim=(-limit, limit), ylim=(-limit, limit), title=title)
        axis_plot.set_aspect("equal")
        axis_plot.tick_params(colors="#a1a1aa", labelsize=7)
        for spine in axis_plot.spines.values():
            spine.set_color("#39393f")
        axis_plot.title.set_color("#dedee3")
        axis_plot.title.set_fontsize(10)
        axis_plot.set_xlabel("AU", color="#a1a1aa", fontsize=8)
    figure.tight_layout(pad=1.2)
    return figure


def planet_population_figure(reference_planets, target_planets):
    """Place a loaded system in the local mass--period planet population."""
    background_periods = []
    background_masses = []
    for planet in reference_planets:
        mass, _ = orbital_mass(planet)
        try:
            period = float(planet.get("pl_orbper"))
        except (TypeError, ValueError):
            continue
        if (
            mass is None
            or not math.isfinite(period)
            or period <= 0
            or not math.isfinite(mass)
            or mass <= 0
        ):
            continue
        background_periods.append(period)
        background_masses.append(mass)

    if not background_periods:
        return None, [], 0

    target_points = []
    for planet in target_planets:
        mass, mass_kind = orbital_mass(planet)
        try:
            period = float(planet.get("pl_orbper"))
        except (TypeError, ValueError):
            continue
        if (
            mass is None
            or not math.isfinite(period)
            or period <= 0
            or not math.isfinite(mass)
            or mass <= 0
        ):
            continue
        target_points.append((period, mass, mass_kind, planet))

    figure, axis = plt.subplots(figsize=(6.1, 4.65), dpi=120)
    figure.patch.set_facecolor("#151518")
    axis.set_facecolor("#151518")
    axis.scatter(
        background_periods, background_masses,
        s=13, color="#d4d4d8", alpha=.22, linewidths=0,
        label="Confirmed planets",
    )

    plotted_target = False
    for period, mass, mass_kind, planet in target_points:
        label = "Loaded planet" if not plotted_target else None
        plotted_target = True
        axis.scatter(
            period, mass, s=92,
            color="#ef4444" if mass_kind == "mass" else "#151518",
            edgecolor="#ef4444", linewidth=1.65, zorder=4, label=label,
        )
        suffix = str(planet.get("pl_name") or "").split()[-1]
        axis.annotate(
            suffix, (period, mass), xytext=(6, 5), textcoords="offset points",
            color="#eeeeef", fontsize=9, zorder=5,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Orbital period (days)", fontsize=13, color="#d4d4d8")
    axis.set_ylabel("Planet mass or M sin i (M⊕)", fontsize=13, color="#d4d4d8")
    axis.tick_params(colors="#a1a1aa", labelsize=11)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("#3f3f46")
    axis.legend(
        facecolor="#19191e", edgecolor="#3f3f46", labelcolor="#d4d4d8",
        frameon=True, fontsize=9, loc="best",
    )
    figure.tight_layout()
    return figure, target_points, len(background_periods)


def multiplicity_table(values, criteria):
    """Show each Gaia parameter and its criterion in a compact table."""
    rows = [
        '<section class="runa-indicator-table">'
        '<div class="runa-indicator-row runa-indicator-head">'
        '<span>Criterion</span><span>Parameter</span><span>Value</span>'
        '<span>Observed</span></div>'
    ]
    indicators = {
        1: (("RUWE", "> 2", "ruwe"),),
        2: (
            ("IPD amplitude", "> 0.1", "ipd_gof_harmonic_amplitude"),
            ("RUWE", "> 1.4", "ruwe"),
        ),
        3: (("IPD multi-peak", "> 30", "ipd_frac_multi_peak"),),
        4: (
            ("RV p-value", "< 0.01", "rv_chisq_pvalue"),
            ("RV GOF", "> 4", "rv_renormalised_gof"),
            ("RV transits", "≥ 10", "rv_nb_transits"),
        ),
        5: (("RV error", "≥ 10 km s⁻¹", "radial_velocity_error"),),
        6: (("Duplicated source", "= 1", "duplicated_source"),),
        7: (("NSS flag", "set", "non_single_star"),),
    }
    for number, parameter_rows in indicators.items():
        status = "Unavailable" if criteria[number] is None else (
            "Met" if criteria[number] else "Not met"
        )
        row_class = {
            "Met": "runa-indicator-met",
            "Not met": "runa-indicator-not-met",
            "Unavailable": "runa-indicator-unavailable",
        }[status]
        for row_number, (parameter, condition, field) in enumerate(parameter_rows):
            value = values[field]
            if value is None:
                shown = "—"
            elif field == "duplicated_source":
                shown = "yes" if value else "no"
            elif isinstance(value, float):
                shown = f"{value:.3g}"
            else:
                shown = str(value)
            criterion_cell = str(number) if row_number == 0 else ""
            rows.append(
                f'<div class="runa-indicator-row {row_class}">'
                f'<span class="runa-indicator-number">{criterion_cell}</span>'
                f'<span class="runa-indicator-label">{escape(parameter)}</span>'
                f'<span class="runa-indicator-condition">{escape(condition)}</span>'
                f'<span class="runa-indicator-observed">{escape(shown)}</span>'
                '</div>'
            )
    rows.append("</section>")
    st.markdown("".join(rows), unsafe_allow_html=True)


MULTIPLICITY_REPORT_FILE = Path(__file__).resolve().parent / "data" / "models" / "multiplicity_evaluation.json"


@st.cache_resource(show_spinner=False)
def load_multiplicity_deployment_model():
    """Fit the selected classifier on all labelled rows for interactive scoring."""
    if not MULTIPLICITY_REPORT_FILE.exists():
        raise FileNotFoundError("The offline multiplicity evaluation report is missing.")
    report = json.loads(MULTIPLICITY_REPORT_FILE.read_text(encoding="utf-8"))
    features, target, _, _, _, _ = prepare_training_data()
    model = model_definitions()[report["selected_model"]]
    model.fit(features, target)
    return model, features.columns.tolist(), report.get("decision_threshold", 0.5)


def multiplicity_ml_result(values, scoring=None):
    """Score Gaia DR3 diagnostics and show the classifier's thresholded decision."""
    astrometric_fields = (
        "ruwe", "ipd_gof_harmonic_amplitude", "ipd_frac_multi_peak",
        "duplicated_source",
    )
    radial_velocity_fields = (
        "rv_chisq_pvalue", "rv_renormalised_gof", "rv_nb_transits",
        "radial_velocity_error",
    )
    astrometric_available = sum(values[field] is not None for field in astrometric_fields)
    radial_velocity_available = sum(values[field] is not None for field in radial_velocity_fields)
    total_available = astrometric_available + radial_velocity_available

    if total_available == 0:
        st.warning("Gaia DR3 provides none of the model inputs for this source; no score was produced.")
        return

    if scoring is None:
        st.caption("Scoring Gaia DR3 diagnostics…")
        return
    score = scoring["score"]
    threshold = scoring["threshold"]

    is_unresolved = score >= threshold
    decision_col, score_col = st.columns(2)
    decision_col.markdown(
        '<div class="runa-model-decision">'
        f'<div class="runa-model-decision-label">Model decision · cut {threshold:.2f}</div>'
        f'<div class="runa-model-decision-value">{"Candidate" if is_unresolved else "Single"}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="runa-score-summary">'
        f'<strong>Unresolved-binary score <em>{score:.2f}</em></strong>'
        f' <span>· based on <em>{total_available}/8</em> available Gaia diagnostics.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="runa-score-track" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="1" aria-valuenow="{score:.3f}">'
        f'<div class="runa-score-fill" style="width:{100 * score:.2f}%"></div>'
        '<span class="runa-score-threshold" aria-label="0.50 threshold"></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This model targets close unresolved components. A ‘Single’ label does not rule out a companion."
    )


def radius_label(radius_arcsec):
    """Show a nearby-source radius in the unit used by the selector."""
    if radius_arcsec < 60:
        return f"{radius_arcsec} arcsec"
    return f"{radius_arcsec / 60:g} arcmin"


def catalogue_rows(star):
    """Return a short, useful selection of SIMBAD identifiers."""
    identifiers = list(star.get("identifiers", []))
    if star.get("gaia_id"):
        identifiers.append("Gaia DR3 " + str(star["gaia_id"]))
    if star.get("twomass_id"):
        identifiers.append(star["twomass_id"])

    catalogues = (
        ("Gaia DR3 ", "Gaia DR3", "optical catalogue"),
        ("2MASS J", "2MASS", "near-IR"),
        ("WISEA J", "WISE", "mid-IR"),
        ("TIC ", "TIC", "TESS target list"),
        ("HIP ", "Hipparcos", "stellar catalogue"),
        ("HD ", "Henry Draper", "spectral catalogue"),
        ("GJ ", "Gliese–Jahreiß", "nearby-star list"),
        ("Gl ", "Gliese", "nearby-star list"),
        ("LHS ", "LHS", "high-proper-motion list"),
    )
    rows = []
    for prefix, label, tag in catalogues:
        for identifier in identifiers:
            if identifier.startswith(prefix):
                value = identifier[len(prefix):].strip()
                rows.append((label, value, tag))
                break
    return rows


def source_name(bibcode, default="SIMBAD"):
    """Give familiar names to the Gaia references used most often by SIMBAD."""
    names = {
        "2020yCat.1350....0G": "Gaia EDR3",
        "2018A&A...616A...1G": "Gaia DR2",
        "2018A&A...616A...7S": "Gaia DR2",
        "2016A&A...595A...2G": "Gaia DR1",
    }
    if bibcode:
        return names.get(str(bibcode), str(bibcode))
    return default


def sexagesimal(value, right_ascension=False):
    """Format decimal degrees as hours or signed degrees."""
    factor = 240 if right_ascension else 3600
    total = round(abs(float(value)) * factor * 100)
    first, remainder = divmod(total, 360000)
    minute, second = divmod(remainder, 6000)
    if right_ascension:
        first %= 24
        sign = ""
    else:
        sign = "+" if value >= 0 else "−"
    return f"{sign}{first:02d}:{minute:02d}:{second / 100:05.2f}"



def photometry_summary_metric(label, value, error=None):
    """Show a photometric value with secondary uncertainty and unit styling."""
    label_html = (
        escape(str(label))
        .replace("M_G", "M<sub>G</sub>")
        .replace("M_J", "M<sub>J</sub>")
    )
    if value is None or np.ma.is_masked(value):
        value_html = '<span class="runa-summary-value">—</span>'
        error_html = ""
        unit_html = ""
    else:
        value_html = f'<span class="runa-summary-value">{float(value):.3f}</span>'
        error_html = (
            f'<span class="runa-summary-error">({float(error):.3f})</span>'
            if error is not None and not np.ma.is_masked(error)
            else ""
        )
        unit_html = '<span class="runa-summary-unit">mag</span>'
    st.markdown(
        '<div class="runa-summary-metric">'
        f'<div class="runa-summary-label">{label_html}</div>'
        f'<div>{value_html}{error_html}{unit_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

def object_panel_title(star, heading="Object"):
    """Show the selected object without making it compete with section titles."""
    label = star["main_id"] if star is not None else "Input"
    st.markdown(
        f'<div class="runa-panel-object"><span>{escape(heading)}</span>{escape(str(label))}</div>',
        unsafe_allow_html=True,
    )


def object_type_name(star):
    """Use SIMBAD's long object type, with a small fallback for older sessions."""
    if star.get("otype_name"):
        return str(star["otype_name"])
    if star.get("otype_txt") and star["otype_txt"] != star.get("otype"):
        return str(star["otype_txt"])
    fallback = {
        "PM*": "High proper-motion Star",
        "*": "Star",
        "**": "Double or multiple star",
        "V*": "Variable Star",
        "WD*": "White Dwarf",
    }
    return fallback.get(str(star.get("otype")), str(star.get("otype") or "Unknown"))


@st.cache_data(ttl=86400, show_spinner=False)
def nearby_gaia_sources_v3(ra, dec, radius_arcsec):
    """Cache a Gaia cone query with VizieR fallback for one day."""
    return query_gaia_neighbours(ra, dec, radius_arcsec)


class GaiaPhotometryUnavailable(RuntimeError):
    """Raised when neither Gaia ESA nor VizieR returns a DR3 source."""


@st.cache_data(ttl=86400, show_spinner=False)
def gaia_source_photometry(gaia_id):
    """Cache Gaia photometry and observation counts with VizieR fallback."""
    photometry = lookup_gaia_photometry(gaia_id)
    if gaia_id and not photometry:
        raise GaiaPhotometryUnavailable(
            "Neither Gaia ESA nor the VizieR mirrors returned this DR3 source."
        )
    return photometry


@st.cache_resource(show_spinner=False)
def analysis_executor():
    """Keep independent object analyses running across page changes."""
    return ThreadPoolExecutor(max_workers=8, thread_name_prefix="runa-analysis")


def catalogue_photometry_object_key(star):
    """Return a stable key for the object whose catalogue data is requested."""
    if not star:
        return None
    return str(star.get("gaia_id") or star.get("main_id") or star.get("query_name"))


ANALYSIS_SECTIONS = {
    "Photometry": ("photometry_catalogues", "photometry_cmd"),
    "Parameters": ("parameters_calculation", "parameters_literature", "photometry_catalogues"),
    "Kinematics": ("kinematics_population",),
    "Multiplicity": (
        "multiplicity_criteria", "multiplicity_neighbours", "multiplicity_epoch",
        "multiplicity_wds", "multiplicity_archives", "multiplicity_rvs",
        "multiplicity_model",
    ),
    "Planets": ("planets_nasa", "planets_oec", "planets_population", "planets_tess"),
}


def _score_multiplicity(criteria_future):
    values, _ = criteria_future.result()
    model, feature_names, threshold = load_multiplicity_deployment_model()
    features = gaia_feature_row(values, feature_names)
    return {"score": float(model.predict_proba(features)[0, 1]), "threshold": threshold}


def start_object_analysis(star):
    """Submit every independent query once for the current object."""
    old_jobs = st.session_state.get("analysis_jobs", {})
    for future in old_jobs.values():
        future.cancel()
    source = {
        key: deepcopy(star.get(key)) for key in (
            "main_id", "query_name", "gaia_id", "identifiers", "ra", "dec",
            "sp_type", "gaia_photometry", "twomass_photometry", "plx_value", "plx_err",
            "twomass_id", "johnson_photometry",
        )
    }
    executor = analysis_executor()
    jobs = {
        "photometry_catalogues": executor.submit(fetch_catalogue_photometry, source),
        "photometry_cmd": executor.submit(load_gaia_reference_sample),
        "parameters_calculation": executor.submit(calculate_parameters, source),
        "parameters_literature": executor.submit(lookup_simbad_measurements, source["main_id"]),
        "kinematics_population": executor.submit(run_banyan, source),
        "planets_nasa": executor.submit(
            lookup_planets, source["query_name"], source["main_id"],
            tuple(source["identifiers"] or ()), source["gaia_id"],
        ),
        "planets_oec": executor.submit(
            lookup_open_exoplanet_catalogue, source["query_name"],
            source["main_id"], tuple(source["identifiers"] or ()),
        ),
        "planets_population": executor.submit(load_planet_population),
    }
    if tess_identifier(source):
        jobs["planets_tess"] = executor.submit(fetch_tess_light_curve, source)
    gaia_id = source.get("gaia_id")
    if gaia_id:
        gaia = source.get("gaia_photometry") or {}
        ra = gaia.get("ra") or source.get("ra")
        dec = gaia.get("dec") or source.get("dec")
        jobs["multiplicity_criteria"] = executor.submit(check_multiplicity, gaia_id)
        jobs["multiplicity_neighbours"] = executor.submit(query_gaia_neighbours, ra, dec, 180)
        jobs["multiplicity_epoch"] = executor.submit(query_gaia_epoch_photometry, gaia_id)
        jobs["multiplicity_rvs"] = executor.submit(query_rvs_reference)
        jobs["multiplicity_model"] = executor.submit(
            _score_multiplicity, jobs["multiplicity_criteria"]
        )
    jobs["multiplicity_wds"] = executor.submit(
        closest_primary_component, source.get("ra"), source.get("dec"),
        tuple(source.get("identifiers") or ()),
    )
    gaia = source.get("gaia_photometry") or {}
    jobs["multiplicity_archives"] = executor.submit(
        search_spectroscopy_archives,
        gaia.get("ra") or source.get("ra"), gaia.get("dec") or source.get("dec"),
        gaia.get("pmra"), gaia.get("pmdec"), gaia.get("ref_epoch"),
    )
    st.session_state["analysis_jobs"] = jobs
    st.session_state["analysis_results"] = {}
    st.session_state["analysis_object"] = catalogue_photometry_object_key(star)


def analysis_result(task):
    """Return (value, state) without waiting for a network query."""
    result = st.session_state.get("analysis_results", {}).get(task)
    if result is None:
        return None, "pending"
    if "error" in result:
        return None, "error"
    return result["value"], "complete"


def analysis_section_status(section):
    if st.session_state.get("star") is None:
        return "idle"
    if section == "Identification":
        return "complete"
    tasks = ANALYSIS_SECTIONS.get(section, ())
    results = st.session_state.get("analysis_results", {})
    jobs = st.session_state.get("analysis_jobs", {})
    if any(task in jobs and task not in results for task in tasks):
        return "pending"
    if any(task in results and "error" in results[task] for task in tasks):
        return "error"
    return "complete" if tasks else "idle"


@st.fragment(run_every="1s")
def analysis_progress():
    """Collect completed jobs even while another section is open."""
    jobs = st.session_state.get("analysis_jobs", {})
    results = st.session_state.get("analysis_results", {})
    changed = False
    for task, future in jobs.items():
        if task in results or not future.done():
            continue
        try:
            results[task] = {"value": future.result()}
        except Exception as exc:
            LOGGER.exception("Background analysis %s failed", task)
            results[task] = {"error": str(exc)}
        changed = True
    if changed:
        st.session_state["analysis_results"] = results
        if "photometry_catalogues" in results:
            record = results["photometry_catalogues"]
            st.session_state["catalogue_photometry"] = record.get("value")
        st.rerun()


st.set_page_config(
    page_title="RUNA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 5rem;}
    .runa-brand {font-size: 1rem; color: #a1a1aa; margin-bottom: 2.8rem;}
    .runa-brand strong {color: #fafafa; font-weight: 600;}
    .runa-brand span {color: #6ee7b7;}
    div[data-testid="stSpinner"],
    div[data-testid="stSpinner"] * {color:#fafafa !important;}
    div[data-testid="stSpinner"] svg,
    div[data-testid="stSpinner"] svg * {
        color:#64dfb4 !important; fill:#64dfb4 !important;
    }
    .runa-tools {border-bottom: 1px solid #27272a; margin: 1.1rem 0 2rem; padding-bottom: .75rem;}
    .runa-tools span {font-size: 1.02rem; color: #92929d;}
    .stMarkdown a, .stCaption a {color: inherit !important;
                                 text-decoration-color: #60606a;
                                 text-underline-offset: .16em;}
    .stMarkdown a:hover, .stCaption a:hover {color: #f4f4f5 !important;}
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171a1c 0%, #121517 100%);
        border-right: 1px solid #2a2d30;
    }
    section[data-testid="stSidebar"][data-collapsed="false"] {
        width: 15.25rem !important;
        min-width: 15.25rem !important;
    }
    section[data-testid="stSidebar"][data-collapsed="true"] {
        width: 0 !important;
        min-width: 0 !important;
        border-right: 0;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] p {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }
    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stExpandSidebarButton"] button {
        width: 1.75rem; height: 1.75rem; min-height: 1.75rem;
        border-radius: 6px; background: transparent !important;
        color: #777780 !important; box-shadow: none !important;
    }
    div[data-testid="stSidebarCollapseButton"] button:hover,
    div[data-testid="stExpandSidebarButton"] button:hover {
        background: #242428 !important; color: #e4e4e7 !important;
    }
    div[data-testid="stSidebarCollapseButton"] svg,
    div[data-testid="stExpandSidebarButton"] svg {
        width: 1rem; height: 1rem;
    }
    div[data-testid="stSidebarHeader"] {
        position: relative; height: 0 !important; min-height: 0 !important;
        padding: 0 !important; margin: 0 !important;
    }
    div[data-testid="stLogoSpacer"] {display: none !important;}
    div[data-testid="stSidebarCollapseButton"] {
        position: absolute; top: .78rem; right: .72rem; z-index: 10;
    }
    section[data-testid="stSidebar"] > div {padding-top: .45rem;}
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
        padding: .4rem .7rem 1rem;
    }
    .runa-linear-header {
        display: flex; align-items: center; justify-content: space-between;
        min-height: 2.65rem; margin: .05rem .48rem 1rem;
    }
    .runa-linear-workspace {
        display: flex; align-items: center; gap: .62rem;
        color: #eceeef; font-size: 1rem; font-weight: 620;
        letter-spacing: -.01em;
    }
    .runa-linear-mark {
        display: inline-block; width: 1.15rem; height: 1.15rem; border-radius: 50%;
        flex: 0 0 1.15rem;
        background: repeating-linear-gradient(45deg, #e7e9ea 0 2px,
                    #7d8287 2px 3px); opacity: .95;
    }
    .runa-linear-workspace-chevron {
        color: #74797f; font-size: .9rem; font-weight: 500;
        transform: translateY(-.05rem);
    }
    .runa-linear-section {
        color: #7d8288; font-size: .82rem; font-weight: 540;
        letter-spacing: -.005em; line-height: 1.3;
        margin: 1.22rem .48rem .4rem;
    }
    .runa-linear-section span {font-size: .62rem; margin-left: .2rem;}
    .runa-note {color: #a1a1aa; font-size: .9rem;}
    .runa-filter-reference {color: #a1a1aa; font-size: .875rem;
                            line-height: 1.4; margin: 0 0 .65rem;}
    .runa-filter-reference a,
    .runa-filter-reference a:hover {text-decoration: none !important;}
    .runa-data-source-reference {display: block;}
    .runa-warning-toggle {position:absolute; opacity:0; pointer-events:none;}
    .runa-warning-badge {display:inline; margin-left:.25rem; padding:0;
                         border:0; color:#f4cf75; background:transparent;
                         font-size:.78rem; font-weight:600; cursor:pointer;}
    .runa-warning-tooltip {
        display: block; box-sizing: border-box; margin-top: .45rem; padding: .7rem .85rem;
        border: 1px solid #4b4226; border-radius: 8px; background: #111114;
        color: #f4cf75; font-size: .88rem; font-weight: 400; line-height: 1.5;
        white-space: normal; text-align: left; letter-spacing: normal;
    }
    .runa-warning-toggle:not(:checked) ~ .runa-warning-tooltip {display: none;}
    .runa-warning-badge:focus-visible {outline: 2px solid #f4cf75; outline-offset: 2px;}
    .st-key-photometry_estimated_distance .runa-data-section,
    .st-key-photometry_estimated_distance .runa-parameter-section {overflow: visible;}
    .runa-learning-note {display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;
                         margin:.2rem 0 .8rem;color:#92929d;font-size:.84rem;}
    .runa-learning-badge {display:inline-flex;align-items:center;padding:.22rem .55rem;
                          border:1px solid #426b5b;border-radius:999px;background:#18231f;
                          color:#64dfb4;font-size:.72rem;font-weight:650;
                          letter-spacing:.035em;text-transform:uppercase;}
    .runa-caution-note {display:flex;align-items:flex-start;gap:.65rem;margin:.9rem 0;
                        padding:.72rem .82rem;border:1px solid #5f4820;border-radius:9px;
                        background:rgba(245,158,11,.065);color:#d5c6a8;font-size:.82rem;
                        line-height:1.42;}
    .runa-caution-icon {display:inline-flex;align-items:center;justify-content:center;
                        flex:0 0 1.25rem;width:1.25rem;height:1.25rem;border-radius:50%;
                        background:#6b4b16;color:#ffd38a;font-size:.76rem;font-weight:750;}
    .runa-spectral-result {display:flex;align-items:flex-end;justify-content:space-between;
                           gap:1.2rem;margin:.45rem 0 .85rem;}
    .runa-spectral-estimate {margin:0;}
    .runa-spectral-estimate span {display:block;color:#92929d;font-size:.82rem;
                                  margin-bottom:.18rem;}
    .runa-spectral-estimate strong {display:block;color:#f4cf75;font-size:2rem;
                                    font-weight:620;line-height:1.1;}
    .runa-distance-flow {display:grid;grid-template-columns:minmax(0,1fr) 24px
                         minmax(0,1.25fr) 24px minmax(0,1fr);align-items:stretch;
                         gap:.45rem;margin:.75rem 0 .3rem;}
    .runa-distance-step {min-width:0;border:1px solid #333338;border-radius:10px;
                         padding:.85rem .9rem;background:#18181b;}
    .runa-distance-step small {display:block;color:#92929d;font-size:.72rem;
                               letter-spacing:.06em;text-transform:uppercase;
                               margin-bottom:.4rem;}
    .runa-distance-step strong {display:block;color:#e4e4e7;font-size:1.08rem;
                                font-weight:570;line-height:1.25;}
    .runa-distance-step > span:not(.runa-warning-wrap) {display:block;color:#92929d;
                                font-size:.8rem;line-height:1.35;margin-top:.3rem;}
    .runa-distance-error {display:inline !important;color:#92929d !important;
                          font-size:.82em !important;font-weight:430;}
    .runa-distance-calibration,.runa-distance-result {border-color:#595443;}
    .runa-distance-calibration strong {color:#f4cf75;}
    .runa-distance-result strong {color:#f4cf75;font-size:1.72rem;font-weight:620;}
    .runa-distance-result strong span {display:inline;color:#a1a1aa;font-size:.8rem;
                                       font-weight:430;}
    .runa-distance-arrow {align-self:center;justify-self:center;color:#72727c;}
    .runa-volume {border: 1px solid #303037; border-radius: 12px; background: #19191e;
                  padding: .8rem 1rem; margin: .65rem 0 .9rem;}
    .runa-volume-title {color: #f4f4f5; font-size: 1rem; font-weight: 650;
                        margin-bottom: .25rem;}
    .runa-volume-text {color: #a1a1aa; font-size: .86rem; line-height: 1.45;}
    .runa-volume-text a {color: #6ee7b7; text-decoration: none;}
    .runa-volume-multiple {border-color: #6a3638; background: #2a191a;}
    .runa-volume-multiple .runa-volume-title {color: #f0aaa2;}
    .runa-gaia-id {color: #c7c7cf; font-size: 1.1rem; font-weight: 560;
                   margin: .3rem 0 .9rem;}
    .runa-gaia-id strong {color: #f4f4f5; font-weight: 680;}
    .runa-spectral-colour {color:#92929d;font-size:.8rem;margin:0 0 .12rem;
                           text-align:right;white-space:nowrap;}
    .runa-spectral-colour strong {display:block;color:#d4d4d8;font-size:1.1rem;
                                  font-weight:560;margin-top:.16rem;}
    .runa-large-metric {padding: .15rem 0 .8rem;}
    .runa-large-metric-label {color: #a1a1aa; font-size: .82rem; margin-bottom: .2rem;}
    .runa-large-metric-label sub {font-size: .72em; vertical-align: sub;}
    .runa-large-metric-value {color: #f4f4f5; font-size: 1.7rem; font-weight: 620;
                              line-height: 1.18;}
    .runa-large-metric-error {color: #92929d; font-size: .82rem; margin-top: .22rem;}
    .runa-summary-metric {padding: .15rem 0 .75rem;}
    .runa-summary-label {color: #a1a1aa; font-size: .82rem; margin-bottom: .18rem;}
    .runa-summary-label sub {font-size: .72em; vertical-align: sub;}
    .runa-summary-value {color: #f4f4f5; font-size: 1.65rem; font-weight: 620;
                         line-height: 1.15; font-variant-numeric: tabular-nums;}
    .runa-summary-error {color: #92929d; font-size: .82rem; font-weight: 430;
                         margin-left: .3rem; font-variant-numeric: tabular-nums;}
    .runa-summary-unit {color: #92929d; font-size: .78rem; font-weight: 430;
                        margin-left: .32rem;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        border-color: #303036;
        background: #151518;
    }
    .runa-status {display: inline-flex; align-items: center; gap: .5rem; margin: .4rem 0 .2rem;}
    .runa-status-icon {display: inline-flex; align-items: center; justify-content: center;
                       width: 1.25rem; height: 1.25rem; border: 2px solid currentColor;
                       border-radius: 50%; font-size: .75rem; font-weight: 700; line-height: 1;}
    .runa-met {color: #47d990;}
    .runa-not-met {color: #d8a099;}
    .runa-unavailable {color: #9999a3;}
    .runa-status-label {color: #e4e4e7; font-weight: 600;}
    .runa-status-text {color: #a1a1aa; font-size: .88rem;}
    .runa-indicator-table {overflow: hidden; border: 1px solid #2d2d33;
                           border-radius: 12px; margin: .3rem 0 .9rem;
                           background: #121215;}
    .runa-indicator-row {display: grid; grid-template-columns: 4.3rem minmax(10rem, 1.25fr)
                         minmax(4.4rem, .6fr) minmax(5.5rem, .8fr); gap: .65rem;
                         align-items: center; padding: .72rem .8rem;
                         color: #ededf0; font-size: .88rem;}
    .runa-indicator-row + .runa-indicator-row {border-top: 1px solid #28282e;}
    .runa-indicator-head {color: #8b8b95; font-size: .74rem; font-weight: 650;
                          letter-spacing: .08em; text-transform: uppercase;}
    .runa-indicator-number {color: #777780; font-size: .86rem; font-variant-numeric: tabular-nums;}
    .runa-indicator-label {color: #d4d4d8; font-size: .88rem; line-height: 1.25;}
    .runa-indicator-condition {color: #a1a1aa; font-size: .86rem; font-variant-numeric: tabular-nums;
                               white-space: nowrap;}
    .runa-indicator-observed {color: #a1a1aa; font-size: .86rem; font-variant-numeric: tabular-nums;
                              overflow-wrap: anywhere;}
    .runa-indicator-met {border-left: 3px solid #e87979;}
    .runa-indicator-met .runa-indicator-label,
    .runa-indicator-met .runa-indicator-condition,
    .runa-indicator-met .runa-indicator-observed {color: #ef9999;}
    .runa-indicator-not-met {border-left: 3px solid #47d990;}
    .runa-indicator-not-met .runa-indicator-label,
    .runa-indicator-not-met .runa-indicator-condition,
    .runa-indicator-not-met .runa-indicator-observed {color: #73dfae;}
    .runa-indicator-unavailable {border-left: 3px solid #777780;}
    .runa-object-head {margin: 0;}
    .runa-panel-object {color: #d4d4d8; font-size: 1.08rem; font-weight: 590;
                        margin: .05rem 0 1rem; letter-spacing: -.01em;}
    .runa-panel-heading {color: #81818b; font-size: .82rem; font-weight: 650;
                         letter-spacing: .08em; text-transform: uppercase;
                         margin: .05rem 0 .65rem;}
    .runa-panel-object span {color: #81818b; font-size: .67rem; font-weight: 650;
                             letter-spacing: .08em; text-transform: uppercase;
                             margin-right: .5rem;}
    .runa-identification-head {display: grid; grid-template-columns: repeat(3, 1fr);
                              gap: .7rem; align-items: stretch; margin-bottom: 1rem;}
    .runa-identification-head .runa-object-head {border: 1px solid #303037;
                              border-radius: 12px; background: #19191e; padding: .85rem 1rem;}
    .runa-classification {display: contents;}
    .runa-class-card {border: 1px solid #303037; border-radius: 12px;
                      background: #19191e; padding: .85rem 1rem;}
    .runa-class-label {color: #81818b; font-size: .68rem; letter-spacing: .08em;
                       text-transform: uppercase; margin-bottom: .35rem;}
    .runa-class-value {color: #f5f5f6; font-size: 1.35rem; font-weight: 620;
                       line-height: 1.25;}
    .runa-class-meta {display: flex; justify-content: space-between; gap: .7rem;
                      color: #85858f; font-size: .72rem; margin-top: .35rem;}
    .runa-class-source {justify-content: flex-end;}
    .runa-chip {display: inline-flex; align-items: center; border: 1px solid #3a3a42;
                border-radius: 999px; color: #c7c7cf; background: #1d1d22;
                font-size: .72rem; line-height: 1.4; padding: .12rem .48rem;}
    .runa-data-section {overflow: hidden; border: 1px solid #2d2d33; border-radius: 12px;
                        margin: 0 0 .85rem; background: #121215;}
    .runa-section-title {color: #8b8b95; font-size: .7rem; font-weight: 650;
                         letter-spacing: .09em; text-transform: uppercase;
                         padding: .75rem 1rem .55rem;}
    .runa-data-row {display: grid; grid-template-columns: minmax(9rem, .9fr)
                    minmax(0, 1.25fr) minmax(3.4rem, .45fr)
                    minmax(6.5rem, .9fr); gap: .8rem;
                    align-items: center; padding: .76rem 1rem;}
    .runa-data-row + .runa-data-row {border-top: 1px solid #28282e;}
    .runa-data-label {color: #a0a0aa; font-size: .94rem; padding-left: .2rem;}
    .runa-data-value {color: #ededf0; font-size: .9rem; font-weight: 520;
                      overflow-wrap: anywhere;}
    .runa-error {color: #92929d; font-size: .9rem; font-weight: 430; margin-left: .45rem;}
    .runa-data-unit {color: #a4a4ad; font-size: .82rem; font-weight: 430;
                     white-space: nowrap;}
    .runa-data-source {color: #777782; font-size: .72rem; text-align: right;
                       white-space: nowrap; padding-top: .08rem; justify-self: end;}
    .runa-identification-data .runa-section-title {color:#81818b;font-size:.82rem;
                         font-weight:650;letter-spacing:.08em;text-transform:uppercase;
                         padding-top:.75rem;padding-bottom:.55rem;}
    .runa-identification-column-marker {display:none;}
    .runa-identification-data .runa-data-row {
        grid-template-columns: minmax(7.4rem, .75fr) minmax(0, 1.5fr)
                               minmax(3rem, .38fr) minmax(5.3rem, .78fr);
        gap: .6rem;
    }
    .runa-identification-data .runa-data-row-no-unit {
        grid-template-columns: minmax(7.4rem, .75fr) minmax(0, 1.65fr)
                               minmax(5.3rem, .78fr);
    }
    .runa-identification-data .runa-data-row-no-unit .runa-data-unit {display:none;}
    .runa-identification-data .runa-data-label {font-size: 1.04rem;}
    .runa-identification-data .runa-data-value {font-size: 1rem;}
    .runa-identification-data .runa-error {font-size: 1rem;}
    .runa-identification-data .runa-data-unit {font-size: .9rem;}
    .runa-identification-data .runa-data-source {font-size: .8rem;}
    .runa-parameter-section {overflow: hidden; border: 1px solid #2d2d33;
                             border-radius: 12px; margin: 0 0 .7rem;
                             background: #121215;}
    .runa-parameter-published {border-color: #5d5968; border-radius: 10px;
                               background: #18181b;}
    .runa-parameter-estimated {border-color: #595443; border-radius: 10px;
                               background: #18181b;}
    .runa-parameter-kind {color: #8b8b95; font-size: .68rem; font-weight: 680;
                          letter-spacing: .09em; text-transform: uppercase;
                          padding: .7rem .85rem .35rem;}
    .runa-parameter-published .runa-parameter-kind {color: #c9c4e8;}
    .runa-parameter-estimated .runa-parameter-kind {color: #f4cf75;}
    .runa-parameter-row {padding: .55rem .85rem .7rem;}
    .runa-parameter-row + .runa-parameter-row {border-top: 1px solid #28282e;}
    .runa-parameter-label {color: #9a9aa4; font-size: .78rem; margin-bottom: .16rem;}
    .runa-parameter-value {color: #f1f1f3; font-size: 1.12rem; font-weight: 620;
                           line-height: 1.25; font-variant-numeric: tabular-nums;}
    .runa-parameter-published .runa-parameter-value {color: #c9c4e8;}
    .runa-parameter-estimated .runa-parameter-value {color: #f4cf75;}
    .runa-parameter-estimated .runa-photometric-distance-input .runa-parameter-value {
        color: #f1f1f3;
    }
    .runa-photometric-distance-value {font-size: 1.72rem; line-height: 1.12;}
    .st-key-kinematics_object_card,
    .st-key-multiplicity_object_card {border-color:#426b5b !important;border-radius:10px !important;background:#18181b;}
    .st-key-kinematics_object_card .runa-panel-object,
    .st-key-multiplicity_object_card .runa-panel-object {color:#64dfb4;font-size:1.15rem;font-weight:550;}
    .st-key-kinematics_object_card .runa-panel-object span,
    .st-key-multiplicity_object_card .runa-panel-object span {color:#92929d;font-size:.76rem;font-weight:400;letter-spacing:.06em;}
    .st-key-planets_registered_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .st-key-planets_registered_card h3 {color:#c9c4e8;font-size:1.15rem;font-weight:550;}
    .st-key-planets_population_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .st-key-planets_population_card h3 {color:#c9c4e8;font-size:1.15rem;font-weight:550;}
    .st-key-planets_tess_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .st-key-planets_tess_card h3 {color:#c9c4e8;font-size:1.15rem;font-weight:550;}
    .st-key-planets_object_card {border-color:#426b5b !important;border-radius:10px !important;background:#18181b;}
    .st-key-planets_object_card .runa-panel-object {color:#64dfb4;font-size:1.15rem;font-weight:550;}
    .st-key-planets_object_card .runa-panel-object span {color:#92929d;font-size:.76rem;font-weight:400;letter-spacing:.06em;}
    .st-key-kinematics_membership_card,
    .st-key-multiplicity_prediction_card {border-color:#595443 !important;border-radius:10px !important;background:#18181b;}
    .st-key-multiplicity_archive_card {border-color:#426b5b !important;border-radius:10px !important;background:#18181b;}
    .st-key-multiplicity_archive_card h3 {color:#64dfb4;font-size:1.15rem;font-weight:550;}
    .st-key-multiplicity_wds_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .st-key-multiplicity_rvs_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .runa-ml-draft-status {margin:.1rem 0 .8rem;padding:.8rem .9rem;border:1px solid #4b4226;border-radius:8px;background:#121215;}
    .runa-ml-draft-status small {display:block;color:#92929d;font-size:.72rem;letter-spacing:.06em;margin-bottom:.25rem;}
    .runa-ml-draft-status strong {display:block;color:#f4cf75;font-size:1rem;font-weight:570;}
    .runa-ml-draft-status span {display:block;color:#b7b7c0;font-size:.86rem;line-height:1.4;margin-top:.35rem;}
    .runa-model-decision {padding-top:.15rem;}
    .runa-model-decision-label {color:#92929d;font-size:.82rem;line-height:1.3;}
    .runa-model-decision-value {color:#f4cf75;font-size:1.75rem;font-weight:600;line-height:1.25;}
    .runa-score-summary em {color:#f4cf75;font-style:normal;font-weight:650;}
    .runa-score-track {position:relative;height:.58rem;margin:.7rem 0 .35rem;
                       border-radius:999px;background:#2a2a2f;overflow:visible;}
    .runa-score-fill {height:100%;border-radius:999px;background:#f4cf75;opacity:.9;}
    .runa-score-threshold {position:absolute;left:50%;top:-.18rem;bottom:-.18rem;
                           width:1px;background:rgba(244,244,245,.55);z-index:2;}
    .runa-score-caption {color:#92929d;font-size:.82rem;line-height:1.35;}
    .runa-model-coverage {color:#f1f1f3;font-size:1rem;font-weight:560;line-height:1.4;margin:.55rem 0 .35rem;}
    .runa-observation-list {display:grid;gap:.32rem;margin:.65rem 0 1.1rem;}
    .runa-observation-row {display:flex;align-items:baseline;gap:.5rem;color:#d4d4d8;font-size:.86rem;line-height:1.35;}
    .runa-observation-number {color:#f1f1f3;font-size:1.38rem;font-weight:680;font-variant-numeric:tabular-nums;white-space:nowrap;}
    .st-key-kinematics_membership_card .runa-large-metric-label,
    .st-key-kinematics_membership_card .runa-large-metric-value {color:#f4cf75;}
    .st-key-kinematics_membership_card .runa-large-metric-label {color:#92929d;}
    .st-key-kinematics_epoch_card {border-color:#595443 !important;border-radius:10px !important;background:#18181b;}
    .st-key-kinematics_epoch_card h3 {color:#f4cf75;font-size:1.15rem;font-weight:550;}
    .st-key-multiplicity_nearby_card {border-color:#426b5b !important;border-radius:10px !important;background:#18181b;}
    .st-key-multiplicity_nearby_card h3 {color:#64dfb4;font-size:1.15rem;font-weight:550;}
    .st-key-multiplicity_epoch_card {border-color:#5d5968 !important;border-radius:10px !important;background:#18181b;}
    .st-key-multiplicity_epoch_card h3 {color:#c9c4e8;font-size:1.15rem;font-weight:550;}
    .runa-parameter-error {color: #92929d; font-size: .9rem; font-weight: 430;
                           margin-left: .4rem;}
    .runa-parameter-unit {color: #a4a4ad; font-size: .82rem; font-weight: 430;
                          margin-left: .42rem;}
    .runa-parameter-source {color: #777782; font-size: .72rem; margin-top: .2rem;}
    .runa-parameter-empty {color: #85858f; font-size: .8rem; line-height: 1.4;
                           padding: .45rem .85rem .75rem;}
    .runa-neighbour-table {overflow: hidden; border: 1px solid #2d2d33;
                           border-radius: 12px; margin: .7rem 0 .85rem;
                           background: #121215;}
    .runa-neighbour-row {display: grid; grid-template-columns: minmax(0, 2.2fr)
                         minmax(4.5rem, .8fr) minmax(6rem, 1fr) minmax(4.5rem, .8fr); gap: .8rem;
                         align-items: center; padding: .72rem 1rem;
                         color: #ededf0; font-size: .86rem;}
    .runa-neighbour-row + .runa-neighbour-row {border-top: 1px solid #28282e;}
    .runa-neighbour-head {color: #8b8b95; font-size: .74rem; font-weight: 650;
                          letter-spacing: .09em; text-transform: uppercase;}
    .runa-radius-head {text-transform: none;}
    .runa-neighbour-heading {text-align:center;}
    .runa-neighbour-heading small {display: block; color: #a4a4ad; font-size: .68rem;
                                   font-weight: 430; letter-spacing: 0; text-transform: none;
                                   margin-top: .12rem;}
    .runa-neighbour-row > span:first-child {white-space: nowrap; font-size: .78rem;}
    .runa-neighbour-target {background: #1b2a24; color: #f4f4f5;}
    .runa-neighbour-check {background:#30291c; color:#f4cf75;}
    .runa-neighbour-target small {color: #a7c8b7;}
    .runa-neighbour-badge {display: inline-flex; margin-left: .45rem; padding: .12rem .42rem;
                           border: 1px solid #3c8d6a; border-radius: 999px; color: #6ee7b7;
                           font-size: .7rem; font-weight: 650; vertical-align: middle;}
    .runa-companion-check {color:#f4cf75; border-color:#a88a43; background:#30291c;}
    .runa-planet-table {overflow: hidden; border: 1px solid #2d2d33;
                        border-radius: 12px; margin: .7rem 0 .85rem;
                        background: #121215;}
    .runa-planet-row {display: grid; grid-template-columns: minmax(8rem, 1.7fr)
                      minmax(5rem, .8fr) repeat(5, minmax(3rem, .6fr)); gap: .45rem;
                      align-items: center; padding: .65rem .7rem;
                      color: #ededf0; font-size: .92rem; overflow-wrap: anywhere;}
    .runa-planet-row + .runa-planet-row {border-top: 1px solid #28282e;}
    .runa-planet-head {color: #8b8b95; font-size: .82rem; font-weight: 650;
                       letter-spacing: .09em; text-transform: uppercase;}
    .runa-planet-head .runa-orbital-symbol {text-transform:none;letter-spacing:0;}
    .runa-planet-row small {display: block; color: #a4a4ad; font-size: .76rem;
                            font-weight: 430; letter-spacing: 0; text-transform: none;
                            margin-top: .12rem;}
    .runa-planet-row > span:first-child {font-weight: 560; white-space: nowrap;}
    .runa-planet-row > span:first-child small {white-space:normal;}
    .runa-planet-status-confirmed {color:#6ee7b7;font-weight:650;}
    .runa-planet-status-candidate {color:#f4cf75;font-weight:650;}
    .runa-planet-count {color:#aaaab3; font-size:.88rem; margin:.05rem 0 .4rem;}
    .runa-planet-count-main {display:flex;align-items:baseline;gap:.55rem;}
    .runa-planet-count-value {color:#6ee7b7;font-size:2.25rem;font-weight:650;line-height:1.1;}
    .runa-planet-count-label {color:#f4f4f5;font-size:.95rem;font-weight:450;}
    .runa-planet-discovery-reference {display:block;color:#a4a4ad;font-size:.84rem;line-height:1.5;}
    .runa-planet-count-candidates {display:block;color:#92929d;margin-top:.3rem;}
    .runa-catalogue-note {color: #777780; font-size: .78rem; margin: -.2rem 0 1rem;}
    .runa-cifuentes-scroll {overflow-x: auto; border: 1px solid #2d2d33;
                            border-radius: 12px; margin: .7rem 0 1rem;
                            background: #121215;}
    .runa-cifuentes-table {border-collapse: separate; border-spacing: 0;
                           width: max-content; min-width: 2400px;
                           color: #ededf0; font-size: .96rem;
                           font-variant-numeric: tabular-nums;}
    .runa-cifuentes-table th,
    .runa-cifuentes-table td {padding: .72rem .8rem; text-align: center;
                              white-space: nowrap; border-top: 1px solid #28282e;}
    .runa-cifuentes-table thead th {color: #e2e2e5; font-size: 1.08rem;
                                    font-weight: 650; border-top: 0;
                                    border-bottom: 1px solid #36363d;}
    .runa-cifuentes-table th small {display: block; color: #9b9ba5;
                                    font-size: .86rem; font-weight: 500;
                                    margin-top: .16rem;}
    .runa-cifuentes-table th sub {font-size: .74em;}
    .runa-cifuentes-table th sub span {font-size: .72em; margin-left: .05em;}
    .runa-cifuentes-band {min-width: 6.2rem;}
    .runa-cifuentes-error {color: #8f8f99; font-size: inherit; font-weight: inherit;}
    .runa-cifuentes-missing {color: #686872;}
    .runa-cifuentes-table .runa-cifuentes-type {position: sticky; left: 0;
                                                z-index: 2; min-width: 7.2rem;
                                                text-align: left; color: #f3f3f5;
                                                font-weight: 620; background: #121215;
                                                box-shadow: 1px 0 #34343a;}
    .runa-cifuentes-table thead .runa-cifuentes-type {z-index: 3;
                                                      text-align: center;}
    .runa-photometry-table {overflow: visible; border: 1px solid #2d2d33;
                            border-radius: 12px; margin: .7rem 0 .85rem;
                            background: #121215;}
    .runa-photometry-row {display: grid; grid-template-columns: minmax(5.5rem, .9fr)
                          minmax(2rem, .35fr) minmax(3.8rem, .45fr)
                          minmax(4.3rem, .55fr) minmax(3.8rem, .45fr)
                          minmax(4.7rem, .6fr); gap: .35rem;
                          align-items: center; padding: .58rem .65rem;
                          color: #ededf0; font-size: .8rem; overflow-wrap: anywhere;}
    .runa-photometry-row + .runa-photometry-row {border-top: 1px solid #28282e;}
    .runa-photometry-head {color: #8b8b95; font-size: .73rem; font-weight: 650;
                           letter-spacing: .08em; text-transform: uppercase;}
    .runa-photometry-catalogue {color: #d4d4d8; font-weight: 560;}
    .runa-band-name {color: var(--band-colour); font-weight: 700;}
    .runa-photometry-wavelength {color: #d4d4d8; font-variant-numeric: tabular-nums;}
    .runa-photometry-head small {display: block; color: #a4a4ad; font-size: .68rem;
                                 font-weight: 430; letter-spacing: 0; text-transform: none;
                                 margin-top: .12rem;}
    .runa-photometry-error {color: #92929d; font-variant-numeric: tabular-nums;}
    .runa-photometry-range {color: #777780; font-size: .78rem;}
    .runa-photometry-missing {color: #777780;}
    .runa-photometry-missing .runa-photometry-catalogue,
    .runa-photometry-missing .runa-band-name {color: #777780;}
    .runa-identification-head .runa-class-card {
        border-color: #426b5b; border-radius: 10px;
        background: #18181b; padding: .9rem 1rem;
    }
    .runa-identification-head .runa-class-label {
        color: #92929d; font-size: .76rem; font-weight: 400;
        letter-spacing: .06em; margin-bottom: .45rem;
    }
    .runa-identification-head .runa-class-value {
        color: #64dfb4; font-size: 1.35rem; font-weight: 550;
    }
    .runa-data-section.runa-result-published {
        border-color: #5d5968; border-radius: 10px; background: #18181b;
    }
    .runa-result-published .runa-data-label {color: #e4e4e7; font-weight: 400;}
    .runa-result-published .runa-data-source,
    .runa-result-published .runa-data-unit,
    .runa-result-published .runa-error {color: #92929d;}
    .st-key-photometry_published_catalogues,
    .st-key-photometry_published_tess {
        border-color: #5d5968 !important; border-radius: 10px !important;
        background: #18181b;
    }
    .st-key-photometry_estimated_spectral,
    .st-key-photometry_estimated_distance,
    .st-key-photometry_estimated_cmd {
        border-color: #595443 !important; border-radius: 10px !important;
        background: #18181b;
    }
    .st-key-photometry_published_catalogues h3,
    .st-key-photometry_published_tess h3 {
        color: #c9c4e8; font-size: 1.15rem; font-weight: 550;
    }
    .st-key-photometry_published_summary {
        border-color: #426b5b !important; border-radius: 10px !important;
        background: #18181b;
    }
    .st-key-photometry_published_summary .runa-panel-object {
        color: #64dfb4; font-size: 1.15rem; font-weight: 550;
    }
    .st-key-photometry_published_summary .runa-panel-object span {
        color: #92929d; font-size: .76rem; font-weight: 400; letter-spacing: .06em;
    }
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-"][class*="_card"],
    div[class*="st-key-photometry_published_"],
    div[class*="st-key-photometry_estimated_"],
    div[class*="st-key-kinematics_"],
    div[class*="st-key-multiplicity_"],
    div[class*="st-key-planets_"],
    .runa-warning-tooltip,
    .runa-learning-badge,
    .runa-caution-note,
    .runa-distance-step,
    .runa-volume,
    .runa-indicator-table,
    .runa-indicator-met,
    .runa-indicator-not-met,
    .runa-indicator-unavailable,
    .runa-identification-head .runa-object-head,
    .runa-class-card,
    .runa-data-section,
    .runa-parameter-section,
    .runa-ml-draft-status,
    .runa-neighbour-table,
    .runa-neighbour-badge,
    .runa-planet-table,
    .runa-cifuentes-scroll,
    .runa-photometry-table,
    .runa-chip {
        border: 0 !important;
        box-shadow: none !important;
    }
    @media (max-width: 700px) {
        .runa-distance-flow {grid-template-columns:1fr;gap:.4rem;}
        .runa-distance-arrow {transform:rotate(90deg);}
        .runa-identification-head {grid-template-columns: 1fr; gap: .7rem;}
        .runa-data-row {grid-template-columns: 1fr; gap: .2rem;}
        .runa-neighbour-row {grid-template-columns: 1fr; gap: .2rem;}
        .runa-planet-row {grid-template-columns: 1fr; gap: .2rem;}
        .runa-data-source {text-align: left;}
        .runa-indicator-row {grid-template-columns: 3.8rem minmax(8.2rem, 1.2fr)
                             minmax(3.8rem, .6fr) minmax(4.3rem, .8fr); gap: .45rem;
                             padding-left: .65rem; padding-right: .65rem;}
        .runa-photometry-row {grid-template-columns: 1fr 1fr; gap: .2rem;}
    }
    @media (max-width: 1050px) {
        div[data-testid="stHorizontalBlock"]:has(.runa-identification-column-marker) {
            flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"]:has(.runa-identification-column-marker)
        > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    <div class="runa-brand"><a class="runa-home" href="https://ccifuentesr.github.io/index.html" target="_blank">C. Cifuentes<span>_</span></a><span>/</span><a href="https://ccifuentesr.github.io/projects.html" target="_blank">Projects</a><span>/ RUNA</span></div>
    """,
    unsafe_allow_html=True,
)

if "star" not in st.session_state:
    st.session_state["star"] = None
if "catalogue_photometry" not in st.session_state:
    st.session_state["catalogue_photometry"] = None
if "analysis_jobs" not in st.session_state:
    st.session_state["analysis_jobs"] = {}
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {}
if "analysis_object" not in st.session_state:
    st.session_state["analysis_object"] = None

available_tools = [
    "Identification", "Photometry", "Parameters", "Kinematics", "Multiplicity", "Planets", "Utilities",
]
current_tool = st.session_state.get("runa_tool", "Identification")
if current_tool not in [*available_tools, "Data", "Models"]:
    current_tool = "Identification"
if "runa_tool" not in st.session_state:
    st.session_state["runa_tool"] = current_tool
if "utility_tool" not in st.session_state:
    st.session_state["utility_tool"] = "Coordinates at epoch"
if "utility_reference_catalogue" not in st.session_state:
    st.session_state["utility_reference_catalogue"] = "Gaia DR3"
if "utility_manual_reference_epoch" not in st.session_state:
    st.session_state["utility_manual_reference_epoch"] = "2016.0"
if "utility_reference_epoch_text" not in st.session_state:
    st.session_state["utility_reference_epoch_text"] = "2016.0"


def select_tool(tool_name):
    """Select a RUNA section from the persistent left navigation."""
    st.session_state["runa_tool"] = tool_name


def select_utility(tool_name):
    """Select one tool in the workspace Utilities menu."""
    st.session_state["runa_tool"] = "Utilities"
    st.session_state["utility_tool"] = tool_name


def select_utility_reference():
    """Keep the epoch field in step with the selected reference."""
    mode = st.session_state["utility_reference_mode"]
    st.session_state["utility_reference_catalogue"] = mode
    st.session_state["utility_reference_epoch_text"] = (
        "2016.0" if mode == "Gaia DR3" else
        "2022.5" if mode == "Gaia DR4" else
        st.session_state.get("utility_manual_reference_epoch", "2016.0")
    )


def save_manual_utility_reference():
    if st.session_state.get("utility_reference_mode") == "Manual":
        st.session_state["utility_manual_reference_epoch"] = st.session_state["utility_reference_epoch_text"]


def sexagesimal_one_decimal(value, right_ascension=False):
    """Format coordinate output to a tenth of a second."""
    total = round(abs(float(value)) * (240 if right_ascension else 3600) * 10)
    first, remainder = divmod(total, 36000)
    minute, tenth_seconds = divmod(remainder, 600)
    sign = "" if right_ascension else ("−" if math.copysign(1, value) < 0 else "+")
    return f"{sign}{first % 24 if right_ascension else first:02d}:{minute:02d}:{tenth_seconds / 10:04.1f}"


def request_object_search():
    """Run the object lookup when the sidebar field is submitted."""
    st.session_state["runa_search_requested"] = True


def adjust_coordinate_epoch(key, step):
    """Change the displayed Julian year in fixed one-year steps."""
    try:
        epoch = float(st.session_state.get(key, "2016.0"))
    except (TypeError, ValueError):
        epoch = 2016.0
    st.session_state[key] = f"{min(2200.0, max(1800.0, epoch + step)):.1f}"


navigation_sections = (
    (
        None,
        (
            ("Identification", ":material/manage_search:"),
            ("Photometry", ":material/gradient:"),
            ("Parameters", ":material/tune:"),
            ("Kinematics", ":material/route:"),
            ("Multiplicity", ":material/hub:"),
            ("Planets", ":material/public:"),
        ),
    ),
    (
        "Reference",
        (
            ("Data", ":material/table_view:"),
            ("Models", ":material/model_training:"),
            ("Utilities", ":material/build:"),
        ),
    ),
)

loaded_star = st.session_state.get("star")
if loaded_star and st.session_state.get("analysis_object") != catalogue_photometry_object_key(loaded_star):
    start_object_analysis(loaded_star)

status_styles = []
search_in_progress = bool(st.session_state.get("runa_search_requested"))
for section in available_tools:
    state = "pending" if search_in_progress else analysis_section_status(section)
    selector = f'.st-key-runa_linear_{section.lower()} button::after'
    if state == "pending":
        status_styles.append(
            f'{selector} {{content:"";position:absolute;right:12px;top:11px;'
            'width:.72rem;height:.72rem;border:1.5px solid #4b4b52;border-top-color:#a6a7b0;'
            'border-radius:50%;animation:runa-spin .8s linear infinite;}'
        )
    elif state == "complete":
        status_styles.append(
            f'{selector} {{content:"";position:absolute;right:10px;top:50%;'
            'width:16px;height:16px;transform:translateY(-50%);pointer-events:none;'
            'background:url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' '
            'viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%236ee7b7\' '
            'stroke-width=\'1.8\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3E'
            '%3Ccircle cx=\'12\' cy=\'12\' r=\'9\'/%3E'
            '%3Cpath d=\'m8.5 12 2.3 2.3 4.7-4.7\'/%3E%3C/svg%3E") '
            'center/contain no-repeat;}'
        )
    elif state == "error":
        status_styles.append(
            f'{selector} {{content:"";position:absolute;right:10px;top:50%;'
            'width:16px;height:16px;transform:translateY(-50%);pointer-events:none;'
            'background:url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' '
            'viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%23d6a557\' '
            'stroke-width=\'1.8\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3E'
            '%3Ccircle cx=\'12\' cy=\'12\' r=\'9\'/%3E'
            '%3Cpath d=\'M12 8v4\'/%3E%3Ccircle cx=\'12\' cy=\'16\' r=\'.6\' fill=\'%23d6a557\' stroke=\'none\'/%3E'
            '%3C/svg%3E") center/contain no-repeat;}'
        )

utility_tool = st.session_state.get("utility_tool", "Coordinates at epoch")

st.markdown(
    "<style>" + (Path(__file__).resolve().parent / "assets" / "portfolio.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)
st.markdown("<style>" + "".join(status_styles) + "</style>", unsafe_allow_html=True)

with st.sidebar:
    st.image(
        str(Path(__file__).resolve().parent / "assets" / "runa-logo-sidebar.png"),
        width=320,
    )
    top_name = st.text_input(
        "Object name", placeholder="Search object…",
        label_visibility="collapsed", key="runa_object_name",
        on_change=request_object_search,
    )
    top_submitted = bool(
        st.session_state.pop("runa_search_requested", False) and top_name.strip()
    )
    for section_name, section_items in navigation_sections:
        if section_name:
            st.markdown(
                f'<div class="runa-sidebar-group runa-sidebar-{section_name.lower()}">{section_name}</div>',
                unsafe_allow_html=True,
            )
        for tool_name, icon in section_items:
            st.button(
                tool_name,
                use_container_width=True,
                type="primary" if current_tool == tool_name else "secondary",
                key=f"runa_linear_{tool_name.lower()}",
                on_click=select_tool,
                args=(tool_name,),
            )
    st.markdown(
        '<div class="runa-sidebar-link"><a href="https://ccifuentesr.github.io/runa.html" target="_blank">About RUNA ↗</a><br><a href="https://ccifuentesr.github.io/index.html" target="_blank">C. Cifuentes ↗</a></div>'
        '<div class="runa-sidebar-credit" aria-label="Copyright">© 2026 C. Cifuentes</div>',
        unsafe_allow_html=True,
    )
    analysis_progress()

tool = st.session_state["runa_tool"]
star = st.session_state["star"]
context_object = (
    f'<span>{escape(str(star["main_id"]))}</span>'
    if star and tool != "Utilities" else
    '<span>No object loaded</span>' if not star and tool != "Utilities" else ""
)
st.markdown(
    '<div class="runa-section-context">'
    f'<strong>{escape(tool)}</strong>'
    f'{context_object}</div>',
    unsafe_allow_html=True,
)
search_error = None
if top_submitted:
    with st.spinner("Loading star and catalogue data...", show_time=True):
        for future in st.session_state.get("analysis_jobs", {}).values():
            future.cancel()
        st.session_state["analysis_jobs"] = {}
        st.session_state["analysis_results"] = {}
        st.session_state["analysis_object"] = None
        st.session_state["star"] = None
        st.session_state["catalogue_photometry"] = None
        star = None
        try:
            data = lookup_star(top_name)
        except ValueError as error:
            search_error = str(error)
        except Exception:
            search_error = "The SIMBAD query failed. Please try again later."
        else:
            if data is None:
                search_error = "No object was found for that name."
            else:
                data["query_name"] = top_name.strip()
                try:
                    data["gaia_photometry"] = gaia_source_photometry(data["gaia_id"])
                except GaiaPhotometryUnavailable as error:
                    data["gaia_photometry"] = {}
                    data["gaia_photometry_error"] = str(error)
                data["twomass_photometry"] = lookup_2mass_j(data["twomass_id"])
                data["johnson_photometry"] = lookup_johnson_v(data["main_id"])
                data["survey_image"] = fetch_survey_image(data["ra"], data["dec"])
                st.session_state["star"] = data
                st.rerun()

if search_error:
    st.error(search_error)

if tool in ("Identification", "Data", "Models", "Parameters", "Utilities"):
    input_card = None
    result_card = (
        st.container(border=False)
        if tool not in ("Data", "Models") else None
    )
    tess_card = None
else:
    column_widths = [0.62, 1.38] if tool == "Parameters" else [1, 1]
    inputs, results = st.columns(column_widths, gap="small")
    input_card_key = (
        "photometry_published_summary" if tool == "Photometry" else
        "kinematics_object_card" if tool == "Kinematics" else
        "multiplicity_object_card" if tool == "Multiplicity" else
        "planets_object_card" if tool == "Planets" else None
    )
    if tool == "Multiplicity":
        input_card = inputs.container(border=True, key=input_card_key)
    else:
        input_card = inputs.container(border=True, key=input_card_key)
    nearby_card = inputs.container(border=True, key="multiplicity_nearby_card") if tool == "Multiplicity" else None
    diagnostics_card = inputs.container(
        border=True, key="multiplicity_rvs_card"
    ) if tool == "Multiplicity" else None
    epoch_card = inputs.container(border=True, key="multiplicity_epoch_card") if tool == "Multiplicity" else None
    coordinate_epoch_card = (
        inputs.container(border=True, key="kinematics_epoch_card")
        if tool == "Kinematics" else None
    )
    cmd_card = (
        inputs.container()
        if tool == "Photometry" else None
    )
    additional_catalogue_card = (
        inputs.container(border=True, key="photometry_published_catalogues") if tool == "Photometry" else None
    )
    prediction_card = results.container(
        border=True, key="multiplicity_prediction_card"
    ) if tool == "Multiplicity" else None
    reported_multiplicity = published_multiplicity(star) if tool == "Multiplicity" else None
    reported_card = results.container(
        border=True, key="multiplicity_reported_card"
    ) if tool == "Multiplicity" else None
    archive_card = results.container(
        border=True, key="multiplicity_archive_card"
    ) if tool == "Multiplicity" else None
    result_card = None if tool == "Multiplicity" else results.container(
        border=tool not in ("Photometry", "Parameters"),
        key=("kinematics_membership_card" if tool == "Kinematics" else
             "planets_registered_card" if tool == "Planets" else None),
    )
    planet_population_card = (
        inputs.container(border=True, key="planets_population_card")
        if tool == "Planets" else None
    )
    tess_card = inputs.container(
        border=True, key="planets_tess_card"
    ) if tool == "Planets" else None

if tool == "Data":
    st.markdown("### Reference data")
    st.caption(
        "Average astrophysical parameters and absolute magnitudes for K5 V–L2.0 "
        "dwarfs from Cifuentes et al. (2020), Tables 6 and 7."
    )
    cifuentes20_table(load_reference_data())
    st.markdown(
        "These reference values describe single, main-sequence K5 V–L2.0 stars "
        "with no companions within 5 arcsec, no extreme-youth overluminosity, "
        "and good Gaia DR2 astrometric and photometric quality. **N** is the "
        "number of well-behaved stars in each spectral-type bin."
    )
    st.markdown(
        "From M0.0 V to L2.0, each value is a three-subtype rolling median after "
        "2.5σ clipping. K5 V and K7 V use simple medians. Grey parentheses show "
        "the standard deviations, which include intrinsic main-sequence scatter "
        "and become less precise in sparsely populated late-type bins. The unsmoothed "
        "K5 V boundary values should be treated with care."
    )
    st.markdown(
        "The 14 representative absolute magnitudes can be combined into colours, "
        "for example G−J = M<sub>G</sub> − M<sub>J</sub>.",
        unsafe_allow_html=True,
    )

elif tool == "Models":
    show_models()

elif tool == "Identification":
    with result_card:
        if star is None:
            pass
        else:
            object_type = object_type_name(star)
            spectral_type = escape(str(star.get("sp_type") or "—"))
            st.markdown(
                '<div class="runa-identification-head">'
                '<div class="runa-object-head runa-class-card">'
                '<div class="runa-class-label">Object</div>'
                f'<div class="runa-class-value">{escape(str(star["main_id"]))}</div></div>'
                '<div class="runa-classification">'
                '<div class="runa-class-card">'
                '<div class="runa-class-label">Object type</div>'
                f'<div class="runa-class-value">{escape(object_type)}</div>'
                '</div>'
                '<div class="runa-class-card">'
                '<div class="runa-class-label">Spectral type</div>'
                f'<div class="runa-class-value">{spectral_type}</div>'
                '</div></div></div>',
                unsafe_allow_html=True,
            )

            result_left, result_right = st.columns(2, gap="small")

            with result_left:
                st.markdown(
                    '<span class="runa-identification-column-marker"></span>',
                    unsafe_allow_html=True,
                )
                survey_image = star.get("survey_image")
                if survey_image is not None:
                    st.markdown(
                        '<div class="runa-identification-data">'
                        '<div class="runa-section-title">Image (visible)</div></div>',
                        unsafe_allow_html=True,
                    )
                    survey_image_view(survey_image["data"])
                    st.caption(survey_image["caption"])
                else:
                    st.markdown(
                        '<div class="runa-identification-data">'
                        '<div class="runa-section-title">Survey image</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.info("No SDSS or DSS2 image could be loaded.")

                polar_fig = polar_position_plot(star)
                if polar_fig is not None:
                    st.pyplot(polar_fig, use_container_width=True)
                    plt.close(polar_fig)
                    st.caption(
                        "RA gives the polar angle and the radius is the distance from the Sun in pc. "
                        "The proper-motion arrow shows the change expected over 100,000 years."
                    )
                    volume = local_volume_status(star)
                    if volume["inside_10pc"]:
                        multiple = volume["multiple_match"]
                else:
                    st.info("A positive parallax and coordinates are needed for the polar view.")

            identifier_rows = [
                ("SIMBAD ID", star["main_id"], "primary name"),
                *catalogue_rows(star),
            ]
            with result_right:
                st.markdown(
                    '<span class="runa-identification-column-marker"></span>',
                    unsafe_allow_html=True,
                )
                data_section(
                    "Identifiers", identifier_rows,
                    class_name="runa-identification-data runa-result-published",
                )

            parallax_text = None
            parallax_error = None
            if star["plx_value"] is not None:
                parallax_text = f"{star['plx_value']:.3f}"
                if star.get("plx_err") is not None:
                    parallax_error = f"{star['plx_err']:.3f}"
            distance = None
            distance_error = None
            if star["distance_pc"] is not None:
                distance = f"{star['distance_pc']:.4f}"
                if star.get("distance_error_pc") is not None:
                    distance_error = f"{star['distance_error_pc']:.4f}"

            ra_error = None
            dec_error = None
            if star.get("coo_err_maj") is not None:
                if star.get("coo_err_angle") == 90:
                    ra_error = f"{star['coo_err_maj']:.3f}"
                    if star.get("coo_err_min") is not None:
                        dec_error = f"{star['coo_err_min']:.3f}"
                else:
                    ellipse = f"{star['coo_err_maj']:.3f}"
                    if star.get("coo_err_min") is not None:
                        ellipse += f" × {star['coo_err_min']:.3f}"
                    ra_error = dec_error = ellipse

            pmra_error = None
            pmdec_error = None
            if star.get("pm_err_maj") is not None:
                if star.get("pm_err_angle") == 90:
                    pmra_error = f"{star['pm_err_maj']:.3f}"
                    if star.get("pm_err_min") is not None:
                        pmdec_error = f"{star['pm_err_min']:.3f}"
                else:
                    pmra_error = f"{star['pm_err_maj']:.3f}"
                    if star.get("pm_err_min") is not None:
                        pmra_error += f" × {star['pm_err_min']:.3f}"
                    pmdec_error = pmra_error
            radial_velocity = None
            radial_velocity_error = None
            if star["rvz_radvel"] is not None:
                radial_velocity = f"{star['rvz_radvel']:.2f}"
                if star.get("rvz_err") is not None:
                    radial_velocity_error = f"{star['rvz_err']:.3f}"
            coordinate_source = source_name(star.get("coo_bibcode"))
            parallax_source = source_name(star.get("plx_bibcode"))
            motion_source = source_name(star.get("pm_bibcode"))
            distance_source = parallax_source
            distance_notice = ""
            volume = local_volume_status(star)
            if volume["inside_10pc"]:
                multiple = volume["multiple_match"]
                if multiple:
                    notice_text = (
                        'Component of multiple system within 10 pc in '
                        '<a href="https://ui.adsabs.harvard.edu/abs/2026MNRAS.549ag838G/abstract" '
                        'target="_blank" rel="noopener noreferrer">González-Payo et al. (2026)</a>.'
                    )
                elif volume["sample_match"]:
                    notice_text = (
                        'Object within 10 parsec in '
                        '<a href="https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.201R/abstract" '
                        'target="_blank" rel="noopener noreferrer">Reylé et al. (2021)</a>.'
                    )
                else:
                    notice_text = "Distance is within 10 pc; no match in the cited samples."
                distance_notice = (
                    '<span class="runa-distance-notice" tabindex="0" '
                    'aria-label="Nearby-star sample notice; hover for details">i'
                    f'<span class="runa-distance-notice-text">{notice_text}</span></span>'
                )
            with result_right:
                data_section(
                    "Astrometry",
                    [
                        ("RA", sexagesimal(star["ra"], True),
                         coordinate_source, ra_error, "mas"),
                        ("Dec", sexagesimal(star["dec"]),
                         coordinate_source, dec_error, "mas"),
                        ("Parallax", parallax_text, parallax_source,
                         parallax_error, "mas"),
                        ("Distance", distance, distance_source,
                         distance_error, "pc", distance_notice),
                        ("pm RA", f"{star['pmra']:.2f}"
                         if star["pmra"] is not None else None,
                         motion_source, pmra_error, "mas/yr"),
                        ("pm Dec", f"{star['pmdec']:.2f}"
                         if star["pmdec"] is not None else None,
                         motion_source, pmdec_error, "mas/yr"),
                        ("RV", radial_velocity,
                         source_name(star.get("rvz_bibcode")), radial_velocity_error,
                         "km/s"),
                    ],
                    class_name="runa-identification-data runa-result-published",
                )

            gaia = star["gaia_photometry"]
            twomass = star["twomass_photometry"]
            johnson = star.get("johnson_photometry", {})
            g_mag = gaia.get("phot_g_mean_mag")
            bp_mag = gaia.get("phot_bp_mean_mag")
            rp_mag = gaia.get("phot_rp_mean_mag")
            j_mag = twomass.get("j_mag")
            j_quality = twomass.get("j_quality")
            g_error = gaia.get("phot_g_mean_mag_error")
            bp_error = gaia.get("phot_bp_mean_mag_error")
            rp_error = gaia.get("phot_rp_mean_mag_error")
            j_error = twomass.get("j_error")
            v_mag = johnson.get("v_mag")
            v_error = johnson.get("v_error")
            with result_right:
                data_section(
                    "Photometry",
                    [
                        ("Gaia G", f"{g_mag:.3f}" if g_mag is not None else None,
                         "Gaia DR3", f"{g_error:.4f}" if g_error else None, "mag"),
                        ("Gaia BP", f"{bp_mag:.3f}" if bp_mag is not None else None,
                         "Gaia DR3", f"{bp_error:.4f}" if bp_error else None, "mag"),
                        ("Gaia RP", f"{rp_mag:.3f}" if rp_mag is not None else None,
                         "Gaia DR3", f"{rp_error:.4f}" if rp_error else None, "mag"),
                        ("2MASS J", f"{j_mag:.3f}" if j_mag is not None else None,
                         f"2MASS (Q={j_quality})" if j_quality else "2MASS",
                         f"{j_error:.3f}" if j_error is not None else None, "mag"),
                        ("Johnson V", f"{v_mag:.3f}" if v_mag is not None else None,
                         "UCAC4", f"{v_error:.3f}" if v_error is not None else None, "mag"),
                    ],
                    class_name="runa-identification-data runa-result-published",
                )

            metallicity = None
            if star["fe_h"] is not None:
                metallicity = f"{star['fe_h']:.2f}"
            with result_right:
                data_section(
                    "Other parameters",
                    [
                        ("[Fe/H]", metallicity, "SIMBAD compilation", None, "dex"),
                    ],
                    class_name="runa-identification-data",
                )

            with result_left:
                simbad_server = star.get("simbad_server") or "simbad.cds.unistra.fr"
                simbad_url = (
                    f"https://{simbad_server}/simbad/sim-id?Ident="
                    + quote(star["query_name"])
                )
                st.link_button("Open SIMBAD record ↗", simbad_url)


elif tool == "Photometry":
    position = star_cmd_position(star) if star is not None else None
    photometry_parameters = calculate_parameters(star) if star is not None else None
    gaia = star.get("gaia_photometry", {}) if star is not None else {}
    twomass = star.get("twomass_photometry", {}) if star is not None else {}

    with input_card:
        object_panel_title(star, "Photometric overview")
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            johnson = star.get("johnson_photometry", {})
            bp_rp = None
            if gaia.get("phot_bp_mean_mag") is not None and gaia.get("phot_rp_mean_mag") is not None:
                bp_rp = gaia["phot_bp_mean_mag"] - gaia["phot_rp_mean_mag"]
            absolute_g = photometry_parameters.get("M_G")
            absolute_j = photometry_parameters.get("M_J")
            first_row = st.columns(2)
            second_row = st.columns(2)
            third_row = st.columns(2)
            with first_row[0]:
                photometry_summary_metric("Gaia G", gaia.get("phot_g_mean_mag"))
            with first_row[1]:
                photometry_summary_metric("2MASS J", twomass.get("j_mag"))
            with second_row[0]:
                photometry_summary_metric("Johnson V", johnson.get("v_mag"))
            with second_row[1]:
                photometry_summary_metric("BP−RP colour", bp_rp)
            with third_row[0]:
                photometry_summary_metric(
                    "M_G",
                    absolute_g["value"] if absolute_g is not None else None,
                    absolute_g.get("error") if absolute_g is not None else None,
                )
            with third_row[1]:
                photometry_summary_metric(
                    "M_J",
                    absolute_j["value"] if absolute_j is not None else None,
                    absolute_j.get("error") if absolute_j is not None else None,
                )

    with additional_catalogue_card:
        panel_heading("Additional catalogue photometry")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            st.markdown(
                '<p class="runa-filter-reference">The '
                '<a href="https://svo2.cab.inta-csic.es/theory/fps/" '
                'target="_blank">Carlos Rodrigo Filter Profile Service</a> '
                'is the reference for the filter profiles.</p>',
                unsafe_allow_html=True,
            )
            catalogue_records, catalogue_state = analysis_result("photometry_catalogues")
            if catalogue_state == "pending":
                st.markdown(
                    '<div class="runa-loading-status" role="status">'
                    '<span class="runa-loading-spinner" aria-hidden="true"></span>'
                    'Searching photometric catalogues…</div>',
                    unsafe_allow_html=True,
                )
            elif catalogue_state == "error":
                st.warning("The photometric catalogue search could not be completed.")
            else:
                st.caption("Photometric catalogue search complete.")
            if catalogue_records is None:
                pass
            else:
                catalogue_records = flag_sed_anomalies(catalogue_records)
                st.write("#### Spectral energy distribution")
                sed_points = build_sed_points(catalogue_records)
                st.caption(
                    "Optical preference per band: APASS → SkyMapper → SDSS9 → "
                    "UCAC4, by availability. All measurements remain in the table. "
                    "Catalogue preference does not replace quality checks."
                )
                if len(sed_points) < 2:
                    st.info(
                        "There are not enough calibrated bands to build the SED. "
                        "Bands without a VOSA zero point are left out."
                    )
                else:
                    show_sed_bands = st.checkbox(
                        "Show photometric bands in the SED",
                        value=False,
                        key="show_sed_bands",
                    )
                    sed_fig = sed_figure(
                        sed_points,
                        star["main_id"] if star is not None else "Loaded object",
                        show_bands=show_sed_bands,
                    )
                    st.pyplot(sed_fig, use_container_width=True)
                    plt.close(sed_fig)
                    st.caption(
                        "Fluxes use the catalogue magnitude system: AB for APASS "
                        "gri and SkyMapper, and the existing VOSA zero points for "
                        "the other bands. SkyMapper positions use the central "
                        "wavelengths in Onken et al. (2024), Table 1; its filter "
                        "curves are not displayed. The plot shows λFλ in W m⁻²; "
                        "points are ordered by effective wavelength. This is the "
                        "observed SED prepared for a future VOSA fit."
                    )
                sort_by_wavelength = st.checkbox(
                    "Sort bands by effective wavelength",
                    value=False,
                    key="sort_catalogue_photometry",
                )
                catalogue_photometry_table(
                    catalogue_records,
                    sort_by_wavelength=sort_by_wavelength,
                )
                st.download_button(
                    "Download all catalogue photometry (.csv)",
                    data=catalogue_photometry_csv(catalogue_records, star),
                    file_name=f"runa_{catalogue_photometry_object_key(star)}_photometry.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_all_catalogue_photometry",
                )
                vosa_file = build_vosa_input(star, catalogue_records)
                st.write("#### VOSA")
                st.caption(
                    "VOSA (VO SED Analyzer) builds and analyses spectral energy "
                    "distributions, compares them with stellar models and can "
                    "estimate physical properties such as bolometric luminosity."
                )
                vosa_actions = st.columns(2, gap="small")
                with vosa_actions[0]:
                    if vosa_file is None:
                        st.button(
                            "↓ Download VOSA input",
                            disabled=True,
                            use_container_width=True,
                            key="download_vosa_input_disabled",
                        )
                    else:
                        st.download_button(
                            "↓ Download VOSA input",
                            data=vosa_file["content"],
                            file_name=vosa_file["filename"],
                            mime="text/plain",
                            use_container_width=True,
                            key="download_vosa_input",
                        )
                with vosa_actions[1]:
                    st.link_button(
                        "Open VOSA ↗",
                        "https://svo2.cab.inta-csic.es/theory/vosa/",
                        use_container_width=True,
                    )
                if vosa_file is None:
                    st.info(
                        "A positive Gaia parallax and at least one mapped band with "
                        "an uncertainty are needed to create a VOSA input file."
                    )
                else:
                    distance_text = f"{vosa_file['distance']:.3f} pc"
                    if vosa_file["distance_error"] is not None:
                        distance_text += f" ({vosa_file['distance_error']:.3f} pc)"
                    st.caption(
                        f"{vosa_file['bands']} mapped bands · Gaia distance: {distance_text}. "
                        "Upload this file to VOSA as magnitudes. Extinction is left blank."
                    )


    with cmd_card:
        with st.container(border=True, key="photometry_estimated_cmd"):
            panel_heading("Colour–magnitude diagram")
            if star is None:
                st.write("Choose a star in **Identification** first.")
            elif position is None:
                st.info("A positive parallax and Gaia BP/RP measurements are needed for the diagram.")
            else:
                reference, reference_state = analysis_result("photometry_cmd")
                if reference_state == "pending":
                    st.markdown(
                        '<div class="runa-loading-status" role="status">'
                        '<span class="runa-loading-spinner" aria-hidden="true"></span>'
                        'Loading Gaia reference stars…</div>',
                        unsafe_allow_html=True,
                    )
                elif reference_state == "error":
                    st.error("The Gaia reference sample could not be loaded.")
                else:
                    colour, absolute_g = position
                    fig, ax = plt.subplots(figsize=(6.6, 6.2))
                    fig.patch.set_facecolor("#151518")
                    ax.set_facecolor("#151518")
                    ax.scatter(
                        reference["colour"], reference["absolute_g"],
                        s=5, color="#d4d4d8", alpha=0.22, linewidths=0,
                        label="Gaia reference",
                    )
                    ax.scatter(
                        colour, absolute_g, s=115, color="#ef4444",
                        edgecolor="#111113", linewidth=1.2,
                        label=str(star["main_id"]), zorder=3,
                    )
                    ax.set_xlabel("Gaia BP−RP colour (mag)", fontsize=13)
                    ax.set_ylabel("Absolute Gaia G magnitude (mag)", fontsize=13)
                    ax.invert_yaxis()
                    ax.grid(False)
                    ax.tick_params(colors="#a1a1aa", labelsize=11)
                    ax.xaxis.label.set_color("#d4d4d8")
                    ax.yaxis.label.set_color("#d4d4d8")
                    for spine in ax.spines.values():
                        spine.set_color("#3f3f46")
                    ax.legend(
                        facecolor="#19191e", edgecolor="#3f3f46",
                        labelcolor="#d4d4d8", loc="lower left", fontsize=10,
                    )
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        "Light-grey points are Gaia DR3 stars with parallax S/N > 20, "
                        "RUWE < 1.4, more than eight visibility periods and high-S/N G, BP and RP photometry."
                    )

elif tool == "Parameters":
    parameters, parameters_state = analysis_result("parameters_calculation") if star is not None else (None, "idle")
    published_parameters = published_stellar_parameters(star)
    additional_measurements = None
    if star is not None:
        additional_measurements, literature_state = analysis_result("parameters_literature")
        if literature_state == "error":
            additional_measurements = {"unavailable": ["all"]}

    def preferred_published(entries):
        """Prefer Cifuentes et al. (2025) when both bundled catalogues match."""
        return next(
            (entry for entry in entries if entry["reference"].endswith("(2025)")),
            entries[0] if entries else None,
        )

    preferred_published_values = {
        name: preferred_published(entries)
        for name, entries in published_parameters.items()
    }
    preferred_luminosity = preferred_published_values.get("luminosity")
    preferred_temperature = preferred_published_values.get("effective_temperature")

    estimated_temperature = None
    temperature_for_radius = preferred_temperature
    if star is not None and temperature_for_radius is None:
        spectral_type = str(star.get("sp_type") or "").strip()
        spectral_type_origin = "published spectral type"
        if spectral_type in ("", "~", "--"):
            spectral_type = None
            spectral_type_origin = "RUNA spectral-type estimate"
            gaia = star.get("gaia_photometry", {})
            twomass = star.get("twomass_photometry", {})
            g_mag = gaia.get("phot_g_mean_mag")
            j_mag = twomass.get("j_mag")
            if g_mag is not None and j_mag is not None:
                try:
                    _, subtype = estimate_subtype(float(g_mag), float(j_mag))
                    spectral_type = f"M{subtype:.1f}"
                except ValueError:
                    spectral_type = None

        spectral_key = spectral_type_key(spectral_type)
        table_temperature = None
        if spectral_key is not None:
            if spectral_key[0] == "M":
                table_temperature = temperature_from_spectral_type(spectral_type)
                temperature_reference = "Cifuentes et al. (2020), Table 6"
            else:
                table_temperature = temperature_from_mamajek(spectral_type)
                temperature_reference = (
                    "Pecaut & Mamajek (2013) · Mamajek dwarf sequence"
                )
        if table_temperature is not None:
            estimated_temperature = {
                **table_temperature,
                "origin": spectral_type_origin,
                "reference": temperature_reference,
            }
            temperature_for_radius = estimated_temperature

    photometric_luminosity = None
    photometric_luminosity_band = None
    if parameters is not None:
        for band in ("M_G", "M_J"):
            estimate = parameters["luminosities"].get(band)
            if estimate is not None:
                photometric_luminosity = {
                    "value": estimate["luminosity"],
                    "error": estimate["luminosity_error"],
                }
                photometric_luminosity_band = band
                break

    estimated_radius = None
    if photometric_luminosity is not None and temperature_for_radius is not None:
        estimated_radius = radius_from_luminosity_temperature(
            photometric_luminosity["value"],
            photometric_luminosity.get("error"),
            temperature_for_radius["value"],
            temperature_for_radius.get("error"),
        )
    eker_mass = (
        mass_from_luminosity_eker(
            photometric_luminosity["value"], photometric_luminosity.get("error")
        ) if photometric_luminosity else None
    )
    catalogue_records = st.session_state.get("catalogue_photometry")
    metallicity = None
    metallicity_inputs = []
    if star is not None and catalogue_records is not None and parameters is not None:
        gaia = star.get("gaia_photometry", {})
        w1 = catalogue_band(catalogue_records, "AllWISE", "W1")
        w2 = catalogue_band(catalogue_records, "AllWISE", "W2")
        bp = catalogue_band(catalogue_records, "Gaia DR3", "BP")
        rp = catalogue_band(catalogue_records, "Gaia DR3", "RP")
        metallicity = metallicity_from_gaia_wise(
            gaia.get("phot_g_mean_mag"),
            gaia.get("phot_g_mean_mag_error"),
            bp.get("value") if bp else None,
            bp.get("error") if bp else None,
            rp.get("value") if rp else None,
            rp.get("error") if rp else None,
            w1.get("value") if w1 else None,
            w1.get("error") if w1 else None,
            w2.get("value") if w2 else None,
            w2.get("error") if w2 else None,
            parameters["parallax"],
            parameters["parallax_error"],
        )
        for catalogue, label, source_label in (
            ("Gaia DR3", "BP", "Gaia DR3"),
            ("Gaia DR3", "RP", "Gaia DR3"),
            ("AllWISE", "W1", "AllWISE"),
            ("AllWISE", "W2", "AllWISE"),
        ):
            band = catalogue_band(catalogue_records, catalogue, label)
            if band and band.get("value") is not None:
                metallicity_inputs.append(
                    (
                        label,
                        f"{band['value']:.3f}",
                        source_label,
                        f"{band['error']:.3f}" if band.get("error") is not None else None,
                        "mag",
                    )
                )

    with result_card:
        if star is None:
            st.info("Choose a star in **Identification** first.")
            stellar_parameter_table([
                ("Luminosity", "L☉", [], []),
                ("Teff", "K", [], []),
                ("Radius", "R☉", [], []),
                ("Mass", "M☉", [], []),
                ("Metallicity", "dex", [], []),
            ])
        else:
            luminosity_rows = []
            if preferred_luminosity is not None:
                luminosity_rows.append(
                    (
                        "Bolometric luminosity",
                        f'{preferred_luminosity["value"]:.6g}',
                        preferred_luminosity["reference"],
                        (
                            f'{preferred_luminosity["error"]:.3g}'
                            if preferred_luminosity["error"] is not None else None
                        ),
                        "L☉",
                    )
                )
            luminosity_estimates = []
            if photometric_luminosity is not None:
                luminosity_value, luminosity_error = format_value_and_error(
                    photometric_luminosity["value"],
                    photometric_luminosity.get("error"),
                )
                luminosity_estimates.append(
                    (
                        f"From {photometric_luminosity_band}",
                        luminosity_value,
                        "Cifuentes et al. (2020)",
                        luminosity_error,
                        "L☉",
                    )
                )

            preferred_radius = preferred_published_values.get("radius")
            radius_rows = []
            if preferred_radius is not None:
                radius_rows.append(
                    (
                        "Published value",
                        f'{preferred_radius["value"]:.4f}',
                        preferred_radius["reference"],
                        (
                            f'{preferred_radius["error"]:.4f}'
                            if preferred_radius["error"] is not None else None
                        ),
                        "R☉",
                    )
                )
            radius_estimates = []
            if estimated_radius is not None:
                radius_estimates.append(
                    (
                        f"From {photometric_luminosity_band} and T_eff",
                        f'{estimated_radius["value"]:.4f}',
                        "Stefan–Boltzmann law",
                        f'{estimated_radius["error"]:.4f}'
                        if estimated_radius["error"] is not None else None,
                        "R☉",
                    )
                )
            preferred_mass = preferred_published_values.get("mass")
            mass_rows = []
            if preferred_mass is not None:
                mass_rows.append(
                    (
                        "Published value",
                        f'{preferred_mass["value"]:.4f}',
                        preferred_mass["reference"],
                        (
                            f'{preferred_mass["error"]:.4f}'
                            if preferred_mass["error"] is not None else None
                        ),
                        "M☉",
                    )
                )
            mass_estimates = []
            if eker_mass is not None:
                mass_estimates.append(
                    (
                        f"From {photometric_luminosity_band} luminosity",
                        f'{eker_mass["value"]:.4f}',
                        "Eker et al. (2018)",
                        f'{eker_mass["error"]:.4f}',
                        "M☉",
                    )
                )

            temperature_rows = []
            if preferred_temperature is not None:
                temperature_rows.append(
                    (
                        "Published value",
                        f'{preferred_temperature["value"]:.0f}',
                        preferred_temperature["reference"],
                        (
                            f'{preferred_temperature["error"]:.0f}'
                            if preferred_temperature["error"] is not None else None
                        ),
                        "K",
                    )
                )
            temperature_estimates = []
            if preferred_temperature is not None and estimated_radius is not None:
                temperature_estimates.append(
                    (
                        "Published Teff used for R",
                        f'{preferred_temperature["value"]:.0f}',
                        preferred_temperature["reference"],
                        (
                            f'{preferred_temperature["error"]:.0f}'
                            if preferred_temperature["error"] is not None else None
                        ),
                        "K",
                        "input",
                    )
                )
            elif estimated_temperature is not None:
                temperature_label = (
                    "From published SpT"
                    if estimated_temperature["origin"] == "published spectral type"
                    else "From RUNA SpT"
                )
                temperature_estimates.append(
                    (
                        f'{temperature_label} {estimated_temperature["spectral_type"]}',
                        f'{estimated_temperature["value"]:.0f}',
                        estimated_temperature["reference"],
                        (
                            f'{estimated_temperature["error"]:.0f}'
                            if estimated_temperature["error"] is not None else None
                        ),
                        "K",
                    )
                )
            preferred_metallicity = preferred_published_values.get("metallicity")
            metallicity_rows = []
            if preferred_metallicity is not None:
                metallicity_rows.append(
                    (
                        "[Fe/H]",
                        f'{preferred_metallicity["value"]:.3f}',
                        preferred_metallicity["reference"],
                        (
                            f'{preferred_metallicity["error"]:.3f}'
                            if preferred_metallicity["error"] is not None else None
                        ),
                        "dex",
                    )
                )
            metallicity_estimates = []
            if metallicity is not None:
                metallicity_estimates.append(
                    (
                        "[Fe/H]",
                        f"{metallicity['value']:.3f}",
                        "Duque-Arribas et al. (2023)",
                        f"{metallicity['error']:.3f}",
                        "dex",
                    )
                )

            object_panel_title(star)
            stellar_parameter_table([
                ("Luminosity", "L☉", luminosity_rows, luminosity_estimates),
                ("Teff", "K", temperature_rows, temperature_estimates),
                ("Radius", "R☉", radius_rows, radius_estimates),
                ("Mass", "M☉", mass_rows, mass_estimates),
                ("Metallicity", "dex", metallicity_rows, metallicity_estimates),
            ], additional_measurements=additional_measurements)
            if catalogue_records is None:
                st.caption("Photometric catalogues are loading for the metallicity estimate.")
            with st.expander("Model scopes", expanded=False):
                st.caption(
                    "Published values use Cifuentes et al. (2025) when both bundled "
                    "catalogues contain the star. Estimated luminosity uses the applicable "
                    "Cifuentes et al. (2020) M_G or M_J relation. Radius uses the "
                    "Stefan–Boltzmann law and the published temperature when available. "
                    "Otherwise, M dwarfs use Cifuentes et al. (2020), Table 6, and other "
                    "dwarfs use the Mamajek sequence. Mass uses Eker et al. (2018) only "
                    "inside its 0.179–31 M☉ main-sequence calibration. RUNA does not "
                    "extrapolate it or apply a mass–radius fallback. Formal uncertainties "
                    "from the photometric polynomial are not a calibrated per-star scatter."
                )


elif tool == "Kinematics":
    banyan_result, banyan_state = analysis_result("kinematics_population") if star is not None else (None, "idle")
    banyan_error = (
        st.session_state.get("analysis_results", {}).get("kinematics_population", {}).get("error")
        if banyan_state == "error" else None
    )
    with input_card:
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            values = banyan_inputs(star)
            input_rows = []
            for label, key, unit, error_key in (
                ("RA", "ra", "deg", None),
                ("Dec", "dec", "deg", None),
                ("pm RA", "pmra", "mas/yr", "epmra"),
                ("pm Dec", "pmdec", "mas/yr", "epmdec"),
                ("Parallax", "plx", "mas", "eplx"),
                ("Radial velocity", "rv", "km/s", "erv"),
            ):
                value = values.get(key)
                error = values.get(error_key) if error_key else None
                if value is not None:
                    input_rows.append(
                        (
                            label,
                            f"{value:.3f}",
                            "Gaia DR3" if key != "rv" else "SIMBAD",
                            f"{error:.3f}" if error is not None else None,
                            unit,
                        )
                    )
            data_section(
                "Kinematic inputs",
                input_rows,
                class_name="runa-identification-data runa-kinematic-inputs",
            )

    with result_card:
        panel_heading("Kinematic population prediction")
        if banyan_error:
            st.error(banyan_error)
        elif star is None:
            st.write("Choose a star in **Identification** first.")
        elif banyan_state == "pending":
            st.write("Calculating kinematic population membership…")
        elif banyan_result is None:
            st.write("No population-membership prediction is available.")
        else:
            banyan = banyan_result
            best_hypothesis = banyan["best_hypothesis"]
            best_association = banyan["best_association"]
            probability = banyan["association_probability"]
            if best_hypothesis == "FIELD":
                st.markdown(
                    '<div class="runa-field-hypothesis">Field</div>',
                    unsafe_allow_html=True,
                )
            else:
                large_metric(
                    "Most likely population",
                    best_hypothesis,
                    f"{probability:.1%} probability",
                )
                if banyan.get("age_myr") is not None:
                    st.caption(f"Approximate population age: {banyan['age_myr']:g} Myr")

            st.caption(
                "Most likely hypothesis based on the Bayesian BANYAN Σ classifier, "
                "using Gaia position, proper motion and parallax when available, "
                "with the SIMBAD radial velocity when available."
            )

            probability_values = [
                ("Field", banyan["field_probability"]),
                *banyan["probabilities"][:6],
            ]
            probability_max = max(probability_values, key=lambda item: item[1])[0]
            probability_rows = "".join(
                '<div class="runa-probability-row">'
                f'<span>{escape(str(name))}</span>'
                f'<strong class="{"runa-probability-maximum" if name == probability_max else ""}">'
                f'{value:.1%}</strong></div>'
                for name, value in probability_values
            )
            with st.expander("Population probabilities", expanded=False):
                st.markdown(
                    '<div class="runa-probability-list">' + probability_rows + "</div>",
                    unsafe_allow_html=True,
                )

            velocity_rows = []
            for label in ("U", "V", "W"):
                value = banyan.get(label)
                error = banyan.get(f"E{label}")
                if value is not None:
                    velocity_rows.append(
                        (
                            label,
                            f"{value:.2f}",
                            "BANYAN Σ",
                            f"{error:.2f}" if error is not None else None,
                            "km/s",
                        )
                    )
            with st.expander("Galactic velocity", expanded=False):
                st.markdown(
                    '<section class="runa-data-section runa-identification-data">'
                    + "".join(data_row(*row) for row in velocity_rows)
                    + "</section>",
                    unsafe_allow_html=True,
                )

    with coordinate_epoch_card:
        epoch_label, epoch_controls = st.columns((1.35, .65), gap="small", vertical_alignment="center")
        with epoch_label:
            panel_heading("Coordinates at epoch:")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            gaia_epoch = star.get("gaia_photometry") or {}
            epoch_key = f"coordinate_epoch_{star['main_id']}"
            with epoch_controls:
                epoch_value, epoch_buttons = st.columns(
                    (1.5, .7), gap=0, vertical_alignment="center"
                )
                with epoch_value:
                    target_epoch_text = st.text_input(
                        "Target epoch (Julian year)", value="2016.0",
                        label_visibility="collapsed", key=epoch_key,
                    )
                with epoch_buttons:
                    epoch_down, epoch_up = st.columns(
                        (1, 1), gap=0, vertical_alignment="center"
                    )
                    with epoch_down:
                        st.button(
                            "−", key=f"coordinate_epoch_down_{star['main_id']}",
                            on_click=adjust_coordinate_epoch,
                            args=(epoch_key, -0.5), use_container_width=True,
                        )
                    with epoch_up:
                        st.button(
                            "+", key=f"coordinate_epoch_up_{star['main_id']}",
                            on_click=adjust_coordinate_epoch,
                            args=(epoch_key, 0.5), use_container_width=True,
                        )
            try:
                target_epoch = float(target_epoch_text)
                if not 1800 <= target_epoch <= 2200:
                    raise ValueError
            except ValueError:
                st.error("Enter a Julian year between 1800.0 and 2200.0, using a decimal point.")
                target_epoch = None
            reference_epoch = gaia_epoch.get("ref_epoch")
            propagated = (
                coordinates_at_epoch(
                    gaia_epoch.get("ra"), gaia_epoch.get("dec"),
                    gaia_epoch.get("pmra"), gaia_epoch.get("pmdec"),
                    reference_epoch, target_epoch,
                ) if target_epoch is not None else None
            )
            if target_epoch is not None and propagated is None:
                st.info(
                    "Gaia DR3 coordinates, proper motion and reference "
                    "epoch are needed for this calculation."
                )
            elif propagated is not None:
                source = f"Gaia DR3 · {float(reference_epoch):.1f}"
                coordinate_rows = [
                    ("RA", sexagesimal(propagated["ra"], True), source, None, None),
                    ("Dec", sexagesimal(propagated["dec"]), source, None, None),
                ]
                st.markdown(
                    '<section class="runa-data-section runa-identification-data runa-kinematic-inputs">'
                    + "".join(data_row(*row) for row in coordinate_rows)
                    + "</section>", unsafe_allow_html=True,
                )

elif tool == "Utilities":
    utility_menu, utility_workspace = st.columns((1, 2), gap="large")
    with utility_menu:
        for utility_name, key_name in (
            ("Coordinates at epoch", "coordinates"),
            ("Spectral type estimation", "spectral_type"),
        ):
            st.button(
                utility_name,
                use_container_width=True,
                type="primary" if utility_tool == utility_name else "secondary",
                key=f"workspace_utility_{key_name}",
                on_click=select_utility,
                args=(utility_name,),
            )

    with utility_workspace:
        with st.container(border=True, key="utilities_selected_card"):
            if utility_tool == "Coordinates at epoch":
                ra_label, ra_fields = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with ra_label:
                    st.markdown('<div class="runa-utility-row-label">RA</div>', unsafe_allow_html=True)
                with ra_fields:
                    ra_parts = st.columns((1, 1, 1.35), gap="small")
                with ra_parts[0]:
                    ra_hour = st.number_input("h", min_value=0, max_value=23, value=0, key="utility_ra_hour")
                with ra_parts[1]:
                    ra_minute = st.number_input("m", min_value=0, max_value=59, value=0, key="utility_ra_minute")
                with ra_parts[2]:
                    ra_second = st.number_input("s", min_value=0.0, max_value=59.9, value=0.0, step=0.1, format="%.1f", key="utility_ra_second")
                input_ra = (ra_hour + ra_minute / 60 + ra_second / 3600) * 15

                dec_label, dec_fields = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with dec_label:
                    st.markdown('<div class="runa-utility-row-label">Dec</div>', unsafe_allow_html=True)
                with dec_fields:
                    dec_parts = st.columns((1, 1, 1.35), gap="small")
                with dec_parts[0]:
                    dec_degree_text = st.text_input("°", value="+00", key="utility_dec_degree_signed", placeholder="+00 or −00")
                with dec_parts[1]:
                    dec_minute = st.number_input("m", min_value=0, max_value=59, value=0, key="utility_dec_minute")
                with dec_parts[2]:
                    dec_second = st.number_input("s", min_value=0.0, max_value=59.9, value=0.0, step=0.1, format="%.1f", key="utility_dec_second")
                try:
                    signed_degrees = dec_degree_text.strip().replace("−", "-")
                    dec_degree = int(signed_degrees)
                    if abs(dec_degree) > 90 or (abs(dec_degree) == 90 and (dec_minute or dec_second)):
                        raise ValueError
                    input_dec = (-1 if signed_degrees.startswith("-") else 1) * (abs(dec_degree) + dec_minute / 60 + dec_second / 3600)
                except ValueError:
                    st.error("Enter Dec degrees from −90 to +90, with a sign if needed.")
                    input_dec = None

                pmra_label, pmra_field = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with pmra_label:
                    st.markdown('<div class="runa-utility-row-label">pm RA (mas/yr)</div>', unsafe_allow_html=True)
                with pmra_field:
                    input_pmra = st.number_input("pm RA (mas/yr)", value=0.0, format="%.3f", key="utility_input_pmra", label_visibility="collapsed")
                pmdec_label, pmdec_field = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with pmdec_label:
                    st.markdown('<div class="runa-utility-row-label">pm Dec (mas/yr)</div>', unsafe_allow_html=True)
                with pmdec_field:
                    input_pmdec = st.number_input("pm Dec (mas/yr)", value=0.0, format="%.3f", key="utility_input_pmdec", label_visibility="collapsed")

                ref_label, ref_fields = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with ref_label:
                    st.markdown('<div class="runa-utility-row-label">Reference epoch</div>', unsafe_allow_html=True)
                reference_catalogue = st.session_state.get("utility_reference_catalogue", "Gaia DR3")
                with ref_fields:
                    reference_controls = st.columns((2.6, 1.1), gap="small", vertical_alignment="center")
                with reference_controls[0]:
                    st.segmented_control(
                        "Reference", ("Gaia DR3", "Gaia DR4", "Manual"),
                        default=reference_catalogue,
                        key="utility_reference_mode",
                        on_change=select_utility_reference,
                        label_visibility="collapsed",
                        width="stretch",
                    )
                with reference_controls[1]:
                    st.text_input(
                        "Reference epoch", key="utility_reference_epoch_text",
                        on_change=save_manual_utility_reference,
                        disabled=reference_catalogue != "Manual",
                        label_visibility="collapsed",
                    )
                try:
                    reference_epoch_value = float(st.session_state["utility_reference_epoch_text"])
                except (TypeError, ValueError):
                    reference_epoch_value = None

                target_label, target_fields = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                with target_label:
                    st.markdown('<div class="runa-utility-row-label">Target epoch</div>', unsafe_allow_html=True)
                with target_fields:
                    target_parts = st.columns((1.5, .32, .32), gap=None)
                with target_parts[0]:
                    target_epoch_text = st.text_input("Target epoch", value="2016.0", key="utility_target_epoch", label_visibility="collapsed")
                with target_parts[1]:
                    st.button("−", key="utility_epoch_down", on_click=adjust_coordinate_epoch, args=("utility_target_epoch", -0.5), use_container_width=True)
                with target_parts[2]:
                    st.button("+", key="utility_epoch_up", on_click=adjust_coordinate_epoch, args=("utility_target_epoch", 0.5), use_container_width=True)
                try:
                    target_epoch = float(target_epoch_text)
                    if not 1800 <= target_epoch <= 2200:
                        raise ValueError
                except ValueError:
                    st.error("Enter a Julian year between 1800.0 and 2200.0, using a decimal point.")
                    target_epoch = None
                if reference_epoch_value is not None and not 1800 <= reference_epoch_value <= 2200:
                    st.error("Enter a Julian reference year between 1800.0 and 2200.0.")
                    reference_epoch_value = None
                propagated = coordinates_at_epoch(input_ra, input_dec, input_pmra, input_pmdec, reference_epoch_value, target_epoch) if target_epoch is not None and reference_epoch_value is not None and input_dec is not None else None
                if propagated is not None:
                    result_label, result_fields = st.columns((1.3, 3), gap="small", vertical_alignment="center")
                    with result_label:
                        st.markdown('<div class="runa-utility-row-label">Results</div>', unsafe_allow_html=True)
                    with result_fields:
                        output_ra = sexagesimal_one_decimal(propagated["ra"], True)
                        output_dec = sexagesimal_one_decimal(propagated["dec"])
                        copy_html = f'''<html><body style="margin:0;background:transparent"><div class="result"><span>RA</span><button type="button" onclick="copyCoordinate('ra')" title="Copy RA" aria-label="Copy RA">{escape(output_ra)}</button><span>Dec</span><button type="button" onclick="copyCoordinate('dec')" title="Copy Dec" aria-label="Copy Dec">{escape(output_dec)}</button></div><script>const values={{ra:{json.dumps(output_ra)},dec:{json.dumps(output_dec)}}};async function copyCoordinate(which){{const value=values[which];try{{await navigator.clipboard.writeText(value)}}catch(e){{const t=document.createElement('textarea');t.value=value;document.body.appendChild(t);t.select();document.execCommand('copy');t.remove()}}}}</script><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#a1a1aa}}.result{{display:grid;grid-template-columns:auto minmax(0,1fr) auto minmax(0,1fr);gap:10px;align-items:center;height:36px}}span{{font-size:16px}}button{{background:none;border:0;padding:0;text-align:left;color:#f4f4f5;font:500 22px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;cursor:copy}}button:hover{{color:#6ee7b7}}</style></body></html>'''
                        components.html(copy_html, height=40)
            else:
                gaia_magnitudes = (star or {}).get("gaia_photometry") or {}
                twomass_magnitudes = (star or {}).get("twomass_photometry") or {}
                default_g = gaia_magnitudes.get("phot_g_mean_mag")
                default_j = twomass_magnitudes.get("j_mag")
                magnitude_columns = st.columns(2, gap="medium")
                with magnitude_columns[0]:
                    g_text = st.text_input(
                        "Gaia G (mag)",
                        value=f"{float(default_g):.3f}" if default_g is not None else "",
                        placeholder="G magnitude",
                        key="utility_spt_g_magnitude",
                    )
                with magnitude_columns[1]:
                    j_text = st.text_input(
                        "2MASS J (mag)",
                        value=f"{float(default_j):.3f}" if default_j is not None else "",
                        placeholder="J magnitude",
                        key="utility_spt_j_magnitude",
                    )
                if g_text.strip() and j_text.strip():
                    try:
                        color, subtype = estimate_subtype(float(g_text), float(j_text))
                    except (ValueError, TypeError) as error:
                        st.caption(str(error))
                    else:
                        st.markdown(
                            '<div class="runa-spectral-result">'
                            '<div class="runa-spectral-estimate">'
                            '<span>Estimated spectral type</span>'
                            f'<strong>M{subtype:.1f}</strong></div></div>',
                            unsafe_allow_html=True,
                        )

elif tool == "Planets":
    with input_card:
        panel_heading("Planets and candidates")
        planet_count_slot = st.empty()

    with result_card:
        panel_heading("Planets and candidates")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            nasa_planets, nasa_state = analysis_result("planets_nasa")
            oec_planets, oec_state = analysis_result("planets_oec")
            nasa_error = nasa_state == "error"
            oec_error = oec_state == "error"
            planets_pending = "pending" in (nasa_state, oec_state)
            planets = merge_planets(nasa_planets or [], oec_planets or [])
            if planets_pending:
                st.markdown(
                    '<div class="runa-loading-status" role="status">'
                    '<span class="runa-loading-spinner" aria-hidden="true"></span>'
                    'Searching planetary catalogues…</div>',
                    unsafe_allow_html=True,
                )
            if not planets_pending and not (nasa_error and oec_error):
                confirmed_count = sum(
                    planet.get("status") == "Confirmed" for planet in planets
                )
                candidate_count = len(planets) - confirmed_count
                citations = discovery_references(planets)
                citation_html = "".join(
                    f'<span class="runa-planet-discovery-reference">{escape(citation)}</span>'
                    for citation in citations
                )
                candidate_html = (
                    f'<span class="runa-planet-count-candidates">'
                    f'({candidate_count} candidate planets)</span>'
                    if candidate_count else ""
                )
                planet_count_slot.markdown(
                    '<div class="runa-planet-count">'
                    '<div class="runa-planet-count-main">'
                    f'<strong class="runa-planet-count-value">{confirmed_count}</strong>'
                    '<span class="runa-planet-count-label">confirmed planets</span></div>'
                    f'{citation_html}{candidate_html}</div>',
                    unsafe_allow_html=True,
                )
            if not planets and not planets_pending:
                if nasa_error and oec_error:
                    st.error("Neither planetary catalogue could be reached right now.")
                else:
                    st.info("No planets or candidates were found for this object.")
            else:
                planet_table(planets)
                orbit_figure = planet_orbit_figure(planets)
                if orbit_figure is not None:
                    st.pyplot(orbit_figure, use_container_width=True)
                    plt.close(orbit_figure)
                    st.caption(
                        "Face-on orbital sketch in AU, using catalogue semimajor axes and eccentricities. "
                        "The star is at one focus; orientation and planet positions are illustrative. "
                        "A dashed circle means eccentricity is unavailable."
                    )
                    st.caption("Green: confirmed · yellow: unconfirmed. Coloured fill: measured mass; "
                               "dark fill: minimum or unavailable mass. Circle area is proportional to planet mass; "
                               "when mass is unavailable, circle radius is proportional to planet radius on a separate scale. "
                               "If both are missing, a fixed small circle is used.")
            if nasa_error:
                st.warning("NASA Exoplanet Archive is temporarily unavailable.")
            if oec_error:
                st.warning("Open Exoplanet Catalogue is temporarily unavailable.")

    with planet_population_card:
        panel_heading("Planet population")
        population, population_state = analysis_result("planets_population") if star is not None else (load_planet_population(), "complete")
        if population_state == "pending":
            population_figure, target_points, population_size = None, [], 0
            st.caption("Loading the planet population…")
        else:
            population_figure, target_points, population_size = planet_population_figure(
                population,
                planets if star is not None else [],
            )
        if population_state == "pending":
            pass
        elif population_figure is None:
            st.info("The bundled exoplanet population is unavailable.")
        else:
            st.pyplot(population_figure, use_container_width=True)
            plt.close(population_figure)
            if star is None:
                st.caption(
                    f"{population_size:,} confirmed planets with a measured mass or M sin i."
                )
            elif target_points:
                st.caption(
                    f"{star['main_id']} is shown over {population_size:,} confirmed planets. "
                    "Open markers denote M sin i."
                )
            else:
                st.caption(
                    f"{population_size:,} confirmed planets with a measured mass or M sin i. "
                    f"No mass or M sin i is available for {star['main_id']}."
                )
            st.caption(
                "Local NASA Exoplanet Archive PSCompPars snapshot retrieved 2026-10-28."
            )

    with tess_card:
        panel_heading("TESS light curve")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            tess_name = tess_identifier(star)
            if tess_name is None:
                st.info("SIMBAD did not return a TIC or TOI identifier for this object.")
            else:
                st.caption(
                    f"SIMBAD identifier: {tess_name['label']}."
                )
                tess_curve, tess_state = analysis_result("planets_tess")
                if tess_state == "pending":
                    st.caption("Querying MAST for a TESS light curve…")
                elif tess_state == "error":
                    st.info("The TESS light-curve query could not be completed.")
                elif tess_curve.get("error"):
                    st.info(tess_curve["error"])
                else:
                    tess_fig = light_curve_figure(tess_curve, star["main_id"])
                    st.pyplot(tess_fig, use_container_width=True)
                    plt.close(tess_fig)
                    st.caption(
                        "Relative PDCSAP flux from one recent TESS SPOC sector via MAST. "
                        "The y-axis follows the median and robust scatter, so isolated "
                        "peaks do not dominate the view. This is not a transit fit."
                    )

    st.divider()
    st.caption(
        "Sources: NASA Exoplanet Archive (PSCompPars) and Open Exoplanet Catalogue. "
        "NASA values take precedence when an object appears in both catalogues."
    )

elif tool == "Multiplicity":
    def render_wds_report():
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            wds_pair, wds_state = analysis_result("multiplicity_wds")
            if wds_state == "pending":
                st.caption("Searching the WDS catalogue…")
            elif wds_state == "error":
                st.warning("The WDS search could not be completed. Try again later.")
            else:
                if wds_pair:
                    separation_col, year_col = st.columns(2)
                    with separation_col:
                        with st.container(key="wds_closest_metric"):
                            st.metric(
                                "Closest WDS pair",
                                f'{wds_pair["separation_arcsec"]:.1f} arcsec',
                            )
                    year = wds_pair.get("last_observation_year")
                    year_col.metric("Last WDS observation", str(year) if year else "—")
                    discoverer = " ".join(wds_pair["discoverer"].split()) or "—"
                    st.markdown(
                        '<div style="color:#a1a1aa;font-size:0.9rem;line-height:1.8">'
                        f'<span style="color:#ffffff">WDS {escape(wds_pair["wds"])}</span>'
                        ' &nbsp; · &nbsp; Component <span style="color:#ffffff;font-weight:600">'
                        f'{escape(wds_pair["component"])}</span>'
                        ' &nbsp; · &nbsp; Discoverer <span style="color:#86cfa5;font-weight:600">'
                        f'{escape(discoverer)}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No usable WDS pair involving component A was returned for this object.")

    query_error = None
    answer = None
    if star is not None and star["gaia_id"]:
        answer, criteria_state = analysis_result("multiplicity_criteria")
        if criteria_state == "error":
            query_error = "The Gaia query failed. Please try again later."

    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in Identification first.")
        elif not star["gaia_id"]:
            st.write("No Gaia DR3 identifier in SIMBAD.")
        else:
            st.markdown(
                f'<div class="runa-gaia-id">Gaia DR3 identifier: '
                f'<strong>{escape(str(star["gaia_id"]))}</strong></div>',
                unsafe_allow_html=True,
            )
            try:
                gaia_counts = gaia_source_photometry(star["gaia_id"])
            except GaiaPhotometryUnavailable as error:
                gaia_counts = {}
                st.warning(str(error))

            def formatted_count(field):
                value = gaia_counts.get(field)
                try:
                    return f"{int(value):,}" if value is not None else None
                except (OverflowError, TypeError, ValueError):
                    return None

            n_obs_al = formatted_count("astrometric_n_obs_al")
            photometric_counts = " / ".join(
                formatted_count(field) or "Unavailable"
                for field in ("phot_g_n_obs", "phot_bp_n_obs", "phot_rp_n_obs")
            )
            observation_rows = (
                (
                    formatted_count("astrometric_matched_transits"),
                    "Gaia field-of-view passages used to measure <strong>astrometric</strong> <strong>position</strong>",
                ),
                (n_obs_al, "observations used in the <strong>astrometric</strong> solution"),
                (
                    photometric_counts,
                    "transits contributing to <strong>mean flux</strong> in G / BP / RP",
                ),
                (
                    formatted_count("rv_nb_transits"),
                    "observing epochs used to calculate the <strong>radial velocity</strong>",
                ),
            )
            rows_html = "".join(
                '<div class="runa-observation-row">'
                f'<strong class="runa-observation-number">{escape(value or "Unavailable")}</strong>'
                f'<span>{description}</span></div>'
                for value, description in observation_rows
            )
            st.markdown(
                '<div class="runa-observation-list" aria-label="Gaia observation counts">'
                + rows_html
                + "</div>",
                unsafe_allow_html=True,
            )
    with nearby_card:
        panel_heading("Nearby Gaia sources")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        elif not star["gaia_id"]:
            st.write("A Gaia DR3 ID is needed to query nearby sources.")
        else:
            radius_arcsec = st.selectbox(
                "Nearby-source radius",
                [5, 10, 30, 60, 180, 300],
                index=4,
                format_func=radius_label,
                key=f"multiplicity_radius_{star['main_id']}",
            )
            gaia_ra = star["gaia_photometry"].get("ra", star["ra"])
            gaia_dec = star["gaia_photometry"].get("dec", star["dec"])
            if radius_arcsec == 180:
                nearby, nearby_state = analysis_result("multiplicity_neighbours")
            else:
                try:
                    nearby = nearby_gaia_sources_v3(gaia_ra, gaia_dec, radius_arcsec)
                    nearby_state = "complete"
                except Exception:
                    nearby, nearby_state = None, "error"
            if nearby_state == "pending":
                st.caption("Loading nearby Gaia sources…")
            elif nearby_state == "error":
                st.error("The nearby Gaia query could not be completed.")
            else:
                nearby = prepare_neighbours(nearby, gaia_ra, gaia_dec)
                if nearby:
                    local_fig = local_proper_motion_plot(
                        nearby, radius_arcsec, star.get("gaia_id")
                    )
                    if local_fig is not None:
                        st.pyplot(local_fig, use_container_width=True)
                        plt.close(local_fig)
                    with st.expander("Gaia source table", expanded=False):
                        neighbour_table(nearby, star.get("gaia_id"))
                    st.caption(
                        f"{len(nearby)} Gaia source(s) within "
                        f"{radius_label(radius_arcsec)}. "
                        "No parallax filter applied."
                    )
                else:
                    st.info("No Gaia sources were found in this radius.")

    with epoch_card:
        panel_heading("Gaia epoch photometry")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        elif not star.get("gaia_id"):
            st.info("A Gaia DR3 ID is needed to retrieve the time series.")
        else:
            epoch_data, epoch_state = analysis_result("multiplicity_epoch")
            if epoch_state == "pending":
                st.caption("Loading Gaia G, BP and RP time series…")
            elif epoch_state == "error":
                st.warning("The Gaia epoch-photometry query could not be completed. Try again later.")
            else:
                if epoch_data.empty:
                    st.info("No published Gaia DR3 epoch photometry was found for this source. "
                            "Time series are available for only a subset of Gaia sources.")
                else:
                    epoch_fig = gaia_epoch_figure(epoch_data)
                    st.pyplot(epoch_fig, use_container_width=True)
                    plt.close(epoch_fig)
                    st.caption("Crosses are rejected by Gaia variability processing")
                    st.download_button("Download Gaia time series", epoch_data.to_csv(index=False),
                                       file_name=f"gaia_dr3_{star['gaia_id']}_epoch_photometry.csv",
                                       mime="text/csv", key="download_gaia_epoch_photometry")

    values = None
    catalogue_multiplicity = reported_multiplicity
    with prediction_card:
        st.markdown(
            '<div class="runa-prediction-heading">'
            '<span class="runa-panel-heading">Multiplicity prediction</span>'
            '<span class="runa-learning-badge">Supervised learning</span></div>',
            unsafe_allow_html=True,
        )
        if query_error:
            st.error(query_error)
        elif star is None:
            st.write("Choose a star in **Identification** first.")
        elif not star["gaia_id"]:
            st.write("A Gaia DR3 ID is needed to score this source.")
        elif answer is None:
            st.caption("Loading Gaia multiplicity criteria…")
        else:
            values, _ = answer
            scoring, scoring_state = analysis_result("multiplicity_model")
            if scoring_state == "error":
                st.warning("The multiplicity model could not score this source.")
            else:
                multiplicity_ml_result(values, scoring)
        with st.expander("Multiplicity criteria", expanded=False):
            st.caption("Based on Cifuentes et al. (2025), Table 3.")
            if query_error:
                st.error(query_error)
            elif star is None:
                st.write("Choose a star in **Identification** first.")
            elif not star["gaia_id"]:
                st.write("A Gaia DR3 ID is needed for these indicators.")
            elif answer is None:
                st.caption("Loading Gaia multiplicity criteria…")
            else:
                values, criteria = answer
                strong_indicators = any(criteria[number] for number in (1, 2, 3, 4, 5, 7))
                if strong_indicators:
                    st.write("**One or more multiplicity indicators are met.**")
                elif criteria[6]:
                    st.markdown(
                        '<p class="runa-criteria-caution">Only the supporting '
                        'duplicated-source flag is met.</p>',
                        unsafe_allow_html=True,
                    )
                elif all(result is None for result in criteria.values()):
                    st.markdown(
                        '<p class="runa-criteria-caution">Gaia has no values '
                        'for these criteria.</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p class="runa-criteria-caution">No available '
                        'criterion is met.</p>',
                        unsafe_allow_html=True,
                    )
                multiplicity_table(values, criteria)

    with reported_card:
        panel_heading("Multiplicity reports")
        if catalogue_multiplicity and catalogue_multiplicity["type"].startswith("Multiple"):
            sb = catalogue_multiplicity["sb"]
            if sb:
                if "SB" in sb:
                    classification = "spectroscopic binary"
                elif sb.startswith("ST"):
                    classification = "spectroscopic triple"
                else:
                    classification = "spectroscopic multiple"
                first_report = catalogue_multiplicity["sb_first_report"]
                bibcode = catalogue_multiplicity["sb_first_bibcode"]
                if first_report and bibcode:
                    attribution = (
                        ' by <a href="https://ui.adsabs.harvard.edu/abs/'
                        f'{escape(bibcode)}/abstract">{escape(first_report)}</a>'
                    )
                elif first_report:
                    attribution = f" by {escape(first_report)}"
                else:
                    attribution = " in the Cifuentes+25 catalogue"
                st.markdown(
                    'Reported as <strong style="color:#f4cf75">'
                    f'{classification}</strong> ({escape(sb)}){attribution}.',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span style='color:#f4cf75'>No</span> "
                    "spectroscopic-binary classification reported.",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<strong style='color:#f4cf75'>Known multiple</strong> "
                "in Cifuentes et al. (2025)",
                unsafe_allow_html=True,
            )
        render_wds_report()

    with archive_card:
        panel_heading("Spectroscopic archives")
        st.caption("Searching public spectroscopic archives automatically.")
        st.markdown(
            '<div class="runa-caution-note" role="note">'
            '<span class="runa-caution-icon" aria-hidden="true">!</span>'
            '<span>Archive coverage describes available observations, not whether their '
            'radial velocities have been analysed and not whether the star is single.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            gaia_astrometry = star.get("gaia_photometry") or {}
            archive_ra = gaia_astrometry.get("ra", star.get("ra"))
            archive_dec = gaia_astrometry.get("dec", star.get("dec"))
            archive_result, archive_state = analysis_result("multiplicity_archives")
            if archive_state == "pending":
                st.caption("Searching archival spectroscopy…")
            elif archive_state == "error":
                st.warning("The archive search could not be completed. Try again later.")
            else:
                st.markdown("**Spectroscopic archives**")
                if archive_result is not None:
                    archive_rows = archive_result["rows"]
                    st.dataframe(
                        archive_result["sources"],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Link": st.column_config.LinkColumn(
                                "Archive", display_text="Open"
                            ),
                        },
                    )
                    st.caption(
                        "ESO, CARMENES DR1 and NOT/FIES are queried automatically. "
                        "HARPS-N/TNG, WHT/ING and the full Calar Alto archive are linked "
                        "for manual review because RUNA has no stable documented query for them."
                    )
                    if not archive_rows:
                        st.info(
                            "No positionally matched spectra were found in the automated "
                            "sources. This does not exclude observations outside their stated "
                            "coverage or in archives that require a manual search."
                        )
                    else:
                        archive_summary = summarise_eso_spectroscopy(archive_rows)
                        files_col, epochs_col, public_col = st.columns(3)
                        files_col.metric("Science files", archive_summary["files"])
                        epochs_col.metric("Observing dates", archive_summary["epochs"])
                        matched_archives = len({row.get("archive") for row in archive_rows})
                        public_col.metric("Archives matched", matched_archives)
                        if archive_summary["first"] and archive_summary["last"]:
                            st.caption(
                                f"Observed from {archive_summary['first']} to "
                                f"{archive_summary['last']} · "
                                f"{archive_summary['proprietary']} file(s) not yet public"
                            )
                        st.dataframe(
                            archive_summary["instruments"],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Files": st.column_config.NumberColumn(format="%d"),
                                "UTC dates": st.column_config.NumberColumn(format="%d"),
                                "Exposure h": st.column_config.NumberColumn(format="%.2f"),
                                "Public": st.column_config.NumberColumn(format="%d"),
                            },
                        )
                        with st.expander("Observing programmes", expanded=False):
                            st.dataframe(
                                archive_summary["programmes"],
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.markdown(
                                "Use ESO programme IDs above to check refereed use in the "
                                f"[ESO Telescope Bibliography]({ESO_TELBIB_URL})."
                            )
                        with st.expander("Matched raw files", expanded=False):
                            st.dataframe(
                                eso_file_details(archive_rows),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Exposure s": st.column_config.NumberColumn(format="%.1f"),
                                    "Offset arcsec": st.column_config.NumberColumn(format="%.2f"),
                                    "S/N": st.column_config.NumberColumn(format="%.1f"),
                                    "Archive": st.column_config.LinkColumn(
                                        "Archive", display_text="Open"
                                    ),
                                },
                            )
                            st.caption(
                                "ESO raw metadata do not provide a uniform resolving power "
                                "or wavelength range for every exposure. Technique and mode "
                                "are shown as recorded by the archive."
                            )
                        st.caption(
                            "Candidate positions are propagated from Gaia DR3 and matched "
                            "within 10 arcsec. "
                            "High-resolution RV marks instrument capability, not an existing "
                            "RV time-series analysis."
                        )

    with diagnostics_card:
        panel_heading("Gaia multiplicity diagnostics")
        target_rvs_plot = None
        if star is None or not star.get("gaia_id") or values is None:
            st.info("A Gaia DR3 source with multiplicity criteria is needed for this plot.")
        else:
            rvs_reference, rvs_state = analysis_result("multiplicity_rvs")
            if rvs_state == "pending":
                st.caption("Loading Gaia stars within 10 pc…")
            elif rvs_state == "error":
                st.error("The Gaia RVS comparison sample could not be loaded.")
            else:
                rvs_reference = [
                    source for source in rvs_reference
                    if source.get("source_id") != str(star["gaia_id"])
                ]
                target_rvs_plot = rvs_ruwe_plot(
                    rvs_reference, values, star["main_id"]
                )
        if target_rvs_plot is not None:
            st.pyplot(target_rvs_plot, use_container_width=True)
            plt.close(target_rvs_plot)
            st.caption(
                "Light-grey points: Gaia DR3 sources within 10 pc with "
                "the three RV measurements available. No parallax S/N cut "
                "is applied. Dashed lines mark RUWE = 2 and RVS = 2."
            )
        elif star is not None and star.get("gaia_id") and values is not None:
            st.info(
                "An RVS score needs the Gaia RV p-value, renormalised GOF "
                "and at least one transit count."
            )

    st.divider()
    st.caption("Method: the seven Gaia DR3 statistical criteria in Cifuentes et al. (2025), Table 3. Missing values remain unavailable.")

else:
    st.error("Choose a RUNA tool from the navigation.")
