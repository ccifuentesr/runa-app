"""Web interface for RUNA."""

from html import escape
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
from gaia_neighbours import (
    local_proper_motion_plot,
    prepare_neighbours,
    query_gaia_neighbours,
)
from multiplicity import (
    check_multiplicity,
    query_rvs_reference,
    rvs_ruwe_plot,
)
from parameters import (
    calculate_parameters,
    mass_from_luminosity_eker,
    metallicity_from_gaia_wise,
    photometric_distance_from_gj,
)
from photometry import (
    build_sed_points,
    fetch_catalogue_photometry,
    load_filter_curve,
    load_gaia_reference_sample,
    star_cmd_position,
)
from planets import lookup_planets
from position_plots import polar_position_plot
from predict_star import estimate_subtype
from train_model import load_stars
from vosa_input import build_vosa_input
from cifuentes20 import load_table6
from local_volume import local_volume_status
from published_parameters import published_stellar_parameters


def data_row(label, value, source=None, error=None, unit=None):
    """Build one compact row for the identification panel."""
    label_html = escape(str(label)).replace("M_G", "M<sub>G</sub>")
    label_html = label_html.replace("M_J", "M<sub>J</sub>")
    label_html = label_html.replace("T_eff", "T<sub>eff</sub>")
    error_html = (
        f'<span class="runa-error">({escape(error)})</span>' if error else ""
    )
    unit_html = escape(unit) if unit else ""
    source_html = escape(source) if source else ""
    return (
        '<div class="runa-data-row">'
        f'<span class="runa-data-label">{label_html}</span>'
        '<span class="runa-data-value">'
        f'{escape(str(value))}{error_html}</span>'
        f'<span class="runa-data-unit">{unit_html}</span>'
        f'<span class="runa-data-source">{source_html}</span></div>'
    )


def data_section(title, rows):
    """Show a titled group of identification values."""
    content = "".join(data_row(*row) for row in rows if row[1] is not None)
    if content:
        title_html = escape(str(title)).replace("M_G", "M<sub>G</sub>")
        title_html = title_html.replace("M_J", "M<sub>J</sub>")
        st.markdown(
            f'<section class="runa-data-section"><div class="runa-section-title">'
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
        source_html = escape(source) if source else "Source unavailable"
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


def cifuentes20_table(rows):
    """Render Cifuentes et al. (2020), Table 6 in RUNA's data-table style."""
    header = (
        '<section class="runa-cifuentes-table">'
        '<div class="runa-cifuentes-row runa-cifuentes-head">'
        '<span>Spectral type</span><span>BC<sub>G</sub><small>mag</small></span>'
        '<span>BC<sub>J</sub><small>mag</small></span>'
        '<span>L<small>10<sup>−4</sup> L<sub>☉</sub></small></span>'
        '<span>T<sub>eff</sub><small>K</small></span><span>R<small>R<sub>☉</sub></small></span>'
        '<span>M<small>M<sub>☉</sub></small></span><span>N</span></div>'
    )
    body = []
    for row in rows:
        value = lambda name: escape(row[name])
        pair = lambda name, error: (
            f'<span>{value(name)} <small>({value(error)})</small></span>'
        )
        body.append(
            '<div class="runa-cifuentes-row">'
            f'<span class="runa-cifuentes-type">{value("sp_type")}</span>'
            f'{pair("bc_g", "bc_g_error")}{pair("bc_j", "bc_j_error")}'
            f'{pair("lbol", "lbol_error")}{pair("teff", "teff_error")}'
            f'{pair("radius", "radius_error")}{pair("mass", "mass_error")}'
            f'<span>{value("n")}</span></div>'
        )
    st.markdown(header + "".join(body) + "</section>", unsafe_allow_html=True)


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
        rows.append(
            f'<div class="runa-photometry-row{catalogue_class}">'
            f'<span class="runa-photometry-catalogue">{escape(band["catalogue"])}</span>'
            f'<span class="runa-band-name" style="--band-colour: {colour}">'
            f'{escape(band["label"])}</span>'
            f'<span class="runa-photometry-wavelength">{escape(wavelength_text)}</span>'
            f'<span>{escape(value_text)}</span>'
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
    rows = [
        '<section class="runa-neighbour-table">'
        '<div class="runa-neighbour-row runa-neighbour-head">'
        '<span>Gaia DR3 ID</span>'
        '<span class="runa-neighbour-heading runa-radius-head">'
        'r<small>arcsec</small></span>'
        '<span class="runa-neighbour-heading">Parallax<small>mas</small></span>'
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


def planet_table(planets):
    """Show the archive planets in a compact aligned table."""
    rows = [
        '<section class="runa-planet-table">'
        '<div class="runa-planet-row runa-planet-head">'
        '<span>Planet</span><span>Discovery</span>'
        '<span>Period<small>days</small></span>'
        '<span>Radius<small>R⊕</small></span>'
        '<span>Mass<small>M⊕</small></span>'
        '<span>Year</span></div>'
    ]
    for planet in planets:
        period = planet.get("pl_orbper")
        radius = planet.get("pl_rade")
        mass = planet.get("pl_masse")
        year = planet.get("disc_year")
        period_text = f"{float(period):.2f}" if period is not None else "—"
        radius_text = f"{float(radius):.2f}" if radius is not None else "—"
        mass_text = f"{float(mass):.2f}" if mass is not None else "—"
        year_text = str(year) if year is not None else "—"
        rows.append(
            '<div class="runa-planet-row">'
            f'<span>{escape(str(planet.get("pl_name") or "—"))}</span>'
            f'<span>{escape(str(planet.get("discoverymethod") or "—"))}</span>'
            f'<span>{escape(period_text)}</span>'
            f'<span>{escape(radius_text)}</span>'
            f'<span>{escape(mass_text)}</span>'
            f'<span>{escape(year_text)}</span></div>'
        )
    rows.append("</section>")
    st.markdown("".join(rows), unsafe_allow_html=True)


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



def magnitude_value(value):
    """Format an apparent magnitude for a compact metric."""
    if value is None or np.ma.is_masked(value):
        return "—"
    return f"{float(value):.3f} mag"

def object_panel_title(star):
    """Show the selected object without making it compete with section titles."""
    label = star["main_id"] if star is not None else "Input"
    st.markdown(
        f'<div class="runa-panel-object"><span>Object</span>{escape(str(label))}</div>',
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
def gaia_reference_sample():
    """Keep the Gaia comparison sample for one day."""
    return load_gaia_reference_sample()


@st.cache_data(ttl=86400, show_spinner=False)
def nearby_gaia_sources_v2(ra, dec, radius_arcsec):
    """Keep a small Gaia cone query for one day."""
    return query_gaia_neighbours(ra, dec, radius_arcsec)


@st.cache_data(ttl=86400, show_spinner=False)
def rvs_reference_sample():
    """Keep the ten-parsec Gaia RV comparison sample for one day."""
    return query_rvs_reference()


@st.cache_data(ttl=86400, show_spinner=False)
def exoplanet_archive_planets(query_name, main_id, identifiers):
    """Keep one NASA Exoplanet Archive query for one day."""
    return lookup_planets(query_name, main_id, identifiers)


st.set_page_config(page_title="RUNA · Stellar characterisation", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 5rem;}
    .runa-brand {font-size: 1rem; color: #a1a1aa; margin-bottom: 2.8rem;}
    .runa-brand strong {color: #fafafa; font-weight: 600;}
    .runa-brand span {color: #6ee7b7;}
    .runa-eyebrow {color: #a1a1aa; font-size: .75rem; letter-spacing: .1em; text-transform: uppercase;}
    .runa-title {color: #fafafa; font-size: 3.25rem; font-weight: 700;
                 letter-spacing: -.035em; line-height: 1; margin-top: .95rem;}
    .runa-version {color: #a1a1aa; font-size: .9rem; margin: .42rem 0 1.1rem;}
    .runa-tools {border-bottom: 1px solid #27272a; margin: 1.1rem 0 2rem; padding-bottom: .75rem;}
    .runa-tools span {font-size: 1.02rem; color: #92929d;}
    .runa-note {color: #a1a1aa; font-size: .9rem;}
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
    .runa-spectral-colour {color: #c7c7cf; font-size: 1.08rem; margin: .9rem 0 .7rem;}
    .runa-spectral-colour strong {color: #f4f4f5; font-size: 1.16rem; font-weight: 650;}
    .runa-large-metric {padding: .15rem 0 .8rem;}
    .runa-large-metric-label {color: #a1a1aa; font-size: .82rem; margin-bottom: .2rem;}
    .runa-large-metric-label sub {font-size: .72em; vertical-align: sub;}
    .runa-large-metric-value {color: #f4f4f5; font-size: 1.7rem; font-weight: 620;
                              line-height: 1.18;}
    .runa-large-metric-error {color: #92929d; font-size: .82rem; margin-top: .22rem;}
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
    .runa-preparing {color: #f0aaa2; background: #2a191a; border: 1px solid #6a3638;
                     border-radius: 10px; padding: .55rem .75rem; margin: 0 0 1rem;
                     font-size: .88rem; font-weight: 560;}
    .runa-panel-object {color: #d4d4d8; font-size: 1.08rem; font-weight: 590;
                        margin: .05rem 0 1rem; letter-spacing: -.01em;}
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
    .runa-parameter-section {overflow: hidden; border: 1px solid #2d2d33;
                             border-radius: 12px; margin: 0 0 .7rem;
                             background: #121215;}
    .runa-parameter-kind {color: #8b8b95; font-size: .68rem; font-weight: 680;
                          letter-spacing: .09em; text-transform: uppercase;
                          padding: .7rem .85rem .35rem;}
    .runa-parameter-estimated .runa-parameter-kind {color: #6ee7b7;}
    .runa-parameter-row {padding: .55rem .85rem .7rem;}
    .runa-parameter-row + .runa-parameter-row {border-top: 1px solid #28282e;}
    .runa-parameter-label {color: #9a9aa4; font-size: .78rem; margin-bottom: .16rem;}
    .runa-parameter-value {color: #f1f1f3; font-size: 1.12rem; font-weight: 620;
                           line-height: 1.25; font-variant-numeric: tabular-nums;}
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
    .runa-neighbour-heading small {display: block; color: #a4a4ad; font-size: .68rem;
                                   font-weight: 430; letter-spacing: 0; text-transform: none;
                                   margin-top: .12rem;}
    .runa-neighbour-row > span:first-child {white-space: nowrap; font-size: .78rem;}
    .runa-neighbour-target {background: #1b2a24; color: #f4f4f5;}
    .runa-neighbour-target small {color: #a7c8b7;}
    .runa-neighbour-badge {display: inline-flex; margin-left: .45rem; padding: .12rem .42rem;
                           border: 1px solid #3c8d6a; border-radius: 999px; color: #6ee7b7;
                           font-size: .7rem; font-weight: 650; vertical-align: middle;}
    .runa-planet-table {overflow: hidden; border: 1px solid #2d2d33;
                        border-radius: 12px; margin: .7rem 0 .85rem;
                        background: #121215;}
    .runa-planet-row {display: grid; grid-template-columns: minmax(6rem, 1.5fr)
                      minmax(5rem, 1.1fr) repeat(4, minmax(3rem, .65fr)); gap: .45rem;
                      align-items: center; padding: .65rem .7rem;
                      color: #ededf0; font-size: .8rem; overflow-wrap: anywhere;}
    .runa-planet-row + .runa-planet-row {border-top: 1px solid #28282e;}
    .runa-planet-head {color: #8b8b95; font-size: .74rem; font-weight: 650;
                       letter-spacing: .09em; text-transform: uppercase;}
    .runa-planet-row small {display: block; color: #a4a4ad; font-size: .68rem;
                            font-weight: 430; letter-spacing: 0; text-transform: none;
                            margin-top: .12rem;}
    .runa-planet-row > span:first-child {font-weight: 560; white-space: nowrap;}
    .runa-catalogue-note {color: #777780; font-size: .78rem; margin: -.2rem 0 1rem;}
    .runa-cifuentes-table {overflow-x: auto; border: 1px solid #2d2d33;
                            border-radius: 12px; margin: .7rem 0 .85rem; background: #121215;}
    .runa-cifuentes-row {display: grid; grid-template-columns: minmax(5.6rem, .9fr)
                          repeat(2, minmax(6.4rem, 1fr)) minmax(5.7rem, .9fr)
                          minmax(5.5rem, .85fr) repeat(2, minmax(5rem, .8fr)) minmax(2rem, .3fr);
                          gap: .55rem; align-items: center; min-width: 790px;
                          padding: .62rem .85rem; color: #ededf0; font-size: .84rem;
                          font-variant-numeric: tabular-nums;}
    .runa-cifuentes-row + .runa-cifuentes-row {border-top: 1px solid #28282e;}
    .runa-cifuentes-head {color: #8b8b95; font-size: .7rem; font-weight: 650;
                           letter-spacing: .07em; text-transform: uppercase;}
    .runa-cifuentes-row small {display: block; color: #92929d; font-size: .7rem;
                                font-weight: 430; letter-spacing: 0; text-transform: none;
                                margin-top: .1rem;}
    .runa-cifuentes-row:not(.runa-cifuentes-head) small {display: inline; margin-left: .28rem;}
    .runa-cifuentes-type {color: #f3f3f5; font-weight: 590;}
    .runa-photometry-table {overflow: hidden; border: 1px solid #2d2d33;
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
    @media (max-width: 700px) {
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
    </style>
    <div class="runa-brand"><strong>C. Cifuentes</strong><span>_</span> / Projects / RUNA</div>
    <div class="runa-eyebrow">Stellar characterisation</div>
    """,
    unsafe_allow_html=True,
)

if "star" not in st.session_state:
    st.session_state["star"] = None
if "multiplicity_result" not in st.session_state:
    st.session_state["multiplicity_result"] = None
if "show_gaia_parameters" not in st.session_state:
    st.session_state["show_gaia_parameters"] = False
if "catalogue_photometry" not in st.session_state:
    st.session_state["catalogue_photometry"] = None
if "banyan_result" not in st.session_state:
    st.session_state["banyan_result"] = None
if "tess_light_curve" not in st.session_state:
    st.session_state["tess_light_curve"] = None

available_tools = [
    "Identification", "Photometry", "Parameters", "Kinematics", "Multiplicity", "Planets",
]
current_tool = st.session_state.get("runa_tool", "Identification")
if current_tool not in [*available_tools, "Data"]:
    current_tool = "Identification"
if "runa_tool" not in st.session_state:
    st.session_state["runa_tool"] = current_tool
if "runa_main_nav" not in st.session_state:
    st.session_state["runa_main_nav"] = (
        current_tool if current_tool in available_tools else None
    )


def select_main_tool():
    st.session_state["runa_tool"] = st.session_state["runa_main_nav"]


def select_data_tool():
    st.session_state["runa_tool"] = "Data"
    st.session_state["runa_main_nav"] = None


header_title, header_search = st.columns([1.15, 1], gap="large")
with header_title:
    st.markdown(
        '<div class="runa-title">RUNA</div>'
        '<div class="runa-version">v20261022</div>',
        unsafe_allow_html=True,
    )
    st.write("Explore stellar properties from measured data.")
with header_search:
    search_name, search_button = st.columns([2.2, 0.75], gap="small")
    with search_name:
        top_name = st.text_input(
            "Object name", placeholder="e.g. Proxima Cen",
            label_visibility="collapsed", key="runa_object_name",
        )
    with search_button:
        top_submitted = st.button("RUN", use_container_width=True)

menu_col, data_col = st.columns([7, 0.85], gap="small")
with menu_col:
    st.segmented_control(
        "RUNA tool", available_tools, label_visibility="collapsed",
        key="runa_main_nav", on_change=select_main_tool,
    )
with data_col:
    st.button(
        "Data", use_container_width=True,
        type="secondary",
        on_click=select_data_tool,
    )

tool = st.session_state["runa_tool"]
star = st.session_state["star"]
search_error = None
if top_submitted:
    st.session_state["star"] = None
    st.session_state["multiplicity_result"] = None
    st.session_state["banyan_result"] = None
    st.session_state["show_gaia_parameters"] = False
    st.session_state["catalogue_photometry"] = None
    st.session_state["tess_light_curve"] = None
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
            data["gaia_photometry"] = lookup_gaia_photometry(data["gaia_id"])
            data["twomass_photometry"] = lookup_2mass_j(data["twomass_id"])
            data["johnson_photometry"] = lookup_johnson_v(data["main_id"])
            data["survey_image"] = fetch_survey_image(data["ra"], data["dec"])
            st.session_state["star"] = data
            st.rerun()

st.markdown(
    """
    <div class="runa-tools" aria-label="RUNA project details">
      <span>Python · Streamlit · Astropy · astroquery · scikit-learn · Matplotlib · ADQL/SQL</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if search_error:
    st.error(search_error)

if tool == "Parameters":
    st.markdown(
        '<div class="runa-preparing">Section in preparation · estimates may change.</div>',
        unsafe_allow_html=True,
    )

if tool in ("Identification", "Data"):
    input_card = None
    parameters_card = None
    result_card = st.container(border=True) if tool == "Identification" else None
else:
    column_widths = [0.62, 1.38] if tool == "Parameters" else [1, 1]
    inputs, results = st.columns(column_widths, gap="small")
    input_card = inputs.container(border=True)
    parameters_card = inputs.container(border=True) if tool == "Multiplicity" else None
    spectral_type_card = (
        inputs.container(border=True) if tool == "Photometry" else None
    )
    additional_catalogue_card = (
        inputs.container(border=True) if tool == "Photometry" else None
    )
    result_card = results.container(
        border=tool not in ("Photometry", "Parameters")
    )

if tool == "Data":
    st.markdown("### Reference data")
    st.caption(
        "Average astrophysical parameters for K5 V–L2.0 dwarfs from "
        "Cifuentes et al. (2020), Table 6."
    )
    cifuentes20_table(load_table6())
    st.caption(
        "Uncertainties are shown in grey parentheses. L is reported in "
        "10⁻⁴ L☉; N is the number of objects in each spectral-type bin."
    )

elif tool == "Identification":
    with result_card:
        if star is None:
            st.write("Enter a name and select **RUN**.")
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
                survey_image = star.get("survey_image")
                if survey_image is not None:
                    st.write(f"#### {survey_image['catalogue']} image")
                    st.image(survey_image["data"])
                    st.caption(survey_image["caption"])
                else:
                    st.write("#### Survey image")
                    st.info("No SDSS or DSS2 image could be loaded.")

                polar_fig = polar_position_plot(star)
                if polar_fig is not None:
                    st.pyplot(polar_fig, use_container_width=True)
                    plt.close(polar_fig)
                    st.caption("RA gives the polar angle; the radius is the distance from the Sun in pc.")
                    volume = local_volume_status(star)
                    distance_text = f'{volume["distance_pc"]:.3f} pc'
                    if volume["inside_10pc"]:
                        sample = volume["sample_match"]
                        sample_text = "Matched in Table C1." if sample else (
                            "No exact Table C1 match was found."
                        )
                        st.markdown(
                            '<div class="runa-volume">'
                            '<div class="runa-volume-title">Within 10 pc</div>'
                            f'<div class="runa-volume-text">Distance: {distance_text}. '
                            f'{sample_text} See the '
                            '<a href="https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.201R/abstract" '
                            'target="_blank">Reylé et al. (2021) 10 pc sample</a>.</div></div>',
                            unsafe_allow_html=True,
                        )
                        multiple = volume["multiple_match"]
                        if multiple:
                            multiple_name = escape(
                                multiple.get("Name") or multiple.get("SimbadName") or "this object"
                            )
                            component = escape(str(multiple.get("Comp") or "").strip())
                            component_text = f" · component {component}" if component else ""
                            st.markdown(
                                '<div class="runa-volume runa-volume-multiple">'
                                '<div class="runa-volume-title">Known multiple system</div>'
                                f'<div class="runa-volume-text">{multiple_name}{component_text} is listed '
                                'among the individual components of multiple systems within 10 pc in '
                                '<a href="https://ui.adsabs.harvard.edu/abs/2026MNRAS.549ag838G/abstract" '
                                'target="_blank">González-Payo et al. (2026)</a> '
                                '(<a href="https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/549/G838" '
                                'target="_blank">J/MNRAS/549/G838</a>).</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(
                                "No match was found in Table C2 of known multiple-system components. "
                                "This does not by itself establish that the star is single."
                            )
                    else:
                        st.markdown(
                            '<div class="runa-volume">'
                            '<div class="runa-volume-title">Outside the 10 pc volume</div>'
                            f'<div class="runa-volume-text">Distance: {distance_text} from the current '
                            'parallax.</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("A positive parallax and coordinates are needed for the polar view.")

            identifier_rows = [
                ("SIMBAD main ID", star["main_id"], "primary name"),
                *catalogue_rows(star),
            ]
            with result_right:
                data_section("Identifiers", identifier_rows)

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
            with result_right:
                data_section(
                    "Astrometry",
                    [
                        ("Right ascension", sexagesimal(star["ra"], True),
                         coordinate_source, ra_error, "mas"),
                        ("Declination", sexagesimal(star["dec"]),
                         coordinate_source, dec_error, "mas"),
                        ("Parallax", parallax_text, parallax_source,
                         parallax_error, "mas"),
                        ("Distance", distance, parallax_source + " · derived",
                         distance_error, "pc"),
                        ("Proper motion RA", f"{star['pmra']:.2f}"
                         if star["pmra"] is not None else None,
                         motion_source, pmra_error, "mas/yr"),
                        ("Proper motion Dec", f"{star['pmdec']:.2f}"
                         if star["pmdec"] is not None else None,
                         motion_source, pmdec_error, "mas/yr"),
                        ("Radial velocity", radial_velocity,
                         source_name(star.get("rvz_bibcode")), radial_velocity_error,
                         "km/s"),
                    ],
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
                         f"2MASS · Q={j_quality}" if j_quality else "2MASS",
                         f"{j_error:.3f}" if j_error is not None else None, "mag"),
                        ("Johnson V", f"{v_mag:.3f}" if v_mag is not None else None,
                         "UCAC4", f"{v_error:.3f}" if v_error is not None else None, "mag"),
                    ],
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
                )

            with result_left:
                st.markdown(
                    '<p class="runa-catalogue-note">Catalogue tags describe the '
                    'available data source; they are not a physical classification.</p>',
                    unsafe_allow_html=True,
                )

                simbad_url = (
                    "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
                    + quote(star["query_name"])
                )
                st.link_button("Open SIMBAD record ↗", simbad_url)


    st.divider()
    st.caption("Sources: SIMBAD, Gaia DR3, 2MASS, UCAC4, SDSS DR17 and NASA SkyView/DSS2. Distance is 1000 / SIMBAD parallax (mas), for positive parallax only.")

elif tool == "Photometry":
    position = star_cmd_position(star) if star is not None else None
    photometric_distance = None
    if star is not None:
        gaia = star.get("gaia_photometry", {})
        twomass = star.get("twomass_photometry", {})
        photometric_distance = photometric_distance_from_gj(
            gaia.get("phot_g_mean_mag"),
            gaia.get("phot_g_mean_mag_error"),
            twomass.get("j_mag"),
            twomass.get("j_error"),
        )

    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            johnson = star.get("johnson_photometry", {})
            bp_rp = None
            if gaia.get("phot_bp_mean_mag") is not None and gaia.get("phot_rp_mean_mag") is not None:
                bp_rp = gaia["phot_bp_mean_mag"] - gaia["phot_rp_mean_mag"]
            first_row = st.columns(2)
            second_row = st.columns(2)
            with first_row[0]:
                st.metric("Gaia G", magnitude_value(gaia.get("phot_g_mean_mag")))
            with first_row[1]:
                st.metric("2MASS J", magnitude_value(twomass.get("j_mag")))
            with second_row[0]:
                st.metric("Johnson V", magnitude_value(johnson.get("v_mag")))
            with second_row[1]:
                st.metric("BP−RP colour", magnitude_value(bp_rp))

    with spectral_type_card:
        st.subheader("Spectral type estimation")
        mode = st.radio(
            "Mode", ["Loaded star", "Manual"], horizontal=True,
            key="spectral_input_source", label_visibility="collapsed",
        )
        ready = False
        message = None
        if mode == "Loaded star":
            if star is None:
                message = "Load an object in Identification, or use Manual input."
            else:
                g_mag = star["gaia_photometry"].get("phot_g_mean_mag")
                j_mag = star["twomass_photometry"].get("j_mag")
                quality = star["twomass_photometry"].get("j_quality")
                if quality and quality != "A":
                    st.warning(f"2MASS J has quality flag {quality}; check this estimate carefully.")
                if g_mag is None or j_mag is None:
                    message = "This object needs both Gaia G and 2MASS J measurements."
                else:
                    ready = True
        else:
            st.write("Estimate from two magnitudes without changing the selected object.")
            with st.form("manual_spectral_type"):
                g_text = st.text_input("Gaia G magnitude", placeholder="e.g. 10.5")
                j_text = st.text_input("2MASS J magnitude", placeholder="e.g. 7.7")
                submitted = st.form_submit_button("Estimate spectral type")
            if submitted:
                try:
                    if not g_text.strip() or not j_text.strip():
                        raise ValueError("Enter both magnitudes.")
                    g_mag = float(g_text)
                    j_mag = float(j_text)
                    ready = True
                except ValueError:
                    message = "Enter two numeric magnitudes."
            else:
                message = "Enter two magnitudes and select **Estimate spectral type**."

        if not ready:
            if message:
                st.write(message)
        else:
            try:
                color, subtype = estimate_subtype(g_mag, j_mag)
            except ValueError as error:
                st.error(f"No estimate was made. {error}")
            else:
                st.metric("Estimated spectral type", f"M{subtype:.1f}")
                st.markdown(
                    f'<div class="runa-spectral-colour">G−J colour: '
                    f'<strong>{color:.2f} mag</strong></div>',
                    unsafe_allow_html=True,
                )
                if mode == "Loaded star":
                    st.caption(f"Estimate for {star['main_id']} from its catalogue magnitudes.")
                else:
                    st.caption("Manual estimate; the selected RUNA object is unchanged.")

                stars = load_stars()
                fig, ax = plt.subplots(figsize=(6.5, 4))
                fig.patch.set_facecolor("#09090b")
                ax.set_facecolor("#09090b")
                ax.scatter(stars["G-J"], stars["SpTnum"], s=12, alpha=0.35, color="#a1a1aa")
                ax.set_ylim(-0.35, max(5.5, subtype + 0.25))
                ax.vlines(
                    color, -0.35, subtype, color="#6ee7b7",
                    linestyle=(0, (4, 3)), linewidth=1.0,
                    alpha=0.85, zorder=2,
                )
                ax.scatter(
                    color, subtype, s=85, color="#6ee7b7",
                    label="Your estimate", zorder=3,
                )
                ax.set_xlabel("G−J colour (mag)", fontsize=13)
                ax.set_ylabel("Spectral subtype", fontsize=13)
                ax.set_yticks(range(6), ["M0", "M1", "M2", "M3", "M4", "M5"])
                ax.tick_params(colors="#a1a1aa", labelsize=11)
                ax.xaxis.label.set_color("#d4d4d8")
                ax.yaxis.label.set_color("#d4d4d8")
                for spine in ax.spines.values():
                    spine.set_color("#3f3f46")
                    spine.set_zorder(4)
                ax.legend(
                    facecolor="#09090b", edgecolor="#3f3f46",
                    labelcolor="#d4d4d8", fontsize=10,
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Photometric estimate for an ordinary M0–M5.5 dwarf. Confirm with spectroscopy.")

        st.markdown(
            '<p class="runa-note">Check measurement quality before interpreting the estimate.</p>',
            unsafe_allow_html=True,
        )

    with additional_catalogue_card:
        st.subheader("Additional catalogue photometry")
        st.caption(
            "Searches prefer exact SIMBAD catalogue identifiers. "
            "Gaia DR3 values come from the loaded source."
        )
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            sort_by_wavelength = st.checkbox(
                "Sort bands by effective wavelength",
                value=False,
                key="sort_catalogue_photometry",
            )
            show_sed_bands = st.checkbox(
                "Show photometric bands in the SED",
                value=False,
                key="show_sed_bands",
            )
            st.caption(
                "Effective wavelengths follow Cifuentes et al. (2020), Table 1."
            )
            if st.button(
                "Search photometric catalogues",
                key="search_catalogue_photometry",
                use_container_width=True,
            ):
                with st.spinner("Querying photometric catalogues..."):
                    st.session_state["catalogue_photometry"] = (
                        fetch_catalogue_photometry(star)
                    )
            catalogue_records = st.session_state.get("catalogue_photometry")
            if catalogue_records is None:
                st.write(
                    "Select the button to query AllWISE, 2MASS, SDSS9, UCAC4 and GALEX."
                )
            else:
                st.write("#### Spectral energy distribution")
                sed_points = build_sed_points(catalogue_records)
                if len(sed_points) < 2:
                    st.info(
                        "There are not enough calibrated bands to build the SED. "
                        "Bands without a VOSA zero point are left out."
                    )
                else:
                    sed_fig = sed_figure(
                        sed_points,
                        star["main_id"] if star is not None else "Loaded object",
                        show_bands=show_sed_bands,
                    )
                    st.pyplot(sed_fig, use_container_width=True)
                    plt.close(sed_fig)
                    st.caption(
                        "Magnitudes are converted to Vega fluxes using the VOSA "
                        "zero points used in the TFM. The plot shows λFλ in W m⁻²; "
                        "points are ordered by effective wavelength. This is the "
                        "observed SED prepared for a future VOSA fit."
                    )
                catalogue_photometry_table(
                    catalogue_records,
                    sort_by_wavelength=sort_by_wavelength,
                )
                st.caption(
                    "Band colours indicate the approximate spectral range: UV, "
                    "optical, near-IR and mid-IR. Missing bands remain unavailable."
                )

                st.write("#### VOSA")
                vosa_file = build_vosa_input(star, catalogue_records)
                vosa_column, link_column = st.columns(2, gap="small")
                with link_column:
                    st.link_button(
                        "Open VOSA ↗",
                        "https://svo2.cab.inta-csic.es/theory/vosa/",
                        use_container_width=True,
                    )
                if vosa_file is None:
                    with vosa_column:
                        st.button(
                            "Download VOSA input",
                            disabled=True,
                            use_container_width=True,
                            key="download_vosa_input_disabled",
                        )
                    st.info(
                        "A positive Gaia parallax and at least one mapped band with "
                        "an uncertainty are needed to create a VOSA input file."
                    )
                else:
                    with vosa_column:
                        st.download_button(
                            "Download VOSA input",
                            data=vosa_file["content"],
                            file_name=vosa_file["filename"],
                            mime="text/plain",
                            use_container_width=True,
                        )
                    distance_text = f"{vosa_file['distance']:.3f} pc"
                    if vosa_file["distance_error"] is not None:
                        distance_text += f" ({vosa_file['distance_error']:.3f} pc)"
                    st.caption(
                        f"{vosa_file['bands']} mapped bands · Gaia distance: {distance_text}. "
                        "Upload this file to VOSA as magnitudes. Extinction is left blank."
                    )


    with result_card:
        with st.container(border=True):
            st.subheader("Colour–magnitude diagram")
            if star is None:
                st.write("Choose a star in **Identification** first.")
            elif position is None:
                st.info("A positive parallax and Gaia BP/RP measurements are needed for the diagram.")
            else:
                try:
                    with st.spinner("Loading Gaia reference stars..."):
                        reference = gaia_reference_sample()
                except Exception:
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
                        colour, absolute_g, s=115, color="#6ee7b7",
                        edgecolor="#111113", linewidth=1.2,
                        label=str(star["main_id"]), zorder=3,
                    )
                    ax.set_xlabel("Gaia BP−RP colour (mag)", fontsize=13)
                    ax.set_ylabel("Absolute Gaia G magnitude (mag)", fontsize=13)
                    ax.invert_yaxis()
                    ax.grid(color="#3f3f46", alpha=0.28, linewidth=0.6)
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

        if star is not None:
            with st.container(border=True):
                st.subheader("Photometric distance")
                if photometric_distance is None:
                    st.info("Gaia G and 2MASS J are needed for this estimate.")
                elif not photometric_distance["in_range"]:
                    relation = photometric_distance["relation"]
                    st.info(
                        f"The G−J colour ({photometric_distance['colour']:.2f} mag) is "
                        f"outside the calibration range "
                        f"({relation['minimum_colour']:.1f}–{relation['maximum_colour']:.1f} mag)."
                    )
                else:
                    relation = photometric_distance["relation"]
                    colour_error = photometric_distance.get("colour_error")
                    data_section(
                        "Estimate",
                        [
                            (
                                "Distance",
                                f"{photometric_distance['distance_pc']:.1f}",
                                "Cifuentes+20",
                                "36% typical",
                                "pc",
                            ),
                            (
                                "G−J colour",
                                f"{photometric_distance['colour']:.3f}",
                                "Gaia + 2MASS",
                                f"{colour_error:.3f}"
                                if colour_error is not None else None,
                                "mag",
                            ),
                            (
                                "M_G from colour",
                                f"{photometric_distance['absolute_g']:.2f}",
                                f"R² = {relation['r_squared']:.4f}",
                                None,
                                "mag",
                            ),
                        ],
                    )
                    st.warning(
                        "Approximate photometric distance. The Cifuentes et al. (2020) "
                        "relation is intended for solar-metallicity stars with good "
                        "photometry; its typical accuracy is about 36%. Use a trigonometric "
                        "parallax when available. Extinction is assumed negligible."
                    )


        with st.container(border=True):
            st.subheader("TESS light curve")
            if star is None:
                st.write("Choose a star in **Identification** first.")
            else:
                tess_name = tess_identifier(star)
                if tess_name is None:
                    st.info("SIMBAD did not return a TIC or TOI identifier for this object.")
                else:
                    st.caption(
                        f"SIMBAD identifier: {tess_name['label']}. "
                        "The curve is loaded only when requested."
                    )
                    if st.button(
                        "Load TESS light curve",
                        key="load_tess_light_curve",
                        use_container_width=True,
                    ):
                        with st.spinner("Querying MAST for a TESS light curve..."):
                            st.session_state["tess_light_curve"] = fetch_tess_light_curve(star)

                    tess_curve = st.session_state.get("tess_light_curve")
                    if tess_curve is None:
                        st.write("Select the button to query the TESS SPOC archive.")
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
        "Method: BP−RP colour and M_G = G + 5 log10(parallax) − 10, "
        "with parallax in milliarcseconds. Photometric distance uses M_G versus G−J "
        "from Cifuentes et al. (2020), with no extinction correction."
    )

elif tool == "Parameters":
    parameters = calculate_parameters(star) if star is not None else None
    published_parameters = published_stellar_parameters(star)
    published_luminosity = published_parameters.get("luminosity", [])
    preferred_luminosity = next(
        (entry for entry in published_luminosity if entry["reference"].endswith("(2025)")),
        published_luminosity[0] if published_luminosity else None,
    )
    eker_mass = (
        mass_from_luminosity_eker(
            preferred_luminosity["value"], preferred_luminosity["error"]
        ) if preferred_luminosity else None
    )
    catalogue_records = st.session_state.get("catalogue_photometry")
    metallicity = None
    metallicity_inputs = []
    if star is not None and catalogue_records is not None:
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

    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            magnitudes = [("M_G", parameters["M_G"]), ("M_J", parameters["M_J"])]
            magnitude_columns = st.columns(2, gap="small")
            for column, (label, magnitude) in zip(magnitude_columns, magnitudes):
                with column:
                    if magnitude is None:
                        large_metric(label, "—", None)
                    else:
                        large_metric(
                            label,
                            f"{magnitude['value']:.3f} mag",
                            f"{magnitude['error']:.3f} mag"
                            if magnitude["error"] is not None else None,
                        )
            st.caption(
                "Derived from Gaia DR3 and 2MASS photometry with positive "
                "Gaia DR3 parallax."
            )
            if metallicity_inputs:
                data_section("Metallicity inputs", metallicity_inputs)
                st.caption(
                    "M_G above and these Gaia/WISE magnitudes are used for "
                    "the photometric metallicity estimate."
                )

    with result_card:
        if star is None:
            st.info("Choose a star in **Identification** first.")
        else:
            luminosity_rows = [
                (
                    "Bolometric luminosity",
                    f'{entry["value"]:.6g}',
                    entry["reference"],
                    f'{entry["error"]:.3g}' if entry["error"] is not None else None,
                    "L☉",
                )
                for entry in published_luminosity
            ]
            luminosity_estimates = []
            for band in ("M_G", "M_J"):
                estimate = parameters["luminosities"].get(band)
                if estimate is not None:
                    luminosity_estimates.append(
                        (
                            f"From {band}",
                            f"{estimate['luminosity']:.6g}",
                            "Cifuentes et al. (2020), Table 5",
                            f"{estimate['luminosity_error']:.3g}",
                            "L☉",
                        )
                    )

            radius_rows = [
                (
                    "Published value",
                    f'{entry["value"]:.4f}',
                    entry["reference"],
                    f'{entry["error"]:.4f}' if entry["error"] is not None else None,
                    "R☉",
                )
                for entry in published_parameters.get("radius", [])
            ]
            mass_rows = [
                (
                    "Published value",
                    f'{entry["value"]:.4f}',
                    entry["reference"],
                    f'{entry["error"]:.4f}' if entry["error"] is not None else None,
                    "M☉",
                )
                for entry in published_parameters.get("mass", [])
            ]
            mass_estimates = []
            if eker_mass is not None:
                mass_estimates.append(
                    (
                        "From luminosity",
                        f'{eker_mass["value"]:.4f}',
                        "Eker et al. (2018)",
                        f'{eker_mass["error"]:.4f}',
                        "M☉",
                    )
                )

            temperature_rows = [
                (
                    "Published value",
                    f'{entry["value"]:.0f}',
                    entry["reference"],
                    f'{entry["error"]:.0f}' if entry["error"] is not None else None,
                    "K",
                )
                for entry in published_parameters.get("effective_temperature", [])
            ]
            metallicity_rows = [
                (
                    "[Fe/H]",
                    f'{entry["value"]:.3f}',
                    entry["reference"],
                    f'{entry["error"]:.3f}' if entry["error"] is not None else None,
                    "dex",
                )
                for entry in published_parameters.get("metallicity", [])
            ]
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

            first_left, first_right = st.columns(2, gap="small")
            with first_left:
                with st.container(border=True):
                    st.subheader("Luminosity")
                    parameter_section(
                        "Published",
                        luminosity_rows,
                        "No published value is available.",
                    )
                    parameter_section(
                        "Estimated",
                        luminosity_estimates,
                        "No estimate is available for the current inputs.",
                    )

            with first_right:
                with st.container(border=True):
                    st.subheader("Radius")
                    parameter_section(
                        "Published",
                        radius_rows,
                        "No published value is available.",
                    )
                    parameter_section(
                        "Estimated",
                        [],
                        "No radius relation is currently implemented.",
                    )

            second_left, second_right = st.columns(2, gap="small")
            with second_left:
                with st.container(border=True):
                    st.subheader("Mass")
                    parameter_section(
                        "Published",
                        mass_rows,
                        "No published value is available.",
                    )
                    mass_empty = "A published luminosity is required."
                    if preferred_luminosity:
                        mass_empty = "Outside the Eker et al. (2018) calibration."
                    parameter_section("Estimated", mass_estimates, mass_empty)
                    if eker_mass is not None:
                        st.caption(
                            f'Calibration: {eker_mass["minimum"]:.3f}–'
                            f'{eker_mass["maximum"]:.2f} M☉. Input luminosity: '
                            f'{preferred_luminosity["reference"]}.'
                        )

            with second_right:
                with st.container(border=True):
                    st.subheader("Effective temperature")
                    parameter_section(
                        "Published",
                        temperature_rows,
                        "No published value is available.",
                    )
                    parameter_section(
                        "Estimated",
                        [],
                        "No temperature relation is currently implemented.",
                    )

            with st.container(border=True):
                st.subheader("Metallicity")
                parameter_section(
                    "Published",
                    metallicity_rows,
                    "No published value is available.",
                )
                metallicity_empty = (
                    "Search photometric catalogues in Photometry first."
                    if catalogue_records is None else
                    "The downloaded data do not contain all required inputs."
                )
                parameter_section(
                    "Estimated",
                    metallicity_estimates,
                    metallicity_empty,
                )

    st.divider()
    st.caption(
        "Published values retain their individual source. Estimated values show the "
        "relation used. Absolute magnitudes use M = m + 5 log10(parallax) − 10, with "
        "parallax in milliarcseconds. No radius or effective-temperature relation is "
        "currently implemented."
    )

elif tool == "Kinematics":
    banyan_error = None
    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            values = banyan_inputs(star)
            input_rows = []
            for label, key, unit, error_key in (
                ("Right ascension", "ra", "deg", None),
                ("Declination", "dec", "deg", None),
                ("Proper motion RA", "pmra", "mas/yr", "epmra"),
                ("Proper motion Dec", "pmdec", "mas/yr", "epmdec"),
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
            data_section("BANYAN inputs", input_rows)
            if st.button("Run BANYAN Σ", use_container_width=True):
                st.session_state["banyan_result"] = None
                try:
                    with st.spinner("Calculating population membership..."):
                        st.session_state["banyan_result"] = run_banyan(star)
                except ValueError as error:
                    banyan_error = str(error)
                except RuntimeError as error:
                    banyan_error = str(error)
                except Exception:
                    banyan_error = "The BANYAN Σ calculation failed. Please try again."

    with result_card:
        st.subheader("BANYAN Σ membership")
        if banyan_error:
            st.error(banyan_error)
        elif star is None:
            st.write("Choose a star in **Identification** first.")
        elif st.session_state["banyan_result"] is None:
            st.write("Select **Run BANYAN Σ** to estimate population membership.")
        else:
            banyan = st.session_state["banyan_result"]
            best_hypothesis = banyan["best_hypothesis"]
            best_association = banyan["best_association"]
            probability = banyan["association_probability"]
            if best_hypothesis == "FIELD":
                large_metric("Most likely hypothesis", "Field")
            else:
                large_metric(
                    "Most likely population",
                    best_hypothesis,
                    f"{probability:.1%} probability",
                )
                if banyan.get("age_myr") is not None:
                    st.caption(f"Approximate population age: {banyan['age_myr']:g} Myr")

            membership_rows = [
                ("Field", f"{banyan['field_probability']:.1%}", "BANYAN Σ", None, None),
            ]
            membership_rows.extend(
                (
                    association,
                    f"{value:.1%}",
                    "BANYAN Σ",
                    None,
                    None,
                )
                for association, value in banyan["probabilities"][:6]
            )
            data_section("Population probabilities", membership_rows)

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
            data_section("Galactic velocity", velocity_rows)

            with st.expander("Coordinates at epoch", expanded=False):
                gaia_epoch = star.get("gaia_photometry") or {}
                target_epoch = st.number_input(
                    "Target epoch (Julian year)",
                    min_value=1800.0,
                    max_value=2200.0,
                    value=2026.0,
                    step=1.0,
                    format="%.1f",
                    key=f"coordinate_epoch_{star['main_id']}",
                )
                reference_epoch = gaia_epoch.get("ref_epoch")
                propagated = coordinates_at_epoch(
                    gaia_epoch.get("ra"),
                    gaia_epoch.get("dec"),
                    gaia_epoch.get("pmra"),
                    gaia_epoch.get("pmdec"),
                    reference_epoch,
                    target_epoch,
                )
                if propagated is None:
                    st.info(
                        "Gaia DR3 coordinates, proper motion and reference "
                        "epoch are needed for this calculation."
                    )
                else:
                    source = f"Gaia DR3 · {float(reference_epoch):.1f}"
                    data_section(
                        f"Coordinates at epoch {target_epoch:.1f}",
                        [
                            (
                                "Right ascension",
                                sexagesimal(propagated["ra"], True),
                                source,
                                None,
                                None,
                            ),
                            (
                                "Declination",
                                sexagesimal(propagated["dec"]),
                                source,
                                None,
                                None,
                            ),
                        ],
                    )
                    st.caption(
                        "Gaia DR3 coordinates propagated with its proper motion; "
                        "the target is a Julian year."
                    )

    st.divider()
    st.caption(
        "Method: BANYAN Σ (Gagné et al. 2018), using Gaia position, proper motion "
        "and parallax when available, with the SIMBAD radial velocity when available."
    )

elif tool == "Planets":
    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in **Identification** first.")
        else:
            st.markdown(
                '<p class="runa-note">The loaded object name and its catalogue '
                'identifiers are matched against the NASA Exoplanet Archive.</p>',
                unsafe_allow_html=True,
            )
            st.caption("The search uses the archive's PSCompPars table.")

    with result_card:
        st.subheader("Registered planets")
        if star is None:
            st.write("Choose a star in **Identification** first.")
        else:
            try:
                with st.spinner("Searching NASA Exoplanet Archive..."):
                    planets = exoplanet_archive_planets(
                        star.get("query_name"),
                        star.get("main_id"),
                        tuple(star.get("identifiers", [])),
                    )
            except Exception:
                st.error(
                    "The NASA Exoplanet Archive query failed. Please try again later."
                )
            else:
                if not planets:
                    st.info(
                        "No planets were found for this object in the NASA Exoplanet Archive."
                    )
                else:
                    host_names = sorted({
                        str(planet.get("hostname"))
                        for planet in planets
                        if planet.get("hostname")
                    })
                    if host_names:
                        st.caption("Archive host name: " + ", ".join(host_names))
                    planet_table(planets)
                    st.caption(
                        f"{len(planets)} planet(s) listed in the NASA Exoplanet Archive."
                    )

    st.divider()
    st.caption(
        "Source: NASA Exoplanet Archive, Planetary Systems Composite Parameters table."
    )

elif tool == "Multiplicity":
    with input_card:
        object_panel_title(star)
        if star is None:
            st.write("Load an object in Identification first.")
            submitted = False
            view_gaia = False
        elif not star["gaia_id"]:
            st.write("No Gaia DR3 identifier in SIMBAD.")
            submitted = False
            view_gaia = False
        else:
            st.markdown(
                f'<div class="runa-gaia-id">Gaia DR3 ID: '
                f'<strong>{escape(str(star["gaia_id"]))}</strong></div>',
                unsafe_allow_html=True,
            )
            check_col, view_col = st.columns(2, gap="small")
            with check_col:
                submitted = st.button("Check indicators", use_container_width=True)
            with view_col:
                view_gaia = st.button("View nearby sources", use_container_width=True)
            if view_gaia:
                st.session_state["show_gaia_parameters"] = True
                st.rerun()
            if st.session_state["show_gaia_parameters"]:
                radius_arcsec = st.selectbox(
                    "Nearby-source radius",
                    [5, 10, 30, 60, 180, 300],
                    index=4,
                    format_func=radius_label,
                    key=f"multiplicity_radius_{star['main_id']}",
                )

    query_error = None
    if submitted:
        st.session_state["multiplicity_result"] = None
        try:
            st.session_state["multiplicity_result"] = check_multiplicity(star["gaia_id"])
        except ValueError as error:
            query_error = str(error)
        except Exception:
            query_error = "The Gaia query failed. Please try again later."

    nearby_error = None
    nearby = None
    if (st.session_state["show_gaia_parameters"] and star is not None
            and star["gaia_id"]):
        gaia_ra = star["gaia_photometry"].get("ra", star["ra"])
        gaia_dec = star["gaia_photometry"].get("dec", star["dec"])
        try:
            with st.spinner("Loading nearby Gaia sources..."):
                nearby = nearby_gaia_sources_v2(
                    gaia_ra, gaia_dec, radius_arcsec
                )
            nearby = prepare_neighbours(nearby, gaia_ra, gaia_dec)
        except Exception:
            nearby_error = "The nearby Gaia query could not be completed."

    show_nearby = st.session_state["show_gaia_parameters"] and star is not None
    answer = st.session_state["multiplicity_result"]

    with parameters_card:
        st.subheader("Multiplicity criteria")
        if query_error:
            st.error(query_error)
        elif star is None:
            st.write("Choose a star in **Identification** first.")
        elif not star["gaia_id"]:
            st.write("A Gaia DR3 ID is needed for these indicators.")
        elif answer is None:
            st.write("Select **Check indicators** for the loaded star.")
        else:
            values, criteria = answer
            multiplicity_table(values, criteria)

            strong_indicators = any(criteria[number] for number in (1, 2, 3, 4, 5, 7))
            if strong_indicators:
                st.write("**One or more multiplicity indicators are met.**")
            elif criteria[6]:
                st.write("**Only the supporting duplicated-source flag is met.**")
            elif all(result is None for result in criteria.values()):
                st.write("**Gaia has no values for these criteria.**")
            else:
                st.write("**No available criterion is met.**")
            st.caption("These indicators suggest possible multiplicity; they do not confirm a companion.")
            st.caption("Red: criterion met · Green: criterion not met · Grey: unavailable.")
            st.caption("Source: Cifuentes et al. (2025), Table 3. Missing Gaia values remain unavailable.")

            st.divider()
            st.subheader("RVS score vs. RUWE")
            target_rvs_plot = None
            try:
                with st.spinner("Loading Gaia stars within 10 pc..."):
                    rvs_reference = rvs_reference_sample()
                rvs_reference = [
                    source for source in rvs_reference
                    if source.get("source_id") != str(star["gaia_id"])
                ]
                target_rvs_plot = rvs_ruwe_plot(
                    rvs_reference, values, star["main_id"]
                )
            except Exception:
                st.error("The Gaia RVS comparison sample could not be loaded.")
            if target_rvs_plot is not None:
                st.pyplot(target_rvs_plot, use_container_width=True)
                plt.close(target_rvs_plot)
                st.caption(
                    "Light-grey points: Gaia DR3 sources within 10 pc with "
                    "the three RV measurements available. No parallax S/N cut "
                    "is applied. Dashed lines mark RUWE = 2 and RVS = 2."
                )
            else:
                st.info(
                    "An RVS score needs the Gaia RV p-value, renormalised GOF "
                    "and at least one transit count."
                )

    with result_card:
        if show_nearby:
            st.write("### Nearby Gaia sources")
            if nearby_error:
                st.error(nearby_error)
            elif nearby:
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
        else:
            st.subheader("Result")
            st.write("Select **View nearby sources** to inspect nearby Gaia sources.")

    st.divider()
    st.caption("Method: the seven Gaia DR3 statistical criteria in Cifuentes et al. (2025), Table 3. Missing values remain unavailable.")

else:
    st.error("Choose a RUNA tool from the navigation.")
