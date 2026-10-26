"""Compact comparison of published and estimated stellar parameters."""

import csv
from io import StringIO
from pathlib import Path
from decimal import Decimal
from html import escape

import streamlit as st
from parameters import radius_from_luminosity_temperature, mass_from_radius_schweitzer


BIBCODES_FILE = Path(__file__).with_name("data") / "bibcodes.csv"


def load_bibcodes():
    """Load the small local bibliography used by exported results."""
    with BIBCODES_FILE.open(newline="", encoding="utf-8") as source:
        return {
            row["reference"]: row["bibcode"]
            for row in csv.DictReader(source)
        }


def reference_bibcode(reference, bibcodes):
    """Match a displayed method reference to its ADS bibcode."""
    text = str(reference or "").lower()
    references = (
        ("cifuentes et al. (2020)", "cifuentes20"),
        ("cifuentes et al. (2025)", "cifuentes25"),
        ("schweitzer et al. (2019)", "schweitzer19"),
        ("eker et al. (2018)", "eker18"),
        ("duque-arribas et al. (2023)", "duque_arribas23"),
    )
    for label, key in references:
        if label in text:
            return bibcodes.get(key, "")
    return ""


def estimate_source(label):
    """Describe the measured or derived input behind one estimate."""
    text = str(label)
    if "M_G" in text:
        return "M_G"
    if "M_J" in text:
        return "M_J"
    if "manual L and T_eff" in text or "L and T_eff" in text:
        return "L + T_eff"
    if "manual radius" in text or "radius" in text.lower():
        return "R"
    if "luminosity" in text.lower():
        return "L"
    if "SpT" in text:
        return "SpT"
    if text == "[Fe/H]":
        return "G / BP / RP / W1 / W2 / parallax"
    return text.removeprefix("From ")


def estimated_parameters_csv(groups):
    """Return an inspectable CSV containing the currently displayed estimates."""
    bibcodes = load_bibcodes()
    parameter_labels = {
        "Luminosity": "L",
        "Radius": "R",
        "Mass": "M",
        "Effective temperature": "Teff",
        "Metallicity": "[Fe/H]",
    }
    unit_labels = {
        "L☉": "L_sol",
        "R☉": "R_sol",
        "M☉": "M_sol",
    }
    export_order = {
        "Luminosity": 0,
        "Effective temperature": 1,
        "Radius": 2,
        "Mass": 3,
        "Metallicity": 4,
    }
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=("parameter", "value", "error", "unit", "source", "reference"),
        delimiter=";",
        lineterminator="\n",
    )
    writer.writeheader()
    ordered_groups = sorted(
        enumerate(groups),
        key=lambda item: (export_order.get(item[1][0], 99), item[0]),
    )
    for _, (name, unit, _, estimated_rows) in ordered_groups:
        for row in estimated_rows:
            writer.writerow({
                "parameter": parameter_labels.get(name, name),
                "value": row[1],
                "error": row[3] if row[3] is not None else "",
                "unit": unit_labels.get(unit, unit),
                "source": estimate_source(row[0]),
                "reference": reference_bibcode(row[2], bibcodes),
            })
    return output.getvalue()


def panel_heading(title):
    """Use the same small heading for result panels across RUNA."""
    st.markdown(
        f'<div class="runa-panel-heading">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def decimal_text(value):
    """Expand scientific notation without changing the supplied precision."""
    if value is None:
        return "—"
    return format(Decimal(str(value)), "f")


def measurement_cell(row):
    if row is None:
        return '<span class="parameter-missing">—</span>'
    value = escape(decimal_text(row[1]))
    error = ""
    if row[3] is not None:
        error = f'<span class="parameter-uncertainty"> ({escape(decimal_text(row[3]))})</span>'
    return f'<span class="parameter-number">{value}{error}</span>'


def reference_cell(row, estimated=False):
    if row is None:
        return "—"
    reference = escape(str(row[2] or "Source unavailable"))
    if estimated:
        label = escape(str(row[0]))
        label = label.replace("M_G", "M<sub>G</sub>").replace("M_J", "M<sub>J</sub>")
        label = label.replace("T_eff", "T<sub>eff</sub>")
        return f'<span class="parameter-method">{label}</span> · {reference}'
    return reference


def manual_parameter_groups():
    """Calculate radius and mass from independent user inputs."""
    first, second = st.columns(2, gap="small")
    with first:
        luminosity = st.number_input(
            "L (solar units)", min_value=0.00000001, value=0.01,
            format="%.8f", key="manual_parameters_luminosity",
        )
        luminosity_error = st.number_input(
            "L uncertainty (optional)", min_value=0.0, value=None,
            format="%.8f", key="manual_parameters_luminosity_error",
        )
    with second:
        temperature = st.number_input(
            "Teff (K)", min_value=1.0, value=3300.0,
            format="%.1f", key="manual_parameters_temperature",
        )
        temperature_error = st.number_input(
            "Teff uncertainty (optional)", min_value=0.0, value=None,
            format="%.1f", key="manual_parameters_temperature_error",
        )
    radius = radius_from_luminosity_temperature(
        luminosity, luminosity_error, temperature, temperature_error,
    )
    mass = mass_from_radius_schweitzer(radius["value"], radius["error"])
    st.caption("Manual calculation · independent of the loaded star.")
    st.caption("Mass relation: main-sequence M dwarfs; not suitable for very young stars.")
    if luminosity_error is None or temperature_error is None:
        st.caption("Errors use available input uncertainties only; mass also includes coefficient errors.")
    radius_row = (
        "From manual L and T_eff", f'{radius["value"]:.4f}',
        "Stefan–Boltzmann law",
        f'{radius["error"]:.4f}' if radius["error"] is not None else None, "R☉",
    )
    mass_rows = []
    if mass is not None:
        mass_rows.append((
            "From manual radius", f'{mass["value"]:.4f}', mass["reference"],
            f'{mass["error"]:.4f}', "M☉",
        ))
    return [
        ("Radius", "R☉", [], [radius_row]),
        ("Mass", "M☉", [], mass_rows),
    ]


def stellar_parameter_table(groups):
    """Show published values and estimates in two simple panels."""
    st.markdown(
        '<style>'
        '.parameter-block {padding:.9rem 0;border-top:1px solid #303037;}'
        '.parameter-name {font-size:1.05rem;font-weight:600;color:#d4d4d8;'
        'margin-bottom:.55rem;}'
        '.parameter-entry {display:flex;align-items:baseline;justify-content:space-between;'
        'gap:1rem;flex-wrap:wrap;}'
        '.parameter-measurement {flex:0 1 auto;}'
        '.parameter-entry + .parameter-entry {margin-top:.8rem;}'
        '.parameter-number {font-size:1.35rem;font-weight:600;color:#f3f3f5;'
        'font-variant-numeric:tabular-nums;overflow-wrap:anywhere;}'
        '.parameter-uncertainty {color:#92929d;font-weight:400;}'
        '.parameter-unit {font-size:1.1rem;color:#a4a4ad;margin-left:.5rem;white-space:nowrap;}'
        '.parameter-unit sub {font-size:.75em;}'
        '.parameter-reference {font-size:.85rem;color:#92929d;text-align:right;'
        'flex:1 1 12rem;margin-left:auto;}'
        '.parameter-method {color:#c7c7cf;}'
        '.parameter-missing {font-size:.9rem;color:#85858f;}'
        '.st-key-parameters_published {border-color:#5d5968 !important;'
        'border-radius:10px !important;background:#18181b;}'
        '.st-key-parameters_estimated {border-color:#595443 !important;'
        'border-radius:10px !important;background:#18181b;}'
        '.st-key-parameters_published .runa-panel-heading {'
        'color:#c9c4e8;font-size:1.15rem;font-weight:550;letter-spacing:.04em;}'
        '.st-key-parameters_estimated .runa-panel-heading {'
        'color:#f4cf75;font-size:1.15rem;font-weight:550;letter-spacing:.04em;}'
        '.st-key-parameters_estimated [data-testid="stWidgetLabel"] {display:none;}'
        '.st-key-parameters_estimated [data-testid="stRadio"] {'
        'margin-top:-.45rem;margin-bottom:-.85rem;}'
        '.st-key-parameters_published .parameter-number {color:#c9c4e8;}'
        '.st-key-parameters_estimated .parameter-number {color:#f4cf75;}'
        '</style>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2, gap="small")
    for column, title, row_index in zip(columns, ("Published", "Estimated"), (2, 3)):
        with column:
            with st.container(border=True, key=f"parameters_{title.lower()}"):
                displayed_groups = groups
                if title == "Estimated":
                    heading_column, mode_column = st.columns((0.82, 1.18), gap="small")
                    with heading_column:
                        panel_heading(title)
                    with mode_column:
                        mode = st.radio(
                            "Mode", ["Loaded star", "Manual"], horizontal=True,
                            label_visibility="collapsed",
                            key="parameters_estimation_mode",
                        )
                    if mode == "Manual":
                        displayed_groups = manual_parameter_groups()
                else:
                    panel_heading(title)
                for group in displayed_groups:
                    name, unit = group[:2]
                    unit_html = escape(unit)
                    if unit in ("L☉", "R☉", "M☉"):
                        unit_html = f'{unit[0]}<sub>☉</sub>'
                    rows = group[row_index]
                    entries = []
                    for row in rows:
                        entries.append(
                            '<div class="parameter-entry">'
                            '<div class="parameter-measurement">'
                            f'{measurement_cell(row)}'
                            f'<span class="parameter-unit">{unit_html}</span></div>'
                            '<div class="parameter-reference">'
                            f'{reference_cell(row, title == "Estimated")}</div></div>'
                        )
                    if not entries:
                        message = "No published value" if title == "Published" else "No estimate available"
                        entries.append(f'<div class="parameter-missing">{message}</div>')
                    st.markdown(
                        '<section class="parameter-block">'
                        f'<div class="parameter-name">{escape(name)}</div>'
                        + "".join(entries) + '</section>',
                        unsafe_allow_html=True,
                    )
                if title == "Estimated":
                    export = estimated_parameters_csv(displayed_groups)
                    if export.count("\n") > 1:
                        st.download_button(
                            "Download estimated parameters (.csv)",
                            data=export,
                            file_name="runa_estimated_parameters.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_estimated_parameters",
                        )
