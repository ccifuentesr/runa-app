"""Compact comparison of published and estimated stellar parameters."""

import csv
from io import StringIO
from pathlib import Path
from decimal import Decimal
from html import escape
import re
from urllib.parse import quote

import streamlit as st


BIBCODES_FILE = Path(__file__).with_name("data") / "bibcodes.csv"


def compact_paper_citation(text):
    """Shorten author-year citations for compact table source columns."""
    value = str(text or "")
    return re.sub(
        r"\s+et al\.\s*\((20\d{2})\)",
        lambda match: "+" + match.group(1)[-2:],
        value,
    )


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
        ("pecaut & mamajek (2013)", "pecaut_mamajek13"),
        ("duque-arribas et al. (2023)", "duque_arribas23"),
    )
    for label, key in references:
        if label in text:
            return bibcodes.get(key, "")
    return ""


def estimate_source(label):
    """Describe the measured or derived input behind one estimate."""
    text = str(label)
    if "M_G" in text and "T_eff" in text:
        return "M_G + Teff"
    if "M_J" in text and "T_eff" in text:
        return "M_J + Teff"
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
        "Teff": "Teff",
        "Metallicity": "[Fe/H]",
    }
    unit_labels = {
        "L☉": "L_sol",
        "R☉": "R_sol",
        "M☉": "M_sol",
    }
    export_order = {
        "Luminosity": 0,
        "Teff": 1,
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
            if len(row) > 5 and row[5] == "input":
                continue
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
    role_class = " parameter-input" if len(row) > 5 and row[5] == "input" else ""
    return f'<span class="parameter-number{role_class}">{value}{error}</span>'


def reference_cell(row):
    if row is None:
        return "—"
    return escape(compact_paper_citation(row[2] or "Source unavailable"))


def estimate_origin_cell(row):
    """Show the photometric origin beneath the estimate reference."""
    if row is None:
        return ""
    label = escape(estimate_source(row[0]))
    label = label.replace("M_G", "M<sub>G</sub>").replace("M_J", "M<sub>J</sub>")
    label = label.replace("T_eff", "T<sub>eff</sub>")
    return f'<small class="runa-estimate-origin">{label}</small>'


def additional_measurements_panel(measurements):
    """Show the original SIMBAD literature measurements beneath published values."""
    sections = (
        ("metallicity", "[Fe/H]", "dex"),
        ("vsini", "v sin i", "km/s"),
        ("rotation_period", "P<sub>rot</sub>", "d"),
        ("spectral_type", "SpT", ""),
    )
    with st.expander("Additional literature measurements", expanded=False):
        rows = []
        for key, label, unit in sections:
            for record in measurements.get(key, []):
                value = record["value"]
                if key == "metallicity":
                    value_text = f"{value:.3f}"
                elif key == "vsini":
                    value_text = f"{value:.4g}"
                elif key == "rotation_period":
                    value_text = f"{value:.5g}"
                else:
                    value_text = str(value)
                qualifier = record.get("qualifier")
                if qualifier in ("<", ">"):
                    value_text = qualifier + value_text
                error = record.get("error")
                error_html = (
                    f'<span class="parameter-uncertainty"> ({error:.3g})</span>'
                    if error is not None else ""
                )
                bibcode = str(record["bibcode"])
                short_bibcode = bibcode.split(".", 1)[0]
                reference_url = (
                    "https://ui.adsabs.harvard.edu/abs/"
                    + quote(bibcode, safe="") + "/abstract"
                )
                unit_html = escape(unit) if unit else ""
                rows.append(
                    '<div class="runa-data-row runa-parameter-compact-row">'
                    f'<span class="runa-data-label">{label}</span>'
                    f'<span class="runa-data-value">{escape(value_text)}{error_html}</span>'
                    f'<span class="runa-data-unit">{unit_html}</span>'
                    '<span class="runa-data-source">'
                    f'<a href="{escape(reference_url, quote=True)}" target="_blank" '
                    f'rel="noopener noreferrer" title="{escape(bibcode, quote=True)}">'
                    f'{escape(short_bibcode)}</a></span></div>'
                )
        if rows:
            st.markdown(
                '<section class="runa-data-section runa-parameter-compact runa-literature-measurements">'
                + "".join(rows) + '</section>',
                unsafe_allow_html=True,
            )
            st.caption("SIMBAD measurement collections. Rotation periods include ROT-classified entries only.")
            if measurements.get("unavailable"):
                st.caption("Some SIMBAD collections could not be loaded.")
        elif measurements.get("unavailable"):
            st.caption("SIMBAD literature measurements are temporarily unavailable.")
        else:
            st.caption("No non-Gaia measurements found in these SIMBAD collections.")


def stellar_parameter_table(groups, additional_measurements=None):
    """Show published values and estimates in two simple panels."""
    st.markdown(
        '<style>'
        '.parameter-number {font-variant-numeric:tabular-nums;overflow-wrap:anywhere;}'
        '.parameter-uncertainty {color:#92929d;font-weight:400;}'
        '.parameter-missing {font-size:.9rem;color:#85858f;}'
        '.st-key-parameters_published {border-color:#5d5968 !important;'
        'border-radius:10px !important;background:#18181b;}'
        '.st-key-parameters_estimated {border-color:#595443 !important;'
        'border-radius:10px !important;background:#18181b;}'
        '.st-key-parameters_additional {border-color:#5d5968 !important;'
        'border-radius:10px !important;background:#18181b;}'
        '.st-key-parameters_additional .runa-panel-heading {color:#bdb6dd;}'
        '.parameter-additional-value {font-size:1.23rem;font-weight:560;'
        'color:#c9c4e8;font-variant-numeric:tabular-nums;}'
        '</style>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2, gap="small")
    for column, title in zip(columns, ("Estimated", "Published")):
        with column:
            with st.container(border=True, key=f"parameters_{title.lower()}"):
                if title == "Estimated":
                    panel_heading("Stellar parameter estimation")
                else:
                    panel_heading("Published stellar parameters")
                estimated = title == "Estimated"
                rows = []
                for name, unit, published_rows, estimated_rows in groups:
                    unit_html = escape(unit)
                    if unit in ("L☉", "R☉", "M☉"):
                        unit_html = f'{unit[0]}<sub>☉</sub>'
                    values = estimated_rows if estimated else published_rows
                    for row in values or (None,):
                        row_class = (
                            f' runa-estimate-{name.lower()}' if estimated else ""
                        )
                        value_html = (
                            measurement_cell(row) if row is not None else
                            '<span class="parameter-missing">'
                            + ("No estimate available" if estimated else "No published value")
                            + '</span>'
                        )
                        source_html = reference_cell(row) if row is not None else "—"
                        if estimated and row is not None:
                            source_html += estimate_origin_cell(row)
                        rows.append(
                            f'<div class="runa-data-row runa-parameter-compact-row{row_class}">'
                            f'<span class="runa-data-label">{escape(name)}</span>'
                            f'<span class="runa-data-value">{value_html}</span>'
                            f'<span class="runa-data-unit">{unit_html}</span>'
                            f'<span class="runa-data-source">{source_html}</span></div>'
                        )
                st.markdown(
                    '<section class="runa-data-section runa-parameter-compact">'
                    + "".join(rows) + '</section>',
                    unsafe_allow_html=True,
                )
                if title == "Estimated":
                    export = estimated_parameters_csv(groups)
                    if export.count("\n") > 1:
                        st.download_button(
                            "Download estimated parameters (.csv)",
                            data=export,
                            file_name="runa_estimated_parameters.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_estimated_parameters",
                        )
            if title == "Published" and additional_measurements is not None:
                additional_measurements_panel(additional_measurements)
