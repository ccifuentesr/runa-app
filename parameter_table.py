"""Compact comparison of published and estimated stellar parameters."""

from decimal import Decimal
from html import escape

import streamlit as st


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
        '</style>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2, gap="small")
    for column, title, row_index in zip(columns, ("Published", "Estimated"), (2, 3)):
        with column:
            with st.container(border=True):
                st.subheader(title)
                for group in groups:
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
