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
        return f'<span class="parameter-method">{label}</span><br>{reference}'
    return reference


def stellar_parameter_table(groups):
    """Show every available reference without implying paired measurements."""
    rows = []
    for name, unit, published, estimated in groups:
        for index in range(max(len(published), len(estimated), 1)):
            pub = published[index] if index < len(published) else None
            est = estimated[index] if index < len(estimated) else None
            group_class = ' class="parameter-group"' if index == 0 else ""
            label = escape(name) if index == 0 else ""
            unit_text = escape(unit) if index == 0 else ""
            rows.append(
                f'<tr{group_class}><th scope="row">{label}</th><td>{unit_text}</td>'
                f'<td>{measurement_cell(pub)}</td><td>{reference_cell(pub)}</td>'
                f'<td>{measurement_cell(est)}</td><td>{reference_cell(est, True)}</td></tr>'
            )
    st.markdown(
        '<style>'
        '.parameter-table-scroll {overflow-x:auto;}'
        '.parameter-comparison {width:100%;border-collapse:collapse;color:#ededf0;}'
        '.parameter-comparison th,.parameter-comparison td {padding:1rem .8rem;'
        'text-align:left;vertical-align:top;border-bottom:1px solid #28282e;}'
        '.parameter-comparison thead th {font-size:1.05rem;font-weight:650;'
        'background:#19191e;color:#f3f3f5;}'
        '.parameter-comparison tbody th {font-size:1rem;font-weight:600;min-width:8rem;}'
        '.parameter-comparison td {font-size:.95rem;}'
        '.parameter-comparison td:nth-child(4),.parameter-comparison td:nth-child(6) '
        '{color:#a4a4ad;font-size:.85rem;min-width:10rem;}'
        '.parameter-number {font-size:1.1rem;font-weight:600;white-space:nowrap;'
        'font-variant-numeric:tabular-nums;}'
        '.parameter-uncertainty {color:#92929d;font-weight:400;}'
        '.parameter-method {color:#d4d4d8;}'
        '.parameter-missing {color:#777782;}'
        '.parameter-group th,.parameter-group td {border-top:2px solid #3a3a42;}'
        '</style><div class="parameter-table-scroll">'
        '<table class="parameter-comparison" aria-label="Stellar parameters">'
        '<thead><tr><th scope="col">Parameter</th><th scope="col">Unit</th>'
        '<th scope="col">Published</th><th scope="col">Source</th>'
        '<th scope="col">Estimated</th><th scope="col">Method / source</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></div>',
        unsafe_allow_html=True,
    )
