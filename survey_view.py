"""Display a survey cutout with an unobtrusive centre marker."""

import base64

import streamlit as st


def survey_image_view(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    mime = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    st.markdown(
        '<style>'
        '.runa-survey-view {position:relative;width:512px;max-width:100%;}'
        '.runa-survey-view img {display:block;width:100%;height:auto;}'
        '.runa-survey-view .centre-mark {position:absolute;pointer-events:none;'
        'background:#d4d4d8;opacity:.9;}'
        '.runa-survey-view .centre-right {left:52%;top:50%;width:5%;height:2px;'
        'transform:translateY(-50%);}'
        '.runa-survey-view .centre-lower {left:50%;top:52%;height:5%;width:2px;'
        'transform:translateX(-50%);}'
        '</style><div class="runa-survey-view">'
        f'<img src="data:{mime};base64,{encoded}" '
        'alt="Survey cutout; two light-grey marks indicate the image centre.">'
        '<span class="centre-mark centre-right" aria-hidden="true"></span>'
        '<span class="centre-mark centre-lower" aria-hidden="true"></span>'
        '</div>',
        unsafe_allow_html=True,
    )
