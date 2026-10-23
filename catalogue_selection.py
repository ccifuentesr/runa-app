"""Choose one available optical measurement per band."""


OPTICAL_PRIORITY = ("APASS DR9", "SkyMapper DR4", "SDSS9", "UCAC4")


def preferred_photometry(records, allowed_filters=None):
    """Keep all non-optical bands and prefer optical catalogues by availability.

    This is a catalogue preference, not a photometric quality assessment.
    VOSA can restrict the selection to filters it supports.
    """
    records = records or []
    selected = {}
    for catalogue in OPTICAL_PRIORITY:
        for record in records:
            if record["catalogue"] != catalogue:
                continue
            for band in record.get("bands", []):
                label = band["label"]
                if allowed_filters is not None:
                    if (catalogue, label) not in allowed_filters:
                        continue
                if band.get("value") is not None and label not in selected:
                    selected[label] = catalogue

    result = []
    for record in records:
        catalogue = record["catalogue"]
        bands = record.get("bands", [])
        if catalogue in OPTICAL_PRIORITY:
            bands = [band for band in bands
                     if selected.get(band["label"]) == catalogue]
        result.append({**record, "bands": bands})
    return result
