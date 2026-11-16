"""Published stellar parameters included with RUNA."""

import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")
CATALOGUES = (
    {
        "filename": "cifuentes25_dataset.csv",
        "reference": "Cifuentes et al. (2025)",
        "gaia_column": "GaiaDR3",
        "columns": {
            "luminosity": ("LBol", "e_LBol"),
            "radius": ("Rad", "e_Rad"),
            "mass": ("Mass", "e_Mass"),
            "effective_temperature": ("Teff", "e_Teff"),
            "metallicity": ("[Fe/H]", None),
        },
    },
    {
        "filename": "cifuentes20_dataset.csv",
        "reference": "Cifuentes et al. (2020)",
        "gaia_column": None,
        "columns": {
            "luminosity": ("Lbol", "e_Lbol"),
            "radius": ("Radius", "e_Radius"),
            "mass": ("Mass", "e_Mass"),
            "effective_temperature": ("Teff", None),
        },
    },
)

# Table A.4 of Cifuentes et al. (2025) defines these codes and normally lists
# the original identification first. Keep the paper's author/year mapping here
# rather than interpreting ambiguous abbreviations from the code alone.
SB_FIRST_REPORTS = {
    "Abt70": "Abt (1970)",
    "Abt76": "Abt & Levy (1976)",
    "Bar18": "Baroch et al. (2018)",
    "Bar21": "Baroch et al. (2021)",
    "Bel97": "Belopolsky (1897)",
    "Ben08": "Bender & Simon (2008)",
    "Bon13a": "Bonfils et al. (2013)",
    "Bop77": "Bopp & Fekel (1977)",
    "Dav14": "Davison et al. (2014)",
    "Del98": "Delfosse et al. (1998)",
    "Del99a": "Delfosse et al. (1999b)",
    "Del99b": "Delfosse et al. (1999a)",
    "Del99c": "Delfosse et al. (1999c)",
    "Duq88": "Duquennoy & Mayor (1988)",
    "Fou18": "Fouqué et al. (2018)",
    "Gaia22a": "Gaia Collaboration (2022)",
    "Giz02": "Gizis et al. (2002)",
    "Giz97b": "Gizis (1997)",
    "GR96": "Gizis & Reid (1996)",
    "Gri85": "Griffin et al. (1985)",
    "Har96": "Harlow (1996)",
    "Her65": "Herbig & Moorhead (1965)",
    "Irw09": "Irwin et al. (2009)",
    "Jef18": "Jeffers et al. (2018)",
    "Jen09": "Jenkins et al. (2009)",
    "Joy26": "Joy & Sanford (1926)",
    "Joy47": "Joy (1947)",
    "Kat13": "Katoh et al. (2013)",
    "Mal14a": "Malo et al. (2014a)",
    "Mar87": "Marcy et al. (1987)",
    "Mer09": "Mermilliod et al. (2009)",
    "Moc02": "Mochnacki et al. (2002)",
    "Mon18": "Montes et al. (2018)",
    "Mor09": "Morales et al. (2009)",
    "New16": "Newton et al. (2016)",
    "Nid02": "Nidever et al. (2002)",
    "Rei09": "Reiners & Basri (2009)",
    "Rein12": "Reiners et al. (2012)",
    "Rei97b": "Reid & Gizis (1997)",
    "She85": "Shen et al. (1985)",
    "Shk09": "Shkolnik et al. (2009)",
    "Shk10": "Shkolnik et al. (2010)",
    "Ski18": "Skinner et al. (2018)",
    "Spe19": "Sperauskas et al. (2019)",
    "Sta97": "Stauffer et al. (1997)",
    "Str03": "Strassmeier & Rice (2003)",
    "TP86": "Tomkin & Pettersen (1986)",
    "Win20": "Winters et al. (2020)",
    "Zak79": "Zakhozhaj (1979)",
}
SB_FIRST_BIBCODES = {"Abt70": "1970ApJS...19..387A"}


def _normalise_identifier(value):
    """Normalise common stellar names while retaining numeric groups."""
    text = str(value or "").lower().replace("−", "-")
    parts = re.findall(r"[a-z]+|\d+", text)
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def _float_or_none(value, positive=False):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if positive and number <= 0:
        return None
    return number


def _catalogue_rows(filename):
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _matches(row, star, gaia_column):
    gaia_id = str(star.get("gaia_id") or "").strip()
    if gaia_column and gaia_id and row.get(gaia_column, "").strip() == gaia_id:
        return True

    names = [star.get("main_id"), star.get("query_name"), *star.get("identifiers", [])]
    normalised_names = {_normalise_identifier(name) for name in names if name}
    candidates = [row.get("Name"), row.get("SimbadName"), row.get("Karmn"), row.get("GJ")]
    return any(
        _normalise_identifier(candidate) in normalised_names
        for candidate in candidates if candidate
    )


def published_multiplicity(star):
    """Return the published multiplicity description for a matched star."""
    if not star:
        return None
    for row in _catalogue_rows("cifuentes25_dataset.csv"):
        if _matches(row, star, "GaiaDR3"):
            sb_references = [
                reference.strip()
                for reference in row.get("r_SB", "").split(",")
                if reference.strip()
            ]
            # Castor's first code, Her03, credits Herschel's visual discovery;
            # Bel97 is the first spectroscopic report in that reference list.
            first_sb_code = next(
                (code for code in sb_references if code != "Her03"), None
            )
            return {
                "type": row.get("Type", "").strip(),
                "sb": row.get("SB", "").strip(),
                "sb_references": sb_references,
                "sb_first_report": (
                    SB_FIRST_REPORTS.get(first_sb_code, first_sb_code)
                    if first_sb_code else None
                ),
                "sb_first_bibcode": SB_FIRST_BIBCODES.get(first_sb_code),
                "candidate": row.get("Candidate", "").strip() == "T",
                "reference": "Cifuentes et al. (2025)",
            }
    return None


def published_stellar_parameters(star):
    """Return matching published values grouped by stellar parameter."""
    if not star:
        return {}

    results = {
        "luminosity": [],
        "radius": [],
        "mass": [],
        "effective_temperature": [],
        "metallicity": [],
    }
    for catalogue in CATALOGUES:
        for row in _catalogue_rows(catalogue["filename"]):
            if not _matches(row, star, catalogue["gaia_column"]):
                continue
            for parameter, columns in catalogue["columns"].items():
                value_column, error_column = columns
                value = _float_or_none(
                    row.get(value_column),
                    positive=parameter != "metallicity",
                )
                if value is None:
                    continue
                error = None
                if error_column:
                    error = _float_or_none(row.get(error_column), positive=True)
                results[parameter].append({
                    "reference": catalogue["reference"],
                    "value": value,
                    "error": error,
                    "name": (
                        row.get("Name") or row.get("SimbadName") or row.get("Karmn")
                    ),
                })
            break
    return results
