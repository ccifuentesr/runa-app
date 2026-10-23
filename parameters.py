"""Magnitude, luminosity, distance, and metallicity relations used by RUNA."""

from math import isfinite, log, log10, sqrt


LN10 = log(10.0)
SOLAR_EFFECTIVE_TEMPERATURE = 5772.0


# The coefficients and ranges are the empirical fits in Table 5.
LUMINOSITY_RELATIONS = {
    "M_J": (
        {
            "minimum": 4.4,
            "maximum": 11.2,
            "coefficients": (2.051, -0.662, 0.0267, -0.00102),
            "coefficient_errors": (0.075, 0.030, 0.0039, 0.00016),
            "r_squared": 0.9923,
            "range_text": "[4.4, 11.2)",
        },
        {
            "minimum": 11.2,
            "maximum": 14.8,
            "coefficients": (-3.906, 0.334, -0.0263),
            "coefficient_errors": (0.998, 0.156, 0.0061),
            "r_squared": 0.9477,
            "range_text": "[11.2, 14.8]",
        },
    ),
    "M_G": (
        {
            "minimum": 6.4,
            "maximum": 14.0,
            "coefficients": (0.145, 0.074, -0.0382, 0.00119),
            "coefficient_errors": (0.201, 0.060, 0.0060, 0.00019),
            "r_squared": 0.9901,
            "range_text": "[6.4, 14.0)",
        },
        {
            "minimum": 14.0,
            "maximum": 20.2,
            "coefficients": (-2.329, 0.092, -0.0103),
            "coefficient_errors": (0.687, 0.084, 0.0025),
            "r_squared": 0.9782,
            "range_text": "[14.0, 20.2]",
        },
    ),
}


# M_G as a function of G-J from Table 5.  The fit is calibrated for
# solar-metallicity stars in the colour range 2.0 <= G-J <= 4.0.
PHOTOMETRIC_DISTANCE_RELATION = {
    "coefficients": (16.24, -13.04, 5.64, -0.622),
    "coefficient_errors": (4.57, 4.80, 1.66, 0.188),
    "minimum_colour": 2.0,
    "maximum_colour": 4.0,
    "r_squared": 0.9308,
    "typical_error_fraction": 0.36,
}


# Duque-Arribas et al. (2023), ApJ 944, 106.  This is the same relation
# used in paper_cif03/cif03.calculator_metallicity_duq23.py.
METALLICITY_RELATION = {
    "name": "Duque-Arribas et al. (2023)",
    "coefficients": (5.87, -1.40, 5.08, -2.45, -0.788, 0.1094),
    "coefficient_errors": (0.44, 0.12, 0.24, 0.14, 0.052, 0.0071),
}


def absolute_magnitude(magnitude, magnitude_error, parallax, parallax_error):
    """Return an absolute magnitude and its propagated error.

    Parallax is in milliarcseconds.  The relation is only used for a
    positive parallax; a missing error simply leaves that term out.
    """
    if magnitude is None or parallax is None or float(parallax) <= 0:
        return None

    magnitude = float(magnitude)
    parallax = float(parallax)
    value = magnitude + 5.0 * log10(parallax) - 10.0

    error_terms = []
    if magnitude_error is not None:
        error_terms.append(float(magnitude_error) ** 2)
    if parallax_error is not None:
        parallax_term = 5.0 / LN10
        error_terms.append((parallax_term * float(parallax_error) / parallax) ** 2)
    error = sqrt(sum(error_terms)) if error_terms else None
    return {"value": value, "error": error}


def _relation_for(value, relation_name):
    """Find the Cifuentes et al. relation that covers one magnitude."""
    relations = LUMINOSITY_RELATIONS[relation_name]
    for relation in relations:
        minimum = relation["minimum"]
        maximum = relation["maximum"]
        if relation is relations[-1]:
            if minimum <= value <= maximum:
                return relation
        elif minimum <= value < maximum:
            return relation
    return None


def luminosity_from_absolute_magnitude(magnitude, magnitude_error, relation_name):
    """Estimate luminosity from M_G or M_J and propagate fit errors."""
    if magnitude is None:
        return None

    relation = _relation_for(float(magnitude), relation_name)
    if relation is None:
        return None

    coefficients = relation["coefficients"]
    coefficient_errors = relation["coefficient_errors"]
    log_luminosity = sum(
        coefficient * float(magnitude) ** power
        for power, coefficient in enumerate(coefficients)
    )

    derivative = sum(
        power * coefficient * float(magnitude) ** (power - 1)
        for power, coefficient in enumerate(coefficients)
        if power > 0
    )
    coefficient_error = sqrt(sum(
        (coefficient_error * float(magnitude) ** power) ** 2
        for power, coefficient_error in enumerate(coefficient_errors)
    ))
    magnitude_term = 0.0
    if magnitude_error is not None:
        magnitude_term = derivative * float(magnitude_error)
    log_error = sqrt(coefficient_error ** 2 + magnitude_term ** 2)

    # Table 5 gives log10(L/L☉); convert the fit back to linear luminosity.
    luminosity = 10.0 ** log_luminosity
    luminosity_error = LN10 * luminosity * log_error
    return {
        "log_luminosity": log_luminosity,
        "log_error": log_error,
        "luminosity": luminosity,
        "luminosity_error": luminosity_error,
        "relation": relation,
    }


EKER_MLR = (
    # (minimum mass, maximum mass, a, b, scatter in log L/Lsun)
    (0.179, 0.45, 2.028, -0.976, 0.076),
    (0.45, 0.72, 4.572, -0.102, 0.109),
    (0.72, 1.05, 5.743, -0.007, 0.129),
    (1.05, 2.40, 4.329, 0.010, 0.140),
    (2.40, 7.00, 3.967, 0.093, 0.165),
    (7.00, 31.00, 2.865, 1.105, 0.152),
)


def mass_from_luminosity_eker(luminosity, luminosity_error=None):
    """Invert the Eker et al. (2018) main-sequence mass--luminosity relation.

    The function intentionally returns None outside the published 0.179--31
    solar-mass intervals rather than extrapolating to late M dwarfs or
    substellar objects.
    """
    try:
        luminosity = float(luminosity)
    except (TypeError, ValueError):
        return None
    if not isfinite(luminosity) or luminosity <= 0:
        return None

    log_luminosity = log10(luminosity)
    for minimum, maximum, slope, intercept, scatter in EKER_MLR:
        log_mass = (log_luminosity - intercept) / slope
        mass = 10.0 ** log_mass
        if minimum <= mass <= maximum:
            log_luminosity_error = 0.0
            try:
                if luminosity_error is not None and float(luminosity_error) > 0:
                    log_luminosity_error = float(luminosity_error) / (luminosity * LN10)
            except (TypeError, ValueError):
                pass
            log_mass_error = sqrt(
                (log_luminosity_error / slope) ** 2 + (scatter / slope) ** 2
            )
            return {
                "value": mass,
                "error": mass * LN10 * log_mass_error,
                "minimum": minimum,
                "maximum": maximum,
            }
    return None


def radius_from_luminosity_temperature(
    luminosity,
    luminosity_error,
    temperature,
    temperature_error,
):
    """Estimate radius from luminosity and temperature with the SB law."""
    try:
        luminosity = float(luminosity)
        temperature = float(temperature)
    except (TypeError, ValueError):
        return None
    if not isfinite(luminosity) or not isfinite(temperature):
        return None
    if luminosity <= 0 or temperature <= 0:
        return None

    radius = sqrt(luminosity) * (
        SOLAR_EFFECTIVE_TEMPERATURE / temperature
    ) ** 2
    relative_terms = []
    try:
        if luminosity_error is not None and float(luminosity_error) > 0:
            relative_terms.append(
                (0.5 * float(luminosity_error) / luminosity) ** 2
            )
    except (TypeError, ValueError):
        pass
    try:
        if temperature_error is not None and float(temperature_error) > 0:
            relative_terms.append(
                (2.0 * float(temperature_error) / temperature) ** 2
            )
    except (TypeError, ValueError):
        pass

    error = radius * sqrt(sum(relative_terms)) if relative_terms else None
    return {"value": radius, "error": error}


def photometric_distance_from_gj(g_mag, g_error, j_mag, j_error):
    """Estimate distance from G and J using the Cifuentes et al. fit.

    The result is an approximate photometric distance.  The quoted 36 per cent
    uncertainty represents the typical accuracy of the colour sequence, rather
    than a precise measurement error for an individual star.
    """
    if g_mag is None or j_mag is None:
        return None

    g_mag = float(g_mag)
    j_mag = float(j_mag)
    colour = g_mag - j_mag
    relation = PHOTOMETRIC_DISTANCE_RELATION
    if not relation["minimum_colour"] <= colour <= relation["maximum_colour"]:
        return {
            "colour": colour,
            "in_range": False,
            "relation": relation,
        }

    coefficients = relation["coefficients"]
    absolute_g = sum(
        coefficient * colour ** power
        for power, coefficient in enumerate(coefficients)
    )
    coefficient_error = sqrt(sum(
        (coefficient_error * colour ** power) ** 2
        for power, coefficient_error in enumerate(relation["coefficient_errors"])
    ))

    colour_error = None
    if g_error is not None or j_error is not None:
        g_term = float(g_error) if g_error is not None else 0.0
        j_term = float(j_error) if j_error is not None else 0.0
        colour_error = sqrt(g_term ** 2 + j_term ** 2)

    distance_pc = 10.0 ** ((g_mag - absolute_g + 5.0) / 5.0)
    distance_error = relation["typical_error_fraction"] * distance_pc
    return {
        "colour": colour,
        "colour_error": colour_error,
        "absolute_g": absolute_g,
        "absolute_g_error": coefficient_error,
        "distance_pc": distance_pc,
        "distance_error_pc": distance_error,
        "in_range": True,
        "relation": relation,
    }


def metallicity_from_gaia_wise(
    g_mag,
    g_error,
    bp_mag,
    bp_error,
    rp_mag,
    rp_error,
    w1_mag,
    w1_error,
    w2_mag,
    w2_error,
    parallax,
    parallax_error,
):
    """Estimate [Fe/H] with the published Gaia/WISE colour relation.

    The relation uses X = W1-W2, Y = BP-RP and Z = M_G.  It is a direct,
    readable port of the calculation in the existing Duque-Arribas script;
    the error combines the input errors and the published coefficient errors.
    """
    values = (
        g_mag, bp_mag, rp_mag, w1_mag, w2_mag, parallax,
    )
    if any(value is None for value in values):
        return None

    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in values) or values[-1] <= 0:
        return None

    g_mag, bp_mag, rp_mag, w1_mag, w2_mag, parallax = values
    absolute_g = absolute_magnitude(
        g_mag, g_error, parallax, parallax_error
    )
    if absolute_g is None:
        return None

    x_colour = w1_mag - w2_mag
    y_colour = bp_mag - rp_mag
    z_absolute_g = absolute_g["value"]
    a, b, c, d, e, f = METALLICITY_RELATION["coefficients"]
    fe_h = (
        a + b * x_colour + c * y_colour + d * z_absolute_g
        + e * y_colour ** 2 + f * z_absolute_g ** 2
    )

    def error_or_zero(error):
        try:
            return float(error) if error is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    sigma_x = sqrt(error_or_zero(w1_error) ** 2 + error_or_zero(w2_error) ** 2)
    sigma_y = sqrt(error_or_zero(bp_error) ** 2 + error_or_zero(rp_error) ** 2)
    sigma_z = absolute_g["error"] or 0.0
    coefficient_errors = METALLICITY_RELATION["coefficient_errors"]
    coefficient_error = sqrt(sum(
        (error * feature) ** 2
        for error, feature in zip(
            coefficient_errors,
            (1.0, x_colour, y_colour, z_absolute_g,
             y_colour ** 2, z_absolute_g ** 2),
        )
    ))
    input_error = sqrt(
        (b * sigma_x) ** 2
        + ((c + 2 * e * y_colour) * sigma_y) ** 2
        + ((d + 2 * f * z_absolute_g) * sigma_z) ** 2
    )

    return {
        "value": fe_h,
        "error": sqrt(coefficient_error ** 2 + input_error ** 2),
        "M_G": absolute_g,
        "W1_minus_W2": x_colour,
        "BP_minus_RP": y_colour,
        "relation": METALLICITY_RELATION,
    }


def calculate_parameters(star):
    """Calculate M_G, M_J and available Cifuentes et al. luminosities."""
    gaia = star.get("gaia_photometry", {})
    twomass = star.get("twomass_photometry", {})
    parallax = gaia.get("parallax", star.get("plx_value"))
    parallax_error = gaia.get("parallax_error", star.get("plx_err"))
    g_mag = gaia.get("phot_g_mean_mag")
    g_error = gaia.get("phot_g_mean_mag_error")
    j_mag = twomass.get("j_mag")
    j_error = twomass.get("j_error")

    absolute_g = absolute_magnitude(
        g_mag, g_error, parallax, parallax_error
    )
    absolute_j = absolute_magnitude(
        j_mag, j_error, parallax, parallax_error
    )
    luminosities = {}
    if absolute_g is not None:
        estimate = luminosity_from_absolute_magnitude(
            absolute_g["value"], absolute_g["error"], "M_G"
        )
        if estimate is not None:
            luminosities["M_G"] = estimate
    if absolute_j is not None:
        estimate = luminosity_from_absolute_magnitude(
            absolute_j["value"], absolute_j["error"], "M_J"
        )
        if estimate is not None:
            luminosities["M_J"] = estimate

    return {
        "parallax": parallax,
        "parallax_error": parallax_error,
        "g_mag": g_mag,
        "g_error": g_error,
        "j_mag": j_mag,
        "j_error": j_error,
        "M_G": absolute_g,
        "M_J": absolute_j,
        "luminosities": luminosities,
    }
