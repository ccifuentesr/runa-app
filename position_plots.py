"""Small position plots for the loaded star.

The polar view follows the RA--distance plot already used in the KOBE
scripts.  It is kept here as a small function so the web interface does not
need to know how the plot is built.
"""

import matplotlib.pyplot as plt
import numpy as np


MAS_PER_RAD = 206264806.0
TAIL_YEARS = 1.0e5
REFERENCE_RINGS = (1, 2, 3, 5, 10, 20, 50)


def _proper_motion_endpoint(star, alpha, distance):
    """Return a short proper-motion tail in the RA--distance plane."""
    if star.get("pmra") is None or star.get("pmdec") is None:
        return None
    if star.get("dec") is None:
        return None

    delta = np.deg2rad(float(star["dec"]))
    pmra = float(star["pmra"])
    pmdec = float(star["pmdec"])

    # This is the same projection used by kobe.plot_polar_ra.py.
    vx = -pmra * np.sin(alpha) - pmdec * np.cos(alpha) * np.sin(delta)
    vy = pmra * np.cos(alpha) - pmdec * np.sin(alpha) * np.sin(delta)
    x0 = distance * np.cos(alpha)
    y0 = distance * np.sin(alpha)
    x1 = x0 + TAIL_YEARS * vx * distance / MAS_PER_RAD
    y1 = y0 + TAIL_YEARS * vy * distance / MAS_PER_RAD
    return np.arctan2(y1, x1), np.hypot(x1, y1)


def _polar_limit(distance):
    """Choose a readable radial limit for the star's distance."""
    if distance < 10:
        return 10.0
    if distance < 20:
        return 20.0
    if distance <= 30:
        return 30.0
    if distance <= 50:
        return 50.0
    return 1.2 * distance


def polar_position_plot(star):
    """Plot the star's distance and RA around the Sun.

    A missing distance or coordinate means there is not enough information to
    place the object, so the caller can show a normal Streamlit message.
    """
    if star.get("distance_pc") is None or star.get("ra") is None:
        return None

    distance = float(star["distance_pc"])
    alpha = np.deg2rad(float(star["ra"]))
    rmax = _polar_limit(distance)

    fig, ax = plt.subplots(figsize=(5.4, 5.4),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, rmax)

    rings = [radius for radius in REFERENCE_RINGS if radius <= rmax]
    ax.set_rticks(rings)
    ax.set_yticklabels([f"{radius} pc" for radius in rings],
                       fontsize=10, color="#a1a1aa")
    ax.set_rlabel_position(22)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 60)))
    ax.set_xticklabels([f"{hour}h" for hour in range(0, 24, 4)],
                       fontsize=10, color="#a1a1aa")

    red = "#d8a099"
    ax.yaxis.grid(True, color=red, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_color(red)
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_linestyle((0, (4, 3)))
    ax.spines["polar"].set_alpha(0.7)

    tail = _proper_motion_endpoint(star, alpha, distance)
    if tail is not None:
        theta_end, radius_end = tail
        ax.plot([alpha, theta_end], [distance, radius_end],
                color=red, lw=1.0, alpha=0.9, zorder=3)

    ax.scatter([0], [0], s=85, c="#f4b942", marker="*",
               edgecolor="#111113", linewidth=0.6, zorder=5)
    ax.scatter([alpha], [distance], s=70, c="#6ee7b7",
               edgecolor="#111113", linewidth=0.8, zorder=6)
    ax.set_title("Position around the Sun", color="#f4f4f5",
                 fontsize=13, pad=14)
    fig.tight_layout()
    return fig
