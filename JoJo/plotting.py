"""Plot the projected silhouettes used by the public transit models.

Matplotlib is imported only when :func:`plot_shape` is called, so computing a
light curve still requires NumPy alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .core import validation

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _rotate(x, y, theta):
    """Rotate body-frame coordinates into the sky frame."""
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


def _outline(x, y, theta, center):
    x, y = _rotate(x, y, theta)
    return x + center[0], y + center[1]


def plot_shape(
    model: str,
    transit_parameters: ArrayLike,
    *,
    ax: Axes | None = None,
    show_star: bool = False,
) -> tuple[Figure, Axes]:
    """Draw an oblate, egg, or ringed planet from its light-curve parameters.

    Parameters
    ----------
    model : {'oblate', 'egg', 'ring'}
        Name of the light-curve model. ``spherical_lc`` is not included.
    transit_parameters : sequence of float
        The complete ordered parameter vector accepted by the matching
        ``oblate_lc``, ``egg_lc``, or ``ring_lc`` function. Radii are in stellar
        radii and ``theta`` is in radians. The drawing uses the projected shape,
        orientation, ring ``alpha``, and impact parameter when applicable.
    ax : matplotlib.axes.Axes, optional
        Draw on this axes; create a figure and axes when omitted.
    show_star : bool, default False
        Draw the unit stellar disk and put the planet center at ``(0, impact)``.
        Otherwise center the planet at the origin for a closer shape view.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the drawing.
    ax : matplotlib.axes.Axes
        Axes containing the drawing, with equal coordinate scales and arrows
        through the planet center. The positive x-axis points along transit.

    Notes
    -----
    The drawing shows the geometric projection at the reference transit phase.
    The dashed lines follow the ellipse's own axes, rotated by ``theta``.
    For rings these axes are drawn on the outer ellipse, with
    ``b_out = r_out * (1 - f_r)``.
    The ring fill uses the model's ``alpha`` and the edges remain visible at
    ``alpha=0``. The egg has a right half-axis ``a_p``, a left half-axis ``a_m``,
    and full vertical height ``2 * b_egg``.
    """
    layouts = {"oblate": (11, 6, 8), "egg": (12, 7, 9), "ring": (14, 8, 10)}
    if not isinstance(model, str) or model not in layouts:
        raise ValueError("model must be 'oblate', 'egg', or 'ring'")
    if not isinstance(show_star, (bool, np.bool_)):
        raise ValueError("show_star must be a boolean")
    size, ecc_index, u1_index = layouts[model]
    p = validation.parameters(transit_parameters, size)
    impact = p[1]
    validation.orbit(p[2], p[ecc_index], p[u1_index], p[u1_index + 1])

    if model == "oblate":
        rp_me, f, theta = p[3:6]
        validation.positive_radius(rp_me, "rp_me")
        validation.flattening(f)
        a = rp_me / np.sqrt(1 - f)
        validation.positive_radius(a, "rp_eq")
        b = a * (1 - f)
    elif model == "egg":
        a_p, a_m, b, theta = p[3:7]
        for value, name in ((a_p, "a_p"), (a_m, "a_m"), (b, "b_egg")):
            validation.positive_radius(value, name)
    else:
        rp, ri, ro, f, theta = p[3:8]
        for value, name in ((rp, "r_p"), (ri, "r_in"), (ro, "r_out")):
            validation.positive_radius(value, name)
        validation.flattening(f)
        if not rp <= ri <= ro:
            raise ValueError("Require r_p <= r_in <= r_out")
        alpha = p[13]
        validation.ring_alpha(alpha)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.path import Path

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    center = (0.0, float(impact) if show_star else 0.0)
    if show_star:
        ax.add_patch(
            Circle((0, 0), 1, facecolor="#fff5dc", edgecolor="#b58b43",
                   linewidth=1.7, zorder=0)
        )

    phase = np.linspace(0, 2 * np.pi, 513)
    if model == "oblate":
        x, y = _outline(a * np.cos(phase), b * np.sin(phase), theta, center)
        ax.fill(x, y, color="#4b84c1", alpha=0.42)
        ax.plot(x, y, color="#245c9e", linewidth=2)
        limits = (x, y)
        shape_axes = (
            ((-a, a), (0, 0), (None, r"$r_{\mathrm{eq}}$")),
            ((0, 0), (-b, b), (None, r"$r_{\mathrm{pol}}$")),
        )
    elif model == "egg":
        a_side = np.where(np.cos(phase) >= 0, a_p, a_m)
        x, y = _outline(a_side * np.cos(phase), b * np.sin(phase), theta, center)
        ax.fill(x, y, color="#4b84c1", alpha=0.42)
        ax.plot(x, y, color="#245c9e", linewidth=2)
        limits = (x, y)
        shape_axes = (
            ((-a_m, a_p), (0, 0), (r"$a_-$", r"$a_+$")),
            ((0, 0), (-b, b), (None, r"$b_{\mathrm{egg}}$")),
        )
    else:
        b_out = ro * (1 - f)
        xo, yo = _outline(ro * np.cos(phase), b_out * np.sin(phase), theta, center)
        xi, yi = _outline(ri * np.cos(phase), ri * (1 - f) * np.sin(phase), theta, center)
        outer = np.column_stack((xo, yo))
        inner = np.column_stack((xi, yi))
        vertices = np.vstack((outer, inner[::-1]))
        codes = np.full(len(vertices), Path.LINETO, dtype=np.uint8)
        codes[0] = codes[len(outer)] = Path.MOVETO
        codes[len(outer) - 1] = codes[-1] = Path.CLOSEPOLY
        ax.add_patch(
            PathPatch(Path(vertices, codes), facecolor="#4b84c1",
                      edgecolor="none", alpha=alpha, zorder=1)
        )
        ax.plot(xo, yo, color="#245c9e", linewidth=2, zorder=2)
        ax.plot(xi, yi, color="#245c9e", linewidth=1.5, zorder=2)
        ax.add_patch(
            Circle(center, rp, facecolor="#de8b55", edgecolor="#a8552a",
                   linewidth=1.7, zorder=3)
        )
        limits = (
            np.r_[xo, center[0] - rp, center[0] + rp],
            np.r_[yo, center[1] - rp, center[1] + rp],
        )
        shape_axes = (
            ((-ro, ro), (0, 0), (None, r"$r_{\mathrm{out}}$")),
            ((0, 0), (-b_out, b_out), (None, r"$b_{\mathrm{out}}$")),
        )

    # Use the body frame to place the dashed axes and rotate them with the outline.
    axis_color = "#5c316b"
    cx, cy = center
    for local_x, local_y, labels in shape_axes:
        dx, dy = _rotate(np.asarray(local_x), np.asarray(local_y), theta)
        ax.plot(cx + dx, cy + dy, "--", color=axis_color, linewidth=1.4, zorder=4)
        for x, y, label in zip(dx, dy, labels):
            if label is None:
                continue
            radius = np.hypot(x, y)
            ux, uy = x / radius, y / radius
            offset = (12 * ux - 7 * uy, 12 * uy + 7 * ux)
            ax.annotate(
                label, (cx + x, cy + y), offset, textcoords="offset points",
                ha="center", va="center", color=axis_color, zorder=6,
            )

    # The sky axes extend well beyond the silhouette in both directions.
    xlim, ylim = limits
    length = 1.4 * max(np.max(np.abs(xlim - cx)), np.max(np.abs(ylim - cy)))
    xmin, xmax = cx - length, cx + length
    ymin, ymax = cy - length, cy + length
    arrows = dict(arrowstyle="->", color="0.25", lw=1, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(xmax, cy), xytext=(xmin, cy), arrowprops=arrows, zorder=5)
    ax.annotate("", xy=(cx, ymax), xytext=(cx, ymin), arrowprops=arrows, zorder=5)
    ax.annotate("$x$", (xmax, cy), (4, 0), textcoords="offset points", va="center")
    ax.annotate("$y$", (cx, ymax), (0, 4), textcoords="offset points", ha="center")

    if show_star:
        xmin, xmax = min(xmin, -1), max(xmax, 1)
        ymin, ymax = min(ymin, -1), max(ymax, 1)
    side = max(xmax - xmin, ymax - ymin)
    midx, midy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = side * 0.60
    ax.set_xlim(midx - half, midx + half)
    ax.set_ylim(midy - half, midy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return fig, ax
