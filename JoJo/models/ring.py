"""Spherical planets surrounded by projected elliptical rings."""

import numpy as np
from ..core.geometry import (
    _arc_signs,
    solve_star_ellipse_intersections,
    solve_planet_ring_intersections,
)
from ..core.integration import ellipse_integral
from ..core.integration import star_limb_primitive


def _selected_arcs(x0, y0, a, b, intersections, residual, u1, u2, n_step, stellar=False):
    """Partition a complete boundary at crossings and integrate retained arcs.

    Include pi to split a closed ellipse into nonzero-endpoint arcs. Duplicate
    roots/tangencies yield zero-length intervals and contribute nothing.
    """
    angles = []
    for x, y in intersections:
        angle = np.mod(np.arctan2((y - y0[:, None]) / b, (x - x0[:, None]) / a), 2 * np.pi)
        angles.append(np.where(np.isfinite(angle), angle, 2 * np.pi))
    stops = np.broadcast_to([0.0, np.pi, 2 * np.pi], (len(x0), 3))
    edges = np.sort(np.concatenate([stops] + angles, axis=1), axis=1)
    total = np.zeros_like(x0)
    for i in range(edges.shape[1] - 1):
        start, end = edges[:, i], edges[:, i + 1]
        signs = _arc_signs(start, end, lambda t: residual(x0 + a * np.cos(t), y0 + b * np.sin(t)))
        use = (end - start > 1e-12) & (signs < 0)
        if not np.any(use):
            continue
        if stellar:
            total[use] += star_limb_primitive(end[use], u1, u2) - star_limb_primitive(
                start[use], u1, u2
            )
        else:
            bounds = np.column_stack((end[use], start[use]))
            xb = x0[use, None] + a * np.cos(bounds)
            yb = y0[use, None] + b * np.sin(bounds)
            total[use] += ellipse_integral(xb, yb, x0[use], y0[use], a, b, u1, u2, n_step)
    return total


def ring_occultation(x0, y0, r_p, r_in, r_out, f_r, u1, u2, opacity=1.0, n_step=32):
    """Unnormalized planet + visible-ring flux; projected ring/planet overlap counted once."""
    from .oblate import ellipse_occultation
    from ..core import validation

    validation.positive_integer(n_step, "n_step")
    validation.flattening(f_r)
    for value, name in ((r_p, "r_p"), (r_in, "r_in"), (r_out, "r_out")):
        validation.positive_radius(value, name)
    if not r_p <= r_in <= r_out:
        raise ValueError("Require r_p <= r_in <= r_out")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("opacity must lie in [0, 1]")
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )
    x0, y0 = np.abs(x0), np.abs(y0)
    planet, _ = ellipse_occultation(x0, y0, r_p, r_p, u1, u2, n_step)
    if opacity == 0 or r_in == r_out or not x0.size:
        return planet
    if r_p <= r_in * (1 - f_r):
        outer, _ = ellipse_occultation(x0, y0, r_out, r_out * (1 - f_r), u1, u2, n_step)
        inner, _ = ellipse_occultation(x0, y0, r_in, r_in * (1 - f_r), u1, u2, n_step)
        return planet + opacity * (outer - inner)
    _, x_sp, y_sp, alpha_sp = solve_star_ellipse_intersections(x0, y0, r_p, 0, 1e-10)
    _, x_si, y_si, alpha_si = solve_star_ellipse_intersections(x0, y0, r_in, f_r, 1e-10)
    _, x_so, y_so, alpha_so = solve_star_ellipse_intersections(x0, y0, r_out, f_r, 1e-10)
    x_pi, y_pi, x_po, y_po = solve_planet_ring_intersections(x0, y0, r_p, r_in, r_out, f_r)

    def outside_planet_in_star(x, y):
        return np.maximum(x * x + y * y - 1, r_p * r_p - (x - x0) ** 2 - (y - y0) ** 2)

    def in_annulus(x, y):
        radius2 = (x - x0) ** 2 + ((y - y0) / (1 - f_r)) ** 2
        return np.maximum(r_in * r_in - radius2, radius2 - r_out * r_out)

    # Positive outer/stellar boundaries; negative inner/planet-hole boundaries.
    outer = _selected_arcs(
        x0,
        y0,
        r_out,
        r_out * (1 - f_r),
        [(x_so, y_so), (x_po, y_po)],
        outside_planet_in_star,
        u1,
        u2,
        n_step,
    )
    inner = _selected_arcs(
        x0,
        y0,
        r_in,
        r_in * (1 - f_r),
        [(x_si, y_si), (x_pi, y_pi)],
        outside_planet_in_star,
        u1,
        u2,
        n_step,
    )
    planet_edge = _selected_arcs(
        x0,
        y0,
        r_p,
        r_p,
        [(x_sp, y_sp), (x_pi, y_pi), (x_po, y_po)],
        lambda x, y: np.maximum(x * x + y * y - 1, in_annulus(x, y)),
        u1,
        u2,
        n_step,
    )
    star = _selected_arcs(
        np.zeros_like(x0),
        np.zeros_like(y0),
        1,
        1,
        [(x_sp, y_sp), (x_si, y_si), (x_so, y_so)],
        lambda x, y: np.maximum(in_annulus(x, y), r_p * r_p - (x - x0) ** 2 - (y - y0) ** 2),
        u1,
        u2,
        n_step,
        stellar=True,
    )
    ring = outer - inner - planet_edge + star
    return planet + opacity * ring


def ring_lc(
    transit_parameters,
    time_array,
    photo_ecc=False,
    n_step=32,
    *,
    exp_time=None,
    supersample_factor=5,
):
    """Return normalized planet + ring flux.

    Parameters: [tc, impact, period, r_p, r_in, r_out, f_r, theta, ecc, omega,
    u1, u2, log10_rho_star, opacity]. Ring radii are equatorial semi-axes.
    All components use the same exposure samples. n_step is the actual number
    of Gauss-Legendre nodes per arc.
    Only exposures > 5/1440 days use supersampling; None infers the median cadence.
    """
    from ..core import validation
    from ..core.orbit import exposure_grid, sky_position

    p = validation.parameters(transit_parameters, 14)
    tc, impact, period, rp, ri, ro, f, theta, ecc, omega, u1, u2, density, opacity = p
    validation.orbit(period, ecc, u1, u2)
    time, samples, count = exposure_grid(time_array, exp_time, supersample_factor)
    x, y, front = sky_position(tc, impact, period, ecc, omega, density, samples, photo_ecc)
    xc = x * np.cos(theta) + y * np.sin(theta)
    yc = -x * np.sin(theta) + y * np.cos(theta)
    xc = np.where(front, xc, 3.0)
    blocked = ring_occultation(xc, yc, rp, ri, ro, f, u1, u2, opacity, n_step)
    if not time.size:
        return np.empty(0)
    flux = 1 - blocked / (np.pi * (1 - u1 / 3 - u2 / 6))
    return flux.reshape(-1, count).mean(axis=1)
