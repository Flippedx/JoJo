"""Spherical and oblate planet transits, with shared geometry and quadrature."""

import numpy as np
from ..core._occultquad import occultquad
from ..core import validation
from ..core.orbit import exposure_grid, sky_position, contact_times
from ..core.geometry import solve_star_ellipse_intersections
from ..core.integration import full_occultation, partial_occultation


def ellipse_occultation(x0, y0, a, b, u1, u2, n_step=32):
    """Unnormalized blocked flux and flags for an axis-aligned ellipse."""
    validation.positive_integer(n_step, "n_step")
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )
    f = 1 - b / a
    flags, xi, yi, alpha = solve_star_ellipse_intersections(x0, y0, a, f, 1e-10)
    blocked = np.zeros_like(x0)
    full, part = flags == 1, flags == 0
    if np.any(full):
        blocked[full] = full_occultation(flags[full], x0[full], y0[full], a, f, u1, u2, n_step)
    if np.any(part):
        blocked[part] = partial_occultation(
            flags[part], alpha[part], xi[part], yi[part], x0[part], y0[part], a, f, u1, u2, n_step
        )
    return blocked, flags


def _inputs(transit_parameters, time_array, photo_ecc, exp_time, supersample_factor):
    p = validation.parameters(transit_parameters, 11)
    tc, b, period, rp, f, theta, ecc, omega, u1, u2, density = p
    validation.flattening(f)
    validation.positive_radius(rp, "rp_me")
    validation.positive_radius(rp / np.sqrt(1 - f), "rp_eq")
    validation.orbit(period, ecc, u1, u2)
    time, samples, count = exposure_grid(time_array, exp_time, supersample_factor)
    x, y, front = sky_position(tc, b, period, ecc, omega, density, samples, photo_ecc)
    return p, time, samples, count, x, y, front


def oblate_lc(
    transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=32
):
    """Return (normalized flux, sampled contacts) for an oblate planet.

    Parameters: [tc, impact, period, rp_me, f, theta, ecc, omega, u1, u2,
    log10_rho_star]. Radii are in stellar-radius units; rp_me is the equal-area
    radius, angles radians, times days, density g/cm^3. n_step is the actual
    Gauss-Legendre node count per boundary arc (also for full occultation).
    exp_time=0 selects instantaneous flux; None infers exposure from cadence.
    Only exposures > 5/1440 days (five minutes) use supersampling.
    photo_ecc=True uses the constant-velocity approximation near transit.
    """
    validation.positive_integer(n_step, "n_step")
    p, time, samples, count, x, y, front = _inputs(
        transit_parameters, time_array, photo_ecc, exp_time, supersample_factor
    )
    tc, impact, period, rp, f, theta, ecc, omega, u1, u2, density = p
    if not time.size:
        return np.empty(0), []
    a = rp / np.sqrt(1 - f)
    # Retained historical approximation; unsuitable for shape gradients below
    # this threshold. The low-level ellipse kernel itself handles near-circles.
    if f * a * a < 1e-6:
        return spherical_lc(p, time, photo_ecc, exp_time, supersample_factor)
    xc = x * np.cos(theta) + y * np.sin(theta)
    yc = -x * np.sin(theta) + y * np.cos(theta)
    blocked, flags = ellipse_occultation(xc, yc, a, a * (1 - f), u1, u2, n_step)
    blocked[~front], flags[~front] = 0, -1
    flux = 1 - blocked / (np.pi * (1 - u1 / 3 - u2 / 6))
    return flux.reshape(-1, count).mean(axis=1), contact_times(flags, samples)


def spherical_lc(
    transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5
):
    """Equal-area spherical comparison; parameter order matches oblate_lc."""
    p, time, samples, count, x, y, front = _inputs(
        transit_parameters, time_array, photo_ecc, exp_time, supersample_factor
    )
    if not time.size:
        return np.empty(0), []
    rp, u1, u2 = p[3], p[8], p[9]
    d = np.hypot(x, y)
    d = np.where(front, d, 2.0)
    flux = occultquad(d, u1, u2, rp)
    flags = np.where(d >= 1 + rp, -1, np.where(d <= 1 - rp, 1, 0))
    return flux.reshape(-1, count).mean(axis=1), contact_times(flags, samples)
