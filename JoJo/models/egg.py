"""Transits of two joined half-ellipses sharing a vertical semi-axis."""

import numpy as np
from ..core.orbit import exposure_grid, sky_position, contact_times
from ..core import validation
from ..core.geometry import _arc_signs, solve_star_ellipse_intersections
from ..core.integration import ellipse_integral, star_limb_primitive


def point_inside_egg(x, y, x0, y0, a_p, a_m, b_egg):
    """This is the function describing the shape of the egg.
    U,v stands for the coordinates with origin on the center of the egg.
    a_side determins the points are in a_p or a_m
    return value is the main formula of the egg shape. a_p: right half elipse; a_m: left half elipse
    """
    u = x - x0
    v = y - y0
    a_side = np.where(u >= 0, a_p, a_m)
    return (u / a_side) ** 2 + (v / b_egg) ** 2 <= 1


def classify_star_egg_position(x0, y0, a_p, a_m, b_egg):
    """This is using the same flag idea in jojo (line 54-56,76-77) to illustrate 3 cases"""
    n_points = len(x0)
    d0 = np.sqrt(x0**2 + y0**2)
    r_max = max(
        a_p, a_m, b_egg
    )  # The maximums distance a point inside the egg can reach is either of these 3 values.

    flags = np.zeros(n_points, dtype=int)
    flags[d0 >= (1 + r_max)] = -1  # fully outside
    flags[d0 <= (1 - r_max)] = 1  # fully inside

    return flags


def solve_star_egg_intersections(x0, y0, a_p, a_m, b_egg, epsilon=1e-8):
    """this is the funciton solving egg intersection with circle, corresponding to solve_star_ellipse_intersections function in jojo"""
    for value, name in ((a_p, "a_p"), (a_m, "a_m"), (b_egg, "b_egg")):
        validation.positive_radius(value, name)
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )
    if x0.ndim != 1 or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(y0)):
        raise ValueError("Centers must be finite scalars or one-dimensional arrays")

    n_points = len(
        x0
    )  # this is line 78-80 in jojo. Here there are 8 points, 4 for each half-elipse
    alphas = np.full((n_points, 8), np.nan)
    x_intersect = np.full((n_points, 8), np.nan)
    y_intersect = np.full((n_points, 8), np.nan)

    flags = classify_star_egg_position(
        x0, y0, a_p, a_m, b_egg
    )  # line 76-77 in jojo. I created an isolated function for flags.

    intersect_indices = np.where(flags == 0)[0]  # line 83-88 in jojo.
    if len(intersect_indices) == 0:
        return (flags, x_intersect, y_intersect, alphas)
    xi = x0[intersect_indices]
    yi = y0[intersect_indices]

    f_p = 1.0 - b_egg / a_p
    f_m = 1.0 - b_egg / a_m  # 2 different flattening coefficients

    """This is the new algorithm calculating intersections of the 2 half-elipses respectively"""
    # 1st ellipse
    flags_p, x_p, y_p, alpha_p = solve_star_ellipse_intersections(
        x0=xi, y0=yi, rp_eq=a_p, f=f_p, epsilon=epsilon
    )  # use the intersection function
    right = (~np.isnan(x_p)) & ((x_p - xi[:, None]) >= 0)
    x_p = np.where(right, x_p, np.nan)  # use nan to keep the right side
    y_p = np.where(right, y_p, np.nan)
    alpha_p = np.where(right, alpha_p, np.nan)  # keep only positive x of the first ellipse

    # 2nd ellipse
    flags_m, x_m, y_m, alpha_m = solve_star_ellipse_intersections(
        x0=xi, y0=yi, rp_eq=a_m, f=f_m, epsilon=epsilon
    )
    left = (~np.isnan(x_m)) & ((x_m - xi[:, None]) < 0)
    x_m = np.where(left, x_m, np.nan)  # use nan to keep the left side
    y_m = np.where(left, y_m, np.nan)
    alpha_m = np.where(left, alpha_m, np.nan)

    # combine the intersections from the two half ellipses
    x_all = np.hstack((x_p, x_m))
    y_all = np.hstack((y_p, y_m))
    alpha_all = np.hstack((alpha_p, alpha_m))

    # arrange the intersections according to their angles on the stellar limb: line 121-124 in jojo
    alpha_for_sort = np.where(
        np.isnan(alpha_all), np.inf, alpha_all
    )  # For unreasonable alpha_all, set them infinity
    order = np.argsort(alpha_for_sort, axis=1)  # rise order so that infinities at the back
    x_all = np.take_along_axis(x_all, order, axis=1)  # X,Y coordinates follow the order of alpha.
    y_all = np.take_along_axis(y_all, order, axis=1)
    alpha_all = np.take_along_axis(alpha_all, order, axis=1)

    # count the number of valid intersections: line 112 in jojo
    n_intersections = np.sum(~np.isnan(x_all), axis=1)

    # cases with fewer than two intersections need to be classified again
    d_subset = np.sqrt(xi**2 + yi**2)
    no_crossing = n_intersections < 2
    full = no_crossing & (d_subset < 1)  # full occultation
    outside = no_crossing & (d_subset >= 1)  # outside of the star

    flags[intersect_indices[full]] = 1
    flags[intersect_indices[outside]] = -1

    """arranged valid subset back to original time array"""
    n_columns = x_all.shape[1]
    x_intersect[intersect_indices, :n_columns] = x_all
    y_intersect[intersect_indices, :n_columns] = y_all
    alphas[intersect_indices, :n_columns] = alpha_all

    return flags, x_intersect, y_intersect, alphas


def full_egg_occultation(x0, y0, a_p, a_m, b_egg, u1, u2, n_step=100):
    """flux integration: Call ellipse_integral function:
    integrate the right half from bottom to right tip to top;
    left half from top to left tip to bottom."""
    if len(x0) == 0:
        return np.zeros_like(x0, dtype=float)

    # right half: bottom to top, anticlockwise
    x_right = np.column_stack((x0, x0))
    y_right = np.column_stack((y0 + b_egg, y0 - b_egg))
    flux_right = ellipse_integral(x_right, y_right, x0, y0, a_p, b_egg, u1, u2, n_step)

    # left half: top to bottom, anticlockwise
    x_left = np.column_stack((x0, x0))
    y_left = np.column_stack((y0 - b_egg, y0 + b_egg))
    flux_left = ellipse_integral(x_left, y_left, x0, y0, a_m, b_egg, u1, u2, n_step)

    return flux_right + flux_left


def green_ay(x, y, u1, u2):
    """greens theorem: appendix equation A5 A6 A11 in Liu et al 2025 paper"""
    r2 = x**2 + y**2

    if np.any(r2 > 1 + 1e-8):
        raise ValueError("Green function can only be evaluated inside the stellar disk")

    mu = np.sqrt(np.clip(1 - r2, 0, None))

    first_term = (1 - u1 - 2 * u2) * x
    second_term = u2 / 3 * (3 * y**2 + x**2) * x
    third_term = (u1 / 2 + u2) * (mu * x + (1 - y**2) * np.arctan2(x, mu))

    return first_term + second_term + third_term


def egg_boundary_coordinates(E, x0, y0, a_p, a_m, b_egg):
    """Return points on the counterclockwise egg boundary in the egg-aligned frame."""
    a_side = np.where(np.cos(E) >= 0, a_p, a_m)
    x = x0 + a_side * np.cos(E)
    y = y0 + b_egg * np.sin(E)
    return x, y


def green_line_integral(x, y, u1, u2):
    """Numerically integrate Ay dy."""
    Ay = green_ay(x, y, u1, u2)
    dy = np.diff(y, axis=-1)
    Ay_average = 0.5 * (Ay[..., 1:] + Ay[..., :-1])
    return np.sum(Ay_average * dy, axis=-1)


def egg_boundary_parameter(x, y, x0, y0, a_p, a_m, b_egg):
    """line 204-209 in jojo: Convert a point on the egg boundary to the anticlockwise egg parameter E."""
    u = x - x0
    v = y - y0
    a_side = np.where(u >= 0, a_p, a_m)

    E = np.arctan2(v / b_egg, u / a_side)
    E = np.where(E < -np.pi / 2, E + 2 * np.pi, E)

    return E


def rotate_to_egg_frame(x, y, theta_egg):
    """line 332-333 in jojo: rotation clockwisely by theta, rotate the center of the planet to egg frame"""
    x_rot = x * np.cos(theta_egg) + y * np.sin(theta_egg)
    y_rot = -x * np.sin(theta_egg) + y * np.cos(theta_egg)

    return x_rot, y_rot


def partial_egg_occultation(
    alphas, x_intersect, y_intersect, x0, y0, a_p, a_m, b_egg, u1, u2, n_step=30
):
    """"""
    if len(x0) == 0:
        return np.zeros_like(x0, dtype=float)

    n_intersections = np.sum(~np.isnan(x_intersect), axis=1)

    if np.any(n_intersections != 2):
        raise ValueError("Partial occultation currently requires exactly two intersections")

    # identify the stellar boundary arc inside the egg
    alpha_0 = alphas[:, 0]
    alpha_1 = alphas[:, 1]
    star_direct = (
        _arc_signs(
            alpha_0,
            alpha_1,
            lambda t: ((np.cos(t) - x0) / np.where(np.cos(t) >= x0, a_p, a_m)) ** 2
            + ((np.sin(t) - y0) / b_egg) ** 2
            - 1,
        )
        < 0
    )

    alpha_start = np.where(star_direct, alpha_0, alpha_1)
    alpha_end = np.where(star_direct, alpha_1, alpha_0 + 2 * np.pi)

    flux_star = star_limb_primitive(alpha_end, u1, u2) - star_limb_primitive(alpha_start, u1, u2)
    # use star_limb_primitive function to calculate the contribution from stellar arc

    # Identify the egg boundary arc using open-arc samples, not a possible tangent.
    e_0 = egg_boundary_parameter(x_intersect[:, 0], y_intersect[:, 0], x0, y0, a_p, a_m, b_egg)
    e_1 = egg_boundary_parameter(x_intersect[:, 1], y_intersect[:, 1], x0, y0, a_p, a_m, b_egg)

    e_low = np.minimum(e_0, e_1)
    e_high = np.maximum(e_0, e_1)
    egg_direct = (
        _arc_signs(
            e_low,
            e_high,
            lambda t: (x0 + np.where(np.cos(t) >= 0, a_p, a_m) * np.cos(t)) ** 2
            + (y0 + b_egg * np.sin(t)) ** 2
            - 1,
        )
        < 0
    )

    e_start = np.where(egg_direct, e_low, e_high)
    e_end = np.where(egg_direct, e_high, e_low + 2 * np.pi)

    flux_egg = np.zeros_like(x0, dtype=float)

    # split the egg boundary at the connection points of the two half ellipses
    intervals = [
        (-np.pi / 2, np.pi / 2, a_p),
        (np.pi / 2, 3 * np.pi / 2, a_m),
        (3 * np.pi / 2, 5 * np.pi / 2, a_p),
        (5 * np.pi / 2, 7 * np.pi / 2, a_m),
    ]

    for E_left, E_right, a_side in intervals:
        segment_start = np.maximum(e_start, E_left)
        segment_end = np.minimum(e_end, E_right)

        VALID = segment_end > segment_start + 1e-12

        if not np.any(VALID):
            continue

        x_center = x0[VALID]
        y_center = y0[VALID]

        x_start = x_center + a_side * np.cos(segment_start[VALID])
        y_start = y_center + b_egg * np.sin(segment_start[VALID])

        x_end = x_center + a_side * np.cos(segment_end[VALID])
        y_end = y_center + b_egg * np.sin(segment_end[VALID])

        # mirror negative-x segments only during integration
        MIRROR = x_center < 0

        x_bound = np.column_stack(
            (np.where(MIRROR, -x_start, x_end), np.where(MIRROR, -x_end, x_start))
        )
        y_bound = np.column_stack(
            (np.where(MIRROR, y_start, y_end), np.where(MIRROR, y_end, y_start))
        )

        flux_egg[VALID] += ellipse_integral(
            x_bound, y_bound, np.abs(x_center), y_center, a_side, b_egg, u1, u2, n_step
        )

    return flux_star + flux_egg


def egg_occultation(x0, y0, a_p, a_m, b_egg, u1, u2, n_step=80, epsilon=1e-8):
    """combine full and partial occultation together."""
    validation.positive_integer(n_step, "n_step")
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )

    flags, x_intersect, y_intersect, alphas = solve_star_egg_intersections(
        x0, y0, a_p, a_m, b_egg, epsilon
    )

    delta_flux = np.zeros_like(x0, dtype=float)

    full = flags == 1
    intersect = flags == 0

    if np.any(full):
        delta_flux[full] = full_egg_occultation(x0[full], y0[full], a_p, a_m, b_egg, u1, u2, n_step)

    if np.any(intersect):
        delta_flux[intersect] = partial_egg_occultation(
            alphas[intersect],
            x_intersect[intersect],
            y_intersect[intersect],
            x0[intersect],
            y0[intersect],
            a_p,
            a_m,
            b_egg,
            u1,
            u2,
            n_step,
        )

    return delta_flux, flags


def egg_lc(
    transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=64
):
    """Return (flux, sampled contacts) for joined half-ellipses.

    Parameters: [tc, impact, period, a_p, a_m, b_egg, theta, ecc, omega,
    u1, u2, log10_rho_star]. Angles radians, time days, radii / R_star.
    n_step is the actual Gauss-Legendre node count per half-ellipse arc.
    """
    p = validation.parameters(transit_parameters, 12)
    tc, impact, period, ap, am, b, theta, ecc, omega, u1, u2, density = p
    validation.orbit(period, ecc, u1, u2)
    validation.positive_integer(n_step, "n_step")
    for value, name in ((ap, "a_p"), (am, "a_m"), (b, "b_egg")):
        validation.positive_radius(value, name)
    time, samples, count = exposure_grid(time_array, exp_time, supersample_factor)
    if not time.size:
        return np.empty(0), []
    x, y, front = sky_position(tc, impact, period, ecc, omega, density, samples, photo_ecc)
    x, y = rotate_to_egg_frame(x, y, theta)
    # Geometry behind the star does not contribute a primary transit.
    x = np.where(front, x, 3.0)
    blocked, flags = egg_occultation(x, y, ap, am, b, u1, u2, n_step, 1e-10)
    flux = 1 - blocked / (np.pi * (1 - u1 / 3 - u2 / 6))
    return flux.reshape(-1, count).mean(axis=1), contact_times(flags, samples)
