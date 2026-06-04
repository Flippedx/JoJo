"""JAX implementation of the oblate-planet light-curve model.

This module is intentionally separate from ``jojo_oblate`` so normal NumPy
usage does not require importing JAX. The first target is differentiable flux
evaluation for gradient-based samplers and Fisher-matrix calculations.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ._solve_quartic_jax import get_roots


def intransit_flag(t0, period, duration, time):
    time_debias = (time - t0) % period
    time_debias = jnp.where(time_debias > period / 2, time_debias - period, time_debias)
    return (time_debias >= -duration / 2) & (time_debias <= duration / 2)


def mean_to_true(mean_anomaly, ecc, n_iter=20):
    """Convert mean anomaly to true anomaly with a fixed Newton solve."""
    mean_anomaly = jnp.asarray(mean_anomaly)

    def body(_, eccentric_anomaly):
        residual = eccentric_anomaly - ecc * jnp.sin(eccentric_anomaly) - mean_anomaly
        derivative = 1 - ecc * jnp.cos(eccentric_anomaly)
        return eccentric_anomaly - residual / derivative

    eccentric_anomaly = jax.lax.fori_loop(0, n_iter, body, mean_anomaly)
    true_anomaly = 2 * jnp.arctan2(
        jnp.sqrt(1 + ecc) * jnp.sin(eccentric_anomaly / 2),
        jnp.sqrt(1 - ecc) * jnp.cos(eccentric_anomaly / 2),
    )
    return jnp.where(ecc == 0, mean_anomaly, true_anomaly)


def true_to_mean(true_anomaly, ecc):
    eccentric_anomaly = 2 * jnp.arctan2(
        jnp.sqrt(1 - ecc) * jnp.sin(true_anomaly / 2),
        jnp.sqrt(1 + ecc) * jnp.cos(true_anomaly / 2),
    )
    return eccentric_anomaly - ecc * jnp.sin(eccentric_anomaly)


def _solve_star_circle_intersections(x0, y0, rp_eq):
    """Analytic fixed-shape intersections for the spherical limit."""
    d0 = jnp.sqrt(x0**2 + y0**2)
    outside = d0 >= (1 + rp_eq)
    inside = d0 <= (1 - rp_eq)
    candidate = ~(outside | inside)

    safe_d0 = jnp.where(candidate, d0, 1.0)
    cos_delta = (1 + safe_d0**2 - rp_eq**2) / (2 * safe_d0)
    delta_alpha = jnp.arccos(jnp.clip(cos_delta, -1, 1))
    alpha0 = jnp.arctan2(y0, x0)
    alpha_low = alpha0 - delta_alpha
    alpha_high = alpha0 + delta_alpha

    alphas = jnp.stack([alpha_low, alpha_high, jnp.zeros_like(alpha0), jnp.zeros_like(alpha0)], axis=1)
    x_intersect = jnp.cos(alphas)
    y_intersect = jnp.sin(alphas)
    flags = jnp.zeros_like(x0, dtype=jnp.int32)
    flags = jnp.where(outside, -1, flags)
    flags = jnp.where(inside, 1, flags)
    return jax.lax.stop_gradient(flags), x_intersect, y_intersect, alphas


def _solve_star_ellipse_intersections(x0, y0, rp_eq, oblateness, epsilon=1e-8, polish_iter=10):
    """Fixed-shape JAX solver for a genuinely oblate ellipse."""
    d0 = jnp.sqrt(x0**2 + y0**2)
    rough_outside = d0 >= (1 + rp_eq)
    rough_inside = d0 <= (1 - rp_eq)
    candidate = ~(rough_outside | rough_inside)

    x_solve = jnp.where(candidate, x0, 0.5)
    y_solve = jnp.where(candidate, y0, 0.25)
    b2 = (1 - oblateness) ** 2
    r2 = rp_eq**2
    c0 = jnp.full_like(x_solve, 1 - b2, dtype=jnp.complex128)
    c1 = 4 * b2 * x_solve - 4j * y_solve
    c2 = -2 * (1 + b2 - 2 * b2 * r2 + 2 * b2 * x_solve**2 + 2 * y_solve**2)
    c3 = 4 * b2 * x_solve + 4j * y_solve
    c4 = c0
    coefficients = jnp.stack([c0, c1, c2, c3, c4], axis=1)
    roots = get_roots(coefficients, max_iter=polish_iter)
    xp = jnp.real(roots)
    yp = jnp.imag(roots)
    dist_squared = (xp - x_solve[:, None]) ** 2 + (yp - y_solve[:, None]) ** 2 / b2
    valid = candidate[:, None] & (jnp.abs(dist_squared - r2) < epsilon)
    n_intersection = jnp.sum(valid, axis=1)

    angles = jnp.angle(roots)
    sort_key = jnp.where(valid, angles, jnp.inf)
    order = jnp.argsort(sort_key, axis=1)
    angles = jnp.take_along_axis(angles, order, axis=1)
    roots = jnp.take_along_axis(roots, order, axis=1)

    safe_angles = jnp.where(jnp.isfinite(jnp.take_along_axis(sort_key, order, axis=1)), angles, 0.0)
    safe_x = jnp.real(roots)
    safe_y = jnp.imag(roots)
    safe_x = jnp.where(jnp.isfinite(jnp.take_along_axis(sort_key, order, axis=1)), safe_x, 1.0)
    safe_y = jnp.where(jnp.isfinite(jnp.take_along_axis(sort_key, order, axis=1)), safe_y, 0.0)

    flags = jnp.zeros_like(x0, dtype=jnp.int32)
    flags = jnp.where(rough_outside, -1, flags)
    flags = jnp.where(rough_inside, 1, flags)
    flags = jnp.where(candidate & (n_intersection < 2) & (d0 < 1), 1, flags)
    flags = jnp.where(candidate & (n_intersection < 2) & (d0 >= 1), -1, flags)
    return (
        jax.lax.stop_gradient(flags),
        safe_x,
        safe_y,
        safe_angles,
    )


def solve_star_ellipse_intersections(x0, y0, rp_eq, oblateness, epsilon=1e-8, polish_iter=10):
    """Solve stellar-limb intersections, including the spherical limit."""
    return jax.lax.cond(
        jnp.abs(oblateness) < 1e-12,
        lambda: _solve_star_circle_intersections(x0, y0, rp_eq),
        lambda: _solve_star_ellipse_intersections(x0, y0, rp_eq, oblateness, epsilon, polish_iter),
    )


def full_occultation(x0, y0, rp_eq, oblateness, u1, u2):
    a2 = rp_eq**2
    d2 = x0**2 + y0**2
    delta_flux_analytic = jnp.pi * (1 - oblateness) * a2 * (
        1 - u1 - 2 * u2 + u2 / 4 * ((2 - 2 * oblateness + oblateness**2) * a2 + 4 * d2)
    )

    n_step = 30
    angles_edge = jnp.linspace(0, 2 * jnp.pi, n_step + 1)
    delta_angle = 2 * jnp.pi / n_step
    angles = (angles_edge[:-1] + angles_edge[1:]) / 2
    delta_x = rp_eq * jnp.cos(angles)
    delta_y = (1 - oblateness) * rp_eq * jnp.sin(angles)
    prefac = (u1 / 2 + u2) * (1 - oblateness)
    xp = x0[:, None] + delta_x
    yp = y0[:, None] + delta_y
    r2 = jnp.minimum(xp**2 + yp**2, 1 - 1e-10)
    mu_p = jnp.sqrt(1 - r2)
    integrand = (mu_p * xp + (1 - yp**2) * jnp.arctan(xp / mu_p)) * delta_x[None, :]
    delta_flux_numeric = jnp.sum(integrand, axis=1) * prefac * delta_angle
    return delta_flux_analytic + delta_flux_numeric


def partial_occultation(alphas, x_intersect, y_intersect, x0, y0, rp_eq, oblateness, u1, u2, n_step=100):
    alphas = alphas[:, :2]
    x_intersect = x_intersect[:, :2]
    y_intersect = y_intersect[:, :2]

    delta_alpha = alphas[:, 1] - alphas[:, 0]
    delta_flux_limb = (
        (0.5 - u1 / 2 - 0.75 * u2) * delta_alpha
        + (0.5 - u1 / 2 - 11 * u2 / 12) * (x_intersect[:, 1] * y_intersect[:, 1] - x_intersect[:, 0] * y_intersect[:, 0])
        + u2 / 6 * (x_intersect[:, 1] * y_intersect[:, 1] ** 3 - x_intersect[:, 0] * y_intersect[:, 0] ** 3)
    )
    y0_i = y_intersect[:, 0]
    y1_i = y_intersect[:, 1]
    limb_negative = (u1 + 2 * u2) * jnp.pi / 4 * (4 / 3 - y1_i + y1_i**3 / 3 - y0_i + y0_i**3 / 3)
    limb_positive = (u1 + 2 * u2) * jnp.pi / 4 * (y1_i - y1_i**3 / 3 - y0_i + y0_i**3 / 3)
    delta_flux_limb = delta_flux_limb + jnp.where(x_intersect[:, 1] < 0, limb_negative, limb_positive)

    cos_e = (x_intersect - x0[:, None]) / rp_eq
    sin_e = (y_intersect - y0[:, None]) / rp_eq / (1 - oblateness)
    cos_e = jnp.clip(cos_e, -1, 1)
    e_anom = jnp.arccos(cos_e)
    e_anom = jnp.where(sin_e < 0, -e_anom, e_anom)
    e_anom = e_anom.at[:, 0].set(jnp.where(e_anom[:, 1] > e_anom[:, 0], e_anom[:, 0] + 2 * jnp.pi, e_anom[:, 0]))
    delta_e = e_anom[:, 0] - e_anom[:, 1]

    first_integral = (
        0.5 * (1 - oblateness) * rp_eq**2 * delta_e
        + 0.5 * (x_intersect[:, 0] * y_intersect[:, 0] - x_intersect[:, 1] * y_intersect[:, 1])
        + 0.5 * x0 * (y_intersect[:, 0] - y_intersect[:, 1])
        - 0.5 * y0 * (x_intersect[:, 0] - x_intersect[:, 1])
    )

    b = (1 - oblateness) * rp_eq
    a2 = rp_eq**2
    b2 = b**2
    ab = (1 - oblateness) * a2
    x2 = x0**2
    y2 = y0**2
    e0 = e_anom[:, 0]
    e1 = e_anom[:, 1]
    second_integral = (
        3 / 8 * ab * (a2 + b2 + 4 * x2 + 4 * y2) * delta_e
        - 1.5 * rp_eq * b2 * y0 * (jnp.cos(e0) - jnp.cos(e1))
        - 1.5 * b2 * x0 * y0 * (jnp.cos(2 * e0) - jnp.cos(2 * e1))
        - 0.5 * rp_eq * b2 * y0 * (jnp.cos(3 * e0) - jnp.cos(3 * e1))
        + 0.25 * b * x0 * (9 * a2 + 3 * b2 + 4 * x2 + 12 * y2) * (jnp.sin(e0) - jnp.sin(e1))
        + 0.25 * ab * (a2 + 3 * x2 + 3 * y2) * (jnp.sin(2 * e0) - jnp.sin(2 * e1))
        + 0.25 * b * x0 * (a2 - b2) * (jnp.sin(3 * e0) - jnp.sin(3 * e1))
        + 0.03125 * ab * (a2 - 3 * b2) * (jnp.sin(4 * e0) - jnp.sin(4 * e1))
    )
    delta_flux_analytic = (1 - u1 - 2 * u2) * first_integral + u2 / 3 * second_integral

    delta_angles = delta_e / n_step
    prefac = (u1 / 2 + u2) * (1 - oblateness)
    frac = (jnp.arange(n_step) + 0.5) / n_step
    angle = e_anom[:, 1, None] + delta_e[:, None] * frac[None, :]
    xp = x0[:, None] + rp_eq * jnp.cos(angle)
    yp = y0[:, None] + rp_eq * jnp.sin(angle) * (1 - oblateness)
    r2 = jnp.minimum(xp**2 + yp**2, 1 - 1e-10)
    mu_p = jnp.sqrt(1 - r2)
    integrand = (mu_p * xp + (1 - yp**2) * (jnp.arctan(xp / mu_p) - jnp.pi / 2)) * (xp - x0[:, None])
    delta_flux_numeric = jnp.sum(integrand, axis=1) * delta_angles * prefac
    delta_flux_numeric += jnp.pi / 2 * (u1 / 2 + u2) * (y0_i - y0_i**3 / 3 - y1_i + y1_i**3 / 3)
    return delta_flux_limb + delta_flux_analytic + delta_flux_numeric


def _sky_position(transit_parameters, time_array, photo_ecc=False):
    tc, b, period, rp_me, oblateness, obliquity, ecc, omega, u1, u2, log10_rho_star = transit_parameters
    rho_star = 10**log10_rho_star
    rho_circ = jnp.where(
        photo_ecc,
        rho_star * ((1 + ecc * jnp.sin(omega)) / jnp.sqrt(1 - ecc**2)) ** 3,
        rho_star,
    )
    a_over_rstar = 3.753 * (period**2 * rho_circ) ** (1 / 3)

    def photoecc_position():
        time_debias = (time_array - tc) % period
        time_debias = jnp.where(time_debias > period / 2, time_debias - period, time_debias)
        return time_debias / period * 2 * jnp.pi * a_over_rstar, jnp.ones_like(time_array) * b

    def eccentric_position():
        tp = tc - period / (2 * jnp.pi) * true_to_mean(jnp.pi / 2 - omega, ecc)
        mean_anomaly = 2 * jnp.pi / period * (time_array - tp)
        true_anomaly = mean_to_true(mean_anomaly, ecc)
        radius_factor = a_over_rstar * (1 - ecc**2) / (1 + ecc * jnp.cos(true_anomaly))
        x_ini = -radius_factor * jnp.cos(omega + true_anomaly)
        y_ini = (1 + ecc * jnp.sin(omega)) / (1 + ecc * jnp.cos(true_anomaly)) * jnp.sin(omega + true_anomaly) * b
        return x_ini, y_ini

    x_ini, y_ini = jax.lax.cond(photo_ecc, photoecc_position, eccentric_position)
    x0 = x_ini * jnp.cos(obliquity) + y_ini * jnp.sin(obliquity)
    y0 = -x_ini * jnp.sin(obliquity) + y_ini * jnp.cos(obliquity)
    return jnp.abs(x0), jnp.abs(y0)


def _oblate_lc_instant(transit_parameters, time_array, photo_ecc=False, n_step=100):
    """Return a differentiable oblate-planet light curve.

    This JAX API returns only the flux array. It intentionally omits contact
    times because contact extraction is discrete and not useful for gradients.
    Exactly spherical planets use an analytic circle-intersection branch.
    """
    _, _, _, rp_me, oblateness, _, _, _, u1, u2, _ = transit_parameters
    rp_eq = rp_me / jnp.sqrt(1 - oblateness)
    use_spherical = oblateness * rp_eq**2 < 1e-6
    effective_rp = jnp.where(use_spherical, rp_me, rp_eq)
    effective_oblateness = jnp.where(use_spherical, 0.0, oblateness)
    x0, y0 = _sky_position(transit_parameters, time_array, photo_ecc=photo_ecc)

    flags, x_intersect, y_intersect, alphas = solve_star_ellipse_intersections(x0, y0, effective_rp, effective_oblateness)
    flux_total = jnp.pi * (6 - 2 * u1 - u2) / 6

    is_partial = flags == 0
    safe_x0 = jnp.where(is_partial, x0, 0.0)
    safe_y0 = jnp.where(is_partial, y0, 0.0)
    fallback_x = jnp.array([0.5 * effective_rp, -0.5 * effective_rp, 0.5 * effective_rp, -0.5 * effective_rp])
    fallback_y = jnp.array([0.25 * effective_rp, 0.25 * effective_rp, -0.25 * effective_rp, -0.25 * effective_rp])
    fallback_alpha = jnp.array([0.25, 0.75, 0.0, 0.0])
    safe_x_intersect = jnp.where(is_partial[:, None], x_intersect, fallback_x[None, :])
    safe_y_intersect = jnp.where(is_partial[:, None], y_intersect, fallback_y[None, :])
    safe_alphas = jnp.where(is_partial[:, None], alphas, fallback_alpha[None, :])
    partial_flux = partial_occultation(
        safe_alphas,
        safe_x_intersect,
        safe_y_intersect,
        safe_x0,
        safe_y0,
        effective_rp,
        effective_oblateness,
        u1,
        u2,
        n_step,
    )
    full_flux = full_occultation(x0, y0, effective_rp, effective_oblateness, u1, u2)
    delta_flux = jnp.where(is_partial, partial_flux, 0.0)
    delta_flux = jnp.where(flags == 1, full_flux, delta_flux)
    return 1 - delta_flux / flux_total


@partial(jax.jit, static_argnames=("photo_ecc", "exp_time", "supersample_factor", "n_step"))
def oblate_lc(transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=100):
    """Return a differentiable oblate-planet light curve.

    This JAX API returns only the flux array. It intentionally omits contact
    times because contact extraction is discrete and not useful for gradients.
    Pass a static ``exp_time`` >= 0.003 to enable supersampling that mirrors the
    NumPy long-exposure path. When ``exp_time`` is ``None`` no automatic
    cadence-based supersampling is applied.

    Intersection roots are refined with an Aberth-Ehrlich ``custom_root``.
    JAX therefore differentiates the converged polynomial root condition
    implicitly instead of backpropagating through the iterative solver.
    Gradients at exact contact points remain non-smooth because the occultation
    topology changes there.
    """
    transit_parameters = jnp.asarray(transit_parameters)
    time_array = jnp.asarray(time_array)
    if exp_time is not None and exp_time >= 0.003:
        exp_time_begin = time_array - 0.5 * exp_time
        exp_time_end = time_array + 0.5 * exp_time
        time_array_supersample = jnp.linspace(exp_time_begin, exp_time_end, supersample_factor).T.flatten()
        flux = _oblate_lc_instant(transit_parameters, time_array_supersample, photo_ecc=photo_ecc, n_step=n_step)
        return jnp.mean(flux.reshape((-1, supersample_factor)), axis=1)

    return _oblate_lc_instant(transit_parameters, time_array, photo_ecc=photo_ecc, n_step=n_step)


jax_oblate_lc = oblate_lc


def flux_jacobian(transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=100):
    return jax.jacobian(
        lambda p: oblate_lc(
            p,
            time_array,
            photo_ecc=photo_ecc,
            exp_time=exp_time,
            supersample_factor=supersample_factor,
            n_step=n_step,
        )
    )(transit_parameters)


def fisher_matrix(transit_parameters, time_array, flux_err, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=100):
    jac = flux_jacobian(
        transit_parameters,
        time_array,
        photo_ecc=photo_ecc,
        exp_time=exp_time,
        supersample_factor=supersample_factor,
        n_step=n_step,
    )
    weights = jnp.broadcast_to(1 / jnp.asarray(flux_err) ** 2, time_array.shape)
    return jac.T @ (jac * weights[:, None])
