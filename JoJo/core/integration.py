"""Green boundary integrals for quadratic stellar limb darkening."""

from functools import lru_cache
import numpy as np
from .geometry import _arc_signs


@lru_cache(maxsize=32)
def _quadrature_rule(n_step, method="gauss"):
    """Read-only nodes and weights on [0, 1]; n_step is the actual node count."""
    if isinstance(n_step, (bool, np.bool_)) or int(n_step) != n_step or n_step < 1:
        raise ValueError("n_step must be a positive integer")
    n_step = int(n_step)
    if method == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(n_step)
        nodes, weights = 0.5 * (nodes + 1), 0.5 * weights
    elif method == "midpoint":
        nodes = (np.arange(n_step) + 0.5) / n_step
        weights = np.full(n_step, 1.0 / n_step)
    else:
        raise ValueError("method must be 'gauss' or 'midpoint'")
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def ellipse_integral(x_bound, y_bound, x0, y0, a, b, u1, u2, n_step, *, method="gauss"):
    """Calculate the integral on the ellipse boundary where the stellar and ellipse intersect.
    From bound[:, :, 1] to bound[:, :, 0]"""
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )
    x_bound, y_bound = np.asarray(x_bound, dtype=float), np.asarray(y_bound, dtype=float)
    nodes, weights = _quadrature_rule(n_step, method)
    delta_flux_ellipse = np.zeros_like(x0)
    if x_bound.ndim == 2:
        x_bound = x_bound[:, None]
        y_bound = y_bound[:, None]

    # Ay is odd in x. Reflection plus endpoint reversal preserves Ay dy and
    # keeps the pi/2-subtracted integrand well conditioned for both methods.
    mirror = x0 < 0
    x_bound = np.where(mirror[:, None, None], -x_bound[..., ::-1], x_bound)
    y_bound = np.where(mirror[:, None, None], y_bound[..., ::-1], y_bound)
    x0 = np.abs(x0)

    a2, b2 = a**2, b**2
    ab = a * b
    x2, y2 = x0**2, y0**2

    for i in range(x_bound.shape[1]):
        x_intersect = x_bound[:, i]
        y_intersect = y_bound[:, i]
        cosE, sinE = (x_intersect - x0[:, None]) / a, (y_intersect - y0[:, None]) / b
        cosE = np.clip(cosE, -1, 1)
        E = np.arccos(cosE)
        E[sinE < 0] *= -1.0  # the formal definition [-pi, pi)
        E[E[:, 1] > E[:, 0], 0] += 2 * np.pi  # make sure E_low > E_up

        delta_E = E[:, 0] - E[:, 1]
        cosE0, cosE1 = np.cos(E[:, 0]), np.cos(E[:, 1])
        sinE0, sinE1 = np.sin(E[:, 0]), np.sin(E[:, 1])
        cos2E0, cos2E1 = np.cos(2 * E[:, 0]), np.cos(2 * E[:, 1])
        sin2E0, sin2E1 = np.sin(2 * E[:, 0]), np.sin(2 * E[:, 1])
        cos3E0, cos3E1 = np.cos(3 * E[:, 0]), np.cos(3 * E[:, 1])
        sin3E0, sin3E1 = np.sin(3 * E[:, 0]), np.sin(3 * E[:, 1])
        sin4E0, sin4E1 = np.sin(4 * E[:, 0]), np.sin(4 * E[:, 1])

        first_integral = (
            0.5 * ab * delta_E
            + 0.5 * (x_intersect[:, 0] * y_intersect[:, 0] - x_intersect[:, 1] * y_intersect[:, 1])
            + 0.5 * x0 * (y_intersect[:, 0] - y_intersect[:, 1])
            - 0.5 * y0 * (x_intersect[:, 0] - x_intersect[:, 1])
        )

        second_integral = (
            3.0 / 8.0 * ab * (a2 + b2 + 4 * x2 + 4 * y2) * delta_E
            - 1.5 * a * b2 * y0 * (cosE0 - cosE1)
            - 1.5 * b2 * x0 * y0 * (cos2E0 - cos2E1)
            - 0.5 * a * b2 * y0 * (cos3E0 - cos3E1)
            + 0.25 * b * x0 * (9 * a2 + 3 * b2 + 4 * x2 + 12 * y2) * (sinE0 - sinE1)
            + 0.25 * ab * (a2 + 3 * x2 + 3 * y2) * (sin2E0 - sin2E1)
            + 0.25 * b * x0 * (a2 - b2) * (sin3E0 - sin3E1)
            + 0.03125 * ab * (a2 - 3 * b2) * (sin4E0 - sin4E1)
        )

        delta_flux_analytic = (1 - u1 - 2 * u2) * first_integral + u2 / 3.0 * second_integral

        prefac = (u1 / 2.0 + u2) * b / a

        angle = E[:, 1, None] + delta_E[:, None] * nodes[None, :]

        xp = x0[:, None] + a * np.cos(angle)
        yp = y0[:, None] + b * np.sin(angle)

        r2 = xp**2 + yp**2
        invalid_points = r2 > 1
        r2[invalid_points] = 1 - 1e-10

        mu_p = np.sqrt(1 - r2)

        integrand = (mu_p * xp + (1 - yp**2) * (np.arctan2(xp, mu_p) - np.pi / 2.0)) * (
            xp - x0[:, None]
        )

        delta_flux_numeric = np.sum(integrand * weights[None, :], axis=1) * delta_E * prefac
        delta_flux_numeric += (
            np.pi
            / 2
            * (u1 / 2.0 + u2)
            * (
                y_intersect[:, 0]
                - y_intersect[:, 0] ** 3 / 3.0
                - y_intersect[:, 1]
                + y_intersect[:, 1] ** 3 / 3.0
            )
        )

        delta_flux_part = np.nan_to_num(delta_flux_analytic + delta_flux_numeric)
        delta_flux_ellipse += delta_flux_part

    return delta_flux_ellipse


def star_limb_primitive(alpha, u1, u2):
    """Antiderivative of Ay dy along the CCW stellar limb, including wraps."""
    alpha = np.asarray(alpha)
    k = np.floor((alpha + np.pi / 2) / np.pi)
    local = alpha - k * np.pi
    sn = np.sin(local)
    return (
        (1 - u1 - 2 * u2) * (alpha / 2 + np.sin(2 * alpha) / 4)
        + u2 / 3 * (3 * alpha / 4 + np.sin(2 * alpha) / 4 - np.sin(4 * alpha) / 16)
        + (u1 / 2 + u2) * np.pi / 2 * (4 * k / 3 + sn - sn**3 / 3)
    )


def full_occultation(flags, x0, y0, rp_eq, f, u1, u2, n_step=32, *, method="gauss"):
    """Blocked flux when the complete ellipse is inside the star (unnormalized)."""
    x0, y0 = np.asarray(x0, dtype=float), np.asarray(y0, dtype=float)
    b = rp_eq * (1 - f)
    xb = np.column_stack((x0, x0))
    yb = np.column_stack((y0 + b, y0 - b))
    return ellipse_integral(
        xb, yb, x0, y0, rp_eq, b, u1, u2, n_step, method=method
    ) + ellipse_integral(xb, yb[:, ::-1], x0, y0, rp_eq, b, u1, u2, n_step, method=method)


def partial_occultation(
    flags, alphas, x_intersect, y_intersect, x0, y0, rp_eq, f, u1, u2, n_step=32, *, method="gauss"
):
    """Integrate every interior arc; supports both two and four crossings."""
    x0, y0 = np.asarray(x0, dtype=float), np.asarray(y0, dtype=float)
    result = np.zeros_like(x0)
    if not len(x0):
        return result
    a, b = rp_eq, rp_eq * (1 - f)
    counts = np.sum(np.isfinite(alphas), axis=1)
    if np.any(~np.isin(counts, [2, 4])):
        raise ValueError("Partial ellipse occultation requires two or four distinct intersections")
    for count in (2, 4):
        mask = counts == count
        if not np.any(mask):
            continue
        xc, yc = x0[mask], y0[mask]
        alpha = np.sort(alphas[mask, :count], axis=1)
        alpha_end = np.concatenate((alpha[:, 1:], alpha[:, :1] + 2 * np.pi), axis=1)
        inside = (
            _arc_signs(
                alpha,
                alpha_end,
                lambda t: ((np.cos(t) - xc[:, None]) / a) ** 2
                + ((np.sin(t) - yc[:, None]) / b) ** 2
                - 1,
            )
            < 0
        )
        value = np.sum(
            np.where(
                inside,
                star_limb_primitive(alpha_end, u1, u2) - star_limb_primitive(alpha, u1, u2),
                0,
            ),
            axis=1,
        )
        e = np.sort(
            np.arctan2(
                (y_intersect[mask, :count] - yc[:, None]) / b,
                (x_intersect[mask, :count] - xc[:, None]) / a,
            ),
            axis=1,
        )
        e_end = np.concatenate((e[:, 1:], e[:, :1] + 2 * np.pi), axis=1)
        inside = (
            _arc_signs(
                e,
                e_end,
                lambda t: (xc[:, None] + a * np.cos(t)) ** 2
                + (yc[:, None] + b * np.sin(t)) ** 2
                - 1,
            )
            < 0
        )
        for j in range(count):
            use = inside[:, j]
            if not np.any(use):
                continue
            xb = xc[use, None] + a * np.cos(np.column_stack((e_end[use, j], e[use, j])))
            yb = yc[use, None] + b * np.sin(np.column_stack((e_end[use, j], e[use, j])))
            value[use] += ellipse_integral(
                xb, yb, xc[use], yc[use], a, b, u1, u2, n_step, method=method
            )
        result[mask] = value
    return result
