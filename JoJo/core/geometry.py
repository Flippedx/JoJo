"""Intersections in the ellipse-aligned frame (stellar radius = 1)."""

import numpy as np
from ._solve_quartic import multi_quartic


def _arc_signs(start, end, residual):
    """Classify open arcs without mistaking an isolated tangent for their interior.

    Boundaries must already be split at crossings. Sample three interior points
    and use the residual farthest from zero. Zero denotes roundoff-level geometry,
    not a reliable inside/outside decision; the tolerance is independent of the
    root solver's (much larger) convergence tolerance.
    """
    values = np.stack(
        [residual(start + fraction * (end - start)) for fraction in (0.25, 0.5, 0.75)],
        axis=-1,
    )
    strongest = np.take_along_axis(values, np.abs(values).argmax(axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    tol = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(values).max(axis=-1))
    return np.where(np.abs(strongest) > tol, np.sign(strongest), 0)


def _crossing_angles(e, x, y, a, b):
    """Remove non-crossing contacts, including numerically split double roots."""
    e = np.sort(np.mod(e, 2 * np.pi))
    if len(e) < 2:
        return np.empty(0)
    end = np.r_[e[1:], e[0] + 2 * np.pi]
    signs = _arc_signs(e, end, lambda t: (x + a * np.cos(t)) ** 2 + (y + b * np.sin(t)) ** 2 - 1)
    # A microscopic gap between split tangent roots may be indistinguishable
    # from the boundary. Collapse it only when both neighboring resolved arcs
    # agree. Do not merge a close pair just because its angular spacing is small.
    resolved = np.flatnonzero(signs)
    if not len(resolved):
        raise ArithmeticError("Ellipse boundary is unresolved at floating-point precision")
    for i in np.flatnonzero(signs == 0):
        before = resolved[np.searchsorted(resolved, i) - 1]
        after = resolved[np.searchsorted(resolved, i) % len(resolved)]
        if signs[before] != signs[after]:
            raise ArithmeticError("Unresolved crossing between near-coincident roots")
        signs[i] = signs[before]
    e = e[signs != np.roll(signs, 1)]
    return np.sort(np.arctan2(y + b * np.sin(e), x + a * np.cos(e)))


def _crossing_angles_batch(z, good, x, y, a, b):
    """Classify candidate ellipse roots in a batch, including double contacts."""
    count = good.sum(axis=1)
    valid_count = np.isin(count, (0, 2, 4))
    e = np.mod(
        np.arctan2((z.imag - y[:, None]) / b, (z.real - x[:, None]) / a), 2 * np.pi
    )
    e = np.sort(np.where(good, e, np.nan), axis=1)
    slots = np.arange(4)[None, :]
    valid = slots < count[:, None]
    next_slot = (slots + 1) % np.maximum(count[:, None], 1)
    end = np.take_along_axis(e, next_slot, axis=1) + 2 * np.pi * (next_slot == 0)
    fractions = np.array([0.25, 0.5, 0.75])
    samples = e[..., None] + (end - e)[..., None] * fractions
    values = (x[:, None, None] + a * np.cos(samples)) ** 2
    values += (y[:, None, None] + b * np.sin(samples)) ** 2 - 1
    strongest = np.take_along_axis(values, np.abs(values).argmax(axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    tol = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(values).max(axis=-1))
    signs = np.where(valid & (np.abs(strongest) > tol), np.sign(strongest), 0)

    # For a numerically split tangent, the microscopic arc has no reliable
    # sign. Its two nearest resolved neighbors must agree before removing it.
    before = np.zeros_like(signs)
    after = np.zeros_like(signs)
    modulus = np.maximum(count[:, None], 1)
    for distance in range(1, 5):
        previous = np.take_along_axis(signs, (slots - distance) % modulus, axis=1)
        following = np.take_along_axis(signs, (slots + distance) % modulus, axis=1)
        before = np.where((before == 0) & valid, previous, before)
        after = np.where((after == 0) & valid, following, after)
    unresolved = ~valid_count | np.any(
        valid & (signs == 0) & ((before == 0) | (before != after)), axis=1
    )
    signs = np.where(valid & (signs == 0), before, signs)
    previous_sign = np.take_along_axis(signs, (slots - 1) % modulus, axis=1)
    crossing = valid & (signs != previous_sign)
    crossing_count = crossing.sum(axis=1)
    unresolved |= ~np.isin(crossing_count, (0, 2, 4))
    alpha = np.arctan2(y[:, None] + b * np.sin(e), x[:, None] + a * np.cos(e))
    alpha = np.sort(np.where(crossing, alpha, np.nan), axis=1)
    return alpha, crossing_count, unresolved


def _parameter_intersections(x, y, a, b, epsilon):
    """Fallback quartic in tan(E/2), well conditioned in the near-circle limit."""
    coeff = [
        (x - a) ** 2 + y * y - 1,
        4 * b * y,
        2 * (x * x + y * y - 1) - 2 * a * a + 4 * b * b,
        4 * b * y,
        (x + a) ** 2 + y * y - 1,
    ]
    roots = np.roots(coeff)
    real = roots.real[np.abs(roots.imag) < 1e-7 * (1 + np.abs(roots.real))]
    e = 2 * np.arctan(real)
    # tan(E/2) does not represent E=pi.
    if abs(coeff[0]) < min(epsilon, 1e-12):
        e = np.append(e, np.pi)
    xp, yp = x + a * np.cos(e), y + b * np.sin(e)
    good = np.abs(xp * xp + yp * yp - 1) < max(epsilon, 1e-12)
    return _crossing_angles(e[good], x, y, a, b)


def _parameter_roots_batch(x, y, a, b, epsilon):
    """Solve the near-circle parameter quartics together; flag degenerate rows."""
    coeff = np.column_stack(
        (
            (x - a) ** 2 + y * y - 1,
            4 * b * y,
            2 * (x * x + y * y - 1) - 2 * a * a + 4 * b * b,
            4 * b * y,
            (x + a) ** 2 + y * y - 1,
        )
    )
    leading = np.abs(coeff[:, 0]) > 1e-10 * np.maximum(1, np.max(np.abs(coeff), axis=1))
    z = np.full((len(x), 4), np.nan + 1j * np.nan)
    good = np.zeros((len(x), 4), dtype=bool)
    if np.any(leading):
        monic = coeff[leading, 1:] / coeff[leading, :1]
        companion = np.zeros((len(monic), 4, 4))
        companion[:, 0, :] = -monic
        companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1
        try:
            roots = np.linalg.eigvals(companion)
        except np.linalg.LinAlgError:
            return z, good, np.ones(len(x), dtype=bool)
        angle = 2 * np.arctan(roots.real)
        xp = x[leading, None] + a * np.cos(angle)
        yp = y[leading, None] + b * np.sin(angle)
        z[leading] = xp + 1j * yp
        good[leading] = (
            (np.abs(roots.imag) < 1e-7 * (1 + np.abs(roots.real)))
            & (np.abs(xp * xp + yp * yp - 1) < max(epsilon, 1e-12))
        )
    return z, good, ~leading


def solve_star_ellipse_intersections(x0, y0, rp_eq, f, epsilon=1e-8):
    """Return flags, x, y, alpha, with crossings sorted by stellar angle.

    Flags: -1 outside, 0 crossing, 1 fully inside. Input centers broadcast as
    1-D arrays. Semi-axes must lie in (0, 1); a vertical major axis is allowed.
    Circular results have two columns, elliptical results four (NaN padded).
    Tangencies with no crossing use the inside/outside flag.
    """
    x0, y0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x0, dtype=float)), np.atleast_1d(np.asarray(y0, dtype=float))
    )
    if x0.ndim != 1 or not np.all(np.isfinite(x0)) or not np.all(np.isfinite(y0)):
        raise ValueError("Centers must be finite scalars or one-dimensional arrays")
    a, b = float(rp_eq), float(rp_eq * (1 - f))
    if not (0 < a < 1 and 0 < b < 1) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require semi-axes in (0, 1) and epsilon > 0")
    d = np.hypot(x0, y0)
    flags = np.zeros(len(x0), dtype=int)
    radius = max(a, b)
    flags[d >= 1 + radius] = -1
    flags[d <= 1 - radius] = 1
    cols = 2 if f == 0 else 4
    alpha = np.full((len(x0), cols), np.nan)
    candidate = np.flatnonzero(flags == 0)
    if f == 0:
        dc = d[candidate]
        da = np.arccos(np.clip((1 + dc * dc - a * a) / (2 * dc), -1, 1))
        center = np.arctan2(y0[candidate], x0[candidate])
        alpha[candidate] = np.sort(
            np.angle(np.exp(1j * np.column_stack((center - da, center + da)))), axis=1
        )
        return flags, np.cos(alpha), np.sin(alpha), alpha
    if not len(candidate):
        return flags, np.cos(alpha), np.sin(alpha), alpha
    xi, yi = x0[candidate], y0[candidate]
    if abs(f) < 1e-3:
        z, good, needs_fallback = _parameter_roots_batch(xi, yi, a, b, epsilon)
    else:
        b2, r2 = (b / a) ** 2, a * a
        c0 = np.full_like(xi, 1 - b2, dtype=complex)
        c1 = 4 * b2 * xi - 4j * yi
        c2 = -2 * (1 + b2 - 2 * b2 * r2 + 2 * b2 * xi * xi + 2 * yi * yi)
        c3, c4 = np.conjugate(c1), c0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = multi_quartic(c0, c1, c2, c3, c4)
            for _ in range(10):
                value = (
                    c0[:, None] * z**4
                    + c1[:, None] * z**3
                    + c2[:, None] * z * z
                    + c3[:, None] * z
                    + c4[:, None]
                )
                deriv = (
                    4 * c0[:, None] * z**3
                    + 3 * c1[:, None] * z * z
                    + 2 * c2[:, None] * z
                    + c3[:, None]
                )
                dz = np.divide(value, deriv, out=np.zeros_like(value), where=np.abs(deriv) > 1e-15)
                z -= dz
                if np.all(np.abs(dz) < epsilon):
                    break
            residual = (z.real - xi[:, None]) ** 2 + (z.imag - yi[:, None]) ** 2 / b2 - r2
        good = (np.abs(residual) < epsilon) & (np.abs(np.abs(z) ** 2 - 1) < epsilon)
        needs_fallback = np.zeros(len(xi), dtype=bool)

    # Nearly parallel normals signal a possible multiple root. The batched
    # result is sufficient for ordinary crossings; only ambiguous rows need
    # the slower topology check or alternative polynomial.
    nx = (z.real - xi[:, None]) / (a * a)
    ny = (z.imag - yi[:, None]) / (b * b)
    near_tangent = np.any(
        good & (np.abs(z.real * ny - z.imag * nx) <= 1e-4 * np.hypot(nx, ny)), axis=1
    )
    counts = np.sum(good, axis=1)
    angles = np.sort(np.where(good, np.angle(z), np.nan), axis=1)
    adjacent = np.any(np.diff(angles, axis=1) < 1e-7, axis=1)
    last = angles[np.arange(len(xi)), np.maximum(counts - 1, 0)]
    wrapped = 2 * np.pi - last + angles[:, 0] < 1e-7
    duplicates = (counts > 1) & (adjacent | wrapped)
    invalid = (
        needs_fallback
        | ~np.isin(counts, (0, 2, 4))
        | duplicates
        | ~np.all(np.isfinite(z), axis=1)
    )
    exceptional = invalid | near_tangent

    regular = ~exceptional
    alpha[candidate[regular]] = angles[regular]
    empty = regular & (counts == 0)
    flags[candidate[empty]] = np.where(d[candidate[empty]] < 1, 1, -1)
    if np.any(near_tangent):
        selected = np.flatnonzero(near_tangent)
        refined_z, refined_good, failed = _parameter_roots_batch(
            xi[selected], yi[selected], a, b, epsilon
        )
        refined_angles, refined_counts, unresolved = _crossing_angles_batch(
            refined_z, refined_good, xi[selected], yi[selected], a, b
        )
        accepted = ~(failed | unresolved) & np.any(refined_good, axis=1)
        accepted_rows = selected[accepted]
        alpha[candidate[accepted_rows]] = refined_angles[accepted]
        no_crossing = accepted_rows[refined_counts[accepted] == 0]
        flags[candidate[no_crossing]] = np.where(d[candidate[no_crossing]] < 1, 1, -1)
        exceptional[accepted_rows] = False
    for j in np.flatnonzero(exceptional):
        row = angles[j, : counts[j]]
        if invalid[j]:
            row = _parameter_intersections(xi[j], yi[j], a, b, epsilon)
        else:
            e = np.arctan2((np.sin(row) - yi[j]) / b, (np.cos(row) - xi[j]) / a)
            try:
                row = _crossing_angles(e, xi[j], yi[j], a, b)
            except ArithmeticError:
                row = _parameter_intersections(xi[j], yi[j], a, b, epsilon)
        if len(row) < 2:
            flags[candidate[j]] = 1 if d[candidate[j]] < 1 else -1
        elif len(row) in (2, 4):
            alpha[candidate[j], : len(row)] = row
        else:
            raise ArithmeticError("Unresolved ellipse intersection topology")
    return flags, np.cos(alpha), np.sin(alpha), alpha


def solve_planet_ring_intersections(x_0, y_0, r_p, r_rin, r_rout, f_r):
    """Analytic intersections of concentric circles/ellipses; NaN for no crossing.

    Each boundary returns four symmetry-related points in the historical order.
    Tangencies can repeat points; arc selection ignores zero-length intervals.
    Coincident circles have no isolated intersections and return NaNs.
    """
    from .validation import flattening, positive_radius

    flattening(f_r)
    for value, name in ((r_p, "r_p"), (r_rin, "r_in"), (r_rout, "r_out")):
        positive_radius(value, name)
    if not r_p <= r_rin <= r_rout:
        raise ValueError("Require r_p <= r_in <= r_out")
    x_0, y_0 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x_0, dtype=float)), np.atleast_1d(np.asarray(y_0, dtype=float))
    )
    if x_0.ndim != 1 or not np.all(np.isfinite(x_0)) or not np.all(np.isfinite(y_0)):
        raise ValueError("Centers must be finite scalars or one-dimensional arrays")

    def boundary(radius):
        if f_r == 0 or r_p < radius * (1 - f_r) or r_p > radius:
            return np.full((len(x_0), 4), np.nan), np.full((len(x_0), 4), np.nan)
        denom = 2 * f_r - f_r * f_r
        dx = np.sqrt(max(0.0, (r_p * r_p - radius * radius * (1 - f_r) ** 2) / denom))
        dy = np.sqrt(max(0.0, (1 - f_r) ** 2 * (radius * radius - r_p * r_p) / denom))
        return (
            x_0[:, None] + np.array([-dx, -dx, dx, dx]),
            y_0[:, None] + np.array([-dy, dy, -dy, dy]),
        )

    return (*boundary(r_rin), *boundary(r_rout))
