"""Independent area integration: vertical slices, no JoJo geometry/Green code."""

import numpy as np
from scipy.integrate import quad


def area_flux(x0, y0, a_right, a_left, b, u1=0.3, u2=0.2, *, ring=None, planet_radius=None):
    """Return unnormalized blocked flux for an egg/ellipse or planet + ring.

    ring=(inner_a, outer_a, axis_ratio, opacity) specifies concentric ellipses.
    The y integral is analytic; adaptive scipy quad integrates vertical slices
    in x, independently of the production boundary integration.
    """

    def primitive(y, x):
        s = max(0.0, 1 - x * x)
        if s == 0:
            return 0.0
        y = np.clip(y, -np.sqrt(s), np.sqrt(s))
        return (
            (1 - u1 - 2 * u2 + u2 * x * x) * y
            + u2 * y**3 / 3
            + (u1 + 2 * u2)
            / 2
            * (y * np.sqrt(max(0, s - y * y)) + s * np.arcsin(np.clip(y / np.sqrt(s), -1, 1)))
        )

    def interval(x, a, semi_b):
        v = 1 - ((x - x0) / a) ** 2
        if v <= 0:
            return None
        h = semi_b * np.sqrt(v)
        sh = np.sqrt(max(0, 1 - x * x))
        lo, hi = max(-sh, y0 - h), min(sh, y0 + h)
        return (lo, hi) if hi > lo else None

    def integral(bounds, x):
        return 0.0 if bounds is None else primitive(bounds[1], x) - primitive(bounds[0], x)

    def intersect(a, b):
        if a is None or b is None:
            return None
        lo, hi = max(a[0], b[0]), min(a[1], b[1])
        return (lo, hi) if hi > lo else None

    if ring is None:
        widths = [a_left, a_right]

        def slice_flux(x):
            return integral(interval(x, a_right if x >= x0 else a_left, b), x)

    else:
        rin, rout, ratio, opacity = ring
        rp = planet_radius
        widths = [rin, rout, rp]

        def slice_flux(x):
            p = interval(x, rp, rp)
            ei, eo = interval(x, rin, rin * ratio), interval(x, rout, rout * ratio)
            return integral(p, x) + opacity * (
                integral(eo, x)
                - integral(ei, x)
                - integral(intersect(eo, p), x)
                + integral(intersect(ei, p), x)
            )

    lo, hi = max(-1, x0 - max(widths)), min(1, x0 + max(widths))
    if hi <= lo:
        return 0.0
    edges = sorted(set([lo, hi] + [v for w in widths for v in (x0 - w, x0, x0 + w) if lo < v < hi]))
    return sum(
        quad(slice_flux, l, r, epsabs=2e-10, epsrel=2e-8, limit=250)[0]
        for l, r in zip(edges[:-1], edges[1:])
    )
