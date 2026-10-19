"""Orbital anomaly conversions and transit phase helpers."""

import numpy as np
from .validation import positive_integer


def exposure_grid(time, exp_time=None, supersample_factor=5):
    """Sorted times and midpoint exposure samples, with explicit instantaneous mode.

    None retains cadence inference for compatibility; one time implies zero
    exposure. Explicit positive exposures are integrated regardless of duration.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(np.diff(time) < 0):
        raise ValueError("time_array must be a finite, sorted one-dimensional array")
    positive_integer(supersample_factor, "supersample_factor")
    if exp_time is None:
        exp_time = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    if not np.isscalar(exp_time) or not np.isfinite(exp_time) or exp_time < 0:
        raise ValueError("exp_time must be a nonnegative finite scalar")
    count = int(supersample_factor) if exp_time > 0 else 1
    offsets = ((np.arange(count) + 0.5) / count - 0.5) * exp_time
    return time, (time[:, None] + offsets).reshape(-1), count


def sky_position(tc, b, period, ecc, omega, log10_rho_star, time, photo_ecc=False):
    """Projected center and foreground mask. Angles in radians, time in days.

    tc uses the historical true anomaly pi/2-omega convention. photo_ecc=True
    uses a constant projected velocity near transit; False uses Kepler motion.
    """
    density = 10.0**log10_rho_star
    if photo_ecc:
        density *= ((1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc * ecc)) ** 3
    a = 3.753 * (period**2 * density) ** (1 / 3)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("Invalid density or period: a/R_star is not finite and positive")
    cos_i = b / a * (1 + ecc * np.sin(omega)) / (1 - ecc * ecc)
    if not photo_ecc and abs(cos_i) > 1:
        raise ValueError("Impact parameter implies an impossible orbital inclination")
    if photo_ecc:
        phase = (time - tc + period / 2) % period - period / 2
        return (
            phase / period * 2 * np.pi * a,
            np.full_like(time, b),
            np.ones(time.shape, dtype=bool),
        )
    tp = tc - period / (2 * np.pi) * true_to_mean(np.pi / 2 - omega, ecc)
    ta = mean_to_true(2 * np.pi / period * (time - tp), ecc)
    r = a * (1 - ecc * ecc) / (1 + ecc * np.cos(ta))
    phase = omega + ta
    return -r * np.cos(phase), r * np.sin(phase) * cos_i, np.sin(phase) > 0


def contact_times(flags, time):
    """Sampled contacts (not root-refined; grazing contacts may be absent)."""
    contacts = []
    for before, after in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        indices = np.flatnonzero((flags[:-1] == before) & (flags[1:] == after))
        if indices.size:
            contacts.append(time[indices[0]])
    return contacts


def mean_to_true(m, e, tol=1e-7, max_iter=100):
    m = np.array(m)
    if e == 0:
        return m

    E = m  # + e * np.sin(m) + 0.5 * e**2 * np.sin(2 * m)
    for _ in range(max_iter):
        f_val = E - e * np.sin(E) - m
        df = 1 - e * np.cos(E)
        delta = f_val / df
        E_new = E - delta
        E = E_new
        if np.all(np.abs(delta) < tol):
            break
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return f


def true_to_mean(f, e):
    # m = f - 2*e*np.sin(f) + 3/4*e**2*np.sin(2*f) - 1/3*e**3*np.sin(3*f)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2))
    m = E - e * np.sin(E)
    return m
