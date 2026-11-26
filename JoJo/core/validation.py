"""Input checks shared by the public NumPy light-curve models."""

import numpy as np


def parameters(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected {size} finite transit parameters")
    return values


def positive_radius(value, name):
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1 (stellar radius units)")


def flattening(value):
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("flattening must satisfy 0 <= f < 1")


def ring_alpha(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("alpha must lie in [0, 1]")


def orbit(period, ecc, u1, u2):
    if period <= 0 or not 0 <= ecc < 1:
        raise ValueError("Require period > 0 and 0 <= ecc < 1")
    if 1 - u1 / 3 - u2 / 6 <= 0:
        raise ValueError("The total stellar flux must be positive")


def positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or int(value) != value
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
