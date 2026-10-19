import numpy as np
import pytest

from JoJo.core.geometry import solve_star_ellipse_intersections, solve_planet_ring_intersections
from JoJo.models.egg import solve_star_egg_intersections


@pytest.mark.parametrize("f", [0, 1e-8, 1e-6, 1e-4, 0.1, 0.4, -0.4])
def test_intersections_satisfy_both_curves(f):
    rng = np.random.default_rng(1234)
    angle = rng.uniform(-np.pi, np.pi, 100)
    distance = rng.uniform(0.95, 1.05, 100)
    x, y = distance * np.cos(angle), distance * np.sin(angle)
    a, b = 0.12, 0.12 * (1 - f)
    flags, xi, yi, alpha = solve_star_ellipse_intersections(x, y, a, f)
    assert np.all(flags == 0)
    good = np.isfinite(alpha)
    np.testing.assert_allclose((xi * xi + yi * yi)[good], 1, atol=2e-10)
    residual = ((xi - x[:, None]) / a) ** 2 + ((yi - y[:, None]) / b) ** 2
    np.testing.assert_allclose(residual[good], 1, atol=2e-7)
    assert np.all(np.diff(alpha[:, :2], axis=1) > 0)
    assert np.all(good.sum(axis=1) == 2)


def test_circle_contacts_and_empty_centers():
    flags, x, y, alpha = solve_star_ellipse_intersections([0, 0.9, 1.0, 1.1], 0, 0.1, 0)
    np.testing.assert_array_equal(flags, [1, 1, 0, -1])
    assert np.isfinite(x[2]).all()
    flags, x, y, alpha = solve_star_ellipse_intersections([], [], 0.1, 0.2)
    assert x.shape == (0, 4)


def test_four_intersections_are_retained():
    flags, x, y, alpha = solve_star_ellipse_intersections([0], [0.89], 0.8, 0.875)
    assert flags[0] == 0
    assert np.isfinite(x).sum() == 4


def test_tangent_root_is_not_an_extra_crossing():
    flags, x, y, alpha = solve_star_ellipse_intersections([0.0] * 128, [0.9] * 128, 0.8, 0.875)
    assert np.all(flags == 0)
    np.testing.assert_array_equal(np.isfinite(x).sum(axis=1), 2)


@pytest.mark.parametrize("size", [1, 2, 16, 128, 129])
@pytest.mark.parametrize("epsilon", [1e-8, 1e-10])
def test_tangent_topology_is_batch_independent(size, epsilon):
    # The two extra crossings approach a double root from below; they must
    # survive until the geometry is actually tangent (not merely close).
    offsets = np.array([-1e-6, -1e-10, -1e-12, 0, 1e-12, 1e-10, 1e-6])
    y = np.tile(0.9 + offsets, size)
    flags, x, _, _ = solve_star_ellipse_intersections(0, y, 0.8, 0.875, epsilon)
    assert np.all(flags == 0)
    np.testing.assert_array_equal(np.isfinite(x).sum(axis=1), np.tile([4, 4, 4, 2, 2, 2, 2], size))


@pytest.mark.parametrize(
    "center,a,b",
    [((0, 0.9), 0.8, 0.1), ((0, -0.9), 0.8, 0.1), ((0.9, 0), 0.1, 0.8), ((-0.9, 0), 0.1, 0.8)],
)
def test_tangent_crossings_under_reflection_and_axis_swap(center, a, b):
    _, x, y, _ = solve_star_ellipse_intersections(
        np.full(128, center[0]), np.full(128, center[1]), a, 1 - b / a
    )
    np.testing.assert_array_equal(np.isfinite(x).sum(axis=1), 2)
    along = y if center[0] == 0 else x
    across = x if center[0] == 0 else y
    np.testing.assert_allclose(np.abs(along[:, :2]), 29 / 35, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        np.abs(across[:, :2]), np.sqrt(1 - (29 / 35) ** 2), atol=1e-12, rtol=0
    )


def test_tall_egg_is_not_misclassified_as_outside():
    flags, x, y, alpha = solve_star_egg_intersections(0, 1.10, 0.08, 0.1, 0.15)
    assert flags[0] == 0
    assert np.isfinite(x).sum() == 2


@pytest.mark.parametrize("a,f", [(0, 0), (-0.1, 0.2), (0.1, 1), (0.9, -0.5)])
def test_invalid_axes_fail_explicitly(a, f):
    with pytest.raises(ValueError):
        solve_star_ellipse_intersections([0], [0], a, f)


@pytest.mark.parametrize("f", [0, 0.2, 0.6])
def test_concentric_ring_intersections(f):
    points = solve_planet_ring_intersections(0, 0.1, 0.1, 0.12, 0.2, f)
    for (x, y), a in zip((points[:2], points[2:]), (0.12, 0.2)):
        valid = np.isfinite(x)
        np.testing.assert_allclose((x * x + (y - 0.1) ** 2)[valid], 0.1**2, atol=1e-14)
        np.testing.assert_allclose(
            (x * x / a**2 + (y - 0.1) ** 2 / (a * (1 - f)) ** 2)[valid], 1, atol=1e-14
        )
    if f == 0:
        assert all(np.isnan(v).all() for v in points)
