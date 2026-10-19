import numpy as np
import pytest

from JoJo.core.integration import _quadrature_rule, star_limb_primitive
from JoJo.models.oblate import ellipse_occultation
from .reference import area_flux


@pytest.mark.parametrize("method", ["gauss", "midpoint"])
def test_rules_are_normalized_cached_and_immutable(method):
    nodes, weights = _quadrature_rule(32, method)
    assert nodes is _quadrature_rule(32, method)[0]
    assert np.all((nodes > 0) & (nodes < 1))
    assert weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        nodes[0] = 0.5


def test_gauss_polynomial_exactness():
    x, w = _quadrature_rule(8)
    for degree in range(16):
        assert np.sum(w * x**degree) == pytest.approx(1 / (degree + 1), abs=1e-14)


@pytest.mark.parametrize("alpha", [-7.0, -np.pi, -0.1, 0.0, 2.0, 8.0])
def test_complete_star_limb_has_correct_flux(alpha):
    u1, u2 = 0.3, 0.2
    got = star_limb_primitive(alpha + 2 * np.pi, u1, u2) - star_limb_primitive(alpha, u1, u2)
    assert got == pytest.approx(np.pi * (1 - u1 / 3 - u2 / 6), abs=2e-14)


@pytest.mark.parametrize(
    "center",
    [
        (0, 0),
        (0.8, 0.3),
        (-0.8, 0.3),
        (0.95, 0.2),
        (-0.95, 0.2),
        (0.01, 0.96),
        (1.1199, 0),
        (0.8799, 0),
    ],
)
def test_ellipse_against_independent_area_integration(center):
    x, y = center
    got, _ = ellipse_occultation(x, y, 0.12, 0.07, 0.3, 0.2, n_step=64)
    ref = area_flux(x, y, 0.12, 0.12, 0.07)
    np.testing.assert_allclose(got, ref, atol=2e-8, rtol=0)


def test_four_crossing_flux_against_area_integration():
    got, _ = ellipse_occultation(0, 0.89, 0.8, 0.1, 0.3, 0.2, n_step=64)
    assert got[0] == pytest.approx(area_flux(0, 0.89, 0.8, 0.8, 0.1), abs=2e-8)


@pytest.mark.parametrize("u1,u2", [(0, 0), (0.3, 0.2), (0.6226, 0.0498)])
def test_tangent_flux_with_exact_crossings(u1, u2):
    # Bypass root solving: even the exact two crossings previously selected
    # an entire exterior ellipse arc because its midpoint was the tangent.
    from JoJo.core.integration import partial_occultation

    y = np.full((1, 2), 29 / 35)
    x = np.array([[np.sqrt(1 - (29 / 35) ** 2), -np.sqrt(1 - (29 / 35) ** 2)]])
    got = partial_occultation(
        np.array([0]),
        np.arctan2(y, x),
        x,
        y,
        np.array([0.0]),
        np.array([0.9]),
        0.8,
        0.875,
        u1,
        u2,
        64,
    )[0]
    reference = area_flux(0, 0.9, 0.8, 0.8, 0.1, u1, u2)
    assert got == pytest.approx(reference, abs=2e-8)


@pytest.mark.parametrize("swap", [False, True])
def test_flux_continuity_across_tangency_and_batch_sizes(swap):
    positions = 0.9 + np.array([-1e-6, -1e-10, -1e-12, 0, 1e-12, 1e-10, 1e-6])
    a, b = (0.1, 0.8) if swap else (0.8, 0.1)
    x = positions if swap else np.zeros_like(positions)
    y = np.zeros_like(positions) if swap else positions
    reference = np.array([area_flux(xi, yi, a, a, b) for xi, yi in zip(x, y)])
    scalar = np.array([ellipse_occultation(xi, yi, a, b, 0.3, 0.2)[0][0] for xi, yi in zip(x, y)])
    np.testing.assert_allclose(scalar, reference, atol=2e-8, rtol=0)
    for repeat in [1, 16, 128]:
        batch, _ = ellipse_occultation(np.tile(x, repeat), np.tile(y, repeat), a, b, 0.3, 0.2)
        # Near-double roots are ill-conditioned; allow sub-0.00001-ppm
        # differences from root polishing, far below the area-reference tolerance.
        np.testing.assert_allclose(batch, np.tile(scalar, repeat), atol=2e-11, rtol=0)


def test_public_oblate_lightcurve_at_tangency():
    from JoJo import oblate_lc

    parameters = [0, 0.9, 10, np.sqrt(0.8 * 0.1), 0.875, 0, 0, np.pi / 2, 0.3, 0.2, 0]
    reference = 1 - area_flux(0, 0.9, 0.8, 0.8, 0.1) / (np.pi * (1 - 0.3 / 3 - 0.2 / 6))
    for size in [1, 16, 128]:
        flux, _ = oblate_lc(parameters, np.zeros(size), photo_ecc=True, exp_time=0)
        np.testing.assert_allclose(flux, reference, atol=1e-8, rtol=0)


def test_uniform_full_ellipse_area():
    got, _ = ellipse_occultation([-0.5, 0, 0.5], [0.1, 0.2, 0.3], 0.12, 0.07, 0, 0)
    np.testing.assert_allclose(got, np.pi * 0.12 * 0.07, atol=2e-14)


def test_both_quadratures_converge_to_same_area():
    from JoJo.core.geometry import solve_star_ellipse_intersections
    from JoJo.core.integration import partial_occultation

    x, y = np.array([0.01]), np.array([0.96])
    flags, xi, yi, al = solve_star_ellipse_intersections(x, y, 0.12, 1 - 0.07 / 0.12)
    ref = area_flux(x[0], y[0], 0.12, 0.12, 0.07)
    errors = {}
    for method, nodes in [("midpoint", 64), ("midpoint", 512), ("gauss", 32)]:
        value = partial_occultation(
            flags, al, xi, yi, x, y, 0.12, 1 - 0.07 / 0.12, 0.3, 0.2, nodes, method=method
        )[0]
        errors[method, nodes] = abs(value - ref)
    assert errors["midpoint", 512] < errors["midpoint", 64]
    assert errors["gauss", 32] < 1e-8
    assert errors["gauss", 32] < errors["midpoint", 512]
