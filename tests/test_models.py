import numpy as np
import pytest

from JoJo import oblate_lc, spherical_lc, egg_lc, ring_lc
from JoJo.models.egg import egg_occultation
from JoJo.models.ring import ring_occultation
from .reference import area_flux

OBLATE = [0, 0.3, 10, np.sqrt(0.12 * 0.07), 1 - 0.07 / 0.12, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]
EGG = [0, 0.3, 10, 0.12, 0.08, 0.07, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]
RING = [0, 0.3, 10, 0.1, 0.12, 0.2, 0.4, 0.4, 0, np.pi / 2, 0.3, 0.2, 0, 0.5]
TIME = np.linspace(-0.15, 0.15, 301)


@pytest.mark.parametrize("angle", [0, 0.4, np.pi / 2, 2.4])
def test_symmetric_egg_matches_oblate(angle):
    pe, po = EGG.copy(), OBLATE.copy()
    pe[4], pe[6], po[5] = 0.12, angle, angle
    fe = egg_lc(pe, TIME, exp_time=0)[0]
    fo = oblate_lc(po, TIME, exp_time=0)[0]
    np.testing.assert_allclose(fe, fo, atol=2e-8, rtol=0)


@pytest.mark.parametrize(
    "center",
    [(-1.08, 0), (-0.95, 0.2), (0, 0), (0.8, 0.3), (0.95, 0.2), (0.05, 0.95), (-0.05, -0.95)],
)
def test_egg_against_independent_area(center):
    x, y = center
    got, _ = egg_occultation(x, y, 0.12, 0.08, 0.07, 0.3, 0.2, n_step=64)
    assert got[0] == pytest.approx(area_flux(x, y, 0.12, 0.08, 0.07), abs=2e-8)


def test_egg_area_and_reflection():
    got, _ = egg_occultation(0, 0, 0.12, 0.08, 0.07, 0, 0)
    assert got[0] == pytest.approx(np.pi * 0.07 * (0.12 + 0.08) / 2, abs=1e-14)
    x = np.linspace(-1.2, 1.2, 121)
    left = egg_occultation(x, 0.3, 0.12, 0.08, 0.07, 0.3, 0.2)[0]
    right = egg_occultation(-x, -0.3, 0.08, 0.12, 0.07, 0.3, 0.2)[0]
    np.testing.assert_allclose(left, right, atol=2e-8, rtol=0)


@pytest.mark.parametrize("f", [0, 0.2, 0.6])
@pytest.mark.parametrize("center", [(0, 0), (0.93, 0.2), (0.03, 1.02)])
def test_ring_against_independent_area(f, center):
    x, y = center
    got = ring_occultation(x, y, 0.1, 0.12, 0.2, f, 0.3, 0.2, 0.5, 64)
    ref = area_flux(x, y, 0.2, 0.2, 0.2, ring=(0.12, 0.2, 1 - f, 0.5), planet_radius=0.1)
    assert got[0] == pytest.approx(ref, abs=3e-8)


@pytest.mark.parametrize("center", [(0, 0), (0.95, 0.1)])
def test_ring_inner_radius_equals_planet_radius(center):
    x, y = center
    got = ring_occultation(x, y, 0.1, 0.1, 0.2, 0.4, 0.3, 0.2, 0.5, 64)
    ref = area_flux(x, y, 0.2, 0.2, 0.2, ring=(0.1, 0.2, 0.6, 0.5), planet_radius=0.1)
    assert got[0] == pytest.approx(ref, abs=3e-8)


def test_ring_transparent_zero_width_and_opacity_linearity():
    ps = [0, 0.3, 10, 0.1, 0, 0, 0, np.pi / 2, 0.3, 0.2, 0]
    sphere = spherical_lc(ps, TIME, exp_time=0)[0]
    p = RING.copy()
    p[-1] = 0
    np.testing.assert_allclose(ring_lc(p, TIME, exp_time=0), sphere, atol=3e-8, rtol=0)
    p[-1] = 1
    opaque = ring_lc(p, TIME, exp_time=0)
    p[-1] = 0.5
    np.testing.assert_allclose(
        ring_lc(p, TIME, exp_time=0), (sphere + opaque) / 2, atol=3e-8, rtol=0
    )
    p[4] = p[5]
    np.testing.assert_allclose(ring_lc(p, TIME, exp_time=0), sphere, atol=3e-8, rtol=0)


@pytest.mark.parametrize("model,p", [(oblate_lc, OBLATE), (egg_lc, EGG), (ring_lc, RING)])
def test_bounds_exposures_and_no_secondary_transit(model, p):
    def flux(t, **kw):
        result = model(p, t, **kw)
        return result[0] if isinstance(result, tuple) else result

    out = flux(TIME, exp_time=0)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0) & (out <= 1 + 1e-10))
    np.testing.assert_allclose(out[[0, -1]], 1, atol=1e-12)
    assert flux(np.array([p[2] / 2]), exp_time=0)[0] == pytest.approx(1, abs=1e-12)
    times = np.array([-0.09, 0.0, 0.09])
    exp, count = 0.004, 7
    samples = (times[:, None] + ((np.arange(count) + 0.5) / count - 0.5) * exp).ravel()
    averaged = flux(samples, exp_time=0).reshape(-1, count).mean(axis=1)
    np.testing.assert_allclose(
        flux(times, exp_time=exp, supersample_factor=count), averaged, atol=1e-13
    )
    np.testing.assert_allclose(
        flux(times, exp_time=exp, supersample_factor=1), flux(times, exp_time=0)
    )
    assert flux(np.array([]), exp_time=0).size == 0
    assert np.isfinite(flux(np.array([0.0]))).all()


def test_near_circle_limit():
    x = np.linspace(0.89, 1.11, 301)
    from JoJo.models.oblate import ellipse_occultation

    reference = ellipse_occultation(x, 0.02, 0.1, 0.1, 0.3, 0.2)[0]
    for f in [1e-8, 1e-6, 1e-4]:
        got = ellipse_occultation(x, 0.02, 0.1, 0.1 * (1 - f), 0.3, 0.2)[0]
        assert np.max(abs(got - reference)) < 0.03 * f + 2e-8


def test_tall_egg_flux_and_explicit_four_crossing_limit():
    got, _ = egg_occultation(0, 1.10, 0.08, 0.1, 0.15, 0.3, 0.2, n_step=64)
    assert got[0] == pytest.approx(area_flux(0, 1.10, 0.08, 0.1, 0.15), abs=2e-8)
    with pytest.raises(ValueError, match="exactly two"):
        egg_occultation(0, 0.89, 0.8, 0.8, 0.1, 0.3, 0.2)


@pytest.mark.parametrize("r_in", [0.1, 0.3])
def test_ring_tangent_flux_with_and_without_planet_overlap(r_in):
    # The first inner radius exercises ring arc selection; the second uses
    # the disjoint annulus path. The outer boundary is tangent at (0, 1).
    from JoJo.models.ring import ring_occultation

    got = ring_occultation(0, 0.9, 0.03, r_in, 0.8, 0.875, 0.3, 0.2)[0]
    reference = area_flux(0, 0.9, 1, 1, 1, ring=(r_in, 0.8, 0.125, 1), planet_radius=0.03)
    assert got == pytest.approx(reference, abs=3e-8)


@pytest.mark.parametrize("a_p,a_m", [(0.8, 0.8), (0.8, 0.7)])
def test_egg_tangent_flux(a_p, a_m):
    from JoJo.models.egg import egg_occultation

    got, _ = egg_occultation(0, 0.9, a_p, a_m, 0.1, 0.3, 0.2, n_step=64)
    reference = area_flux(0, 0.9, a_p, a_m, 0.1)
    np.testing.assert_allclose(got, reference, atol=3e-8, rtol=0)


def test_randomized_ring_boundaries_against_independent_area():
    rng = np.random.default_rng(437)
    for _ in range(12):
        f = rng.uniform(0.1, 0.85)
        angle = rng.uniform(-np.pi, np.pi)
        distance = rng.uniform(0.8, 1.15)
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        got = ring_occultation(x, y, 0.1, 0.12, 0.2, f, 0.3, 0.2, 0.7, 64)
        ref = area_flux(x, y, 0.2, 0.2, 0.2, ring=(0.12, 0.2, 1 - f, 0.7), planet_radius=0.1)
        assert got[0] == pytest.approx(ref, abs=3e-8)


@pytest.mark.parametrize("model,p", [(oblate_lc, OBLATE), (egg_lc, EGG), (ring_lc, RING)])
def test_public_input_validation(model, p):
    with pytest.raises(ValueError):
        model(p, [0, -0.1], exp_time=0)
    with pytest.raises(ValueError):
        model(p, [0], n_step=0)
    with pytest.raises(ValueError):
        model(p, [0], supersample_factor=0)
    bad = p.copy()
    bad[2] = -1
    with pytest.raises(ValueError):
        model(bad, [0])
