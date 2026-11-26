"""The shape viewer must use the same geometry and parameter order as JoJo."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch

from JoJo import plot_shape


OBLATE = [0, 0.3, 10, np.sqrt(0.12 * 0.07), 1 - 0.07 / 0.12, 0, 0, np.pi / 2, 0.3, 0.2, 0]
EGG = [0, 0.3, 10, 0.12, 0.08, 0.07, 0, 0, np.pi / 2, 0.3, 0.2, 0]
RING = [0, 0.3, 10, 0.1, 0.12, 0.2, 0.4, 0, 0, np.pi / 2, 0.3, 0.2, 0, 0.5]


def test_oblate_outline_uses_equal_area_radius_and_rotation():
    fig, ax = plot_shape("oblate", OBLATE)
    x, y = ax.lines[0].get_data()
    assert max(x) == pytest.approx(0.12, abs=1e-10)
    assert max(y) == pytest.approx(0.07, abs=1e-10)
    assert ax.get_aspect() == 1
    plt.close(fig)

    turned = OBLATE.copy()
    turned[5] = np.pi / 2
    fig, ax = plot_shape("oblate", turned)
    x, y = ax.lines[0].get_data()
    assert max(abs(x)) == pytest.approx(0.07, abs=1e-10)
    assert max(abs(y)) == pytest.approx(0.12, abs=1e-10)
    plt.close(fig)


def test_egg_outline_preserves_different_halves_and_vertical_semi_axis():
    fig, ax = plot_shape("egg", EGG)
    x, y = ax.lines[0].get_data()
    assert max(x) == pytest.approx(0.12, abs=1e-10)
    assert min(x) == pytest.approx(-0.08, abs=1e-10)
    assert max(y) == pytest.approx(0.07, abs=1e-10)
    assert min(y) == pytest.approx(-0.07, abs=1e-10)
    plt.close(fig)


def test_egg_axes_follow_the_rotated_halves():
    turned = EGG.copy()
    turned[6] = np.pi / 2
    fig, ax = plot_shape("egg", turned)
    horizontal, vertical = ax.lines[1:]
    np.testing.assert_allclose(horizontal.get_xydata(), [(0, -0.08), (0, 0.12)], atol=1e-10)
    np.testing.assert_allclose(vertical.get_xydata(), [(0.07, 0), (-0.07, 0)], atol=1e-10)
    plt.close(fig)


def test_ring_outlines_and_planet_use_projected_radii():
    fig, ax = plot_shape("ring", RING)
    outer_x, outer_y = ax.lines[0].get_data()
    inner_x, inner_y = ax.lines[1].get_data()
    assert max(outer_x) == pytest.approx(0.2, abs=1e-10)
    assert max(outer_y) == pytest.approx(0.12, abs=1e-10)
    assert max(inner_x) == pytest.approx(0.12, abs=1e-10)
    assert max(inner_y) == pytest.approx(0.072, abs=1e-10)
    assert any(isinstance(p, Circle) and p.radius == 0.1 for p in ax.patches)
    plt.close(fig)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.75, 1.0])
def test_ring_fill_uses_model_alpha_and_planet_is_in_front(alpha):
    parameters = RING.copy()
    parameters[-1] = alpha
    fig, ax = plot_shape("ring", parameters)
    ring = next(p for p in ax.patches if isinstance(p, PathPatch))
    planet = next(p for p in ax.patches if isinstance(p, Circle))
    assert ring.get_alpha() == pytest.approx(alpha)
    assert planet.get_zorder() > ring.get_zorder()
    assert planet.get_zorder() > max(edge.get_zorder() for edge in ax.lines[:2])
    plt.close(fig)


def test_ring_theta_rotates_the_projected_annulus():
    turned = RING.copy()
    turned[7] = np.pi / 2
    fig, ax = plot_shape("ring", turned)
    outer_x, outer_y = ax.lines[0].get_data()
    assert max(abs(outer_x)) == pytest.approx(0.12, abs=1e-10)
    assert max(abs(outer_y)) == pytest.approx(0.2, abs=1e-10)
    plt.close(fig)


def test_show_star_places_planet_at_impact_and_reuses_axes():
    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_shape("egg", EGG, ax=ax, show_star=True)
    assert returned_fig is fig and returned_ax is ax
    x, y = ax.lines[0].get_data()
    assert max(y) == pytest.approx(0.3 + 0.07, abs=1e-10)
    assert any(isinstance(p, Circle) and p.radius == 1 for p in ax.patches)
    assert not ax.axison
    assert ax.get_legend() is None
    arrows = [annotation for annotation in ax.texts if annotation.arrow_patch is not None]
    horizontal, vertical = arrows
    assert horizontal.xyann[0] < min(x) < max(x) < horizontal.xy[0]
    assert horizontal.xyann[1] == horizontal.xy[1] == EGG[1]
    assert vertical.xyann[1] < min(y) < max(y) < vertical.xy[1]
    assert vertical.xyann[0] == vertical.xy[0] == 0
    assert {label.get_text() for label in ax.texts if label.get_text()} == {
        "$x$", "$y$", "$a_-$", "$a_+$", r"$b_{\mathrm{egg}}$",
    }
    plt.close(fig)


@pytest.mark.parametrize("model,params", [("oblate", OBLATE), ("egg", EGG), ("ring", RING)])
def test_plot_shape_rejects_wrong_size_and_invalid_period(model, params):
    with pytest.raises(ValueError, match="Expected"):
        plot_shape(model, params[:-1])
    invalid = params.copy()
    invalid[2] = -1
    with pytest.raises(ValueError, match="period"):
        plot_shape(model, invalid)


def test_plot_shape_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="model"):
        plot_shape("unknown", OBLATE)
    invalid = OBLATE.copy()
    invalid[4] = 0.999
    with pytest.raises(ValueError, match="rp_eq"):
        plot_shape("oblate", invalid)
    invalid = EGG.copy()
    invalid[5] = -0.05
    with pytest.raises(ValueError, match="b_egg"):
        plot_shape("egg", invalid)
    invalid = RING.copy()
    invalid[-1] = 1.1
    with pytest.raises(ValueError, match="alpha"):
        plot_shape("ring", invalid)
