import numpy as np
import pytest

from JoJo.core.orbit import mean_to_true, true_to_mean, exposure_grid


@pytest.mark.parametrize("ecc", [0, 0.2, 0.6, 0.8])
def test_anomaly_roundtrip(ecc):
    mean = np.linspace(-np.pi, np.pi, 301)
    true = mean_to_true(mean, ecc)
    recovered = true_to_mean(true, ecc)
    np.testing.assert_allclose(recovered, mean, atol=1e-7, rtol=0)


def test_exposure_samples_midpoints_even_for_short_exposure():
    _, samples, count = exposure_grid([0.0], 0.001, 4)
    np.testing.assert_allclose(samples, [-0.000375, -0.000125, 0.000125, 0.000375])
    assert count == 4


def test_inferred_exposure_matches_explicit_cadence():
    time = np.linspace(0, 1, 11)
    _, implicit, count = exposure_grid(time)
    _, explicit, count2 = exposure_grid(time, 0.1)
    np.testing.assert_allclose(implicit, explicit)
    assert count == count2
