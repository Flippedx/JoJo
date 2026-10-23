import numpy as np
import pytest

from JoJo.core.orbit import mean_to_true, true_to_mean, exposure_grid


@pytest.mark.parametrize("ecc", [0, 0.2, 0.6, 0.8])
def test_anomaly_roundtrip(ecc):
    mean = np.linspace(-np.pi, np.pi, 301)
    true = mean_to_true(mean, ecc)
    recovered = true_to_mean(true, ecc)
    np.testing.assert_allclose(recovered, mean, atol=1e-7, rtol=0)


def test_long_exposure_samples_midpoints():
    _, samples, count = exposure_grid([0.0], 0.01, 4)
    np.testing.assert_allclose(samples, [-0.00375, -0.00125, 0.00125, 0.00375])
    assert count == 4


@pytest.mark.parametrize("exposure", [0, 1 / 1440, 0.003, 5 / 1440])
def test_short_exposure_uses_only_center(exposure):
    time = np.array([0.0, 0.1, 0.2])
    _, samples, count = exposure_grid(time, exposure, 7)
    np.testing.assert_array_equal(samples, time)
    assert count == 1


def test_supersampling_threshold_is_strict():
    _, _, count = exposure_grid([0.0], np.nextafter(5 / 1440, np.inf), 5)
    assert count == 5


@pytest.mark.parametrize("minutes,count", [(1, 1), (5, 1), (6, 5)])
def test_inferred_exposure_uses_five_minute_threshold(minutes, count):
    time = np.array([0.0, minutes / 1440])
    _, samples, actual_count = exposure_grid(time)
    assert actual_count == count
    if count == 1:
        np.testing.assert_array_equal(samples, time)


def test_inferred_exposure_matches_explicit_cadence():
    time = np.linspace(0, 1, 11)
    _, implicit, count = exposure_grid(time)
    _, explicit, count2 = exposure_grid(time, 0.1)
    np.testing.assert_allclose(implicit, explicit)
    assert count == count2
