# Models and parameters

JoJo computes primary-transit light curves for projected silhouettes crossing a unit-radius star with quadratic limb darkening. The model functions take an ordered parameter vector and a one-dimensional array of observation times. Each flux array has one value per observation time and is normalized to one outside transit.

| Model | Parameter order | Return value |
| --- | --- | --- |
| [Oblate](models/oblate.md) and equal-area sphere | `tc, impact, period, rp_me, f, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| [Egg](models/egg.md) | `tc, impact, period, a_p, a_m, b_egg, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| [Ring](models/ring.md) | `tc, impact, period, r_p, r_in, r_out, f_r, theta, ecc, omega, u1, u2, log10_rho_star, alpha` | `flux` |

## Shared orbital and stellar inputs

| Parameter | Meaning |
| --- | --- |
| `tc` | Reference transit time in days, in the same time system as `time_array`. For an eccentric orbit, it is the time at true anomaly `π/2 − omega`. Minimum projected separation can occur at a different time. |
| `impact` | Transit impact parameter, conventionally written *b*, in stellar-radius units. |
| `period` | Orbital period in days. |
| `ecc`, `omega` | Eccentricity in `[0, 1)` and argument of periastron in radians. |
| `u1`, `u2` | Quadratic stellar limb-darkening coefficients. |
| `log10_rho_star` | Base-10 logarithm of the stellar density expressed in g/cm³. |

Shape dimensions are in stellar-radius units. The sky-plane orientation `theta` is in radians and increases counterclockwise. The model pages explain the dimensions specific to each silhouette.

## Kepler-167e setup

Run this block before any model example. It reproduces the time grid and the orbital, stellar, and reference-radius inputs used in the [example notebook](https://github.com/Flippedx/JoJo/blob/master/examples/basic_usage.ipynb). The three examples assign illustrative shape parameters to this system and compute model curves for those assumptions.

```python
import matplotlib.pyplot as plt
import numpy as np

from JoJo import egg_lc, oblate_lc, plot_shape, ring_lc, spherical_lc

tc, impact, period = 0.0, 0.233, 1071.23228
r0 = 0.1281
ecc, omega = 0.062, np.deg2rad(201)
u1, u2, log10_rho_star = 0.6226, 0.0498, 0.46

exp_time = 1 / 1440  # one minute in days
time = tc + np.arange(-0.5, 0.5, exp_time)
```

Install the [plotting extra](quick-start.md#install) to use Matplotlib and `plot_shape`.

## Compare a model with an equal-area sphere

Each model example computes `flux` and `sphere_flux` on the same time grid, with the same orbit and limb darkening. The difference `1e6 * (flux - sphere_flux)` measures the effect of changing the shape at fixed inputs, in parts per million (ppm). A positive value means the model with the specified shape blocks less light than the sphere at that time. Two silhouettes can cover the same area and still block different amounts of light because stellar intensity varies across the disk.

After you run a model example, the following code draws its silhouette, plots both light curves, and shows their difference.

```python
plot_shape(model, params)

fig, (ax_flux, ax_diff) = plt.subplots(
    2, 1, figsize=(7, 5), sharex=True, height_ratios=(2, 1),
    layout="constrained",
)
time_from_transit = time - tc
ax_flux.plot(time_from_transit, flux, label=model.capitalize())
ax_flux.plot(time_from_transit, sphere_flux, "--", label="Equal-area sphere")
ax_flux.set_ylabel("Normalized flux")
ax_flux.legend()

ax_diff.plot(time_from_transit, 1e6 * (flux - sphere_flux))
ax_diff.axhline(0, color="0.5", linestyle=":")
ax_diff.set_xlabel("Time from reference transit [days]")
ax_diff.set_ylabel("Model − sphere [ppm]")
plt.show()
```

Pass `show_star=True` to `plot_shape` to draw the unit stellar disk with the planet centered at `(0, impact)` for reference.

## Sampling and numerical settings

- Pass finite, sorted times in days. `photo_ecc=False` computes Keplerian positions. Set it to `True` to use a constant projected velocity near transit.
- Set `exp_time` in days. Use `0` for instantaneous flux or `None` to infer the median cadence. Exposures longer than five minutes use `supersample_factor` midpoint samples per exposure, five by default. The one-minute examples use center-time flux.
- `n_step=32` is the default quadrature resolution per boundary arc for the oblate, egg, and ring models. Increase it and compare the curves when you need to check numerical convergence.
- Oblate, egg, and spherical models estimate contact times from the sampled geometry. Grazing transits or sparse sampling can yield fewer than four contacts. The ring model returns flux alone.

See the [API reference](api.md) for the full call signatures.
