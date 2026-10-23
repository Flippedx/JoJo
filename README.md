# JoJo

JoJo calculates primary-transit light curves for oblate planets, asymmetric planets formed from two half-ellipses (egg), and spherical planets with rings. Models assume a unit-radius spherical star with quadratic limb darkening.

## Installation

Python 3.10 or newer is required. NumPy is the only core dependency. From the repository directory:

```bash
python -m pip install -e .
# Include plotting support for the examples:
python -m pip install -e '.[plot]'
```

## Quick start

```python
import numpy as np
from JoJo import oblate_lc, spherical_lc, egg_lc, ring_lc

time = np.linspace(-0.15, 0.15, 301)
parameters = [0, 0.3, 10, 0.1, 0.2, 0.4, 0, np.pi/2, 0.3, 0.2, 0]
flux, contacts = oblate_lc(parameters, time, exp_time=0)
```

Run all three models with `python -m examples.basic`. The [basic usage notebook](examples/basic_usage.ipynb) explains each model and compares Kepler-167e shape signals against equal-area spherical planets. Open it with Jupyter or a notebook-capable editor; install Jupyter separately if needed.

## Parameters and outputs

All radii are in stellar-radius units, times and periods in days, and angles in radians. `log10_rho_star` is log10 stellar density in g/cm³; `b` is the impact parameter; `u1, u2` are quadratic limb-darkening coefficients.

| Model | Ordered parameter vector | Return |
| --- | --- | --- |
| `oblate_lc`, `spherical_lc` | `tc, b, period, rp_me, f, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| `egg_lc` | `tc, b, period, a_p, a_m, b_egg, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| `ring_lc` | `tc, b, period, r_p, r_in, r_out, f_r, theta, ecc, omega, u1, u2, log10_rho_star, opacity` | `flux` |

For oblate planets, `rp_me = rp_eq * sqrt(1-f)` is the equal-area radius. Ring `r_in, r_out` are equatorial semi-axes; their projected polar semi-axes are `r * (1-f_r)`. The ring parameter `f_r` is flattening, not inclination; opacity ranges from zero to one. Egg `a_p, a_m` are right/left horizontal semi-axes and `b_egg` is their shared vertical semi-axis.

Use `exp_time=0` for instantaneous flux. Only exposures strictly longer than five minutes (`exp_time > 5/1440` days) use midpoint subsamples controlled by `supersample_factor` (default 5). Shorter exposures, including exactly five minutes, use the center-time flux without averaging. `None` infers exposure from the median cadence and applies the same threshold. Ellipse integrals use Gauss–Legendre quadrature with `n_step=32` nodes per arc by default; pass a larger value for a convergence check. `photo_ecc=False` uses Kepler positions; `True` uses a constant-velocity approximation near transit.

## Important limits

- Radii and semi-axes must be strictly between zero and one. Oblate/ring flattening must be in [0, 1); ring radii must satisfy `r_p <= r_in <= r_out`.
- Oblate and egg support two or four boundary intersections.
- Oblate uses a spherical approximation when `f * rp_eq**2 < 1e-6`; do not rely on shape derivatives across this switch.
- Contacts are sampled estimates. For eccentric orbits, `tc` follows the reference true anomaly `pi/2 - omega`, not an exactly minimized projected separation.

## Theory

The Green-theorem derivation is described by [Liu et al. (2024)](https://iopscience.iop.org/article/10.3847/1538-3881/ad9b8c). The spherical `occultquad` implementation is retained; see [Eastman, Gaudi & Agol (2013)](https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E/abstract).
