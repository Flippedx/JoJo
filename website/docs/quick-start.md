# Quick start

## Install

JoJo requires Python 3.10 or newer. Install the package from the [source repository](https://github.com/Flippedx/JoJo):

```bash
git clone https://github.com/Flippedx/JoJo.git
cd JoJo
python -m pip install -e '.[plot]'
```

The `plot` extra installs Matplotlib for the examples. For light-curve calculations alone, `python -m pip install -e .` is enough.

## Compute and compare a transit

```python
import numpy as np
import matplotlib.pyplot as plt
from JoJo import oblate_lc, spherical_lc

time = np.linspace(-0.15, 0.15, 301)  # days relative to mid-transit
params = [0, 0.3, 10, 0.1, 0.2, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]

oblate_flux, contacts = oblate_lc(params, time, exp_time=0)
sphere_flux, _ = spherical_lc(params, time, exp_time=0)

plt.plot(time, oblate_flux, label="Oblate")
plt.plot(time, sphere_flux, label="Equal-area sphere", linestyle="--")
plt.xlabel("Time from reference transit [days]")
plt.ylabel("Normalized flux")
plt.legend()
plt.show()
```

The ordered vector is `tc, b, period, rp_me, f, theta, ecc, omega, u1, u2, log10_rho_star`. In this example `f=0.2` is projected flattening, `theta=0.4` radians is the projected orientation, and `rp_me=0.1` is the equal-area radius in stellar-radius units. `spherical_lc` ignores `f` and `theta` while using the same radius.

## More examples

For a runnable comparison of all three non-spherical models, see `examples/basic.py` and `examples/basic_usage.ipynb` in the [repository](https://github.com/Flippedx/JoJo/tree/master/examples).

!!! note "Exposure times"
    `exp_time=0` evaluates the model at each requested time. With `exp_time=None`, JoJo infers the median cadence. Midpoint supersampling begins only when the exposure is strictly longer than five minutes; adjust `supersample_factor` if needed.
