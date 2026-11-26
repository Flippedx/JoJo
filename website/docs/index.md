<div class="jojo-hero" markdown>

<span class="jojo-eyebrow">Exoplanet transit modeling</span>

# JoJo

Model primary transits of **oblate**, **asymmetric**, and **ringed** planets with a small NumPy-based Python package. Compare an oblate planet with an equal-area sphere using the same orbital inputs.

<div class="jojo-actions" markdown>

[Get started](quick-start.md){ .md-button .md-button--primary }
[Explore the models](models.md){ .md-button }

</div>

</div>

<div class="jojo-grid" markdown="block">
<div class="jojo-card" markdown>

### Oblate planets

Calculate light curves for projected ellipses with adjustable flattening and sky-plane orientation.

</div>
<div class="jojo-card" markdown>

### Asymmetric planets

Represent a silhouette with two joined half-ellipses and independently chosen left and right radii.

</div>
<div class="jojo-card" markdown>

### Planetary rings

Calculate transits of a spherical planet surrounded by a projected ring with adjustable attenuation.

</div>
</div>

## A first light curve

```python
import numpy as np
from JoJo import oblate_lc

time = np.linspace(-0.15, 0.15, 301)
params = [0, 0.3, 10, 0.1, 0.2, 0.4, 0, np.pi / 2, 0.3, 0.2, 0]
flux, contacts = oblate_lc(params, time, exp_time=0)
```

Here, `flux` is a normalized light curve sampled at `time`. The optional `contacts` list contains approximate contact times from that grid. See the [quick start](quick-start.md) for installation, plotting, and an equal-area spherical comparison.

## What's included

- A unit-radius spherical star with quadratic limb darkening.
- Keplerian sky positions, with an optional constant-velocity approximation near transit.
- Exposure-time averaging for exposures longer than five minutes.

Read the [model guide](models.md) for the parameter order, units, and important limits, or the [API reference](api.md) for the public functions.
