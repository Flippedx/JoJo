# API reference

Import the public functions directly from `JoJo`:

```python
from JoJo import oblate_lc, spherical_lc, egg_lc, ring_lc, plot_shape
```

| Function | Purpose | Return value |
| --- | --- | --- |
| `oblate_lc(transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=32)` | Oblate primary transit | `(flux, contacts)` |
| `spherical_lc(transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5)` | Equal-area spherical comparison | `(flux, contacts)` |
| `egg_lc(transit_parameters, time_array, photo_ecc=False, exp_time=None, supersample_factor=5, n_step=32)` | Asymmetric primary transit | `(flux, contacts)` |
| `ring_lc(transit_parameters, time_array, photo_ecc=False, n_step=32, *, exp_time=None, supersample_factor=5)` | Ringed primary transit | `flux` |
| `plot_shape(model, transit_parameters, *, ax=None, show_star=False)` | Projected silhouette for `"oblate"`, `"egg"`, or `"ring"` | `(figure, axes)` |

`time_array` must be a finite, sorted one-dimensional array in days. `flux` is a NumPy array normalized to one outside transit. See [Models](models.md) for the exact parameter vectors and units.

## Examples

The [quick start](quick-start.md) contains an oblate transit and spherical comparison. The [model guide](models.md) walks through all three models using Kepler-167e, with examples of `plot_shape` and equal-area comparisons.
