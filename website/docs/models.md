# Models and parameters

JoJo takes an **ordered vector** of parameters rather than keyword arguments for the physical model. Times and periods are in days; angles are in radians; radii are divided by the stellar radius. The star has unit radius and quadratic limb-darkening coefficients `u1, u2`. `log10_rho_star` is the base-10 log of stellar density in g/cm³.

| Model | Parameter order | Output |
| --- | --- | --- |
| Oblate and sphere | `tc, b, period, rp_me, f, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| Asymmetric egg | `tc, b, period, a_p, a_m, b_egg, theta, ecc, omega, u1, u2, log10_rho_star` | `(flux, contacts)` |
| Ringed planet | `tc, b, period, r_p, r_in, r_out, f_r, theta, ecc, omega, u1, u2, log10_rho_star, opacity` | `flux` |

## Oblate and spherical

`oblate_lc` models a projected ellipse. Its equal-area radius is `rp_me = rp_eq * sqrt(1-f)`, where `rp_eq` is the horizontal equatorial semi-axis before rotation. The projected polar semi-axis is `rp_eq * (1-f)`. `theta` rotates the ellipse counterclockwise on the sky. `spherical_lc` uses `rp_me` as the sphere radius and ignores `f` and `theta`.

## Asymmetric egg

`egg_lc` joins a right half-ellipse of width `a_p` to a left half-ellipse of width `a_m`. Both share vertical semi-axis `b_egg`; the full height is `2*b_egg`. `theta` rotates the projected silhouette.

## Ringed planet

`ring_lc` combines an opaque spherical planet of radius `r_p` with a projected ring. `r_in` and `r_out` are its inner and outer horizontal semi-axes. Their vertical semi-axes are scaled by `1-f_r`; `theta` sets sky-plane orientation. `opacity` is the fraction of incident stellar light blocked by the ring outside the planet: zero is transparent and one is fully blocking. The ring model returns only the flux array.

## Numerical options and limits

- `photo_ecc=False` uses Keplerian positions. `True` uses a constant projected velocity approximation around transit.
- `n_step=32` sets the Gauss–Legendre quadrature nodes per boundary arc. Increase it to check numerical convergence.
- Contact times are estimates on the supplied sample grid; grazing or sparse events may yield fewer contacts.
- Radii must be between zero and one; flattening is in `[0, 1)`; rings require `r_p <= r_in <= r_out` and `opacity` in `[0, 1]`.
- The oblate model switches to a spherical approximation when `f * rp_eq**2 < 1e-6`. Shape derivatives should not be interpreted across this switch.
- For eccentric orbits, `tc` is defined at a reference true anomaly of `pi/2 - omega`; it need not be the instant of minimum projected separation.

See the [API reference](api.md) for each public function's arguments.
