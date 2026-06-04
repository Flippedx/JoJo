"""JAX root-polishing utilities for differentiable polynomial roots."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def multi_quadratic(a0, b0, c0):
    a = b0 / a0
    b = c0 / a0
    a_mid = -0.5 * a
    sqrt_delta = jnp.sqrt(a_mid * a_mid - b + 0j)
    return a_mid - sqrt_delta, a_mid + sqrt_delta


def fast_cubic(a, b, c, d, all_roots=True):
    d0 = b**2 - 3 * a * c
    d1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    d2 = jnp.sqrt(d1**2 - 4 * d0**3 + 0j)
    d3 = jnp.where(d2 == d1, d1 + d2, d1 - d2)
    cap_c = (d3 / 2) ** (1 / 3)
    d4 = jnp.where((cap_c == 0) & (d0 == 0), 0, d0 / cap_c)
    pcru = (-1 - jnp.sqrt(jnp.asarray(-3 + 0j))) / 2
    x0 = -1 / (3 * a) * (b + cap_c + d4)
    if not all_roots:
        return x0
    x1 = -1 / (3 * a) * (b + cap_c * pcru + d4 / pcru)
    x2 = -1 / (3 * a) * (b + cap_c * pcru**2 + d4 / pcru**2)
    return jnp.stack([x0, x1, x2])


def multi_quartic(a0, b0, c0, d0, e0):
    """Closed-form JAX initializer for batches of quartic roots."""
    a = b0 / a0
    b = c0 / a0
    c = d0 / a0
    d = e0 / a0

    a_quarter = 0.25 * a
    a_quarter2 = a_quarter * a_quarter
    p = 3 * a_quarter2 - 0.5 * b
    q = a * a_quarter2 - b * a_quarter + 0.5 * c
    r = 3 * a_quarter2 * a_quarter2 - b * a_quarter2 + c * a_quarter - d

    z0 = fast_cubic(1, p, r, p * r - 0.5 * q * q, all_roots=False)
    s = jnp.sqrt(2 * p + 2 * z0 + 0j)
    t = jnp.where(jnp.abs(s) < 1e-8, z0 * z0 + r, -q / s)

    r0, r1 = multi_quadratic(1, s, z0 + t)
    r2, r3 = multi_quadratic(1, -s, z0 - t)
    return jnp.stack([r0, r1, r2, r3], axis=-1) - a_quarter[..., None]


@jax.jit
def AE_roots0(coff):
    """Compute deterministic initial guesses for Aberth-Ehrlich roots."""

    def uv_bounds(coefficients):
        upper = 1 + 1 / jnp.abs(coefficients[0]) * jnp.max(jnp.abs(coefficients[:-1]))
        lower = jnp.abs(coefficients[-1]) / (
            jnp.abs(coefficients[-1]) + jnp.max(jnp.abs(coefficients[:-1]))
        )
        return upper, lower

    upper, lower = uv_bounds(coff)
    r = jax.random.uniform(jax.random.PRNGKey(0), shape=(coff.shape[0] - 1,), minval=lower, maxval=upper)
    phi = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(coff.shape[0] - 1,),
        minval=0,
        maxval=2 * jnp.pi,
    )
    return r * jnp.exp(1j * phi)


@partial(jax.jit, static_argnames=("max_iter",))
def Aberth_Ehrlich(coff, roots, max_iter=50, tol=2e-14):
    """Polish polynomial roots with implicit differentiation.

    Adopted from the Aberth-Ehrlich method described in
    https://arxiv.org/abs/2206.00482. ``jax.lax.custom_root`` gives
    derivatives from the converged polynomial equation instead of
    differentiating through the iterative polishing loop.
    """
    derp = jnp.polyder(coff)
    off_diagonal = ~jnp.eye(roots.shape[0], dtype=bool)

    def loop_body(carry):
        roots, correction, n_iter = carry
        ratio = jnp.polyval(coff, roots) / jnp.polyval(derp, roots)
        differences = roots[:, None] - roots[None, :]
        safe_differences = jnp.where(off_diagonal, differences, 1)
        inverse_differences = jnp.where(off_diagonal, 1 / safe_differences, 0)
        sum_term = jnp.sum(inverse_differences, axis=1)
        correction = ratio / (1 - ratio * sum_term)
        correction = jnp.where(jnp.isfinite(correction), correction, 0)
        return roots - correction, correction, n_iter + 1

    def cond_fun(carry):
        _, correction, n_iter = carry
        return jnp.any(jnp.abs(correction) > tol) & (n_iter < max_iter)

    def solve(_, roots0):
        initial_correction = jnp.full_like(roots0, jnp.inf)
        return lax.while_loop(cond_fun, loop_body, (roots0, initial_correction, 0))[0]

    def tangent_solve(g, y):
        # Each root independently satisfies P(root) = 0, so the linearized
        # Jacobian is diagonal. Evaluating g(1) extracts its diagonal.
        return y / g(jnp.ones_like(y))

    polynomial = lambda x: jnp.polyval(coff, x)
    return lax.custom_root(polynomial, roots, solve=solve, tangent_solve=tangent_solve)


@partial(jax.jit, static_argnames=("max_iter",))
def get_roots(coefficients, max_iter=10):
    """Return polished quartic roots for coefficient rows ``a,b,c,d,e``."""
    roots0 = jax.lax.stop_gradient(
        multi_quartic(
            coefficients[:, 0],
            coefficients[:, 1],
            coefficients[:, 2],
            coefficients[:, 3],
            coefficients[:, 4],
        )
    )
    roots = jax.vmap(lambda coeff, initial: Aberth_Ehrlich(coeff, initial, max_iter=max_iter))(
        coefficients,
        roots0,
    )
    return jnp.where(jnp.isfinite(jnp.real(roots)) & jnp.isfinite(jnp.imag(roots)), roots, 1.0 + 0.0j)
