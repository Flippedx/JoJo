import numpy as np
import jax.numpy as jnp
import jax
from jax import lax

# Aberth-Ehrlich method
@jax.jit
def AE_roots0(coff):
    """
    Computes the initial guesses using the Aberth-Ehrlich method.

    Args:
        coff (ndarray): Coefficients of the polynomial.

    Returns:
        ndarray: Array of initial guesses for the roots of the polynomial.
    """
    def UV(coff):
        U = 1 + 1 / jnp.abs(coff[0]) * jnp.max(jnp.abs(coff[:-1]))
        V = jnp.abs(coff[-1]) / (jnp.abs(coff[-1]) + jnp.max(jnp.abs(coff[:-1])))
        return U, V
    def Roots0(coff):
        U , V = UV(coff)
        r = jax.random.uniform(jax.random.PRNGKey(0),shape=(coff.shape[0]-1,),minval=V,maxval=U)
        phi = jax.random.uniform(jax.random.PRNGKey(0),shape=(coff.shape[0]-1,),minval=0,maxval=2*jnp.pi)
        return r * jnp.exp(1j * phi)
    roots = Roots0(coff)
    return roots
@jax.jit
def Aberth_Ehrlich(coff, roots, MAX_ITER=50):
    """
    Solves a polynomial equation using the Aberth-Ehrlich method.
    Adopted from https://arxiv.org/abs/2206.00482 Hossein Fatheddin
    Use jax.lax.custom_root to get precise derivative in automatic differentiation

    Args:
        coff (ndarray): Coefficients of the polynomial equation.
        roots (ndarray): Initial guesses for the roots of the polynomial equation.
        MAX_ITER (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        ndarray: The roots of the polynomial equation.

    """
    derp = jnp.polyder(coff)
    mask = 1 - jnp.eye(roots.shape[0])
    # alpha = jnp.abs(coff)*((2*jnp.sqrt(2))*1j+1)

    def loop_body(carry):
        roots, coff, cond, ratio_old, n_iter = carry
        # h = jnp.polyval(coff, roots)
        # b = jnp.polyval(alpha, jnp.abs(roots))
        ratio = jnp.polyval(coff, roots) / jnp.polyval(derp, roots)

        sum_term = jnp.nansum(mask * 1 / (roots - roots[:, None]), axis=0)
        w = ratio / (1 - (ratio * sum_term))
        cond = jnp.abs(w) > 2e-14
        # cond = jnp.abs(h) > 1e-15*b
        roots -= w
        return (roots, coff, cond, ratio, n_iter + 1)

    def cond_fun(carry):
        roots, coff, cond, ratio, n_iter = carry
        return cond.any() & (n_iter < MAX_ITER)

    f = lambda x: jnp.polyval(coff, x)
    solution = lambda f, x0: lax.while_loop(cond_fun, loop_body, (x0, coff, jnp.ones_like(x0, dtype=bool), x0, 0))[0]
    sclar = lambda g, y: jnp.linalg.solve(jax.jacobian(g, holomorphic=True)(y), y)

    return lax.custom_root(f, roots, solve=solution, tangent_solve=sclar)

# closed-form roots for quadratic, cubic, and quartic polynomials
# multi_quadratic and multi_quartic adapted from https://github.com/NKrvavica/fqs
# fast_cubic rewritten for complex polynomials
# https://arxiv.org/abs/2207.12412
# author: Keming Zhang

def multi_quadratic(a0, b0, c0):
    ''' Analytical solver for multiple quadratic equations
    (2nd order polynomial), based on `numpy` functions.
    Parameters
    ----------
    a0, b0, c0: array_like
        Input data are coefficients of the Quadratic polynomial::
            a0*x^2 + b0*x + c0 = 0
    Returns
    -------
    r1, r2: ndarray
        Output data is an array of two roots of given polynomials.
    '''
    ''' Reduce the quadratic equation to to form:
        x^2 + ax + b = 0'''
    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = np.sqrt(delta + 0j)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2

def fast_cubic(a, b, c, d, all_roots=True):
    d0 = b**2-3*a*c
    d1 = 2*b**3-9*a*b*c+27*a**2*d
    d2 = np.sqrt(d1**2-4*d0**3+0j)
    
    d3 = d1-d2
    d3[d2 == d1] += 2*d2[d2 == d1]
    
    C = (d3/2)**(1/3)
    d4 = d0/C
    d4[(C == 0)*(d0 == 0)] = 0
    pcru = (-1-(-3)**0.5)/2
    if all_roots:
        x0 = -1/3/a*(b+C+d4)
        x1 = -1/3/a*(b+C*pcru+d4/pcru)
        x2 = -1/3/a*(b+C*pcru**2+d4/pcru**2)
        return np.array([x0, x1, x2])
    else:
        return -1/3/a*(b+C+d4)

def multi_quartic(a0, b0, c0, d0, e0):
    ''' Analytical closed-form solver for multiple quartic equations
    (4th order polynomial), based on `numpy` functions. Calls
    `multi_cubic` and `multi_quadratic`.
    Parameters
    ----------
    a0, b0, c0, d0, e0: array_like
        Input data are coefficients of the Quartic polynomial::
            a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0
    Returns
    -------
    r1, r2, r3, r4: ndarray
        Output data is an array of four roots of given polynomials.
    '''

    ''' Reduce the quartic equation to to form:
        x^4 + ax^3 + bx^2 + cx + d = 0'''
    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = fast_cubic(1, p, r, p*r - 0.5*q*q, all_roots=False)

    # Additional variables
    s = np.sqrt(2*p + 2*z0 + 0j)
    t = np.zeros_like(s)
    mask = (np.abs(s) < 1e-8)
    t[mask] = z0[mask]*z0[mask] + r[mask]
    t[~mask] = -q[~mask] / s[~mask]

    # Compute roots by quadratic equations
    r0, r1 = multi_quadratic(1, s, z0 + t) - a0
    r2, r3 = multi_quadratic(1, -s, z0 - t) - a0

    return np.array([r0, r1, r2, r3]).T