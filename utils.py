import jax.numpy as jnp
from params import *
from jax import config
config.update("jax_enable_x64", True)

def h(phi):
    """KKS interpolation h(φ) = φ^2(3 - 2φ)."""
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return -jnp.float64(2.0)*phi**3 + jnp.float64(3.0)*phi**2

def h_p(phi):
    """First derivative of h(φ)."""
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return -jnp.float64(6.0)*phi**2 + jnp.float64(6.0)*phi

def h_pp(phi):
    """Second derivative of h(φ)."""
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return -jnp.float64(12.0)*phi + jnp.float64(6.0)

def g_p(phi):
    """Derivative of double-well potential."""
    phi = jnp.asarray(phi, dtype=jnp.float64)
    return jnp.float64(2.0) * phi * (1 - phi)**2 - jnp.float64(2.0) * (1 - phi) * phi**2

def compute_phi_ic(x, t):
    """Initial condition for φ with x in [-0.5, 0.5]."""
    x = jnp.asarray(x, dtype=jnp.float64)
    scale = jnp.sqrt(omega_phi) / jnp.sqrt(jnp.float64(2.0) * alpha_phi)
    arg = scale * x / x_coef  # normalize by x_coef to match spatial scaling
    return jnp.float64(0.5) * (jnp.float64(1.0) - jnp.tanh(arg))

def compute_c_ic(x, t):
    """Initial condition for concentration c."""
    return h(compute_phi_ic(x, t)) * c_s  # 0 * (1 - h_phi) dropped for clarity

def compute_phi_bc(x, t):
    """Boundary condition for φ with x in [-0.5, 0.5]."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.float64(0.5) - x

def compute_c_bc(x, t):
    """Boundary condition for c, same as φ."""
    return compute_phi_bc(x, t)