import jax.numpy as jnp
from jax import config
# config.update("jax_enable_x64", True)

# Model parameters (cast to float64 explicitly)
alpha_phi = jnp.float64(9.62e-5)
omega_phi = jnp.float64(1.663e7)
M         = jnp.float64(8.5e-10) / (2 * jnp.float64(5.35e7))
A         = jnp.float64(5.35e7)
L         = jnp.float64(1e-11)
c_se       = jnp.float64(1.0)
c_le       = jnp.float64(5100/1.43e5)
dc        = c_se - c_le  # already float64 since c_s, c_l are

inp_idx = {'x':0, 't': 1, 'L': 2, 'M': 3}
out_idx = {'phi': 0, 'c':1}
in_axes = (0,0,None,None)
phys_span = {
    "x": (-0.5e5, 0.5e5),   # x runs from 0 to 1
    "t": (0.0, 100000),   # t runs from 0 to 2
    "L": (0, 2.5),   # L varies between 0.5 and 2.5
    "M": (0, 5.0),   # M varies between 1.0 and 5.0
}
# Normalized domains (e.g. we’ll map each phys span into [-1, +1])
norm_span = {
    "x": (-0.5, 0.5),
    "t": (0, 1.0),
    "L": (0, 1.0),
    "M": (0, 1.0),
}
