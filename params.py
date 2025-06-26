from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Model parameters (cast to float64 explicitly)
alpha_phi = jnp.float64(9.62e-5)
omega_phi = jnp.float64(1.663e7)
M         = jnp.float64(8.5e-10) / (2 * jnp.float64(5.35e7))
A         = jnp.float64(5.35e7)
L         = jnp.float64(1e-11)
c_s       = jnp.float64(1.0)
c_l       = jnp.float64(5100/1.43e5)
dc        = c_s - c_l  # already float64 since c_s, c_l are

# normalized domain for x (microns)
x_min = jnp.float64(-0.5)
x_max = jnp.float64(0.5)

# normalized domain for t (seconds)
t_min = jnp.float64(0.0)
t_max = jnp.float64(1)

# denormalizing constants:
x_coef = 1e4
t_coef = 1e-5

# Spans
x_span = (x_min, x_max)
t_span = (t_min, t_max)
