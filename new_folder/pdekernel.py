import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import diffrax as dfx
from functools import partial

# config.update("jax_enable_x64", True)

class PDEKernel:
    def __init__(self, params):
        """
        Initialize PDE solver with arbitrary parameters from a dictionary.
        
        Parameters
        ----------
        params : dict
            Dictionary of physical parameters of the pde, will be set as attributes.
        """
        for key, value in params.items():
            setattr(self, key, value)

    @staticmethod
    def h(phi):
        """ Interpolation function h(phi). """
        return -2 * phi**3 + 3 * phi**2

    @staticmethod
    def h_p(phi):
        """ First derivative of h(phi) wrt phi. """
        return -6 * phi**2 + 6 * phi

    @staticmethod
    def h_pp(phi):
        """ Second derivative of h(phi) wrt phi. """
        return -12.0 * phi + 6.0

    @staticmethod
    def g_p(phi):
        """ First derivative of double-well potential g(phi) wrt phi. """
        return 2 * phi * (1 - phi) * (2 * phi - 1)

    def phi_ic(self, x, t, x_init_interface=0.0):
        """ Initial condition for phi. """
        xd = x - x_init_interface
        scale = (self.omega_phi)**0.5 / (2.0 * self.alpha_phi)**0.5
        return 0.5 * (1.0 - jnp.tanh(scale * xd))

    def c_ic(self, x, t, x_init_interface=0.0):
        """ Initial condition for c. """
        phi = self.phi_ic(x, t, x_init_interface)
        return self.h(phi) * self.c_se

    def phi_bc(self, x, t):
        """ Boundary condition for phi. """
        return jnp.where(x < 0, 1.0, 0.0)

    def c_bc(self, x, t):
        """ Boundary condition for c. """
        return jnp.where(x < 0, 1.0, 0.0)
