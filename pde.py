import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

class PDE:
    """
    Phase-field crystal model PDE parameters and helper functions.

    This class defines key physical constants and interpolation functions 
    used in phase-field models, including initial and boundary conditions 
    for `phi` and `c`.

    Notes
    -----
    - Inputs `x` and `t` are assumed to be in physical (not normalized) coordinates.
    - All parameters are cast to float64 for numerical precision.
    - Functions are written to support scalar or array inputs via JAX vectorization.
    """

    def __init__(self, **kwargs):
        """
        Initialize PDE parameters as 64-bit floats for precision.

        Parameters
        ----------
        alpha_phi : float
            Gradient energy coefficient for phi.
        omega_phi : float
            Double-well potential strength for phi.
        M : float
            Mobility coefficient.
        A : float
            Coupling constant (unused here but typical in CH/PFC models).
        L : float
            Kinetic coefficient.
        c_se : float
            Solid-phase equilibrium concentration.
        c_le : float
            Liquid-phase equilibrium concentration.
        """
        self.alpha_phi = jnp.float64(kwargs["alpha_phi"])
        self.omega_phi = jnp.float64(kwargs["omega_phi"])
        self.M = jnp.float64(kwargs["M"])
        self.A = jnp.float64(kwargs["A"])
        self.L = jnp.float64(kwargs["L"])
        self.c_se = jnp.float64(kwargs["c_se"])
        self.c_le = jnp.float64(kwargs["c_le"])

    @staticmethod
    def h(phi):
        """
        Smooth interpolation function h(phi).

        Represents a phase-field interpolation between solid and liquid phases.

        Parameters
        ----------
        phi : float or jnp.ndarray
            Phase-field variable.

        Returns
        -------
        jnp.ndarray
            Interpolated values: -2 * phi^3 + 3 * phi^2
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        return -2.0 * phi**3 + 3.0 * phi**2

    @staticmethod
    def h_p(phi):
        """
        First derivative of h(phi) with respect to phi.

        Parameters
        ----------
        phi : float or jnp.ndarray
            Phase-field variable.

        Returns
        -------
        jnp.ndarray
            Derivative: -6 * phi^2 + 6 * phi
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        return -6.0 * phi**2 + 6.0 * phi

    @staticmethod
    def h_pp(phi):
        """
        Second derivative of h(phi) with respect to phi.

        Parameters
        ----------
        phi : float or jnp.ndarray
            Phase-field variable.

        Returns
        -------
        jnp.ndarray
            Second derivative: -12 * phi + 6
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        return -12.0 * phi + 6.0

    @staticmethod
    def g_p(phi):
        """
        Derivative of phase-field coupling function g(phi).

        Parameters
        ----------
        phi : float or jnp.ndarray
            Phase-field variable.

        Returns
        -------
        jnp.ndarray
            g'(phi) = 2 * phi * (1 - phi)^2 - 2 * (1 - phi) * phi^2
        """
        phi = jnp.asarray(phi, dtype=jnp.float64)
        return 2.0 * phi * (1 - phi)**2 - 2.0 * (1 - phi) * phi**2

    def phi_ic(self, x, t):
        """
        Initial condition for phi at (x, t).

        Represents a tanh interface between solid and liquid.

        Parameters
        ----------
        x : float or jnp.ndarray
            Spatial coordinate(s).
        t : float or jnp.ndarray
            Time coordinate(s). (Unused, included for consistency)

        Returns
        -------
        jnp.ndarray
            Initial phi values.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        scale = jnp.sqrt(self.omega_phi) / jnp.sqrt(2.0 * self.alpha_phi)
        return 0.5 * (1.0 - jnp.tanh(scale * x))

    def c_ic(self, x, t):
        """
        Initial condition for concentration c at (x, t).

        Uses interpolated h(phi) scaled by solid equilibrium concentration.

        Parameters
        ----------
        x : float or jnp.ndarray
            Spatial coordinate(s).
        t : float or jnp.ndarray
            Time coordinate(s).

        Returns
        -------
        jnp.ndarray
            Initial concentration profile.
        """
        phi = self.phi_ic(x, t)
        return PDE.h(phi) * self.c_se

    def phi_bc(self, x, t):
        """
        Boundary condition for phi at (x, t).

        Defines solid (phi = 1) for x < 0 and liquid (phi = 0) for x ≥ 0.

        Parameters
        ----------
        x : float or jnp.ndarray
            Spatial coordinate(s).
        t : float or jnp.ndarray
            Time coordinate(s). (Unused)

        Returns
        -------
        jnp.ndarray
            Boundary phi values.
        """
        return jnp.where(x < 0, 1.0, 0.0)
    
    def c_bc(self, x, t):
        """
        Boundary condition for concentration c at (x, t).

        Uses h(phi_bc) scaled by solid equilibrium concentration.

        Parameters
        ----------
        x : float or jnp.ndarray
            Spatial coordinate(s).
        t : float or jnp.ndarray
            Time coordinate(s).

        Returns
        -------
        jnp.ndarray
            Boundary concentration values.
        """
        return self.h(self.phi_bc(x, t)) * self.c_se
