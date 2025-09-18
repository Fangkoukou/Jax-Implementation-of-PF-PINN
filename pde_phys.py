import jax
import jax.numpy as jnp
from jax import config, random
import matplotlib.pyplot as plt
import diffrax as dfx

# JAX configuration
config.update("jax_enable_x64", True)

class PDE_phys:
    """
    Physical domain PDE solver for the coupled Cahn-Hilliard and Allen-Cahn equations.

    This class sets up and solves the system on a 1D spatial grid over time
    using the original physical parameters without non-dimensionalization.
    """
    def __init__(self, params):
        """
        Initialize the PDE solver with physical and discretization parameters.

        Args:
            params (dict): A dictionary containing all necessary physical parameters
                           (e.g., A, M, L, c_se, c_le, alpha_phi, omega_phi) and
                           simulation settings (x_range, t_range, nx, nt).
        """
        for key, value in params.items():
            setattr(self, key, value)
            
        self.x         = jnp.linspace(self.x_range[0], self.x_range[1], self.nx)
        self.t_eval    = jnp.linspace(self.t_range[0], self.t_range[1], self.nt)
        self.dx        = self.x[1] - self.x[0]

    def print_params(self, **overrides):
        """
        Print selected physical and dimensionless parameters of the PDE solver.
        """
        phys_params = [
            "alpha_phi", "omega_phi", "M", "L", "A",
            "c_se", "c_le", "x_range", "t_range", "l_0", "t_0"
        ]
        dimless_params = [
            "P_CH", "P_AC1", "P_AC2", "P_AC3",
            "x_range_nd", "t_range_nd"
        ]
    
        print("==== Physical parameters ====")
        for attr in phys_params:
            value = getattr(self, attr, None)
            print(f"{attr:<15} : {value}")
            
    # --- Auxiliary functions for the double-well potential ---
    @staticmethod
    def h(phi):
        """Interpolation function h(phi)."""
        return -2 * phi**3 + 3 * phi**2
    
    @staticmethod
    def h_p(phi):
        """First derivative of h(phi) with respect to phi."""
        return -6 * phi**2 + 6 * phi
    
    @staticmethod
    def h_pp(phi):
        """Second derivative of h(phi) with respect to phi."""
        return -12.0 * phi + 6.0
    
    @staticmethod
    def g_p(phi):
        """Derivative of the double-well potential g(phi)."""
        return 2 * phi * (1 - phi) * (2 * phi - 1)

    # --- Initial and Boundary Conditions (Physical) ---
    def phi_ic(self, x, t, **overrides):
        """Initial condition for the phase-field variable phi."""
        x_init = overrides.get("x_init",0.0)
        xd = x - x_init
        # The width of the interface is determined by the ratio of gradient to potential energy
        omega_phi = overrides.get("omega_phi",self.omega_phi)
        alpha_phi = overrides.get("alpha_phi",self.alpha_phi)
        K = jnp.sqrt(omega_phi / (2.0 * alpha_phi))
        return 0.5 * (1.0 - jnp.tanh(K * xd))

    def c_ic(self, x, t, **overrides):
        """Initial condition for the concentration c."""
        phi = self.phi_ic(x, t, **overrides)
        return self.h(phi) * self.c_se
                              
    def phi_bc(self, x, t):
        """Boundary condition for phi (1 on the left, 0 on the right)."""
        # Assumes the domain is centered around x=0
        return jnp.where(x <= self.x_range[0], 1.0, 0.0)

    def c_bc(self, x, t):
        """Boundary condition for c (c_se on the left, c_le on the right)."""
        # Assumes the domain is centered around x=0
        return jnp.where(x <= self.x_range[0], 1.0, 0.0)

    # --- Core Numerical Methods ---
    def laplacian(self, f):
        """
        Computes the 1D Laplacian of a function f on the grid
        using a second-order finite difference scheme.
        """
        lap = jnp.zeros_like(f)
        interior = (f[2:] - 2 * f[1:-1] + f[:-2]) / self.dx**2
        lap = lap.at[1:-1].set(interior)
        return lap

    def rhs(self, t, y, p):
        """
        The right-hand-side of the PHYSICAL PDE system for the ODE solver.
        """
        # Unpack physical parameters
        M, L, alpha_phi, omega_phi = p["M"], p["L"], p["alpha_phi"], p["omega_phi"]
        A, c_se, c_le = self.A, self.c_se, self.c_le
        c_diff = self.c_se - self.c_le

        # Split state vector into c and phi
        c, phi = jnp.split(y, 2)
        phi = phi.at[0].set(self.phi_bc(self.x[0],  t))
        phi = phi.at[-1].set(self.phi_bc(self.x[-1], t))
        c   = c.at[0].set(self.c_bc(self.x[0],      t))
        c   = c.at[-1].set(self.c_bc(self.x[-1],    t))
        h_phi     = self.h(phi)
        dh_dphi   = self.h_p(phi)
        lap_c     = self.laplacian(c)
        lap_phi   = self.laplacian(phi)
        lap_h_phi = self.laplacian(h_phi)
        dc_dt = 2 * A * M * (lap_c - (c_se - c_le) * lap_h_phi)
        reaction = 2 * A * (c_se - c_le) * (c - h_phi * (c_se - c_le) - c_le)
        dphi_dt = L * (alpha_phi * lap_phi + reaction * dh_dphi + omega_phi * self.g_p(phi))
        bc_idx = jnp.array([0, -1])
        dc_dt = dc_dt.at[bc_idx].set(0.0)
        dphi_dt = dphi_dt.at[bc_idx].set(0.0)
        return jnp.concatenate([dc_dt, dphi_dt])

    def solve(self, **overrides):
        """
        Solve the physical PDE system.

        Args:
            **overrides: Optional keyword arguments to override initial physical
                         parameters for this specific solve run (e.g., L=0.1).

        Returns:
            dict: A dictionary containing the physical solution arrays for
                  x, t, c, and phi.
        """
        p = dict(M=self.M, L=self.L, alpha_phi=self.alpha_phi, omega_phi=self.omega_phi)
        p.update(overrides)
        # Create and apply boundary conditions to initial conditions
        phi0 = self.phi_ic(self.x, 0.0, **overrides)
        phi0 = phi0.at[0].set(self.phi_bc(self.x[0],  0.0))
        phi0 = phi0.at[-1].set(self.phi_bc(self.x[-1], 0.0))
        
        c0 = self.c_ic(self.x, 0.0, **overrides)
        c0 = c0.at[0].set(self.c_bc(self.x[0], 0.0))
        c0 = c0.at[-1].set(self.c_bc(self.x[-1], 0.0))
        
        y0 = jnp.concatenate([c0, phi0])
        
        # --- Solve the system ---
        term   = dfx.ODETerm(self.rhs)
        solver = dfx.Kvaerno5() # A good implicit solver for stiff systems
        sol    = dfx.diffeqsolve(
            term, solver,
            t0=self.t_eval[0], t1=self.t_eval[-1], dt0=1e-5, # A small initial dt is often stable
            y0=y0, args=p,
            stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-9),
            saveat=dfx.SaveAt(ts=self.t_eval),
            max_steps=400000,
        )
        
        # Unpack results
        y_sol = sol.ys
        c_sol = y_sol[:, :self.nx]
        phi_sol = y_sol[:, self.nx:]
        
        return {
            "x": self.x,
            "t": self.t_eval,
            "phi": phi_sol,
            "c": c_sol
        }
