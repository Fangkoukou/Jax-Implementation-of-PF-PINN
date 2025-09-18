import jax
import jax.numpy as jnp
from jax import config, random, vmap
import matplotlib.pyplot as plt
import diffrax as dfx
from contextlib import nullcontext
# Local utility functions for subsampling and mapping spans
from util import subset, map_span

# Enable 64-bit precision for JAX
config.update("jax_enable_x64", True)

class PDE_dimless:
    """
    Dimensionless PDE solver for coupled Cahn-Hilliard (CH) and Allen-Cahn (AC) equations.

    This class:
      - Converts physical parameters into dimensionless form.
      - Defines initial and boundary conditions in dimensionless units.
      - Implements the dimensionless PDE system (rhs_nd) and Laplacian operator.
      - Solves the PDE system in 1D using diffrax ODE solver.
      - Provides utilities to convert dimensionless solutions back to physical units.
      - Supports subsampling for training data generation.

    Attributes:
        xs_nd (jnp.ndarray): 1D dimensionless spatial grid.
        ts_nd (jnp.ndarray): Dimensionless time evaluation points.
        dx_nd (float): Dimensionless spatial step size.
        c_bc_left_nd, c_bc_right_nd (float): Dimensionless boundary values for concentration.
        c_diff (float): Difference between solute concentrations at equilibrium.
        K_nd (float): Width parameter for initial φ profile in dimensionless units.
    """

    def __init__(self, params):
        """
        Initialize the PDE solver with physical parameters and dimensionless grids.

        Args:
            params (dict): Dictionary containing physical parameters:
                - alpha_phi, omega_phi, M, L, A, c_se, c_le, l_0, t_0
                - x_range: physical spatial domain (tuple)
                - t_range: physical time domain (tuple)
                - nx, nt: number of spatial and temporal points
        """
        # Store all parameters as attributes
        for key, value in params.items():
            setattr(self, key, value)

        # Compute dimensionless spatial and temporal ranges
        self.x_range_nd = (self.x_range[0]/self.l_0, self.x_range[1]/self.l_0)
        self.t_range_nd = (self.t_range[0]/self.t_0, self.t_range[1]/self.t_0)

        # Create 1D dimensionless spatial and temporal grids
        self.xs_nd = jnp.linspace(self.x_range_nd[0], self.x_range_nd[1], self.nx)
        self.ts_nd = jnp.linspace(self.t_range_nd[0], self.t_range_nd[1], self.nt)
        self.dx_nd = self.xs_nd[1] - self.xs_nd[0]

        # Compute concentration difference for scaling
        self.c_diff = self.c_se - self.c_le

        # Boundary values for concentration in dimensionless units
        self.c_bc_left_nd  = (1.0 - self.c_le) / self.c_diff
        self.c_bc_right_nd = (0.0 - self.c_le) / self.c_diff

    # ------------------------------------------------------------
    # Utilities for printing and extracting parameters
    # ------------------------------------------------------------
    def show(self, **overrides):
        """
        Print physical and dimensionless parameters side by side with 5 significant figures.

        Overrides can be provided to temporarily change values for display.
        """
        phys_params = [
            "alpha_phi", "omega_phi", "M", "L", "A",
            "c_se", "c_le", "x_range", "t_range", "l_0", "t_0"
        ]
        dimless_params = [
            "P_CH", "P_AC1", "P_AC2", "P_AC3",
            "x_range_nd", "t_range_nd"
        ]

        # Pad shorter list for alignment
        max_len = max(len(phys_params), len(dimless_params))
        phys_params += [""] * (max_len - len(phys_params))
        dimless_params += [""] * (max_len - len(dimless_params))

        # Header
        print(f"{'Physical Parameter':<20}{'Value':<25}|| {'Dimless Parameter':<20}{'Value':<25}")
        print("=" * 95)

        def fmt(val):
            """Format numeric or array-like values for display."""
            if isinstance(val, (int, float)):
                return f"{val:.5g}"
            if isinstance(val, (list, tuple)):
                return "[" + ", ".join(f"{v:.5g}" if isinstance(v, (int, float)) else str(v) for v in val) + "]"
            try:
                import jax.numpy as jnp
                import numpy as np
                if isinstance(val, (jnp.ndarray, np.ndarray)):
                    flat = val.flatten()
                    return "[" + ", ".join(f"{v:.5g}" for v in flat[:5]) + (" ...]" if flat.size > 5 else "]")
            except ImportError:
                pass
            return str(val)

        # Print each parameter pair
        for p_attr, d_attr in zip(phys_params, dimless_params):
            p_val = getattr(self, p_attr, "") if p_attr else ""
            d_val = getattr(self, d_attr, "") if d_attr else ""
            if callable(d_val):
                try:
                    d_val = d_val(**overrides)
                except TypeError:
                    d_val = "<function requires args>"
            print(f"{p_attr:<20}{fmt(p_val):<25}|| {d_attr:<20}{fmt(d_val):<25}")

    def dimless_params(self, **overrides):
        """
        Return dictionary of main dimensionless PDE coefficients: P_CH, P_AC1, P_AC2, P_AC3.
        """
        keys = ["P_CH", "P_AC1", "P_AC2", "P_AC3"]
        out = {}
        for k in keys:
            val = getattr(self, k, None)
            if callable(val):
                try:
                    val = val(**overrides)
                except TypeError:
                    val = None
            out[k] = val
        return out

    # ------------------------------------------------------------
    # Nonlinear free energy functions
    # ------------------------------------------------------------
    @staticmethod
    def h(phi_nd):
        """Nonlinear term for Allen-Cahn equation."""
        return -2 * phi_nd**3 + 3 * phi_nd**2

    @staticmethod
    def h_p(phi_nd):
        """Derivative of h(phi) w.r.t phi."""
        return -6 * phi_nd**2 + 6 * phi_nd

    @staticmethod
    def h_pp(phi_nd):
        """Second derivative of h(phi) w.r.t phi."""
        return -12.0 * phi_nd + 6.0

    @staticmethod
    def g_p(phi_nd):
        """Derivative of double-well potential function g(phi)."""
        return 2 * phi_nd * (1 - phi_nd) * (2 * phi_nd - 1)

    # ------------------------------------------------------------
    # Dimensionless PDE coefficients
    # ------------------------------------------------------------
    def P_CH(self, **overrides):
        """Dimensionless Cahn-Hilliard coefficient."""
        M = overrides.get('M', self.M)
        return (2 * self.A * M * self.t_0) / (self.l_0**2)

    def P_AC1(self, **overrides):
        """Coefficient multiplying (c - h(phi)) h'(phi) term in Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        return 2 * self.A * L * self.t_0 * self.c_diff**2

    def P_AC2(self, **overrides):
        """Coefficient multiplying g'(phi) term in Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        omega_phi = overrides.get('omega_phi', self.omega_phi)
        return L * omega_phi * self.t_0

    def P_AC3(self, **overrides):
        """Coefficient multiplying Laplacian(phi) term in Allen-Cahn equation."""
        L = overrides.get('L', self.L)
        alpha_phi = overrides.get('alpha_phi', self.alpha_phi)
        return (L * alpha_phi * self.t_0) / (self.l_0**2)

    def K_nd(self, **overrides):
        """Width parameter for initial phi profile in dimensionless units."""
        omega_phi = overrides.get('omega_phi', self.omega_phi)
        alpha_phi = overrides.get('alpha_phi', self.alpha_phi)
        return self.l_0 * jnp.sqrt(omega_phi / (2 * alpha_phi))

    # ------------------------------------------------------------
    # Initial and boundary conditions (dimensionless)
    # ------------------------------------------------------------
    def phi_ic_nd(self, xs_nd, ts_nd, **overrides):
        """Initial φ profile (tanh interface)."""
        x_init_nd = overrides.get("x_init", 0.0) / self.l_0
        xd_nd = xs_nd - x_init_nd
        K = self.K_nd(**overrides)
        return 0.5 * (1.0 - jnp.tanh(K * xd_nd))

    def c_ic_nd(self, xs_nd, ts_nd, **overrides):
        """Initial concentration field c(x,0) based on φ(x,0)."""
        phi_nd = self.phi_ic_nd(xs_nd, ts_nd, **overrides)
        h_val = self.h(phi_nd)
        return (self.c_se * h_val - self.c_le) / self.c_diff

    def phi_bc_nd(self, xs_nd, ts_nd):
        """Boundary condition for φ at domain edges."""
        return jnp.where(xs_nd <= self.x_range_nd[0], 1.0, 0.0)

    def c_bc_nd(self, xs_nd, ts_nd):
        """Boundary condition for c at domain edges."""
        return jnp.where(xs_nd <= self.x_range_nd[0], self.c_bc_left_nd, self.c_bc_right_nd)

    # ------------------------------------------------------------
    # Core numerical methods
    # ------------------------------------------------------------
    def laplacian_nd(self, f_nd):
        """Compute second derivative (Laplacian) for 1D dimensionless grid."""
        lap = jnp.zeros_like(f_nd)
        interior = (f_nd[2:] - 2 * f_nd[1:-1] + f_nd[:-2]) / self.dx_nd**2
        lap = lap.at[1:-1].set(interior)
        return lap

    def rhs_nd(self, ts_nd, y_nd, p):
        """
        Right-hand-side of dimensionless PDE system.

        Args:
            ts_nd: current time
            y_nd: concatenated state vector [c_nd, phi_nd]
            p: dictionary of dimensionless coefficients
        """
        # Unpack PDE coefficients
        P_CH, P_AC1, P_AC2, P_AC3 = p["P_CH"], p["P_AC1"], p["P_AC2"], p["P_AC3"]

        # Split state vector into c' and phi'
        c_nd, phi_nd = jnp.split(y_nd, 2)

        # Enforce Dirichlet boundary conditions
        phi_nd = phi_nd.at[0].set(self.phi_bc_nd(self.xs_nd[0], ts_nd))
        phi_nd = phi_nd.at[-1].set(self.phi_bc_nd(self.xs_nd[-1], ts_nd))
        c_nd   = c_nd.at[0].set(self.c_bc_nd(self.xs_nd[0], ts_nd))
        c_nd   = c_nd.at[-1].set(self.c_bc_nd(self.xs_nd[-1], ts_nd))

        # Compute nonlinear terms
        h_phi = self.h(phi_nd)
        dh_dphi = self.h_p(phi_nd)
        lap_c_nd = self.laplacian_nd(c_nd)
        lap_phi_nd = self.laplacian_nd(phi_nd)
        lap_h_phi = self.laplacian_nd(h_phi)

        # Dimensionless Cahn-Hilliard equation
        dc_dt_nd = P_CH * (lap_c_nd - lap_h_phi)

        # Dimensionless Allen-Cahn equation
        reaction_term = P_AC1 * (c_nd - h_phi) * dh_dphi
        potential_term = P_AC2 * self.g_p(phi_nd)
        gradient_term = P_AC3 * lap_phi_nd
        dphi_dt_nd = reaction_term + potential_term + gradient_term

        # Set boundary derivatives to zero
        bc_idx = jnp.array([0, -1])
        dc_dt_nd = dc_dt_nd.at[bc_idx].set(0.0)
        dphi_dt_nd = dphi_dt_nd.at[bc_idx].set(0.0)

        return jnp.concatenate([dc_dt_nd, dphi_dt_nd])

    # ------------------------------------------------------------
    # PDE solver
    # ------------------------------------------------------------
    def solve(self, force_cpu: bool = False, **overrides):
        """
        Solve the dimensionless PDE system using diffrax.

        Args:
            force_cpu: if True, force computation on CPU
            overrides: optional physical parameter overrides (e.g., L, x_init)
        Returns:
            dict: dimensionless solution {'x', 't', 'phi', 'c'}
        """
        # Compute dimensionless coefficients
        p = dict(
            P_CH=self.P_CH(**overrides),
            P_AC1=self.P_AC1(**overrides),
            P_AC2=self.P_AC2(**overrides),
            P_AC3=self.P_AC3(**overrides)
        )

        # Initial conditions
        phi0_nd = self.phi_ic_nd(self.xs_nd, 0.0, **overrides)
        c0_nd   = self.c_ic_nd(self.xs_nd, 0.0, **overrides)

        # Enforce boundary conditions
        phi0_nd = phi0_nd.at[0].set(self.phi_bc_nd(self.xs_nd[0], 0.0))
        phi0_nd = phi0_nd.at[-1].set(self.phi_bc_nd(self.xs_nd[-1], 0.0))
        c0_nd   = c0_nd.at[0].set(self.c_bc_nd(self.xs_nd[0], 0.0))
        c0_nd   = c0_nd.at[-1].set(self.c_bc_nd(self.xs_nd[-1], 0.0))

        # Concatenate state vector
        y0_nd = jnp.concatenate([c0_nd, phi0_nd])

        # ODE solver setup
        term   = dfx.ODETerm(self.rhs_nd)
        solver = dfx.Kvaerno5()
        context = jax.default_device(jax.devices("cpu")[0]) if force_cpu else nullcontext()

        with context:
            sol = dfx.diffeqsolve(
                term, solver,
                t0=self.ts_nd[0], t1=self.ts_nd[-1], dt0=1e-5,
                y0=y0_nd, args=p,
                stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-9),
                saveat=dfx.SaveAt(ts=self.ts_nd),
                max_steps=400000,
            )

        # Extract solution
        y_nd_sol = sol.ys
        c_nd_sol = y_nd_sol[:, :self.nx]
        phi_nd_sol = y_nd_sol[:, self.nx:]

        return {
            "x": self.xs_nd,
            "t": self.ts_nd,
            "phi": phi_nd_sol,
            "c": c_nd_sol
        }

    # ------------------------------------------------------------
    # Conversion to physical units
    # ------------------------------------------------------------
    def to_phys(self, result_nd):
        """
        Convert dimensionless solution back to physical units.

        Args:
            result_nd: dict with keys 'x', 't', 'phi', 'c'
        Returns:
            dict with physical units
        """
        x = map_span(result_nd['x'], self.x_range_nd, self.x_range)
        t = map_span(result_nd['t'], self.t_range_nd, self.t_range)
        phi = result_nd['phi']
        c = result_nd['c'] * self.c_diff + self.c_le
        return {"x": x, "t": t, "phi": phi, "c": c}

    # ------------------------------------------------------------
    # Training data generation (subsampling)
    # ------------------------------------------------------------
    def generate_training_data(self, key, params, num_train):
        """
        Generate PDE training data for multiple parameter sets.

        Args:
            key: JAX PRNG key
            params: dictionary of parameter sets
            num_train: number of points to subsample per simulation
        Returns:
            (solution, training_data) for each parameter set
        """
        num_simulations = list(params.values())[0].shape[0]
        keys = random.split(key, num_simulations)

        def generate_data_single(_key, p):
            # Solve PDE
            sol = self.solve(**p)
            xs_nd, ts_nd, phi_nd, c_nd = sol['x'], sol['t'], sol['phi'], sol['c']

            # Subsample
            xs_nd, ts_nd = jnp.meshgrid(xs_nd, ts_nd)
            sol_flat = (xs_nd.ravel(), ts_nd.ravel(), phi_nd.ravel(), c_nd.ravel())
            x_sub, t_sub, phi_sub, c_sub = subset(_key, sol_flat, num_train)

            train_data = {
                'x': x_sub,
                't': t_sub,
                'phi': phi_sub,
                'c': c_sub
            }
            return sol, train_data

        return vmap(generate_data_single)(keys, params)
