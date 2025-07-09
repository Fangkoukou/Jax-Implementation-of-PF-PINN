import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap, grad, config, debug
from jax.flatten_util import ravel_pytree
config.update("jax_enable_x64", True)

class Residual:
    """
    Encapsulates computation of PDE residuals and related losses
    for physics-informed neural networks (PINNs) on a 2-component model.

    The residuals correspond to:
      - Initial conditions (IC)
      - Boundary conditions (BC)
      - Physics PDE residuals (split into two components: 'ac' and 'ch')

    Also supports:
      - Adaptive sampling based on residual or gradient magnitude
      - NTK-based weight computation for loss balancing
    """
    
    def __init__(self, x_coef, t_coef, pde, derivative):
        """
        Initialize Residual instance.

        Parameters:
        -----------
        x_coef, t_coef : float
            Normalizing coefficients for spatial and temporal inputs.
        pde : object
            PDE object providing IC, BC, and PDE terms (phi_ic, c_ic, phi_bc, c_bc, h, h_p, etc.)
        derivative : object
            Object to compute required derivatives of model outputs.
        """
        self.x_coef = jnp.float64(x_coef)
        self.t_coef = jnp.float64(t_coef)
        self.pde = pde
        self.deriv = derivative
        self.deriv.set_coef(x_coef, t_coef)

    def res_ic(self, model, x, t):
        """
        Compute initial condition residuals.

        Parameters:
        -----------
        model : callable
            PINN model predicting (phi, c) given (x, t).
        x, t : arrays

        Returns:
        --------
        dict with key 'ic' containing residual vector for initial condition enforcement.

        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        phi_pred, c_pred = model(x, t)
        phi_ic = self.pde.phi_ic(x / self.x_coef, t / self.t_coef)
        c_ic   = self.pde.c_ic(x / self.x_coef, t / self.t_coef)
        # Residuals: predicted - true initial condition values
        return {'ic': jnp.concatenate([phi_pred - phi_ic, c_pred - c_ic]).ravel()}

    def res_bc(self, model, x, t):
        """
        Compute boundary condition residuals.

        Parameters:
        -----------
        model : callable
            PINN model predicting (phi, c) given (x, t).
        x, t : arrays

        Returns:
        --------
        dict with key 'bc' containing residual vector for initial condition enforcement.
        
        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        phi_pred, c_pred = model(x, t)
        phi_bc = self.pde.phi_bc(x * self.x_coef, t * self.t_coef)
        c_bc   = self.pde.c_bc(x * self.x_coef, t * self.t_coef)
        return {'bc': jnp.concatenate([phi_pred - phi_bc, c_pred - c_bc]).ravel()}

    def res_phys(self, model, x, t):
        """
        Compute PDE residuals enforcing physics at collocation points.

        Residuals correspond to two equations (denoted 'ac' and 'ch') arising from the PDE system.

        Parameters:
        -----------
        model : callable
        x, t : jnp.arrays
        
        Returns:
        --------
        dict with keys 'ac' and 'ch' containing physics residual vectors.
        
        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        phi, c = model(x, t)

        # Evaluate required derivatives
        d = self.deriv.evaluate(model, x, t, ["phi_t", "phi_x", "phi_2x", "c_t", "c_x", "c_2x"])

        # Extract PDE parameters for compactness
        P = {k: getattr(self.pde, k) for k in ["A", "L", "c_se", "c_le", "omega_phi", "alpha_phi", "M"]}
        dc = P["c_se"] - P["c_le"]

        # Nonlinear functions and their derivatives of phi
        h   = self.pde.h(phi)
        h_p = self.pde.h_p(phi)
        h_xx = self.pde.h_pp(phi) * d["phi_x"]**2 + h_p * d["phi_2x"]
        g_p = self.pde.g_p(phi)

        # Residual for 'phi' equation (ac)
        res_phi = (
            d["phi_t"]
            - 2 * P["A"] * P["L"] * (c - h * dc - P["c_le"]) * dc * h_p
            + P["L"] * P["omega_phi"] * g_p
            - P["L"] * P["alpha_phi"] * d["phi_2x"]
        )

        # Residual for 'c' equation (ch)
        res_c = (
            d["c_t"]
            - 2 * P["A"] * P["M"] * d["c_2x"]
            + 2 * P["A"] * P["M"] * dc * h_xx
        )

        return {
            'ac': res_phi.ravel(),
            'ch': res_c.ravel()
        }

    def compute_loss(self, model, x, t):
        """
        Compute MSE losses for each residual type over sampled points.

        Parameters:
        -----------
        model : callable
        x, t : dict of jnp.array keyed by ['ic', 'bc', 'colloc', 'adapt']

        Returns:
        --------
        dict with MSE losses: {'ic', 'bc', 'ac', 'ch'}

        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        # Initial and boundary condition residual losses
        res_ic = self.res_ic(model, x['ic'], t['ic'])['ic']
        res_bc = self.res_bc(model, x['bc'], t['bc'])['bc']
        loss_ic = jnp.mean(res_ic**2)
        loss_bc = jnp.mean(res_bc**2)

        # Physics residual loss (concatenate collocation and adaptive points)
        x_phys = jnp.concatenate([x['colloc'], x['adapt']])
        t_phys = jnp.concatenate([t['colloc'], t['adapt']])
        phys = self.res_phys(model, x_phys, t_phys)
        loss_ac = jnp.mean(phys['ac']**2)
        loss_ch = jnp.mean(phys['ch']**2)

        return {
            'ic': loss_ic,
            'bc': loss_bc,
            'ac': loss_ac,
            'ch': loss_ch
        }

    def get_noisy_points(self, model, x, t, k, which_criterion="residual"):
        """
        Adaptive sampling: select points with largest residual or gradient magnitude.

        Parameters:
        -----------
        model : callable
        x, t : arrays of candidate points (noramlized)
        k : int
            Number of points to select
        which_criterion : str
            'residual' or 'gradient'

        Returns:
        --------
        x_top, t_top : arrays of selected points

        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        topk = lambda arr, num: jnp.argsort(-arr)[:num]

        k1 = k // 2
        k2 = k - k1

        if which_criterion == "residual":
            res = self.res_phys(model, x, t)
            idx1 = topk(jnp.abs(res['ac']), k1)
            idx2 = topk(jnp.abs(res['ch']), k2)

        elif which_criterion == "gradient":
            grads = self.deriv.evaluate(model, x, t, ["phi_t", "phi_x", "c_t", "c_x"])
            mag_dt = jnp.sqrt(grads["phi_t"]**2 + grads["c_t"]**2)
            mag_dx = jnp.sqrt(grads["phi_x"]**2 + grads["c_x"]**2)
            idx1 = topk(mag_dt, k1)
            idx2 = topk(mag_dx, k2)

        else:
            raise ValueError(f"Invalid criterion '{which_criterion}', must be 'residual' or 'gradient'.")

        x_top = jnp.concatenate([x[idx1], x[idx2]])
        t_top = jnp.concatenate([t[idx1], t[idx2]])
        return x_top, t_top

    def ntk_residual_wrappers(self):
        """
        Prepare wrapped residual functions for NTK weight computation.

        Returns:
        --------
        dict of functions keyed by residual type.
        Each function accepts (flat_params, recon, static, x, t)
        """
        def wrap(fn, key):
            def wrapped(fp, recon, static, x_, t_):
                model = eqx.combine(recon(fp), static)
                return fn(model, x_, t_)[key]
            return wrapped

        return {k: wrap(getattr(self, f"res_{k}") if k in ['ic','bc'] else self.res_phys, k)
                for k in ['ic', 'bc', 'ac', 'ch']}

    def compute_ntk_weights(self, model, x, t, batch_size=32):
        """
        Compute Neural Tangent Kernel (NTK) trace weights for residual terms.

        These weights can be used to balance losses dynamically based on gradient magnitudes.

        Parameters:
        -----------
        model : Equinox model
        x, t : dict of jnp.array keyed by ['ic', 'bc', 'colloc', 'adapt']
        batch_size : int

        Returns:
        --------
        dict of NTK weights keyed by residual type.

        Notes:
        --------
        Input x,t are assumed to have been normalized using self.x_coef, self.t_coef
        """
        eps = 1e-8
        # Separate trainable parameters from static parts
        params, static = eqx.partition(model, eqx.is_inexact_array)
        flat_p, recon = ravel_pytree(params)
        fns = self.ntk_residual_wrappers()

        # Prepare input splits per residual type
        x_phys = jnp.concatenate([x['colloc'], x['adapt']])
        t_phys = jnp.concatenate([t['colloc'], t['adapt']])
        splits = {
            'ic': (x['ic'],      t['ic']),
            'bc': (x['bc'],      t['bc']),
            'ac': (x_phys,       t_phys),
            'ch': (x_phys,       t_phys),
        }

        def trace_sq(J): 
            return jnp.sum(J**2)

        def batched_trace(fn, x_, t_):
            n = x_.shape[0]
            total = 0.0
            # Batch over input points to compute squared Jacobian norm traces
            for i in range(0, n, batch_size):
                xb, tb = x_[i:i+batch_size], t_[i:i+batch_size]
                def local(p, xi, ti): 
                    return fn(p, recon, static, xi[None], ti[None])
                def single(xi, ti): 
                    # Jacobian w.r.t. parameters for a single point
                    return trace_sq(jax.jacfwd(lambda p: local(p, xi, ti))(flat_p))
                total += jnp.sum(vmap(single)(xb, tb))
            return total, n

        results = {k: batched_trace(fns[k], *splits[k]) for k in splits}
        traces = {k: v[0] for k, v in results.items()}
        Ns     = {k: v[1] for k, v in results.items()}

        # Inverse trace weighting normalized by sample size
        inv    = {k: (traces[k]+eps)/Ns[k] for k in splits}
        unnorm = {k: Ns[k]/(traces[k]+eps) for k in splits}
        total_inv = sum(inv.values())

        # Return weights scaled to keep total sum balanced
        return {k: unnorm[k]*total_inv for k in splits}
