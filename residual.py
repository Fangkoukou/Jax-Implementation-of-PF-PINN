import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap, grad, config, lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves

# Note: enabling 64-bit precision can improve accuracy for PDE solvers
# but may reduce performance depending on hardware.
# config.update("jax_enable_x64", True)

class Residual:
    """
    Computes residuals for PINNs applied to Allen–Cahn (AC) and Cahn–Hilliard (CH).
    Includes:
      - Initial condition residual
      - Boundary condition residual
      - PDE residual (AC + CH)
      - Data residual
    Also provides NTK-based weighting for balancing losses.
    """

    def __init__(self, span_pde, span_model, pdekernel, derivative):
        # Store spans for mapping between PDE space and normalized model space
        self.span_pde = span_pde       # Physical spans (dimensionless PDE space)
        self.span_model = span_model   # Model input spans (normalized space)
        self.pdekernel = pdekernel     # PDE kernel with physics functions (e.g. h, g, P_AC1...)
        self.deriv = derivative        # Derivative evaluator (computes temporal/spatial derivatives)

    # --------------------- utility functions --------------------- #
    @staticmethod
    def _map_span(u, src, tgt):
        """Map a value u from a source interval [a,b] to a target interval [c,d]."""
        a, b = src
        c, d = tgt
        return (u - a) / (b - a) * (d - c) + c

    def denorm(self, D, keys_to_denorm=None):
        """
        Map inputs from normalized model space back to PDE space.

        Args:
            D (dict): dictionary of normalized values.
            keys_to_denorm (list, optional): subset of keys to denormalize.
        Returns:
            dict: dictionary with selected values denormalized.
        """
        out_dict = D.copy()
        keys_to_process = keys_to_denorm if keys_to_denorm is not None else D.keys()

        for key in keys_to_process:
            if key in self.span_pde and key in self.span_model and key in D:
                normalized_value = D[key]
                out_dict[key] = self._map_span(normalized_value,
                                               self.span_model[key],
                                               self.span_pde[key])
        return out_dict

    # --------------------- residual functions --------------------- #
    def res_ic(self, model, inp_model):
        """Residual for initial conditions (IC)."""
        # Model predictions
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        # Ground truth ICs in PDE space
        inp_pde = self.denorm(inp_model)
        phi = self.pdekernel.phi_ic_nd(inp_pde['x'], inp_pde['t'], **inp_pde)
        c = self.pdekernel.c_ic_nd(inp_pde['x'], inp_pde['t'], **inp_pde)

        # Residual = prediction - ground truth
        res = jnp.stack([phi_pred - phi, c_pred - c], axis=0).ravel()
        return {'ic': res}

    def res_bc(self, model, inp_model):
        """Residual for boundary conditions (BC)."""
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        inp_pde = self.denorm(inp_model)
        phi = self.pdekernel.phi_bc_nd(inp_pde['x'], inp_pde['t'])
        c = self.pdekernel.c_bc_nd(inp_pde['x'], inp_pde['t'])

        res = jnp.stack([phi_pred - phi, c_pred - c], axis=0).ravel()
        return {'bc': res}

    def res_pde(self, model, inp_model):
        """
        Residual for PDEs:
          - Allen–Cahn (AC)
          - Cahn–Hilliard (CH)
        """
        # Model predictions
        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        # Get derivatives (e.g. phi_t, phi_x, phi_xx, c_t, c_xx)
        sorted_inp_names = sorted(self.deriv.inp_idx.keys(), key=self.deriv.inp_idx.get)
        ordered_args = [inp_model[name] for name in sorted_inp_names]
        derivs = self.deriv.evaluate(
            model, *ordered_args,
            function_names=['phi_t', 'phi_x', 'phi_2x', 'c_t', 'c_2x']
        )

        # Nonlinear terms from PDE kernel
        h = self.pdekernel.h(phi_pred)
        h_p = self.pdekernel.h_p(phi_pred)
        h_xx = self.pdekernel.h_pp(phi_pred) * derivs['phi_x']**2 + h_p * derivs['phi_2x']
        g_p = self.pdekernel.g_p(phi_pred)

        inp_pde = self.denorm(inp_model)

        # Allen–Cahn residual
        res_phi = (
            derivs['phi_t']
            - self.pdekernel.P_AC1(**inp_pde) * (c_pred - h) * h_p
            - self.pdekernel.P_AC2(**inp_pde) * g_p
            - self.pdekernel.P_AC3(**inp_pde) * derivs['phi_2x']
        )

        # Cahn–Hilliard residual
        res_c = derivs['c_t'] - (derivs['c_2x'] - h_xx) * self.pdekernel.P_CH(**inp_pde)

        return {'ac': res_phi.ravel(), 'ch': res_c.ravel()}

    def res_data(self, model, inp_model):
        """Residual for supervised data (φ, c values)."""
        if not inp_model:   # Handle empty input dict
            return {'data': jnp.array(0.0)}

        pred = model.predict(inp_model)
        phi_pred, c_pred = pred['phi'], pred['c']

        res = jnp.stack([phi_pred - inp_model['phi'],
                         c_pred - inp_model['c']], axis=0).ravel()
        return {'data': res}

    # --------------------- loss computation --------------------- #
    def compute_loss(self, model, combined_inp):
        """
        Compute mean squared error (MSE) loss for all residual components.
        Combines IC, BC, PDE, and data residuals.
        """
        residuals = {}
        residuals.update(self.res_ic(model, combined_inp['ic']))
        residuals.update(self.res_bc(model, combined_inp['bc']))
        residuals.update(self.res_pde(model, combined_inp['colloc']))
        residuals.update(self.res_data(model, combined_inp['data']))
        return {k: jnp.mean(v**2) for k, v in residuals.items()}

    # --------------------- NTK utilities --------------------- #
    def ntk_residual_wrappers(self):
        """
        Build wrappers for each residual to compute NTK (Jacobian trace).
        Each wrapper takes flattened parameters, reconstructs the model, and
        outputs the residual vector for a specific component.
        """
        def wrap(fn, key):
            def wrapped(flat_p, recon, static, inp):
                model = eqx.combine(recon(flat_p), static)
                return fn(model, inp)[key]
            return wrapped

        return {
            'ic': wrap(self.res_ic, 'ic'),
            'bc': wrap(self.res_bc, 'bc'),
            'ac': wrap(self.res_pde, 'ac'),
            'ch': wrap(self.res_pde, 'ch'),
            'data': wrap(self.res_data, 'data'),
        }

    def compute_ntk_weights(self, model, inp, batch_size=64):
        """
        Compute NTK-based weights for each residual term.

        NTK weight for component k:
            w_k ∝ n_k / Tr(J_k J_k^T)
        where:
            n_k = number of samples
            J_k = Jacobian of residual wrt parameters

        Args:
            model: the PINN model
            inp: dict of training inputs (ic, bc, colloc, data)
            batch_size: minibatch size for large datasets
        """
        eps = 1e-8
        params, static = eqx.partition(model, eqx.is_inexact_array)
        flat_p, recon = ravel_pytree(params)
        res_fns = self.ntk_residual_wrappers()
        inps = {
            'ic': inp['ic'],
            'bc': inp['bc'],
            'ac': inp['colloc'],
            'ch': inp['colloc'],
            'data': inp['data'],
        }

        def batched_trace(fn, _inp):
            """
            Compute NTK trace for one residual component.
            Uses batching to avoid memory blowup.
            """
            leaves = tree_leaves(_inp)
            if not leaves: return 0.0, 0
            n = leaves[0].shape[0]
            if n == 0: return 0.0, 0

            def single_point_trace(__inp_slice):
                local = lambda p: fn(p, recon, static, __inp_slice)
                return jnp.sum(jax.jacrev(local)(flat_p) ** 2)

            batch_trace_fn = vmap(single_point_trace)

            if n <= batch_size:
                total = jnp.sum(batch_trace_fn(_inp))
                return total, n
            else:
                # Split into batches
                num_full_batches = n // batch_size
                n_processed = num_full_batches * batch_size
                truncated_inp = tree_map(lambda x: x[:n_processed], _inp)
                batched_inp = tree_map(
                    lambda x: x.reshape(num_full_batches, batch_size, *x.shape[1:]),
                    truncated_inp
                )

                def scan_body(accumulated_total, one_batch):
                    batch_total = jnp.sum(batch_trace_fn(one_batch))
                    return accumulated_total + batch_total, None

                final_total, _ = lax.scan(scan_body, 0.0, batched_inp)
                return final_total, n_processed

        # Compute raw NTK weights
        raw_weight = {}
        scale = 0
        for k, _inp in inps.items():
            total, n = batched_trace(res_fns[k], _inp)
            raw_weight[k] = n / (total + eps)
            scale += total / (n + eps)

        # Normalize weights by scale factor
        return {k: u * scale for k, u in raw_weight.items()}
