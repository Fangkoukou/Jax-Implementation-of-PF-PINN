import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap, grad, config, lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves
# Ensure JAX is configured for 64-bit precision
config.update("jax_enable_x64", True)

class Residual:
    """
    Computes residuals for initial/boundary conditions and the pdekernel.
    This version treats physical parameters L and M as scalars.
    """
    def __init__(self, phys_span, norm_span, pdekernel, derivative):
        # Store spans for normalization/denormalization
        self.phys_span = phys_span
        self.norm_span = norm_span
        self.pdekernel = pdekernel
        self.deriv = derivative

    @staticmethod
    def _map_span(u, src, tgt):
        """Helper function to map a value from a source span to a target span."""
        a, b = src
        c, d = tgt
        return (u - a) / (b - a) * (d - c) + c

    def denorm(self, *pairs):
        """
        Denormalize multiple values.
        """
        out = []
        for u, key in pairs:
            out.append(self._map_span(u, self.norm_span[key], self.phys_span[key]))
        return tuple(out)

    def res_ic(self, model, inp):
        """Compute residual for initial conditions."""
        # prediction
        pred = model.predict(inp['x'],inp['t'])
        phi_pred, c_pred = pred['phi'], pred['c']
        
        # ground truth
        x_phys, t_phys = self.denorm((inp['x'],'x'),(inp['t'],'t'))
        phi = self.pdekernel.phi_ic(x_phys, t_phys)
        c = self.pdekernel.c_ic(x_phys, t_phys)

        # residual
        res = jnp.stack([phi_pred - phi, c_pred - c],axis=0).ravel()
        return {'ic': res}

    def res_bc(self, model, inp):
        """Compute residual for boundary conditions."""
        # prediction
        pred = model.predict(inp['x'], inp['t'])
        phi_pred, c_pred = pred['phi'], pred['c']
        
        # ground truth
        x_phys, t_phys = self.denorm((inp['x'],"x"),(inp['t'],"t"))
        phi = self.pdekernel.phi_bc(x_phys, t_phys)
        c = self.pdekernel.c_bc(x_phys, t_phys)

        # residual
        res = jnp.stack([phi_pred - phi, c_pred - c],axis=0).ravel()
        return {'bc': res}

    def res_phys(self, model, inp):
        """Compute residual for the physical PDE (Allen-Cahn & Cahn-Hilliard)."""
        # unpack parameters
        P = {k: getattr(self.pdekernel, k) for k in [
            'A','L','c_se','c_le','dc','omega_phi','alpha_phi','M'
        ]}

        # prediction
        pred = model.predict(inp['x'], inp['t'])
        phi_pred, c_pred = pred['phi'], pred['c']

        # derivatives
        derivs = self.deriv.evaluate(
            model, inp['x'], inp['t'],
            function_names=['phi_t', 'phi_x', 'phi_2x', 'c_t', 'c_2x']
        )
        h = self.pdekernel.h(phi_pred)
        h_p = self.pdekernel.h_p(phi_pred)
        h_xx = self.pdekernel.h_pp(phi_pred) * derivs['phi_x']**2 + h_p * derivs['phi_2x']
        g_p = self.pdekernel.g_p(phi_pred)

        # residuals
        res_phi = (
            derivs['phi_t']
            - 2 * P['A'] * P['L'] * (c_pred - h * P['dc'] - P['c_le']) * P['dc'] * h_p
            - P['L'] * P['omega_phi'] * g_p
            - P['L'] * P['alpha_phi'] * derivs['phi_2x']
        )
        res_c = (
            derivs['c_t']
            - 2 * P['A'] * P['M'] * derivs['c_2x']
            + 2 * P['A'] * P['M'] * P['dc'] * h_xx
        )

        return {'ac': res_phi.ravel(), 'ch': res_c.ravel()}

    def res_data(self, model, inp):
        # If the input dict is empty
        if not inp:
            return {'data': jnp.array(0.0)}
    
        # Flatten all leaves of the (possibly nested) dict
        leaves = jax.tree_util.tree_leaves(inp)
        
        # If all leaves are empty arrays, return 0
        if all(leaf.size == 0 for leaf in leaves):
            return {'data': jnp.array(0.0)}
    
        # Prediction
        pred = model.predict(inp['x'], inp['t'])
        phi_pred, c_pred = pred['phi'], pred['c']
    
        # Residual
        res = jnp.stack([phi_pred - inp['phi'], c_pred - inp['c']], axis=0).ravel()
        return {'data': res}

    def compute_loss(self, model, inp):
        """Compute total loss as MSE of all residual components."""
        residuals = {}
        residuals.update(self.res_ic(model, inp['ic']))
        residuals.update(self.res_bc(model, inp['bc']))
        residuals.update(self.res_phys(model, inp['colloc']))
        residuals.update(self.res_data(model, inp['data']))
        return {k: jnp.mean(v**2).astype(jnp.float32) for k, v in residuals.items()}

    def ntk_residual_wrappers(self):
        """Create NTK-compatible wrappers for each residual key."""
        def wrap(fn, key):
            def wrapped(flat_p, recon, static, inp):
                model = eqx.combine(recon(flat_p), static)
                return fn(model, inp)[key]
            return wrapped

        return {
            'ic': wrap(self.res_ic, 'ic'),
            'bc': wrap(self.res_bc, 'bc'),
            'ac': wrap(self.res_phys, 'ac'),
            'ch': wrap(self.res_phys, 'ch'),
            'data': wrap(self.res_data, 'data'),
        }

    def compute_ntk_weights(self, model, inp, batch_size=32):
        """Compute NTK-based weights for each loss component."""
        # unpack inputs
        eps = 1e-8
        params, static = eqx.partition(model, eqx.is_inexact_array)
        flat_p, recon = ravel_pytree(params)
        res_fns = self.ntk_residual_wrappers()
        inps = {
            'ic': inp['ic'],
            'bc': inp['bc'],
            'ac': inp['colloc'],
            'ch': inp['colloc'],
            'data': inp['data']
        }

        def batched_trace(fn, _inp):
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

        # compute ntk weight
        raw_weight = {}
        scale = 0
        for k, _inp in inps.items():
            total, n = batched_trace(res_fns[k], _inp)
            raw_weight[k] = n/(total + eps)
            scale += (total)/(n+eps)
        return {k: u * scale for k, u in raw_weight.items()}