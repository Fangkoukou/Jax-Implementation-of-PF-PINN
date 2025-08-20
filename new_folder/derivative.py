import jax
import jax.numpy as jnp
import re
from jax import grad, vmap, config

config.update("jax_enable_x64", True)

class Derivative:
    """
    Generic derivative generator for JAX-based models, with dynamic batching based on input dimensions.

    Usage:
        derive = Derivative(
            inp_idx={'x': 0, 't': 1, 'L': 2, 'M': 3},
            out_idx={'phi': 0, 'c': 1},
            phys_span={'x': (0.0, 1.0), 't': (0.0, 1.0), 'L': (0.5, 2.0), 'M': (0.5, 1.5)},
            norm_span={'x': (0.0, 1.0), 't': (0.0, 1.0), 'L': (0.0, 1.0), 'M': (0.0, 1.0)},
        )
        derive.create_deriv_fn('phi_x2_t')
        dx = derive.phi_x2_t(model, x, t, L, M)
    """

    def __init__(self, inp_idx, out_idx, phys_span, norm_span):
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        self.phys_span = phys_span
        self.norm_span = norm_span
        self.norm_coefs = {
            name: self._get_coef(norm_span[name], phys_span[name])
            for name in inp_idx
        }

    def _get_coef(self, norm_range: tuple, phys_range: tuple) -> float:
        return (norm_range[1] - norm_range[0]) / (phys_range[1] - phys_range[0])

    def _parse_name(self, name_str: str):
        parts = name_str.split('_')
        out_name = parts[0]
        deriv_order = {}
        for token in parts[1:]:
            # Support formats like 'x2' or '2x'
            m = re.fullmatch(r"([a-zA-Z]+)(\d*)", token)
            if m and m.group(1):
                var, pow_str = m.groups()
            else:
                m2 = re.fullmatch(r"(\d+)([a-zA-Z]+)", token)
                if m2:
                    pow_str, var = m2.groups()
                else:
                    raise ValueError(f"Invalid token '{token}' in '{name_str}'")
            order = int(pow_str) if pow_str else 1
            deriv_order[var] = deriv_order.get(var, 0) + order
        return out_name, deriv_order

    def create_deriv_fn(self, name_str: str):
        """
        Generates and registers `self.<name_str>(model, *args)`, auto-batched based on input dims.
        """
        out_name, deriv_order = self._parse_name(name_str)

        # Base scalar function (one sample)
        def base_fn(model, *args):
            # model(*args) -> tuple or array with outputs
            out = model._forward(*args)
            return out[self.out_idx[out_name]]

        # Build gradient chain
        grad_fn = base_fn
        for var, order in deriv_order.items():
            argnum = self.inp_idx[var] + 1  # skip model arg
            for _ in range(order):  # repeated differentiation
                grad_fn = grad(grad_fn, argnums=argnum)

        # Scaling for normalization
        scale = jnp.prod(jnp.array([self.norm_coefs[v] ** o for v, o in deriv_order.items()]))

        def scalar_fn(model, *args):
            """Compute single-sample derivative."""
            return scale * grad_fn(model, *args)

        # Wrapped function that auto-vectorizes if inputs are arrays
        def deriv_fn(model, *args):
            # Convert to array and inspect dims
            arrs = [jnp.asarray(a) for a in args]
            # If all scalars, just compute directly
            if all(jnp.ndim(a) ==0 for a in arrs):
                return scalar_fn(model, *arrs)
            else:
                in_axes = (None,) + tuple(None if jnp.ndim(a) == 0 else 0 for a in arrs)
                return jax.vmap(scalar_fn, in_axes = in_axes)(model, *arrs)

        # Attach to instance
        setattr(self, name_str, deriv_fn)

    def evaluate(self, model, *args, function_names: list):
        """
        Compute multiple registered derivatives.
        Returns a dict mapping name -> array.
        """
        results = {}
        for name in function_names:
            fn = getattr(self, name, None)
            results[name] = fn(model, *args) if fn else jnp.array([])
        return results
