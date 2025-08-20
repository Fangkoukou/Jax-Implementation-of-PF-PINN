import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Tuple, Sequence, Union

Array = jnp.ndarray

class PINN(eqx.Module):
    """
    Generalized Physics-Informed Neural Network.

    Args:
        inp_idx: mapping from input names to positional indices
        out_idx: mapping from output names to indices in network output
        in_axes: tuple matching inp_idx keys for vmap axes (0 or None)
        width: hidden layer width
        depth: number of hidden layers
        key: PRNG key for initialization
    """
    net: eqx.nn.MLP
    inp_idx: Dict[str,int]
    out_idx: Dict[str,int]

    def __init__(self, inp_idx, out_idx, width, depth, key = None):
        key = jax.random.PRNGKey(0) if key is None else key
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        eqx_mlp_args = dict(
            in_size=len(inp_idx),
            out_size=len(out_idx),
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.net = eqx.nn.MLP(**eqx_mlp_args)

    def _forward(self, *args):
        x = jnp.stack(args, axis=0)
        return tuple(self.net(x))

    def __call__(self, *args):
        # cast everything into array
        arrs = tuple(jnp.asarray(a) for a in args)
        if all(jnp.ndim(a) == 0 for a in arrs):
            return self._forward(*arrs)
        else:
            in_axes = tuple(None if jnp.ndim(a) == 0 else 0 for a in arrs)
            return jax.vmap(self._forward, in_axes=in_axes)(*arrs)

    def predict(self, *args, names = None):
        """
        Returns outputs as a dict mapping names to arrays.
        If names is None, returns all out_idx keys.
        """
        outs = self(*args)
        keys = names or list(self.out_idx.keys())
        return {k: outs[self.out_idx[k]] for k in keys}