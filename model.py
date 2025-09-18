import jax
import jax.numpy as jnp
import equinox as eqx


class PINN(eqx.Module):
    """
    Physics-Informed Neural Network (PINN) built with Equinox.
    
    This module wraps an MLP inside a structured model that:
      - Uses named indices for inputs and outputs.
      - Provides multiple evaluation modes (call, eval, predict).
      - Supports validation on space-time grids for PDEs.
    """

    net: eqx.nn.MLP    # The underlying neural network
    inp_idx: dict      # Mapping of input names -> index
    out_idx: dict      # Mapping of output names -> index

    def __init__(self, inp_idx, out_idx, width, depth, key=None):
        """
        Args:
            inp_idx (dict): Mapping of input names (e.g. {"x":0, "t":1, ...}).
            out_idx (dict): Mapping of output names (e.g. {"phi":0, "c":1}).
            width (int): Hidden layer width.
            depth (int): Number of hidden layers.
            key (jax.random.PRNGKey): Optional PRNG key for initialization.
        """
        key = jax.random.PRNGKey(0) if key is None else key
        self.inp_idx = inp_idx
        self.out_idx = out_idx
        # Define the MLP with tanh activation (common in PINNs)
        self.net = eqx.nn.MLP(
            in_size=len(inp_idx),
            out_size=len(out_idx),
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, *args):
        """
        Forward pass for ordered inputs.

        Args:
            *args: Inputs in the order defined by inp_idx.

        Returns:
            tuple: Raw outputs of the network.
        
        Notes:
            - Only the first len(inp_idx) args are used.
            - Inputs are stacked into a single feature vector.
        """
        n = len(self.inp_idx)
        args = args[:n]  # truncate if too many args given
        x = jnp.stack(args, axis=-1)
        return tuple(self.net(x))

    def eval(self, *args):
        """
        Forward pass for scalars or batched inputs.

        Args:
            *args: Inputs (scalars or arrays). Only first len(inp_idx) used.

        Returns:
            tuple: Outputs with broadcasting/vmap support.

        Notes:
            - If all inputs are scalars, behaves like __call__.
            - If some inputs are arrays, vectorized evaluation is applied.
        """
        n = len(self.inp_idx)
        arrs = tuple(jnp.asarray(a) for a in args[:n])
        # Case 1: all scalars
        if all(jnp.ndim(a) == 0 for a in arrs):
            return self.__call__(*arrs)
        # Case 2: at least one batched input
        in_axes = tuple(None if jnp.ndim(a) == 0 else 0 for a in arrs)
        return jax.vmap(self.__call__, in_axes=in_axes)(*arrs)

    def predict(self, inp, names=None):
        """
        Forward pass with named inputs and outputs.

        Args:
            inp (dict): Dictionary of named inputs.
            names (list, optional): List of output names to return.
                                    Defaults to all outputs.

        Returns:
            dict: Mapping of output names -> predictions.

        Notes:
            - Input dictionary is ordered using inp_idx.
            - Output dictionary uses out_idx mapping.
        """
        # order inputs according to inp_idx
        sorted_inp_names = sorted(self.inp_idx.keys(), key=self.inp_idx.get)
        ordered_args = [inp[name] for name in sorted_inp_names]

        outs = self.eval(*ordered_args)
        keys = names or list(self.out_idx.keys())
        return {k: outs[self.out_idx[k]] for k in keys}

    def validation(self, x, t, *scalars):
        """
        Evaluate the network on a space-time grid with extra scalar inputs.

        Args:
            x (1D array): Spatial coordinates.
            t (1D array): Temporal coordinates.
            *scalars: Any extra scalar parameters (broadcasted across grid).

        Returns:
            dict with fields:
                - "x": input x
                - "t": input t
                - "phi": predicted phi(x,t) over grid
                - "c": predicted c(x,t) over grid

        Notes:
            - Builds meshgrid from (x, t).
            - Flattens grid, evaluates network, reshapes outputs back to grid.
        """
        X, T = jnp.meshgrid(x, t, indexing="xy")
        flat_inp = [X.ravel(), T.ravel(), *scalars]
        outs = self.eval(*flat_inp)
        return {
            "x": x,
            "t": t,
            "phi": outs[0].reshape(X.shape),
            "c": outs[1].reshape(X.shape),
        }
