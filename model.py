import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

class PINN(eqx.Module):
    """
    Physics-Informed Neural Network mapping (x, t) → (phi, psi).
    
    Attributes:
        net: eqx.nn.MLP approximating the PDE solution.
    """
    net: eqx.nn.MLP

    def __init__(self, key=None, width=16, depth=4):
        """
        Args:
            key: PRNGKey for initialization (default: jax.random.PRNGKey(0)).
            width: Hidden layer width.
            depth: Number of hidden layers.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        self.net = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key
        )

    def _forward_scalar(self, x, t):
        """
        Forward pass for a single input.

        Args:
            x, t: Scalars.

        Returns:
            phi, c: Scalars.
        """
        out = self.net(jnp.array([x, t]))
        return out[0], out[1]

    def __call__(self, x, t):
        """
        Evaluate model at (x, t).

        Args:
            x, t: Scalars or arrays. Assume x,t are normalized

        Returns:
            phi, c: Scalars or arrays matching input shape.
        """
        x, t = jnp.asarray(x), jnp.asarray(t)
        if x.ndim == 0:
            return self._forward_scalar(x, t)
        return jax.vmap(self._forward_scalar)(x, t)

    def validation(self, x, t):
        X, T = jnp.meshgrid(x, t, indexing='xy')
        X_flat = X.ravel()
        T_flat = T.ravel()
    
        phi, c = self(X_flat, T_flat)
        P = phi.reshape(T.shape)
        C = c.reshape(T.shape)
        return P, C
