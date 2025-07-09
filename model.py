import jax
import jax.numpy as jnp
import equinox as eqx

class PINN(eqx.Module):
    """
    Physics-Informed Neural Network (PINN) model.

    This class defines a neural network architecture that maps spatial and 
    temporal coordinates (x, t) to two output quantities (phi, psi), which 
    typically correspond to physical fields in a PDE system.
    
    Attributes:
    -----------
    net : eqx.nn.MLP
        Multi-layer perceptron used to approximate the solution (phi, psi).
    """

    net: eqx.nn.MLP

    def __init__(self, key=None, width=16, depth=4):
        """
        Initialize PINN instance.

        Parameters:
        -----------
        key : jax.random.PRNGKey or None
            Random key for network weight initialization. If None, uses a default key.
        width : int
            Number of hidden units in each hidden layer.
        depth : int
            Number of hidden layers in the MLP architecture.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        self.net = eqx.nn.MLP(
            in_size=2,               # Inputs: x and t
            out_size=2,              # Outputs: phi and psi
            width_size=width,       # Hidden layer width
            depth=depth,            # Number of hidden layers
            activation=jax.nn.tanh, # Nonlinearity used in each layer
            key=key
        )

    def _forward_single(self, x, t):
        """
        Forward pass for a single input coordinate (x, t).

        Parameters:
        -----------
        x : float
            Spatial coordinate.
        t : float
            Temporal coordinate.

        Returns:
        --------
        out_phi, out_psi : float
            Predicted values of phi and psi at the input point.
        """
        inp = jnp.stack([x, t], axis=-1)   # Shape (2,)
        out = self.net(inp)                # Shape (2,)
        out_phi = out[..., 0]
        out_psi = out[..., 1]
        return out_phi, out_psi

    def __call__(self, x, t):
        """
        Evaluate the network at input (x, t), supporting both scalar and batched inputs.

        Parameters:
        -----------
        x : float or array-like
            Spatial coordinate(s).
        t : float or array-like
            Temporal coordinate(s).

        Returns:
        --------
        out_phi, out_psi : float or array
            Predicted values of phi and psi. If inputs are batched, returns arrays.
        """
        x, t = jnp.asarray(x), jnp.asarray(t)
        if x.ndim == 0:
            return self._forward_single(x, t)  # Scalar case
        else:
            out_phi, out_psi = jax.vmap(self._forward_single)(x, t)  # Batched case
            return out_phi, out_psi
