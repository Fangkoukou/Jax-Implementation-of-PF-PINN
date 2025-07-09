import jax
import jax.numpy as jnp
from jax import vmap, grad, config
config.update("jax_enable_x64", True)

class Derivative:
    """
    Computes on-demand partial derivatives of a neural network using JAX.

    This class is designed to compute partial derivatives with respect to 
    the inputs of a neural network using automatic differentiation. It assumes
    normalized inputs (x, t) and applies appropriate denormalization factors 
    when returning derivatives. The model is expected to take exactly two inputs 
    (x, t) and return exactly two outputs (phi, c).
    """

    def __init__(self, x_coef=1.0, t_coef=1.0):
        """
        Initialize Derivative instance.

        Parameters:
        -----------
        x_coef, t_coef : float
            Scaling coefficients for spatial and temporal inputs.
        """
        self.x_coef = jnp.float64(x_coef)
        self.t_coef = jnp.float64(t_coef)

    def set_coef(self, x_coef, t_coef):
        """
        Update scaling coefficients at runtime.

        Parameters:
        -----------
        x_coef, t_coef : float
            New scaling coefficients for spatial and temporal axes.
        """
        self.x_coef = jnp.float64(x_coef)
        self.t_coef = jnp.float64(t_coef)

    def _phi(self, model, x, t):
        """
        Extract the 'phi' component from the model output.

        Parameters:
        -----------
        model : callable
            Model that returns (phi, c) given (x, t).
        x, t : float
            Spatial and temporal coordinates.

        Returns:
        --------
        float
            Value of phi at (x, t).
        """
        return model(jnp.asarray(x), jnp.asarray(t))[0]

    def _c(self, model, x, t):
        """
        Extract the 'c' component from the model output.

        Parameters:
        -----------
        model : callable
            Model that returns (phi, c) given (x, t).
        x, t : float
            Spatial and temporal coordinates.

        Returns:
        --------
        float
            Value of c at (x, t).
        """
        return model(jnp.asarray(x), jnp.asarray(t))[1]

    def _dimension_compitable_return(self, fn, x, t):
        """
        Apply scalar function `fn` over input arrays via `vmap`.

        Parameters:
        -----------
        fn : callable
            Scalar-valued function of two arguments (x, t).
        x, t : float or jnp.ndarray
            Spatial and temporal inputs, scalar or batched.

        Returns:
        --------
        jnp.ndarray
            Output of `fn` applied over inputs with matching shape.
        """
        x, t = jnp.asarray(x), jnp.asarray(t)
        if x.ndim == 0 and t.ndim == 0:
            return fn(x, t)
        elif x.ndim == 0:
            return vmap(lambda ti: fn(x, ti))(t)
        elif t.ndim == 0:
            return vmap(lambda xi: fn(xi, t))(x)
        else:
            return vmap(fn)(x, t)

    def phi(self, model, x, t):
        """
        Evaluate phi(x, t).

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            phi values at input locations.
        """
        fn = lambda xi, ti: self._phi(model, xi, ti)
        return self._dimension_compitable_return(fn, x, t)

    def c(self, model, x, t):
        """
        Evaluate c(x, t).

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            c values at input locations.
        """
        fn = lambda xi, ti: self._c(model, xi, ti)
        return self._dimension_compitable_return(fn, x, t)

    def phi_t(self, model, x, t):
        """
        Compute d/dt(phi)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Time derivative of phi, scaled by t_coef.
        """
        fn = lambda xi, ti: grad(lambda tii: self._phi(model, xi, tii))(ti) * self.t_coef
        return self._dimension_compitable_return(fn, x, t)

    def phi_x(self, model, x, t):
        """
        Compute d/dx(phi)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Spatial derivative of phi, scaled by x_coef.
        """
        fn = lambda xi, ti: grad(lambda xii: self._phi(model, xii, ti))(xi) * self.x_coef
        return self._dimension_compitable_return(fn, x, t)

    def phi_2x(self, model, x, t):
        """
        Compute d^2/dx^2(phi)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Second spatial derivative of phi, scaled by x_coef².
        """
        fn = lambda xi, ti: grad(grad(lambda xii: self._phi(model, xii, ti)))(xi) * self.x_coef**2
        return self._dimension_compitable_return(fn, x, t)

    def c_t(self, model, x, t):
        """
        Compute d/dt(c)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Time derivative of c, scaled by t_coef.
        """
        fn = lambda xi, ti: grad(lambda tii: self._c(model, xi, tii))(ti) * self.t_coef
        return self._dimension_compitable_return(fn, x, t)

    def c_x(self, model, x, t):
        """
        Compute d/dx(c)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Spatial derivative of c, scaled by x_coef.
        """
        fn = lambda xi, ti: grad(lambda xii: self._c(model, xii, ti))(xi) * self.x_coef
        return self._dimension_compitable_return(fn, x, t)

    def c_2x(self, model, x, t):
        """
        Compute d^2/dx^2(c)

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.

        Returns:
        --------
        jnp.ndarray
            Second spatial derivative of c, scaled by x_coef².
        """
        fn = lambda xi, ti: grad(grad(lambda xii: self._c(model, xii, ti)))(xi) * self.x_coef**2
        return self._dimension_compitable_return(fn, x, t)

    def evaluate(self, model, x, t, function_names):
        """
        Evaluate a list of derivatives on given inputs.

        Parameters:
        -----------
        model : callable
            Neural network returning (phi, c).
        x, t : float or jnp.ndarray
            Spatial and temporal coordinates.
        function_names : List[str]
            List of derivative names to evaluate. Valid options include:
            'phi', 'phi_t', 'phi_x', 'phi_2x',
            'c', 'c_t', 'c_x', 'c_2x'.

        Returns:
        --------
        dict[str, jnp.ndarray]
            Dictionary mapping function names to their computed results.
            Invalid function names return an empty array.
        """
        results = {}
        for name in function_names:
            fn = getattr(self, name, None)
            if fn is None:
                results[name] = jnp.array([])
            else:
                results[name] = fn(model, x, t)
        return results
