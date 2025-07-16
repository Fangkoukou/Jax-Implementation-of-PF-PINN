import jax
import jax.numpy as jnp
from jax import grad, config
config.update("jax_enable_x64", True)


class Derivative:
    """
    Compute phi, c, and their derivatives from a neural network model.
    """

    def __init__(self, x_coef=1.0, t_coef=1.0):
        """
        Initialize the normalizing constant for the space and time
        """
        self.x_coef = x_coef
        self.t_coef = t_coef

    def set_coef(self, x_coef, t_coef):
        """ set the x_coef and t_coef """
        self.x_coef = x_coef
        self.t_coef = t_coef

    def _vectorize(self, scalar_fn):
        """
        Wrapper function of scalar function that maps (x,t) to a float.
        Uses jnp.frompyfunc to allow broadcasting.

        Returns:
            Callable accepting scalar or array-like x, t
        """
        ufunc = jnp.frompyfunc(scalar_fn, 2, 1)

        def wrapped(x, t):
            out = ufunc(x, t)
            return jnp.asarray(out, dtype=jnp.float64)

        return wrapped

    # ------------------ Scalar functions ------------------

    def _phi(self, model, x, t):
        return model(x, t)[0]

    def _c(self, model, x, t):
        return model(x, t)[1]

    # ------------------ Evaluation interface ------------------

    def phi(self, model, x, t):
        """ Evaluate phi(x, t). """
        fn = lambda x, t: self._phi(model, x, t)
        return self._vectorize(fn)(x, t)

    def c(self, model, x, t):
        """ Evaluate c(x, t). """
        fn = lambda x, t: self._c(model, x, t)
        return self._vectorize(fn)(x, t)

    def phi_t(self, model, x, t):
        """ Evaluate dphi/dt at (x,t) """
        fn = lambda x, t: grad(lambda t_: self._phi(model, x, t_))(t) * self.t_coef
        return self._vectorize(fn)(x, t)

    def phi_x(self, model, x, t):
        """ Evaluate dphi/dx at (x,t) """
        fn = lambda x, t: grad(lambda x_: self._phi(model, x_, t))(x) * self.x_coef
        return self._vectorize(fn)(x, t)

    def phi_2x(self, model, x, t):
        """ Evaluate d2phi/dx^2 at (x,t) """
        fn = lambda x, t: grad(grad(lambda x_: self._phi(model, x_, t)))(x) * self.x_coef**2
        return self._vectorize(fn)(x, t)

    def c_t(self, model, x, t):
        """ Evaluate dc/dt at (x,t) """
        fn = lambda x, t: grad(lambda t_: self._c(model, x, t_))(t) * self.t_coef
        return self._vectorize(fn)(x, t)

    def c_x(self, model, x, t):
        """ Evaluate dc/dx at (x,t) """
        fn = lambda x, t: grad(lambda x_: self._c(model, x_, t))(x) * self.x_coef
        return self._vectorize(fn)(x, t)

    def c_2x(self, model, x, t):
        """ Evaluate d2c/dx^2 at (x,t) """
        fn = lambda x, t: grad(grad(lambda x_: self._c(model, x_, t)))(x) * self.x_coef**2
        return self._vectorize(fn)(x, t)

    def evaluate(self, model, x, t, function_names):
        """
        Evaluate multiple quantities (phi, phi_x, c_2x, etc) at (x, t).

        Args:
            model: callable(x, t) -> (phi, c)
            x, t: scalar or array-like
            function_names: list of str

        Returns:
            dict[str, jnp.ndarray]
        """
        results = {}
        for name in function_names:
            fn = getattr(self, name, None)
            if fn is None:
                results[name] = jnp.array([])
            else:
                results[name] = fn(model, x, t)
        return results
