import jax
import jax.numpy as jnp
from jax import random, config
from residual import *
config.update("jax_enable_x64", True)

class Sampler:
    """
    Sampler for generating training data for PINNs, including:
    - Initial condition (IC) points
    - Boundary condition (BC) points
    - Collocation points for residual loss
    - Adaptive residual points based on model residuals or gradients

    Supports Latin Hypercube Sampling (LHS) or noisy uniform sampling.
    """

    def __init__(self, x_span, t_span, sample_size, subsample_size, res_instance, noise=1.0):
        """
        Initialize the sampler.

        Parameters
        ----------
        x_span : tuple of float
            Normalized spatial domain, e.g., (-0.5, 0.5).
        t_span : tuple of float
            Normalized temporal domain, e.g., (0.0, 1.0).
        sample_size : dict
            Number of full samples per category: 'ic', 'bc', 'colloc_x', 'colloc_t', 'adapt'.
        subsample_size : dict
            Number of samples to use per batch per category: same keys as sample_size.
        res_instance : object
            Residual computation object with `get_noisy_points(model, x, t, k, criterion)` method.
        noise : float
            Noise amplitude for jittered uniform grid (0 ≤ noise ≤ 1).
            If noise < 0, Latin Hypercube Sampling is used instead.
        """
        self.x_span = x_span
        self.t_span = t_span
        self.sample_size = sample_size
        self.subsample_size = subsample_size
        self.noise = float(min(noise, 1.0))
        self.res = res_instance

    def _denormalize(self, x, span):
        """
        Denormalize from [0, 1] to given range.

        Parameters
        ----------
        x : float or jnp.ndarray
            Normalized input.
        span : tuple of float
            Domain bounds (min, max).

        Returns
        -------
        jnp.ndarray
            Value(s) rescaled to original range.
        """
        return x * jnp.float64(span[1] - span[0]) + jnp.float64(span[0])

    def _lhs(self, key, dim, num=1):
        """
        Latin Hypercube Sampling in [0, 1]^dim.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        dim : int
            Number of dimensions.
        num : int
            Number of samples.

        Returns
        -------
        tuple of jnp.ndarray or jnp.ndarray
            Sampled values in [0, 1]^dim.
        """
        keys = random.split(key, dim)
        samples = []
        for i in range(dim):
            perm = random.permutation(keys[i], jnp.arange(num, dtype=jnp.int32))
            u = random.uniform(keys[i], (num,), dtype=jnp.float64)
            samples.append((perm + u) / jnp.float64(num))
        return samples[0] if dim == 1 else tuple(samples)

    def _make_uniform_grid(self, key, num, span):
        """
        Generate 1D grid with uniform spacing and noise jitter.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        num : int
            Number of grid points.
        span : tuple of float
            Domain bounds (min, max).

        Returns
        -------
        jnp.ndarray
            Jittered uniform grid points within span.
        """
        dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
        base = jnp.linspace(span[0], span[1], num, endpoint=False, dtype=jnp.float64)
        shift = random.uniform(key, shape=(), minval=0.0, maxval=dx, dtype=jnp.float64)
        return jnp.clip(base + shift * self.noise, span[0], span[1])

    def _get_ic(self, key, local_span=(-0.1, 0.1)):
        """
        Sample initial condition points at t = t_min, spatially localized.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        local_span : tuple of float, optional
            Interval for dense center sampling. Default is (-0.1, 0.1).

        Returns
        -------
        x, t : jnp.ndarray
            Sampled spatial coordinates and corresponding time array,
            subject to self.x_span and self.t_span
        """
        num_x = self.sample_size['ic']
        x_left = self._make_uniform_grid(key, num_x // 3, (self.x_span[0], local_span[0]))
        x_local = self._make_uniform_grid(key, num_x // 3, local_span)
        x_right = self._make_uniform_grid(key, num_x // 3, (local_span[1], self.x_span[1]))
        x = jnp.sort(jnp.concatenate([x_left, x_local, x_right]))
        t = jnp.full_like(x, self.t_span[0], dtype=jnp.float64)
        return x, t

    def _get_bc(self, key):
        """
        Sample boundary condition points at domain edges across time.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.

        Returns
        -------
        x, t : jnp.ndarray
            Spatial and temporal coordinates of boundary points,
            subject to self.x_span and self.t_span
        """
        num_t = self.sample_size['bc'] // 2
        if self.noise < 0:
            t_single = self._lhs(key, 1, num_t).ravel()
        else:
            t_single = self._make_uniform_grid(key, num_t, self.t_span)

        x = jnp.array([self.x_span[0], self.x_span[1]], dtype=jnp.float64).repeat(num_t)
        t = jnp.concatenate([t_single, t_single])
        return x, t

    def _get_colloc(self, key):
        """
        Sample collocation points for residual enforcement.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.

        Returns
        -------
        x, t : jnp.ndarray
            Spatial and temporal coordinates of collocation points,
            subject to self.x_span and self.t_span
        """
        num_x = self.sample_size['colloc_x']
        num_t = self.sample_size['colloc_t']
        if self.noise < 0:
            x_norm, t_norm = self._lhs(key, 2, num_x * num_t)
            x = self._denormalize(x_norm, self.x_span)
            t = self._denormalize(t_norm, self.t_span)
        else:
            subkey1, subkey2 = random.split(key)
            x = self._make_uniform_grid(subkey1, num_x, self.x_span)
            t = self._make_uniform_grid(subkey2, num_t, self.t_span)
            X, T = jnp.meshgrid(x, t, indexing="ij")
            x = X.ravel()
            t = T.ravel()
        return x, t

    def _get_adapt(self, key, model, which_criterion="residual"):
        """
        Sample adaptive points based on model residual or gradient.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        model : Callable
            PINN model with signature (x, t) → (phi, c).
        which_criterion : str, optional
            Adaptivity criterion: 'residual' or 'gradient'.

        Returns
        -------
        x, t : jnp.ndarray
            Adaptively selected sample coordinates, subject 
            to self.x_span and self.t_span
        """
        x, t = self._lhs(key, 2, 5 * self.sample_size['adapt'])
        return self.res.get_noisy_points(model, x, t, self.sample_size['adapt'], which_criterion)

    def get_sample(self, key, model, which_criterion="residual"):
        """
        Generate a full training set with IC, BC, collocation, and adaptive points.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        model : Callable
            PINN model for adaptive sampling.
        which_criterion : str, optional
            Criterion for adaptivity ('residual' or 'gradient').

        Returns
        -------
        x, t : dict of jnp.ndarray
            Dictionaries with keys ['ic','bc','colloc','adapt'] containing
            normalized spatial and temporal coordinates subject to
            self.x_span and self.t_span.
        """
        key1, key2, key3, key4 = random.split(key, 4)
        x, t = {}, {}
        x['ic'], t['ic'] = self._get_ic(key1)
        x['bc'], t['bc'] = self._get_bc(key2)
        x['colloc'], t['colloc'] = self._get_colloc(key3)
        x['adapt'], t['adapt'] = self._get_adapt(key4, model, which_criterion)
        return x, t

    def get_subsample(self, key, x: dict, t: dict):
        """
        Randomly subsample from full dataset.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random seed.
        x, t : dict of jnp.ndarray
            Full sample coordinates returned from `get_sample`.

        Returns
        -------
        x_rand, t_rand : dict of jnp.ndarray
            Subsampled spatial and temporal points for each category.
        """
        keys = random.split(key, 4)
        x_rand, t_rand = {}, {}

        idx_ic = random.choice(keys[0], self.sample_size['ic'], shape=(self.subsample_size['ic'],), replace=False)
        x_rand['ic'] = x['ic'][idx_ic]
        t_rand['ic'] = t['ic'][idx_ic]

        idx_bc = random.choice(keys[1], self.sample_size['bc'] * 2, shape=(self.subsample_size['bc'],), replace=False)
        x_rand['bc'] = x['bc'][idx_bc]
        t_rand['bc'] = t['bc'][idx_bc]

        total_colloc = self.sample_size['colloc_x'] * self.sample_size['colloc_t']
        idx_colloc = random.choice(keys[2], total_colloc, shape=(self.subsample_size['colloc'],), replace=False)
        x_rand['colloc'] = x['colloc'][idx_colloc]
        t_rand['colloc'] = t['colloc'][idx_colloc]

        idx_adapt = random.choice(keys[3], self.sample_size['adapt'], shape=(self.subsample_size['adapt'],), replace=False)
        x_rand['adapt'] = x['adapt'][idx_adapt]
        t_rand['adapt'] = t['adapt'][idx_adapt]

        return x_rand, t_rand
