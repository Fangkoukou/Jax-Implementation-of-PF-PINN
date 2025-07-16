import jax
import jax.numpy as jnp
from jax import random, config
from residual import *
config.update("jax_enable_x64", True)

class Sampler:
    """
    Generates training samples for PINNs:
    - IC: initial condition points at t = t_min (1D spatial + scalar time)
    - BC: boundary condition points at spatial domain edges (1D spatial + 1D time)
    - Collocation: interior points in space-time domain (1D spatial + 1D time)
    - Adaptive: points selected by residual or gradient criteria
    
    Sampling modes:
    - jittered uniform grids (default)
    - Latin Hypercube Sampling if noise < 0
    """

    def __init__(self, x_span, t_span, sample_size, subsample_size, res_instance, noise=1.0):
        """
        Args:
            x_span (tuple[float, float]): spatial domain bounds (min, max)
            t_span (tuple[float, float]): temporal domain bounds (min, max)
            sample_size (dict): number of points per category ('ic', 'bc', 'colloc_x', 'colloc_t', 'adapt')
            subsample_size (dict): number of points to subsample per batch, same keys as sample_size
            res_instance (object): residual evaluator with method get_noisy_points(model, x, t, k, criterion)
            noise (float): jitter amplitude for uniform grid (0 to 1), negative for LHS
        """
        self.x_span = x_span
        self.t_span = t_span
        self.sample_size = sample_size
        self.subsample_size = subsample_size
        self.noise = min(noise, 1.0)
        self.res = res_instance

    def _denormalize(self, x, span):
        """
        Rescale normalized coords [0,1] to domain span.
        
        Args:
            x (jnp.ndarray): shape (...,), normalized values in [0,1]
            span (tuple): (min, max) domain interval
            
        Returns:
            jnp.ndarray: rescaled values in domain span, same shape as x
        """
        return x * (span[1] - span[0]) + span[0]

    def _lhs(self, key, dim, num=1):
        """
        Latin Hypercube Sampling in [0,1]^dim.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            dim (int): dimensionality of samples
            num (int): number of samples
            
        Returns:
            jnp.ndarray or tuple of jnp.ndarray: 
                If dim=1, returns (num,) array; 
                else tuple of dim arrays each shape (num,)
        """
        keys = random.split(key, dim)
        samples = []
        for i in range(dim):
            perm = random.permutation(keys[i], jnp.arange(num))
            u = random.uniform(keys[i], (num,))
            samples.append((perm + u) / num)
        return samples[0] if dim == 1 else tuple(samples)

    def _make_uniform_grid(self, key, num, span):
        """
        Uniform 1D grid with optional noise jitter.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            num (int): number of points
            span (tuple): (min, max) domain interval
            
        Returns:
            jnp.ndarray: shape (num,), points within span
        """
        dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
        base = jnp.linspace(span[0], span[1], num, endpoint=False)
        shift = random.uniform(key, (), minval=0.0, maxval=dx)
        return jnp.clip(base + shift * self.noise, span[0], span[1])

    def _get_ic(self, key, local_span=(-0.1, 0.1)):
        """
        Sample initial condition points at t = t_min, spatially localized.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            local_span (tuple): sub-interval in x_span for denser sampling
            
        Returns:
            x (jnp.ndarray): shape (N,), spatial coords within x_span
            t (jnp.ndarray): shape (N,), time coords all equal to t_min
        """
        num_x = self.sample_size['ic']
        x_left = self._make_uniform_grid(key, num_x // 3, (self.x_span[0], local_span[0]))
        x_local = self._make_uniform_grid(key, num_x // 3, local_span)
        x_right = self._make_uniform_grid(key, num_x // 3, (local_span[1], self.x_span[1]))
        x = jnp.sort(jnp.concatenate([x_left, x_local, x_right]))
        t = jnp.full_like(x, self.t_span[0])
        return x, t

    def _get_bc(self, key):
        """
        Sample boundary condition points at spatial domain edges over time.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            
        Returns:
            x (jnp.ndarray): shape (M,), spatial coords at domain boundaries repeated over time
            t (jnp.ndarray): shape (M,), time coords within t_span
        """
        num_t = self.sample_size['bc'] // 2
        if self.noise < 0:
            t_single = self._lhs(key, 1, num_t).ravel()
        else:
            t_single = self._make_uniform_grid(key, num_t, self.t_span)
        x = jnp.array([self.x_span[0], self.x_span[1]]).repeat(num_t)
        t = jnp.concatenate([t_single, t_single])
        return x, t

    def _get_colloc(self, key):
        """
        Sample collocation points inside the spatial-temporal domain.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            
        Returns:
            x (jnp.ndarray): shape (num_x*num_t,), spatial coords inside x_span
            t (jnp.ndarray): shape (num_x*num_t,), temporal coords inside t_span
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
        Sample adaptive points guided by residual or gradient.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            model (callable): PINN model with signature (x, t) → outputs
            which_criterion (str): 'residual' or 'gradient'
            
        Returns:
            x, t (jnp.ndarray): shape (k,), adaptively chosen coords in domain
        """
        x, t = self._lhs(key, 2, 5 * self.sample_size['adapt'])
        return self.res.get_noisy_points(model, x, t, self.sample_size['adapt'], which_criterion)

    def get_sample(self, key, model, which_criterion="residual"):
        """
        Generate full training set with all categories.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            model (callable): PINN model for adaptive sampling
            which_criterion (str): criterion for adaptivity
            
        Returns:
            x, t (dict): keys = ['ic','bc','colloc','adapt'], each jnp.ndarray arrays
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
        Randomly subsample points from full dataset for batching.
        
        Args:
            key (jax.random.PRNGKey): RNG seed
            x, t (dict): full samples dict, keys ['ic','bc','colloc','adapt'], each jnp.ndarray
            
        Returns:
            x_rand, t_rand (dict): subsampled points, same keys as input
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
