import jax
import jax.numpy as jnp
from jax import random, config
from variable import *
from residual import *
config.update("jax_enable_x64", True)

class Sampler:
    def __init__(self, x_span, t_span, size, size_rand, noise=1.0):
        self.x_span = tuple(map(jnp.float64, x_span))
        self.t_span = tuple(map(jnp.float64, t_span))
        self.size = size
        self.size_rand = size_rand
        self.noise = float(noise)

    def _denormalize(self, x, span):
        return x * jnp.float64(span[1] - span[0]) + jnp.float64(span[0])

    def _lhs(self, key, dim, num=1):
        keys = random.split(key, dim)
        samples = []
        for i in range(dim):
            perm = random.permutation(keys[i], jnp.arange(num, dtype=jnp.int32))
            u = random.uniform(keys[i], (num,), dtype=jnp.float64)
            samples.append((perm + u) / jnp.float64(num))
        return samples[0] if dim == 1 else tuple(samples)

    def _make_uniform_grid(self, key, num, span):
        """Create a uniform grid in the given span with a global shift."""
        dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
        base = jnp.linspace(span[0], span[1], num, endpoint = False, dtype=jnp.float64)
        shift = random.uniform(key, shape=(), minval=0.0, maxval=dx, dtype=jnp.float64)
        return jnp.clip(base + shift*self.noise, span[0], span[1])

    def _get_ic(self, key):
        num_x = self.size.ic
        subkey1, subkey2, subkey3 = random.split(key, 3)
    
        # Determine counts per region (you can adjust these ratios)
        num_local = num_x // 3
        num_left = num_x // 3
        num_right = num_x - num_local - num_left  # Ensure total adds up
    
        # Local center region: [-0.1, 0.1]
        xs_local = self._make_uniform_grid(subkey1, num_local, (-0.1, 0.1))
    
        # Left region: [x_min, -0.1)
        x_min, x_max = self.x_span
        left_upper = -0.1
        xs_left = self._make_uniform_grid(subkey2, num_left, (x_min, left_upper))
    
        # Right region: (0.1, x_max]
        right_lower = 0.1
        xs_right = self._make_uniform_grid(subkey3, num_right, (right_lower, x_max))
    
        # Combine and sort
        xs_all = jnp.sort(jnp.concatenate([xs_left, xs_local, xs_right]))
        ts_all = jnp.full(num_x, self.t_span[0], dtype=jnp.float64)
        return xs_all, ts_all

    def _get_bc(self, key):
        num_t = self.size.bc
        if self.noise < 0:
            t_single = self._lhs(key, 1, num_t).ravel()
            t_single = self._denormalize(t_single, self.t_span)
        else:
            t_single = self._make_uniform_grid(key, num_t, self.t_span)
    
        x_bc = jnp.concatenate([
            jnp.full(num_t, self.x_span[0], dtype=jnp.float64),
            jnp.full(num_t, self.x_span[1], dtype=jnp.float64)
        ])
        t_bc = jnp.concatenate([t_single, t_single])
        return x_bc, t_bc

    def _get_colloc(self, key):
        num_x, num_t = self.size.colloc_x, self.size.colloc_t
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
        """Adaptive sampling based on top-k residuals or gradients from a dense candidate grid."""
        x_norm, t_norm = self._lhs(key, dim=2, num=20 * self.size.adapt)
        # Use residual module to get top-k indices
        idx = Residual.get_criterion(model, x_norm, t_norm, self.size.adapt, which_criterion)
        return x_norm[idx], t_norm[idx]

    def get_sample(self, key, model, which_criterion="residual"):
        key1, key2, key3, key4 = random.split(key, 4)
        x = Variable()
        t = Variable()
        x.ic, t.ic = self._get_ic(key1)
        x.bc, t.bc = self._get_bc(key2)
        x.colloc, t.colloc = self._get_colloc(key3)
        x.adapt, t.adapt = self._get_adapt(key4, model, which_criterion)
        return x, t

    def get_subsample(self, key, x, t):
        keys = random.split(key, 4)
        x_rand = Variable()
        t_rand = Variable()

        idx_ic = random.choice(keys[0], self.size.ic, shape=(self.size_rand.ic,), replace=False)
        x_rand.ic, t_rand.ic = x.ic[idx_ic], t.ic[idx_ic]

        idx_bc = random.choice(keys[1], self.size.bc * 2, shape=(self.size_rand.bc,), replace=False)
        x_rand.bc, t_rand.bc = x.bc[idx_bc], t.bc[idx_bc]

        total_colloc = self.size.colloc_x * self.size.colloc_t
        idx_colloc = random.choice(keys[2], total_colloc, shape=(self.size_rand.colloc,), replace=False)
        x_rand.colloc, t_rand.colloc = x.colloc[idx_colloc], t.colloc[idx_colloc]

        idx_adapt = random.choice(keys[3], self.size.adapt, shape=(self.size_rand.adapt,), replace=False)
        x_rand.adapt, t_rand.adapt = x.adapt[idx_adapt], t.adapt[idx_adapt]

        return x_rand, t_rand