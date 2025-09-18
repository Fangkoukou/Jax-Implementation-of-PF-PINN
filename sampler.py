import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy.stats import qmc


class Sampler:
    """
    General sampler for PINNs.
    Provides:
      - Option A: per-point jitter
      - Option B: global jitter
      - sample_hypercube: uniform, Sobol, or Halton sampling
    """

    def _make_uniform_grid(self, num, span):
        """
        Construct a deterministic 1D grid within [span[0], span[1]].
        - num=0 → return empty
        - num=1 → return left endpoint
        - num>1 → return evenly spaced grid including both endpoints
        """
        if num == 0:
            return jnp.array([])
        if num == 1:
            return jnp.array([span[0]])
        return jnp.linspace(span[0], span[1], num, endpoint=True)

    def get_jitter(self, key, nums, spans, noise=1.0):
        """
        Option A: per-point jitter.
        Each grid point is perturbed independently within its cell width.
        """
        keys = random.split(key, len(nums))
        samples = []
        for k, num, span in zip(keys, nums, spans):
            base = self._make_uniform_grid(num, span)
            dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
            # Random offset for each grid point, up to dx
            jitter = random.uniform(k, base.shape, minval=0.0, maxval=dx)
            arr = jnp.clip(base + jitter * noise, span[0], span[1])
            samples.append(arr)
        return jnp.concatenate(samples, axis=0)

    def get(self, key, nums, spans, noise=1.0):
        """
        Option B: global jitter.
        All grid points along a span are shifted by the same random offset.
        """
        keys = random.split(key, len(nums))
        samples = []
        for k, num, span in zip(keys, nums, spans):
            base = self._make_uniform_grid(num, span)
            dx = (span[1] - span[0]) / (num - 1) if num > 1 else 0.0
            # Single random offset applied to all points
            shift = random.uniform(k, (), minval=0.0, maxval=dx)
            arr = jnp.clip(base + shift * noise, span[0], span[1])
            samples.append(arr)
        return jnp.concatenate(samples, axis=0)

    def sample_hypercube(self, key, ranges, n_samples=1, method="uniform"):
        """
        Sample points in a hypercube defined by 'ranges'.
        Supported methods:
          - uniform: JAX-native uniform sampling
          - sobol: low-discrepancy Sobol sequence (SciPy QMC)
          - halton: low-discrepancy Halton sequence (SciPy QMC)
        Returns: dict mapping parameter names → sampled values
        """
        param_names = list(ranges.keys())
        dim = len(param_names)

        lows = jnp.array([ranges[k][0] for k in param_names])
        highs = jnp.array([ranges[k][1] for k in param_names])

        if method == "uniform":
            # Generate uniform samples in [0,1]^dim using JAX
            unit_points = random.uniform(key, shape=(n_samples, dim))

        elif method in ["sobol", "halton"]:
            # Use SciPy's QMC sampler (not JIT-compatible)
            seed_key, _ = random.split(key)
            seed = random.randint(
                seed_key, shape=(), minval=0, maxval=jnp.iinfo(jnp.int32).max
            )
            if method == "sobol":
                sampler = qmc.Sobol(d=dim, scramble=True, seed=int(seed))
            else:  # halton
                sampler = qmc.Halton(d=dim, scramble=True, seed=int(seed))
            unit_points = sampler.random(n_samples)

        else:
            raise ValueError("Method must be one of 'uniform', 'sobol', or 'halton'")

        # Scale samples to the target ranges
        scaled_points = lows + unit_points * (highs - lows)

        # Return dictionary mapping parameter names to 1D arrays
        return {name: scaled_points[:, i] for i, name in enumerate(param_names)}
