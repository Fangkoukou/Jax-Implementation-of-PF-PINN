import jax
import jax.numpy as jnp
from jax import random, config
from residual import *
# config.update("jax_enable_x64", True)

class Sampler:
    """
    A general sampler
    """
    def _make_uniform_grid(self, key, num, span, noise = 0):
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
        return jnp.clip(base + shift * noise, span[0], span[1])

    def get_sample(self, key, nums, spans, noise=1.0):
        """
        Sample each dimension separately and concatenate.

        Args:
            key (jax.random.PRNGKey): RNG seed
            nums (list[int]): number of points per dimension
            spans (list[tuple]): (min, max) for each dimension
            noise (float): jitter amplitude

        Returns:
            jnp.ndarray: concatenated samples, shape (sum(nums),)
        """
        keys = random.split(key, len(nums))
        samples = []
        for k, num, span in zip(keys, nums, spans):
            arr = self._make_uniform_grid(k, num, span, noise)
            samples.append(arr)

        return jnp.concatenate(samples, axis=0)

    def subsample_inp(key, inp, sizes):
        """
        Subsample each top-level group in `inp`.
        - inp: dict like {'ic': {...}, 'bc': {...}, ...}, leaves are jnp arrays with same leading dim per group.
        - sizes: either {'ic': 10, 'bc': 20, ...} or {'ic': {'x':10,'t':30}, ...} with same structure as inp[group].
        Returns new dict with same nested structure and subsampled leaves.
        """
        out = {}
        top_keys = list(inp.keys())
        keys = random.split(key, len(top_keys))
        for k_top, group in zip(keys, top_keys):
            group_tree = inp[group]
            size_spec = sizes[group]
    
            # normalize to a tree with same structure as group_tree where each leaf is an int
            if isinstance(size_spec, int):
                size_tree = tree_util.tree_map(lambda _: int(size_spec), group_tree)
            else:
                size_tree = size_spec
    
            sizes_leaves, treedef = tree_util.tree_flatten(size_tree)
            arr_leaves, treedef2 = tree_util.tree_flatten(group_tree)
            if treedef != treedef2:
                raise ValueError(f"sizes[{group}] must match structure of inp[{group}]")
    
            # one subkey per leaf
            leaf_keys = random.split(k_top, len(sizes_leaves))
            new_leaves = []
            for subkey, s, arr in zip(leaf_keys, sizes_leaves, arr_leaves):
                s = int(s)
                m = arr.shape[0]
                if s > m:
                    raise ValueError(f"Requested {s} > available {m} for group {group}")
                idx = random.permutation(subkey, m)[:s]
                new_leaves.append(arr[idx])
    
            out[group] = treedef.unflatten(new_leaves)
        return out