# ==============================
# Standard library imports
# ==============================
import os
import time
from functools import partial
import warnings

# ==============================
# Third-party imports: numerical and plotting
# ==============================
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from jax import config, random
from jax.tree_util import tree_leaves, tree_map
import diffrax as dfx
import equinox as eqx

# ==============================
# Visualization imports
# ==============================
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# ==============================================================
# Utility functions for value mapping and data manipulation
# ==============================================================

def map_span(u, src, tgt):
    """
    Linearly map a scalar or array from one span (interval) to another.

    Args:
        u: Scalar or array-like. The value(s) to map.
        src: Tuple (a, b). The source span.
        tgt: Tuple (c, d). The target span.

    Returns:
        Array-like object with values mapped from [a, b] to [c, d].
    """
    a, b = src
    c, d = tgt
    return (u - a) / (b - a) * (d - c) + c
        

def subset(key, pytree, n):
    """
    Select a random subset of elements along the leading dimension
    from every array in a pytree.

    Args:
        key: JAX PRNGKey used for randomness.
        pytree: Nested pytree (e.g., dict, list, or custom object) 
                whose leaves are array-like objects with a leading dimension.
        n: Number of elements to sample.

    Returns:
        A pytree of the same structure, where each array leaf has been subset
        along the first axis using the same random indices.
    """
    leaves = jax.tree_util.tree_leaves(pytree)

    if not leaves:
        return pytree

    # Ensure all leaves have consistent leading dimensions
    leading_dims = [leaf.shape[0] for leaf in leaves]
    m = leading_dims[0]

    if any(ld != m for ld in leading_dims):
        warnings.warn(
            f"Inconsistent leading dimensions in pytree leaves: {leading_dims}",
            UserWarning
        )

    idx = random.permutation(key, m)[:n]
    return jax.tree_util.tree_map(lambda leaf: leaf[idx], pytree)


def get_i(pytree, idx):
    """
    Select the element at index `idx` from each array leaf in the pytree.

    Args:
        pytree: Nested pytree whose leaves are array-like with leading dimension.
        idx: Integer index or slice to extract.

    Returns:
        A pytree of the same structure with each leaf indexed by `idx`.
    """
    return jax.tree_util.tree_map(lambda leaf: leaf[idx], pytree)


def get_len(pytree):
    """
    Get the length of the leading dimension across all leaves of a pytree.

    Args:
        pytree: Nested pytree whose leaves are array-like with a leading dimension.

    Returns:
        Integer length of the leading dimension, if consistent across all leaves.

    Raises:
        TypeError: If any leaf does not have a valid shape.
        ValueError: If leaves have inconsistent lengths.
    """
    leaves = tree_leaves(pytree)

    if not leaves:
        return 0

    try:
        lengths = {leaf.shape[0] for leaf in leaves}
    except (AttributeError, IndexError) as e:
        raise TypeError(
            "All leaves of the pytree must be array-like and have a non-empty shape."
        ) from e

    if len(lengths) > 1:
        raise ValueError(f"Pytree leaves have inconsistent lengths: {sorted(list(lengths))}")
        
    return lengths.pop()


# ==============================================================
# Model save/load utilities for Equinox
# ==============================================================

def save_model(model: eqx.Module, filename: str):
    """
    Save the trainable parameters (state) of an Equinox model to a file.

    This uses `eqx.tree_serialise_leaves`, which traverses the model
    as a pytree and saves only array leaves (e.g., JAX arrays).

    Args:
        model: The Equinox model instance to save.
        filename: Path to the output file. By convention, use `.eqx` extension.
    """
    eqx.tree_serialise_leaves(filename, model)
    print(f"Model saved successfully to '{filename}'")


def load_model(skeleton_model: eqx.Module, filename: str) -> eqx.Module:
    """
    Load the saved state into a skeleton Equinox model.

    Args:
        skeleton_model: An instance of the model with the correct architecture
                        and initialized parameters (can be random). This provides
                        the pytree structure into which parameters are loaded.
        filename: Path to the file from which to load model parameters.

    Returns:
        A new model instance with parameters replaced by the loaded state.
    """
    loaded_model = eqx.tree_deserialise_leaves(filename, skeleton_model)
    print(f"Model loaded successfully from '{filename}'")
    return loaded_model


# ==============================================================
# Validation data generation
# ==============================================================

def generate_validation_data(filename, num_params, pdekernel, span_model, span_pde, recompute=False):
    """
    Generate or load validation data for PDE-based simulations.

    Args:
        filename: Path where validation data is stored or will be saved.
        num_params: Number of parameter points to generate in dimensionless space.
        pdekernel: PDE kernel object with a `generate_training_data` method.
        span_model: Dict mapping parameter names to tuples (low, high)
                    in normalized [0, 1] model space.
        span_pde: Dict mapping parameter names to tuples (low, high)
                  in physical PDE parameter space.
        recompute: If True, regenerate data even if `filename` exists.

    Returns:
        P_validation_dimless: Dict of dimensionless validation parameters.
        sols: Generated or loaded solutions corresponding to those parameters.
    """
    P_validation_dimless = {
        'L': jnp.linspace(0, 1, num_params, dtype=jnp.float64)
    }
    
    # Case 1: Load precomputed data
    if os.path.exists(filename) and not recompute:
        print(f"Found existing validation data. Loading from '{filename}'...")
        with open(filename, 'rb') as f:
            sols = pickle.load(f)
        print("Loading complete.")
        return P_validation_dimless, sols

    # Case 2: Generate new data
    print(f"Validation data file '{filename}' not found.")
    print(f"Generating {num_params} new validation simulations...")

    start_time = time.time()

    # Map dimensionless parameters to physical parameters
    P_validation_phys = {
        key: map_span(value, span_model[key], span_pde[key])
        for key, value in P_validation_dimless.items()
    }

    # Generate solutions using PDE solver
    key_default = jax.random.PRNGKey(0)
    sols, _ = pdekernel.generate_training_data(
        key_default, P_validation_phys, num_train=1
    )

    print("Generation complete.")

    # Save results to file
    print(f"Saving new validation data to '{filename}'...")
    with open(filename, 'wb') as f:
        pickle.dump(sols, f)
    print("Save complete.")

    elapsed = time.time() - start_time
    print(f"Validation data generation took {elapsed:.2f} seconds.")

    return P_validation_dimless, sols


# ==============================================================
# Type conversion utilities
# ==============================================================

def to_f32(x):
    """
    Convert a floating-point array to float32 if applicable.

    Args:
        x: Input array-like or other object.

    Returns:
        x converted to float32 if it is a floating JAX array,
        otherwise returns the input unchanged.
    """
    return x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x


def tree_to_f32(tree):
    """
    Recursively convert all floating-point leaves in a pytree to float32.

    Args:
        tree: Pytree of arrays.

    Returns:
        New pytree with all floating-point arrays cast to float32.
    """
    return jax.tree_util.tree_map(to_f32, tree)
