import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import jax
import jax.numpy as jnp
from jax import config, random
import matplotlib.pyplot as plt
import diffrax as dfx
from functools import partial

def map_span(u, src, tgt):
    """Helper function to map a value from a source span to a target span."""
    a, b = src
    c, d = tgt
    return (u - a) / (b - a) * (d - c) + c
        
def subset(key, n, *arrays):
    """
    Random subset without replacement across multiple arrays.

    Args:
        key (jax.random.PRNGKey): RNG seed
        n (int): number of elements to select
        *arrays (jnp.ndarray): arrays with same leading dimension

    Returns:
        tuple[jnp.ndarray]: arrays subset to size n
    """
    m = arrays[0].shape[0]
    if not all(arr.shape[0] == m for arr in arrays):
        raise ValueError("All arrays must have same length in axis 0")

    idx = random.permutation(key, m)[:n]
    return tuple(arr[idx] for arr in arrays)

        
def animate_2D_heatmaps(data, step=5, vmin=None, vmax=None,
                        label="Field", title="Heatmap"):

    data = np.array(data)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    total_frames = data.shape[0]
    frames_to_use = np.arange(0, total_frames, step)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(data[0], vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title(f"{label} (frame 0)")

    # Title text artist above plot
    text_artist = ax.text(
        0.5, 1.05,
        f"{title}",
        ha='center', va='bottom',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold'
    )

    def update(frame_idx):
        frame = frames_to_use[frame_idx]
        im.set_data(data[frame])
        ax.set_title(f"{label} (frame {frame})")
        text_artist.set_text(f"{title}")
        return im, text_artist

    anim = FuncAnimation(
        fig, update,
        frames=len(frames_to_use),
        interval=200,
        blit=True,
        repeat=False,
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())

def data_filter(*raw_arrays):
    """
    Filters out time frames where any of the input 3D arrays contain NaNs.

    Parameters:
    - raw_arrays: any number of 3D arrays of shape (T, H, W)

    Returns:
    - Tuple of filtered arrays, each with shape (T_valid, H, W)
    """
    import numpy as np

    # Convert to numpy and check shapes
    arrays = [np.array(arr) for arr in raw_arrays]
    if not all(arr.shape == arrays[0].shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape.")

    # Identify valid frames: no NaN in any array at any spatial location
    nan_mask = np.stack([np.isnan(arr) for arr in arrays], axis=0)  # shape: (N, T, H, W)
    valid_frames = ~np.any(nan_mask, axis=(0, 2, 3))  # shape: (T,)

    # Filter each array
    filtered = tuple(arr[valid_frames] for arr in arrays)
    return filtered
    
def plot_loss_dict_series(loss_dict, 
                          label="Run", 
                          step=10,
                          title="Loss Components Over Training", 
                          figsize=(14, 8)):
    import matplotlib.pyplot as plt
    import itertools

    plt.figure(figsize=figsize)

    keys = list(loss_dict.keys())
    # Use matplotlib color cycle for consistent colors
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = itertools.cycle(color_cycle)

    for k, color in zip(keys, colors):
        v = loss_dict[k][::step]
        x = range(0, len(loss_dict[k]), step)

        plt.plot(x, v, label=f"{label} - {k}", linestyle='-', color=color)

    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel("Loss Value (log scale)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()