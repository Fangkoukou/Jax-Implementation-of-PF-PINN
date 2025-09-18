import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Tuple, List
from jax.experimental import io_callback
from util import tree_to_f32, to_f32


# ==============================================================
# Configuration container
# ==============================================================

@dataclass
class TrainConfig:
    """
    Configuration container for the training loop.

    Attributes:
        total_steps: Total number of training iterations to run.
        sp1: Frequency (in steps) for updating training samples.
        sp2: Frequency (in steps) for updating loss weights.
        sp3: Frequency (in steps) for printing logs.
        num_data: Number of data samples to use during training.
        P_model: Model parameter dictionary (dimensionless space).
        train_data: Dataset dictionary.
        static: Static model inputs (unchanged during training).
        optimizer: Optax optimizer object.
        update_input: Callable to refresh the training batch.
        update_weight: Callable to refresh loss weights.
        loss_fn: Callable computing loss and auxiliary info.
        log_keys_order: List of keys specifying print/log order.
    """
    total_steps: int = 0
    sp1: int = 0
    sp2: int = 0
    sp3: int = 0
    num_data: int = 0
    P_model: dict = field(default_factory=dict)
    train_data: dict = field(default_factory=dict)
    static: any = None
    optimizer: any = None
    update_input: callable = None
    update_weight: callable = None
    loss_fn: callable = None
    log_keys_order: List[str] = field(default_factory=list)


# ==============================================================
# Logging utilities
# ==============================================================

# Global state for timing
LOG_TIMES = []   # list of (step, total_elapsed, epoch_elapsed)
LOG_START = 0.0  # absolute start time
LOG_LAST = 0.0   # last log time


def _log_on_host(step, geometric_loss, loss_vals, weight_vals, *, keys_tuple):
    """
    Host-side function (non-JAX) that handles formatted logging.

    Args:
        step: Current training step.
        geometric_loss: Scalar geometric mean loss.
        loss_vals: Array of individual loss values.
        weight_vals: Array of corresponding weights.
        keys_tuple: Ordered tuple of keys matching the above arrays.

    Side-effects:
        - Prints formatted training status to console.
        - Updates global LOG_TIMES with elapsed timing.
    """
    global LOG_START, LOG_LAST, LOG_TIMES

    # 1. Timing
    now = time.time()
    total_elapsed = now - LOG_START
    epoch_elapsed = now - LOG_LAST
    LOG_TIMES.append((int(step), float(total_elapsed), float(epoch_elapsed)))
    LOG_LAST = now

    # 2. Loss reconstruction
    keys = list(keys_tuple)
    prod_vals = loss_vals * weight_vals
    total_loss = np.sum(prod_vals)

    # 3. Formatting
    col_width = 12
    formatted_keys = " | ".join([f"{k:^{col_width}}" for k in keys])

    def format_array_line(arr):
        return " | ".join([f"{x:<{col_width}.4e}" for x in arr])

    weights_line = format_array_line(weight_vals)
    losses_line = format_array_line(loss_vals)
    prod_line = format_array_line(prod_vals)

    # 4. Print output
    header = (
        f"Step {step:<6} total_loss = {total_loss:<10.4f}  "
        f"geom_loss = {geometric_loss:<10.4e}  "
        f"epoch_elapsed = {epoch_elapsed:.3f}s  "
        f"total_elapsed = {total_elapsed:.3f}s"
    )
    
    print("=" * len(header))
    print(header)
    print(
        f"Keys   : {formatted_keys}\n"
        f"Weights: {weights_line}\n"
        f"Losses : {losses_line}\n"
        f"W * L  : {prod_line}"
    )


# ==============================================================
# Training loop
# ==============================================================

class Train:
    """
    Stateless training loop utility.
    
    Provides methods for logging, performing a single training step,
    and executing a full training loop.
    """

    @staticmethod
    def log(cond, step, geometric_mean_loss, loss_dict, weight_dict, log_keys_order: List[str]):
        """
        Trigger host-side logging inside a JAX-traced computation.

        Args:
            cond: Boolean-like, determines whether logging should run.
            step: Current training step.
            geometric_mean_loss: Scalar geometric mean of all losses.
            loss_dict: Dict of individual losses {key: scalar}.
            weight_dict: Dict of weights {key: scalar}.
            log_keys_order: Order of keys for log formatting (static).
        """
        def true_fn(s, g_loss, l_dict, w_dict):
            ordered_keys = log_keys_order or list(l_dict.keys())
            loss_vals = jnp.array([l_dict[k] for k in ordered_keys])
            weight_vals = jnp.array([w_dict[k] for k in ordered_keys])

            io_callback(
                lambda step, geom, loss, weight: _log_on_host(step, geom, loss, weight, keys_tuple=tuple(ordered_keys)),
                None,  # no return value
                s,
                g_loss,
                loss_vals,
                weight_vals,
                ordered=True
            )
        
        def false_fn(s, g_loss, l_dict, w_dict):
            return None

        jax.lax.cond(cond, true_fn, false_fn, step, geometric_mean_loss, loss_dict, weight_dict)

    @staticmethod
    def train_step(carry, step, config):
        """
        Execute one training step.

        Args:
            carry: Tuple of training state:
                (key, sample, weight_dict, params, opt_state, best_loss, best_params).
            step: Current step index.
            config: TrainConfig object with all user-defined functions and parameters.

        Returns:
            new_carry: Updated training state tuple.
            loss_dict: Dict of individual loss components at this step.
        """
        key, sample, weight_dict, params, opt_state, best_loss, best_params = carry
        key1, key2, key3 = jax.random.split(key, 3)

        # 1. Update sample periodically
        sample = jax.lax.cond(
            step % config.sp1 == 0,
            lambda _: config.update_input(key1, config.P_model, config.train_data, config.num_data),
            lambda _: sample,
            operand=None,
        )

        # 2. Update weights periodically
        weight_dict = jax.lax.cond(
            step % config.sp2 == 0,
            lambda _: config.update_weight(key2, params, config.static, sample),
            lambda _: weight_dict,
            operand=None,
        )

        # 3. Compute total loss
        def total_loss(p):
            return config.loss_fn(p, config.static, sample, weight_dict)
            
        (loss, loss_dict), grads = jax.value_and_grad(total_loss, has_aux=True)(params)

        # 4. Optimizer step
        updates, opt_state = config.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # 5. Track best params by geometric mean of losses
        epsilon = 1e-12
        log_losses = jnp.log(jnp.asarray(list(loss_dict.values())) + epsilon)
        geometric_mean_loss = jnp.exp(jnp.mean(log_losses))
        best_loss, best_params = jax.lax.cond(
            geometric_mean_loss < best_loss,
            lambda _: (geometric_mean_loss, params),
            lambda _: (best_loss, best_params),
            operand=None
        )
        
        # 6. Logging
        Train.log(step % config.sp3 == 0, step, geometric_mean_loss, loss_dict, weight_dict, config.log_keys_order)

        # 7. Return updated state
        new_carry = (key3, sample, weight_dict, params, opt_state, best_loss, best_params)
        return new_carry, loss_dict

    @staticmethod
    def train(config, carry, total_steps=-1):
        """
        Execute a full training loop.

        Args:
            config: TrainConfig object with training setup.
            carry: Initial training state tuple.
            total_steps: Override for total training steps. If <= 0, use config.total_steps.

        Returns:
            final_carry: Final training state.
            loss_history: PyTree of loss_dicts over all steps.
        """
        global LOG_START, LOG_LAST
        total_steps = config.total_steps if total_steps <= 0 else total_steps

        step_fn = lambda carry, step: Train.train_step(carry, step, config)

        LOG_START = time.time()
        LOG_LAST = LOG_START

        start_time = time.time()
        final_carry, loss_history = jax.lax.scan(step_fn, carry, xs=jnp.arange(total_steps + 1))
        print(f"The training time is {(time.time() - start_time)/60:.2f} minutes")
        return final_carry, loss_history
