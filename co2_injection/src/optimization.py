import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad

from .solver import run_simulation
from .config import SimConfig


def loss_function(trainable_params, cfg, strategy, static_params):
    all_params = {**static_params, **trainable_params}

    _, history = run_simulation(cfg, strategy, all_params)
    sat_history, p_history = history

    loss_saturation = -jnp.mean(sat_history[-100:])
    loss_energy = jnp.mean(p_history**2) * 0.1

    return loss_saturation + loss_energy, sat_history


def run_optimization(cfg: SimConfig, strategy: str):
    print(f"--- Optimizing {strategy} Strategy ---")

    if strategy == "freeform":
        params = {"schedule": jnp.zeros(cfg.STEPS)}
        optimizer = optax.adam(learning_rate=cfg.LEARNING_RATE)
    else:
        params = {"freq": 10.0}
        optimizer = optax.adam(learning_rate=0.1)

    opt_state = optimizer.init(params)
    static_params = {"amp": 0.1}

    @jax.jit
    def update_step(params, opt_state):
        (loss, _), grads = value_and_grad(loss_function, has_aux=True)(
            params, cfg, strategy, static_params
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for i in range(50):
        params, opt_state, loss = update_step(params, opt_state)
        if i % 10 == 0:
            print(f"Epoch {i}: Loss = {loss:.4f}")

    return params, loss
