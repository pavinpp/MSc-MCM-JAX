import jax.numpy as jnp


def get_injection_pressure(step_idx, total_steps, strategy, params):
    t = step_idx / total_steps
    amp = jnp.asarray(params.get("amp", 0.0))
    freq = jnp.asarray(params.get("freq", 0.0))

    if strategy == "constant":
        return amp

    if strategy == "square":
        period = 1.0 / jnp.maximum(freq, 1e-6)
        is_on = (t % period) < (period * 0.5)
        return jnp.where(is_on, amp, 0.0)

    if strategy == "hammer":
        period = 1.0 / jnp.maximum(freq, 1e-6)
        cycle_t = (t % period) / period
        return amp * (1.0 - cycle_t)

    if strategy == "biomimetic":
        period = 1.0 / jnp.maximum(freq, 1e-6)
        cycle_t = (t % period) / period * 2.0 * jnp.pi
        base = jnp.sin(cycle_t)
        notch = 0.5 * jnp.sin(2.0 * cycle_t)
        wave = jnp.maximum(0.0, base + notch)
        return amp * wave

    if strategy == "freeform":
        schedule = jnp.asarray(params.get("schedule", jnp.zeros(total_steps)))
        raw_val = jnp.take(schedule, step_idx, mode="clip")
        safe_limit = jnp.asarray(params.get("safe_limit", 0.15))
        return safe_limit * jnp.tanh(raw_val)

    return 0.0
