import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, lax

from .config import SimConfig
from .physics import boundaries, lbm, transport
from . import injection


def init_simulation(cfg: SimConfig):
    key = jax.random.PRNGKey(0)
    noise = jax.random.uniform(key, (cfg.NX, cfg.NY))
    mask = jnp.where(noise > 0.7, 1.0, 0.0)

    mask = mask.at[0:5, :].set(0.0)
    mask = mask.at[cfg.NX - 5 : cfg.NX, :].set(0.0)

    rho = jnp.ones((cfg.NX, cfg.NY)) * cfg.RHO_BRINE
    ux = jnp.zeros((cfg.NX, cfg.NY))
    uy = jnp.zeros((cfg.NX, cfg.NY))

    f = lbm.get_equilibrium(rho, ux, uy)

    salt = jnp.ones((cfg.NX, cfg.NY))

    return f, salt, mask


@partial(jit, static_argnames=["cfg", "strategy"])
def run_simulation(cfg: SimConfig, strategy: str, params: dict):
    f_init, salt_init, mask_init = init_simulation(cfg)
    init_carry = (f_init, salt_init, mask_init)

    def step_fn(carry, step_idx):
        f, salt, mask = carry

        p_inlet_delta = injection.get_injection_pressure(step_idx, cfg.STEPS, strategy, params)

        rho_inlet = cfg.RHO_CO2 + p_inlet_delta
        f = boundaries.zou_he_inlet_pressure(f, rho_inlet, cfg.INLET_Y_START, cfg.INLET_Y_END)

        f_collided = lbm.collision_bgk(f, cfg.TAU_CO2)
        f_streamed = lbm.stream(f_collided)

        f = boundaries.apply_bounce_back(f_streamed, mask)

        rho = lbm.get_density(f)
        ux, uy = lbm.get_momentum(f, rho)

        salt = transport.update_salt_advection_diffusion(salt, ux, uy, cfg.D_SALT)
        salt = salt.at[0, cfg.INLET_Y_START : cfg.INLET_Y_END].set(0.0)
        salt, mask = transport.check_precipitation(salt, mask, cfg.K_SP)

        co2_saturation = jnp.mean(1.0 - salt)

        new_carry = (f, salt, mask)
        output_metrics = (co2_saturation, p_inlet_delta)
        return new_carry, output_metrics

    final_carry, history = lax.scan(step_fn, init_carry, jnp.arange(cfg.STEPS))
    return final_carry, history
