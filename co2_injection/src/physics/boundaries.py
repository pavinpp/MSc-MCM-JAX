import jax.numpy as jnp
from jax import jit
from functools import partial

from .lbm import get_density, get_momentum, get_equilibrium, W


@jit
def apply_bounce_back(f, mask):
    idx_bounce = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    f_bounced = f[..., idx_bounce]
    mask_exp = mask[..., None]
    return jnp.where(mask_exp, f_bounced, f)


@partial(jit, static_argnames=["y_start", "y_end"])
def zou_he_inlet_pressure(f, rho_inlet, y_start, y_end):
    f_slice = f[0, y_start:y_end, :]
    rho = rho_inlet
    u_y = 0.0

    col_sum = f_slice[..., 0] + f_slice[..., 2] + f_slice[..., 4]
    west_sum = f_slice[..., 3] + f_slice[..., 6] + f_slice[..., 7]
    u_x = 1.0 - (col_sum + 2.0 * west_sum) / rho

    f1 = f_slice[..., 3] + (2.0 / 3.0) * rho * u_x
    f5 = f_slice[..., 7] - 0.5 * (f_slice[..., 2] - f_slice[..., 4]) + (1.0 / 6.0) * rho * u_x
    f8 = f_slice[..., 6] + 0.5 * (f_slice[..., 2] - f_slice[..., 4]) + (1.0 / 6.0) * rho * u_x

    f_new = f
    f_new = f_new.at[0, y_start:y_end, 1].set(f1)
    f_new = f_new.at[0, y_start:y_end, 5].set(f5)
    f_new = f_new.at[0, y_start:y_end, 8].set(f8)
    return f_new
