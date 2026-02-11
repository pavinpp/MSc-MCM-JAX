import jax.numpy as jnp
from jax import jit


@jit
def update_salt_advection_diffusion(salt, ux, uy, D_salt):
    ux_safe = jnp.clip(ux, -0.2, 0.2)
    uy_safe = jnp.clip(uy, -0.2, 0.2)

    grad_x = (jnp.roll(salt, -1, axis=0) - jnp.roll(salt, 1, axis=0)) / 2.0
    grad_y = (jnp.roll(salt, -1, axis=1) - jnp.roll(salt, 1, axis=1)) / 2.0

    laplacian = (
        jnp.roll(salt, -1, axis=0)
        + jnp.roll(salt, 1, axis=0)
        + jnp.roll(salt, -1, axis=1)
        + jnp.roll(salt, 1, axis=1)
        - 4.0 * salt
    )

    delta_salt = -(ux_safe * grad_x + uy_safe * grad_y) + (D_salt * laplacian)
    return salt + delta_salt


@jit
def check_precipitation(salt, mask, K_sp):
    excess_indices = (salt > K_sp) & (mask < 0.5)
    new_mask = jnp.where(excess_indices, 1.0, mask)
    new_salt = jnp.where(excess_indices, K_sp, salt)
    return new_salt, new_mask
