import jax.numpy as jnp
from jax import jit

W = jnp.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
CX = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])


@jit
def get_density(f):
    return jnp.sum(f, axis=-1)


@jit
def get_momentum(f, rho):
    ux = jnp.dot(f, CX) / rho
    uy = jnp.dot(f, CY) / rho
    return ux, uy


@jit
def get_equilibrium(rho, ux, uy):
    u_sq = ux**2 + uy**2
    cu = (ux[..., None] * CX) + (uy[..., None] * CY)
    return W * rho[..., None] * (1.0 + 3.0 * cu + 4.5 * (cu**2) - 1.5 * u_sq[..., None])


@jit
def collision_bgk(f, tau):
    rho = get_density(f)
    ux, uy = get_momentum(f, rho)
    f_eq = get_equilibrium(rho, ux, uy)
    return f - (f - f_eq) / tau


@jit
def stream(f):
    f_new = []
    for i in range(9):
        f_i = jnp.roll(f[..., i], shift=(CX[i], CY[i]), axis=(0, 1))
        f_new.append(f_i)
    return jnp.stack(f_new, axis=-1)
