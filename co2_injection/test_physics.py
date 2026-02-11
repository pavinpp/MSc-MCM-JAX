import jax.numpy as jnp

from src.physics import lbm


def test_conservation():
    rho = jnp.ones((10, 10))
    u = jnp.zeros((10, 10))
    v = jnp.zeros((10, 10))

    f = lbm.get_equilibrium(rho, u, v)
    rho_check = lbm.get_density(f)
    print(f"Mean Density: {jnp.mean(rho_check)}")


if __name__ == "__main__":
    test_conservation()
