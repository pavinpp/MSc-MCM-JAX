"""
Single component 2D capillary rise example where a channel is initially submerged in liquid. The density of each region is computed using Maxwell's Construction. The density profile is initialized with smooth profile with specified
interface thickness.
"""

import os

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.tree import map
from functools import partial

from src.lattice import LatticeD2Q9
from src.multiphase import MultiphaseBGK
from src.eos import Carnahan_Starling
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


class CapillaryRise2D(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (y - L) / width)
        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[:, :, 0] = rho_profile

        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)

        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        psi = lambda rho: 4 * (1 - jnp.exp(-(200 / rho)))
        psi_tree = map(psi, rho_tree)
        U_tree = map(lambda psi: jnp.zeros(psi), psi_tree)
        return psi_tree, U_tree

    def set_boundary_conditions(self):
        left_wall = np.array(
            [[self.nx // 2 - channel_width // 2, i + offset] for i in range(channel_height)],
            dtype=np.int32,
        )
        right_wall = np.array(
            [[self.nx // 2 + channel_width // 2, i + offset] for i in range(channel_height)],
            dtype=np.int32,
        )
        walls = np.concatenate((
            left_wall,
            right_wall,
            self.boundingBoxIndices["top"],
            self.boundingBoxIndices["bottom"],
        ))
        walls = tuple(walls.T)
        self.BCs[0].append(
            BounceBack(
                walls,
                self.gridInfo,
                self.precisionPolicy,
                theta[walls],
                phi[walls],
                delta_rho[walls],
            )
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_prev_tree"][0][0, :, 1:-1, :])
        p = np.array(kwargs["p_tree"][0][:, 1:-1, :])
        u = np.array(kwargs["u_tree"][0][0, :, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "flag": self.solid_mask_streamed[0][:, 1:-1, 0],
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        print(f"Max velocity: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        save_fields_vtk(
            timestep,
            fields,
            "output_",
            "data",
        )


if __name__ == "__main__":
    nx = 800
    ny = 500

    channel_width = 36
    channel_height = 600
    offset = 100  # Distance of channel bottom from domain bottom
    L = offset - 20  # 1/L of domain is filled with liquid, rest is vapor

    width = 5  # Initial Liquid-vapor interface thickness

    ratio = 6.11
    diff = 438.686
    rho_g = diff / (ratio - 1)
    rho_l = ratio * rho_g

    # Define contact angle matrix: I do not want any contact angle at the domain top and bottom, which are defined as walls.
    theta = 30 * (np.pi / 180) * np.ones((nx, ny, 1))
    theta[:, [0, ny - 1], 0] = 90 * (np.pi / 180)

    # Same goes for phi
    phi = 1.4 * np.ones((nx, ny, 1))
    phi[:, [0, ny - 1], 0] = 1.0

    # This is not used,
    delta_rho = np.zeros((nx, ny, 1))

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "omega": [1.0],
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "body_force": [0.0, -2e-8],
        "g_kkprime": -120.0 * np.ones((1, 1)),
        "k": [0.0],
        "A": 0.0 * np.ones((1, 1)),
        "kappa": [0.0],
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output* *.vtk")
    sim = CapillaryRise2D(**kwargs)
    sim.run(50000)
