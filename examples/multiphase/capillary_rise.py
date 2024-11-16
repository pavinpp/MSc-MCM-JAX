import os

import numpy as np
import jax.numpy as jnp

from src.lattice import LatticeD2Q9
from src.multiphase import Carnahan_Starling
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk

channel_width = 12
channel_height = 100
offset = 20
init_liq_height = 40

nx = 200
ny = 120

a = 1.0
b = 4.0
R = 1.0

rho_g = 0.000048042
rho_l = 0.503994922
Tc = 0.0943287031
T = 0.4 * Tc


class CapillaryRise2D(Carnahan_Starling):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        x, y = x.T, y.T

        r = 5000
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y + 0.996 * r) ** 2)
        width = 3

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (dist - r) / width
        )

        rho = rho.reshape((nx, ny, 1))

        rho_tree = []
        rho_tree.append(rho)
        u_tree = [np.zeros((self.nx, self.ny, 2))]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        left_wall = np.array(
            [
                [self.nx // 2 - channel_width // 2, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )
        right_wall = np.array(
            [
                [self.nx // 2 + channel_width // 2, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )

        walls = np.concatenate(
            (
                left_wall,
                right_wall,
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
            )
        )
        self.BCs[0].append(
            BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )

    def get_force(self):
        """
        Gravity force
        """
        return jnp.array(
            np.array([0.0, -1.2e-4]),
            dtype=self.precisionPolicy.compute_dtype,
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_prev_tree"][0])
        p = np.array(kwargs["p_tree"][0])
        u = np.array(kwargs["u_tree"][0])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(
            timestep,
            fields,
            "output_",
            "data",
        )


precision = "f32/f32"
kwargs = {
    "lattice": LatticeD2Q9(precision),
    "omega": [1.0],
    "nx": nx,
    "ny": ny,
    "nz": 0,
    "n_components": 1,
    "g_kkprime": -1.0 * np.ones((1, 1)),
    "g_ks": [-0.01],
    "a": a,
    "b": b,
    "R": R,
    "T": T,
    "k": 0.27,
    "A": -0.229,
    "precision": precision,
    "io_rate": 10000,
    "print_info_rate": 10000,
    "checkpoint_rate": 10000,
    "checkpoint_dir": os.path.abspath("./checkpoints_"),
    "restore_checkpoint": False,
}
sim = CapillaryRise2D(**kwargs)
sim.run(100000)
