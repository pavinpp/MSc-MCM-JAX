import os

import numpy as np
import jax.numpy as jnp

from src.lattice import LatticeD2Q9
from src.multiphase import Carnahan_Starling
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


r = 30
nx = 200
ny = 200

a = 1.0
b = 4.0
R = 1.0

rho_g = 0.000048042
rho_l = 0.503994922
Tc = 0.0943287031
T = 0.4 * Tc


class DropletOnWall2D(Carnahan_Starling):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        dist = np.sqrt((x - 5) ** 2 + (y - self.ny / 2) ** 2)
        width = 5

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (dist - r) / width
        )

        rho = rho.reshape((nx, ny, 1))

        rho_tree.append(rho)

        u_tree = [
            np.zeros((self.nx, self.ny, 2)),
        ]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate(
            (self.boundingBoxIndices["bottom"], self.boundingBoxIndices["top"])
        )
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0])
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
    "g_kkprime": -1.0 * np.ones((1, 1)),
    "g_ks": [0.1],
    "a": a,
    "b": b,
    "R": R,
    "T": T,
    "k": 0.072,
    "A": -0.28,
    "precision": precision,
    "io_rate": 10000,
    "print_info_rate": 10000,
    "checkpoint_rate": -1,
    "checkpoint_dir": os.path.abspath("./checkpoints_"),
    "restore_checkpoint": False,
}
sim = DropletOnWall2D(**kwargs)
sim.run(30000)
