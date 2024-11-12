import os

import numpy as np
import jax.numpy as jnp

from src.lattice import LatticeD2Q9
from src.multiphase import Carnahan_Starling
from src.boundary_conditions import ZouHe, BounceBackHalfway
from src.utils import save_fields_vtk

channel_width = 24
channel_height = 70

nx = channel_width + 2
ny = 200

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

        rho_tree = []

        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[1 : self.nx - 2, 0 : channel_height // 2, :] = rho_l
        rho_tree.append(rho)

        u_tree = [np.zeros((self.nx, self.ny, 2))]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        left_wall = np.array([[0, i] for i in range(self.ny)], dtype=np.int32)
        right_wall = np.array(
            [[self.nx - 1, i] for i in range(self.ny)], dtype=np.int32
        )
        walls = np.concatenate((left_wall, right_wall))
        # apply bounce back boundary condition to the walls
        self.BCs.append(
            BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )

        bottom_capillary_inlet = np.array(
            [[i, 0] for i in range(1, self.nx - 1)], dtype=np.int32
        )

        pressure = (
            self.lattice.cs2 * rho_l * np.ones((np.shape(bottom_capillary_inlet)[0], 1))
        )
        self.BCs.append(
            ZouHe(
                tuple(bottom_capillary_inlet.T),
                self.gridInfo,
                self.precisionPolicy,
                "pressure",
                pressure,
            )
        )

    # def get_force(self):
    #     """
    #     Gravity force
    #     """
    #     return jnp.array(
    #         np.array([0.0, -1.102e-3]),
    #         dtype=self.precisionPolicy.compute_dtype,
    #     )

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][1:-1, 1:-1, :])
        p = np.array(kwargs["p_tree"][0][1:-1, 1:-1])
        u = np.array(kwargs["u_tree"][0][1:-1, 1:-1, :])
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
    "k": 0.07,
    "A": -0.28,
    "precision": precision,
    "io_rate": 10000,
    "print_info_rate": 10000,
    "checkpoint_rate": 30000,
    "checkpoint_dir": os.path.abspath("./checkpoints_"),
    "restore_checkpoint": False,
}
sim = CapillaryRise2D(**kwargs)
sim.run(30000)
