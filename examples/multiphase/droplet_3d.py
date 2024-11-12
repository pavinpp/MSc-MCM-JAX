import os

import numpy as np

from src.lattice import LatticeD3Q19
from src.multiphase import (
    Carnahan_Starling,
    Peng_Robinson,
    Redlich_Kwong,
    Redlich_Kwong_Soave,
    VanderWaal,
)
from src.utils import save_fields_vtk


r = 30
nx = 100
ny = 100
nz = 100

a = 1.0
b = 4.0
R = 1.0

rho_g = 0.000048042
rho_l = 0.503994922
Tc = 0.0943287031
T = 0.4 * Tc


class Droplet3D(Carnahan_Starling):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z)

        rho_tree = []

        dist = np.sqrt(
            (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2
        )
        width = 5

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (dist - r) / width
        )

        rho = rho.reshape((self.nx, self.ny, self.nz, 1))

        rho_tree.append(rho)

        u_tree = [
            np.zeros((self.nx, self.ny, self.nz, 3)),
        ]
        return rho_tree, u_tree

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
            "uz": u[..., 2],
        }
        save_fields_vtk(
            timestep,
            fields,
            "output_",
            "data",
        )


precision = "f32/f32"
kwargs = {
    "lattice": LatticeD3Q19(precision),
    "omega": [1.0],
    "nx": nx,
    "ny": ny,
    "nz": nz,
    "g_kkprime": -1.0 * np.ones((1, 1)),
    "g_ks": [0.0],
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
sim = Droplet3D(**kwargs)
sim.run(30000)
