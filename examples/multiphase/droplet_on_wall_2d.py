import os
import numpy as np

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

rho_g = 0.000626568
rho_l = 0.454078426
Tc = 0.0943287031
T = 0.5 * Tc


class DropletOnWall2D(Carnahan_Starling):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        dist = np.sqrt((x - r) ** 2 + (y - self.ny / 2) ** 2)
        width = 3

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (dist - r) / width
        )
        # rho[:, 0] = 2.0 * rho_g

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
        rho = np.array(kwargs["rho_tree"][0][:, 2:-2, :])
        p = np.array(kwargs["p_tree"][0][:, 2:-2, :])
        u = np.array(kwargs["u_tree"][0][:, 2:-2, :])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        u_sp = np.sqrt(np.sum(np.square(u), axis=-1))
        rho_g_pred = np.min(rho)
        rho_l_pred = np.max(rho)
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Spurious velocity Max: {np.max(u_sp)}")
        print(
            f"% Error: {(rho_g_pred - rho_g)*100 /rho_g} Max: {(rho_l_pred - rho_l)*100/rho_l}"
        )
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
    "g_ks": [0.15],
    "a": a,
    "b": b,
    "R": R,
    "T": T,
    "k": 0.035,
    "A": -0.32,
    # "k": 0.1,
    # "A": -0.31,
    "precision": precision,
    "io_rate": 40000,
    "print_info_rate": 40000,
    "checkpoint_rate": -1,  # Disable checkpointing
    "checkpoint_dir": os.path.abspath("./checkpoints_"),
    "restore_checkpoint": False,
}
sim = DropletOnWall2D(**kwargs)
sim.run(280000)
