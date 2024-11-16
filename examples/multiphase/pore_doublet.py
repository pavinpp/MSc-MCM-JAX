import os
import numpy as np
import jax.numpy as jnp

from src.lattice import LatticeD2Q9
from src.multiphase import Carnahan_Starling
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk

a = 1.0
b = 4.0
R = 1.0

rho_g = 0.000626568
rho_l = 0.454078426
Tc = 0.0943287031
T = 0.5 * Tc

offset = -10
channel_width = 20
nx = 500
ny = 120

# Using elliptical obstacle
a_inner = 85
b_inner = 35
a_outer = 100
b_outer = 55


class PoreDoublet2D(Carnahan_Starling):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        rho = rho_g * jnp.ones((nx, ny, 1))
        rho_tree.append(rho)
        u_tree = [
            np.zeros((self.nx, self.ny, 2)),
        ]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x, y = x.T, y.T

        ellipse_inner = ((x - self.nx / 2) / a_inner) ** 2 + (
            (y - self.ny / 2 + offset) / b_inner
        ) ** 2
        ellipse_outer = ((x - self.nx / 2) / a_outer) ** 2 + (
            (y - self.ny / 2) / b_outer
        ) ** 2
        ind = np.where((ellipse_inner > 1) & (ellipse_outer <= 1))
        pore = np.array(ind).T
        inlet_channel = np.array(
            [
                [x, y]
                for x in range(self.nx // 2 - a_outer + 5)
                for y in range(
                    self.ny // 2 - channel_width // 2,
                    self.ny // 2 + channel_width // 2 + 1,
                )
            ]
        )
        outlet_channel = np.array(
            [
                [x, y]
                for x in range(self.nx // 2 + a_outer - 5, self.nx)
                for y in range(
                    self.ny // 2 - channel_width // 2,
                    self.ny // 2 + channel_width // 2 + 1,
                )
            ]
        )
        fluid_nodes = np.concatenate((pore, inlet_channel, outlet_channel), axis=0)
        solid = np.ones((self.nx, self.ny))
        solid[fluid_nodes[:, 0], fluid_nodes[:, 1]] = 0
        walls = np.array(np.where(solid == 1))

        self.BCs[0].append(
            BounceBack(tuple(walls), self.gridInfo, self.precisionPolicy)
        )

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
            "solid_mask": np.array(self.solid_mask_streamed[..., 0]),
        }
        save_fields_vtk(
            timestep,
            fields,
            "output_",
            "data",
        )


precision = "f32/f32"
kwargs = {
    "n_components": 2,
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
    "precision": precision,
    "io_rate": 1,
    "print_info_rate": 1,
    "checkpoint_rate": -1,  # Disable checkpointing
    "checkpoint_dir": os.path.abspath("./checkpoints_"),
    "restore_checkpoint": False,
}
sim = PoreDoublet2D(**kwargs)
sim.run(0)
