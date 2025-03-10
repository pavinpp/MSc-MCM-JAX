"""
Analytical benchmark for a 2D channel.
"""

import os
import operator
import numpy as np

from src.lattice import LatticeD2Q9
from src.utils import save_fields_vtk
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack

from functools import partial
from jax import jit, vmap
from jax.tree import map, reduce
import jax.numpy as jnp


class CocurrentFlow(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        rho = rho_c * np.ones((self.nx, self.ny, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init(
            (self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u
        )
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)

        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate(
            (
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
            )
        )
        # apply bounce back boundary condition to the walls
        self.BCs[0].append(
            BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(
                operator.add, map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree)
            )

        return map(
            lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt,
            rho_tree,
            psi_tree,
            list(vmap(f, in_axes=(0,))(self.g_kkprime)),
        )

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho_total = np.array(kwargs.get("rho_total")[0, :, 1:-1, :])
        rho = np.array(kwargs["rho_tree"][0][0, :, 1:-1, :])
        p = np.array(kwargs["p"][0, 1:-1, 1:-1])
        u = np.array(kwargs["u_tree"][0][0, :, 1:-1, :])
        u_total = np.array(kwargs["u_total"][0, :, 1:-1, :])
        timestep = kwargs["timestep"]
        # fields = {
        #     "p": p[..., 0],
        #     "rho": rho[..., 0],
        #     "rho_total": rho_total[..., 0],
        #     "ux": u[..., 0],
        #     "uy": u[..., 1],
        #     "ux_total": u_total[..., 0],
        #     "uy_total": u_total[..., 1],
        # }
        # save_fields_vtk(
        #     timestep,
        #     fields,
        #     f"output_channel_{wetting_type}_visc_ratio_{visc_ratio}",
        #     "data",
        # )
        if timestep == 20000:
            Q = np.sum(u[self.nx // 2, :, 0])
            file.write(f"{wetting_type},{visc_ratio},{Q}\n")


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 150
    ny = 50

    rho_nw = 1.0  # Displaced fluid
    rho_w = 1.0  # Invading fluid

    rho_t = rho_w + rho_nw
    fraction = 0.97

    g_kkprime = 0 * np.ones((1, 1))
    fx = 1e-5

    Wetting = ["wetting", "non-wetting"]
    Rho = [rho_w, rho_nw]
    G_ks = [0.12, 0.0]

    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]

    os.system("rm -rf Flow*")
    # Simulate for a range of viscosity ratio and wetting conditions
    file = open("Flow_rate_data.csv", "w")
    file.write("Wetting_type,Viscosity Ratio,Flow Rate(LU)\n")
    for visc_ratio in [0.1, 1.0, 10]:
        tau_w = 5.0
        v_w = (tau_w - 0.5) / 3

        v_nw = v_w / visc_ratio
        tau_nw = 3 * v_nw + 0.5

        Tau = [tau_w, tau_nw]
        for i in range(2):
            tau = Tau[i]
            wetting_type = Wetting[i]
            rho_c = Rho[i]
            g_ks = G_ks[i]
            s_rho = [0.0]  # Mass
            s_e = [0.4]
            s_eta = [1.0]
            s_j = [0.0]  # Momentum
            s_q = [1.0]
            s_v = [1 / tau]
            kwargs = {
                "n_components": 1,
                "lattice": LatticeD2Q9(precision),
                "nx": nx,
                "ny": ny,
                "nz": 0,
                "g_kkprime": g_kkprime,
                "g_ks": [g_ks],
                "body_force": [fx, 0.0],
                "omega": [1 / tau],
                "precision": precision,
                "M": [M],
                "s_rho": s_rho,
                "s_e": s_e,
                "s_eta": s_eta,
                "s_j": s_j,
                "s_q": s_q,
                "s_v": s_v,
                "k": [0],
                "A": np.zeros((1, 1)),
                "kappa": [0.0],
                "io_rate": 20000,
                "compute_MLUPS": False,
                "print_info_rate": 20000,
                "checkpoint_rate": -1,
                "checkpoint_dir": os.path.abspath("./checkpoints_"),
                "restore_checkpoint": False,
            }
            sim = CocurrentFlow(**kwargs)
            sim.run(20000)
    file.close()
