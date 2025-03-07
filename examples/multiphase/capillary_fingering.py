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


nx = 500
ny = 75

tau_2 = 1.9
v_2 = (tau_2 - 0.5) / 3

visc_ratio = 10.0  # viscosity ratio
v_1 = v_2 / visc_ratio
tau_1 = 3 * v_1 + 0.5

rho_2 = 1.0  # Displaced fluid
rho_1 = 1.0  # Invading fluid

rho_t = rho_1 + rho_2

fraction = 0.97


class CapillaryFingering(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        # Invading fluid
        rho = (1 - fraction) * rho_t * np.ones((self.nx, self.ny, 1))
        rho[0:Lx, :, :] = fraction * rho_t
        rho = self.distributed_array_init(
            (self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # Displaced fluid
        rho = fraction * rho_t * np.ones((self.nx, self.ny, 1))
        rho[0:Lx, :, :] = (1 - fraction) * rho_t
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
        u_tree.append(u)
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        # _, yy = coord[:, 0], coord[:, 1]
        # poiseuille_profile = lambda x, x0, d, umax: np.maximum(
        #     0.0, 4.0 * umax / (d**2) * ((x - x0) * d - (x - x0) ** 2)
        # )

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
        self.BCs[1].append(
            BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )

        # apply inlet equilibrium boundary condition at the left
        # inlet = self.boundingBoxIndices["left"]
        # yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        # rho_inlet = (
        #     fraction
        #     * rho_t
        #     * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        # )
        # vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        # vel_inlet[:, 0] = poiseuille_profile(
        #     yy_inlet,
        #     yy_inlet.min(),
        #     yy_inlet.max() - yy_inlet.min(),
        #     3.0 / 2.0 * prescribed_vel,
        # )
        # self.BCs[0].append(
        #     EquilibriumBC(
        #         tuple(inlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_inlet,
        #         vel_inlet,
        #     )
        # )
        # rho_inlet = (
        #     (1 - fraction)
        #     * rho_t
        #     * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        # )
        # vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        # vel_inlet[:, 0] = poiseuille_profile(
        #     yy_inlet,
        #     yy_inlet.min(),
        #     yy_inlet.max() - yy_inlet.min(),
        #     3.0 / 2.0 * prescribed_vel,
        # )
        # self.BCs[1].append(
        #     EquilibriumBC(
        #         tuple(inlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_inlet,
        #         vel_inlet,
        #     )
        # )

        # # Same at the outlet
        # outlet = self.boundingBoxIndices["right"]
        # yy_outlet = yy.reshape(self.nx, self.ny)[tuple(outlet.T)]
        # rho_outlet = (
        #     fraction
        #     * rho_t
        #     * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        # )
        # vel_outlet = np.zeros(outlet.shape, dtype=self.precisionPolicy.compute_dtype)
        # vel_outlet[:, 0] = poiseuille_profile(
        #     yy_outlet,
        #     yy_outlet.min(),
        #     yy_outlet.max() - yy_outlet.min(),
        #     3.0 / 2.0 * prescribed_vel,
        # )
        # self.BCs[0].append(
        #     EquilibriumBC(
        #         tuple(outlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_outlet,
        #         vel_outlet,
        #     )
        # )
        # rho_outlet = (
        #     (1 - fraction)
        #     * rho_t
        #     * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        # )
        # vel_outlet = np.zeros(outlet.shape, dtype=self.precisionPolicy.compute_dtype)
        # vel_outlet[:, 0] = poiseuille_profile(
        #     yy_outlet,
        #     yy_outlet.min(),
        #     yy_outlet.max() - yy_outlet.min(),
        #     3.0 / 2.0 * prescribed_vel,
        # )
        # self.BCs[1].append(
        #     EquilibriumBC(
        #         tuple(outlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_outlet,
        #         vel_outlet,
        #     )
        # )

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
        rho = np.array(kwargs.get("rho_total")[0, 1:-1, 1:-1, :])
        rho_d = np.array(kwargs["rho_tree"][0][0, 1:-1, 1:-1, :])
        rho_i = np.array(kwargs["rho_tree"][1][0, 1:-1, 1:-1, :])
        p_d = np.array(kwargs["p_tree"][0][1:-1, 1:-1])
        p_i = np.array(kwargs["p_tree"][1][1:-1, 1:-1])
        p = np.array(kwargs["p"][0, 1:-1, 1:-1])
        u_d = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, :])
        u_i = np.array(kwargs["u_tree"][1][0, 1:-1, 1:-1, :])
        u = np.array(kwargs["u_total"][0, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "p_d": p_d[..., 0],
            "p_i": p_i[..., 0],
            "rho": rho[..., 0],
            "rho_displaced": rho_d[..., 0],
            "rho_invading": rho_i[..., 0],
            "ux_displaced": u_d[..., 0],
            "uy_displaced": u_d[..., 1],
            "ux_invading": u_i[..., 0],
            "uy_invading": u_i[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(
            timestep,
            fields,
            f"output_fx_{fx}_visc_ratio_{visc_ratio}",
            "data",
        )


if __name__ == "__main__":
    precision = "f32/f32"
    g_kkprime = -0.027 * np.ones((2, 2))
    g = 0.57
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g
    Lx = nx // 2
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

    s_rho = [0.0, 0.0]  # Mass
    s_e = [1.0, 1.0]
    s_eta = [1.0, 1.0]
    s_j = [0.0, 0.0]  # Momentum
    s_q = [1.0, 1.0]
    s_v = [1 / tau_1, 1 / tau_2]
    for fx in [0.8, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        kwargs = {
            "n_components": 2,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "g_kkprime": g_kkprime,
            "g_ks": [0.0, 0.0],
            "body_force": [fx * 1e-5, 0.0],
            "omega": [1 / tau_1, 1 / tau_2],
            "precision": precision,
            "M": [M, M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0.0, 0.0],
            "k": [0, 0],
            "A": np.zeros((2, 2)),
            "io_rate": 10,
            "compute_MLUPS": False,
            "print_info_rate": 30000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        # os.system("rm -rf output*/ *.vtk")
        try:
            sim = CapillaryFingering(**kwargs)
            sim.run(30000)
        except FloatingPointError as _:  # Larger forces can lead to instability
            continue
