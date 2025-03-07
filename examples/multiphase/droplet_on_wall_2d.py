"""
Multi-component droplet on wall example. Used for parameter tuning (fluid-wall interaction coefficient) for setting contact angle. Parameters set here are used in channe_rel_perm example.
Unlike Free-Energy method, contact angle in Shan-Chen method is dependent on the interaction coefficient between wall and the fluid and it changes with lattice size. A good approach to
set interaction strengths can be use equivalent size domain and then compute contact angle v/s inital volume (radius) if it convergences after some volume sizes, that intetraction value is
good enough.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.tree import map, reduce
from functools import partial
import operator

from src.lattice import LatticeD2Q9
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


# Initial semi circular droplet specification
r = 20
nx = 100
ny = 100

# Viscosity ratio
M = 0.1

# Wetting phase
tau_w = 1.98
v_w = (tau_w - 0.5) / 3

# Non-wetting phase
v_nw = v_w / M
tau_nw = 3 * v_nw + 0.5

# Density
rho_w = 1.0
rho_nw = 1.0

# Fraction of total density in the bulk of one component
# It needs to be non-zero value to prevent NaN values.
fraction = 0.98


class DropletOnWall2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        dist = x**2 + (y - self.ny / 2) ** 2 - r**2

        # Wall indices where density as 1.0
        wall = np.concatenate(
            (self.boundingBoxIndices["bottom"], self.boundingBoxIndices["top"])
        )

        # Initialize the domain with a semi-circular droplet
        # Wetting phase
        rho = (1 - fraction) * rho_w * np.ones((self.nx, self.ny, 1))
        rho[dist <= 0] = fraction * rho_w
        # rho[tuple(wall.T)] = 1.0  # Set wall density as 1.0
        rho = rho.reshape((nx, ny, 1))
        rho_tree.append(rho)

        # Non-wetting phase
        rho = fraction * rho_nw * np.ones((self.nx, self.ny, 1))
        rho[dist <= 0] = (1 - fraction) * rho_nw
        # rho[tuple(wall.T)] = 1.0  # Set wall density as 1.0
        rho = rho.reshape((nx, ny, 1))
        rho_tree.append(rho)

        u_tree = [
            np.zeros((self.nx, self.ny, 2)),
            np.zeros((self.nx, self.ny, 2)),
        ]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate(
            (self.boundingBoxIndices["bottom"], self.boundingBoxIndices["top"])
        )
        self.BCs[0].append(
            BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )
        self.BCs[1].append(
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
        rho_w = np.array(kwargs["rho_tree"][0][0, :, 2:-2, :])
        rho_nw = np.array(kwargs["rho_tree"][1][0, :, 2:-2, :])
        rho = np.array(kwargs["rho_total"][0, :, 2:-2, :])
        u_w = np.array(kwargs["u_tree"][0][0, :, 2:-2, :])
        u_nw = np.array(kwargs["u_tree"][1][0, :, 2:-2, :])
        u = np.array(kwargs["u_total"][0, :, 2:-2, :])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "rho_w": rho_w[..., 0],
            "rho_nw": rho_nw[..., 0],
            "ux_w": u_w[..., 0],
            "uy_w": u_w[..., 1],
            "ux_nw": u_nw[..., 0],
            "uy_nw": u_nw[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        u_sp = np.sqrt(np.sum(np.square(u_w), axis=-1))
        print(f"Max spurious velocity for wetting phase: {np.max(u_sp)}")
        u_sp = np.sqrt(np.sum(np.square(u_nw), axis=-1))
        print(f"Max spurious velocity for non-wetting phase: {np.max(u_sp)}")
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


if __name__ == "__main__":
    precision = "f32/f32"
    # Intra & inter component interaction coefficients, must be a symmetric matrix.
    g_kkprime = 0 * np.ones((2, 2))  # No intra-component interaction
    g = 1.295
    g_kkprime[0, 1] = g  # Inter-component interaction coefficient
    g_kkprime[1, 0] = g  # Inter-component interaction coefficient

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
    s_e = [0.3, 0.3]
    s_eta = [1.0, 1.0]
    s_j = [0.0, 0.0]  # Momentum
    s_q = [1.0, 1.0]
    s_v = [1 / tau_w, 1 / tau_nw]
    kwargs = {
        "lattice": LatticeD2Q9(precision),
        "omega": [1 / tau_w, 1 / tau_nw],
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "n_components": 2,
        "g_kkprime": g_kkprime,
        "g_ks": [
            0.3,
            -0.3,
        ],  # Wall density is as 1.0, change this to change contact angle
        "M": [M, M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0, 0],
        # values not used, can be set as anything
        "k": [0.0, 0.0],
        "A": 0 * np.zeros((2, 2)),
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,  # Disable checkpointing
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    os.system("rm -rf output*/")
    sim = DropletOnWall2D(**kwargs)
    sim.run(15000)
