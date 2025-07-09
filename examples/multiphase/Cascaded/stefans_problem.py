"""
Multi-component 2D and 3D isothermal Stefans problem where liquid undergoes evaporation due to the prescribed vapor pressure gradient.
The boundary conditions are: periodic (left and right), no slip-wall (at the bottom)

The collision matrix is based on:
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of heterogeneity on evaporation in porous media.
Journal of Fluid Mechanics 983, A6 (2024).
"""

import os
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, config
from jax.tree import map

from src.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation
from src.eos import Peng_Robinson
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.multiphase import MultiphaseCascade
from src.utils import save_fields_vtk

# config.update("jax_default_matmul_precision", "float32")


# Estimate surface tension of the water component
class Droplet2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)
        rho = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)
        rho = 0.5 * (rho_a_l + rho_a_g) - 0.5 * (rho_a_l - rho_a_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        u_tree.append(u)
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G),
            self.k,
            p_tree,
            rho_tree,
            self.g_kkprime.diagonal().tolist(),
        )
        psi_tree[1] = rho_tree[1]
        # Zhang-Chen potential
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        if inter_component_interaction:
            F_1 = (
                -g_AB
                * rho_tree[0]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[1], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
            # F_2 = (
            #     -g_AB
            #     * rho_tree[1]
            #     * jnp.dot(
            #         self.G_ff * self.streaming(jnp.repeat(rho_tree[0], repeats=self.lattice.q, axis=-1)),
            #         self.precisionPolicy.cast_to_compute(self.lattice.c.T),
            #     )
            # )
            F_2 = F_1
            F_tree = [F_1, F_2]
            fluid_fluid_force = map(lambda f1, f2: f1 + f2, fluid_fluid_force, F_tree)
        if self.body_force is not None:
            return map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, F_intra_tree, psi_tree):
        def f(F, F_intra, sigma, psi, s_b):
            C = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
            Fx = F_intra[..., 0]
            Fy = F_intra[..., 1]
            eta = 4 * sigma * (Fx**2 + Fy**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
            Fx = F[..., 0]
            Fy = F[..., 1]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(eta)
            C = C.at[..., 6].set(Fy * self.lattice.cs2)
            C = C.at[..., 7].set(Fx * self.lattice.cs2)
            C = C.at[..., 8].set(eta * self.lattice.cs2)
            return C

        return map(lambda F, F_intra, sigma, psi, s_b: f(F, F_intra, sigma, psi, s_b), F_tree, F_intra_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree):
        F_tree = self.compute_force(rho_tree)
        F_intra_tree = self.compute_force(rho_tree, inter_component_interaction=False)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, F_intra_tree, psi_tree)
        Tf_tree = map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = self.eos.EOS(rho_tree)
        p_tree[1] = self.lattice.cs2 * rho_tree[1]
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] + g_AB * rho_tree[0] * rho_tree[1]

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        offset = 90
        p_north = p[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        # print(f"Pressure difference: {pressure_difference}")
        save_fields_vtk(
            timestep,
            fields,
            f"output_{r}",
            "data",
        )
        if timestep == 20000:
            f.write(f"{pressure_difference}, {r}, {1 / r}\n")


class StefansProblem2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho = np.ones((self.nx, self.ny, 1))
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g_boundary) - 0.5 * (rho_w_l - rho_w_g_boundary) * np.tanh(2 * (y - L) / width)
        rho[:, self.ny - 1, 0] = rho_w_g_boundary
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        rho = np.ones((self.nx, self.ny, 1))
        rho[..., 0] = 0.5 * (rho_a_l + rho_a_g_boundary) - 0.5 * (rho_a_l - rho_a_g_boundary) * np.tanh(2 * (y - L) / width)
        # d = self.ny - L - width
        # rho[:, L + width : self.ny, 0] = rho_a_g + np.arange(0, d) * (rho_a_g_boundary - rho_a_g) / d
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        u_tree.append(u)
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        bottom = self.boundingBoxIndices["bottom"]
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))
        self.BCs[1].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))

        # Define indices, used later
        top = self.boundingBoxIndices["top"]
        # top = np.array([[x, 499] for x in range(1, self.nx - 1)], dtype=int)
        rho = rho_w_g_boundary * np.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))
        rho = rho_a_g_boundary * np.ones((top.shape[0], 1))
        self.BCs[1].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))
        self.ind_computed = False

    @partial(jit, static_argnums=(0, 4))
    def apply_bc(self, fout_tree, fin_tree, timestep, implementation_step):
        def _apply_bc_(fin, fout, bc):
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if (bc.implementationStep == implementation_step) and (not isinstance(bc, ExactNonEquilibriumExtrapolation)):
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
            return fout

        def __apply_bc__(fout, fin, BCs):
            for bc in BCs:
                fout = _apply_bc_(fin, fout, bc)
            return fout

        fout_tree = map(lambda fout, fin, BCs: __apply_bc__(fout, fin, BCs), fout_tree, fin_tree, self.BCs)
        # Modify the apply_bc function to separately handle the case of eNEQ boundary condition using mixture velocity instead of component velocity
        if implementation_step == "PostStreaming":
            if not self.ind_computed:
                self.BCs[0][1].find_neighbors()  # Uses normal to identify the neighboring lattice sites
                self.ind = self.BCs[0][1].indices
                nbd = len(self.ind[0])
                bindex = np.arange(nbd)[:, None]
                self.ind_nbr = self.BCs[0][1].indices_nbr
                self.imissing = self.BCs[0][1].imissing
                w_NEQ = self.BCs[0][1].w_NEQ
                self.ind_computed = True

            rho_tree, _ = self.update_macroscopic(fout_tree)
            u_tree = self.macroscopic_velocity(fout_tree, rho_tree)
            u_mixture = self.compute_total_velocity(rho_tree, u_tree)

            def eNEQ(fout, u, prescribed):
                fbd = fout[self.ind]
                f_nbr = fout[self.ind_nbr]
                rho_nbr = jnp.sum(f_nbr, axis=-1, keepdims=True)
                rho = jnp.sum(fbd, axis=-1, keepdims=True)
                feq = self.equilibrium(rho, u_mixture[self.ind_nbr])
                feq_nbr = self.equilibrium(rho_nbr, u[self.ind_nbr])
                fneq_nbr = f_nbr - feq_nbr
                fbd = fbd.at[bindex, self.imissing].set(feq[bindex, self.imissing] + fneq_nbr[bindex, self.imissing])

                # Correction step
                rho_incorrect = jnp.sum(fbd, axis=-1, keepdims=True)
                beta = w_NEQ * jnp.repeat(prescribed - rho_incorrect, axis=-1, repeats=self.lattice.q) / jnp.sum(self.G_ff)
                fbd = fbd.at[bindex, self.imissing].set(fbd[bindex, self.imissing] + beta[bindex, self.imissing])
                fout = fout.at[self.ind].set(fbd)
                return fout

            return map(
                lambda fout, u, prescribed: eNEQ(fout, u, prescribed), fout_tree, u_tree, [self.BCs[0][1].prescribed, self.BCs[1][1].prescribed]
            )
        else:
            return fout_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G),
            self.k,
            p_tree,
            rho_tree,
            self.g_kkprime.diagonal().tolist(),
        )
        psi_tree[1] = rho_tree[1]
        # Zhang-Chen potential
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        if inter_component_interaction:
            F_1 = (
                -g_AB
                * rho_tree[0]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[1], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
            # F_2 = (
            #     -g_AB
            #     * rho_tree[1]
            #     * jnp.dot(
            #         self.G_ff * self.streaming(jnp.repeat(rho_tree[0], repeats=self.lattice.q, axis=-1)),
            #         self.precisionPolicy.cast_to_compute(self.lattice.c.T),
            #     )
            # )
            F_2 = F_1
            F_tree = [F_1, F_2]
            fluid_fluid_force = map(lambda f1, f2: f1 + f2, fluid_fluid_force, F_tree)
        if self.body_force is not None:
            return map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, F_intra_tree, psi_tree):
        def f(F, F_intra, sigma, psi, s_b):
            C = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
            Fx = F_intra[..., 0]
            Fy = F_intra[..., 1]
            eta = 4 * sigma * (Fx**2 + Fy**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
            Fx = F[..., 0]
            Fy = F[..., 1]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(eta)
            C = C.at[..., 6].set(Fy * self.lattice.cs2)
            C = C.at[..., 7].set(Fx * self.lattice.cs2)
            C = C.at[..., 8].set(eta * self.lattice.cs2)
            return C

        return map(lambda F, F_intra, sigma, psi, s_b: f(F, F_intra, sigma, psi, s_b), F_tree, F_intra_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree):
        F_tree = self.compute_force(rho_tree)
        F_intra_tree = self.compute_force(rho_tree, inter_component_interaction=False)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, F_intra_tree, psi_tree)
        Tf_tree = map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = self.eos.EOS(rho_tree)
        p_tree[1] = self.lattice.cs2 * rho_tree[1]
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] + g_AB * rho_tree[0] * rho_tree[1]

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


class StefansProblem3D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        _, y, _ = np.meshgrid(np.linspace(0, self.nx - 1, self.nx, dtype=int), y, np.linspace(0, self.nz - 1, self.nz, dtype=int))
        y = np.transpose(y, axes=(1, 0, 2))
        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g_boundary) - 0.5 * (rho_w_l - rho_w_g_boundary) * np.tanh(2 * (y - L) / width)
        rho[:, self.ny - 1, ...] = rho_w_g_boundary
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho[..., 0] = 0.5 * (rho_a_l + rho_a_g_boundary) - 0.5 * (rho_a_l - rho_a_g_boundary) * np.tanh(2 * (y - L) / width)
        # d = self.ny - L - width
        # rho[:, L + width : self.ny, 0] = rho_a_g + np.arange(0, d) * (rho_a_g_boundary - rho_a_g) / d
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        u_tree.append(u)
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        bottom = self.boundingBoxIndices["front"]
        # bottom = np.array([[x, 0, z] for x in range(self.nx) for z in range(self.nz)], dtype=int)
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))
        self.BCs[1].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))

        # Define indices, used later
        top = self.boundingBoxIndices["back"]
        # top = np.array([[x, self.ny - 1, z] for x in range(self.nx) for z in range(self.nz)], dtype=int)
        rho = rho_w_g_boundary * np.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))
        rho = rho_a_g_boundary * np.ones((top.shape[0], 1))
        self.BCs[1].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))
        self.ind_computed = False

    @partial(jit, static_argnums=(0, 4))
    def apply_bc(self, fout_tree, fin_tree, timestep, implementation_step):
        def _apply_bc_(fin, fout, bc):
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if (bc.implementationStep == implementation_step) and (not isinstance(bc, ExactNonEquilibriumExtrapolation)):
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
            return fout

        def __apply_bc__(fout, fin, BCs):
            for bc in BCs:
                fout = _apply_bc_(fin, fout, bc)
            return fout

        fout_tree = map(lambda fout, fin, BCs: __apply_bc__(fout, fin, BCs), fout_tree, fin_tree, self.BCs)
        # Modify the apply_bc function to separately handle the case of eNEQ boundary condition using mixture velocity instead of component velocity
        if implementation_step == "PostStreaming":
            if not self.ind_computed:
                self.BCs[0][1].find_neighbors()  # Uses normal to identify the neighboring lattice sites
                self.ind = self.BCs[0][1].indices
                nbd = len(self.ind[0])
                bindex = np.arange(nbd)[:, None]
                self.ind_nbr = self.BCs[0][1].indices_nbr
                self.imissing = self.BCs[0][1].imissing
                w_NEQ = self.BCs[0][1].w_NEQ
                self.ind_computed = True

            rho_tree, _ = self.update_macroscopic(fout_tree)
            u_tree = self.macroscopic_velocity(fout_tree, rho_tree)
            u_mixture = self.compute_total_velocity(rho_tree, u_tree)

            def eNEQ(fout, u, prescribed):
                fbd = fout[self.ind]
                f_nbr = fout[self.ind_nbr]
                rho_nbr = jnp.sum(f_nbr, axis=-1, keepdims=True)
                rho = jnp.sum(fbd, axis=-1, keepdims=True)
                feq = self.equilibrium(rho, u_mixture[self.ind_nbr])
                feq_nbr = self.equilibrium(rho_nbr, u[self.ind_nbr])
                fneq_nbr = f_nbr - feq_nbr
                fbd = fbd.at[bindex, self.imissing].set(feq[bindex, self.imissing] + fneq_nbr[bindex, self.imissing])

                # Correction step
                rho_incorrect = jnp.sum(fbd, axis=-1, keepdims=True)
                beta = w_NEQ * jnp.repeat(prescribed - rho_incorrect, axis=-1, repeats=self.lattice.q) / jnp.sum(self.G_ff)
                fbd = fbd.at[bindex, self.imissing].set(fbd[bindex, self.imissing] + beta[bindex, self.imissing])
                fout = fout.at[self.ind].set(fbd)
                return fout

            return map(
                lambda fout, u, prescribed: eNEQ(fout, u, prescribed), fout_tree, u_tree, [self.BCs[0][1].prescribed, self.BCs[1][1].prescribed]
            )
        else:
            return fout_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G),
            self.k,
            p_tree,
            rho_tree,
            self.g_kkprime.diagonal().tolist(),
        )
        psi_tree[1] = rho_tree[1]
        # Zhang-Chen potential
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        if inter_component_interaction:
            F_1 = (
                -g_AB
                * rho_tree[0]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[1], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
            # F_2 = (
            #     -g_AB
            #     * rho_tree[1]
            #     * jnp.dot(
            #         self.G_ff * self.streaming(jnp.repeat(rho_tree[0], repeats=self.lattice.q, axis=-1)),
            #         self.precisionPolicy.cast_to_compute(self.lattice.c.T),
            #     )
            # )
            F_2 = F_1
            F_tree = [F_1, F_2]
            fluid_fluid_force = map(lambda f1, f2: f1 + f2, fluid_fluid_force, F_tree)
        if self.body_force is not None:
            return map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, F_intra_tree, psi_tree):
        def f(F, F_intra, sigma, psi, s_b):
            C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
            Fx = F_intra[..., 0]
            Fy = F_intra[..., 1]
            Fz = F_intra[..., 2]
            eta = 6 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(Fz)
            C = C.at[..., 7].set(eta)
            C = C.at[..., 8].set(eta)
            C = C.at[..., 9].set(eta)
            C = C.at[..., 10].set(Fx * self.lattice.cs2)
            C = C.at[..., 11].set(Fx * self.lattice.cs2)
            C = C.at[..., 12].set(Fy * self.lattice.cs2)
            C = C.at[..., 13].set(Fz * self.lattice.cs2)
            C = C.at[..., 14].set(Fy * self.lattice.cs2)
            C = C.at[..., 15].set(Fz * self.lattice.cs2)
            return C

        return map(lambda F, F_intra, sigma, psi, s_b: f(F, F_intra, sigma, psi, s_b), F_tree, F_intra_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree):
        F_tree = self.compute_force(rho_tree)
        F_intra_tree = self.compute_force(rho_tree, inter_component_interaction=False)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, F_intra_tree, psi_tree)
        Tf_tree = map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = self.eos.EOS(rho_tree)
        p_tree[1] = self.lattice.cs2 * rho_tree[1]
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] + g_AB * rho_tree[0] * rho_tree[1]

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_prev_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_prev_tree"][1][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


class DropletOnWall2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2 - 100) ** 2)
        rho = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = [rho]
        rho = 0.5 * (rho_a_l + rho_a_g) - 0.5 * (rho_a_l - rho_a_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        self.BCs[0].append(
            BounceBack(
                tuple(ind.T),
                self.gridInfo,
                self.precisionPolicy,
                theta[tuple(ind.T)],
                phi[tuple(ind.T)],
                delta_rho[tuple(ind.T)],
            )
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, :, :, :])
        u = np.array(kwargs["u_tree"][0][0, :, :, :])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "flag": np.array(self.solid_mask_streamed[0][..., 0]),
        }
        u_sp = np.sqrt(np.sum(np.square(u), axis=-1))
        print(f"Max spurious velocity: {np.max(u_sp)}")
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


if __name__ == "__main__":
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    # Cascaded LBM collision matrix
    M = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])

    rho_w_l = 6.499210784
    rho_w_g = 0.379598891

    a = [3 / 49, 1]
    b = [2 / 21, 1]
    R = [1, 1]
    Tc = 0.1093785558
    T = 0.86 * Tc

    kwargs = {"a": a, "b": b, "pr_omega": [0.344, 1], "R": R, "T": T}
    eos = Peng_Robinson(**kwargs)

    s2 = 0.8  # This sets the kinematic viscosity
    s_0 = [1.0, 1.0]  # Mass conservation
    s_1 = [1.0, 1.0]  # Fixed
    s_2 = [s2, s2]
    s_b = [0.8, 0.8]
    s_3 = [0.8, 0.8]  # [(16 - 8 * s2) / (8 - s2), (16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [1.0, 1.0]  # No slip

    g_kkprime = -1 * np.ones((2, 2))
    g_kkprime[1, 1] = 1
    g_kkprime[0, 1] = 0
    g_kkprime[1, 0] = 0
    g_AB = -0.15

    precision = "f32/f32"

    # Estimate surface tension
    # rho_a_l = 1e-6
    # rho_a_g = 1e-6
    # nx = 200
    # ny = 200
    # width = 5
    # f = open("pressure_difference.txt", "w")
    # f.write("Pressure Difference, R, 1/R\n")
    # os.system("rm -rf output*/ *.vtk")
    # for r in [30, 35, 40, 45, 50]:
    #     kwargs = {
    #         "n_components": 2,
    #         "lattice": LatticeD2Q9(precision),
    #         "nx": nx,
    #         "ny": ny,
    #         "nz": 0,
    #         "g_kkprime": g_kkprime,
    #         "body_force": [0.0, 0.0],
    #         "omega": [s2, s2],
    #         "EOS": eos,
    #         "k": [1.0, 1.0],
    #         "A": np.zeros((2, 2)),
    #         "M": [M, M],
    #         "s_0": s_0,
    #         "s_1": s_1,
    #         "s_b": s_b,
    #         "s_2": s_2,
    #         "s_3": s_3,
    #         "s_4": s_4,
    #         "sigma": [0.099, 0.0],
    #         "precision": precision,
    #         "io_rate": 10000,
    #         "print_info_rate": 10000,
    #         "compute_MLUPS": False,
    #         "checkpoint_rate": -1,
    #         "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #         "restore_checkpoint": False,
    #     }
    #     sim = Droplet2D(**kwargs)
    #     sim.run(20000)
    # f.close()
    #
    # Isothermal Stefan's Problem benchmark in 2D
    # nx = 5
    # ny = 400
    # width = 5
    # rho_a_l = 0.0019
    # rho_a_g = 0.0019
    # rho_w_g_boundary = rho_w_g - 0.0003
    # rho_a_g_boundary = rho_a_g + 0.0003
    # # initial liquid region width
    # L = ny // 3
    # kwargs = {
    #     "n_components": 2,
    #     "lattice": LatticeD2Q9(precision),
    #     "nx": nx,
    #     "ny": ny,
    #     "nz": 0,
    #     "g_kkprime": g_kkprime,
    #     "body_force": [0.0, 0.0],
    #     "omega": [s2, s2],
    #     "EOS": eos,
    #     "k": [1.0, 1.0],
    #     "A": np.zeros((2, 2)),
    #     "M": [M, M],
    #     "s_0": s_0,
    #     "s_1": s_1,
    #     "s_b": s_b,
    #     "s_2": s_2,
    #     "s_3": s_3,
    #     "s_4": s_4,
    #     "sigma": [0.099, 0.0],
    #     "precision": precision,
    #     "io_rate": 100,
    #     "print_info_rate": 100,
    #     "compute_MLUPS": False,
    #     "checkpoint_rate": -1,
    #     "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #     "restore_checkpoint": False,
    # }
    # os.system("rm -rf output*/ *.vtk")
    # sim = StefansProblem2D(**kwargs)
    # sim.run(2000)
    #
    # D3Q19
    # e = LatticeD3Q19().c.T
    # ex = e[:, 0]
    # ey = e[:, 1]
    # ez = e[:, 2]
    # M = np.zeros((19, 19))
    # M[0, :] = ex**0
    # M[1, :] = ex
    # M[2, :] = ey
    # M[3, :] = ez
    # M[4, :] = ex * ey
    # M[5, :] = ex * ez
    # M[6, :] = ey * ez
    # M[7, :] = ex * ex
    # M[8, :] = ey * ey
    # M[9, :] = ez * ez
    # M[10, :] = ex * ey * ey
    # M[11, :] = ex * ez * ez
    # M[12, :] = ey * ex * ex
    # M[13, :] = ez * ex * ex
    # M[14, :] = ey * ez * ez
    # M[15, :] = ez * ey * ey
    # M[16, :] = ex * ex * ey * ey
    # M[17, :] = ex * ex * ez * ez
    # M[18, :] = ey * ey * ez * ez

    # s2 = 0.8
    # # s3 = (16 - 8 * s2) / (8 - s2)
    # s_0 = [1.0, 1.0]
    # s_1 = [1.0, 1.0]
    # s_2 = [s2, s2]
    # s_b = [0.8, 0.8]
    # s_3 = [1.8, 1.8]
    # s_4 = [1.0, 1.0]

    # nx = 20
    # ny = 500
    # nz = 20
    # width = 5
    # rho_a_l = 0.1
    # rho_a_g = 0.0019
    # rho_w_g_boundary = rho_w_g - 0.0745
    # rho_a_g_boundary = rho_a_g + 0.0745
    # # rho_w_l_boundary = rho_w_l - 0.189
    # # rho_a_l_boundary = rho_a_l + 0.189
    # # initial liquid region width
    # L = 2 * ny // 5
    # kwargs = {
    #     "n_components": 2,
    #     "lattice": LatticeD3Q19(precision),
    #     "nx": nx,
    #     "ny": ny,
    #     "nz": nz,
    #     "g_kkprime": g_kkprime,
    #     "body_force": [0.0, 0.0, 0.0],
    #     "omega": [s2, s2],
    #     "EOS": eos,
    #     "k": [1.0, 1.0],
    #     "A": np.zeros((2, 2)),
    #     "M": [M, M],
    #     "s_0": s_0,
    #     "s_1": s_1,
    #     "s_b": s_b,
    #     "s_2": s_2,
    #     "s_3": s_3,
    #     "s_4": s_4,
    #     "sigma": [0.102, 0.0],
    #     "precision": precision,
    #     "io_rate": 10,
    #     "print_info_rate": 10,
    #     "compute_MLUPS": False,
    #     "checkpoint_rate": -1,
    #     "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #     "restore_checkpoint": False,
    # }
    # os.system("rm -rf output*/ *.vtk")
    # sim = StefansProblem3D(**kwargs)
    # sim.run(1000)

    r = 50
    width = 5
    nx = 300
    ny = 350
    theta = 60 * np.pi / 180 * np.ones((nx, ny, 1))
    phi = 1.1 * np.ones((nx, ny, 1))
    delta_rho = 0.0 * np.ones((nx, ny, 1))

    rho_w_l = 6.499210784
    rho_w_g = 0.379598891
    rho_a_l = 0.1
    rho_a_g = 0.0019
    rho_w_g_boundary = rho_w_g - 0.0745
    rho_a_g_boundary = rho_a_g + 0.0745

    # Circular wall
    R = 70
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T
    circ = (x - nx / 2) ** 2 + (y - ny / 2 + 20) ** 2 - R**2
    ind = np.array(np.where(circ <= 0), dtype=int).T
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": g_kkprime,
        "body_force": [0.0, 0.0],
        "omega": [s2, s2],
        "EOS": eos,
        "k": [1.0, 1.0],
        "A": np.zeros((2, 2)),
        "M": [M, M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "sigma": [0.099, 0.0],
        "precision": precision,
        "io_rate": 10000,
        "print_info_rate": 10000,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim = DropletOnWall2D(*kwargs)
    sim.run(20000)
