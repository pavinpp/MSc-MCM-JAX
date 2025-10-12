"""
Multi-component 2D and 3D isothermal Stefans problem where liquid undergoes evaporation due to the prescribed vapor concentration gradient.
The boundary conditions are: periodic (left and right), no slip-wall (at the bottom).  The gas phase densities are changed to account for
different air mass fraction. The densities are changed by same amounts to keep total pressure constant.

The collision matrix for 2D and 3D cases are respectively based on:
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of heterogeneity on evaporation in porous media.
Journal of Fluid Mechanics 983, A6 (2024).
2. Fei, L., Luo, K. H. & Li, Q. Three-dimensional cascaded lattice Boltzmann method: Improved implementation and consistent
forcing scheme. Phys. Rev. E 97, 053309 (2018).
"""

import os
from functools import partial
import operator

import jax.numpy as jnp
import numpy as np
from jax import config, debug, jit, vmap
from jax.tree import map, reduce

from src.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation
from src.eos import Peng_Robinson
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.multiphase import MultiphaseCascade
from src.utils import save_fields_vtk

config.update("jax_default_matmul_precision", "float32")
# config.update("jax_debug_nans", True)


class SingleComponent(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho = np.ones((self.nx, self.ny, 1))
        # rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (y - L) / width)
        rho[:, 0:L, 0] = rho_w_l
        rho[:, L : self.ny, 0] = rho_w_g
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Define indices, used later
        top = self.boundingBoxIndices["top"]
        rho = 0.75 * rho_w_g * np.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho, "density"))

        bottom = self.boundingBoxIndices["bottom"]
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))

        # Used in apply_bc function
        self.ind = tuple(np.array([[x, self.ny - 1] for x in range(self.nx)], dtype=int).T)
        self.ind_nbr = tuple(np.array([[x, self.ny - 2] for x in range(self.nx)], dtype=int).T)
        self.imissing = np.array([4, 7, 8], dtype=int)
        self.w_NEQ = self.BCs[0][0].w_NEQ

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

        # Handle eNEQ at PostStreaming using post-stream fields and BC-provided indices/weights
        if implementation_step == "PostStreaming":
            # Use post-streaming distributions to compute macroscopic fields and neighbor non-equilibrium
            rho_tree, _ = self.update_macroscopic(fin_tree)
            u_tree = self.macroscopic_velocity(fin_tree, rho_tree)
            u_mixture = self.compute_total_velocity(rho_tree, u_tree)

            def eNEQ(fout, rho, u, prescribed):
                # Neighbor values from post-streaming
                f_nbr = fout[self.ind_nbr]
                rho_nbr = rho[self.ind_nbr]
                u_nbr = u[self.ind_nbr]
                feq_nbr = self.equilibrium(rho_nbr, u_nbr)
                fneq_nbr = f_nbr - feq_nbr

                # Prescribed density at boundary, velocity from interior neighbor
                feq_prescribed = self.equilibrium(prescribed, u_mixture[self.ind_nbr])

                # Set missing directions using eNEQ formula
                f_bdr = fout[self.ind]
                f_bdr = f_bdr.at[..., self.imissing].set(feq_prescribed[..., self.imissing] + fneq_nbr[..., self.imissing])

                # Correction: enforce boundary density by distributing error over missing dirs via NEQ weights
                rho_incorrect = jnp.sum(f_bdr, axis=-1, keepdims=True)
                beta = self.w_NEQ[self.imissing] * jnp.repeat(prescribed - rho_incorrect, axis=-1, repeats=3) / jnp.sum(self.w_NEQ[self.imissing])
                f_bdr = f_bdr.at[..., self.imissing].add(beta)

                fout = fout.at[self.ind].set(f_bdr)
                # fout = fout.at[self.ind_bottom].set(fin[self.ind_bottom])  # Revert the changes made to bottom boundary due to streaming step
                return fout

            fout_tree = map(lambda fout, rho, u, prescribed: eNEQ(fout, rho, u, prescribed), fout_tree, rho_tree, u_tree, [self.BCs[0][0].prescribed])

        return fout_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G), self.k, p_tree, rho_tree, self.g_kkprime.diagonal().tolist()
        )
        # Zhang-Chen potential is not used here
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
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
            # C = C.at[..., 8].set(eta * self.lattice.cs2)
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

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho_water": rho_water[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(timestep, fields, "output", "data")


class StefansProblem2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho = np.ones((self.nx, self.ny, 1))
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (y - L) / width)
        # rho[:, 0:L, 0] = rho_w_l
        # rho[:, L : self.ny, 0] = rho_w_g
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        rho = np.ones((self.nx, self.ny, 1))
        rho[..., 0] = 0.5 * (rho_a_l + rho_a_g) - 0.5 * (rho_a_l - rho_a_g) * np.tanh(2 * (y - L) / width)
        # rho[:, 0:L, 0] = rho_a_l
        # rho[:, L : self.ny, 0] = rho_a_g
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Define indices, used later
        top = self.boundingBoxIndices["top"]
        rho = rho_w_g * np.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho, "density"))
        rho = rho_a_g * np.ones((top.shape[0], 1))
        self.BCs[1].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho, "density"))

        bottom = self.boundingBoxIndices["bottom"]
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))
        self.BCs[1].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))

        # Used in apply_bc function
        self.ind = tuple(np.array([[x, self.ny - 1] for x in range(self.nx)], dtype=int).T)
        self.ind_nbr = tuple(np.array([[x, self.ny - 2] for x in range(self.nx)], dtype=int).T)
        self.ind_nbr[0][0] = 1
        self.ind_nbr[0][self.nx - 1] = self.nx - 2
        self.imissing = np.array([4, 7, 8], dtype=int)
        self.w_NEQ = self.BCs[0][0].w_NEQ

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

        debug.print("\nImplementation step: {}, pre-NEQ application", implementation_step)
        debug.print("rho_out, top (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 1, :], axis=-1))
        debug.print("rho_out, top nbr (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 2, :], axis=-1))
        debug.print("rho_out, top next nbr (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 3, :], axis=-1))
        debug.print("rho_out, top (air): {}", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 1, :], axis=-1))
        debug.print("rho_out, top nbr (air): {}", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 2, :], axis=-1))
        debug.print("rho_out, top next nbr (air): {}", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 3, :], axis=-1))

        fout_tree = map(lambda fout, fin, BCs: __apply_bc__(fout, fin, BCs), fout_tree, fin_tree, self.BCs)

        # Handle eNEQ at PostStreaming using post-stream fields and BC-provided indices/weights
        if implementation_step == "PostStreaming":
            # Use post-streaming distributions to compute macroscopic fields and neighbor non-equilibrium
            rho_tree, _ = self.update_macroscopic(fin_tree)
            u_tree = self.macroscopic_velocity(fin_tree, rho_tree)
            u_mixture = self.compute_total_velocity(rho_tree, u_tree)

            def eNEQ(fout, rho, u, prescribed):
                # Neighbor values from post-streaming
                f_nbr = fout[self.ind_nbr]
                rho_nbr = rho[self.ind_nbr]
                u_nbr = u[self.ind_nbr]
                feq_nbr = self.equilibrium(rho_nbr, u_nbr)
                fneq_nbr = f_nbr - feq_nbr

                # Prescribed density at boundary, velocity from interior neighbor
                feq_prescribed = self.equilibrium(prescribed, u_mixture[self.ind_nbr])

                # Set missing directions using eNEQ formula
                f_bdr = fout[self.ind]
                f_bdr = f_bdr.at[..., self.imissing].set(feq_prescribed[..., self.imissing] + fneq_nbr[..., self.imissing])

                # Correction: enforce boundary density by distributing error over missing dirs via NEQ weights
                rho_incorrect = jnp.sum(f_bdr, axis=-1, keepdims=True)
                beta = self.w_NEQ[self.imissing] * jnp.repeat(prescribed - rho_incorrect, axis=-1, repeats=3) / jnp.sum(self.w_NEQ[self.imissing])
                f_bdr = f_bdr.at[..., self.imissing].add(beta)

                fout = fout.at[self.ind].set(f_bdr)
                return fout

            fout_tree = map(
                lambda fout, rho, u, prescribed: eNEQ(fout, rho, u, prescribed),
                fout_tree,
                rho_tree,
                u_tree,
                [self.BCs[0][0].prescribed, self.BCs[1][0].prescribed],
            )

        debug.print("\nImplementation step: {}, post-NEQ application", implementation_step)
        debug.print("rho_out, top (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 1, :], axis=-1))
        debug.print("rho_out, top nbr (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 2, :], axis=-1))
        debug.print("rho_out, top next nbr (water): {}", jnp.sum(fout_tree[0][self.nx // 2, self.ny - 3, :], axis=-1))
        debug.print("rho_out, top (air): {}, air", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 1, :], axis=-1))
        debug.print("rho_out, top nbr (air): {}, air", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 2, :], axis=-1))
        debug.print("rho_out, top next nbr (air): {}", jnp.sum(fout_tree[1][self.nx // 2, self.ny - 3, :], axis=-1))

        return fout_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)

        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G), self.k, p_tree, rho_tree, self.g_kkprime.diagonal().tolist()
        )

        # This is not used for the air component, it just needs to be non-zero to prevent NaN values during eta calculation in compute_force_central_moments
        psi_tree[1] = rho_tree[1]

        # Zhang-Chen potential is also not used here
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree, U_tree):
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        psi_s_tree = map(lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)), psi_tree)

        def ffk_1(Ai, g_kkprime):
            return reduce(operator.add, map(lambda A, G, psi_s: jnp.dot((1 - A) * G * self.G_ff * psi_s, c), list(Ai), list(g_kkprime), psi_s_tree))

        return map(lambda psi, nt_1: psi * nt_1, psi_tree, list(vmap(ffk_1, in_axes=(0, 0))(self.A, self.g_kkprime)))

    @partial(jit, static_argnums=(0, 2), donate_argnums=(1,))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        fluid_fluid_force[1] = jnp.zeros_like(fluid_fluid_force[1], dtype=self.precisionPolicy.compute_dtype)
        if inter_component_interaction:
            F_1 = (
                -g_AB
                * rho_tree[0]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[1], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
            # F_2 = F_1
            F_2 = (
                -g_AB
                * rho_tree[1]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[0], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
            # F_1 = F_2
            F_tree = [-F_1, -F_2]
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
            # C = C.at[..., 8].set(eta * self.lattice.cs2)

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
        p_tree[1] = rho_tree[1] * self.lattice.cs2
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] + g_AB * rho_tree[0] * rho_tree[1]

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        Y_vapor = rho_water / (rho_water + rho_air)
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "Y_vapor": Y_vapor[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        save_fields_vtk(timestep, fields, "output", "data")


class StefansProblem3D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        _, y, _ = np.meshgrid(np.linspace(0, self.nx - 1, self.nx, dtype=int), y, np.linspace(0, self.nz - 1, self.nz, dtype=int))
        y = np.transpose(y, axes=(1, 0, 2))
        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (y - L) / width)
        rho[:, self.ny - 1, ...] = rho_w_g
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho[..., 0] = 0.5 * (rho_a_l + rho_a_g) - 0.5 * (rho_a_l - rho_a_g) * np.tanh(2 * (y - L) / width)
        rho[:, self.ny - 1, ...] = rho_a_g
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Define indices, used later
        top = self.boundingBoxIndices["back"]
        # top = np.array([[x, self.ny - 1, z] for x in range(self.nx) for z in range(self.nz)], dtype=int)
        rho = rho_w_g * np.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))
        rho = rho_a_g * np.ones((top.shape[0], 1))
        self.BCs[1].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.gridInfo, self.precisionPolicy, rho))

        bottom = self.boundingBoxIndices["front"]
        # bottom = np.array([[x, 0, z] for x in range(self.nx) for z in range(self.nz)], dtype=int)
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))
        self.BCs[1].append(BounceBack(tuple(bottom.T), self.gridInfo, self.precisionPolicy))

        # self.ind =
        # self.ind_nbr = self.BCs[0][1].indices_nbr
        # self.imissing = self.BCs[0][1].imissing
        # self.w_NEQ = self.BCs[0][1].w_NEQ

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

        # Handle eNEQ at PostStreaming using post-stream fields and BC-provided indices/weights
        if implementation_step == "PostStreaming":
            # Use post-streaming distributions to compute macroscopic fields and neighbor non-equilibrium
            rho_tree, _ = self.update_macroscopic(fin_tree)
            u_tree = self.macroscopic_velocity(fin_tree, rho_tree)
            u_mixture = self.compute_total_velocity(rho_tree, u_tree)

            def eNEQ(fout, rho, u, prescribed):
                # Neighbor values from post-streaming
                f_nbr = fout[self.ind_nbr]
                rho_nbr = rho[self.ind_nbr]
                u_nbr = u[self.ind_nbr]
                feq_nbr = self.equilibrium(rho_nbr, u_nbr)
                fneq_nbr = f_nbr - feq_nbr

                # Prescribed density at boundary, velocity from interior neighbor
                feq_prescribed = self.equilibrium(prescribed, u_mixture[self.ind_nbr])

                # Set missing directions using eNEQ formula
                f_bdr = fout[self.ind]
                f_bdr = f_bdr.at[..., self.imissing].set(feq_prescribed[..., self.imissing] + fneq_nbr[..., self.imissing])

                # Correction: enforce boundary density by distributing error over missing dirs via NEQ weights
                rho_incorrect = jnp.sum(f_bdr, axis=-1, keepdims=True)
                beta = self.w_NEQ[self.imissing] * jnp.repeat(prescribed - rho_incorrect, axis=-1, repeats=3) / jnp.sum(self.w_NEQ[self.imissing])
                f_bdr = f_bdr.at[..., self.imissing].add(beta)

                fout = fout.at[self.ind].set(f_bdr)
                return fout

            fout_tree = map(
                lambda fout, rho, u, prescribed: eNEQ(fout, rho, u, prescribed),
                fout_tree,
                rho_tree,
                u_tree,
                [self.BCs[0][0].prescribed, self.BCs[1][0].prescribed],
            )

        return fout_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G), self.k, p_tree, rho_tree, self.g_kkprime.diagonal().tolist()
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
                g_AB
                * rho_tree[0]
                * jnp.dot(
                    self.G_ff * self.streaming(jnp.repeat(rho_tree[1], repeats=self.lattice.q, axis=-1)),
                    self.precisionPolicy.cast_to_compute(self.lattice.c.T),
                )
            )
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
        p_tree[1] = rho_tree[1] * self.lattice.cs2
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] - g_AB * rho_tree[0] * rho_tree[1]

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho_water": rho_water[..., 0], "rho_air": rho_air[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        save_fields_vtk(timestep, fields, "output", "data")


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

    g_AB = 0.15

    precision = "f32/f32"

    # Single component
    # nx = 20
    # ny = 500
    # width = 8
    # rho_w_l = 6.499210784
    # rho_w_g = 0.379598891
    # # initial liquid region width
    # L = 200
    # s2 = 1.4  # This sets the kinematic viscosity
    # s_0 = [1.0]  # Mass conservation
    # s_1 = [1.0]  # Fixed
    # s_2 = [s2]
    # s_b = [0.8]
    # s_3 = [0.8]  # [(16 - 8 * s2) / (8 - s2)]  # No slip
    # s_4 = [1.0]  # No slip
    #
    # g_kkprime = -1 * np.ones((1, 1))
    # a = [3 / 49]
    # b = [2 / 21]
    # R = [1.0]
    # Tc = 0.1093785558
    # T = 0.86 * Tc
    #
    # kwargs = {"a": a, "b": b, "pr_omega": [0.344], "R": R, "T": T}
    # eos = Peng_Robinson(**kwargs)
    # kwargs = {
    #     "n_components": 1,
    #     "lattice": LatticeD2Q9(precision),
    #     "nx": nx,
    #     "ny": ny,
    #     "nz": 0,
    #     "g_kkprime": g_kkprime,
    #     "body_force": [0.0],
    #     "omega": [s2],
    #     "EOS": eos,
    #     "k": [1.0],
    #     "A": np.zeros((1, 1)),
    #     "M": [M],
    #     "s_0": s_0,
    #     "s_1": s_1,
    #     "s_b": s_b,
    #     "s_2": s_2,
    #     "s_3": s_3,
    #     "s_4": s_4,
    #     "sigma": [0.099],
    #     "precision": precision,
    #     "io_rate": 1000,
    #     "print_info_rate": 1000,
    #     "compute_MLUPS": False,
    #     "checkpoint_rate": -1,
    #     "checkpoint_dir": os.path.abspath("./checkpoints_"),
    #     "restore_checkpoint": False,
    # }
    # os.system("rm -rf output*/ *.vtk")
    # sim = SingleComponent(**kwargs)
    # sim.run(2)

    # Isothermal Stefan's Problem benchmark in 2D
    Y_air_liquid = 0.0001  # Fixed
    Y_air_1 = 0.01
    Y_air_vapor = 0.5  # so rho_water_g = rho_air_g
    nx = 10
    ny = 100
    width = 4
    rho_sat_l = 6.629589978
    rho_sat_g = 0.341237545
    rho_w_l = rho_sat_l
    rho_w_g = 0.05829998943954706
    rho_a_l = rho_sat_l * Y_air_liquid / (1 - Y_air_liquid)
    rho_a_g = 0.05829998943954706  # rho_sat_g * Y_air_1 / (1 - Y_air_1)
    # initial liquid region width
    L = 40

    a = [3 / 49, 1.0]
    b = [2 / 21, 1.0]
    R = [1.0, 1.0]
    Tc = 0.1093785558
    T = 0.85 * Tc

    kwargs = {"a": a, "b": b, "pr_omega": [0.344, 1], "R": R, "T": T}
    eos = Peng_Robinson(**kwargs)

    s2 = 1.25  # This sets the kinematic viscosity
    s_0 = [1.0, 1.0]  # Mass conservation
    s_1 = [1.0, 1.0]  # Fixed
    s_2 = [s2, s2]
    s_b = [0.8, 0.8]
    s_3 = [(16 - 8 * s2) / (8 - s2), (16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [1.0, 1.0]  # No slip

    g_kkprime = -1 * np.ones((2, 2))
    g_kkprime[1, 1] = 0
    g_kkprime[0, 1] = 0
    g_kkprime[1, 0] = 0

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
        "io_rate": 1,
        "print_info_rate": 1,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    os.system("rm -rf output*/ *.vtk")
    sim = StefansProblem2D(**kwargs)
    sim.run(10)

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
    #
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
