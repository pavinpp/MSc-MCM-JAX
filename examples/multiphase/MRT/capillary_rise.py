"""
Single component 2D capillary rise example where a channel is initially submerged in liquid. The density of each region is computed using Maxwell's Construction. The density profile is initialized with smooth profile with specified
interface thickness.
"""

import os

import numpy as np

from src.lattice import LatticeD2Q9
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk
from src.eos import VanderWaal


# Estimate surface tension
class Droplet2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        return rho_tree, u_tree

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
        offset_x = 300
        offset_y = 225
        rho_north = rho[self.nx // 2, self.ny // 2 - offset_y, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset_y, 0]
        rho_west = rho[self.nx // 2 - offset_x, self.ny // 2, 0]
        rho_east = rho[self.nx // 2 + offset_x, self.ny // 2, 0]
        rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
        rho_l_pred = rho[self.nx // 2, self.ny // 2, 0]
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset_y, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset_y, 0]
        p_west = p[self.nx // 2 - offset_x, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset_x, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference: {pressure_difference}")
        if timestep == 30000:
            file.write(f"{r},{pressure_difference}\n")
        save_fields_vtk(
            timestep,
            fields,
            f"output_{r}",
            "data",
        )


# Estimate contact angle
class DropletOnSurface2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y + disp) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

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
        save_fields_vtk(
            timestep,
            fields,
            f"output_{disp}",
            "data",
        )


class CapillaryRise2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (y - L) / width)
        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[:, :, 0] = rho_profile

        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = []
        u_tree.append(u)

        return rho_tree, u_tree

    def set_boundary_conditions(self):
        left_wall = np.array(
            [[self.nx // 2 - channel_width // 2, i + offset] for i in range(channel_height)],
            dtype=np.int32,
        )
        right_wall = np.array(
            [[self.nx // 2 + channel_width // 2, i + offset] for i in range(channel_height)],
            dtype=np.int32,
        )
        walls = np.concatenate((
            left_wall,
            right_wall,
            self.boundingBoxIndices["top"],
            self.boundingBoxIndices["bottom"],
        ))
        walls = tuple(walls.T)
        self.BCs[0].append(
            BounceBack(
                walls,
                self.gridInfo,
                self.precisionPolicy,
                theta[walls],
                phi[walls],
                delta_rho[walls],
            )
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_prev_tree"][0][0, :, 1:-1, :])
        p = np.array(kwargs["p_tree"][0][:, 1:-1, :])
        u = np.array(kwargs["u_tree"][0][0, :, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "flag": self.solid_mask_streamed[0][:, 1:-1, 0],
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        print(f"Max velocity: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        save_fields_vtk(
            timestep,
            fields,
            "output_",
            "data",
        )


if __name__ == "__main__":
    nx = 800
    ny = 500
    width = 3  # Initial Liquid-vapor interface thickness

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

    s_rho = [0.0]
    s_e = [0.5]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    a = 9 / 49
    b = 2 / 21
    R = 1.0

    rho_l = 6.764470400
    rho_g = 0.838834226
    Tc = 0.5714285714
    T = 0.8 * Tc

    kwargs = {
        "a": a,
        "b": b,
        "R": R,
        "T": T,
    }
    eos = VanderWaal(**kwargs)

    # Estimate surface tension
    file = open("surface_tension.txt", "w")
    file.write("Radius,Pressure Difference\n")
    for r in [75, 100, 125, 150, 175]:
        precision = "f32/f32"
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "omega": [1.0],
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "body_force": [0.0, 0.0],
            "g_kkprime": -1 * np.ones((1, 1)),
            "k": [0.16],
            "A": -0.032 * np.ones((1, 1)),
            "M": [M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0.0],
            "precision": precision,
            "EOS": eos,
            "io_rate": 10000,
            "print_info_rate": 10000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        sim = Droplet2D(**kwargs)
        sim.run(30000)
    file.close()

    theta = 30 * (np.pi / 180) * np.ones((nx, ny, 1))
    phi = 1.14 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    # Estimate contact angle
    r = 200
    Disp = [0, 25, 50, 75, 100, 125, 150, 175]
    for disp in Disp:
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "omega": [1.0],
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "body_force": [0.0, -2e-8],
            "g_kkprime": -1 * np.ones((1, 1)),
            "k": [0.16],
            "A": -0.032 * np.ones((1, 1)),
            "M": [M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0.0],
            "precision": precision,
            "io_rate": 1000,
            "eos": eos,
            "print_info_rate": 1000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        os.system("rm -rf output* *.vtk")
        sim = DropletOnSurface2D(**kwargs)
        sim.run(10000)

    channel_width = 36
    channel_height = 600
    offset = 100  # Distance of channel bottom from domain bottom
    L = offset - 20  # 1/L of domain is filled with liquid, rest is vapor

    # Define contact angle matrix: I do not want any contact angle at the domain top and bottom, which are defined as walls.
    theta = 30 * (np.pi / 180) * np.ones((nx, ny, 1))
    theta[:, [0, ny - 1], 0] = 90 * (np.pi / 180)
    phi = 1.14 * np.ones((nx, ny, 1))
    phi[:, [0, ny - 1], 0] = 1.0
    delta_rho = np.zeros((nx, ny, 1))

    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "omega": [1.0],
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "body_force": [0.0, -2e-8],
        "g_kkprime": -120.0 * np.ones((1, 1)),
        "k": [0.16],
        "A": -0.032 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0.0],
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    os.system("rm -rf output* *.vtk")
    sim = CapillaryRise2D(**kwargs)
    sim.run(50000)
