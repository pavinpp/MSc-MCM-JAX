import os

import numpy as np
import jax.numpy as jnp

from src.lattice import LatticeD2Q9
from src.multiphase import MultiphaseMRT
from src.eos import Carnahan_Starling
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


class CapillaryRise2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(
            2 * (y - L) / width
        )
        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[:, :, 0] = rho_profile

        rho = self.distributed_array_init(
            (self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
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
        left_wall = np.array(
            [
                [self.nx // 2 - channel_width // 2, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )
        right_wall = np.array(
            [
                [self.nx // 2 + channel_width // 2, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )
        outside_left_wall = np.array(
            [
                [nx // 2 - channel_width // 2 - 1, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )
        outside_right_wall = np.array(
            [
                [nx // 2 + channel_width // 2 + 1, i + offset]
                for i in range(channel_height)
            ],
            dtype=np.int32,
        )
        walls = np.concatenate(
            (
                left_wall,
                right_wall,
                outside_left_wall,
                outside_right_wall,
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
            )
        )
        self.BCs[0].append(
            BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy)
        )

    def get_force(self):
        """
        Gravity force
        """
        return jnp.array(
            np.array([0.0, -1e-4]),
            dtype=self.precisionPolicy.compute_dtype,
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_prev_tree"][0][0, :, 1:-1, :])
        p = np.array(kwargs["p_tree"][0][:, 1:-1, :])
        u = np.array(kwargs["u_tree"][0][0, :, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "flag": self.solid_mask_streamed[:, 1:-1, 0],
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
    nx = 200
    ny = 120

    channel_width = 8
    channel_height = 60
    offset = 20  # Distance of channel bottom from domain bottom
    L = ny // 3  # 1/L of domain is filled with liquid, rest is vapor

    width = 5  # Initial Liquid-vapor interface thickness

    a = 1.0
    b = 4.0
    R = 1.0

    rho_g = 0.000626568
    rho_l = 0.454078426
    Tc = 0.0943287031
    T = 0.5 * Tc

    kwargs = {
        "a": a,
        "b": b,
        "R": R,
        "T": T,
    }
    eos = Carnahan_Starling(**kwargs)

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
    s_e = [1.3]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    # Define contact angle matrix: I do not want any contact angle at the domain top and bottom, which are defined as walls.
    theta = 30 * (np.pi / 180) * np.ones((nx, ny, 1))
    theta[:, [0, ny - 1], 0] = 180 * (np.pi / 180)
    # channel has thickness of 2 lattice units. Outer channel walls are set as neutral wetting to prevent fluid climbing on outside
    outside_left_wall = np.array(
        [[nx // 2 - channel_width // 2 - 1, i + offset] for i in range(channel_height)],
        dtype=np.int32,
    )
    outside_right_wall = np.array(
        [[nx // 2 + channel_width // 2 + 1, i + offset] for i in range(channel_height)],
        dtype=np.int32,
    )
    theta[outside_left_wall] = 180 * (np.pi / 180)
    theta[outside_right_wall] = 180 * (np.pi / 180)

    # Same goes for phi
    phi = 1.4 * np.ones((nx, ny, 1))
    phi[:, [0, ny - 1], 0] = 1.0
    phi[outside_left_wall] = 1.0
    phi[outside_right_wall] = 1.0

    # This is not used,
    delta_rho = np.zeros((nx, ny, 1))

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "omega": [1.0],
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "body_force": [0.0, -0.95e-4],
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "k": [0.01],
        "A": 0.0 * np.ones((1, 1)),
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": [1.0],
        "M": [M],
        "theta": [theta],
        "phi": [phi],
        "delta_rho": [delta_rho],
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
    sim.run(9000)
