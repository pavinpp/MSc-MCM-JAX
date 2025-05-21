"""
Implementation of thermal LBM for arbitrary collision model
"""

from src.base import LBMBase
from src.multiphase import Multiphase
from src.utils import downsample_field


from functools import partial
import jax
from jax import jit
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from termcolor import colored
import time


# Single phase thermal LBM
class Thermal(LBMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fluid_solver = kwargs.get("fluid_solver")

    def set_thermal_boundary_conditions(self):
        """
        This function sets the boundary conditions for thermal simulation only.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        return

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.thermal_BCs = []
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        print("Time to create the grid mask for thermal lattice:", time.time() - start)

        start = time.time()
        for bc in self.thermal_BCs:
            assert bc.implementationStep in ["PostStreaming", "PostCollision"]
            bc.create_local_mask_and_normal_arrays(grid_mask)
        print("Time to create the local masks and normal arrays for thermal lattice:", time.time() - start)

    @partial(jit, static_argnums=(0,))
    def compute_temperature(self, g):
        """
        Compute the temperature field from temperature distributions.

        Parameters
        ----------
        g: jax.numpy.ndarray
            Temperature distribution.
        """
        return jnp.sum(g, axis=-1, keepdims=True)

    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2))
    def thermal_collision(self, gin, u):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        gin: jax.numpy.ndarray
            The pre-collision distribution functions.
        u: jax.numpy.ndarray
            The velocity field.

        Returns
        -------
        gin: jax.numpy.ndarray
            The post-collision distribution functions.
        """
        pass

    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, gout, gin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions for thermal LBM.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        gout: jax.numpy.ndarray
            The post-collision distribution functions.
        gin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        jax.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.thermal_BCs:
            gout = bc.prepare_populations(gout, gin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    gout = bc.apply(gout, gin, timestep)
                else:
                    gout = gout.at[bc.indices].set(bc.apply(gout, gin))

        return gout

    @partial(jit, static_argnums=(0,))
    def initialize_macroscopic_fields(self):
        """
        This function initializes the temperature distribution using prescribed temperature field and fluid velocities.
        The default temperature and density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None, None: The default temperature, density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: temperature = 1, density = 1, fluid velocity = 0")
        print("         To set explicit initial temperature, density and velocity, use self.initialize_macroscopic_fields.")
        return None, None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default
        value (self.w), representing temperature=1 and velocity=0. Otherwise, it uses the provided temperature and velocity to initialize the populations.

        Parameters
        ----------
        None

        Returns
        -------
        g: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the temperature distribution functions for the simulation.
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the fluid distribution functions for the simulation.
        """
        T0, rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)

        if T0 is None or rho0 is None or u0 is None:
            g = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=self.w)
            f = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=self.w)
        else:
            g = self.initialize_populations(T0, u0)
            f = self.initialize_populations(rho0, u0)

        return g, f

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, g_poststreaming, u, timestep, return_gpost=False):
        """
        This function performs a single step of the thermal LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions
        towards their equilibrium values. It then applies the respective boundary conditions to the
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        Parameters
        ----------
        g_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        u: jax.numpy.ndarray
            The velocity field.
        timestep: int
            The current timestep of the simulation.
        return_gpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if
            return_gpost is False.
        """
        g_postcollision = self.thermal_collision(g_poststreaming, u)
        g_postcollision = self.apply_bc(g_postcollision, g_poststreaming, timestep, "PostCollision")
        g_poststreaming = self.streaming(g_postcollision)
        g_poststreaming = self.apply_bc(g_poststreaming, g_postcollision, timestep, "PostStreaming")

        if return_gpost:
            return g_poststreaming, g_postcollision
        else:
            return g_poststreaming, None

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        Returns
        -------
        g: jax.numpy.ndarray
            The distribution functions for temperature after the simulation.
        """
        g, f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {"g": g, "f": f}
                # shardings = map(lambda x: x.sharding, state)
                # restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                try:
                    # f = self.mngr.restore(latest_step, restore_kwargs={'restore_args': restore_args})['f']
                    g = self.mngr.restore(latest_step, args=orb.args.StandardSave(state))["g"]
                    f = self.mngr.restore(latest_step, args=orb.args.StandardSave(state))["f"]
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")
        if self.computeMLUPS:
            start = time.time()
        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0
            checkpoint_flag = self.checkpointRate > 0 and timestep % self.checkpointRate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.fluid_solver.downsamplingFactor)
                u_prev = downsample_field(u_prev, self.fluid_solver.downsamplingFactor)
                T_prev = self.compute_temperature(g)
                T_prev = downsample_field(g, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)
                T_prev = process_allgather(T_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.fluid_solver.step(f, timestep, return_fpost=self.fluid_solver.returnFpost)
            rho, u = self.update_macroscopic(f)
            g, gstar = self.step(g, u, timestep, return_gpost=self.returnFpost)

            # Print the progress of the simulation
            if print_iter_flag:
                print(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                # rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.fluid_solver.downsamplingFactor)
                u = downsample_field(u, self.fluid_solver.downsamplingFactor)
                T = self.compute_temperature(g)
                T = downsample_field(T, self.downsamplingFactor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)
                T = process_allgather(T)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, g, gstar, T, rho, u, T_prev, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {"f": f}
                # self.mngr.save(timestep, state)
                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                jax.block_until_ready(g)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print(
                    colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(colored("MLUPS: ", "blue") + colored(f"{2 * self.nx * self.ny * t_max / (end - start) / 1e6}", "red"))

            elif self.dim == 3:
                print(colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny} x {self.nz}", "green"))
                print(colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny * self.nz}", "green"))
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{2 * self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f

    def handle_io_timestep(self, timestep, f, fstar, g, gstar, T, rho, u, T_prev, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        g: jax.numpy.ndarray
            The post-streaming distribution functions for temperature at the current time step.
        gstar: jax.numpy.ndarray
            The post-collision distribution functions for temperature at the current time step.
        rho: jax.numpy.ndarray
            The density field at the current time step.
        u: jax.numpy.ndarray
            The velocity field at the current time step.
        T: jax.numpy.ndarray
            The temperature field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "T": T,
            "T_prev": T_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar,
            "g_poststreaming": g,
            "g_postcollision": gstar,
        }
        self.output_data(**kwargs)


class BGKSim(Thermal):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def thermal_collision(self, g, u):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        g = self.precisionPolicy.cast_to_compute(g)
        temperature = self.compute_temperature(g)
        geq = self.equilibrium(temperature, u, cast_output=False)
        gneq = g - geq
        gout = g - self.omega * gneq
        # if self.force is not None:
        #     fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(gout)


class MRTSim(Thermal):
    """
    Multi-relaxation time model.
    """

    def __init__(self, **kwargs):
        kwargs.update({"omega": 1.0})
        super().__init__(**kwargs)
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = kwargs.get("s_v")
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precisionPolicy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precisionPolicy.compute_dtype)
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_j, self.s_q, self.s_v, self.s_v]),
                dtype=self.precisionPolicy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = jnp.array(
                np.diag([
                    self.s_rho,
                    self.s_e,
                    self.s_eta,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_v,
                    self.s_v,
                    self.s_m,
                    self.s_m,
                    self.s_m,
                ]),
                dtype=self.precisionPolicy.compute_dtype,
            )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def thermal_collision(self, g, u):
        """
        MRT collision step for lattice.

        Parameters
        ----------
        g: jax.numpy.ndarray
            Temperature distribution.
        u: jax.numpy.ndarray
            Velocity field.
        """
        g = self.precisionPolicy.cast_to_compute(g)
        m = jnp.dot(g, self.M)
        rho, u = self.update_macroscopic(g)
        geq = self.equilibrium(rho, u)
        meq = jnp.dot(geq, self.M)
        mout = -jnp.dot(m - meq, self.S)
        if self.force is not None:
            mout = self.apply_force(mout, meq, rho, u)
        return self.precisionPolicy.cast_to_output(g + jnp.dot(mout, self.M_inv))


# TODO
# class CLBMSim(Thermal):
#     """
#     Central moment (cascaded) collision model
#     """
#
#     def __init__(self, **kwargs):
#         kwargs.update({"omega": 1.0})
#         super().__init__(**kwargs)
#         self.M_inv = jnp.array(
#             np.transpose(np.linalg.inv(kwargs.get("M"))),
#             dtype=self.precisionPolicy.compute_dtype,
#         )
#         self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precisionPolicy.compute_dtype)
#         self.s_0 = kwargs.get("s_0")
#         self.s_1 = kwargs.get("s_1")
#         self.s_b = kwargs.get("s_b")
#         self.s_2 = kwargs.get("s_2")
#         self.s_3 = kwargs.get("s_3")
#         self.s_4 = kwargs.get("s_4")
#         self.s_v = self.omega
#         if isinstance(self.lattice, LatticeD2Q9):
#             self.S = jnp.array(
#                 np.diag([self.s_0, self.s_1, self.s_1, self.s_b, self.s_2, self.s_2, self.s_3, self.s_3, self.s_4]),
#                 dtype=self.precisionPolicy.compute_dtype,
#             )
#         elif isinstance(self.lattice, LatticeD3Q19):
#             self.s_plus = (self.s_b + 2 * self.s_2) / 3
#             self.s_minus = (self.s_b - self.s_2) / 3
#
#             S = np.diag([
#                 self.s_0,
#                 self.s_1,
#                 self.s_1,
#                 self.s_1,
#                 self.s_v,
#                 self.s_v,
#                 self.s_v,
#                 self.s_plus,
#                 self.s_plus,
#                 self.s_plus,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_4,
#                 self.s_4,
#                 self.s_4,
#             ])
#             S[7, 8] = self.s_minus
#             S[7, 9] = self.s_minus
#             S[8, 7] = self.s_minus
#             S[8, 9] = self.s_minus
#             S[9, 7] = self.s_minus
#             S[9, 8] = self.s_minus
#             self.S = jnp.array(S, dtype=self.precisionPolicy.compute_dtype)
#
#         elif isinstance(self.lattice, LatticeD3Q27):
#             self.s_plus = (self.s_b + 2 * self.s_2) / 3
#             self.s_minus = (self.s_b - self.s_2) / 3
#             self.s_3b = kwargs.get("s_3b")
#             self.s_4b = kwargs.get("s_4b")
#             self.s_5 = kwargs.get("s_5")
#             self.s_6 = kwargs.get("s_6")
#
#             S = np.diag([
#                 self.s_0,
#                 self.s_1,
#                 self.s_1,
#                 self.s_1,
#                 self.s_v,
#                 self.s_v,
#                 self.s_v,
#                 self.s_plus,
#                 self.s_plus,
#                 self.s_plus,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3,
#                 self.s_3b,
#                 self.s_4,
#                 self.s_4,
#                 self.s_4,
#                 self.s_4b,
#                 self.s_4b,
#                 self.s_4b,
#                 self.s_5,
#                 self.s_5,
#                 self.s_5,
#                 self.s_6,
#             ])
#             S[7, 8] = self.s_minus
#             S[7, 9] = self.s_minus
#             S[8, 7] = self.s_minus
#             S[8, 9] = self.s_minus
#             S[9, 7] = self.s_minus
#             S[9, 8] = self.s_minus
#             self.S = jnp.array(S, dtype=self.precisionPolicy.compute_dtype)
#
#     @partial(jit, static_argnums=(0,), inline=True)
#     def macroscopic_velocity(self, f, rho):
#         """
#         macroscopic_velocity computes the velocity and incorporates forces into velocity for Exact Difference Method (EDM) (used for SRT and MRT collision) models
#         and the consistent forcing scheme developed by LinLin Fei et. al (for Cascaded LBM). This is used for post-processing only and not for equilibrium distribution computation.
#
#         Parameters
#         ----------
#         f: jax.numpy.ndarray
#             Distribution arrays.
#         rho: jax.numpy.ndarray
#             Density fields.
#
#         Returns
#         -------
#         u: jax.numpy.ndarray
#             Velocity fields.
#         """
#         # rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
#         c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
#         u = jnp.dot(f, c) / rho
#         if self.force is not None:
#             return u + 0.5 * self.force / rho
#         else:
#             return u
#
#     @partial(jit, static_argnums=(0,))
#     def compute_central_moment(self, m, u):
#         if isinstance(self.lattice, LatticeD2Q9):
#
#             def shift(m, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 usq = ux**2 + uy**2
#                 udiff = ux**2 - uy**2
#                 T = jnp.zeros_like(m)
#                 T = T.at[..., 0].set(m[..., 0])
#                 T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
#                 T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
#                 T = T.at[..., 3].set(usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3])
#                 T = T.at[..., 4].set(udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4])
#                 T = T.at[..., 5].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5])
#                 T = T.at[..., 6].set(
#                     -(ux**2) * uy * m[..., 0]
#                     + 2 * ux * uy * m[..., 1]
#                     + ux**2 * m[..., 2]
#                     - 0.5 * uy * m[..., 3]
#                     - 0.5 * uy * m[..., 4]
#                     - 2 * ux * m[..., 5]
#                     + m[..., 6]
#                 )
#                 T = T.at[..., 7].set(
#                     -(uy**2) * ux * m[..., 0]
#                     + uy**2 * m[..., 1]
#                     + 2 * ux * uy * m[..., 2]
#                     - 0.5 * ux * m[..., 3]
#                     + 0.5 * ux * m[..., 4]
#                     - 2 * uy * m[..., 5]
#                     + m[..., 7]
#                 )
#                 T = T.at[..., 8].set(
#                     (uy**2 * ux**2) * m[..., 0]
#                     - 2 * ux * uy**2 * m[..., 1]
#                     - 2 * uy * ux**2 * m[..., 2]
#                     + 0.5 * usq * m[..., 3]
#                     - 0.5 * udiff * m[..., 4]
#                     + 4 * ux * uy * m[..., 5]
#                     - 2 * uy * m[..., 6]
#                     - 2 * ux * m[..., 7]
#                     + m[..., 8]
#                 )
#                 return T
#
#             return shift(m, u)
#
#         elif isinstance(self.lattice, LatticeD3Q19):
#
#             def shift(m, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 uz = u[..., 2]
#                 T = jnp.zeros_like(m)
#                 T = T.at[..., 0].set(m[..., 0])
#                 T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
#                 T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
#                 T = T.at[..., 3].set(-uz * m[..., 0] + m[..., 3])
#                 T = T.at[..., 4].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 4])
#                 T = T.at[..., 5].set(ux * uz * m[..., 0] - uz * m[..., 1] - ux * m[..., 3] + m[..., 5])
#                 T = T.at[..., 6].set(uy * uz * m[..., 0] - uz * m[..., 2] - uy * m[..., 3] + m[..., 6])
#                 T = T.at[..., 7].set((ux**2) * m[..., 0] - 2 * ux * m[..., 1] + m[..., 7])
#                 T = T.at[..., 8].set((uy**2) * m[..., 0] - 2 * uy * m[..., 2] + m[..., 8])
#                 T = T.at[..., 9].set((uz**2) * m[..., 0] - 2 * uz * m[..., 3] + m[..., 9])
#                 T = T.at[..., 10].set(
#                     -ux * (uy**2) * m[..., 0] + (uy**2) * m[..., 1] + 2 * ux * uy * m[..., 2] - 2 * uy * m[..., 4] - ux * m[..., 8] + m[..., 10]
#                 )
#                 T = T.at[..., 11].set(
#                     -ux * (uz**2) * m[..., 0] + (uz**2) * m[..., 1] + 2 * ux * uz * m[..., 3] - 2 * uz * m[..., 5] - ux * m[..., 9] + m[..., 11]
#                 )
#                 T = T.at[..., 12].set(
#                     -(ux**2) * uy * m[..., 0] + 2 * ux * uy * m[..., 1] + (ux**2) * m[..., 2] - 2 * ux * m[..., 4] - uy * m[..., 7] + m[..., 12]
#                 )
#                 T = T.at[..., 13].set(
#                     -(ux**2) * uz * m[..., 0] + 2 * ux * uz * m[..., 1] + (ux**2) * m[..., 3] - 2 * ux * m[..., 5] - uz * m[..., 7] + m[..., 13]
#                 )
#                 T = T.at[..., 14].set(
#                     -uy * (uz**2) * m[..., 0] + (uz**2) * m[..., 2] + 2 * uy * uz * m[..., 3] - 2 * uz * m[..., 6] - uy * m[..., 9] + m[..., 14]
#                 )
#                 T = T.at[..., 15].set(
#                     -(uy**2) * uz * m[..., 0] + 2 * uy * uz * m[..., 2] + (uy**2) * m[..., 3] - 2 * uy * m[..., 6] - uz * m[..., 8] + m[..., 15]
#                 )
#                 T = T.at[..., 16].set(
#                     (ux**2) * (uy**2) * m[..., 0]
#                     - 2 * ux * (uy**2) * m[..., 1]
#                     - 2 * uy * (ux**2) * m[..., 2]
#                     + 4 * ux * uy * m[..., 4]
#                     + (uy**2) * m[..., 7]
#                     + (ux**2) * m[..., 8]
#                     - 2 * ux * m[..., 10]
#                     - 2 * uy * m[..., 12]
#                     + m[..., 16]
#                 )
#                 T = T.at[..., 17].set(
#                     (ux**2) * (uz**2) * m[..., 0]
#                     - 2 * ux * (uz**2) * m[..., 1]
#                     - 2 * uz * (ux**2) * m[..., 3]
#                     + 4 * ux * uz * m[..., 5]
#                     + (uz**2) * m[..., 7]
#                     + (ux**2) * m[..., 9]
#                     - 2 * ux * m[..., 11]
#                     - 2 * uz * m[..., 13]
#                     + m[..., 17]
#                 )
#                 T = T.at[..., 18].set(
#                     (uy**2) * (uz**2) * m[..., 0]
#                     - 2 * uy * (uz**2) * m[..., 2]
#                     - 2 * uz * (uy**2) * m[..., 3]
#                     + 4 * uy * uz * m[..., 6]
#                     + (uz**2) * m[..., 8]
#                     + (uy**2) * m[..., 9]
#                     - 2 * uy * m[..., 14]
#                     - 2 * uz * m[..., 15]
#                     + m[..., 18]
#                 )
#                 return T
#
#             return shift(m, u)
#
#         elif isinstance(self.lattice, LatticeD3Q27):
#
#             def shift(m, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 uz = u[..., 2]
#                 T = jnp.zeros_like(m)
#                 T = T.at[..., 0].set(m[..., 0])
#                 T = T.at[..., 1].set(m[..., 1] - m[..., 0] * ux)
#                 T = T.at[..., 2].set(m[..., 2] - m[..., 0] * uy)
#                 T = T.at[..., 3].set(m[..., 3] - m[..., 0] * uz)
#                 T = T.at[..., 4].set(m[..., 4] - m[..., 2] * ux - m[..., 1] * uy + m[..., 0] * ux * uy)
#                 T = T.at[..., 5].set(m[..., 5] - m[..., 3] * ux - m[..., 1] * uz + m[..., 0] * ux * uz)
#                 T = T.at[..., 6].set(m[..., 6] - m[..., 3] * uy - m[..., 2] * uz + m[..., 0] * uy * uz)
#                 T = T.at[..., 7].set(m[..., 0] * ux**2 - 2 * m[..., 1] * ux + m[..., 7])
#                 T = T.at[..., 8].set(m[..., 0] * uy**2 - 2 * m[..., 2] * uy + m[..., 8])
#                 T = T.at[..., 9].set(m[..., 0] * uz**2 - 2 * m[..., 3] * uz + m[..., 9])
#                 T = T.at[..., 10].set(
#                     m[..., 10] - m[..., 8] * ux - 2 * m[..., 4] * uy + m[..., 1] * uy**2 - m[..., 0] * ux * uy**2 + 2 * m[..., 2] * ux * uy
#                 )
#                 T = T.at[..., 11].set(
#                     m[..., 11] - m[..., 9] * ux - 2 * m[..., 5] * uz + m[..., 1] * uz**2 - m[..., 0] * ux * uz**2 + 2 * m[..., 3] * ux * uz
#                 )
#                 T = T.at[..., 12].set(
#                     m[..., 12] - 2 * m[..., 4] * ux - m[..., 7] * uy + m[..., 2] * ux**2 - m[..., 0] * ux**2 * uy + 2 * m[..., 1] * ux * uy
#                 )
#                 T = T.at[..., 13].set(
#                     m[..., 13] - 2 * m[..., 5] * ux - m[..., 7] * uz + m[..., 3] * ux**2 - m[..., 0] * ux**2 * uz + 2 * m[..., 1] * ux * uz
#                 )
#                 T = T.at[..., 14].set(
#                     m[..., 14] - m[..., 9] * uy - 2 * m[..., 6] * uz + m[..., 2] * uz**2 - m[..., 0] * uy * uz**2 + 2 * m[..., 3] * uy * uz
#                 )
#                 T = T.at[..., 15].set(
#                     m[..., 15] - 2 * m[..., 6] * uy - m[..., 8] * uz + m[..., 3] * uy**2 - m[..., 0] * uy**2 * uz + 2 * m[..., 2] * uy * uz
#                 )
#                 T = T.at[..., 16].set(
#                     m[..., 16]
#                     - m[..., 6] * ux
#                     - m[..., 5] * uy
#                     - m[..., 4] * uz
#                     + m[..., 3] * ux * uy
#                     + m[..., 2] * ux * uz
#                     + m[..., 1] * uy * uz
#                     - m[..., 0] * ux * uy * uz
#                 )
#                 T = T.at[..., 17].set(
#                     m[..., 0] * ux**2 * uy**2
#                     - 2 * m[..., 2] * ux**2 * uy
#                     + m[..., 8] * ux**2
#                     - 2 * m[..., 1] * ux * uy**2
#                     + 4 * m[..., 4] * ux * uy
#                     - 2 * m[..., 10] * ux
#                     + m[..., 7] * uy**2
#                     - 2 * m[..., 12] * uy
#                     + m[..., 17]
#                 )
#                 T = T.at[..., 18].set(
#                     m[..., 0] * ux**2 * uz**2
#                     - 2 * m[..., 3] * ux**2 * uz
#                     + m[..., 9] * ux**2
#                     - 2 * m[..., 1] * ux * uz**2
#                     + 4 * m[..., 5] * ux * uz
#                     - 2 * m[..., 11] * ux
#                     + m[..., 7] * uz**2
#                     - 2 * m[..., 13] * uz
#                     + m[..., 18]
#                 )
#                 T = T.at[..., 19].set(
#                     m[..., 0] * uy**2 * uz**2
#                     - 2 * m[..., 3] * uy**2 * uz
#                     + m[..., 9] * uy**2
#                     - 2 * m[..., 2] * uy * uz**2
#                     + 4 * m[..., 6] * uy * uz
#                     - 2 * m[..., 14] * uy
#                     + m[..., 8] * uz**2
#                     - 2 * m[..., 15] * uz
#                     + m[..., 19]
#                 )
#                 T = T.at[..., 20].set(
#                     m[..., 20]
#                     - 2 * m[..., 16] * ux
#                     - m[..., 13] * uy
#                     - m[..., 12] * uz
#                     + m[..., 6] * ux**2
#                     - m[..., 3] * ux**2 * uy
#                     - m[..., 2] * ux**2 * uz
#                     + 2 * m[..., 5] * ux * uy
#                     + 2 * m[..., 4] * ux * uz
#                     + m[..., 7] * uy * uz
#                     - 2 * m[..., 1] * ux * uy * uz
#                     + m[..., 0] * ux**2 * uy * uz
#                 )
#                 T = T.at[..., 21].set(
#                     m[..., 21]
#                     - m[..., 15] * ux
#                     - 2 * m[..., 16] * uy
#                     - m[..., 10] * uz
#                     + m[..., 5] * uy**2
#                     - m[..., 3] * ux * uy**2
#                     - m[..., 1] * uy**2 * uz
#                     + 2 * m[..., 6] * ux * uy
#                     + m[..., 8] * ux * uz
#                     + 2 * m[..., 4] * uy * uz
#                     - 2 * m[..., 2] * ux * uy * uz
#                     + m[..., 0] * ux * uy**2 * uz
#                 )
#                 T = T.at[..., 22].set(
#                     m[..., 22]
#                     - m[..., 14] * ux
#                     - m[..., 11] * uy
#                     - 2 * m[..., 16] * uz
#                     + m[..., 4] * uz**2
#                     - m[..., 2] * ux * uz**2
#                     - m[..., 1] * uy * uz**2
#                     + m[..., 9] * ux * uy
#                     + 2 * m[..., 6] * ux * uz
#                     + 2 * m[..., 5] * uy * uz
#                     - 2 * m[..., 3] * ux * uy * uz
#                     + m[..., 0] * ux * uy * uz**2
#                 )
#                 T = T.at[..., 23].set(
#                     m[..., 23]
#                     - m[..., 19] * ux
#                     - 2 * m[..., 22] * uy
#                     - 2 * m[..., 21] * uz
#                     + m[..., 11] * uy**2
#                     + m[..., 10] * uz**2
#                     - m[..., 9] * ux * uy**2
#                     - m[..., 8] * ux * uz**2
#                     - 2 * m[..., 4] * uy * uz**2
#                     - 2 * m[..., 5] * uy**2 * uz
#                     + m[..., 1] * uy**2 * uz**2
#                     + 2 * m[..., 14] * ux * uy
#                     + 2 * m[..., 15] * ux * uz
#                     + 4 * m[..., 16] * uy * uz
#                     - 4 * m[..., 6] * ux * uy * uz
#                     + 2 * m[..., 2] * ux * uy * uz**2
#                     + 2 * m[..., 3] * ux * uy**2 * uz
#                     - m[..., 0] * ux * uy**2 * uz**2
#                 )
#                 T = T.at[..., 24].set(
#                     m[..., 24]
#                     - 2 * m[..., 22] * ux
#                     - m[..., 18] * uy
#                     - 2 * m[..., 20] * uz
#                     + m[..., 14] * ux**2
#                     + m[..., 12] * uz**2
#                     - m[..., 9] * ux**2 * uy
#                     - 2 * m[..., 4] * ux * uz**2
#                     - 2 * m[..., 6] * ux**2 * uz
#                     - m[..., 7] * uy * uz**2
#                     + m[..., 2] * ux**2 * uz**2
#                     + 2 * m[..., 11] * ux * uy
#                     + 4 * m[..., 16] * ux * uz
#                     + 2 * m[..., 13] * uy * uz
#                     - 4 * m[..., 5] * ux * uy * uz
#                     + 2 * m[..., 1] * ux * uy * uz**2
#                     + 2 * m[..., 3] * ux**2 * uy * uz
#                     - m[..., 0] * ux**2 * uy * uz**2
#                 )
#                 T = T.at[..., 25].set(
#                     m[..., 25]
#                     - 2 * m[..., 21] * ux
#                     - 2 * m[..., 20] * uy
#                     - m[..., 17] * uz
#                     + m[..., 15] * ux**2
#                     + m[..., 13] * uy**2
#                     - 2 * m[..., 5] * ux * uy**2
#                     - 2 * m[..., 6] * ux**2 * uy
#                     - m[..., 8] * ux**2 * uz
#                     - m[..., 7] * uy**2 * uz
#                     + m[..., 3] * ux**2 * uy**2
#                     + 4 * m[..., 16] * ux * uy
#                     + 2 * m[..., 10] * ux * uz
#                     + 2 * m[..., 12] * uy * uz
#                     - 4 * m[..., 4] * ux * uy * uz
#                     + 2 * m[..., 1] * ux * uy**2 * uz
#                     + 2 * m[..., 2] * ux**2 * uy * uz
#                     - m[..., 0] * ux**2 * uy**2 * uz
#                 )
#                 T = T.at[..., 26].set(
#                     m[..., 0] * ux**2 * uy**2 * uz**2
#                     - 2 * m[..., 3] * ux**2 * uy**2 * uz
#                     + m[..., 9] * ux**2 * uy**2
#                     - 2 * m[..., 2] * ux**2 * uy * uz**2
#                     + 4 * m[..., 6] * ux**2 * uy * uz
#                     - 2 * m[..., 14] * ux**2 * uy
#                     + m[..., 8] * ux**2 * uz**2
#                     - 2 * m[..., 15] * ux**2 * uz
#                     + m[..., 19] * ux**2
#                     - 2 * m[..., 1] * ux * uy**2 * uz**2
#                     + 4 * m[..., 5] * ux * uy**2 * uz
#                     - 2 * m[..., 11] * ux * uy**2
#                     + 4 * m[..., 4] * ux * uy * uz**2
#                     - 8 * m[..., 16] * ux * uy * uz
#                     + 4 * m[..., 22] * ux * uy
#                     - 2 * m[..., 10] * ux * uz**2
#                     + 4 * m[..., 21] * ux * uz
#                     - 2 * m[..., 23] * ux
#                     + m[..., 7] * uy**2 * uz**2
#                     - 2 * m[..., 13] * uy**2 * uz
#                     + m[..., 18] * uy**2
#                     - 2 * m[..., 12] * uy * uz**2
#                     + 4 * m[..., 20] * uy * uz
#                     - 2 * m[..., 24] * uy
#                     + m[..., 17] * uz**2
#                     - 2 * m[..., 25] * uz
#                     + m[..., 26]
#                 )
#
#                 return T
#
#             return shift(m, u)
#
#     @partial(jit, static_argnums=(0,))
#     def compute_central_moment_inverse(self, T, u):
#         if isinstance(self.lattice, LatticeD2Q9):
#
#             def shift_inverse(T, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 usq = ux**2 + uy**2
#                 udiff = ux**2 - uy**2
#                 m = jnp.zeros_like(T)
#                 m = m.at[..., 0].set(T[..., 0])
#                 m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
#                 m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
#                 m = m.at[..., 3].set(usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3])
#                 m = m.at[..., 4].set(udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4])
#                 m = m.at[..., 5].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5])
#                 m = m.at[..., 6].set(
#                     (ux**2) * uy * T[..., 0]
#                     + 2 * ux * uy * T[..., 1]
#                     + ux**2 * T[..., 2]
#                     + 0.5 * uy * T[..., 3]
#                     + 0.5 * uy * T[..., 4]
#                     + 2 * ux * T[..., 5]
#                     + T[..., 6]
#                 )
#                 m = m.at[..., 7].set(
#                     (uy**2) * ux * T[..., 0]
#                     + uy**2 * T[..., 1]
#                     + 2 * ux * uy * T[..., 2]
#                     + 0.5 * ux * T[..., 3]
#                     - 0.5 * ux * T[..., 4]
#                     + 2 * uy * T[..., 5]
#                     + T[..., 7]
#                 )
#                 m = m.at[..., 8].set(
#                     (uy**2 * ux**2) * T[..., 0]
#                     + 2 * ux * uy**2 * T[..., 1]
#                     + 2 * uy * ux**2 * T[..., 2]
#                     + 0.5 * usq * T[..., 3]
#                     - 0.5 * udiff * T[..., 4]
#                     + 4 * ux * uy * T[..., 5]
#                     + 2 * uy * T[..., 6]
#                     + 2 * ux * T[..., 7]
#                     + T[..., 8]
#                 )
#                 return m
#
#             return shift_inverse(T, u)
#
#         elif isinstance(self.lattice, LatticeD3Q19):
#
#             def shift_inverse(T, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 uz = u[..., 2]
#                 m = jnp.zeros_like(T)
#                 m = m.at[..., 0].set(T[..., 0])
#                 m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
#                 m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
#                 m = m.at[..., 3].set(uz * T[..., 0] + T[..., 3])
#                 m = m.at[..., 4].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 4])
#                 m = m.at[..., 5].set(ux * uz * T[..., 0] + uz * T[..., 1] + ux * T[..., 3] + T[..., 5])
#                 m = m.at[..., 6].set(uy * uz * T[..., 0] + uz * T[..., 2] + uy * T[..., 3] + T[..., 6])
#                 m = m.at[..., 7].set((ux**2) * T[..., 0] + 2 * ux * T[..., 1] + T[..., 7])
#                 m = m.at[..., 8].set((uy**2) * T[..., 0] + 2 * uy * T[..., 2] + T[..., 8])
#                 m = m.at[..., 9].set((uz**2) * T[..., 0] + 2 * uz * T[..., 3] + T[..., 9])
#                 m = m.at[..., 10].set(
#                     ux * (uy**2) * T[..., 0] + (uy**2) * T[..., 1] + 2 * ux * uy * T[..., 2] + 2 * uy * T[..., 4] + ux * T[..., 8] + T[..., 10]
#                 )
#                 m = m.at[..., 11].set(
#                     ux * (uz**2) * T[..., 0] + (uz**2) * T[..., 1] + 2 * ux * uz * T[..., 3] + 2 * uz * T[..., 5] + ux * T[..., 9] + T[..., 11]
#                 )
#                 m = m.at[..., 12].set(
#                     (ux**2) * uy * T[..., 0] + 2 * ux * uy * T[..., 1] + (ux**2) * T[..., 2] + 2 * ux * T[..., 4] + uy * T[..., 7] + T[..., 12]
#                 )
#                 m = m.at[..., 13].set(
#                     (ux**2) * uz * T[..., 0] + 2 * ux * uz * T[..., 1] + (ux**2) * T[..., 3] + 2 * ux * T[..., 5] + uz * T[..., 7] + T[..., 13]
#                 )
#                 m = m.at[..., 14].set(
#                     uy * (uz**2) * T[..., 0] + (uz**2) * T[..., 2] + 2 * uy * uz * T[..., 3] + 2 * uz * T[..., 6] + uy * T[..., 9] + T[..., 14]
#                 )
#                 m = m.at[..., 15].set(
#                     (uy**2) * uz * T[..., 0] + 2 * uy * uz * T[..., 2] + (uy**2) * T[..., 3] + 2 * uy * T[..., 6] + uz * T[..., 8] + T[..., 15]
#                 )
#                 m = m.at[..., 16].set(
#                     (ux**2) * (uy**2) * T[..., 0]
#                     + 2 * ux * (uy**2) * T[..., 1]
#                     + 2 * uy * (ux**2) * T[..., 2]
#                     + 4 * ux * uy * T[..., 4]
#                     + (uy**2) * T[..., 7]
#                     + (ux**2) * T[..., 8]
#                     + 2 * ux * T[..., 10]
#                     + 2 * uy * T[..., 12]
#                     + T[..., 16]
#                 )
#                 m = m.at[..., 17].set(
#                     (ux**2) * (uz**2) * T[..., 0]
#                     + 2 * ux * (uz**2) * T[..., 1]
#                     + 2 * uz * (ux**2) * T[..., 3]
#                     + 4 * ux * uz * T[..., 5]
#                     + (uz**2) * T[..., 7]
#                     + (ux**2) * T[..., 9]
#                     + 2 * ux * T[..., 11]
#                     + 2 * uz * T[..., 13]
#                     + T[..., 17]
#                 )
#                 m = m.at[..., 18].set(
#                     (uy**2) * (uz**2) * T[..., 0]
#                     + 2 * uy * (uz**2) * T[..., 2]
#                     + 2 * uz * (uy**2) * T[..., 3]
#                     + 4 * uy * uz * T[..., 6]
#                     + (uz**2) * T[..., 8]
#                     + (uy**2) * T[..., 9]
#                     + 2 * uy * T[..., 14]
#                     + 2 * uz * T[..., 15]
#                     + T[..., 18]
#                 )
#                 return m
#
#             return shift_inverse(T, u)
#
#         elif isinstance(self.lattice, LatticeD3Q27):
#
#             def shift_inverse(T, u):
#                 ux = u[..., 0]
#                 uy = u[..., 1]
#                 uz = u[..., 2]
#                 m = jnp.zeros_like(T)
#                 m = m.at[..., 0].set(T[..., 0])
#                 m = m.at[..., 1].set(T[..., 1] + T[..., 0] * ux)
#                 m = m.at[..., 2].set(T[..., 2] + T[..., 0] * uy)
#                 m = m.at[..., 3].set(T[..., 3] + T[..., 0] * uz)
#                 m = m.at[..., 4].set(T[..., 4] + T[..., 2] * ux + T[..., 1] * uy + T[..., 0] * ux * uy)
#                 m = m.at[..., 5].set(T[..., 5] + T[..., 3] * ux + T[..., 1] * uz + T[..., 0] * ux * uz)
#                 m = m.at[..., 6].set(T[..., 6] + T[..., 3] * uy + T[..., 2] * uz + T[..., 0] * uy * uz)
#                 m = m.at[..., 7].set(T[..., 0] * ux**2 + 2 * T[..., 1] * ux + T[..., 7])
#                 m = m.at[..., 8].set(T[..., 0] * uy**2 + 2 * T[..., 2] * uy + T[..., 8])
#                 m = m.at[..., 9].set(T[..., 0] * uz**2 + 2 * T[..., 3] * uz + T[..., 9])
#                 m = m.at[..., 10].set(
#                     T[..., 10] + T[..., 8] * ux + 2 * T[..., 4] * uy + T[..., 1] * uy**2 + T[..., 0] * ux * uy**2 + 2 * T[..., 2] * ux * uy
#                 )
#                 m = m.at[..., 11].set(
#                     T[..., 11] + T[..., 9] * ux + 2 * T[..., 5] * uz + T[..., 1] * uz**2 + T[..., 0] * ux * uz**2 + 2 * T[..., 3] * ux * uz
#                 )
#                 m = m.at[..., 12].set(
#                     T[..., 12] + 2 * T[..., 4] * ux + T[..., 7] * uy + T[..., 2] * ux**2 + T[..., 0] * ux**2 * uy + 2 * T[..., 1] * ux * uy
#                 )
#                 m = m.at[..., 13].set(
#                     T[..., 13] + 2 * T[..., 5] * ux + T[..., 7] * uz + T[..., 3] * ux**2 + T[..., 0] * ux**2 * uz + 2 * T[..., 1] * ux * uz
#                 )
#                 m = m.at[..., 14].set(
#                     T[..., 14] + T[..., 9] * uy + 2 * T[..., 6] * uz + T[..., 2] * uz**2 + T[..., 0] * uy * uz**2 + 2 * T[..., 3] * uy * uz
#                 )
#                 m = m.at[..., 15].set(
#                     T[..., 15] + 2 * T[..., 6] * uy + T[..., 8] * uz + T[..., 3] * uy**2 + T[..., 0] * uy**2 * uz + 2 * T[..., 2] * uy * uz
#                 )
#                 m = m.at[..., 16].set(
#                     T[..., 16]
#                     + T[..., 6] * ux
#                     + T[..., 5] * uy
#                     + T[..., 4] * uz
#                     + T[..., 3] * ux * uy
#                     + T[..., 2] * ux * uz
#                     + T[..., 1] * uy * uz
#                     + T[..., 0] * ux * uy * uz
#                 )
#                 m = m.at[..., 17].set(
#                     T[..., 0] * ux**2 * uy**2
#                     + 2 * T[..., 2] * ux**2 * uy
#                     + T[..., 8] * ux**2
#                     + 2 * T[..., 1] * ux * uy**2
#                     + 4 * T[..., 4] * ux * uy
#                     + 2 * T[..., 10] * ux
#                     + T[..., 7] * uy**2
#                     + 2 * T[..., 12] * uy
#                     + T[..., 17]
#                 )
#                 m = m.at[..., 18].set(
#                     T[..., 0] * ux**2 * uz**2
#                     + 2 * T[..., 3] * ux**2 * uz
#                     + T[..., 9] * ux**2
#                     + 2 * T[..., 1] * ux * uz**2
#                     + 4 * T[..., 5] * ux * uz
#                     + 2 * T[..., 11] * ux
#                     + T[..., 7] * uz**2
#                     + 2 * T[..., 13] * uz
#                     + T[..., 18]
#                 )
#                 m = m.at[..., 19].set(
#                     T[..., 0] * uy**2 * uz**2
#                     + 2 * T[..., 3] * uy**2 * uz
#                     + T[..., 9] * uy**2
#                     + 2 * T[..., 2] * uy * uz**2
#                     + 4 * T[..., 6] * uy * uz
#                     + 2 * T[..., 14] * uy
#                     + T[..., 8] * uz**2
#                     + 2 * T[..., 15] * uz
#                     + T[..., 19]
#                 )
#                 m = m.at[..., 20].set(
#                     T[..., 20]
#                     + 2 * T[..., 16] * ux
#                     + T[..., 13] * uy
#                     + T[..., 12] * uz
#                     + T[..., 6] * ux**2
#                     + T[..., 3] * ux**2 * uy
#                     + T[..., 2] * ux**2 * uz
#                     + 2 * T[..., 5] * ux * uy
#                     + 2 * T[..., 4] * ux * uz
#                     + T[..., 7] * uy * uz
#                     + 2 * T[..., 1] * ux * uy * uz
#                     + T[..., 0] * ux**2 * uy * uz
#                 )
#                 m = m.at[..., 21].set(
#                     T[..., 21]
#                     + T[..., 15] * ux
#                     + 2 * T[..., 16] * uy
#                     + T[..., 10] * uz
#                     + T[..., 5] * uy**2
#                     + T[..., 3] * ux * uy**2
#                     + T[..., 1] * uy**2 * uz
#                     + 2 * T[..., 6] * ux * uy
#                     + T[..., 8] * ux * uz
#                     + 2 * T[..., 4] * uy * uz
#                     + 2 * T[..., 2] * ux * uy * uz
#                     + T[..., 0] * ux * uy**2 * uz
#                 )
#                 m = m.at[..., 22].set(
#                     T[..., 22]
#                     + T[..., 14] * ux
#                     + T[..., 11] * uy
#                     + 2 * T[..., 16] * uz
#                     + T[..., 4] * uz**2
#                     + T[..., 2] * ux * uz**2
#                     + T[..., 1] * uy * uz**2
#                     + T[..., 9] * ux * uy
#                     + 2 * T[..., 6] * ux * uz
#                     + 2 * T[..., 5] * uy * uz
#                     + 2 * T[..., 3] * ux * uy * uz
#                     + T[..., 0] * ux * uy * uz**2
#                 )
#                 m = m.at[..., 23].set(
#                     T[..., 23]
#                     + T[..., 19] * ux
#                     + 2 * T[..., 22] * uy
#                     + 2 * T[..., 21] * uz
#                     + T[..., 11] * uy**2
#                     + T[..., 10] * uz**2
#                     + T[..., 9] * ux * uy**2
#                     + T[..., 8] * ux * uz**2
#                     + 2 * T[..., 4] * uy * uz**2
#                     + 2 * T[..., 5] * uy**2 * uz
#                     + T[..., 1] * uy**2 * uz**2
#                     + 2 * T[..., 14] * ux * uy
#                     + 2 * T[..., 15] * ux * uz
#                     + 4 * T[..., 16] * uy * uz
#                     + 4 * T[..., 6] * ux * uy * uz
#                     + 2 * T[..., 2] * ux * uy * uz**2
#                     + 2 * T[..., 3] * ux * uy**2 * uz
#                     + T[..., 0] * ux * uy**2 * uz**2
#                 )
#                 m = m.at[..., 24].set(
#                     T[..., 24]
#                     + 2 * T[..., 22] * ux
#                     + T[..., 18] * uy
#                     + 2 * T[..., 20] * uz
#                     + T[..., 14] * ux**2
#                     + T[..., 12] * uz**2
#                     + T[..., 9] * ux**2 * uy
#                     + 2 * T[..., 4] * ux * uz**2
#                     + 2 * T[..., 6] * ux**2 * uz
#                     + T[..., 7] * uy * uz**2
#                     + T[..., 2] * ux**2 * uz**2
#                     + 2 * T[..., 11] * ux * uy
#                     + 4 * T[..., 16] * ux * uz
#                     + 2 * T[..., 13] * uy * uz
#                     + 4 * T[..., 5] * ux * uy * uz
#                     + 2 * T[..., 1] * ux * uy * uz**2
#                     + 2 * T[..., 3] * ux**2 * uy * uz
#                     + T[..., 0] * ux**2 * uy * uz**2
#                 )
#                 m = m.at[..., 25].set(
#                     T[..., 25]
#                     + 2 * T[..., 21] * ux
#                     + 2 * T[..., 20] * uy
#                     + T[..., 17] * uz
#                     + T[..., 15] * ux**2
#                     + T[..., 13] * uy**2
#                     + 2 * T[..., 5] * ux * uy**2
#                     + 2 * T[..., 6] * ux**2 * uy
#                     + T[..., 8] * ux**2 * uz
#                     + T[..., 7] * uy**2 * uz
#                     + T[..., 3] * ux**2 * uy**2
#                     + 4 * T[..., 16] * ux * uy
#                     + 2 * T[..., 10] * ux * uz
#                     + 2 * T[..., 12] * uy * uz
#                     + 4 * T[..., 4] * ux * uy * uz
#                     + 2 * T[..., 1] * ux * uy**2 * uz
#                     + 2 * T[..., 2] * ux**2 * uy * uz
#                     + T[..., 0] * ux**2 * uy**2 * uz
#                 )
#                 m = m.at[..., 26].set(
#                     T[..., 0] * ux**2 * uy**2 * uz**2
#                     + 2 * T[..., 3] * ux**2 * uy**2 * uz
#                     + T[..., 9] * ux**2 * uy**2
#                     + 2 * T[..., 2] * ux**2 * uy * uz**2
#                     + 4 * T[..., 6] * ux**2 * uy * uz
#                     + 2 * T[..., 14] * ux**2 * uy
#                     + T[..., 8] * ux**2 * uz**2
#                     + 2 * T[..., 15] * ux**2 * uz
#                     + T[..., 19] * ux**2
#                     + 2 * T[..., 1] * ux * uy**2 * uz**2
#                     + 4 * T[..., 5] * ux * uy**2 * uz
#                     + 2 * T[..., 11] * ux * uy**2
#                     + 4 * T[..., 4] * ux * uy * uz**2
#                     + 8 * T[..., 16] * ux * uy * uz
#                     + 4 * T[..., 22] * ux * uy
#                     + 2 * T[..., 10] * ux * uz**2
#                     + 4 * T[..., 21] * ux * uz
#                     + 2 * T[..., 23] * ux
#                     + T[..., 7] * uy**2 * uz**2
#                     + 2 * T[..., 13] * uy**2 * uz
#                     + T[..., 18] * uy**2
#                     + 2 * T[..., 12] * uy * uz**2
#                     + 4 * T[..., 20] * uy * uz
#                     + 2 * T[..., 24] * uy
#                     + T[..., 17] * uz**2
#                     + 2 * T[..., 25] * uz
#                     + T[..., 26]
#                 )
#                 return m
#
#             return shift_inverse(T, u)
#
#     @partial(jit, static_argnums=(0,))
#     def compute_eq_central_moments(self, rho):
#         """
#         Calculate the central moments of the equilibrium distribution.
#
#         Parameters:
#         ----------
#         rho: jax.numpy.ndarray
#            Density field.
#
#         Returns:
#         -------
#         T_eq: jax.numpy.ndarray
#             central moment of the equilibrium distribution.
#         """
#
#         if isinstance(self.lattice, LatticeD2Q9):
#             T_eq = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             T_eq = T_eq.at[..., 0].set(rho[..., 0])
#             T_eq = T_eq.at[..., 3].set(2 * rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs**4)
#
#             return T_eq
#
#         elif isinstance(self.lattice, LatticeD3Q19):
#             T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             T_eq = T_eq.at[..., 0].set(rho[..., 0])
#             T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 16].set(rho[..., 0] * self.lattice.cs**4)
#             T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
#             T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)
#
#             return T_eq
#
#         elif isinstance(self.lattice, LatticeD3Q27):
#             T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             T_eq = T_eq.at[..., 0].set(rho[..., 0])
#             T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
#             T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
#             T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)
#             T_eq = T_eq.at[..., 19].set(rho[..., 0] * self.lattice.cs**4)
#             T_eq = T_eq.at[..., 26].set(rho[..., 0] * self.lattice.cs**6)
#
#             return T_eq
#
#     @partial(jit, static_argnums=(0,))
#     def compute_force_central_moments(self, F):
#         """
#         Calculate the central moments of the force distribution. Includes modification to accurately replicate mechanical stability conditions.
#
#         Parameters:
#         ----------
#         F: pytree of jax.numpy.ndarray
#             Force field.
#
#         Returns:
#         -------
#         T_eq: pytree of jax.numpy.ndarray
#             Central moments of the force distribution.
#         """
#
#         if isinstance(self.lattice, LatticeD2Q9):
#             C = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             Fx = F[..., 0]
#             Fy = F[..., 1]
#             C = C.at[..., 1].set(Fx)
#             C = C.at[..., 2].set(Fy)
#             C = C.at[..., 6].set(Fy * self.lattice.cs2)
#             C = C.at[..., 7].set(Fx * self.lattice.cs2)
#
#             return C
#         elif isinstance(self.lattice, LatticeD3Q19):
#             C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             Fx = F[..., 0]
#             Fy = F[..., 1]
#             Fz = F[..., 2]
#             C = C.at[..., 1].set(Fx)
#             C = C.at[..., 2].set(Fy)
#             C = C.at[..., 3].set(Fz)
#             C = C.at[..., 10].set(Fx * self.lattice.cs2)
#             C = C.at[..., 11].set(Fx * self.lattice.cs2)
#             C = C.at[..., 12].set(Fy * self.lattice.cs2)
#             C = C.at[..., 13].set(Fz * self.lattice.cs2)
#             C = C.at[..., 14].set(Fy * self.lattice.cs2)
#             C = C.at[..., 15].set(Fz * self.lattice.cs2)
#
#             return C
#         elif isinstance(self.lattice, LatticeD3Q27):
#             C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
#             Fx = F[..., 0]
#             Fy = F[..., 1]
#             Fz = F[..., 2]
#             C = C.at[..., 1].set(Fx)
#             C = C.at[..., 2].set(Fy)
#             C = C.at[..., 3].set(Fz)
#             C = C.at[..., 10].set(Fx * self.lattice.cs2)
#             C = C.at[..., 11].set(Fx * self.lattice.cs2)
#             C = C.at[..., 12].set(Fy * self.lattice.cs2)
#             C = C.at[..., 13].set(Fz * self.lattice.cs2)
#             C = C.at[..., 14].set(Fy * self.lattice.cs2)
#             C = C.at[..., 15].set(Fz * self.lattice.cs2)
#             C = C.at[..., 23].set(Fx * self.lattice.cs**4)
#             C = C.at[..., 24].set(Fy * self.lattice.cs**4)
#             C = C.at[..., 25].set(Fz * self.lattice.cs**4)
#
#             return C
#
#     @partial(jit, static_argnums=(0,), inline=True)
#     def apply_force(self, Tdash, rho, u):
#         """
#         Modified version of the apply_force defined in LBMBase to account for modified force.
#
#         Parameters
#         ----------
#         Tdash: jax.numpy.ndarray
#             Central moments post-collision distribution functions.
#         rho: jax.numpy.ndarray
#             Density field.
#         u: jax.numpy.ndarray
#            Velocity field.
#
#         Returns
#         -------
#         f_postcollision: jax.numpy.ndarray
#             The post-collision distribution functions with the force applied.
#         """
#         F = self.get_force()
#         if F is None:
#             F = jnp.zeros_like(u)
#         C = self.compute_force_central_moments(F)
#         Tf = jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * self.S)
#         return Tdash + Tf
#
#     @partial(jit, static_argnums=(0,), donate_argnums=(1,))
#     def collision(self, fin):
#         """
#         Cascaded LBM collision step for lattice.
#         """
#         fin = self.precisionPolicy.cast_to_compute(fin)
#         rho, _ = self.update_macroscopic(fin)
#         u = self.macroscopic_velocity(fin, rho)
#         T = jnp.dot(fin, self.M)
#         Tdash = self.compute_central_moment(T, u)
#         Tdash_eq = self.compute_eq_central_moments(rho)
#         Tout = jnp.dot(Tdash, jnp.eye(self.lattice.q) - self.S) + jnp.dot(Tdash_eq, self.S)
#         Tout = self.apply_force(Tout, rho, u)
#         Tout = self.compute_central_moment_inverse(Tout, u)
#         fout = jnp.dot(T, self.M_inv)
#         return self.precisionPolicy.cast_to_output(fout)
