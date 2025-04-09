"""
Definition of Multiphase class for simulating a multiphase flow.
"""

import operator
import time

# System libraries
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, vmap
from jax.experimental.multihost_utils import process_allgather

# Third-party libraries
from jax.tree import map, reduce
from termcolor import colored

# User-defined libraries
from src.boundary_conditions import (
    BounceBack,
    BounceBackHalfway,
    BounceBackMoving,
    InterpolatedBounceBackBouzidi,
    InterpolatedBounceBackDifferentiable,
)
from src.lattice import LatticeD2Q9, LatticeD3Q19
from src.base import LBMBase
from src.utils import downsample_field

# This significantly reduces the performance. Use if necessary
# jax.config.update("jax_debug_nans", True)


class Multiphase(LBMBase):
    """
    Multiphase model, based on the Shan-Chen method. To model the fluid, an equation of state (EOS) is defined by the user.
    Sequence of computation is pressure (EOS, dependent on the density and temperature) --> effective mass (phi).
    Can model both single component multiphase (SCMP) and multi-component multiphase (MCMP).

    Parameters
    ----------
    k: list
        Modification coefficient, used to tune surface tension.
    A: list
       Weighting factor, used for linear combination of Shan-Chen and Zhang-Chen Forces
    g_kk: numpy.ndarray
        Inter component interaction strength. Its a matrix of size n_components x n_components. It must be symmetric.

    References
    ----------
    1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
        Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.”
        Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.

    """

    def __init__(self, **kwargs):
        self.n_components = kwargs.get("n_components")
        super().__init__(**kwargs)
        self.k = kwargs.get("k")
        self.A = kwargs.get("A")
        self.eos = kwargs.get("EOS", None)
        self.g_kkprime = kwargs.get("g_kkprime")  # Fluid-fluid interaction strength
        # self.g_ks = kwargs.get("g_ks")  # Fluid-solid interaction strength
        self.body_force = kwargs.get("body_force", None)

        self.G_ff = self.compute_ff_greens_function()
        # self.G_fs = self.compute_fs_greens_function()

        # self.omega = jnp.array(self.omega, dtype=self.precisionPolicy.compute_dtype)
        self.g_kkprime = jnp.array(self.g_kkprime, dtype=self.precisionPolicy.compute_dtype)

        self.solid_mask_streamed = self.get_solid_mask_streamed()

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if not isinstance(value, list):
            raise ValueError("omega must be a list")
        self._omega = value

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        if value is None:
            raise ValueError("Number of components cannot be None")
        if value <= 0:
            raise ValueError("Number of components must be positive")
        if not isinstance(value, int):
            raise ValueError("Number of components must be an integer")
        self._n_components = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if value is None:
            raise ValueError("Modification coefficient must be provided")
        if isinstance(value, float) or isinstance(value, int):
            if self.n_components != 1:
                raise ValueError("The number of modification coefficients provided does not match the number of components in the system")
            self._k = [value]
        elif isinstance(value, list):
            if len(value) != self.n_components:
                raise ValueError("The number of modification coefficients provided does not match the number of components in the system")
        self._k = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is None:
            raise ValueError("Weight coefficient value must be provided")
        if isinstance(value, np.ndarray):
            if value.shape != (self.n_components, self.n_components):
                raise ValueError("The dimensions of A should match the number of components")
        self._A = jnp.array(value, dtype=self.precisionPolicy.compute_dtype)

    @property
    def body_force(self):
        return self._body_force

    @body_force.setter
    def body_force(self, value):
        if isinstance(value, list):
            self._body_force = jnp.array(np.array(value), dtype=self.precisionPolicy.compute_dtype)
        if isinstance(value, np.ndarray):
            self._body_force = jnp.array(value, dtype=self.precisionPolicy.compute_dtype)

    @property
    def g_kkprime(self):
        return self._g_kkprime

    @g_kkprime.setter
    def g_kkprime(self, value):
        if not isinstance(value, np.ndarray) and not isinstance(value, jax.numpy.ndarray):
            raise ValueError("g_kkprime must be a numpy array or jax.numpy.ndarray")
        if value.shape != (self.n_components, self.n_components):
            raise ValueError("g_kkprime must be a matrix of size n_components x n_components")
        if not np.allclose(value, np.transpose(value), atol=1e-6):
            raise ValueError("g_kkprime must be a symmetric matrix")
        self._g_kkprime = np.array(value)

    # @property
    # def g_ks(self):
    #     return self._g_ks
    #
    # @g_ks.setter
    # def g_ks(self, value):
    #     if len(value) != self.n_components:
    #         raise ValueError("g_ks must be a list size n_components")
    #     if isinstance(value, np.ndarray):
    #         value = self.distributed_array_init(
    #             value.shape, self.precisionPolicy.compute_dtype, value
    #         )
    #     self._g_ks = value

    def get_solid_mask_streamed(self):
        """
        Define the solid mask used for fluid-solid interaction force. Currently the same solid mask is used by all components.
        Modify to have solid_mask_streamed for each component (useful for component specific boundary condition).

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray: solid_mask array. Dimension: (nx, ny, 1) for d == 2 and (nx, ny, nz, 1) for d == 3
        """
        # solid_indices = []
        # for i in range(self.n_components):
        #     for bc in self.BCs[i]:
        #         if (
        #             isinstance(bc, BounceBack)
        #             or isinstance(bc, BounceBackHalfway)
        #             or isinstance(bc, BounceBackMoving)
        #         ):
        #             solid_indices.append(np.array(bc.indices).T)
        # solid_index = None
        # if not len(solid_indices) == 0:
        #     solid_index = np.vstack(solid_indices)
        # if self.dim == 2:
        #     shape = (self.nx, self.ny, 1)
        #     solid_mask = jnp.zeros(shape, dtype=jnp.int8)
        #     if solid_index is not None:
        #         solid_mask = solid_mask.at[solid_index[:, 0], solid_index[:, 1], 0].set(
        #             1
        #         )
        # else:
        #     shape = (self.nx, self.ny, self.nz, 1)
        #     solid_mask = jnp.zeros(shape, dtype=jnp.int8)
        #     if solid_index is not None:
        #         solid_mask = solid_mask.at[
        #             solid_index[:, 0], solid_index[:, 1], solid_index[:, 2], 0
        #         ].set(1)
        # return self.streaming(
        #     jnp.repeat(
        #         solid_mask,
        #         axis=-1,
        #         repeats=self.q,
        #     )
        # )
        solid_mask = []
        solid_indices = [[] for i in range(self.n_components)]
        for i in range(self.n_components):
            for bc in self.BCs[i]:
                if isinstance(bc, BounceBack) or isinstance(bc, BounceBackHalfway) or isinstance(bc, BounceBackMoving):
                    solid_indices[i].append(np.array(bc.indices).T)
        for i in range(self.n_components):
            index = None
            if not len(solid_indices[i]) == 0:
                index = np.vstack(solid_indices[i])
            if self.dim == 2:
                shape = (self.nx, self.ny, 1)
                mask = jnp.zeros(shape, dtype=jnp.int8)
                if index is not None:
                    mask = mask.at[index[:, 0], index[:, 1], 0].set(1)
                mask = self.streaming(jnp.repeat(mask, axis=-1, repeats=self.q))
                solid_mask.append(mask)
            else:
                shape = (self.nx, self.ny, self.nz, 1)
                mask = jnp.zeros(shape, dtype=jnp.int8)
                if index is not None:
                    mask = mask.at[index[:, 0], index[:, 1], index[:, 2], 0].set(1)
                mask = self.streaming(jnp.repeat(mask, axis=-1, repeats=self.q))
                solid_mask.append(mask)
        return solid_mask

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = [[] for _ in range(self.n_components)]
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        for i in range(self.n_components):
            print(f"Component: {i + 1}")
            solid_halo_list = [np.array(bc.indices).T for bc in self.BCs[i] if bc.isSolid]
            solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

            # Create the grid mask on each process
            start = time.time()
            grid_mask = self.create_grid_mask(solid_halo_voxels)
            print("Time to create the grid mask:", time.time() - start)

            start = time.time()
            for bc in self.BCs[i]:
                assert bc.implementationStep in ["PostStreaming", "PostCollision"]
                bc.create_local_mask_and_normal_arrays(grid_mask)
            print("Time to create the local masks and normal arrays:", time.time() - start)

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho_tree, u_tree, cast_output=True):
        """
        Compute the equillibrium distribution function using the given density and velocity pytrees.

        Parameters
        ----------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density values.
        u_tree: jax.numpy.ndarray
            Pytree of velocity values.
        cast_output: bool, optional
            A flag to cast the density and velocity values to the compute and output precision. Default: True

        Returns
        -------
        feq_tree: pytree of jax.numpy.ndarray
            Pytree of equillibrium distribution.
        """
        if cast_output:
            cast = lambda x: self.precisionPolicy.cast_to_compute(x)
            rho_tree = map(cast, rho_tree)
            u_tree = map(cast, u_tree)

        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype)
        cu_tree = map(lambda u: 3.0 * jnp.dot(u, c), u_tree)
        usqr_tree = map(lambda u: 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True), u_tree)
        feq_tree = map(
            lambda rho, udote, udotu: rho * self.w * (1.0 + udote * (1.0 + 0.5 * udote) - udotu),
            rho_tree,
            cu_tree,
            usqr_tree,
        )

        if cast_output:
            return map(lambda f_eq: self.precisionPolicy.cast_to_output(f_eq), feq_tree)
        else:
            return feq_tree

    @partial(jit, static_argnums=(0,))
    def compute_average_density(self, rho_tree):
        rho_s_tree = map(
            lambda rho: self.streaming(jnp.repeat(rho, axis=-1, repeats=self.lattice.q)),
            rho_tree,
        )
        rho_ave_tree = map(
            lambda rho_s, solid_mask: jnp.sum(
                self.G_ff * rho_s * (1 - solid_mask),
                axis=-1,
                keepdims=True,
            )
            / jnp.sum(self.G_ff * (1 - solid_mask), axis=-1, keepdims=True),
            rho_s_tree,
            self.solid_mask_streamed,
        )
        return rho_ave_tree

    @partial(jit, static_argnums=(0,))
    def apply_contact_angle(self, rho_tree):
        """
        Apply the prescribed contact angle to density. The control parameters for this scheme are
        self.theta_tree, self.phi_tree and self.delta_rho_tree. The max and min density fields are identified from initial configuration.

        Parameters:
        ----------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density fields

        Returns:
        -------
        Pytree of density tree with adjusted contact angle values at the boundary nodes.

        Reference:
        ---------
        1. Li, Q., Yu, Y. & Luo, K. H. "Implementation of contact angles in pseudopotential lattice Boltzmann simulations with
        curved boundaries." Phys. Rev. E 100, 053313 (2019).
        """
        rho_ave_tree = self.compute_average_density(rho_tree)

        def set_contact_angle(rho, rho_ave, BC):
            rho_min = jnp.min(rho)
            rho_max = jnp.max(rho)
            for bc in BC:
                if isinstance(
                    bc,
                    (
                        BounceBackHalfway,
                        BounceBack,
                        BounceBackMoving,
                        InterpolatedBounceBackBouzidi,
                        InterpolatedBounceBackDifferentiable,
                    ),
                ):
                    rho = rho.at[bc.indices].set(
                        (bc.theta <= jnp.pi / 2) * (bc.phi * rho_ave[bc.indices]) + (bc.theta > jnp.pi / 2) * (rho_ave[bc.indices] - bc.delta_rho)
                    )
                    rho = jnp.clip(rho, min=rho_min, max=rho_max)
            return rho

        return map(
            lambda rho, rho_ave, BC: set_contact_angle(rho, rho_ave, BC),
            rho_tree,
            rho_ave_tree,
            self.BCs,
        )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        """
        Apply collision step of LBM
        """
        pass

    def compute_ff_greens_function(self):
        """
        Define the Green's function used to compute interaction phase-phase interaction forces.

        The interaction coefficient between k^th and kprime^th component: self.gkkprime[k, kprime]
        During computation, this value is multiplied with corresponding g_kkprime value to get the Green's function:
        G_kkprime = self.g_kk[k, k_prime] * self.G_ff

        G_kkprime(x, x') = g1 * g_kkprime,  if |x - x'| = 1
                         = g2 * g_kkprime,  if |x - x'| = sqrt(2)
                         = 0,               otherwise

        Here d is the dimension of problem and x' are the neighboring points.

        Some examples values could be:
        For D2Q9:
            g1 = 2 and g2 = 1/2
        For D3Q19
            g1 = 1 and g2 = 1/2

        Parameters
        ----------
        None

        Returns
        -------
        G_ff: jax.numpy.ndarray.
            Dimension: (q, )
        """
        c = np.array(self.lattice.c).T
        G_ff = np.zeros((self.q,), dtype=np.float64)
        cl = np.linalg.norm(c, axis=-1)
        if isinstance(self.lattice, LatticeD2Q9):
            g1 = 1 / 3
            g2 = 1 / 12
            G_ff[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_ff[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        elif isinstance(self.lattice, LatticeD3Q19):
            g1 = 1 / 6
            g2 = 1 / 12
            G_ff[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_ff[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        return jnp.array(G_ff, dtype=self.precisionPolicy.compute_dtype)

    def compute_fs_greens_function(self):
        """
        Define the Green's function used to model interaction between kth fluid and solid.

        During computation, this G_fs is multiplied with corresponding g_ks value to get the Green's function:
        G_ks = self.g_ks[k] * self.G_fs

        Green's function used in this case:
        G_ks(x, x') = g1 * g_ks,       if |x - x'| = 1
                    = g2 * g_ks,       if |x - x'| = sqrt(2)
                    = 0,               otherwise

        Here d is the dimension of problem and x' are the neighboring points.
        For D2Q9:
            g1 = 2 and g2 = 1/2
        For D3Q19
            g1 = 1 and g2 = 1/2

        Parameters
        ----------
        None

        Returns
        -------
        G_fs: jax.numpy.ndarray
            Dimension: (q, )
        """
        c = np.array(self.lattice.c).T
        cl = np.linalg.norm(c, axis=-1)
        G_fs = np.zeros((self.q,), dtype=np.float64)
        if isinstance(self.lattice, LatticeD2Q9):
            g1 = 1 / 3
            g2 = 1 / 12
            G_fs[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_fs[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        elif isinstance(self.lattice, LatticeD3Q19):
            g1 = 1 / 6
            g2 = 1 / 12
            G_fs[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_fs[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        return jnp.array(G_fs, dtype=self.precisionPolicy.compute_dtype)

    def assign_fields_sharded(self):
        """
        This function is used to initialize pytree of the distribution arrays using the initial velocities and velocity defined in self.initialize_macroscopic_fields function.
        To do this, function first uses the initialize_macroscopic_fields function to get the initial values of rho (rho0) and velocity (u0).

        If this function is not modified then, the distribution pytree is initialized with density value of 1.0 everywhere and velocity of 0.0 everywhere

        The distribution is initialized with rho0 and u0 values, using the self.equilibrium function.

        Parameters
        ----------
        None

        Returns
        -------
        f: pytree of distributed JAX array of shape: (self.nx, self.ny, self.q) for 2D and (self.nx, self.ny, self.nz, self.q) for 3D.
        """
        rho0_tree, u0_tree = self.initialize_macroscopic_fields()
        if self.dim == 2:
            shape = (self.nx, self.ny, self.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.q)
        f_tree = []
        if rho0_tree is not None and u0_tree is not None:
            assert len(rho0_tree) == self.n_components, "The initial density values for all components must be provided"

            assert len(u0_tree) == self.n_components, "The initial velocity values for all components must be provided."

            for i in range(self.n_components):
                rho0, u0 = rho0_tree[i], u0_tree[i]
                rho0 = self.precisionPolicy.cast_to_compute(rho0)
                u0 = self.precisionPolicy.cast_to_compute(u0)
                f_tree.append(self.initialize_populations(rho0, u0))
        else:
            for i in range(self.n_components):
                f_tree.append(self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=self.w))
        return f_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(
        self,
        f_tree,
    ):
        """
        update_macroscopic from base.py extended to pytrees.

        Parameters
        ----------
        f_tree: pytree of jax.numpy.ndarray
            Pytree of distribution arrays.

        Returns
        -------
        rho_tree: pytree of jax.numpy.ndarray for component densities
        u_tree: pytree of jax.numpy.ndarray for component velocities
        """
        rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        u_tree = map(
            lambda f, rho: jnp.dot(f, c) / rho,
            f_tree,
            rho_tree,
        )  # Component velocity
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def macroscopic_velocity(self, f_tree, rho_tree):
        """
        macroscopic_velocity computes the velocity and incorporates forces
        using the Exact Difference Method (EDM). This is only used for
        post-processing and not for equilibrium distribution computation.

        Parameters
        ----------
        f_tree: pytree of jax.numpy.ndarray
            Pytree of distribution arrays.
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density values.

        Returns
        -------
        u_tree: pytree of jax.numpy.ndarray for component velocities
        """
        # rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        u_tree = map(
            lambda f, rho: jnp.dot(f, c) / rho,
            f_tree,
            rho_tree,
        )
        F_tree = self.compute_force(rho_tree)
        return map(lambda rho, u, F: u + 0.5 * F / rho, rho_tree, u_tree, F_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_density(self, rho_tree):
        """
        Compute the total density using component velocity and density values.

        Parmeters
        ---------
        rho_tree: Pytree of jax.numpy.ndarray
            Pytree of component density values.

        Returns
        -------
        jax.numpy.ndarray
            Total density values.
        """
        return reduce(operator.add, rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_velocity(self, rho_tree, u_tree):
        """
        Compute the total velocity using component velocity and density values.

        Parameters
        ----------
        rho_tree: Pytree of jax.numpy.ndarray
            Pytree of component density values.
        u_tree: Pytree of jax.numpy.ndarray
            Pytree of component velocity values.

        Returns
        -------
        jax.numpy.ndarray
            Total velocity values.
        """
        n = reduce(
            operator.add,
            map(
                lambda rho, u, omega: rho * u * omega,
                rho_tree,
                u_tree,
                self.omega,
            ),
        )
        d = reduce(
            operator.add,
            map(
                lambda rho, omega: rho * omega,
                rho_tree,
                self.omega,
            ),
        )
        return n / d

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        """
        Generalized function for computing pressure. By default it uses equation
        of state but it can be modified if the pseudopotential is computed using
        a different method.

        Parameters
        ----------
        p_tree: pytree of jax.numpy.ndarray
            Pressure values for all components.

        Returns
        -------
        pytree of jax.numpy.ndarray
        """
        return self.eos.EOS(rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree):
        """
        Cpmpute the total combined pressure from all components.

        Parameters
        ----------
        p_tree: pytree of jax.numpy.ndarray
            Pressure values for all components.

        Returns
        -------
        jax.numpy.ndarray
        """
        return reduce(operator.add, p_tree)

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        """
        Compute the potential (psi and U) for each component which is required for computing interaction forces.
        The psi values are obtained using the corresponding EOS.

        Parameters
        ----------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density values.

        Returns
        -------
        psi_tree: pytree of jax.numpy.ndarray
        """
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
        # Zhang-Chen potential
        U_tree = map(
            lambda k, p, rho: k * p - self.lattice.cs2 * rho,
            self.k,
            p_tree,
            rho_tree,
        )
        return psi_tree, U_tree

    # Compute the force using the effective mass (psi) and the interaction potential (phi)
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def compute_force(self, rho_tree):
        """
        Compute the force acting on each component(fluid). This includes fluid-fluid, fluid-solid, and body forces.

        Parameters
        ----------
        f_tree: pytree of jax.numpy.ndarray
            Pytree of distribution array.

        Returns
        -------
        Pytree of jax.numpy.ndarray
        """
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        # fluid_solid_force = self.compute_fluid_solid_force(rho_tree)
        if self.body_force is not None:
            return map(
                lambda ff, rho: ff + self.body_force * rho,
                fluid_fluid_force,
                rho_tree,
            )
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree, U_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi).
        The force calculation is based on the Shan-Chen method using the weighted sum
        of Shan-Chen and Zhang-Chen potential where modified pressure is used:

        modified pressure = k (modification )

        Parameters
        ----------
        psi_tree: pytree of jax.numpy.ndarray
            Pytree of pseudo-potential (Yuan-Schaefer, with modification)
        U_tree: pytree of jax.numpy.ndarray
            Pytree of pseudo-potential (Zhang-Chen, with modification)

        Returns
        -------
        pytree of jax.numpy.ndarray
            Pytree of fluid-fluid interaction force.
        """
        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
        psi_s_tree = map(
            lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)),
            psi_tree,
        )
        U_s_tree = map(lambda U: self.streaming(jnp.repeat(U, axis=-1, repeats=self.q)), U_tree)

        def ffk_1(Ai, g_kkprime):
            """
            Shan-Chen interaction force
            g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components

            Interaction force must only be applied if neighboring nodes are fluid nodes. 1 - solid_mask ensures that only
            fluid nodes are considered.
            """
            return reduce(
                operator.add,
                map(
                    lambda A, G, psi_s: jnp.dot(
                        (1 - A) * G * self.G_ff * psi_s,
                        c,
                    ),
                    list(Ai),
                    list(g_kkprime),
                    psi_s_tree,
                ),
            )

        def ffk_2(Ai):
            """
            Zhang-Chen interaction force.

            Interaction force must only be applied if neighboring nodes are fluid nodes. 1 - solid_mask ensures that only
            fluid nodes are considered.
            """
            return reduce(
                operator.add,
                map(
                    lambda A, U_s: A * jnp.dot(self.G_ff * U_s, c),
                    list(Ai),
                    U_s_tree,
                ),
            )

        return map(
            lambda psi, nt_1, nt_2: psi * nt_1 + nt_2,
            psi_tree,
            list(vmap(ffk_1, in_axes=(0, 0))(self.A, self.g_kkprime)),
            list(vmap(ffk_2, in_axes=(0))(self.A)),
        )

    # @partial(jit, static_argnums=(0,))
    # def compute_fluid_solid_force(self, rho_tree):
    #     """
    #     Compute the fluid-fluid interaction force using the effective mass (psi).

    #     Parameters
    #     ----------
    #     psi_tree: Pytree of jax.numpy.ndarray
    #         Pytree of pseudopotential of all components.

    #     Returns
    #     -------
    #     Pytree of jax.numpy.ndarray
    #         Pytree of fluid-solid interaction force.
    #     """
    #     return map(
    #         lambda g_ks, rho, solid_mask: -g_ks
    #         * rho
    #         * jnp.dot(self.G_fs * solid_mask, self.c.T),
    #         self.g_ks,
    #         rho_tree,
    #         self.solid_mask_streamed,
    #     )
    # psi_tree, _ = self.compute_potential(rho_tree)
    # psi_s_tree = map(
    #     lambda psi: self.streaming(
    #         jnp.repeat(psi, axis=-1, repeats=self.lattice.q)
    #     ),
    #     psi_tree,
    # )
    # return map(
    #     lambda g_ks, psi, psi_s, solid_mask: -g_ks
    #     * psi
    #     * jnp.dot(self.G_fs * solid_mask * psi_s, self.c.T),
    #     self.g_ks,
    #     psi_tree,
    #     psi_s_tree,
    #     self.solid_mask_streamed
    # )

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision_tree, feq_tree, rho_tree, u_tree):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
        f_postcollision_tree: pytree of jax.numpy.ndarray
            pytree of post-collision distribution functions.
        feq_tree: pytree of jax.numpy.ndarray
            pytree of equilibrium distribution functions.
        rho_tree: pytree of jax.numpy.ndarray
            pytree of density field for all components.
        u_tree: pytree of jax.numpy.ndarray
            pytree of velocity field for all components.

        Returns
        -------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions with the force applied.
        """
        F_tree = self.compute_force(rho_tree)

        u_temp_tree = map(lambda u, F, rho: u + F / rho, u_tree, F_tree, rho_tree)
        feq_force_tree = self.equilibrium(rho_tree, u_temp_tree)
        return map(
            lambda f_postcollision, feq_force, feq: f_postcollision + feq_force - feq,
            f_postcollision_tree,
            feq_force_tree,
            feq_tree,
        )

    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, fout_tree, fin_tree, timestep, implementation_step):
        """
        This function extends apply_bc to pytrees.

        Parameters
        ----------
        fout_tree: jax.numpy.ndarray
            The post-collision distribution functions.
        fin_tree: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        jax.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """

        def _apply_bc_(fin, fout, bc):
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
            return fout

        for i in range(self.n_components):
            for bc in self.BCs[i]:
                fout_tree[i] = _apply_bc_(fin_tree[i], fout_tree[i], bc)

        return fout_tree

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming_tree, timestep, return_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions
        towards their equilibrium values. It then applies the respective boundary conditions to the
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        Parameters
        ----------
        fin_tree: pytree of jax.numpy.ndarray
            pytree of post-streaming distribution function.
        timestep: int
            Current timestep

        Returns
        -------
        f_poststreaming_tree: pytree of jax.numpy.ndarray
            pytree of post-streamed distribution function.
        f_collision_tree: pytree of jax.numpy.ndarray {Optional}
            pytree of post-collision distribution function.
        """
        f_postcollision_tree = self.collision(f_poststreaming_tree)
        f_postcollision_tree = self.apply_bc(f_postcollision_tree, f_poststreaming_tree, timestep, "PostCollision")
        f_poststreaming_tree = map(
            lambda f_postcollision: self.streaming(f_postcollision),
            f_postcollision_tree,
        )
        f_poststreaming_tree = self.apply_bc(f_poststreaming_tree, f_postcollision_tree, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming_tree, f_postcollision_tree
        else:
            return f_poststreaming_tree, None

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
        f_tree: pytree of jax.numpy.ndarray
            pytree of distribution functions after the simulation.
        """
        f_tree = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]
                # shardings = jax.map(lambda x: x.sharding, f_tree)
                # restore_args = orb.checkpoint_utils.construct_restore_args(
                #     f_tree, shardings
                # )
                try:
                    f_tree = self.mngr.restore(latest_step, args=orb.args.StandardRestore(state))
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max  = {t_max}). Consider increasing t_max.")

        if self.computeMLUPS:
            start = time.time()

        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0
            checkpoint_flag = self.checkpointRate > 0 and timestep % self.checkpointRate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev_tree, _ = self.update_macroscopic(f_tree)
                u_prev_tree = self.macroscopic_velocity(f_tree, rho_prev_tree)
                rho_prev_tree = map(
                    lambda rho_prev: downsample_field(rho_prev, self.downsamplingFactor),
                    rho_prev_tree,
                )
                psi_prev_tree, _ = self.compute_potential(rho_prev_tree)
                p_prev_tree = self.compute_pressure(rho_prev_tree, psi_prev_tree)
                p_prev_total = self.compute_total_pressure(p_prev_tree)
                p_prev_total = downsample_field(p_prev_total, self.downsamplingFactor)
                u_prev_tree = map(
                    lambda u_prev: downsample_field(u_prev, self.downsamplingFactor),
                    u_prev_tree,
                )
                rho_total_prev = self.compute_total_density(rho_prev_tree)
                u_total_prev = self.compute_total_velocity(rho_prev_tree, u_prev_tree)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_prev_total = process_allgather(p_prev_total)
                rho_prev_tree = map(lambda rho_prev: process_allgather(rho_prev), rho_prev_tree)
                u_prev_tree = map(lambda u_prev: process_allgather(u_prev), u_prev_tree)
                rho_total_prev = process_allgather(rho_total_prev)
                u_total_prev = process_allgather(u_total_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f_tree, fstar_tree = self.step(f_tree, timestep)

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
                rho_tree, _ = self.update_macroscopic(f_tree)
                u_tree = self.macroscopic_velocity(f_tree, rho_tree)
                psi_tree, _ = self.compute_potential(rho_tree)
                p_tree = self.compute_pressure(rho_tree, psi_tree)
                p_total = self.compute_total_pressure(p_tree)
                p_total = downsample_field(p_total, self.downsamplingFactor)
                rho_tree = map(
                    lambda rho: downsample_field(rho, self.downsamplingFactor),
                    rho_tree,
                )
                u_tree = map(lambda u: downsample_field(u, self.downsamplingFactor), u_tree)

                rho_total = self.compute_total_density(rho_tree)
                u_total = self.compute_total_velocity(rho_tree, u_tree)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_total = process_allgather(p_total)
                rho_tree = map(lambda rho: process_allgather(rho), rho_tree)
                u_tree = map(lambda u: process_allgather(u), u_tree)
                rho_total = process_allgather(rho_total)
                u_total = process_allgather(u_total)

                # Save the data
                self.handle_io_timestep(
                    timestep,
                    f_tree,
                    fstar_tree,
                    p_tree,
                    p_total,
                    u_tree,
                    u_total,
                    rho_total,
                    rho_tree,
                    p_prev_tree,
                    p_prev_total,
                    u_total_prev,
                    u_prev_tree,
                    rho_total_prev,
                    rho_prev_tree,
                )

            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]

                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f_tree)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f_tree)
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
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.n_components * self.nx * self.ny * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )

            elif self.dim == 3:
                print(colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny} x {self.nz}", "green"))
                print(colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny * self.nz}", "green"))
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.n_components * self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f_tree

    def handle_io_timestep(
        self,
        timestep,
        f_tree,
        fstar_tree,
        p_tree,
        p_total,
        u_tree,
        u_total,
        rho_total,
        rho_tree,
        p_prev_tree,
        p_prev_total,
        u_total_prev,
        u_prev_tree,
        rho_total_prev,
        rho_prev_tree,
    ):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f_tree: pytree of jax.numpy.ndarray
            Pytree of post-streaming distribution functions at the current time step.
        fstar_tree: pytree of jax.numpy.ndarray
            Pytree of post-collision distribution functions at the current time step.
        p_tree: Pytree of jax.numpy.ndarray
            Pressure field at the current time step for each component.
        p: jax.numpy.ndarray
            Pressure field at the current time step.
        u_total: jax.numpy.ndarray
            Total velocity field at the current time step.
        u_tree: pytree of jax.numpy.ndarray
            Pytree of velocity field at the current time step.
        rho_total: jax.numpy.ndarray
            Total density field at the current time step.
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density field at the current time step.
        p_prev_tree: Pytree of jax.numpy.ndarray
            Pressure field at the previous time step for each component.
        p_prev: jax.numpy.ndarray
            Pressure field at the previous time step.
        u_total_prev: jax.numpy.ndarray
            Total velocity field at the previous time step.
        u_prev_tree: pytree of jax.numpy.ndarray
            Pytree of velocity field at the previous time step.
        rho_total_prev: jax.numpy.ndarray
            Total density field at the previous time step.
        rho_prev_tree: pytree of jax.numpy.ndarray
            Pytree of density field at the previous time step.
        """
        kwargs = {
            "n_components": self.n_components,
            "timestep": timestep,
            "rho_total": rho_total,
            "rho_tree": rho_tree,
            "p_tree": p_tree,
            "p": p_total,
            "u_total": u_total,
            "u_tree": u_tree,
            "rho_total_prev": rho_total_prev,
            "rho_prev_tree": rho_prev_tree,
            "p_prev_tree": p_prev_tree,
            "p_prev": p_prev_total,
            "u_total_prev": u_total_prev,
            "u_prev_tree": u_prev_tree,
            "f_poststreaming_tree": f_tree,
            "f_postcollision_tree": fstar_tree,
        }
        self.output_data(**kwargs)


class MultiphaseBGK(Multiphase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        """
        BGK collision step for lattice, extended to pytrees.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        fin_tree = map(lambda fin: self.precisionPolicy.cast_to_compute(fin), fin_tree)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        fneq_tree = map(lambda feq, fin: feq - fin, feq_tree, fin_tree)
        fout_tree = map(lambda fin, fneq, omega: fin + omega * fneq, fin_tree, fneq_tree, self.omega)

        fout_tree = self.apply_force(fout_tree, feq_tree, rho_tree, u_tree)

        return map(lambda fout: self.precisionPolicy.cast_to_output(fout), fout_tree)


class MultiphaseMRT(Multiphase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kappa = kwargs.get("kappa")
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = kwargs.get("s_v")
        self.M_inv = map(
            lambda M: jnp.array(
                np.linalg.inv(M).T,
                dtype=self.precisionPolicy.compute_dtype,
            ),
            kwargs.get("M"),
        )
        self.M = map(
            lambda M: jnp.array(M.T, dtype=self.precisionPolicy.compute_dtype),
            kwargs.get("M"),
        )
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = map(
                lambda s_rho, s_e, s_eta, s_j, s_q, s_v: jnp.array(
                    np.diag([s_rho, s_e, s_eta, s_j, s_q, s_j, s_q, s_v, s_v]),
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                self.s_rho,
                self.s_e,
                self.s_eta,
                self.s_j,
                self.s_q,
                self.s_v,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = map(
                lambda s_rho, s_e, s_eta, s_j, s_q, s_v, s_pi, s_m: jnp.array(
                    np.diag([
                        s_rho,
                        s_e,
                        s_eta,
                        s_j,
                        s_q,
                        s_j,
                        s_q,
                        s_j,
                        s_q,
                        s_v,
                        s_pi,
                        s_v,
                        s_pi,
                        s_v,
                        s_v,
                        s_v,
                        s_m,
                        s_m,
                        s_m,
                    ]),
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                self.s_rho,
                self.s_e,
                self.s_eta,
                self.s_j,
                self.s_q,
                self.s_v,
                self.s_pi,
                self.s_m,
            )

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        if not isinstance(value, list):
            raise ValueError("Matrix M must be a list")
        if len(value) != self.n_components:
            raise ValueError("Number of components does not match number of matrix M passed")
        else:
            self._M = value

    @partial(jit, static_argnums=(0,))
    def adjust_surface_tension(self, psi_tree):
        psi_s_tree = map(
            lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)),
            psi_tree,
        )
        c = jnp.transpose(self.c)
        if isinstance(self.lattice, LatticeD2Q9):
            tm1 = lambda i, j, psi, psi_s: psi[..., 0] * jnp.dot(self.G_ff * (psi_s - psi), c[:, i] * c[:, j])
            tm2 = lambda i, j, psi, psi_s: jnp.dot(self.G_ff * (psi_s**2 - psi**2), c[:, i] * c[:, j])

            def compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s):
                C = jnp.zeros_like(
                    psi_s,
                    dtype=self.precisionPolicy.compute_dtype,
                )
                qxx = -kappa * ((1 - A) * tm1(0, 0, psi, psi_s) + 0.5 * A * tm2(0, 0, psi, psi_s))
                qxy = -kappa * ((1 - A) * tm1(0, 1, psi, psi_s) + 0.5 * A * tm2(0, 1, psi, psi_s))
                qyy = -kappa * ((1 - A) * tm1(1, 1, psi, psi_s) + 0.5 * A * tm2(1, 1, psi, psi_s))
                C = C.at[..., 1].set(1.5 * s_e * (qxx + qyy))
                C = C.at[..., 2].set(-1.5 * s_eta * (qxx + qyy))
                C = C.at[..., 7].set(-s_v * (qxx - qyy))
                C = C.at[..., 8].set(-s_v * qxy)
                return C

            C_tree = map(
                lambda kappa, A, s_v, s_e, s_eta, psi, psi_s: compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s),
                self.kappa,
                list(self.A.diagonal()),
                self.s_v,
                self.s_e,
                self.s_eta,
                psi_tree,
                psi_s_tree,
            )
            return C_tree
        elif isinstance(self.lattice, LatticeD3Q19):
            tm1 = lambda i, j, psi, psi_s: psi[..., 0] * jnp.dot(self.G_ff * (psi_s - psi), c[:, i] * c[:, j])
            tm2 = lambda i, j, psi, psi_s: jnp.dot(self.G_ff * (psi_s**2 - psi**2), c[:, i] * c[:, j])

            def compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s):
                C = jnp.zeros_like(
                    psi_s,
                    dtype=self.precisionPolicy.compute_dtype,
                )
                qxx = -kappa * ((1 - A) * tm1(0, 0, psi, psi_s) + 0.5 * A * tm2(0, 0, psi, psi_s))
                qxy = -kappa * ((1 - A) * tm1(0, 1, psi, psi_s) + 0.5 * A * tm2(0, 1, psi, psi_s))
                qxz = -kappa * ((1 - A) * tm1(0, 2, psi, psi_s) + 0.5 * A * tm2(0, 2, psi, psi_s))
                qyy = -kappa * ((1 - A) * tm1(1, 1, psi, psi_s) + 0.5 * A * tm2(1, 1, psi, psi_s))
                qyz = -kappa * ((1 - A) * tm1(1, 2, psi, psi_s) + 0.5 * A * tm2(1, 2, psi, psi_s))
                qzz = -kappa * ((1 - A) * tm1(2, 2, psi, psi_s) + 0.5 * A * tm2(2, 2, psi, psi_s))
                C = C.at[..., 1].set((2 / 5) * s_e * (qxx + qyy + qzz))
                C = C.at[..., 9].set(-s_v * (2 * qxx - qyy - qzz))
                C = C.at[..., 11].set(-s_v * (qyy - qzz))
                C = C.at[..., 13].set(-s_v * qxy)
                C = C.at[..., 14].set(-s_v * qyz)
                C = C.at[..., 15].set(-s_v * qxz)
                return C

            C_tree = map(
                lambda kappa, A, s_v, s_e, s_eta, psi, psi_s: compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s),
                self.kappa,
                list(self.A.diagonal()),
                self.s_v,
                self.s_e,
                self.s_eta,
                psi_tree,
                psi_s_tree,
            )
            return C_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, m_tree, meq_tree, rho_tree, u_tree):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
        m_tree: pytree of jax.numpy.ndarray
            pytree of post-collision distribution functions.
        meq_tree: pytree of jax.numpy.ndarray
            pytree of equilibrium distribution functions.
        rho_tree: pytree of jax.numpy.ndarray
            pytree of density field for all components.
        u_tree: pytree of jax.numpy.ndarray
            pytree of velocity field for all components.

        Returns
        -------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions with the force applied.
        """
        F_tree = self.compute_force(rho_tree)

        delta_u_tree = map(lambda F, rho: F / rho, F_tree, rho_tree)
        u_temp_tree = map(lambda u, delta_u: u + delta_u, u_tree, delta_u_tree)
        feq_force_tree = self.equilibrium(rho_tree, u_temp_tree, cast_output=False)
        meq_force_tree = map(lambda feq, M: jnp.dot(feq, M), feq_force_tree, self.M)
        return map(
            lambda m, meq_force, meq: m + meq_force - meq,
            m_tree,
            meq_force_tree,
            meq_tree,
        )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        """
        MRT collision step for lattice.
        """
        fin_tree = map(lambda f: self.precisionPolicy.cast_to_compute(f), fin_tree)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        m_tree = map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        meq_tree = map(lambda feq, M: jnp.dot(feq, M), feq_tree, self.M)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.adjust_surface_tension(psi_tree)
        mout_tree = map(
            lambda m, meq, S: m - jnp.dot(m - meq, S),
            m_tree,
            meq_tree,
            self.S,
        )
        mout_tree = self.apply_force(mout_tree, meq_tree, rho_tree, u_tree)
        fout_tree = map(lambda m, Minv, C: jnp.dot(m + C, Minv), mout_tree, self.M_inv, C_tree)
        # fout_tree = self.apply_force(fout_tree, feq_tree, rho_tree, u_tree)
        return map(
            lambda fout: self.precisionPolicy.cast_to_output(fout),
            fout_tree,
        )


class CascadedLBM(Multiphase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = kwargs.get("sigma")
        self.s_0 = kwargs.get("s_0")
        self.s_1 = kwargs.get("s_1")
        self.s_b = kwargs.get("s_b")
        self.s_2 = kwargs.get("s_2")
        self.s_3 = kwargs.get("s_3")
        self.s_4 = kwargs.get("s_4")
        self.M_inv = map(
            lambda M: jnp.array(
                np.linalg.inv(M).T,
                dtype=self.precisionPolicy.compute_dtype,
            ),
            kwargs.get("M"),
        )
        self.M = map(
            lambda M: jnp.array(M.T, dtype=self.precisionPolicy.compute_dtype),
            kwargs.get("M"),
        )
        self.S = map(
            lambda s_0, s_1, s_b, s_2, s_3, s_4: jnp.array(
                np.diag([s_0, s_1, s_1, s_b, s_2, s_2, s_3, s_3, s_4]),
                dtype=self.precisionPolicy.compute_dtype,
            ),
            self.s_0,
            self.s_1,
            self.s_b,
            self.s_2,
            self.s_3,
            self.s_4,
        )

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        if not isinstance(value, list):
            raise ValueError("Matrix M must be a list")
        if len(value) != self.n_components:
            raise ValueError("Number of components does not match number of matrix M passed")
        self._M = value

    @partial(jit, static_argnums=(0,))
    def compute_normals(self):
        ns = jnp.dot(self.normal_weights * self.solid_mask_streamed_h, self.c_h)
        return ns / jnp.sqrt(jnp.sum(ns**2, axis=-1, keepdims=True))

    @partial(jit, static_argnums=(0,))
    def compute_central_moment(self, m_tree, u_tree):
        if self.lattice.d == 2:

            def f(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
                T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
                T = T.at[..., 3].set(usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3])
                T = T.at[..., 4].set(udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4])
                T = T.at[..., 5].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5])
                T = T.at[..., 6].set(
                    -(ux**2) * uy * m[..., 0]
                    + 2 * ux * uy * m[..., 1]
                    + ux**2 * m[..., 2]
                    - 0.5 * uy * m[..., 3]
                    - 0.5 * uy * m[..., 4]
                    - 2 * ux * m[..., 5]
                    + m[..., 6]
                )
                T = T.at[..., 7].set(
                    -(uy**2) * ux * m[..., 0]
                    + uy**2 * m[..., 1]
                    + 2 * ux * uy * m[..., 2]
                    - 0.5 * ux * m[..., 3]
                    + 0.5 * ux * m[..., 4]
                    - 2 * uy * m[..., 5]
                    + m[..., 7]
                )
                T = T.at[..., 8].set(
                    (uy**2 * ux**2) * m[..., 0]
                    - 2 * ux * uy**2 * m[..., 1]
                    - 2 * uy * ux**2 * m[..., 2]
                    + 0.5 * usq * m[..., 3]
                    - 0.5 * udiff * m[..., 4]
                    + 4 * ux * uy * m[..., 5]
                    - 2 * uy * m[..., 6]
                    - 2 * ux * m[..., 7]
                    + m[..., 8]
                )
                return T

            return map(lambda m, u: f(m, u), m_tree, u_tree)

    @partial(jit, static_argnums=(0,))
    def compute_central_moment_inverse(self, T_tree, u_tree):
        if self.lattice.d == 2:

            def f(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
                m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
                m = m.at[..., 3].set(usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3])
                m = m.at[..., 4].set(udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4])
                m = m.at[..., 5].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5])
                m = m.at[..., 6].set(
                    (ux**2) * uy * T[..., 0]
                    + 2 * ux * uy * T[..., 1]
                    + ux**2 * T[..., 2]
                    + 0.5 * uy * T[..., 3]
                    + 0.5 * uy * T[..., 4]
                    + 2 * ux * T[..., 5]
                    + T[..., 6]
                )
                m = m.at[..., 7].set(
                    (uy**2) * ux * T[..., 0]
                    + uy**2 * T[..., 1]
                    + 2 * ux * uy * T[..., 2]
                    + 0.5 * ux * T[..., 3]
                    - 0.5 * ux * T[..., 4]
                    + 2 * uy * T[..., 5]
                    + T[..., 7]
                )
                m = m.at[..., 8].set(
                    (uy**2 * ux**2) * T[..., 0]
                    + 2 * ux * uy**2 * T[..., 1]
                    + 2 * uy * ux**2 * T[..., 2]
                    + 0.5 * usq * T[..., 3]
                    - 0.5 * udiff * T[..., 4]
                    + 4 * ux * uy * T[..., 5]
                    + 2 * uy * T[..., 6]
                    + 2 * ux * T[..., 7]
                    + T[..., 8]
                )
                return m

            return map(lambda T, u: f(T, u), T_tree, u_tree)

    @partial(jit, static_argnums=(0,))
    def compute_eq_central_moments(self, rho_tree):
        """
        Calculate the central moments of the equilibrium distribution.

        Parameters:
        ----------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density field for all components.

        Returns:
        -------
        T_eq: pytree of jax.numpy.ndarray
            Pytree of central moments of the equilibrium distribution.
        """

        def f(rho):
            if self.lattice.d == 2:
                T_eq = jnp.zeros(
                    (self.nx, self.ny, self.lattice.q),
                    dtype=self.precisionPolicy.compute_dtype,
                )
                T_eq = T_eq.at[..., 0].set(rho[..., 0])
                T_eq = T_eq.at[..., 3].set(2 * rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs**4)

                return T_eq

        return map(lambda rho: f(rho), rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, psi_tree):
        """
        Calculate the central moments of the force distribution. Includes modification to accurately replicate mechanical stability conditions.

        Parameters:
        ----------
        F_tree: pytree of jax.numpy.ndarray
            Pytree of force field for all components.
        psi_tree: pytree of jax.numpy.ndarray
            Pytree of potential field for all components.

        Returns:
        -------
        T_eq: pytree of jax.numpy.ndarray
            Pytree of central moments of the force distribution.
        """

        def f(F, sigma, psi, s_b):
            if self.lattice.d == 2:
                T_eq = jnp.zeros(
                    (self.nx, self.ny, self.lattice.q),
                    dtype=self.precisionPolicy.compute_dtype,
                )
                Fx = F[..., 0]
                Fy = F[..., 1]
                eta = 4 * sigma * (Fx**2 + Fy**2) / (psi[..., 0] ** 2 * (1 / s_b - 0.5))
                T_eq = T_eq.at[..., 1].set(Fx)
                T_eq = T_eq.at[..., 2].set(Fy)
                T_eq = T_eq.at[..., 3].set(eta)
                T_eq = T_eq.at[..., 6].set(Fy * self.lattice.cs2)
                T_eq = T_eq.at[..., 7].set(Fx * self.lattice.cs2)
                T_eq = T_eq.at[..., 8].set(eta * self.lattice.cs2)

                return T_eq

        return map(
            lambda F, sigma, psi, s_b: f(F, sigma, psi, s_b),
            F_tree,
            self.sigma,
            psi_tree,
            self.s_b,
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
        Tdash_tree: pytree of jax.numpy.ndarray
            pytree of central moments post-collision distribution functions.
        rho_tree: pytree of jax.numpy.ndarray
            pytree of density field for all components.
        u_tree: pytree of jax.numpy.ndarray
            pytree of velocity field for all components.

        Returns
        -------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions with the force applied.
        """
        F_tree = self.compute_force(rho_tree)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, psi_tree)
        Tf_tree = map(
            lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S),
            self.S,
            C_tree,
        )
        return map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    # @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    # def collision(self, fin_tree):
    #     """
    #     Cascaded LBM collision step for lattice.
    #     """
    #     fin_tree = map(lambda f: self.precisionPolicy.cast_to_compute(f), fin_tree)
    #     rho_tree, u_tree = self.update_macroscopic(fin_tree)
    #     T_tree = map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
    #     N_tree = self.compute_shift_matrix(u_tree)
    #     Tdash_tree = self.compute_central_moment(T_tree, N_tree)
    #     Tdash_eq_tree = self.compute_eq_central_moments(rho_tree)
    #     Tout_tree = map(
    #         lambda Tdash, Tdash_eq, S: -jnp.dot(Tdash - Tdash_eq, S),
    #         Tdash_tree,
    #         Tdash_eq_tree,
    #         self.S,
    #     )
    #     Tout_tree = self.apply_force(Tout_tree, Tdash_eq_tree, rho_tree, u_tree)
    #     Ninv_tree = self.compute_shift_matrix_inverse(u_tree)
    #     Tout_tree = self.compute_central_moment(Tout_tree, Ninv_tree)
    #     fout_tree = map(
    #         lambda fin, T, Minv: fin + jnp.dot(T, Minv), fin_tree, Tout_tree, self.M_inv
    #     )
    #     return map(
    #         lambda fout: self.precisionPolicy.cast_to_output(fout),
    #         fout_tree,
    #     )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        """
        Cascaded LBM collision step for lattice.
        """
        fin_tree = map(lambda f: self.precisionPolicy.cast_to_compute(f), fin_tree)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        T_tree = map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
        Tdash_tree = self.compute_central_moment(T_tree, u_tree)
        Tdash_eq_tree = self.compute_eq_central_moments(rho_tree)
        Tout_tree = map(
            lambda Tdash, Tdash_eq, S: jnp.dot(Tdash, jnp.eye(self.lattice.q) - S) + jnp.dot(Tdash_eq, S),
            Tdash_tree,
            Tdash_eq_tree,
            self.S,
        )
        Tout_tree = self.apply_force(Tout_tree, rho_tree, u_tree)
        Tout_tree = self.compute_central_moment_inverse(Tout_tree, u_tree)
        fout_tree = map(lambda T, Minv: jnp.dot(T, Minv), Tout_tree, self.M_inv)
        return map(
            lambda fout: self.precisionPolicy.cast_to_output(fout),
            fout_tree,
        )
