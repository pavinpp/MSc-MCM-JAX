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
from jax import jit
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
# Third-party libraries
from jax.sharding import PartitionSpec
from jax.tree import map, reduce
from termcolor import colored

from src.lattice import LatticeD2Q9, LatticeD3Q19
# User-defined libraries
from src.models import BGKSim
from src.utils import downsample_field

jax.config.update("jax_debug_nans", True)


class Multiphase(BGKSim):
    """
    Multiphase model, based on the Shan-Chen method. To model the fluid, an equation of state (EOS) is defined by the user.
    Sequence of computation is pressure (EOS, dependent on the density and temperature) --> effective mass (phi).
    Can model both single component multiphase (SCMP) and multi-component multiphase (MCMP).

    Parameters
    ----------
    R: float
        Gas constant
    T: float
        Temperature
    beta: float
        Weighting factor used for force calculation.
    g_kk: numpy.ndarray
        Inter component interaction strength. Its a matrix of size n_components x n_components. It must be symmetric.
    g_ks: list
        Component-wall interaction strength. Its a vector of size (n_components,).

    References
    ----------
    1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
        Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.”
        Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.

    3. Pan, C., M. Hilpert, and C. T. Miller. “Lattice-Boltzmann Simulation of Two-Phase Flow in Porous Media.”
        Water Resources Research 40, no. 1 (2004). https://doi.org/10.1029/2003WR002120.

    4. Kang, Qinjun, Dongxiao Zhang, and Shiyi Chen. “Displacement of a Two-Dimensional Immiscible Droplet in a Channel.”
        Physics of Fluids 14, no. 9 (September 1, 2002): 3203-14. https://doi.org/10.1063/1.1499125

    5. Gong, Shuai, and Ping Cheng. “Numerical Investigation of Droplet Motion and Coalescence by an Improved Lattice Boltzmann
        Model for Phase Transitions and Multiphase Flows.” Computers & Fluids 53 (January 15, 2012): 93-104.
        https://doi.org/10.1016/j.compfluid.2011.09.013.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R = kwargs.get("R", 0.0)
        self.T = kwargs.get("T", 0.0)
        self.beta = kwargs.get("beta")
        self.n_components = kwargs.get("n_components")
        self.g_kkprime = kwargs.get("g_kkprime")  # Fluid-fluid interaction strength
        self.g_ks = kwargs.get("g_ks")  # Fluid-solid interaction strength
        self.force = kwargs.get("body_force", None)

        self.G_ff = self.compute_ff_greens_function()
        self.G_fs = self.compute_fs_greens_function()

        # self.omega = jnp.array(self.omega, dtype=self.precisionPolicy.compute_dtype)
        self.g_kkprime = jnp.array(
            self.g_kkprime, dtype=self.precisionPolicy.compute_dtype
        )

        self.solid_mask_streamed = self.get_solid_mask_streamed()
        

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if not isinstance(value, list) and not isinstance(value, jax.numpy.ndarray):
            raise ValueError("omega must be a list or jax.numpy.ndarray")
        self._omega = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        if value is None:
            raise ValueError("n_components must be provided for multiphase simulation")
        if not isinstance(value, int):
            raise ValueError("n_components must be an integer")
        self._n_components = value

    @property
    def g_kkprime(self):
        return self._g_kkprime

    @g_kkprime.setter
    def g_kkprime(self, value):
        if not isinstance(value, np.ndarray) and not isinstance(
            value, jax.numpy.ndarray
        ):
            raise ValueError("g_kkprime must be a numpy array or jax.numpy.ndarray")
        if np.shape(value) != (self.n_components, self.n_components):
            raise ValueError(
                "g_kkprime must be a matrix of size n_components x n_components"
            )
        if not np.allclose(value, np.transpose(value), atol=1e-6):
            raise ValueError("g_kkprime must be a symmetric matrix")
        self._g_kkprime = np.array(value)

    @property
    def g_ks(self):
        return self._g_ks

    @g_ks.setter
    def g_ks(self, value):
        if len(value) != self.n_components:
            raise ValueError("g_ks must be a list size n_components")
        self._g_ks = value

    def get_solid_mask_streamed(self):
        """
        Define the solid mask used for fluid-solid interaction force.

        Parameters
        ----------
        None
        
        Returns
        -------
        numpy.ndarray: solid_mask array. Dimension: (nx, ny, 1) for d == 2 and (nx, ny, nz, 1) for d == 3
        """
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None
        shape = (self.nx, self.ny, 1) if self.dim == 2 else (self.nx, self.ny, self.nz, 1)
        self.solid_mask = np.zeros(shape=shape, dtype=np.int8)
        solid_mask_repeated = jnp.repeat(jnp.array(self.solid_mask, dtype=jnp.int8), axis=-1, repeats=self.q)
        return self.streaming(solid_mask_repeated)

    @partial(jit, static_argnums=(0, 3))
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
        feq: pytree of jax.numpy.ndarray
            Pytree of equillibrium distribution.
        """
        if cast_output:
            cast = lambda x: self.precisionPolicy.cast_to_compute(x)
            rho_tree  = map(cast, rho_tree)
            u_tree  = map(cast, u_tree)

        c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype)
        cu_tree = map(lambda u: 3.0 * jnp.dot(u, c), u_tree)
        usqr_tree = map(
            lambda u: 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True), u_tree
        )
        feq_tree = map(
            lambda rho, udote, udotu: rho
                * self.w
                * (1.0 + udote * (1.0 + 0.5 * udote) - udotu),
            rho_tree,
            cu_tree,
            usqr_tree,
        )

        if cast_output:
            return map(
                lambda f_eq: self.precisionPolicy.cast_to_output(f_eq), feq_tree
            )
        else:
            return feq_tree

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        """
        BGK collision step for lattice, extended to pytrees.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        fin_tree = map(
            lambda fin: self.precisionPolicy.cast_to_compute(fin), fin_tree
        )
        rho_tree, u_tree = self.compute_macroscopic_variables(fin_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        fneq_tree = map(lambda feq, fin: feq - fin, feq_tree, fin_tree)
        fout_tree = map(
            lambda fin, fneq, omega: fin + omega * fneq, fin_tree, fneq_tree, self.omega
        )

        fout_tree = self.apply_force(fout_tree, feq_tree, rho_tree, u_tree)

        return map(
            lambda fout: self.precisionPolicy.cast_to_output(fout), fout_tree
        )

    def compute_ff_greens_function(self):
        """
        Define the Green's function used to model interaction of kth fluid with all components.

        During computation, this G_ff is multiplied with corresponding g_kkprime value to get the Green's function:
        G_kkprime = self.g_kk[k, k_prime] * self.G_ff

        Green's function used in this case:

        G_ff(x, x') = g1,                   if |x - x'| = 1
                    = g2,                   if |x - x'| = sqrt(2)
                    = 0,                    otherwise

        which when multiplied with g_kkprime gives:

        G_kkprime(x, x') = g1 * g_kkprime,  if |x - x'| = 1
                         = g2 * g_kkprime,  if |x - x'| = sqrt(2)
                         = 0,               otherwise

        Here d is the dimension of problem and x' are the neighboring points.

        Some examples values could be:
        For D2Q9:
            g1 = 2 and g2 = 1/2
        For D3Q19
            g1 = 1 and g2 = 1/2

        By default, lattice weights (self.lattice.w) is used as G_ff

        Parameters
        ----------
        None

        Returns
        -------
        G_ff: jax.numpy.ndarray.
            Dimension: (q, )
        """
        # e = np.array(self.lattice.e).T
        # G_ff = np.zeros((self.q,), dtype=np.float64)
        # el = np.linalg.norm(e, axis=-1)
        # g2 = 0.5
        # if isinstance(self.lattice, LatticeD2Q9):
        #     g1 = 2.0
        # elif isinstance(self.lattice, LatticeD3Q19):
        #     g1 = 1.0
        # G_ff[np.isclose(el, 1.0, atol=1e-6)] = g1
        # G_ff[np.isclose(el, jnp.sqrt(2.0), atol=1e-6)] = g2
        # return jnp.array(G_ff, dtype=self.precisionPolicy.compute_dtype)
        return jnp.array(self.lattice.w, dtype=self.precisionPolicy.compute_dtype)

    def compute_fs_greens_function(self):
        """
        Define the Green's function used to model interaction between kth fluid and solid.

        During computation, this G_fs is multiplied with corresponding g_ks value to get the Green's function:
        G_ks = self.g_ks[k] * self.G_fs

        Green's function used in this case:

        G_fs(x, x') = g1,              if |x - x'| = 1
                    = g2,              if |x - x'| = sqrt(2)
                    = 0,               otherwise

        which when multiplied with g_ks gives:

        G_ks(x, x') = g1 * g_ks,       if |x - x'| = 1
                    = g2 * g_ks,       if |x - x'| = sqrt(2)
                    = 0,               otherwise

        Here d is the dimension of problem and x' are the neighboring points.
        For D2Q9:
            g1 = 2 and g2 = 1/2
        For D3Q19
            g1 = 1 and g2 = 1/2

        Again, d is the dimension of the problem

        Parameters
        ----------
        None

        Returns
        -------
        G_fs: jax.numpy.ndarray
            Dimension: (q, )
        """
        c = np.array(self.lattice.c).T
        G_fs = np.zeros((self.q,), dtype=np.float64)
        el = np.linalg.norm(c, axis=-1)
        g2 = 0.5
        if isinstance(self.lattice, LatticeD2Q9):
            g1 = 2.0
        elif isinstance(self.lattice, LatticeD3Q19):
            g1 = 1.0
        G_fs[np.isclose(el, 1.0, atol=1e-6)] = g1
        G_fs[np.isclose(el, jnp.sqrt(2.0), atol=1e-6)] = g2
        return jnp.array(G_fs, dtype=self.precisionPolicy.compute_dtype)

    def initialize_macroscopic_fields(self):
        """
        Functions to initialize the pytrees of density and velocity arrays with their corresponding initial values.
        By default, velocities is set as 0 everywhere and density as 1.0.

        Note:
            Function must be overwritten in a subclass or instance of the class to not use the default values.

        Parameters
        ----------
        None by default, can be overwritten as required

        Returns
        -------
        None, None: The default density and velocity values, both None.
        This indicates that the actual values should be set elsewhere.
        """
        print(
            "Default initial conditions assumed for the missing entries in the dictionary: density = 1.0 and velocity = 0.0"
        )
        print(
            "To set explicit initial values for velocity and density, use the self.initialize_macroscopic_fields function"
        )
        return None, None

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
        shape = (self.nx, self.ny, self.q) if self.dim == 2 else (self.nx, self.ny, self.nz, self.q)
        f_tree = []
        if rho0_tree is not None and u0_tree is not None:
            assert (
                len(rho0_tree) == self.n_components
            ), "The initial density values for all components must be provided"

            assert (
                len(u0_tree) == self.n_components
            ), "The initial velocity values for all components must be provided."

            for i in range(self.n_components):
                rho0, u0 = rho0_tree[i], u0_tree[i]
                f_tree.append(self.initialize_populations(rho0, u0))
        else:
            for i in range(self.n_components):
                f_tree.append(
                    self.distributed_array_init(
                        shape, self.precisionPolicy.output_dtype, init_val=self.w
                    )
                )
        return f_tree

    @partial(jit, static_argnums=(0,))
    def compute_rho(self, f_tree):
        """
        Compute the number density for all fluids using the respective distribution function.

        Parameters
        ----------
        f_tree: pytree of jax.numpy.ndarray
            Pytree of distribution arrays.

        Returns
        -------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density values.
        """
        rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        return rho_tree

    @partial(jit, static_argnums=(0,))
    def compute_macroscopic_variables(self, f_tree,):
        """
        compute_macroscopic_variables from base.py extended to pytrees.

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
        F_tree = self.compute_force(f_tree)
        u_tree = map(
            lambda f, rho, F: (jnp.dot(f, self.c.T) + 0.5*F) / rho, f_tree, rho_tree, F_tree
        )  # Component velocity
        return rho_tree, u_tree

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
                lambda omega, rho, u: omega * rho * u, self.omega, rho_tree, u_tree
            ),
        )
        d = reduce(
            operator.add, map(lambda omega, rho: rho * omega, self.omega, rho_tree)
        )
        return n / d

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
            Total velocity values.
        """
        return reduce(
            operator.add, map(lambda rho, omega: omega * rho, rho_tree, self.omega)
        )

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        """
        Define the equation of state for the problem. Defined in sub-classes.

        Parameters
        ----------
        rho_tree: jax.numpy.ndarray
            Pytree of density values.

        Returns
        -------
        p_tree: pytree of jax.numpy.ndarray
            Pytree of pressure values.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho_tree):
        """
        Compute the effective mass (psi) for each component which is required for computing interaction forces.
        The psi values are obtained using the corresponding EOS.

        Parameters
        ----------
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density values.

        Returns
        -------
        psi_tree: pytree of jax.numpy.ndarray
        """
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        p_tree = self.EOS(rho_tree)
        g_tree = list(self.g_kkprime.diagonal())
        psi = lambda p, rho, g: jnp.sqrt(
            jnp.abs(2 * (p - self.lattice.cs2 * rho) / (self.lattice.cs2 * g))
        )
        psi_tree = map(
            psi, p_tree, rho_tree, g_tree
        )  # Exact value of g does not matter
        return psi_tree

    # Compute the force using the effective mass (psi) and the interaction potential (phi)
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def compute_force(self, f_tree):
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
        rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        psi_tree = self.compute_psi(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree)
        fluid_solid_force = self.compute_fluid_solid_force(rho_tree)
        return map(lambda ff, fs: ff + fs, fluid_fluid_force, fluid_solid_force)

    # @partial(jit, static_argnums=(0,))
    # def compute_fluid_fluid_force(self, psi_tree):
    #     """
    #     Compute the fluid-fluid interaction force using the effective mass (psi).
    #     The force calculation is based on the Shan-Chen method:
    #     F(x) = -\psi_k(x) \sum_{x'} G(x, x') \psi(x') (x' - x)

    #     Parameters
    #     ----------
    #     psi_tree: pytree of jax.numpy.ndarray
    #         Pytree of effective mass values.
    #     psi_s_tree: pytree of jax.numpy.ndarray
    #         Pytree of streamed effective mass values.
    #
    #     Returns
    #     -------
    #     pytree of jax.numpy.ndarray
    #         Pytree of fluid-fluid interaction force.
    #     """
    #     psi_s_tree = map(
    #         lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)),
    #         psi_tree,
    #     )
    #
    #     def ffk(g_kkprime):
    #         """
    #         g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components
    #         """
    #         f_int = (
    #             lambda g, psi: -0.5
    #             * g
    #             * jnp.dot(self.G_ff * psi, self.e.T)
    #         )
    #         return tree_reduce(
    #             operator.add,
    #             map(
    #                 f_int,
    #                 list(g_kkprime),
    #                 psi_tree,
    #             ),
    #         )
    #
    #     return map(
    #         lambda psi, nt: -psi * nt,
    #         psi_tree,
    #         list(jax.vmap(ffk, in_axes=0)(self.g_kkprime)),
    #     )

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi). The improved force model works better with lower temperatures,
        (where the original interaction force scheme underperformed, significantly deviating from the Maxwell's co-existence curve).

        Parameters
        ----------
            psi_tree: pytree of jax.numpy.ndarray
                Pytree of effective mass values.
            psi_s_tree: pytree of jax.numpy.ndarray
                Pytree of streamed effective mass values.

        Returns
        -------
        pytree of jax.numpy.ndarray
            Pytree of fluid-fluid interaction force.

        Reference
        ---------
        1. Gong, Shuai, and Ping Cheng. “Numerical Investigation of Droplet Motion and Coalescence by an Improved Lattice Boltzmann
            Model for Phase Transitions and Multiphase Flows.” Computers & Fluids 53 (January 15, 2012): 93-104.
            https://doi.org/10.1016/j.compfluid.2011.09.013.
        """
        psi_s_tree = map(
            lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)),
            psi_tree,
        )
        psi_s2_tree = map(
            lambda psi: self.streaming(
                jnp.repeat(jnp.square(psi), axis=-1, repeats=self.q)
            ),
            psi_tree,
        )

        def ffk_1(g_kkprime):
            """
            g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components
            """
            f_int_1 = (
                lambda g, psi_s: jnp.dot(g * self.G_ff * psi_s, self.c.T)
            )
            return reduce(
                operator.add,
                map(
                    f_int_1,
                    list(g_kkprime),
                    psi_s_tree,
                ),
            )

        def ffk_2(g_kkprime):
            """
            g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components
            """
            f_int_2 = (
                lambda g, psi_s2: jnp.dot(g * self.G_ff * psi_s2, self.c.T)
            )
            return reduce(
                operator.add,
                map(
                    f_int_2,
                    list(g_kkprime),
                    psi_s2_tree,
                ),
            )

        return map(
            lambda psi, nt_1, nt_2: self.beta * psi * nt_1 + 0.5*(1.0 - self.beta) * nt_2,
            psi_tree,
            list(jax.vmap(ffk_1, in_axes=0)(self.g_kkprime)),
            list(jax.vmap(ffk_2, in_axes=0)(self.g_kkprime)),
        )

    @partial(jit, static_argnums=(0,))
    def compute_fluid_solid_force(self, rho_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi).

        Parameters
        ----------
        rho_tree: Pytree of jax.numpy.ndarray
            Pytree of density of all components.

        Returns
        -------
        Pytree of jax.numpy.ndarray
            Pytree of fluid-solid interaction force.
        """
        neighbor_terms = map(
            lambda g_ks: g_ks * jnp.dot(self.G_fs * self.solid_mask_streamed, self.c.T),
            self.g_ks,
        )
        return map(lambda rho, nt: -rho * nt, rho_tree, neighbor_terms)

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
        F_tree = self.compute_force(f_postcollision_tree)
        if self.force is not None:
            delta_u_tree = map(lambda F: (F + self.force), F_tree)

        delta_u_tree = map(lambda F, rho: F / rho, F_tree, rho_tree)
        u_temp_tree = map(lambda u, delta_u: u + delta_u, u_tree, delta_u_tree)
        feq_force_tree = self.equilibrium(rho_tree, u_temp_tree)
        update_distribution = (
            lambda f_postcollision, feq_force, feq: f_postcollision + feq_force - feq
        )
        return map(
            update_distribution, f_postcollision_tree, feq_force_tree, feq_tree
        )

    @partial(jit, static_argnums=(0, 4))
    def apply_bc(
        self, fout_tree, fin_tree, timestep, implementation_step
    ):
        """
        This function extends apply_bc to pytrees.

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        jax.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout_tree = map(
                lambda fin, fout: bc.prepare_populations(
                    fout, fin, implementation_step
                ),
                fin_tree,
                fout_tree,
            )
            if bc.implementation_step == implementation_step:
                if bc.isDynamic:
                    fout_tree = map(
                        lambda fin, fout: bc.apply(fout, fin, timestep),
                        fin_tree,
                        fout_tree,
                    )
                else:
                    fout_tree = map(
                        lambda fin, fout: fout.at[bc.indices].set(
                            bc.apply(fout, fin)
                        ),
                        fin_tree,
                        fout_tree,
                    )
        return fout_tree

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
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
        f_postcollision_tree = self.apply_bc(
            f_postcollision_tree, f_poststreaming_tree, timestep, "post_collision"
        )
        f_poststreaming_tree = map(
            lambda f_postcollision: self.streaming(f_postcollision),
            f_postcollision_tree,
        )
        f_poststreaming_tree = self.apply_bc(
            f_poststreaming_tree, f_postcollision_tree, timestep, "post_streaming"
        )

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
                    raise ValueError(
                        f"Failed to restore checkpoint at step {latest_step}."
                    )

                start_step = latest_step + 1
                if not (t_max  > start_step):
                    raise ValueError(
                        f"Simulation already exceeded maximum allowable steps (t_max  = {t_max}). Consider increasing t_max."
                    )

        if self.computeMLUPS:
            start = time.time()

        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (
                timestep % self.ioRate == 0
                or timestep == t_max
            )
            print_iter_flag = (
                self.printInfoRate > 0 and timestep % self.printInfoRate == 0
            )
            checkpoint_flag = (
                self.checkpointRate > 0 and timestep % self.checkpointRate == 0
            )

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev_tree, u_prev_tree = self.compute_macroscopic_variables(f_tree)
                rho_prev_tree = map(
                    lambda rho_prev: downsample_field(
                        rho_prev, self.downsamplingFactor
                    ),
                    rho_prev_tree,
                )
                p_prev_tree = self.EOS(rho_prev_tree)
                p_prev_tree = map(
                    lambda p_prev: downsample_field(p_prev, self.downsamplingFactor),
                    p_prev_tree,
                )
                u_prev_tree = map(
                    lambda u_prev: downsample_field(u_prev, self.downsamplingFactor),
                    u_prev_tree,
                )
                rho_total_prev = self.compute_total_density(rho_prev_tree)
                u_total_prev = self.compute_total_velocity(rho_prev_tree, u_prev_tree)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_prev_tree = map(
                    lambda p_prev: process_allgather(p_prev), p_prev_tree
                )
                rho_prev_tree = map(
                    lambda rho_prev: process_allgather(rho_prev), rho_prev_tree
                )
                u_prev_tree = map(
                    lambda u_prev: process_allgather(u_prev), u_prev_tree
                )
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
                rho_tree, u_tree = self.compute_macroscopic_variables(f_tree)
                p_tree = self.EOS(rho_tree)
                p_tree = map(
                    lambda p: downsample_field(p, self.downsamplingFactor), p_tree
                )
                rho_tree = map(
                    lambda rho: downsample_field(rho, self.downsamplingFactor),
                    rho_tree,
                )
                u_tree = map(
                    lambda u: downsample_field(u, self.downsamplingFactor), u_tree
                )

                rho_total = self.compute_total_density(rho_tree)
                u_total = self.compute_total_velocity(rho_tree, u_tree)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_tree = map(lambda p: process_allgather(p), p_tree)
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
                    u_total,
                    u_tree,
                    rho_total,
                    rho_tree,
                    p_prev_tree,
                    u_total_prev,
                    u_prev_tree,
                    rho_total_prev,
                    rho_prev_tree,
                )

            if checkpoint_flag:
                # Save the checkpoint
                print(
                    f"Saving checkpoint at timestep {timestep}/{t_max}"
                )
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
                    colored("Domain: ", "blue")
                        + colored(f"{self.nx} x {self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue")
                        + colored(f"{self.nx * self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(
                    colored("MLUPS: ", "blue")
                        + colored(
                            f"{self.nx * self.ny * t_max / (end - start) / 1e6}",
                            "red",
                        )
                )

            elif self.dim == 3:
                print(
                    colored("Domain: ", "blue")
                        + colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue")
                        + colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(
                    colored("MLUPS: ", "blue")
                        + colored(
                            f"{self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                            "red",
                        )
                )

        return f_tree

    def handle_io_timestep(
        self,
        timestep,
        f_tree,
        fstar_tree,
        p_tree,
        u_total,
        u_tree,
        rho_total,
        rho_tree,
        p_prev_tree,
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
        p_tree: pytree of jax.numpy.ndarray
            Pytree of pressure field at the current time step.
        u_total: jax.numpy.ndarray
            Total velocity field at the current time step.
        u_tree: pytree of jax.numpy.ndarray
            Pytree of velocity field at the current time step.
        rho_total: jax.numpy.ndarray
            Total density field at the current time step.
        rho_tree: pytree of jax.numpy.ndarray
            Pytree of density field at the current time step.
        p_prev_tree: pytree of jax.numpy.ndarray
            Pytree of pressure field at the previous time step.
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
            "u_total": u_total,
            "u_tree": u_tree,
            "rho_total_prev": rho_total_prev,
            "rho_prev_tree": rho_prev_tree,
            "p_prev_tree": p_prev_tree,
            "u_total_prev": u_total_prev,
            "u_prev_tree": u_prev_tree,
            "f_poststreaming_tree": f_tree,
            "f_postcollision_tree": fstar_tree,
        }
        self.output_data(**kwargs)


class VanderWaals(Multiphase):
    """
    Define multiphase model using the VanderWaals EOS.

    Parameters
    ----------

    Reference
    ---------
    1. Reprint of: The Equation of State for Gases and Liquids. The Journal of Supercritical Fluids,
    100th year Anniversary of van der Waals' Nobel Lecture, 55, no. 2 (2010): 403–14. https://doi.org/10.1016/j.supflu.2010.11.001.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - a*rho^2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using VanderWaals EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using VanderWaals EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (
            1.0 - self.b * rho
        ) - self.a * jnp.square(rho)
        return map(eos, rho_tree)


class ShanChen(Multiphase):
    """
    Define the multiphase model using the original Shan-Chen EOS. For this class compute_psi is redefined.
    For this case, there is no need to define R and T as they are not used in the EOS.

    Parameters
    ----------
    rho_0: float
        rho_0 used for computing the effective mass (psi)

    Reference
    ---------
    1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
       Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    Notes
    -----
    The expression for psi in this case is:
    psi = rho_0 * (1 - exp(-rho / rho_0))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_0 = kwargs.get("rho_0")

    @property
    def rho_0(self):
        return self._rho_0

    @rho_0.setter
    def rho_0(self, value):
        if value is None:
            raise ValueError("rho_0 value must be provided Shan-Chen EOS")
        self._rho_0 = value

    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho_tree):
        return map(lambda rho: self.rho_0 * (1.0 - jnp.exp(-rho/self.rho_0)), rho_tree)


class Redlich_Kwong(Multiphase):
    """
    Define multiphase model using the Redlich-Kwong EOS.

    Parameters
    ----------

    Reference
    ---------
    1. Redlich O., Kwong JN., "On the thermodynamics of solutions; an equation of state; fugacities of gaseous solutions."
    Chem Rev. 1949 Feb;44(1):233-44. https://doi.org/10.1021/cr60137a013.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*rho^2)/(sqrt(T) * (1 + b*rho))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Redlich-Kwong EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Redlich-Kwong EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * rho**2
        ) / (self.T**0.5 * (1.0 + self.b * rho))
        return map(eos, rho_tree)


class Redlich_Kwong_Soave(Multiphase):
    """
    Define multiphase model using the Redlich-Kwong-Soave EOS.

    Parameters
    ----------

    Reference
    ---------
    1. Giorgio Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state",
    Chemical Engineering Science 27, no. 6(1972), 1197-1203, https://doi.org/10.1016/0009-2509(72)80096-4.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + b*rho)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError(
                "a value must be provided for using Redlich-Kwong-Soave EOS"
            )
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError(
                "b value must be provided for using Redlich-Kwong-Soave EOS"
            )
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Redlich-Kwong EOS")
        self._alpha = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * self.alpha * rho**2
        ) / (1.0 + self.b * rho)
        return map(eos, rho_tree)


class Peng_Robinson(Multiphase):
    """
    Define multiphase model using the Peng-Robinson EOS.

    Parameters
    ----------

    Reference
    ---------
    1. Peng, Ding-Yu, and Donald B. Robinson. "A new two-constant equation of state."
    Industrial & Engineering Chemistry Fundamentals 15, no. 1 (1976): 59-64. https://doi.org/10.1021/i160057a011

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Peng-Robinson EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Peng-Robinson EOS")
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Peng-Robinson EOS")
        self._alpha = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * self.alpha * rho**2
        ) / (1.0 + 2 * self.b * rho - self.b**2 * rho**2)
        return map(eos, rho_tree)


class Carnahan_Starling(Multiphase):
    """
    Define multiphase model using the Carnahan-Starling EOS.

    Parameters
    ----------

    Reference
    ---------
    1.  Carnahan, Norman F., and Kenneth E. Starling. "Equation of state for nonattracting rigid spheres."
    The Journal of chemical physics 51, no. 2 (1969): 635-636. https://doi.org/10.1063/1.1672048

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Carnahan-Starling EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Carnahan-Starling EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = map(
            lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree
        )
        x_tree = map(lambda rho: 0.25 * self.b * rho, rho_tree)
        eos = lambda rho, x: (
            rho * self.R * self.T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)
        ) - (self.a * rho**2)
        return map(eos, rho_tree, x_tree)
