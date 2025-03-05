from functools import partial

from jax import jit
from jax.tree import map

import jax.numpy as jnp


class EOS:
    """
    Base class for all equation of state
    """

    def __init__(self, **kwargs):
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.R = kwargs.get("R")
        self.T = kwargs.get("T")

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if value is None:
            raise ValueError("Gas constant value must be provided")
        if isinstance(value, float) or isinstance(value, int):
            self._R = [value]
        elif isinstance(value, list):
            self._R = value
        else:
            raise ValueError(
                "Gas constant must be int, float or a list (for a multi-component flows)"
            )

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value is None:
            raise ValueError("Temperature value must be provided")
        if value < 0:
            raise ValueError("Temperature cannot be negative")
        self._T = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("EOS parameter a must be provided EOS")
        if isinstance(value, float) or isinstance(value, int):
            self._a = [value]
        elif isinstance(value, list):
            self._a = value
        else:
            raise ValueError(
                "EOS parameter a must be int, float or a list (for multi-component flows)"
            )

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("EOS parameter b must be provided EOS")
        if isinstance(value, float) or isinstance(value, int):
            self._b = [value]
        elif isinstance(value, list):
            self._b = value
        else:
            raise ValueError(
                "EOS parameter b must be int, float or a list (for multi-component flows)"
            )

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        pass


class VanderWaal(EOS):
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

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - a * rho**2
        return map(eos, self.a, self.b, self.R, rho_tree)


class Redlich_Kwong(EOS):
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

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (
            a * rho**2
        ) / (jnp.sqrt(self.T) * (1.0 + b * rho))
        return map(eos, self.a, self.b, self.R, rho_tree)


class Redlich_Kwong_Soave(EOS):
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
        self.rks_omega = kwargs.get("RKS_omega")
        self.set_alpha()

    @property
    def rks_omega(self):
        return self._rks_omega

    @rks_omega.setter
    def rks_omega(self, value):
        if value is None:
            raise ValueError(
                "rks_omega value must be provided for using Redlich-Kwong EOS"
            )
        self._rks_omega = value

    def set_alpha(self):
        Tc_tree = map(
            lambda a, b, R: (a / b) * (0.08664 / 0.42784) * (1 / R),
            self.a,
            self.b,
            self.R,
        )
        self.alpha = map(
            lambda rks_omega, Tc: (
                1
                + (0.480 + 1.574 * rks_omega - 0.176 * rks_omega**2)
                * (1 - jnp.sqrt(self.T / Tc))
            )
            ** 2,
            self.rks_omega,
            Tc_tree,
        )

    def set_temperature(self, T):
        self.T = T
        self.set_alpha()

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (
            a * alpha * rho**2
        ) / (1.0 + b * rho)
        return map(eos, self.a, self.b, self.alpha, self.R, rho_tree)


class Peng_Robinson(EOS):
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
        self.pr_omega = kwargs.get("pr_omega")
        self.set_alpha()

    @property
    def pr_omega(self):
        return self._pr_omega

    @pr_omega.setter
    def pr_omega(self, value):
        if value is None:
            raise ValueError(
                "pr_omega value must be provided for using Peng-Robinson EOS"
            )
        if isinstance(value, int) or isinstance(value, float):
            self._pr_omega = [value]
        if isinstance(value, list):
            self._pr_omega = value

    def set_alpha(self):
        Tc_tree = map(
            lambda a, b, R: (a / b) * (0.0778 / 0.45724) * (1 / R),
            self.a,
            self.b,
            self.R,
        )
        self.alpha = map(
            lambda pr_omega, Tc: (
                1
                + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2)
                * (1 - jnp.sqrt(self.T / Tc))
            )
            ** 2,
            self.pr_omega,
            Tc_tree,
        )

    def set_temperature(self, T):
        self.T = T
        self.set_alpha()

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (
            a * alpha * rho**2
        ) / (1.0 + 2 * b * rho - b**2 * rho**2)
        return map(eos, self.a, self.b, self.alpha, self.R, rho_tree)


class Carnahan_Starling(EOS):
    """
    Define multiphase model using the Carnahan-Starling EOS.

    Parameters
    ----------
    a: float or list
    b: float or list
    R: float or list

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

    def set_temperature(self, T):
        self.T = T

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        x_tree = map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        eos = lambda a, R, rho, x: (
            rho * R * self.T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)
        ) - (a * rho**2)
        return map(eos, self.a, self.R, rho_tree, x_tree)
