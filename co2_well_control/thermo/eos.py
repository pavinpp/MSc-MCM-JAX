# co2_well_control/thermo/eos.py
import jax
import jax.numpy as jnp

@jax.jit
def peng_robinson_pressure(rho, T, Tc, Pc, omega):
    """
    Calculates pressure using the Peng-Robinson EOS.
    All inputs must be JAX arrays or scalars.
    """
    R = 1.0  # Gas constant in lattice units
    
    # PR-EOS Parameters
    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
    alpha = (1 + kappa * (1 - jnp.sqrt(Tr))) ** 2
    
    a = 0.45724 * (R**2 * Tc**2) / Pc * alpha
    b = 0.07780 * R * Tc / Pc
    
    # Specific volume v = 1/rho
    # To avoid division by zero in empty nodes, we use a safe inverse
    v = jnp.where(rho > 1e-6, 1.0 / rho, 1e6)
    
    # The Pressure Equation: P = RT/(v-b) - a/(v(v+b) + b(v-b))
    term1 = (R * T) / (v - b)
    term2 = a / (v * (v + b) + b * (v - b))
    
    return term1 - term2

@jax.jit
def calculate_pseudopotential(rho, T, Tc, Pc, omega, cs2):
    """
    Calculates the Shan-Chen pseudopotential psi(rho).
    In modern LBM, Force ~ -G * psi(x) * sum(weights * psi(neighbor))
    Using EOS, psi = sqrt(2*(P_eos - rho*cs2) / G)
    """
    # Calculate thermodynamic pressure
    p_eos = peng_robinson_pressure(rho, T, Tc, Pc, omega)
    
    # Calculate Ideal Gas Pressure
    p_ideal = rho * cs2
    
    # Effective Mass/Potential for interaction force
    # We return standard density as a proxy for basic SCMP, 
    # but for EOS-coupled, we return the deviation from ideal behavior.
    # Simplified for Phase 1 stability:
    return rho
