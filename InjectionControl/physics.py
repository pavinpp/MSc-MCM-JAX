"""
physics.py
----------
Centralized Differentiable Physics Engine for JAX-LaB.
Contains all physical constants, LBM kernels, and boundary conditions
extracted from Baseline_Comparison.ipynb to ensure consistency across
all experimental notebooks.
"""

import jax
import jax.numpy as jnp
from jax import jit, checkpoint

# --- 1. GLOBAL PHYSICAL CONSTANTS (Single Source of Truth) ---
# Grid Dimensions
NX, NY = 100, 50

# Fluid Properties
TAU_BRINE = 1.0
TAU_CO2 = 0.9
RHO_BRINE = 1.0
RHO_CO2_INIT = 0.1

# Multiphase Interaction
G_INT = -1.0  # Interaction Strength

# Solute & Reaction Properties
D_SALT = 0.05
K_SP = 1.1    # Solubility Product (Precipitation Threshold)

# Lattice Constants (D2Q9)
W = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CX = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

# --- 2. CORE LBM KERNELS (JIT Compiled) ---

@jit
def get_equilibrium(rho, u_x, u_y):
    """Calculates the Equilibrium Distribution Function (Fe)"""
    u_x_exp = jnp.expand_dims(u_x, axis=-1)
    u_y_exp = jnp.expand_dims(u_y, axis=-1)
    rho_exp = jnp.expand_dims(rho, axis=-1)
    
    u_sq = u_x**2 + u_y**2
    u_sq_exp = jnp.expand_dims(u_sq, axis=-1)
    
    eu = (CX * u_x_exp + CY * u_y_exp)
    return rho_exp * W * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq_exp)

@jit
def interaction_force(rho, G):
    """Calculates Shan-Chen Multiphase Interaction Force"""
    rho_safe = jnp.clip(rho, 1e-3, 5.0)
    psi = 1 - jnp.exp(-rho_safe)
    
    psi_xp, psi_xm = jnp.roll(psi, -1, axis=0), jnp.roll(psi, 1, axis=0)
    psi_yp, psi_ym = jnp.roll(psi, -1, axis=1), jnp.roll(psi, 1, axis=1)
    
    fx = -G * psi * (psi_xp - psi_xm)
    fy = -G * psi * (psi_yp - psi_ym)
    return fx, fy

@jit
def collision_stream(f, salt_conc, mask, tau):
    """
    Standard LBM Step: 
    1. Macroscopic Moments
    2. Force Calculation
    3. Collision
    4. Streaming
    5. Bounce-back (Mask)
    """
    rho = jnp.sum(f, axis=-1)
    rho_safe = jnp.maximum(rho, 1e-3)
    
    u_x = jnp.sum(f * CX, axis=-1) / rho_safe
    u_y = jnp.sum(f * CY, axis=-1) / rho_safe
    
    # Stability Clipping
    u_x = jnp.clip(u_x, -0.4, 0.4)
    u_y = jnp.clip(u_y, -0.4, 0.4)
    
    # Multiphase Forces
    fx, fy = interaction_force(rho, G_INT)
    u_x += fx / rho_safe
    u_y += fy / rho_safe
    
    # Salt Drag (Clogging Effect)
    # Reduces velocity in high salt concentration areas
    precip_factor = jax.nn.sigmoid(10 * (salt_conc - K_SP)) 
    u_x *= (1.0 - precip_factor) 
    u_y *= (1.0 - precip_factor)

    # Collision
    f_eq = get_equilibrium(rho, u_x, u_y)
    tau_exp = jnp.expand_dims(tau, axis=-1)
    f_out = f - (f - f_eq) / tau_exp
    
    # Streaming
    for i in range(9):
        f_out = f_out.at[..., i].set(jnp.roll(f_out[..., i], (CX[i], CY[i]), axis=(0, 1)))
        
    # Bounce-Back Boundary (Rock Mask)
    mask_exp = jnp.expand_dims(mask, axis=-1)
    f_out = f_out * (1 - mask_exp) + f * mask_exp
    
    return jnp.nan_to_num(f_out, nan=0.0), rho, u_x, u_y

# --- 3. RIGOROUS BOUNDARY CONDITIONS (Zou-He) ---

@jit
def zou_he_inlet_pressure(f, rho_target):
    """
    Applies Constant Pressure at Left Wall (x=0).
    Fixes Density (rho), Calculates Velocity (u).
    Used for: Hammer, Chirp, Sine, BioMimetic strategies.
    """
    # Populations streaming out of domain (knowns)
    f0, f2, f3, f4, f6, f7 = f[...,0], f[...,2], f[...,3], f[...,4], f[...,6], f[...,7]
    
    # Calculate Velocity based on Fixed Rho
    sum_known = f0 + f2 + f4 + 2*(f3 + f6 + f7)
    u_inlet = 1.0 - (sum_known / rho_target)
    
    # Update Unknowns (f1, f5, f8 pointing East)
    f1 = f3 + (2.0/3.0) * rho_target * u_inlet
    f5 = f7 - 0.5*(f2 - f4) + (1.0/6.0) * rho_target * u_inlet
    f8 = f6 + 0.5*(f2 - f4) + (1.0/6.0) * rho_target * u_inlet
    
    f_out = f
    f_out = f_out.at[0, :, 1].set(f1[0, :])
    f_out = f_out.at[0, :, 5].set(f5[0, :])
    f_out = f_out.at[0, :, 8].set(f8[0, :])
    
    return f_out, u_inlet

@jit
def zou_he_inlet_velocity(f, u_target):
    """
    Applies Constant Rate at Left Wall (x=0).
    Fixes Velocity (u), Calculates Density (rho).
    Used for: Constant Rate Baseline.
    """
    f0, f2, f3, f4, f6, f7 = f[...,0], f[...,2], f[...,3], f[...,4], f[...,6], f[...,7]
    
    # Calculate Rho based on Fixed Velocity
    rho_inlet = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1.0 - u_target)
    
    f1 = f3 + (2.0/3.0) * rho_inlet * u_target
    f5 = f7 - 0.5*(f2 - f4) + (1.0/6.0) * rho_inlet * u_target
    f8 = f6 + 0.5*(f2 - f4) + (1.0/6.0) * rho_inlet * u_target
    
    f_out = f
    f_out = f_out.at[0, :, 1].set(f1[0, :])
    f_out = f_out.at[0, :, 5].set(f5[0, :])
    f_out = f_out.at[0, :, 8].set(f8[0, :])
    
    return f_out, rho_inlet

# --- 4. STANDARDIZED SIMULATION STEP ---

@checkpoint
def lbm_step_pressure(carry, pressure_signal):
    """
    A unified step function for ALL pressure-based optimization strategies
    (Hammer, Sine, Chirp, BioMimetic).
    
    Args:
        carry: Tuple (f, salt, mask)
        pressure_signal: The specific pressure value P(t) for this time step.
        
    Returns:
        (new_carry), (metrics)
    """
    f, salt, mask = carry
    
    # 1. Physics Step
    rho_local = jnp.sum(f, axis=-1)
    
    # Linear Interpolation for Tau based on Saturation
    tau_eff = TAU_CO2 + (TAU_BRINE - TAU_CO2) * (rho_local - RHO_CO2_INIT)/(RHO_BRINE - RHO_CO2_INIT)
    tau_safe = jnp.maximum(tau_eff, 0.52)
    
    f, rho, ux, uy = collision_stream(f, salt, mask, tau_safe)
    
    # 2. Apply Boundary Condition (Standardized Zou-He)
    # Note: RHO_CO2_INIT is the base density. pressure_signal adds to it.
    rho_target = RHO_CO2_INIT + pressure_signal
    f_new, u_inlet_actual = zou_he_inlet_pressure(f, rho_target)
    
    # 3. Advection-Diffusion for Salt
    ux_s = jnp.clip(ux, -0.2, 0.2)
    uy_s = jnp.clip(uy, -0.2, 0.2)
    
    grad_salt_x = (jnp.roll(salt, -1, axis=0) - jnp.roll(salt, 1, axis=0)) / 2.0
    grad_salt_y = (jnp.roll(salt, -1, axis=1) - jnp.roll(salt, 1, axis=1)) / 2.0
    laplacian = (jnp.roll(salt, -1, axis=0) + jnp.roll(salt, 1, axis=0) + 
                 jnp.roll(salt, -1, axis=1) + jnp.roll(salt, 1, axis=1) - 4*salt)
    
    salt_new = salt + (-(ux_s * grad_salt_x + uy_s * grad_salt_y) + D_SALT * laplacian)
    
    # Boundary: Zero Salt at Inlet (Fresh CO2)
    salt_new = salt_new.at[0, :].set(0.0)
    salt_new = jnp.clip(salt_new, 0.0, 5.0)
    
    # 4. Precipitation Logic
    # If Salt > K_SP, porosity reduces (mask increases)
    new_precip = jax.nn.sigmoid(20 * (salt_new - K_SP))
    mask_new = jnp.maximum(mask, new_precip)
    
    # 5. Metrics Calculation
    pore_vol = jnp.sum(1-mask_new) + 1e-6
    # CO2 Saturation: fraction of pore space filled with CO2 (low density fluid)
    s_co2 = jnp.sum((rho < (RHO_BRINE + RHO_CO2_INIT)/2.0) * (1-mask_new)) / pore_vol
    
    # Track injectivity (mean inlet velocity)
    inlet_flux = jnp.mean(u_inlet_actual[0, :])
    
    return (f_new, salt_new, mask_new), (s_co2, inlet_flux)