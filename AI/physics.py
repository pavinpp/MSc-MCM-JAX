"""
physics.py
----------
Centralized Differentiable Physics Engine for JAX-LaB.
Contains all physical constants, LBM kernels, and boundary conditions.
FIX: Lattice constants are now Numpy arrays to prevent JIT Tracing errors.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np  # Required for static lattice constants
import os

# --- 1. GLOBAL PHYSICAL CONSTANTS ---
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

# --- LATTICE CONSTANTS (D2Q9) ---
# MUST be numpy arrays (or lists) to avoid "ConcretizationTypeError" 
# when used inside int() or loop indices during JIT compilation.
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

# --- 2. UTILITIES ---
def ensure_results_dir():
    """Ensures the 'results' directory exists for data passing."""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory.")

# --- 3. CORE LBM KERNELS (JIT Compiled) ---
@jit
def get_equilibrium(rho, u_x, u_y):
    """Calculates the Equilibrium Distribution Function (Fe)."""
    # JAX handles numpy constants (W, CX, CY) automatically here
    u_x_exp = jnp.expand_dims(u_x, axis=-1)
    u_y_exp = jnp.expand_dims(u_y, axis=-1)
    rho_exp = jnp.expand_dims(rho, axis=-1)
    
    u_sq = u_x**2 + u_y**2
    u_sq_exp = jnp.expand_dims(u_sq, axis=-1)
    
    # Dot product c_i * u
    cu = (u_x_exp * CX) + (u_y_exp * CY)
    
    return rho_exp * W * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u_sq_exp)

@jit
def collision_stream(f, salt, mask, tau):
    """Standard LBM Collision and Streaming Step."""
    # 1. Macroscopic Moments
    rho = jnp.sum(f, axis=-1)
    # Add epsilon to avoid division by zero
    ux = jnp.sum(f * CX, axis=-1) / (rho + 1e-9)
    uy = jnp.sum(f * CY, axis=-1) / (rho + 1e-9)
    
    # 2. Collision (BGK)
    fe = get_equilibrium(rho, ux, uy)
    tau_exp = jnp.expand_dims(tau, axis=-1)
    f_post = f - (f - fe) / tau_exp
    
    # 3. Streaming (Periodic wrapper, blocked by walls later)
    # We create a new array for the streamed population
    f_stream = jnp.zeros_like(f_post)
    
    # Unroll the 9 directions
    for i in range(9):
        # int(CX[i]) works now because CX is a numpy array (concrete)
        shift_x = int(CX[i])
        shift_y = int(CY[i])
        
        # JAX roll handles the periodic shift
        f_stream = f_stream.at[:, :, i].set(
            jnp.roll(f_post[:, :, i], shift=(shift_x, shift_y), axis=(0, 1))
        )
        
    # 4. Bounce-Back (Rock Mask)
    # If mask=1 (Rock), reflect directions
    mask_exp = jnp.expand_dims(mask, axis=-1)
    # Simple bounce-back index mapping for D2Q9: [0, 3, 4, 1, 2, 7, 8, 5, 6]
    inv_idx = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    f_bounced = f_stream[:, :, inv_idx]
    
    f_new = f_stream * (1 - mask_exp) + f_bounced * mask_exp
    
    return f_new, rho, ux, uy

@jit
def zou_he_inlet_pressure(f, rho_inlet):
    """Zou-He Pressure Boundary Condition at x=0 (Inlet)."""
    # Simple vertical inlet implementation (x=0)
    # We only assume bounce-back for non-streaming directions for simplicity in this 
    # specific differentiable physics demo.
    
    # Correct Rho, solve for velocity
    rho_curr = jnp.sum(f[0, :, :], axis=-1)
    correction = rho_inlet / (rho_curr + 1e-9)
    f_new = f.at[0, :, :].set(f[0, :, :] * jnp.expand_dims(correction, -1))
    
    return f_new, rho_inlet

@jit
def lbm_step_pressure(carry, pressure_signal):
    """
    Main Simulation Loop Step: Controlled by Pressure.
    
    Args:
        carry: (f, salt, mask) tuple
        pressure_signal: float, The P(t) injection pressure at this step
    """
    f, salt, mask = carry
    
    # 1. Physics Step (Collision & Streaming)
    rho_local = jnp.sum(f, axis=-1)
    
    # Interpolate Viscosity (Tau) based on concentration
    # Brine (High Rho) -> TAU_BRINE, CO2 (Low Rho) -> TAU_CO2
    tau_eff = TAU_CO2 + (TAU_BRINE - TAU_CO2) * \
              (rho_local - RHO_CO2_INIT)/(RHO_BRINE - RHO_CO2_INIT)
    tau_safe = jnp.maximum(tau_eff, 0.52) # Numerical stability floor
    
    f_new, rho, ux, uy = collision_stream(f, salt, mask, tau_safe)
    
    # 2. Boundary Condition: Inlet Pressure
    rho_target = RHO_CO2_INIT + pressure_signal
    f_new, _ = zou_he_inlet_pressure(f_new, rho_target)
    
    # 3. Solute Transport (Salt Advection-Diffusion)
    # Clamp velocities for stability
    ux_s = jnp.clip(ux, -0.2, 0.2)
    uy_s = jnp.clip(uy, -0.2, 0.2)
    
    # Finite Difference Gradients
    grad_salt_x = (jnp.roll(salt, -1, axis=0) - jnp.roll(salt, 1, axis=0)) / 2.0
    grad_salt_y = (jnp.roll(salt, -1, axis=1) - jnp.roll(salt, 1, axis=1)) / 2.0
    laplacian = (jnp.roll(salt, -1, axis=0) + jnp.roll(salt, 1, axis=0) + 
                 jnp.roll(salt, -1, axis=1) + jnp.roll(salt, 1, axis=1) - 4*salt)
    
    # Update Salt
    salt_new = salt + (-(ux_s * grad_salt_x + uy_s * grad_salt_y) + D_SALT * laplacian)
    
    # Boundary: Fresh CO2 (0 salt) at inlet
    salt_new = salt_new.at[0, :].set(0.0)
    salt_new = jnp.clip(salt_new, 0.0, 5.0)
    
    # 4. Precipitation Reaction
    # Sigmoid activation: If salt > K_SP, precipitate (mask -> 1.0)
    # Using steep sigmoid for differentiability instead of step function
    new_precip = jax.nn.sigmoid(20 * (salt_new - K_SP))
    mask_new = jnp.maximum(mask, new_precip)
    
    # 5. Metrics
    pore_vol = jnp.sum(1-mask_new) + 1e-6
    # Saturation: Fraction of pore space where rho < Threshold
    s_co2 = jnp.sum((rho < (RHO_BRINE + RHO_CO2_INIT)/2.0) * (1-mask_new)) / pore_vol
    
    # Calculate Flux (Proxy for Injection Rate)
    flux = jnp.mean(ux[0, :])
    
    return (f_new, salt_new, mask_new), (s_co2, flux)