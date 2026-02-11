# co2_well_control/physics/lbm_core.py
import jax
import jax.numpy as jnp
from jax import lax

# Load configuration for compilation constants
from co2_well_control.config.params import SimulationConfig
cfg = SimulationConfig()

@jax.jit
def equilibrium(rho, u, cx, cy, w, cs2):
    """
    Calculates Equilibrium Distribution Function (f_eq).
    Fixed for correct broadcasting of (nx, ny) vs (9) directions.
    """
    # 1. Expand Velocity for Broadcasting
    # u[0] is (nx, ny) -> We make it (nx, ny, 1) to multiply with cx (9)
    # Resulting u_dot_c will be (nx, ny, 9)
    ux = u[0][..., None] 
    uy = u[1][..., None]
    
    # 2. Calculate Dot Products
    u_dot_c = ux * cx + uy * cy
    
    # 3. Calculate Velocity Squared
    # u_sq is (nx, ny), we expand to (nx, ny, 1) for the final sum
    u_sq = (u[0]**2 + u[1]**2)[..., None]
    
    # 4. BGK Equilibrium Expansion
    term1 = u_dot_c / cs2
    term2 = (u_dot_c**2) / (2 * cs2**2)
    term3 = -u_sq / (2 * cs2)
    
    # f_eq = rho * w * (1 + term1 + term2 + term3)
    # rho must also be expanded to (nx, ny, 1)
    f_eq = rho[..., None] * w * (1 + term1 + term2 + term3)
    
    return f_eq

@jax.jit
def compute_forces(rho_field, other_rho_field, G, w, cx, cy):
    """
    Calculates Shan-Chen interaction force between two components.
    Force_k = -G * rho_k(x) * Sum(w_i * rho_other(x+e_i) * e_i)
    """
    force_x = jnp.zeros_like(rho_field)
    force_y = jnp.zeros_like(rho_field)
    
    for i in range(9):
        # Shift other_rho array opposite to velocity direction to get "neighbor value"
        shifted_rho = jnp.roll(other_rho_field, shift=(-cx[i], -cy[i]), axis=(0, 1))
        
        contribution = w[i] * shifted_rho
        # cx[i] and cy[i] are scalars here, so simple multiplication works
        force_x += contribution * cx[i]
        force_y += contribution * cy[i]
        
    force_x *= -G * rho_field
    force_y *= -G * rho_field
    
    return jnp.stack([force_x, force_y])

@jax.jit
def lbm_step(state, params_tuple):
    """
    One time step of the LBM simulation.
    state: (f1, f2) - Distribution functions for Water and CO2
    params_tuple: Unpacked constants for JIT compatibility
    """
    f1, f2 = state
    # Unpack static params
    (cx, cy, w, cs2, tau, G, mask) = params_tuple
    
    # 1. Macroscopic moments (Density and Velocity)
    rho1 = jnp.sum(f1, axis=-1)
    rho2 = jnp.sum(f2, axis=-1)
    
    # Momentum (j = rho * u)
    jx1 = jnp.sum(f1 * cx, axis=-1)
    jy1 = jnp.sum(f1 * cy, axis=-1)
    jx2 = jnp.sum(f2 * cx, axis=-1)
    jy2 = jnp.sum(f2 * cy, axis=-1)
    
    # 2. Compute Interaction Forces (Shan-Chen)
    F1 = compute_forces(rho1, rho2, G, w, cx, cy)
    F2 = compute_forces(rho2, rho1, G, w, cx, cy)
    
    # 3. Velocity Correction
    rho_tot = rho1 + rho2 + 1e-9
    u_common_x = (jx1 + jx2 + 0.5 * (F1[0] + F2[0])) / rho_tot
    u_common_y = (jy1 + jy2 + 0.5 * (F1[1] + F2[1])) / rho_tot
    
    u_vec = jnp.stack([u_common_x, u_common_y])

    # 4. Collision (BGK with Force)
    feq1 = equilibrium(rho1, u_vec, cx, cy, w, cs2)
    feq2 = equilibrium(rho2, u_vec, cx, cy, w, cs2)
    
    f1_post = f1 - (f1 - feq1) / tau[0]
    f2_post = f2 - (f2 - feq2) / tau[1]
    
    # 5. Bounceback Boundary (Solid Matrix)
    inv_idx = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    
    f1_bounced = f1_post[..., inv_idx]
    f2_bounced = f2_post[..., inv_idx]
    
    # Apply mask: Fluid uses post-collision, Solid uses bounceback
    mask_exp = mask[..., None] # Broadcast mask to (nx, ny, 1)
    f1_out = f1_post * (1 - mask_exp) + f1_bounced * mask_exp
    f2_out = f2_post * (1 - mask_exp) + f2_bounced * mask_exp
    
    # 6. Streaming (Propagate to neighbors)
    for i in range(9):
        f1_out = f1_out.at[..., i].set(jnp.roll(f1_out[..., i], shift=(cx[i], cy[i]), axis=(0, 1)))
        f2_out = f2_out.at[..., i].set(jnp.roll(f2_out[..., i], shift=(cx[i], cy[i]), axis=(0, 1)))

    return (f1_out, f2_out)
