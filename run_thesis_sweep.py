import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.ndimage import gaussian_filter, distance_transform_edt

# --- 1. STABILITY SETTINGS (TUNED) ---
NX, NY = 128, 128
TAU = 1.0            # Increased from 0.8 for stability
MAX_STEPS = 3000     # More steps to settle
BODY_FORCE = 1e-5    # Reduced force to prevent "explosions"

# --- 2. GEOMETRY FUNCTIONS ---
def generate_base_rock(nx, ny, porosity_target=0.6):
    np.random.seed(42)
    noise = np.random.normal(size=(nx, ny))
    smooth = gaussian_filter(noise, sigma=4)
    threshold = np.percentile(smooth, 100 * (1 - porosity_target))
    mask = (smooth > threshold).astype(int)
    return mask

def clog_sequentially(current_mask, mode, pixels_to_remove):
    """
    Takes the EXISTING mask and adds more solids to it.
    This guarantees we are always getting tighter.
    """
    if pixels_to_remove <= 0: return current_mask

    # Create a working copy
    new_mask = np.array(current_mask, copy=True)
    
    if mode == "Coating":
        solid_mask = 1 - new_mask
        dist = distance_transform_edt(1 - solid_mask)
        fluid_dists = dist[new_mask == 1]
        
        if len(fluid_dists) == 0: return new_mask
        
        sorted_dists = np.sort(fluid_dists)
        # Clog the pixels closest to walls
        cutoff = sorted_dists[min(len(sorted_dists)-1, pixels_to_remove)]
        coating = (dist <= cutoff)
        return np.array(1 - coating.astype(int))

    elif mode == "Throats":
        # Preferential Throat Clogging
        dist_map = distance_transform_edt(new_mask)
        fluid_coords = np.argwhere(new_mask == 1)
        dists = dist_map[new_mask == 1]
        
        # Weight: High probability for small distance
        weights = 1.0 / (dists**4 + 0.1) # Stronger weighting
        weights /= weights.sum()
        
        # Choose pixels to clog
        count = min(len(fluid_coords), pixels_to_remove)
        idx = np.random.choice(len(fluid_coords), count, replace=False, p=weights)
        
        rows = fluid_coords[idx, 0]
        cols = fluid_coords[idx, 1]
        new_mask[rows, cols] = 0
        return new_mask

    elif mode == "Stochastic":
        # Random Nucleation
        fluid_coords = np.argwhere(new_mask == 1)
        count = min(len(fluid_coords), pixels_to_remove)
        idx = np.random.choice(len(fluid_coords), count, replace=False)
        rows = fluid_coords[idx, 0]
        cols = fluid_coords[idx, 1]
        new_mask[rows, cols] = 0
        return new_mask
    
    return new_mask

# --- 3. JAX SOLVER ---
@jax.jit
def lbm_step(f, mask):
    w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
    idx_bounce = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6]) 

    rho = jnp.sum(f, axis=-1)
    u = jnp.dot(f, e) / (rho[..., None] + 1e-9)
    u = u.at[..., 0].add(BODY_FORCE * TAU / rho)

    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    u_dot_e = jnp.dot(u, e.T)
    feq = w * rho[..., None] * (1 + 3*u_dot_e + 4.5*u_dot_e**2 - 1.5*u_sq)
    f_post = f - (f - feq) / TAU

    f_stream = jnp.zeros_like(f_post)
    for i in range(9):
        f_stream = f_stream.at[..., i].set(jnp.roll(f_post[..., i], shift=e[i], axis=(0, 1)))

    f_bounced = f_stream[..., idx_bounce]
    mask_exp = mask[..., None]
    return f_stream * mask_exp + f_bounced * (1 - mask_exp)

def get_k(f, mask):
    rho = jnp.sum(f, axis=-1)
    e = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
    u = jnp.dot(f, e) / (rho[..., None] + 1e-9)
    
    # Stability Check: If velocity is insane, return NaN
    u_mag = jnp.sqrt(jnp.sum(u**2, axis=-1))
    max_u = jnp.max(u_mag)
    
    # Calculate Permeability
    u_avg = jnp.mean(u[..., 0] * mask)
    nu = (TAU - 0.5) / 3
    k = u_avg * (nu * 1.0) / BODY_FORCE
    
    return k, max_u

# --- 4. THE SEQUENTIAL SWEEP ---
def main():
    print("--- THESIS SWEEP (Sequential + Stability Fix) ---")
    
    # 1. Generate Initial Rock
    base_mask_original = generate_base_rock(NX, NY, porosity_target=0.60)
    phi_0 = np.mean(base_mask_original)
    print(f"Base Porosity: {phi_0:.1%}")

    # 2. Calculate k0
    w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f = jnp.ones((NX, NY, 9)) * w
    print("Stabilizing Base Rock...")
    for _ in range(MAX_STEPS): f = lbm_step(f, base_mask_original)
    k_0, max_u = get_k(f, base_mask_original)
    print(f"k0 = {k_0:.5f} | Max U = {max_u:.5f}\n")

    results = []
    modes = ["Coating", "Stochastic", "Throats"]
    
    # 3. Run Each Mode Sequentially
    for mode in modes:
        print(f"--- Mode: {mode} ---")
        
        # Reset for this mode
        current_mask = np.array(base_mask_original, copy=True)
        current_k = k_0
        
        # We will take 10 steps, removing 5% of INITIAL fluid each time
        total_fluid_pixels = np.sum(base_mask_original)
        pixels_per_step = int(total_fluid_pixels * 0.05)
        
        for step in range(10):
            # Update Geometry (Remove more pixels from the CURRENT mask)
            current_mask = clog_sequentially(current_mask, mode, pixels_per_step)
            
            actual_phi = np.mean(current_mask)
            
            # Run LBM (Hot start from previous f helps stability)
            # We reset f to equilibrium to avoid shock, but use new mask
            f = jnp.ones((NX, NY, 9)) * w 
            for _ in range(MAX_STEPS): f = lbm_step(f, current_mask)
            
            k_val, max_u = get_k(f, current_mask)
            
            # Check for explosion
            if max_u > 0.15 or jnp.isnan(k_val):
                print(f"  Step {step}: UNSTABLE (Max U={max_u:.2f}). Stopping this branch.")
                break

            norm_phi = actual_phi / phi_0
            norm_k = k_val / k_0
            
            row = {
                "Mode": mode,
                "Phi_Normalized": norm_phi,
                "k_Normalized": norm_k
            }
            results.append(row)
            print(f"  Phi Ratio: {norm_phi:.2f} | k/k0: {norm_k:.4f}")
            
            # Stop if fully clogged
            if actual_phi < 0.05: break

    # --- 5. PLOT ---
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(8, 6))
    for mode in modes:
        subset = df[df["Mode"] == mode]
        plt.plot(subset["Phi_Normalized"], subset["k_Normalized"], 'o-', label=mode)

    # Add Reference Line
    x_ref = np.linspace(0.5, 1.0, 20)
    y_ref = x_ref**3 # Cubic law
    plt.plot(x_ref, y_ref, 'k--', alpha=0.3, label="Cubic Law")

    plt.xlabel(r"Normalized Porosity ($\phi / \phi_0$)")
    plt.ylabel(r"Normalized Permeability ($k / k_0$)")
    plt.title("Permeability Degradation (Laptop Simulation)")
    plt.xlim(1.0, 0.4) # Invert X axis (starts at 1, goes down)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("thesis_money_plot_fixed.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()