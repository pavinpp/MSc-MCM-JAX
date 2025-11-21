import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter, distance_transform_edt

# --- 1. CONFIGURATION ---
NX, NY = 128, 128    
TAU = 0.8            
MAX_STEPS = 3000     # Increased steps for stability
BODY_FORCE = 0.0001  

# --- 2. IMPROVED GEOMETRY GENERATORS ---
def generate_base_rock(nx, ny, porosity_target=0.5):
    np.random.seed(42) 
    noise = np.random.normal(size=(nx, ny))
    smooth = gaussian_filter(noise, sigma=4)
    threshold = np.percentile(smooth, 100 * (1 - porosity_target))
    mask = (smooth > threshold).astype(int)
    return mask

def get_morphologies(base_mask):
    # CASE 1: Uniform Coating (The Baseline)
    solid_mask = 1 - base_mask
    dist = distance_transform_edt(1 - solid_mask)
    # Apply a standard 1.5 pixel coating
    coating = (dist <= 1.5)
    mask_case1 = jnp.array(1 - coating.astype(int))
    
    # CALCULATE TARGETS
    initial_fluid = np.sum(base_mask)
    target_fluid = jnp.sum(mask_case1)
    pixels_to_remove = int(initial_fluid - target_fluid)
    
    print(f"Target: Remove {pixels_to_remove} pixels to match Case 1 porosity.")

    # CASE 2: Preferential Throat Clogging (Weighted Random)
    # Logic: We want to clog narrow spots, but we want to pile up 
    # material there to BRIDGE the throat, not just coat it.
    mask_case2 = np.array(base_mask, copy=True)
    dist_map = distance_transform_edt(mask_case2)
    
    # Identify available fluid pixels
    fluid_coords = np.argwhere(mask_case2 == 1)
    dists = dist_map[mask_case2 == 1]
    
    # Weight probability heavily towards small distances (1/d^4)
    # This makes it aggressively attack throats
    weights = 1.0 / (dists**4 + 0.1) 
    weights /= weights.sum() # Normalize
    
    # Randomly select pixels based on weights
    clog_indices = np.random.choice(len(fluid_coords), pixels_to_remove, replace=False, p=weights)
    
    rows = fluid_coords[clog_indices, 0]
    cols = fluid_coords[clog_indices, 1]
    mask_case2[rows, cols] = 0 # Solidify
    
    # CASE 3: Stochastic Nucleation (Exact Porosity Match)
    mask_case3 = np.array(base_mask, copy=True)
    fluid_coords = np.argwhere(mask_case3 == 1)
    
    # We add crystals one by one until we hit the pixel budget
    pixels_removed = 0
    while pixels_removed < pixels_to_remove:
        # Pick random site
        idx = np.random.randint(len(fluid_coords))
        r, c = fluid_coords[idx]
        
        # Try to place a small 2x2 crystal
        # We check bounds to avoid errors
        r_min, r_max = max(0, r-1), min(NX, r+1)
        c_min, c_max = max(0, c-1), min(NY, c+1)
        
        # Count how many we are about to remove
        current_slice = mask_case3[r_min:r_max, c_min:c_max]
        potential_loss = np.sum(current_slice)
        
        # Only place if it doesn't exceed our budget
        if pixels_removed + potential_loss <= pixels_to_remove:
            mask_case3[r_min:r_max, c_min:c_max] = 0
            pixels_removed += potential_loss
        else:
            # If too big, break loop (close enough) or try single pixel
            mask_case3[r, c] = 0
            pixels_removed += 1
            
    return {
        "Case 1 (Coating)": mask_case1,
        "Case 2 (Throats)": jnp.array(mask_case2),
        "Case 3 (Stochastic)": jnp.array(mask_case3)
    }

# --- 3. JAX LBM SOLVER ---
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

def calculate_permeability(f, mask):
    rho = jnp.sum(f, axis=-1)
    e = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
    u = jnp.dot(f, e) / (rho[..., None] + 1e-9)
    u_x = u[..., 0] * mask 
    u_darcy = jnp.mean(u_x) 
    nu = (TAU - 0.5) / 3
    mu = nu * 1.0 
    k = u_darcy * mu / BODY_FORCE
    return k, u_x

# --- 4. MAIN EXECUTION ---
def main():
    print("--- THESIS PILOT: MATCHED POROSITY STUDY ---\n")
    
    base_mask = generate_base_rock(NX, NY)
    print(f"Base Porosity: {jnp.mean(base_mask):.2%}")
    
    scenarios = get_morphologies(base_mask)
    
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, mask) in enumerate(scenarios.items()):
        porosity = jnp.mean(mask)
        print(f"\nRunning {name}...")
        print(f"  -> Porosity: {porosity:.2%} (Exact Match Attempt)")
        
        w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        f = jnp.ones((NX, NY, 9)) * w
        
        start = time.time()
        for t in range(MAX_STEPS):
            f = lbm_step(f, mask)
        
        k_val, u_field = calculate_permeability(f, mask)
        results[name] = k_val
        
        print(f"  -> Permeability (k): {k_val:.5f}")
        
        ax = axes[i]
        im = ax.imshow(u_field, cmap='turbo', origin='lower')
        ax.set_title(f"{name}\nk={k_val:.5f}")
        ax.axis('off')

    print("\n--- FINAL RESULTS ---")
    base_k = results["Case 1 (Coating)"]
    for name, k in results.items():
        ratio = k / base_k
        print(f"{name}: k = {k:.5f} | Ratio: {ratio:.2f}")
        
    print("\nInterpretation: If Case 2 Ratio < 1.0, your thesis hypothesis is validated.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()