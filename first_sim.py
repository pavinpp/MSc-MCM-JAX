import os

# --- 1. FORCE CPU MODE (CRITICAL FOR YOUR LAPTOP) ---
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import distance_transform_edt, gaussian_filter

print(f"--- JAX-LaB Setup ---")
try:
    print(f"Device detected: {jax.devices()[0]}")
except:
    print("JAX device detection failed (check installation). Defaulting to CPU.")

print(f"Note: This is running on your Intel Core Ultra 7 CPU.")
print(f"It will be slower than the A100 Cluster, but perfect for 2D testing.\n")

# --- 2. PARAMETERS ---
NX, NY = 128, 128   # Small 2D grid
TAU = 0.8           # Relaxation time
MAX_STEPS = 2000    # Time steps

# --- 3. MORPHOLOGY GENERATOR (Thesis Case 1) ---
def generate_coated_rock(nx, ny, thickness=1):
    np.random.seed(42)
    noise = np.random.normal(size=(nx, ny))
    base = gaussian_filter(noise, sigma=5)
    base_mask = (base > 0).astype(int) # 1=Fluid, 0=Solid
    
    solid_mask = 1 - base_mask
    dist = distance_transform_edt(1 - solid_mask)
    
    coating = (dist <= thickness)
    final_fluid_mask = jnp.array(1 - coating)
    return final_fluid_mask

# --- 4. JAX-LBM SOLVER ENGINE ---
@jax.jit
def lbm_step(f, mask):
    # D2Q9 constants
    w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
    idx_bounce = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6]) 

    # Macroscopic Variables
    rho = jnp.sum(f, axis=-1)
    
    # --- FIX IS HERE: Use 'e', not 'e.T' ---
    # f shape: (NX, NY, 9)
    # e shape: (9, 2)
    # Result u shape: (NX, NY, 2)
    u = jnp.dot(f, e) / (rho[..., None] + 1e-9)

    # Force (Driving the fluid rightwards)
    force = 0.0001
    u = u.at[..., 0].add(force * TAU / rho)

    # Equilibrium (BGK)
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    
    # Here we DO use transpose because we need shape (NX, NY, 9)
    u_dot_e = jnp.dot(u, e.T) 
    
    feq = w * rho[..., None] * (1 + 3*u_dot_e + 4.5*u_dot_e**2 - 1.5*u_sq)
    f_post = f - (f - feq) / TAU

    # Streaming
    f_stream = jnp.zeros_like(f_post)
    for i in range(9):
        f_stream = f_stream.at[..., i].set(
            jnp.roll(f_post[..., i], shift=e[i], axis=(0, 1))
        )

    # Bounce-Back
    f_bounced = f_stream[..., idx_bounce]
    mask_exp = mask[..., None]
    
    return f_stream * mask_exp + f_bounced * (1 - mask_exp)

# --- 5. MAIN RUN ---
def main():
    print("Generating Geometry (Coated Rock)...")
    mask = generate_coated_rock(NX, NY, thickness=2)
    porosity = jnp.mean(mask)
    print(f"Porosity: {porosity:.1%}")

    # Initialize Fluid
    w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f = jnp.ones((NX, NY, 9)) * w

    print("Starting Simulation Loop...")
    start = time.time()
    
    for t in range(MAX_STEPS):
        f = lbm_step(f, mask)
        if t % 500 == 0:
            print(f"Step {t}/{MAX_STEPS}")

    fps = (MAX_STEPS * NX * NY) / (time.time() - start) / 1e6
    print(f"Finished! Performance: {fps:.2f} MLUPS")

    # --- 6. VISUALIZE ---
    rho = jnp.sum(f, axis=-1)
    
    # Fix visualization dot product as well
    e = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
    u = jnp.dot(f, e) / rho[..., None]
    
    velocity_mag = jnp.sqrt(jnp.sum(u**2, axis=-1))
    velocity_mag = jnp.where(mask==1, velocity_mag, jnp.nan)

    plt.figure(figsize=(6, 6))
    plt.imshow(velocity_mag, cmap='magma', origin='lower')
    plt.colorbar(label="Fluid Velocity")
    plt.title(f"JAX-LBM Flow on Laptop (Porosity={porosity:.1%})")
    plt.show()

if __name__ == "__main__":
    main()
