# generate_paper_figures.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from co2_well_control.config.params import SimulationConfig
from co2_well_control.geometry.differentiable import get_soft_mask
from co2_well_control.physics.lbm_core import lbm_step, equilibrium

def generate_physics_figures():
    print("Running simulation for Paper Figures (Width=12.0)...")
    cfg = SimulationConfig()
    
    # 1. Setup Domain (Width = 12.0)
    solid_mask = get_soft_mask(12.0, cfg.nx, cfg.ny)
    
    # 2. Initialize Fluids
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0  # Brine
    rho2 = jnp.zeros((cfg.nx, cfg.ny))       # CO2
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    
    params = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, solid_mask)
    
    # 3. Define Loop
    def update_wrapper(carrier, i):
        f1_curr, f2_curr = carrier
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params)
        
        # Continuous Injection
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)
        
        # Record Pressure
        p_inlet = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[0:5, :])
        return (f1_new, f2_new), p_inlet

    # 4. Run Simulation
    (f1_final, f2_final), p_history = jax.lax.scan(update_wrapper, (f1, f2), None, length=cfg.steps)
    
    # --- FIGURE 1: Density Heatmap (The "Cross-section") ---
    print("Generating Figure 1: Density Heatmap...")
    rho_total = jnp.sum(f1_final, axis=-1) + jnp.sum(f2_final, axis=-1)
    
    plt.figure(figsize=(10, 4))
    # We transpose (.T) to show the flow going Left->Right
    # origin='lower' ensures Y=0 is at the bottom
    im = plt.imshow(rho_total.T, cmap='turbo', origin='lower', aspect='equal')
    cbar = plt.colorbar(im, label='Total Density (Lattice Units)')
    
    plt.title("Figure 1: Cross-Section of CO2 Injection (t=2000)")
    plt.xlabel("Distance from Wellbore (Lattice Units)")
    plt.ylabel("Fracture Width / Matrix")
    plt.tight_layout()
    plt.savefig("Figure_1_Density_Heatmap.png", dpi=300)
    print("Saved 'Figure_1_Density_Heatmap.png'")
    
    # --- FIGURE 2: Pressure Transient (The "Graph") ---
    print("Generating Figure 2: Pressure Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(p_history, 'b-', linewidth=2)
    plt.title("Figure 2: Simulated Bottomhole Pressure Response")
    plt.xlabel("Time Step")
    plt.ylabel("Inlet Pressure (Proxy)")
    plt.grid(True, alpha=0.3)
    
    # Annotate the phases (Optional, makes it look pro)
    plt.text(100, jnp.min(p_history), "Inertial Kick", fontsize=9, rotation=90)
    plt.text(500, jnp.mean(p_history), "Compression Phase", fontsize=9)
    plt.text(1500, jnp.max(p_history), "Steady State", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("Figure_2_Pressure_Transient.png", dpi=300)
    print("Saved 'Figure_2_Pressure_Transient.png'")
    
    plt.show()

if __name__ == "__main__":
    generate_physics_figures()