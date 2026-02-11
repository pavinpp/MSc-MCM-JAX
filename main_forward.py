# Updated main_forward.py to match Phase 2/3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from co2_well_control.config.params import SimulationConfig
# CHANGE 1: Use the Differentiable Mask instead of domain.py
from co2_well_control.geometry.differentiable import get_soft_mask 
from co2_well_control.physics.lbm_core import lbm_step, equilibrium

def run_simulation():
    print(f"Initializing Simulation on device: {jax.devices()[0]}")
    cfg = SimulationConfig()
    
    # CHANGE 2: Set Width to 12.0 (Same as Phase 2/3)
    print("Using Standardized Fracture Width: 12.0")
    solid_mask = get_soft_mask(12.0, cfg.nx, cfg.ny)
    
    # ... (Rest of initialization is same) ...
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0 
    rho2 = jnp.zeros((cfg.nx, cfg.ny))
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    
    params = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, solid_mask)
    
    # ... (Rest of loop is same) ...
    def update_wrapper(carrier, i):
        f1_curr, f2_curr = carrier
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params)
        
        # Ensure you have the CONTINUOUS INJECTION block here
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)

        # Output: Inlet Pressure
        p_inlet = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[0:5, :])
        
        return (f1_new, f2_new), p_inlet

    # ... (Run and Plot) ...
    start_time = time.time()
    (f1_final, f2_final), p_history = jax.lax.scan(update_wrapper, (f1, f2), None, length=cfg.steps)
    end_time = time.time()
    
    print(f"Simulation completed. 12.0 pixel width.")
    
    plt.figure(figsize=(6, 4))
    plt.plot(p_history)
    plt.title("Forward Simulation (Width=12.0)")
    plt.xlabel("Time Step")
    plt.ylabel("Pressure")
    plt.show()

if __name__ == "__main__":
    run_simulation()