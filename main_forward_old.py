# main_forward.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from co2_well_control.config.params import SimulationConfig
from co2_well_control.geometry.domain import create_domain
from co2_well_control.physics.lbm_core import lbm_step, equilibrium

def run_simulation():
    print(f"Initializing Simulation on device: {jax.devices()[0]}")
    cfg = SimulationConfig()
    
    # 1. Setup Domain
    solid_mask = create_domain(cfg)
    
    # 2. Initialize Fields
    # Water everywhere initially
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0 
    # Small random noise to break symmetry and allow phase separation if needed
    rho1 += jax.random.uniform(jax.random.PRNGKey(0), (cfg.nx, cfg.ny)) * 0.01
    
    # CO2 Injection at Inlet (Left side)
    rho2 = jnp.zeros((cfg.nx, cfg.ny))
    rho2 = rho2.at[0:5, :].set(cfg.inlet_pressure_co2) # Inject CO2
    
    # Initial zero velocity
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    
    # Initialize Distribution Functions
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    
    # Pack parameters for the JIT function
    params = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, solid_mask)
    
    # 3. Define the Time Loop (using jax.lax.scan) with CONTINUOUS INJECTION
    # This effectively compiles the entire simulation loop into one optimized kernel
    def update_wrapper(carrier, i):
        f1_curr, f2_curr = carrier
        
        # A. Run Physics Step
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params)
        
        # B. Apply Hard Inlet Boundary Condition (Dirichlet)
        # We force the density at the inlet (Left) to stay high, mimicking a pump.
        # But ONLY in the fracture zone (e.g., y=22 to 42)
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        
        # Calculate equilibrium for the forced inlet density
        # We assume new incoming fluid is pure CO2 (rho2=high, rho1=low)
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        
        # Create the forced density block
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        rho1_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * 1.0 # Background water
        
        # Re-calculate distributions for the inlet nodes
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        
        # Force the new distributions into the field (Hard Reset of Inlet)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)

        # Output: Monitor the pressure halfway down the fracture
        rho_mid = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[128, 32:42])
        
        return (f1_new, f2_new), rho_mid

    print("Starting Compilation and Execution...")
    start_time = time.time()
    
    # RUN THE SIMULATION
    # scan returns: (final_state, stacked_history_of_outputs)
    (f1_final, f2_final), p_history = jax.lax.scan(update_wrapper, (f1, f2), None, length=cfg.steps)
    
    end_time = time.time()
    print(f"Simulation completed {cfg.steps} steps in {end_time - start_time:.4f} seconds.")
    print(f"Performance: {cfg.steps * cfg.nx * cfg.ny / (end_time - start_time) / 1e6:.2f} MLUPS")

    # 4. Visualization
    rho_total = jnp.sum(f1_final, axis=-1) + jnp.sum(f2_final, axis=-1)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(rho_total.T, cmap='viridis', origin='lower')
    plt.colorbar(label='Total Density')
    plt.title(f"State at Step {cfg.steps} (Water + CO2)")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    plt.plot(p_history)
    plt.xlabel("Time Step")
    plt.ylabel("Inlet Pressure (Proxy)")
    plt.title("Synthetic Pressure Transient Log")
    plt.show()

if __name__ == "__main__":
    run_simulation()
