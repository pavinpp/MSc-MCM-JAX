# visualize_noise.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import simulation modules
from co2_well_control.config.params import SimulationConfig
from co2_well_control.physics.lbm_core import lbm_step, equilibrium
from co2_well_control.geometry.differentiable import get_soft_mask

def visualize_sensor_data():
    print("Generating clean physics data...")
    cfg = SimulationConfig()
    
    # 1. Setup True Geometry (Width = 12.0)
    TRUE_WIDTH = 12.0
    mask_true = get_soft_mask(TRUE_WIDTH, cfg.nx, cfg.ny)
    
    # 2. Initialize State
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0 
    rho2 = jnp.zeros((cfg.nx, cfg.ny))
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    
    params_true = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, mask_true)
    
    # 3. Run Forward Simulation
    def truth_loop(carrier, i):
        f1_curr, f2_curr = carrier
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params_true)
        
        # Inlet BC (Continuous Injection)
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)
        
        # Record Pressure
        p_inlet = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[0:5, :])
        return (f1_new, f2_new), p_inlet

    print("Running simulation (2000 steps)...")
    _, clean_signal = jax.lax.scan(truth_loop, (f1, f2), jnp.arange(cfg.steps))
    
    # 4. Add 5% Noise
    noise_percentage = 0.05
    mean_val = jnp.mean(clean_signal)
    noise = jax.random.normal(jax.random.PRNGKey(42), clean_signal.shape) * mean_val * noise_percentage
    noisy_signal = clean_signal + noise

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot the noisy "dots" to simulate raw sensor readings
    plt.plot(noisy_signal, 'k.', markersize=2, alpha=0.3, label='Raw Sensor Data (5% Noise)')
    
    # Plot the clean "line" to show the true physics
    plt.plot(clean_signal, 'r-', linewidth=2, label='True Physics (Width=12.0)')
    
    plt.title(f"What the AI Sees: Clean Physics vs. {int(noise_percentage*100)}% Sensor Noise")
    plt.xlabel("Time Step (Simulation Time)")
    plt.ylabel("Inlet Pressure (Lattice Units)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save and Show
    plt.savefig("forward_noise_visualization.png")
    print("Graph saved as 'forward_noise_visualization.png'")
    plt.show()

if __name__ == "__main__":
    visualize_sensor_data()
