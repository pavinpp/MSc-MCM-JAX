# experiments/03_noise_robustness.py
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time

# Import your modules
from co2_well_control.config.params import SimulationConfig
from co2_well_control.physics.lbm_core import lbm_step, equilibrium
from co2_well_control.geometry.differentiable import get_soft_mask
from co2_well_control.inverse.loss import simulation_loss

def run_noise_experiment():
    print(f"Running Noise Robustness Test on {jax.devices()[0]}")
    cfg = SimulationConfig()
    
    # --- 1. Generate Ground Truth (Width=12.0) ---
    TRUE_WIDTH = 12.0
    print(f"Generating clean truth data (Width={TRUE_WIDTH})...")
    
    # (Simplified Forward Pass for Data Generation)
    mask_true = get_soft_mask(TRUE_WIDTH, cfg.nx, cfg.ny)
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0 
    rho2 = jnp.zeros((cfg.nx, cfg.ny))
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    params_true = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, mask_true)
    
    def truth_loop(carrier, i):
        f1_curr, f2_curr = carrier
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params_true)
        # Inlet BC
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)
        
        p_inlet = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[0:5, :])
        return (f1_new, f2_new), p_inlet

    _, clean_signal = jax.lax.scan(truth_loop, (f1, f2), jnp.arange(cfg.steps))
    
    # --- 2. Define Noise Levels to Test ---
    noise_levels = [0.00, 0.01, 0.05] # 0%, 1%, 5% Noise
    results = {}

    for noise_pct in noise_levels:
        print(f"\n--- Testing with {noise_pct*100}% Sensor Noise ---")
        
        # Add Noise
        noise_magnitude = noise_pct * jnp.mean(clean_signal)
        # Use a fixed key for reproducibility
        noise = jax.random.normal(jax.random.PRNGKey(42), clean_signal.shape) * noise_magnitude
        noisy_target = clean_signal + noise
        
        # Setup Optimizer (Identical to Phase 2)
        INITIAL_GUESS = 6.0
        param_width = jnp.array(INITIAL_GUESS)
        optimizer = optax.adam(learning_rate=0.4) # Slightly lower LR to handle noise
        opt_state = optimizer.init(param_width)
        
        @jax.jit
        def step(param, opt_state, target):
            loss_val, grads = jax.value_and_grad(simulation_loss)(param, target)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_param = optax.apply_updates(param, updates)
            return new_param, opt_state, loss_val

        # Run Optimization Loop
        history = []
        for i in range(40):
            param_width, opt_state, loss = step(param_width, opt_state, noisy_target)
            history.append(float(param_width))
            if i % 10 == 0:
                print(f"Iter {i} | Est Width: {param_width:.3f} | Loss: {loss:.6f}")
        
        results[noise_pct] = history
        print(f"Final Result ({noise_pct*100}% Noise): {float(param_width):.3f}")

    # --- 3. Visualization for Paper ---
    plt.figure(figsize=(10, 6))
    for noise_pct, history in results.items():
        plt.plot(history, label=f"Noise {noise_pct*100}%", linewidth=2)
    
    plt.axhline(TRUE_WIDTH, color='k', linestyle='--', label="Ground Truth (12.0)")
    plt.axhline(6.0, color='r', linestyle=':', label="Start Guess (6.0)")
    
    plt.xlabel("Optimization Steps")
    plt.ylabel("Inferred Fracture Width (Lattice Units)")
    plt.title("Robustness of Differentiable Inverse Solver to Sensor Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("noise_robustness_result.png")
    plt.show()

if __name__ == "__main__":
    run_noise_experiment()
