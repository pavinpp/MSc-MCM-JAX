# main_inverse_decay.py
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time

from co2_well_control.inverse.loss import simulation_loss
from co2_well_control.config.params import SimulationConfig
# We need these imports to regenerate the Ground Truth inside the script
from co2_well_control.geometry.differentiable import get_soft_mask
from co2_well_control.physics.lbm_core import lbm_step, equilibrium

def run_inverse_decay():
    print(f"Device: {jax.devices()[0]}")
    cfg = SimulationConfig()
    
    # --- Step 1: Generate "Ground Truth" Data ---
    # We want to find this exact value
    TRUE_WIDTH = 12.0
    print(f"Generating synthetic field data (True Width = {TRUE_WIDTH})...")
    
    # --- (Ground Truth Generation Code - Same as before) ---
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

    _, true_pressure_history = jax.lax.scan(truth_loop, (f1, f2), jnp.arange(cfg.steps))
    print("Truth data generated.")

    # --- Step 2: Setup Optimization with DECAY ---
    INITIAL_GUESS = 6.0
    param_width = jnp.array(INITIAL_GUESS)
    
    # CONFIGURATION
    TOTAL_STEPS = 60
    START_LR = 0.5
    END_LR = 0.001  # Stop moving at the end
    
    # DEFINING THE SCHEDULE
    # Cosine Decay: Starts at 0.5, follows a cosine curve down to 0.001 over 60 steps
    scheduler = optax.cosine_decay_schedule(
        init_value=START_LR, 
        decay_steps=TOTAL_STEPS, 
        alpha=END_LR / START_LR
    )
    
    # Pass the scheduler instead of a float
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(param_width)
    
    @jax.jit
    def step(param, opt_state, target, step_count):
        loss_val, grads = jax.value_and_grad(simulation_loss)(param, target)
        updates, opt_state = optimizer.update(grads, opt_state) # Optax handles the LR internaly using step count
        new_param = optax.apply_updates(param, updates)
        return new_param, opt_state, loss_val

    # --- Step 3: Training Loop ---
    print(f"Starting Inverse Optimization with Decay (Start LR={START_LR} -> End LR={END_LR})...")
    history_param = []
    history_loss = []
    
    for i in range(TOTAL_STEPS):
        # We don't need to pass 'i' to step() unless we manually handle LR, 
        # but Optax's internal state tracks the step count automatically.
        param_width, opt_state, loss = step(param_width, opt_state, true_pressure_history, i)
        
        history_param.append(float(param_width))
        history_loss.append(float(loss))
        
        # Calculate current LR for display (just for our info)
        current_lr = scheduler(i)
        
        if i % 5 == 0:
            print(f"Iter {i:03d} | Width: {param_width:.4f} | Loss: {loss:.6f} | LR: {current_lr:.4f}")

    print(f"Final Estimated Width: {param_width:.4f} (True: {TRUE_WIDTH})")
    
    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_param, label='Estimated Width', linewidth=2)
    plt.axhline(TRUE_WIDTH, color='r', linestyle='--', label='True Width (12.0)')
    plt.axhline(INITIAL_GUESS, color='g', linestyle=':', label='Start (6.0)')
    plt.xlabel("Optimization Steps")
    plt.ylabel("Fracture Width")
    plt.title("Parameter Convergence with Learning Rate Decay")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_inverse_decay()