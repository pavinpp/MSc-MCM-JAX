# main_inverse.py
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time

from co2_well_control.inverse.loss import simulation_loss
from co2_well_control.config.params import SimulationConfig

def run_inverse_problem():
    print(f"Device: {jax.devices()[0]}")
    cfg = SimulationConfig()
    
    # --- Step 1: Generate "Ground Truth" Data ---
    TRUE_WIDTH = 12.0
    print(f"Generating synthetic field data (True Width = {TRUE_WIDTH})...")
    
    # We cheat slightly and call simulation_loss just to get the history
    # To do this, we need to modify simulation_loss to return history, 
    # OR we just copy the forward run code here. 
    # For cleanliness, let's use a helper wrapper around the loss function's logic.
    # A simple hack: Pass a dummy target of zeros to the loss function, 
    # but we need the trajectory.
    
    # Let's just redefine the forward pass briefly for generation to be explicit
    from co2_well_control.geometry.differentiable import get_soft_mask
    from co2_well_control.physics.lbm_core import lbm_step, equilibrium
    
    # ... (Copy of the forward loop for truth generation)
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

    # --- Step 2: Setup Optimization ---
    # Initial Guess (Wrong width)
    INITIAL_GUESS = 6.0
    param_width = jnp.array(INITIAL_GUESS)
    
    # Optimizer (Adam)
    optimizer = optax.adam(learning_rate=0.5)
    opt_state = optimizer.init(param_width)
    
    # JIT-compiled Update Step
    @jax.jit
    def step(param, opt_state, target):
        # Value and Gradient
        loss_val, grads = jax.value_and_grad(simulation_loss)(param, target)
        # Update
        updates, opt_state = optimizer.update(grads, opt_state)
        new_param = optax.apply_updates(param, updates)
        return new_param, opt_state, loss_val, grads

    # --- Step 3: Training Loop ---
    print(f"Starting Inverse Optimization (Initial Guess = {INITIAL_GUESS})...")
    history_param = []
    history_loss = []
    
    start_time = time.time()
    for i in range(50): # 50 Iterations
        param_width, opt_state, loss, grad = step(param_width, opt_state, true_pressure_history)
        
        history_param.append(float(param_width))
        history_loss.append(float(loss))
        
        if i % 5 == 0:
            print(f"Iter {i:03d} | Width: {param_width:.4f} | Loss: {loss:.6f} | Grad: {grad:.4f}")

    total_time = time.time() - start_time
    print(f"Optimization Finished in {total_time:.2f}s")
    print(f"Final Estimated Width: {param_width:.4f} (True: {TRUE_WIDTH})")
    
    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_param, label='Estimated Width')
    plt.axhline(TRUE_WIDTH, color='r', linestyle='--', label='True Width')
    plt.axhline(INITIAL_GUESS, color='g', linestyle='--', label='Initial Guess')
    plt.legend()
    plt.title("Parameter Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Width Parameter")
    
    plt.subplot(1, 2, 2)
    plt.plot(history_loss)
    plt.yscale('log')
    plt.title("Loss Function (MSE)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    run_inverse_problem()
