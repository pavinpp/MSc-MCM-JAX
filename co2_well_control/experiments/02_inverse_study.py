# generate_inverse_figure.py
import os
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import time

from co2_well_control.config.params import SimulationConfig
from co2_well_control.geometry.differentiable import get_soft_mask
from co2_well_control.physics.lbm_core import lbm_step, equilibrium
from co2_well_control.inverse.loss import simulation_loss

def generate_inverse_plot():
    print(f"Generating Figure 3 (Inverse Convergence) on {jax.devices()[0]}...")
    cfg = SimulationConfig()
    
    # 1. Generate Ground Truth (Width=12.0)
    TRUE_WIDTH = 12.0
    print(f"Simulating Ground Truth (Width={TRUE_WIDTH})...")
    
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

    # 2. Run Optimization (Start Guess=6.0)
    INITIAL_GUESS = 6.0
    param_width = jnp.array(INITIAL_GUESS)
    
    # Learning Rate Schedule
    TOTAL_STEPS = 60
    scheduler = optax.cosine_decay_schedule(init_value=0.5, decay_steps=TOTAL_STEPS, alpha=0.002)
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(param_width)
    
    @jax.jit
    def step(param, opt_state, target):
        loss_val, grads = jax.value_and_grad(simulation_loss)(param, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_param = optax.apply_updates(param, updates)
        return new_param, opt_state, loss_val

    print("Running Optimization...")
    history_param = []
    history_loss = []
    
    for i in range(TOTAL_STEPS):
        param_width, opt_state, loss = step(param_width, opt_state, true_pressure_history)
        history_param.append(float(param_width))
        history_loss.append(float(loss))

    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(base_dir, "..", "reports", "figures")
    os.makedirs(save_dir, exist_ok=True)

    # 3. Plotting - TWO PANELS
    print("Saving Figure 3...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left Panel: Parameter Convergence
    ax1.plot(history_param, 'b-', linewidth=2, label='Estimated Width')
    ax1.axhline(TRUE_WIDTH, color='r', linestyle='--', label=f'True Width ({TRUE_WIDTH})')
    ax1.axhline(INITIAL_GUESS, color='g', linestyle=':', label=f'Start ({INITIAL_GUESS})')
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Fracture Width (Lattice Units)")
    ax1.set_title("A) Parameter Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right Panel: Loss Function
    ax2.plot(history_loss, 'k-', linewidth=2)
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.set_title("B) Loss Function Optimization")
    ax2.set_yscale('log') # Log scale makes it look much more professional
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure_3.png"), dpi=300)
    print("Saved 'Figure_3.png'")
    plt.show()

if __name__ == "__main__":
    generate_inverse_plot()