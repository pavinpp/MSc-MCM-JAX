# co2_well_control/inverse/loss.py
import jax
import jax.numpy as jnp
from co2_well_control.physics.lbm_core import lbm_step, equilibrium
from co2_well_control.geometry.differentiable import get_soft_mask
from co2_well_control.config.params import SimulationConfig

cfg = SimulationConfig()

def simulation_loss(width_param, target_pressure_history):
    """
    Runs the full simulation with a specific fracture width
    and returns the Mean Squared Error (MSE) vs target data.
    """
    
    # 1. Generate Geometry based on the parameter
    # This is the "Differentiable" link: Geometry depends on theta
    mask = get_soft_mask(width_param, cfg.nx, cfg.ny)
    
    # 2. Initialize State (Same as Forward Sim)
    rho1 = jnp.ones((cfg.nx, cfg.ny)) * 1.0 
    rho2 = jnp.zeros((cfg.nx, cfg.ny))
    u_init = jnp.zeros((2, cfg.nx, cfg.ny))
    
    f1 = equilibrium(rho1, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    f2 = equilibrium(rho2, u_init, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
    
    # Pack params (Note: we pass the dynamic 'mask' here)
    params = (cfg.cx, cfg.cy, cfg.w, cfg.cs2, cfg.tau, cfg.G_interaction, mask)

    # 3. Define the Loop
    def update_wrapper(carrier, i):
        f1_curr, f2_curr = carrier
        f1_new, f2_new = lbm_step((f1_curr, f2_curr), params)
        
        # --- Continuous Injection (Boundary Condition) ---
        fracture_y_start = cfg.ny // 2 - 10
        fracture_y_end   = cfg.ny // 2 + 10
        
        u_inlet = jnp.zeros((2, 5, fracture_y_end - fracture_y_start))
        rho2_forced = jnp.ones((5, fracture_y_end - fracture_y_start)) * cfg.inlet_pressure_co2
        f2_inlet = equilibrium(rho2_forced, u_inlet, cfg.cx, cfg.cy, cfg.w, cfg.cs2)
        f2_new = f2_new.at[0:5, fracture_y_start:fracture_y_end, :].set(f2_inlet)
        # ------------------------------------------------
        
        # Record Inlet Pressure (Signal)
        p_inlet = jnp.mean(jnp.sum(f1_new + f2_new, axis=-1)[0:5, :])
        
        return (f1_new, f2_new), p_inlet

    # Run Simulation
    # We use fewer steps for optimization to save time (e.g., 500)
    # But for accuracy, let's match the target data length
    steps = target_pressure_history.shape[0]
    final_state, simulated_history = jax.lax.scan(update_wrapper, (f1, f2), jnp.arange(steps))
    
    # 4. Calculate Loss (MSE)
    # We ignore the first 100 steps (initialization noise)
    diff = simulated_history[100:] - target_pressure_history[100:]
    mse = jnp.mean(diff ** 2)
    
    return mse
