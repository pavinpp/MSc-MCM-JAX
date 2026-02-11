# co2_well_control/config/params.py
import jax.numpy as jnp

class SimulationConfig:
    def __init__(self):
        # --- Grid Parameters ---
        self.nx = 256  # Length (Flow direction)
        self.ny = 64   # Height (Fracture width + Matrix)
        self.steps = 2000 # Time steps for Phase 1 testing

        # --- LBM Constants (D2Q9) ---
        # Weights for D2Q9
        self.w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        # Lattice Velocities (c_x, c_y)
        self.cx = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.cs2 = 1.0 / 3.0  # Speed of sound squared

        # --- Multiphase Parameters (Shan-Chen) ---
        # Component 0: Water (Brine)
        # Component 1: Supercritical CO2
        self.tau = jnp.array([1.0, 1.0])  # Relaxation time for each component
        self.G_interaction = -1.2         # Interaction strength (Negative = Repulsive/Immiscible)
        
        # --- Thermodynamics (Peng-Robinson parameters - dimensionless) ---
        # Critical properties (Pseudo-reduced for stability)
        self.Tc = jnp.array([1.5, 1.1])   # [Water, CO2]
        self.Pc = jnp.array([0.5, 0.8])
        self.omega = jnp.array([0.344, 0.225]) # Acentric factors
        
        # --- Boundary Conditions ---
        self.inlet_pressure_co2 = 0.15 # Density proxy for pressure
        self.outlet_pressure = 0.10
