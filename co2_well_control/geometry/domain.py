# co2_well_control/geometry/domain.py
import jax.numpy as jnp

def create_domain(cfg):
    """
    Creates the simulation domain masks.
    Returns:
        solid_mask: 1.0 where solid (matrix), 0.0 where fluid (fracture)
    """
    nx, ny = cfg.nx, cfg.ny
    
    # Initialize all as solid (matrix)
    solid_mask = jnp.ones((nx, ny))
    
    # Create a fracture channel in the middle
    # Fracture width = 20 pixels
    fracture_start = ny // 2 - 10
    fracture_end = ny // 2 + 10
    
    # JAX arrays are immutable, so we use .at[].set()
    solid_mask = solid_mask.at[:, fracture_start:fracture_end].set(0.0)
    
    # Add roughness or blockage (optional for later)
    # solid_mask = solid_mask.at[100:110, fracture_start:fracture_end].set(1.0)
    
    return solid_mask
