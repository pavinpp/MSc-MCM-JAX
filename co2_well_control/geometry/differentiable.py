# co2_well_control/geometry/differentiable.py
import jax.numpy as jnp
import jax

def get_soft_mask(width_param, nx, ny):
    """
    Creates a differentiable mask for a fracture.
    width_param: A scalar float controlling the channel width.
    
    Returns:
    mask: (nx, ny) array where 1.0 = Solid, 0.0 = Fluid.
          Values between 0 and 1 represent the wall interface.
    """
    # Create grid coordinates
    y = jnp.arange(ny)
    center = ny / 2.0
    
    # Calculate distance from center line
    dist = jnp.abs(y - center)
    
    # Sigmoid function to create the walls
    # steepness: controls how sharp the wall is (higher = sharper)
    steepness = 1.0 
    
    # Logic:
    # If dist < width, we want Fluid (0)
    # If dist > width, we want Solid (1)
    # Sigmoid(x) goes 0->1. We want transition at dist = width.
    # So we use sigmoid(steepness * (dist - width))
    
    transverse_profile = jax.nn.sigmoid(steepness * (dist - width_param))
    
    # Broadcast to full domain (nx, ny)
    mask = jnp.tile(transverse_profile[None, :], (nx, 1))
    
    return mask
