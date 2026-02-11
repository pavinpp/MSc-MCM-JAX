import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
import random

# --- 1. BASELINE GEOMETRY GENERATOR ---
def generate_base_rock(nx, ny, porosity_target=0.4):
    """Generates the clean, initial sandstone."""
    np.random.seed(42)
    noise = np.random.normal(size=(nx, ny))
    smooth = gaussian_filter(noise, sigma=4) # Sigma controls grain size
    # Threshold to match target porosity
    threshold = np.percentile(smooth, 100 * (1 - porosity_target))
    mask = (smooth > threshold).astype(int) # 1=Fluid, 0=Solid
    return mask

# --- 2. THE THREE THESIS SCENARIOS ---

def case_1_uniform_coating(mask, thickness):
    """
    Proposal Case 1: Uniform Pore-Wall Coating.
    Grows solid uniformly from existing walls.
    """
    solid_mask = 1 - mask
    # Calculate distance from walls
    dist = distance_transform_edt(1 - solid_mask)
    # Clog pixels that are close to walls
    clogged = (dist <= thickness)
    return 1 - clogged.astype(int)

def case_2_throat_clogging(mask, target_porosity_reduction):
    """
    Proposal Case 2: Preferential Pore-Throat Clogging.
    Identifies the NARROWEST points and fills them first.
    """
    solid_mask = 1 - mask
    # Distance map: Small values = narrow throats / near walls
    # Large values = wide pore bodies
    dist = distance_transform_edt(1 - solid_mask)
    
    # We only want to fill FLUID nodes (dist > 0)
    fluid_indices = np.where(dist > 0)
    fluid_distances = dist[fluid_indices]
    
    # Sort fluid pixels by "narrowness" (distance to wall)
    # We want to clog the SMALLEST distance values first (the throats)
    sorted_indices = np.argsort(fluid_distances)
    
    # Determine how many pixels to clog to match the reduction
    n_to_clog = int(len(fluid_distances) * target_porosity_reduction)
    
    # Create new mask
    new_mask = mask.copy()
    
    # Clog the narrowest N pixels
    rows = fluid_indices[0][sorted_indices[:n_to_clog]]
    cols = fluid_indices[1][sorted_indices[:n_to_clog]]
    new_mask[rows, cols] = 0 # Turn to solid
    
    return new_mask

def case_3_stochastic_nucleation(mask, n_crystals, crystal_radius):
    """
    Proposal Case 3: Stochastic Nucleation.
    Random crystals appear in the middle of pore space.
    """
    new_mask = mask.copy()
    nx, ny = mask.shape
    
    # Find all available fluid points
    fluid_coords = np.argwhere(mask == 1)
    
    # Randomly pick N sites
    if len(fluid_coords) > n_crystals:
        indices = np.random.choice(len(fluid_coords), n_crystals, replace=False)
        sites = fluid_coords[indices]
        
        # Grow a crystal at each site
        for (r, c) in sites:
            # Simple square/circle stamp
            r_min, r_max = max(0, r-crystal_radius), min(nx, r+crystal_radius+1)
            c_min, c_max = max(0, c-crystal_radius), min(ny, c+crystal_radius+1)
            new_mask[r_min:r_max, c_min:c_max] = 0
            
    return new_mask

# --- 3. VISUALIZATION & COMPARISON ---
def main():
    NX, NY = 200, 200
    
    print("Generating Base Rock...")
    base = generate_base_rock(NX, NY, porosity_target=0.5)
    base_porosity = np.mean(base)
    print(f"Base Porosity: {base_porosity:.2%}")

    # --- Generate Scenarios ---
    # 1. Coating: Thickness of 2 pixels
    rock_coating = case_1_uniform_coating(base, thickness=1.5)
    
    # 2. Throats: Reduce porosity by roughly same amount as coating
    # Calculate how much porosity we lost in Case 1 to match it in Case 2
    p_loss = (np.sum(base) - np.sum(rock_coating)) / np.sum(base)
    rock_throats = case_2_throat_clogging(base, target_porosity_reduction=p_loss)
    
    # 3. Nucleation: Add random crystals
    rock_nucleation = case_3_stochastic_nucleation(base, n_crystals=80, crystal_radius=2)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Helper to plot
    def plot_rock(ax, data, title):
        ax.imshow(data, cmap='binary_r') # White=Fluid, Black=Solid
        ax.set_title(f"{title}\nPhi={np.mean(data):.1%}")
        ax.axis('off')

    plot_rock(axes[0], base, "Base Rock")
    plot_rock(axes[1], rock_coating, "Case 1: Uniform Coating")
    plot_rock(axes[2], rock_throats, "Case 2: Throat Clogging")
    plot_rock(axes[3], rock_nucleation, "Case 3: Stochastic")

    plt.tight_layout()
    plt.show()
    print("Comparison image generated. Notice the connectivity differences!")

if __name__ == "__main__":
    main()