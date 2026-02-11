import os

import matplotlib.pyplot as plt
import numpy as np


def save_saturation_map(final_state, filename):
    _, salt, mask = final_state

    salt_np = np.array(salt)
    mask_np = np.array(mask)

    plt.figure(figsize=(10, 5))
    plt.imshow(mask_np.T, cmap="gray", alpha=0.5, origin="lower")
    plt.imshow(salt_np.T, cmap="RdBu", alpha=0.8, origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="Salt Concentration (0=CO2, 1=Brine)")
    plt.title("Final Saturation Distribution")
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_performance_curves(history, strategy_name, save_path):
    sat_current, p_current = history
    steps = np.arange(len(sat_current))

    baseline_path = "data/outputs/baseline_history.npy"
    sat_baseline = None

    if os.path.exists(baseline_path) and strategy_name != "constant":
        data = np.load(baseline_path, allow_pickle=True).item()
        sat_baseline = data.get("saturation")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(steps, p_current, "r-", linewidth=1.5, label=f"{strategy_name} (Current)")
    ax[0].set_title(f"Injection Strategy: {strategy_name}")
    ax[0].set_ylabel("Pressure Offset")
    ax[0].set_xlabel("Time Step")
    ax[0].legend(loc="upper right")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(steps, sat_current, "r-", linewidth=2, label=f"{strategy_name}")

    if sat_baseline is not None:
        limit = min(len(sat_baseline), len(sat_current))
        ax[1].plot(steps[:limit], sat_baseline[:limit], "k:", linewidth=2, label="Baseline")

    ax[1].set_title("Saturation Efficiency")
    ax[1].set_ylabel("CO2 Saturation")
    ax[1].set_xlabel("Time Step")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Performance plot saved to {save_path}")
