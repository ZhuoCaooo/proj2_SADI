'''
Step 5: Construct 2D Density-Based Empirical Probability Map (Corrected)
- Goal: Reproduce the highD density map logic (Probability Bins instead of Time Windows).
- Input: ad4che_cnn_probs_and_truth.pkl (from Step 4)
- Output: 2D Density Maps (LK and LC) and visualization.
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION (Aligned with highD script)
# ============================================================================
INPUT_FILE = "ad4che_cnn_probs_and_truth.pkl"
OUTPUT_FILE = "ad4che_density_maps.pkl"

# Grid Parameters
RESOLUTION = 0.01  # Bin size (e.g., 0.02 = 50x50 grid)
GRID_SIZE = int(1.0 / RESOLUTION) + 1

# Filtering Logic (Critical for reproducing highD method)
# highD script used "Last 50 timesteps".
# With Stride=1, 50 timesteps is approx 3-4 windows.
LC_LAST_N_WINDOWS = 100
USE_LOG_SCALE = True

# ============================================================================
# LOGIC
# ============================================================================

def get_bin_idx(prob):
    """Map a probability (0-1) to a grid index."""
    idx = int(prob / RESOLUTION)
    return min(idx, GRID_SIZE - 1)

def construct_density_map():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run Step 4 first.")
        return

    print("Loading Step 4 data...")
    with open(INPUT_FILE, 'rb') as f:
        # Structure: list of [probs_array, truth_array]
        # probs_array shape: (num_windows, 3) -> [LK, LCL, LCR]
        all_data = pickle.load(f)

    # Initialize 2D Heatmaps (Rows=LCL, Cols=LCR)
    # This matches the axes in your provided highD snippet
    lk_density_map = np.zeros((GRID_SIZE, GRID_SIZE))
    lc_density_map = np.zeros((GRID_SIZE, GRID_SIZE))

    lk_count = 0
    lc_count = 0

    print(f"Processing {len(all_data)} trajectories...")

    for i, (probs, truth) in enumerate(all_data):
        # 1. Identify Trajectory Type (LC or LK)
        # We check the LAST truth label to define the trajectory type
        final_label = truth[-1]

        # 0=LK, 1=LCL, 2=LCR
        is_lc = (final_label != 0)

        num_windows = len(probs)

        # 2. Select Windows to Process
        if is_lc:
            # For LC, we usually focus on the execution phase (last N windows)
            # strictly matching your highD 'last 50 timesteps' logic
            start_idx = max(0, num_windows - LC_LAST_N_WINDOWS)
            windows_to_process = range(start_idx, num_windows)
        else:
            # For LK, we use the whole stable trajectory
            windows_to_process = range(num_windows)

        # 3. Fill the Grid
        for w_idx in windows_to_process:
            # Get probabilities for this window
            p_lk = probs[w_idx][0]
            p_lcl = probs[w_idx][1]
            p_lcr = probs[w_idx][2]

            # Calculate Bins
            # X-axis = LCR Prob, Y-axis = LCL Prob
            bin_x = get_bin_idx(p_lcr)
            bin_y = get_bin_idx(p_lcl)

            if is_lc:
                lc_density_map[bin_y, bin_x] += 1
                lc_count += 1
            else:
                lk_density_map[bin_y, bin_x] += 1
                lk_count += 1

    print(f"Done. Processed {lk_count} LK windows and {lc_count} LC windows.")

    # ========================================================================
    # VISUALIZATION (Reproducing the plots)
    # ========================================================================

    # Apply Log Scale (log1p to handle zeros)
    if USE_LOG_SCALE:
        plot_lk = np.log1p(lk_density_map)
        plot_lc = np.log1p(lc_density_map)
    else:
        plot_lk = lk_density_map
        plot_lc = lc_density_map

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot LK Map
    # origin='lower' puts (0,0) at bottom-left
    im1 = ax1.imshow(plot_lk, cmap='viridis', origin='lower',
                     extent=[0, 1, 0, 1], aspect='auto')
    ax1.set_title(f'LK Density Map ({"Log" if USE_LOG_SCALE else "Linear"})')
    ax1.set_xlabel('Probability LCR')
    ax1.set_ylabel('Probability LCL')
    plt.colorbar(im1, ax=ax1)

    # Plot LC Map
    im2 = ax2.imshow(plot_lc, cmap='viridis', origin='lower',
                     extent=[0, 1, 0, 1], aspect='auto')
    ax2.set_title(f'LC Density Map')
    ax2.set_xlabel('Probability LCR')
    ax2.set_ylabel('Probability LCL')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # SAVE
    # ========================================================================
    save_data = {
        'lk_map': lk_density_map,
        'lc_map': lc_density_map,
        'resolution': RESOLUTION,
        'lc_windows_used': LC_LAST_N_WINDOWS
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Density maps saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    construct_density_map()