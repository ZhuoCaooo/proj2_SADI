'''THIS IS THE MODIFIED CONSTRUCTION OF DENSITY-BASED CELL MAPS
   For LC trajectories, only using the last 50 timesteps'''
import numpy as np
import pickle
import os
from collections import defaultdict

# Define key parameters in one place
RESOLUTION = 0.01  # Cell resolution
LC_TIMESTEPS = 50  # Number of timesteps to use from the end of LC trajectories

# Load data
folder_path = 'cell_maps_train_data'
file_path = os.path.join(folder_path, 'cnn_probs_and_truth_2.0s_march.pkl')

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    predictions_prob_meta = data['predictions_probabilities']
    ground_truth_each_timestep = data['ground_truth']

print(f"Successfully loaded data with {len(ground_truth_each_timestep)} trajectories")
print(f"Using cell resolution: {RESOLUTION}")
print(f"Using last {LC_TIMESTEPS} timesteps for LC trajectories")

def identify_trajectory_type(ground_truth):
    """Identify if trajectory is pure LK or ends with LC."""
    final_label = ground_truth[-1]
    if final_label == 0:
        return 'LK'
    else:
        return 'LC'

def build_modified_density_based_cell_maps(predictions_prob_meta, ground_truth_each_timestep, resolution, lc_timesteps):
    """Build cell maps that track occurrence density for reliability assessment.
    For LC trajectories, only use the last lc_timesteps."""

    # Initialize density maps for all cells
    lk_cell_counts = defaultdict(int)
    lc_cell_counts = defaultdict(int)

    # Track total prediction counts for normalization
    total_lk_predictions = 0
    total_lc_predictions = 0

    # Count trajectories by type
    num_lk_traj = 0
    num_lc_traj = 0

    # Track statistics about trajectory lengths
    lc_traj_lengths = []

    # Process each trajectory
    for traj_idx in range(len(predictions_prob_meta)):
        pred_sequence = predictions_prob_meta[traj_idx]
        true_sequence = ground_truth_each_timestep[traj_idx][:len(pred_sequence)]

        traj_type = identify_trajectory_type(true_sequence)

        if traj_type == 'LK':
            num_lk_traj += 1
            # Use all timesteps for LK trajectories
            for t in range(len(pred_sequence)):
                pred_probs = pred_sequence[t]

                # Map to cell
                n1 = int(pred_probs[1] / resolution)  # LCL
                n2 = int(pred_probs[2] / resolution)  # LCR
                cell = (n1, n2)

                # Update LK map
                lk_cell_counts[cell] += 1
                total_lk_predictions += 1

        else:  # LC trajectory
            num_lc_traj += 1
            lc_traj_lengths.append(len(pred_sequence))

            # For LC trajectories, use only the last lc_timesteps
            start_idx = max(0, len(pred_sequence) - lc_timesteps)

            for t in range(start_idx, len(pred_sequence)):
                pred_probs = pred_sequence[t]

                # Map to cell
                n1 = int(pred_probs[1] / resolution)  # LCL
                n2 = int(pred_probs[2] / resolution)  # LCR
                cell = (n1, n2)

                # Update LC map
                lc_cell_counts[cell] += 1
                total_lc_predictions += 1

    # Create the final density maps
    lk_density_map = {}
    lc_density_map = {}

    # Store just the raw count for each cell in the LK map
    for cell, count in lk_cell_counts.items():
        lk_density_map[cell] = {
            'count': count
        }

    # Store just the raw count for each cell in the LC map
    for cell, count in lc_cell_counts.items():
        lc_density_map[cell] = {
            'count': count
        }

    # Print statistics about LC trajectory lengths
    if lc_traj_lengths:
        print(f"\nLC Trajectory Length Statistics:")
        print(f"  Min length: {min(lc_traj_lengths)}")
        print(f"  Max length: {max(lc_traj_lengths)}")
        print(f"  Mean length: {np.mean(lc_traj_lengths):.2f}")
        print(f"  Using last {lc_timesteps} timesteps for LC trajectories")

    return lk_density_map, lc_density_map, num_lk_traj, num_lc_traj

# Build density-based cell maps with the modified approach
lk_density_map, lc_density_map, num_lk_traj, num_lc_traj = build_modified_density_based_cell_maps(
    predictions_prob_meta,
    ground_truth_each_timestep,
    resolution=RESOLUTION,
    lc_timesteps=LC_TIMESTEPS
)

# Print statistics
print(f"\nTrajectory Statistics:")
print(f"Total LK trajectories: {num_lk_traj}")
print(f"Total LC trajectories: {num_lc_traj}")
print(f"Number of unique cells in LK map: {len(lk_density_map)}")
print(f"Number of unique cells in LC map: {len(lc_density_map)}")

# Analyze some example cells from each map
print("\nExample LK Cell Analysis:")
for cell, info in list(sorted(lk_density_map.items(), key=lambda x: x[1]['count'], reverse=True))[:5]:
    lcl_range = (cell[0] * RESOLUTION, (cell[0] + 1) * RESOLUTION)
    lcr_range = (cell[1] * RESOLUTION, (cell[1] + 1) * RESOLUTION)
    print(f"\nCell {cell} (LCL≈{lcl_range[0]:.2f}-{lcl_range[1]:.2f}, LCR≈{lcr_range[0]:.2f}-{lcr_range[1]:.2f}):")
    print(f"Count: {info['count']}")

print("\nExample LC Cell Analysis:")
for cell, info in list(sorted(lc_density_map.items(), key=lambda x: x[1]['count'], reverse=True))[:5]:
    lcl_range = (cell[0] * RESOLUTION, (cell[0] + 1) * RESOLUTION)
    lcr_range = (cell[1] * RESOLUTION, (cell[1] + 1) * RESOLUTION)
    print(f"\nCell {cell} (LCL≈{lcl_range[0]:.2f}-{lcl_range[1]:.2f}, LCR≈{lcr_range[0]:.2f}-{lcr_range[1]:.2f}):")
    print(f"Count: {info['count']}")

# Also report on cells with very low counts
print("\nLow Count Cell Analysis:")
low_count_lk = [cell for cell, info in lk_density_map.items() if info['count'] < 5]
low_count_lc = [cell for cell, info in lc_density_map.items() if info['count'] < 5]
print(f"LK map cells with count < 5: {len(low_count_lk)} ({len(low_count_lk)/len(lk_density_map)*100:.2f}%)")
print(f"LC map cells with count < 5: {len(low_count_lc)} ({len(low_count_lc)/len(lc_density_map)*100:.2f}%)")

# Save the density-based maps
results = {
    'lk_density_map': lk_density_map,
    'lc_density_map': lc_density_map,
    'num_lk_trajectories': num_lk_traj,
    'num_lc_trajectories': num_lc_traj,
    'parameters': {
        'resolution': RESOLUTION,
        'lc_timesteps': LC_TIMESTEPS
    }
}

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Create the full file path by joining folder path and filename
output_file_path = os.path.join(folder_path, f'density_cell_maps_2.0s_march.pkl')

# Save the pickle file
with open(output_file_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\nModified density-based cell maps saved to {output_file_path}")

# Add correlation analysis after building the maps
lk_counts = []
lc_counts = []
common_cells = set(lk_density_map.keys()).intersection(set(lc_density_map.keys()))

for cell in common_cells:
    lk_count = lk_density_map[cell]['count']
    lc_count = lc_density_map[cell]['count']
    lk_counts.append(lk_count)
    lc_counts.append(lc_count)

# Calculate Pearson correlation
from scipy.stats import pearsonr
corr, p_value = pearsonr(lk_counts, lc_counts)
print(f"\nCorrelation Analysis:")
print(f"Number of common cells: {len(common_cells)}")
print(f"Pearson correlation coefficient: {corr:.4f} (p-value: {p_value:.6f})")

# Optional: Create and save a visualization of the density maps
try:
    import matplotlib.pyplot as plt

    # Create a heatmap visualization of the density
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prepare LK heatmap data
    max_n1 = max(cell[0] for cell in lk_density_map.keys()) + 1
    max_n2 = max(cell[1] for cell in lk_density_map.keys()) + 1
    lk_heatmap = np.zeros((max_n1, max_n2))

    for cell, info in lk_density_map.items():
        # Use log1p for visualization to handle large ranges better
        lk_heatmap[cell[0], cell[1]] = np.log1p(info['count'])

    # Prepare LC heatmap data
    max_n1 = max(max_n1, max(cell[0] for cell in lc_density_map.keys()) + 1)
    max_n2 = max(max_n2, max(cell[1] for cell in lc_density_map.keys()) + 1)
    lc_heatmap = np.zeros((max_n1, max_n2))

    for cell, info in lc_density_map.items():
        # Use log1p for visualization to handle large ranges better
        lc_heatmap[cell[0], cell[1]] = np.log1p(info['count'])

    # Plot heatmaps
    im1 = ax1.imshow(lk_heatmap, cmap='viridis', aspect='auto')
    ax1.set_title('LK Density Map (Log Scale)')
    ax1.set_xlabel('LCR Probability Bin')
    ax1.set_ylabel('LCL Probability Bin')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(lc_heatmap, cmap='viridis', aspect='auto')
    ax2.set_title(f'LC Density Map (Log Scale)')
    ax2.set_xlabel('LCR Probability Bin')
    ax2.set_ylabel('LCL Probability Bin')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    print("Heatmap visualization created.")

    # Save the visualization
    plt.show()

    # Also create a scatter plot to visualize the correlation
    plt.figure(figsize=(10, 8))
    plt.scatter(lk_counts, lc_counts, alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('LK Cell Count (log scale)')
    plt.ylabel('LC Cell Count (log scale)')
    plt.title(f'Correlation between LK and LC Cell Counts\nUsing Last {LC_TIMESTEPS} LC Timesteps')

    # Add line of equality for reference
    max_val = max(max(lk_counts), max(lc_counts))
    min_val = min(min(lk_counts), min(lc_counts))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the correlation plot
    plt.show()

except ImportError:
    print("Matplotlib not available, skipping visualization.")