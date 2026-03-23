'''THIS IS THE CONSTRUCTION OF EMPIRICAL CELL MAPS'''
import numpy as np
import pickle
import os
from collections import defaultdict



# Load data
# Define the folder path
folder_path = 'cell_maps_train_data'

# Create the full file path
file_path = os.path.join(folder_path, 'cnn_probs_and_truth_0.0s_new.pkl')

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    predictions_prob_meta = data['predictions_probabilities']
    ground_truth_each_timestep = data['ground_truth']




def identify_trajectory_type(ground_truth):
    """Identify if trajectory is pure LK or ends with LC."""
    final_label = ground_truth[-1]
    if final_label == 0:
        return 'LK'
    else:
        return 'LC'


def build_separate_cell_maps(predictions_prob_meta, ground_truth_each_timestep, resolution, min_samples,
                             lc_timesteps):
    """Build separate cell maps for LK and LC trajectories."""

    # Initialize separate counters for LK and LC
    lk_cell_counts = defaultdict(lambda: {'total': 0, 'LK': 0, 'LCL': 0, 'LCR': 0})
    lc_cell_counts = defaultdict(lambda: {'total': 0, 'LK': 0, 'LCL': 0, 'LCR': 0})

    # Store samples for analysis
    lk_cell_samples = defaultdict(list)
    lc_cell_samples = defaultdict(list)

    # Process each trajectory
    num_lk_traj = 0
    num_lc_traj = 0

    for traj_idx in range(len(predictions_prob_meta)):
        pred_sequence = predictions_prob_meta[traj_idx]  # Shape (126, 3)
        true_sequence = ground_truth_each_timestep[traj_idx][:126]

        traj_type = identify_trajectory_type(true_sequence)

        if traj_type == 'LK':
            num_lk_traj += 1
            # Use all timesteps for LK trajectories
            for t in range(len(pred_sequence)):
                pred_probs = pred_sequence[t]
                true_label = int(true_sequence[t])

                # Map to cell
                n1 = int(pred_probs[1] / resolution)  # LCL
                n2 = int(pred_probs[2] / resolution)  # LCR
                cell = (n1, n2)

                # Update LK map
                lk_cell_counts[cell]['total'] += 1
                lk_cell_counts[cell]['LK' if true_label == 0 else 'LCL' if true_label == 1 else 'LCR'] += 1

                # Store sample
                lk_cell_samples[cell].append({
                    'traj_idx': traj_idx,
                    'timestep': t,
                    'pred_probs': pred_probs,
                    'true_label': true_label
                })

        else:  # LC trajectory
            num_lc_traj += 1
            # Use only last lc_timesteps timesteps for LC trajectories
            # Or more generally, to ensure we only get LC labels:
            start_idx = next((i for i, label in enumerate(true_sequence) if label != 0), len(true_sequence) - 1)
            for t in range(start_idx, len(pred_sequence)):
                pred_probs = pred_sequence[t]
                true_label = int(true_sequence[t])

                # Map to cell
                n1 = int(pred_probs[1] / resolution)  # LCL
                n2 = int(pred_probs[2] / resolution)  # LCR
                cell = (n1, n2)

                # Update LC map
                lc_cell_counts[cell]['total'] += 1
                lc_cell_counts[cell]['LK' if true_label == 0 else 'LCL' if true_label == 1 else 'LCR'] += 1

                # Store sample
                lc_cell_samples[cell].append({
                    'traj_idx': traj_idx,
                    'timestep': t,
                    'pred_probs': pred_probs,
                    'true_label': true_label
                })

            # Debug cell updates for the first few trajectories
            if traj_idx < 3:  # Only for first 3 trajectories
                print(f"Trajectory {traj_idx} (LC):")
                for t in range(start_idx, min(start_idx + 10, len(pred_sequence))):  # First 10 timesteps
                    pred_probs = pred_sequence[t]
                    true_label = int(true_sequence[t])
                    n1 = int(pred_probs[1] / resolution)
                    n2 = int(pred_probs[2] / resolution)
                    cell = (n1, n2)
                    print(f"  Timestep {t}: true_label={true_label}, cell={cell}")

    # Convert counts to frequencies
    lk_distributions = {}
    lc_distributions = {}

    # Process LK cells
    for cell, counts in lk_cell_counts.items():
        if counts['total'] >= min_samples:
            lk_distributions[cell] = {
                'LK': counts['LK'] / counts['total'],
                'LCL': counts['LCL'] / counts['total'],
                'LCR': counts['LCR'] / counts['total'],
                'sample_count': counts['total'],
                'samples': lk_cell_samples[cell]
            }

    # Process LC cells
    for cell, counts in lc_cell_counts.items():
        if counts['total'] >= min_samples:
            lc_distributions[cell] = {
                'LK': counts['LK'] / counts['total'],
                'LCL': counts['LCL'] / counts['total'],
                'LCR': counts['LCR'] / counts['total'],
                'sample_count': counts['total'],
                'samples': lc_cell_samples[cell]
            }

    return lk_distributions, lc_distributions, num_lk_traj, num_lc_traj



# Build separate cell maps
lk_distributions, lc_distributions, num_lk_traj, num_lc_traj = build_separate_cell_maps(
    predictions_prob_meta,
    ground_truth_each_timestep,
    resolution=0.02,
    min_samples=10,
    lc_timesteps=50
)


print(f"\nTrajectory Statistics:")
print(f"Total LK trajectories: {num_lk_traj}")
print(f"Total LC trajectories: {num_lc_traj}")
print(f"Number of valid LK cells: {len(lk_distributions)}")
print(f"Number of valid LC cells: {len(lc_distributions)}")

# Analyze some example cells from each map
print("\nExample LK Cell Analysis:")
for cell, info in list(lk_distributions.items())[:3]:
    lcl_range = (cell[0] * 0.01, (cell[0] + 1) * 0.01)
    lcr_range = (cell[1] * 0.01, (cell[1] + 1) * 0.01)
    print(f"\nCell {cell} (LCL≈{lcl_range[0]:.2f}-{lcl_range[1]:.2f}, LCR≈{lcr_range[0]:.2f}-{lcr_range[1]:.2f}):")
    print(f"Samples: {info['sample_count']}")
    print(f"Frequencies: LK={info['LK']:.3f}, LCL={info['LCL']:.3f}, LCR={info['LCR']:.3f}")

print("\nExample LC Cell Analysis:")
for cell, info in list(lc_distributions.items())[:3]:
    lcl_range = (cell[0] * 0.01, (cell[0] + 1) * 0.01)
    lcr_range = (cell[1] * 0.01, (cell[1] + 1) * 0.01)
    print(f"\nCell {cell} (LCL≈{lcl_range[0]:.2f}-{lcl_range[1]:.2f}, LCR≈{lcr_range[0]:.2f}-{lcr_range[1]:.2f}):")
    print(f"Samples: {info['sample_count']}")
    print(f"Frequencies: LK={info['LK']:.3f}, LCL={info['LCL']:.3f}, LCR={info['LCR']:.3f}")

# Save the separate distributions
results = {
    'lk_distributions': lk_distributions,
    'lc_distributions': lc_distributions,
    'num_lk_trajectories': num_lk_traj,
    'num_lc_trajectories': num_lc_traj,
    'parameters': {
        'resolution': 0.01,
        'min_samples': 6,
        'lc_timesteps': 50
    }
}

# Define the folder path
folder_path = 'cell_maps_train_data'

# Create the full file path by joining folder path and filename
file_path = os.path.join(folder_path, 'separate_cell_maps_0.0s_new.pkl')

# Save the pickle file
with open(file_path, 'wb') as f:
    pickle.dump(results, f)

import numpy as np
from collections import Counter


# Identify LC trajectories (trajectories that end with a non-zero label)
lc_trajectories = []
lc_indices = []

for i, trajectory in enumerate(ground_truth_each_timestep):
    # Check if the last value is non-zero (indicating LC)
    if trajectory[-1] != 0:
        lc_trajectories.append(trajectory)
        lc_indices.append(i)

print(f"Found {len(lc_trajectories)} lane change trajectories")

# Analyze the first few LC trajectories
sample_size = min(5, len(lc_trajectories))

for i in range(sample_size):
    traj = lc_trajectories[i]
    traj_idx = lc_indices[i]

    # Count labels in the trajectory
    label_counts = Counter(traj)

    # Calculate when the lane change starts (first non-zero label)
    lc_start = next((j for j, label in enumerate(traj) if label != 0), len(traj))

    # Print trajectory information
    print(f"\nTrajectory {traj_idx} (length: {len(traj)}):")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Lane change starts at timestep: {lc_start} (out of {len(traj)})")

    # Print last 60 timesteps to see the transition pattern (simplified)
    last_60 = traj[-60:] if len(traj) >= 60 else traj
    print(f"Last 60 timesteps: {[int(x) for x in last_60]}")

# Analyze LC start points across all LC trajectories
lc_start_points = []
lc_durations = []
label_types = []

for traj in lc_trajectories:
    # Find first non-zero label
    lc_start = next((j for j, label in enumerate(traj) if label != 0), len(traj))
    lc_start_points.append(lc_start)

    # Calculate duration of LC phase
    lc_duration = len(traj) - lc_start
    lc_durations.append(lc_duration)

    # Record the type of LC (1 for LCL, 2 for LCR)
    final_label = traj[-1]
    label_types.append(final_label)

# Calculate statistics
avg_start = sum(lc_start_points) / len(lc_start_points)
avg_duration = sum(lc_durations) / len(lc_durations)

print(f"\nStatistics across {len(lc_trajectories)} LC trajectories:")
print(f"Average LC start point: {avg_start:.2f} timesteps from beginning")
print(f"Average LC duration: {avg_duration:.2f} timesteps")

# Calculate percentiles for LC start points
percentiles = [10, 25, 50, 75, 90]
lc_start_percentiles = np.percentile(lc_start_points, percentiles).tolist()

print("\nLC start point percentiles:")
for p, value in zip(percentiles, lc_start_percentiles):
    print(f"{p}th percentile: {value:.1f}")

# Calculate statistics for the last 50 timesteps
last_50_gt_stats = []

for traj in lc_trajectories:
    last_50 = traj[-50:] if len(traj) >= 50 else traj
    lk_count = sum(1 for label in last_50 if label == 0)
    lcl_count = sum(1 for label in last_50 if label == 1)
    lcr_count = sum(1 for label in last_50 if label == 2)

    last_50_gt_stats.append({
        'LK_percentage': (lk_count / len(last_50)) * 100,
        'LCL_percentage': (lcl_count / len(last_50)) * 100,
        'LCR_percentage': (lcr_count / len(last_50)) * 100
    })

# Calculate average percentages
avg_lk_percentage = sum(stats['LK_percentage'] for stats in last_50_gt_stats) / len(last_50_gt_stats)
avg_lcl_percentage = sum(stats['LCL_percentage'] for stats in last_50_gt_stats) / len(last_50_gt_stats)
avg_lcr_percentage = sum(stats['LCR_percentage'] for stats in last_50_gt_stats) / len(last_50_gt_stats)

print("\nIn the last 50 timesteps of LC trajectories:")
print(f"Average LK percentage: {avg_lk_percentage:.2f}%")
print(f"Average LCL percentage: {avg_lcl_percentage:.2f}%")
print(f"Average LCR percentage: {avg_lcr_percentage:.2f}%")