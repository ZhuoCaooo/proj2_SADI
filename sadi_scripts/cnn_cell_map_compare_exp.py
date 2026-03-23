"""this file construct the empirical map in one map, using different lk lc trajectory ratio, and different t_lc to do
the comparison and visualization """
import numpy as np
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import random

import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define key parameters in one place
from scipy.spatial.distance import jensenshannon

RESOLUTION = 0.02  # Cell resolution
LC_TIMESTEPS_OPTIONS = [25, 50, 75, 100]  # Different options for LC timesteps to label
LK_LC_RATIOS = [(1, 1), (2, 1), (5, 1), (10, 1)]  # Different LK:LC ratios to experiment with

# Set up output directory for visualizations
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load data
folder_path = 'cell_maps_train_data'
file_path = os.path.join(folder_path, 'cnn_probs_and_truth_0.0s_new.pkl')

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    predictions_prob_meta = data['predictions_probabilities']
    ground_truth_each_timestep = data['ground_truth']

print(f"Successfully loaded data with {len(ground_truth_each_timestep)} trajectories")


def identify_trajectory_type(ground_truth):
    """Identify if trajectory is pure LK or ends with LC."""
    final_label = ground_truth[-1]
    if final_label == 0:
        return 'LK'
    else:
        return 'LC'


def build_combined_density_map(predictions_prob_meta, ground_truth_each_timestep,
                               lk_traj_indices, lc_traj_indices, resolution, lc_timesteps=None):
    """
    Build a combined cell map that tracks occurrence density.

    Args:
        predictions_prob_meta: Prediction probabilities for all trajectories
        ground_truth_each_timestep: Ground truth labels for all trajectories
        lk_traj_indices: Indices of LK trajectories to include
        lc_traj_indices: Indices of LC trajectories to include
        resolution: Cell resolution
        lc_timesteps: Number of timesteps to mark as LC in LC trajectories
                     If None, use ground truth labels as is

    Returns:
        combined_map: Combined density map
        class_balance: LK/LC counts
    """
    # Initialize density map for all cells
    combined_map = defaultdict(lambda: defaultdict(int))

    # Track class balance
    lk_count = 0
    lc_count = 0

    # Process each LK trajectory
    for traj_idx in lk_traj_indices:
        pred_sequence = predictions_prob_meta[traj_idx]
        true_sequence = ground_truth_each_timestep[traj_idx][:len(pred_sequence)]

        # Use all timesteps for LK trajectories (all labeled as LK)
        for t in range(len(pred_sequence)):
            pred_probs = pred_sequence[t]

            # Map to cell
            n1 = int(pred_probs[1] / resolution)  # LCL
            n2 = int(pred_probs[2] / resolution)  # LCR
            cell = (n1, n2)

            # Update combined map
            combined_map[cell]['LK'] += 1
            lk_count += 1

    # Process each LC trajectory
    for traj_idx in lc_traj_indices:
        pred_sequence = predictions_prob_meta[traj_idx]
        true_sequence = ground_truth_each_timestep[traj_idx][:len(pred_sequence)]

        # If lc_timesteps is specified, override the ground truth
        if lc_timesteps is not None:
            # All steps are LK except the last lc_timesteps
            modified_labels = np.zeros(len(true_sequence))

            # Mark last lc_timesteps as LC
            # (using label 1 for simplicity - could be either LCL or LCR)
            start_idx = max(0, len(true_sequence) - lc_timesteps)
            modified_labels[start_idx:] = 1
            true_sequence = modified_labels

        # Process each timestep
        for t in range(len(pred_sequence)):
            pred_probs = pred_sequence[t]
            label = true_sequence[t]

            # Map to cell
            n1 = int(pred_probs[1] / resolution)  # LCL
            n2 = int(pred_probs[2] / resolution)  # LCR
            cell = (n1, n2)

            # Update combined map based on label
            if label == 0:  # LK
                combined_map[cell]['LK'] += 1
                lk_count += 1
            else:  # LC (either LCL or LCR)
                combined_map[cell]['LC'] += 1
                lc_count += 1

    # Calculate final ratios for each cell
    for cell in combined_map:
        total = combined_map[cell]['LK'] + combined_map[cell]['LC']
        combined_map[cell]['total'] = total
        combined_map[cell]['lk_ratio'] = combined_map[cell]['LK'] / total if total > 0 else 0
        combined_map[cell]['lc_ratio'] = combined_map[cell]['LC'] / total if total > 0 else 0

    class_balance = {'LK': lk_count, 'LC': lc_count}
    return combined_map, class_balance


def analyze_and_visualize_map(combined_map, class_balance, experiment_type, experiment_value):
    """Analyze and visualize the combined density map."""
    # Get counts
    total_cells = len(combined_map)
    total_samples = class_balance['LK'] + class_balance['LC']
    lk_ratio = class_balance['LK'] / total_samples if total_samples > 0 else 0

    # Print statistics
    title = f"{experiment_type}: {experiment_value}"
    print(f"\n{title} Statistics:")
    print(f"Total cells: {total_cells}")
    print(f"Total samples: {total_samples}")
    print(f"LK samples: {class_balance['LK']} ({lk_ratio:.2%})")
    print(f"LC samples: {class_balance['LC']} ({1 - lk_ratio:.2%})")

    # Calculate entropy for each cell (measure of uncertainty)
    entropies = []
    for cell, info in combined_map.items():
        p_lk = info['lk_ratio']
        p_lc = info['lc_ratio']
        # Shannon entropy calculation
        entropy = 0
        if p_lk > 0:
            entropy -= p_lk * np.log2(p_lk)
        if p_lc > 0:
            entropy -= p_lc * np.log2(p_lc)
        if info['total'] >= 5:  # Only consider cells with enough samples
            entropies.append(entropy)

    avg_entropy = np.mean(entropies) if entropies else 0
    print(f"Average entropy: {avg_entropy:.4f}")

    # Calculate additional metrics for comparison
    lc_dominant_cells = sum(1 for cell, info in combined_map.items()
                            if info['total'] >= 5 and info['lc_ratio'] > 0.5)
    lk_dominant_cells = sum(1 for cell, info in combined_map.items()
                            if info['total'] >= 5 and info['lk_ratio'] > 0.5)

    print(f"LC dominant cells: {lc_dominant_cells} ({lc_dominant_cells / total_cells:.2%} of total)")
    print(f"LK dominant cells: {lk_dominant_cells} ({lk_dominant_cells / total_cells:.2%} of total)")

    # Calculate high-entropy cells (uncertain cells)
    high_entropy_threshold = 0.8
    high_entropy_cells = sum(1 for cell, info in combined_map.items()
                             if info['total'] >= 5 and
                             -info['lk_ratio'] * np.log2(info['lk_ratio'] + 1e-10)
                             - info['lc_ratio'] * np.log2(info['lc_ratio'] + 1e-10) > high_entropy_threshold)

    print(
        f"High entropy cells (>{high_entropy_threshold}): {high_entropy_cells} ({high_entropy_cells / total_cells:.2%} of total)")

    # Prepare visualization data
    max_n1 = max(cell[0] for cell in combined_map.keys()) + 1
    max_n2 = max(cell[1] for cell in combined_map.keys()) + 1

    # Create data arrays for visualization
    density_map = np.zeros((max_n1, max_n2))
    entropy_map = np.zeros((max_n1, max_n2))
    lc_ratio_map = np.zeros((max_n1, max_n2))

    # Fill in the maps
    for cell, info in combined_map.items():
        # Skip cells with too few samples
        if info['total'] < 5:
            continue

        # Log density for better visualization (add 1 to avoid log(0))
        density_map[cell[0], cell[1]] = np.log1p(info['total'])

        # Entropy calculation
        p_lk = info['lk_ratio']
        p_lc = info['lc_ratio']
        entropy = 0
        if p_lk > 0:
            entropy -= p_lk * np.log2(p_lk)
        if p_lc > 0:
            entropy -= p_lc * np.log2(p_lc)

        entropy_map[cell[0], cell[1]] = entropy

        # LC ratio
        lc_ratio_map[cell[0], cell[1]] = info['lc_ratio']

    # Create custom colormaps
    cmap_entropy = plt.cm.viridis
    cmap_lc_ratio = LinearSegmentedColormap.from_list("LK_LC", ["blue", "white", "red"])

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Cell Density
    im1 = axes[0].imshow(density_map, cmap='plasma', aspect='equal')
    axes[0].set_title(f'Cell Sample Count (Log Scale)\n{title}')
    axes[0].set_xlabel('LCR Probability Bin')
    axes[0].set_ylabel('LCL Probability Bin')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    # Plot 2: Entropy
    im2 = axes[1].imshow(entropy_map, cmap=cmap_entropy, vmin=0, vmax=1, aspect='equal')
    axes[1].set_title(f'Cell Entropy (Uncertainty)\n{title}')
    axes[1].set_xlabel('LCR Probability Bin')
    axes[1].set_ylabel('LCL Probability Bin')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    # Plot 3: LC Ratio (Class Balance)
    im3 = axes[2].imshow(lc_ratio_map, cmap=cmap_lc_ratio, vmin=0, vmax=1, aspect='equal')
    axes[2].set_title(f'LC Ratio (Blue=LK, Red=LC)\n{title}')
    axes[2].set_xlabel('LCR Probability Bin')
    axes[2].set_ylabel('LCL Probability Bin')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)

    plt.tight_layout()

    # Save the visualization
    filename = f"{experiment_type.replace(' ', '_')}_{str(experiment_value).replace(':', '_')}.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {file_path}")

    return {
        'cell_count': total_cells,
        'total_samples': total_samples,
        'lk_samples': class_balance['LK'],
        'lc_samples': class_balance['LC'],
        'lk_ratio': lk_ratio,
        'avg_entropy': avg_entropy,
        'lc_dominant_cells': lc_dominant_cells,
        'lk_dominant_cells': lk_dominant_cells,
        'high_entropy_cells': high_entropy_cells
    }


def visualize_comparison(experiment_results, experiment_type):
    """Create comparison visualizations for the experiment results."""
    metrics = ['avg_entropy', 'lk_ratio', 'high_entropy_cells']
    titles = {
        'avg_entropy': 'Average Entropy (Uncertainty)',
        'lk_ratio': 'LK Ratio',
        'high_entropy_cells': 'High Entropy Cells Count'
    }

    # Get experiment values and results
    experiment_values = list(experiment_results.keys())

    # Create a figure for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Get values
        values = [experiment_results[value][metric] for value in experiment_values]

        # For lk_ratio, convert to percentage
        if metric == 'lk_ratio':
            values = [v * 100 for v in values]
            plt.ylabel('Percentage (%)')
        else:
            plt.ylabel('Value')

        # Create bar chart
        bars = plt.bar(experiment_values, values, color='skyblue', width=0.6)

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            if metric == 'lk_ratio':
                label = f"{value:.1f}%"
            elif metric == 'high_entropy_cells':
                label = f"{value}"
            else:
                label = f"{value:.3f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01),
                label,
                ha='center',
                va='bottom',
                fontsize=10
            )

        # Set chart title and labels
        plt.title(f'{titles[metric]} Comparison')
        plt.xlabel(experiment_type)

        # For LK:LC ratio experiment, rotate x-labels for better readability
        if experiment_type == 'LK:LC Ratio':
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save the figure
        filename = f"comparison_{experiment_type.replace(' ', '_')}_{metric}.png"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison chart for {metric} saved to {file_path}")


# Separate trajectories by type
lk_trajectories = []
lc_trajectories = []

for traj_idx in range(len(predictions_prob_meta)):
    true_sequence = ground_truth_each_timestep[traj_idx]
    traj_type = identify_trajectory_type(true_sequence)

    if traj_type == 'LK':
        lk_trajectories.append(traj_idx)
    else:
        lc_trajectories.append(traj_idx)

print(f"Identified {len(lk_trajectories)} LK trajectories and {len(lc_trajectories)} LC trajectories")

# Experiment 1: Compare different LK:LC ratios
ratio_experiment_results = {}

for lk_ratio, lc_ratio in LK_LC_RATIOS:
    # Calculate how many trajectories to sample
    # Use LC trajectories as the base, since they're likely fewer
    n_lc = len(lc_trajectories)
    n_lk = min(len(lk_trajectories), int(n_lc * lk_ratio / lc_ratio))

    # In case we need to downsample LC trajectories to maintain ratio
    if n_lk == len(lk_trajectories):
        n_lc = int(n_lk * lc_ratio / lk_ratio)

    # Randomly sample trajectories
    sampled_lk = random.sample(lk_trajectories, n_lk)
    sampled_lc = random.sample(lc_trajectories, n_lc)

    print(f"\nExperiment with LK:LC ratio {lk_ratio}:{lc_ratio}")
    print(f"Using {n_lk} LK trajectories and {n_lc} LC trajectories")

    # Build and analyze the combined map
    combined_map, class_balance = build_combined_density_map(
        predictions_prob_meta,
        ground_truth_each_timestep,
        sampled_lk,
        sampled_lc,
        resolution=RESOLUTION
    )

    experiment_value = f"{lk_ratio}:{lc_ratio}"
    results = analyze_and_visualize_map(combined_map, class_balance, "LK:LC Ratio", experiment_value)
    ratio_experiment_results[experiment_value] = results

# Create comparison visualizations for ratio experiment
visualize_comparison(ratio_experiment_results, "LK:LC Ratio")

# Experiment 2: Impact of different LC labeling approaches
timesteps_experiment_results = {}

for lc_timesteps in LC_TIMESTEPS_OPTIONS:
    print(f"\nExperiment with {lc_timesteps} timesteps labeled as LC")

    # Use a balanced ratio for this experiment
    n_lc = len(lc_trajectories)
    n_lk = min(len(lk_trajectories), n_lc)  # 1:1 ratio

    sampled_lk = random.sample(lk_trajectories, n_lk)
    sampled_lc = random.sample(lc_trajectories, n_lc)

    # Build and analyze the combined map
    combined_map, class_balance = build_combined_density_map(
        predictions_prob_meta,
        ground_truth_each_timestep,
        sampled_lk,
        sampled_lc,
        resolution=RESOLUTION,
        lc_timesteps=lc_timesteps
    )

    experiment_value = lc_timesteps
    results = analyze_and_visualize_map(combined_map, class_balance, "LC Timesteps", experiment_value)
    timesteps_experiment_results[experiment_value] = results

# Create comparison visualizations for timesteps experiment
visualize_comparison(timesteps_experiment_results, "LC Timesteps")

# Create a table comparison view
print("\n===== EXPERIMENT RESULTS SUMMARY =====")

# LK:LC Ratio Experiment
print("\nLK:LC Ratio Experiment Results:")
print(f"{'Metric':<20} | " + " | ".join(f"{ratio:<10}" for ratio in ratio_experiment_results.keys()))
print("-" * 20 + "-+-" + "-" * 11 * len(ratio_experiment_results))

metrics = ['avg_entropy', 'lk_ratio', 'high_entropy_cells']
for metric in metrics:
    values = [ratio_experiment_results[ratio][metric] for ratio in ratio_experiment_results.keys()]

    if metric == 'lk_ratio':
        print(f"{metric:<20} | " + " | ".join(f"{value * 100:<8.1f}%" for value in values))
    elif metric == 'high_entropy_cells':
        print(f"{metric:<20} | " + " | ".join(f"{value:<10d}" for value in values))
    else:
        print(f"{metric:<20} | " + " | ".join(f"{value:<10.3f}" for value in values))

# LC Timesteps Experiment
print("\nLC Timesteps Experiment Results:")
print(f"{'Metric':<20} | " + " | ".join(f"{ts:<10}" for ts in timesteps_experiment_results.keys()))
print("-" * 20 + "-+-" + "-" * 11 * len(timesteps_experiment_results))

for metric in metrics:
    values = [timesteps_experiment_results[ts][metric] for ts in timesteps_experiment_results.keys()]

    if metric == 'lk_ratio':
        print(f"{metric:<20} | " + " | ".join(f"{value * 100:<8.1f}%" for value in values))
    elif metric == 'high_entropy_cells':
        print(f"{metric:<20} | " + " | ".join(f"{value:<10d}" for value in values))
    else:
        print(f"{metric:<20} | " + " | ".join(f"{value:<10.3f}" for value in values))

print("\nExperiments complete. Visualizations saved to the 'visualizations' directory.")

# ===========================================================================
# EXTENDED ANALYSIS: LK:LC RATIO SENSITIVITY ANALYSIS
# ===========================================================================
print("\n\n" + "=" * 80)
print("EXTENDED ANALYSIS: SENSITIVITY TO LK:LC RATIO (with LC_TIMESTEPS=25)")
print("=" * 80)

# Create a separate output directory for sensitivity analysis
sensitivity_dir = 'ratio_sensitivity_analysis'
os.makedirs(sensitivity_dir, exist_ok=True)

# Define more granular ratios for sensitivity testing
EXTENDED_LK_LC_RATIOS = [(1, 1), (2, 1), (5, 1), (7, 1), (10, 1)]
FIXED_LC_TIMESTEPS = 75  # Keep LC timesteps fixed for this analysis


def analyze_map_for_sensitivity(combined_map, class_balance):
    """
    Analyze the combined density map for sensitivity analysis.
    Returns detailed metrics for sensitivity comparison.
    """
    # Basic counts
    total_cells = len(combined_map)
    total_samples = class_balance['LK'] + class_balance['LC']
    lk_ratio = class_balance['LK'] / total_samples if total_samples > 0 else 0

    # Filter cells with sufficient samples
    valid_cells = {cell: info for cell, info in combined_map.items() if info['total'] >= 5}
    cells_with_samples = len(valid_cells)

    # Calculate entropy for each cell
    entropies = []
    lc_ratios = []
    lk_dominant_count = 0
    lc_dominant_count = 0

    for cell, info in valid_cells.items():
        p_lk = info['lk_ratio']
        p_lc = info['lc_ratio']

        # Shannon entropy calculation
        entropy = 0
        if p_lk > 0:
            entropy -= p_lk * np.log2(p_lk)
        if p_lc > 0:
            entropy -= p_lc * np.log2(p_lc)

        entropies.append(entropy)
        lc_ratios.append(p_lc)

        # Count dominant cells
        if p_lc > 0.5:
            lc_dominant_count += 1
        else:
            lk_dominant_count += 1

    # Calculate entropy statistics
    avg_entropy = np.mean(entropies) if entropies else 0
    median_entropy = np.median(entropies) if entropies else 0
    entropy_std = np.std(entropies) if entropies else 0
    entropy_quantiles = np.percentile(entropies, [25, 50, 75]) if entropies else [0, 0, 0]

    # Calculate LC ratio statistics
    lc_ratio_std = np.std(lc_ratios) if lc_ratios else 0
    lc_ratio_mean = np.mean(lc_ratios) if lc_ratios else 0

    # Calculate threshold metrics
    high_entropy_threshold = 0.8
    high_entropy_cells = sum(1 for e in entropies if e > high_entropy_threshold)
    high_entropy_percentage = high_entropy_cells / cells_with_samples if cells_with_samples > 0 else 0

    # Calculate decision boundary metrics
    medium_entropy_threshold = 0.6
    decision_boundary_cells = sum(1 for e in entropies if e > medium_entropy_threshold)
    decision_boundary_percentage = decision_boundary_cells / cells_with_samples if cells_with_samples > 0 else 0

    # Return all metrics
    return {
        'total_cells': total_cells,
        'cells_with_samples': cells_with_samples,
        'total_samples': total_samples,
        'lk_samples': class_balance['LK'],
        'lc_samples': class_balance['LC'],
        'lk_ratio': lk_ratio,
        'lc_ratio_mean': lc_ratio_mean,
        'lc_ratio_std': lc_ratio_std,
        'avg_entropy': avg_entropy,
        'median_entropy': median_entropy,
        'entropy_std': entropy_std,
        'entropy_q1': entropy_quantiles[0],
        'entropy_q2': entropy_quantiles[1],
        'entropy_q3': entropy_quantiles[2],
        'lk_dominant_cells': lk_dominant_count,
        'lc_dominant_cells': lc_dominant_count,
        'high_entropy_cells': high_entropy_cells,
        'high_entropy_percentage': high_entropy_percentage,
        'decision_boundary_cells': decision_boundary_cells,
        'decision_boundary_percentage': decision_boundary_percentage
    }


def calculate_map_differences(map1, map2):
    """
    Calculate differences between two cell maps.
    Returns the differences in LC ratios and entropy.
    """
    # Find cells present in both maps
    common_cells = set(map1.keys()) & set(map2.keys())

    # Initialize difference measures
    lc_ratio_diffs = []
    entropy_diffs = []
    js_divergence = []

    for cell in common_cells:
        # Skip cells with too few samples in either map
        if map1[cell]['total'] < 5 or map2[cell]['total'] < 5:
            continue

        # Calculate LC ratio difference
        lc_ratio_diff = abs(map1[cell]['lc_ratio'] - map2[cell]['lc_ratio'])
        lc_ratio_diffs.append(lc_ratio_diff)

        # Calculate entropy for each map
        entropy1 = 0
        if map1[cell]['lk_ratio'] > 0:
            entropy1 -= map1[cell]['lk_ratio'] * np.log2(map1[cell]['lk_ratio'])
        if map1[cell]['lc_ratio'] > 0:
            entropy1 -= map1[cell]['lc_ratio'] * np.log2(map1[cell]['lc_ratio'])

        entropy2 = 0
        if map2[cell]['lk_ratio'] > 0:
            entropy2 -= map2[cell]['lk_ratio'] * np.log2(map2[cell]['lk_ratio'])
        if map2[cell]['lc_ratio'] > 0:
            entropy2 -= map2[cell]['lc_ratio'] * np.log2(map2[cell]['lc_ratio'])

        # Calculate entropy difference
        entropy_diff = abs(entropy1 - entropy2)
        entropy_diffs.append(entropy_diff)

        # Calculate Jensen-Shannon divergence between distributions
        p = [map1[cell]['lk_ratio'], map1[cell]['lc_ratio']]
        q = [map2[cell]['lk_ratio'], map2[cell]['lc_ratio']]
        js = jensenshannon(p, q)
        js_divergence.append(js)

    return {
        'common_cells': len(common_cells),
        'mean_lc_ratio_diff': np.mean(lc_ratio_diffs) if lc_ratio_diffs else 0,
        'max_lc_ratio_diff': np.max(lc_ratio_diffs) if lc_ratio_diffs else 0,
        'mean_entropy_diff': np.mean(entropy_diffs) if entropy_diffs else 0,
        'max_entropy_diff': np.max(entropy_diffs) if entropy_diffs else 0,
        'mean_js_divergence': np.mean(js_divergence) if js_divergence else 0,
        'max_js_divergence': np.max(js_divergence) if js_divergence else 0
    }


def create_sensitivity_visualizations(map_data, metrics_data):
    """
    Create visualizations showing the sensitivity to LK:LC ratio changes.
    """
    # 1. Side-by-side comparison of LC ratio maps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Custom colormap for LC ratio
    cmap_lc_ratio = LinearSegmentedColormap.from_list("LK_LC", ["blue", "white", "red"])

    # Find the global max dimensions for consistent plotting
    max_n1 = 0
    max_n2 = 0
    for ratio_str, (combined_map, _) in map_data.items():
        if combined_map:
            max_n1 = max(max_n1, max(cell[0] for cell in combined_map.keys()) + 1)
            max_n2 = max(max_n2, max(cell[1] for cell in combined_map.keys()) + 1)

    # Plot LC ratio maps for each ratio
    for i, ratio_str in enumerate(map_data.keys()):
        if i >= len(axes):
            break

        combined_map, _ = map_data[ratio_str]

        # Create LC ratio map
        lc_ratio_map = np.zeros((max_n1, max_n2))

        # Fill in the map
        for cell, info in combined_map.items():
            if info['total'] >= 5:  # Skip cells with too few samples
                lc_ratio_map[cell[0], cell[1]] = info['lc_ratio']

        # Plot the map
        im = axes[i].imshow(lc_ratio_map, cmap=cmap_lc_ratio, vmin=0, vmax=1, aspect='equal')
        axes[i].set_title(f'LC Ratio Map (Blue=LK, Red=LC)\nLK:LC = {ratio_str}')

        if i >= len(axes) - 4:  # Only add x-labels to bottom row
            axes[i].set_xlabel('LCR Probability Bin')
        if i % 4 == 0:  # Only add y-labels to leftmost column
            axes[i].set_ylabel('LCL Probability Bin')

    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(sensitivity_dir, 'lc_ratio_map_comparison.png'), dpi=300)
    plt.close()

    # 2. Create entropy map comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Custom colormap for entropy
    cmap_entropy = plt.cm.viridis

    # Plot entropy maps for each ratio
    for i, ratio_str in enumerate(map_data.keys()):
        if i >= len(axes):
            break

        combined_map, _ = map_data[ratio_str]

        # Create entropy map
        entropy_map = np.zeros((max_n1, max_n2))

        # Fill in the map
        for cell, info in combined_map.items():
            if info['total'] >= 5:  # Skip cells with too few samples
                p_lk = info['lk_ratio']
                p_lc = info['lc_ratio']

                # Shannon entropy calculation
                entropy = 0
                if p_lk > 0:
                    entropy -= p_lk * np.log2(p_lk)
                if p_lc > 0:
                    entropy -= p_lc * np.log2(p_lc)

                entropy_map[cell[0], cell[1]] = entropy

        # Plot the map
        im = axes[i].imshow(entropy_map, cmap=cmap_entropy, vmin=0, vmax=1, aspect='equal')
        axes[i].set_title(f'Entropy Map (Uncertainty)\nLK:LC = {ratio_str}')

        if i >= len(axes) - 4:  # Only add x-labels to bottom row
            axes[i].set_xlabel('LCR Probability Bin')
        if i % 4 == 0:  # Only add y-labels to leftmost column
            axes[i].set_ylabel('LCL Probability Bin')

    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(sensitivity_dir, 'entropy_map_comparison.png'), dpi=300)
    plt.close()

    # 3. Create sensitivity metrics comparison
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')

    # Add ratio as a column and extract numeric ratio values
    metrics_df['ratio_str'] = metrics_df.index
    metrics_df['ratio_value'] = metrics_df['ratio_str'].apply(
        lambda x: float(x.split(':')[0]) / float(x.split(':')[1]))

    # Sort by ratio value
    metrics_df = metrics_df.sort_values('ratio_value')

    # List of important metrics to plot
    key_metrics = [
        'avg_entropy', 'entropy_std', 'high_entropy_percentage',
        'decision_boundary_percentage', 'lc_ratio_std'
    ]

    metric_titles = {
        'avg_entropy': 'Average Entropy',
        'entropy_std': 'Entropy Standard Deviation',
        'high_entropy_percentage': 'High Entropy Cells (%)',
        'decision_boundary_percentage': 'Decision Boundary Cells (%)',
        'lc_ratio_std': 'LC Ratio Standard Deviation'
    }

    # Plot each metric
    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(key_metrics):
        plt.subplot(3, 2, i + 1)

        # Extract values for plotting
        x = metrics_df['ratio_value']
        y = metrics_df[metric]

        # Plot the metric
        plt.plot(x, y, 'o-', linewidth=2, markersize=8)

        # Format percentage metrics
        if 'percentage' in metric:
            y_formatted = [f"{val:.1%}" for val in y]
            plt.ylabel('Percentage')
        else:
            y_formatted = [f"{val:.3f}" for val in y]
            plt.ylabel('Value')

        # Add value labels
        for i, (x_val, y_val, y_fmt) in enumerate(zip(x, y, y_formatted)):
            plt.annotate(y_fmt, (x_val, y_val), textcoords="offset points",
                         xytext=(0, 10), ha='center')

        plt.title(metric_titles[metric])
        plt.xlabel('LK:LC Ratio Value')
        plt.xscale('log')  # Log scale for ratio
        plt.grid(True, alpha=0.3)

        # Add x-tick labels with original ratio strings
        plt.xticks(x, metrics_df['ratio_str'], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(sensitivity_dir, 'sensitivity_metrics.png'), dpi=300)
    plt.close()

    # 4. Create entropy distribution comparison
    plt.figure(figsize=(12, 8))

    # For each ratio, create kernel density estimate of entropy distribution
    for ratio_str, (combined_map, _) in map_data.items():
        # Extract entropies for cells with sufficient samples
        entropies = []
        for cell, info in combined_map.items():
            if info['total'] >= 5:
                p_lk = info['lk_ratio']
                p_lc = info['lc_ratio']

                # Shannon entropy calculation
                entropy = 0
                if p_lk > 0:
                    entropy -= p_lk * np.log2(p_lk)
                if p_lc > 0:
                    entropy -= p_lc * np.log2(p_lc)

                entropies.append(entropy)

        if entropies:
            # Plot KDE
            sns.kdeplot(entropies, label=ratio_str)

    plt.title('Entropy Distribution by LK:LC Ratio')
    plt.xlabel('Entropy Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend(title='LK:LC Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(sensitivity_dir, 'entropy_distribution_comparison.png'), dpi=300)
    plt.close()


# Run sensitivity analysis with fixed LC timesteps and more granular ratios
print(f"\nRunning extended sensitivity analysis with LC timesteps = {FIXED_LC_TIMESTEPS}")

# Store maps and metrics for each ratio
ratio_maps = {}
ratio_metrics = {}

for lk_ratio, lc_ratio in EXTENDED_LK_LC_RATIOS:
    ratio_str = f"{lk_ratio}:{lc_ratio}"
    print(f"\nAnalyzing sensitivity for LK:LC ratio {ratio_str}")

    # Calculate how many trajectories to sample
    # Ensure same total number of trajectories across experiments
    total_trajectories = min(len(lk_trajectories) + len(lc_trajectories), 2000)  # Limit total to 2000 if needed

    ratio_factor = lk_ratio / (lk_ratio + lc_ratio)
    n_lk = min(len(lk_trajectories), int(total_trajectories * ratio_factor))
    n_lc = min(len(lc_trajectories), total_trajectories - n_lk)

    # Adjust if necessary to maintain exact ratio
    if n_lk / n_lc != lk_ratio / lc_ratio:
        n_lc = min(len(lc_trajectories), int(n_lk * lc_ratio / lk_ratio))
        n_lk = min(len(lk_trajectories), int(n_lc * lk_ratio / lc_ratio))

    print(f"  Using {n_lk} LK trajectories and {n_lc} LC trajectories")

    # Randomly sample trajectories
    sampled_lk = random.sample(lk_trajectories, n_lk)
    sampled_lc = random.sample(lc_trajectories, n_lc)

    # Build combined map
    combined_map, class_balance = build_combined_density_map(
        predictions_prob_meta,
        ground_truth_each_timestep,
        sampled_lk,
        sampled_lc,
        resolution=RESOLUTION,
        lc_timesteps=FIXED_LC_TIMESTEPS
    )

    # Store map data
    ratio_maps[ratio_str] = (combined_map, class_balance)

    # Analyze and store metrics
    metrics = analyze_map_for_sensitivity(combined_map, class_balance)
    ratio_metrics[ratio_str] = metrics

    # Print key metrics
    print(f"  Average entropy: {metrics['avg_entropy']:.4f}")
    print(f"  High entropy cells: {metrics['high_entropy_cells']} ({metrics['high_entropy_percentage']:.2%})")
    print(f"  LC ratio std dev: {metrics['lc_ratio_std']:.4f}")

# Create visualizations showing sensitivity
create_sensitivity_visualizations(ratio_maps, ratio_metrics)

# Create summary table
print("\n===== SENSITIVITY ANALYSIS SUMMARY =====")
print(f"{'Metric':<25} | " + " | ".join(f"{ratio:<10}" for ratio in ratio_metrics.keys()))
print("-" * 25 + "-+-" + "-" * 11 * len(ratio_metrics))

# Key metrics to show in summary
summary_metrics = [
    'avg_entropy', 'entropy_std', 'high_entropy_percentage',
    'decision_boundary_percentage', 'lc_ratio_mean', 'lc_ratio_std'
]

metric_formats = {
    'avg_entropy': lambda x: f"{x:.4f}",
    'entropy_std': lambda x: f"{x:.4f}",
    'high_entropy_percentage': lambda x: f"{x:.2%}",
    'decision_boundary_percentage': lambda x: f"{x:.2%}",
    'lc_ratio_mean': lambda x: f"{x:.4f}",
    'lc_ratio_std': lambda x: f"{x:.4f}"
}

for metric in summary_metrics:
    values = [ratio_metrics[ratio][metric] for ratio in ratio_metrics.keys()]
    print(f"{metric:<25} | " + " | ".join(metric_formats[metric](value) for value in values))

print("\nSensitivity analysis complete. Visualizations saved to the 'ratio_sensitivity_analysis' directory.")