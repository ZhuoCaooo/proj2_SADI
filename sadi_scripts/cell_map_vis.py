import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import os

# Path to the data file
folder_path = 'cell_maps_train_data'
file_path = os.path.join(folder_path, 'density_cell_maps_0.0s.pkl')

# Load the data
with open(file_path, 'rb') as f:
    data = pickle.load(f)

lk_distributions = data['lk_distributions']
lc_distributions = data['lc_distributions']


def create_heatmap_data(distributions, size=100):
    """Convert sparse cell distributions to a dense numpy array for heatmap."""
    dominant_states = np.zeros((size, size))
    dominant_probs = np.zeros((size, size))

    for cell, info in distributions.items():
        x, y = cell
        if x < size and y < size:
            probs = [info['LK'], info['LCL'], info['LCR']]
            max_prob = max(probs)
            dominant_state = probs.index(max_prob) +1

            dominant_states[y, x] = dominant_state
            dominant_probs[y, x] = max_prob

    return dominant_states, dominant_probs


# Create a figure with only the dominant states plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Process data
lk_states, lk_probs = create_heatmap_data(lk_distributions)
lc_states, lc_probs = create_heatmap_data(lc_distributions)

# Custom colormap for states
colors = ['white', 'royalblue', 'limegreen', 'crimson']
state_cmap = ListedColormap(colors)

# Plot LK dominant states
im1 = ax1.imshow(lk_states, cmap=state_cmap, vmin=0, vmax=3)
ax1.set_title('LK Trajectories - Dominant States', fontsize=14, pad=20)
ax1.set_xlabel('LCL Probability', fontsize=12)
ax1.set_ylabel('LCR Probability', fontsize=12)

# Create tick labels with proper formatting
tick_positions = np.arange(0, 101, 10)  # 0 to 100 in steps of 10
tick_labels = [f'{(x / 100):.2f}' for x in tick_positions]  # 0.00 to 1.00

ax1.set_xticks(tick_positions)
ax1.set_yticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45)
ax1.set_yticklabels(tick_labels)
ax1.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)

# Add probability value text for LK
for i in range(lk_states.shape[0]):
    for j in range(lk_states.shape[1]):
        if lk_states[i, j] > 0:
            ax1.text(j, i, f'{lk_probs[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if lk_probs[i, j] > 0.5 else 'black',
                     fontsize=6)

# Plot LC dominant states
im2 = ax2.imshow(lc_states, cmap=state_cmap, vmin=0, vmax=3)
ax2.set_title('LC Trajectories - Dominant States', fontsize=14, pad=20)
ax2.set_xlabel('LCL Probability', fontsize=12)
ax2.set_ylabel('LCR Probability', fontsize=12)

# Set tick labels for LC plot
ax2.set_xticks(tick_positions)
ax2.set_yticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45)
ax2.set_yticklabels(tick_labels)
ax2.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)

# Add probability value text for LC
for i in range(lc_states.shape[0]):
    for j in range(lc_states.shape[1]):
        if lc_states[i, j] > 0:
            ax2.text(j, i, f'{lc_probs[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if lc_probs[i, j] > 0.5 else 'black',
                     fontsize=6)

# Add legend for state colors
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors[1:]]
legend_labels = ['LK', 'LCL', 'LCR']
ax1.legend(legend_elements, legend_labels, loc='upper right')
ax2.legend(legend_elements, legend_labels, loc='upper right')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# Uncomment to save the plot
# plt.savefig('dominant_states_plots.png', dpi=300, bbox_inches='tight')


def analyze_cell_maps(lk_distributions, lc_distributions):
    # 1. Statistical Summary
    def get_stats(distributions):
        stats = {
            'avg_sample_count': np.mean([d['sample_count'] for d in distributions.values()]),
            'max_sample_count': max(d['sample_count'] for d in distributions.values()),
            'num_cells': len(distributions),
            'avg_dominant_prob': np.mean([max(d['LK'], d['LCL'], d['LCR']) for d in distributions.values()]),
            'high_confidence_cells': sum(1 for d in distributions.values()
                                         if max(d['LK'], d['LCL'], d['LCR']) > 0.8)
        }
        return stats

    lk_stats = get_stats(lk_distributions)
    lc_stats = get_stats(lc_distributions)

    # Create summary table
    summary_df = pd.DataFrame({
        'Metric': ['Average Samples/Cell', 'Max Samples/Cell', 'Number of Cells',
                   'Avg Dominant Probability', 'High Confidence Cells (>0.8)'],
        'LK Map': [lk_stats['avg_sample_count'], lk_stats['max_sample_count'],
                   lk_stats['num_cells'], lk_stats['avg_dominant_prob'],
                   lk_stats['high_confidence_cells']],
        'LC Map': [lc_stats['avg_sample_count'], lc_stats['max_sample_count'],
                   lc_stats['num_cells'], lc_stats['avg_dominant_prob'],
                   lc_stats['high_confidence_cells']]
    })

    print("\nCell Map Statistics:")
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))


# Call the function with your data
analyze_cell_maps(lk_distributions, lc_distributions)



from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter



def create_base_heatmap(distributions, size=100):
    """Create initial heatmap arrays."""
    dominant_states = np.zeros((size, size))
    sample_counts = np.zeros((size, size))

    for cell, info in distributions.items():
        x, y = cell
        if x < size and y < size:
            probs = [info['LK'], info['LCL'], info['LCR']]
            max_prob = max(probs)
            dominant_state = probs.index(max_prob) + 1


            dominant_states[y, x] = dominant_state
            sample_counts[y, x] = info['sample_count']

    return dominant_states, sample_counts


def smooth_array(arr, window_size=4):
    """Apply smoothing to array using moving average."""
    smoothed = uniform_filter(arr, size=window_size, mode='constant', cval=0.0)
    return smoothed


def create_custom_colormap():
    """Create a custom colormap from blue to red."""
    colors = ['darkblue', 'blue', 'lightblue', 'white', 'pink', 'red', 'darkred']
    return LinearSegmentedColormap.from_list('custom', colors)


# Create figure
fig = plt.figure(figsize=(20, 10))
gs = plt.GridSpec(1, 4, width_ratios=[2, 1, 2, 1])

ax1 = plt.subplot(gs[0])
ax1_text = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax2_text = plt.subplot(gs[3])

# Process data
lk_states, lk_samples = create_base_heatmap(lk_distributions)
lc_states, lc_samples = create_base_heatmap(lc_distributions)

# Apply smoothing with 4x4 window
window_size = 2
lk_smoothed = smooth_array(lk_samples, window_size)
lc_smoothed = smooth_array(lc_samples, window_size)

# Create custom colormap
cmap = create_custom_colormap()

# Plot smoothed heatmaps with smaller colorbars
im1 = ax1.imshow(np.log1p(lk_smoothed), cmap=cmap)
cbar1 = plt.colorbar(im1, ax=ax1, label='Log(Samples+1)',
                     shrink=0.3,  # Make colorbar shorter
                     aspect=20)  # Make colorbar thinner
cbar1.ax.tick_params(labelsize=8)  # Smaller tick labels
#Smoothed Sample Density (2×2 window)
ax1.set_title('LK Trajectories', fontsize=14, pad=20)
ax1.set_xlabel('LCL Probability', fontsize=12)
ax1.set_ylabel('LCR Probability', fontsize=12)

im2 = ax2.imshow(np.log1p(lc_smoothed), cmap=cmap)
cbar2 = plt.colorbar(im1, ax=ax2, label='Log(Samples+1)',
                     shrink=0.3,  # Make colorbar shorter
                     aspect=20)  # Make colorbar thinner
cbar2.ax.tick_params(labelsize=8)  # Smaller tick labels
ax2.set_title('LC Trajectories', fontsize=14, pad=20)
ax2.set_xlabel('LCL Probability', fontsize=12)
ax2.set_ylabel('LCR Probability', fontsize=12)

# Create tick labels
tick_positions = np.arange(0, 101, 10)
tick_labels = [f'{(x / 100):.2f}' for x in tick_positions]

# Set ticks for both plots
for ax in [ax1, ax2]:
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticklabels(tick_labels)
    ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)


def create_info_text(distributions, samples, smoothed):
    """Create information text with distribution statistics."""
    active_cells = np.sum(samples > 0)
    total_samples = int(np.sum(samples))
    all_counts = np.array([info['sample_count'] for info in distributions.values()])
    all_counts = all_counts[all_counts > 0]

    percentiles = np.percentile(all_counts, [25, 50, 75, 90])

    text = f"Map Statistics:\n\n"
    text += f"Total Active Cells: {active_cells}\n"
    text += f"Total Samples: {total_samples:,}\n"
    text += f"\nOriginal Sample Distribution:\n"
    text += f"25th percentile: {percentiles[0]:.0f}\n"
    text += f"Median: {percentiles[1]:.0f}\n"
    text += f"75th percentile: {percentiles[2]:.0f}\n"
    text += f"90th percentile: {percentiles[3]:.0f}\n"
    text += f"Max samples: {np.max(all_counts):.0f}\n\n"

    text += "Smoothed Statistics (4×4):\n"
    smooth_active = np.sum(smoothed > 0)
    text += f"Active regions: {smooth_active}\n"
    text += f"Max density: {np.max(smoothed):.1f}\n"
    text += f"Mean density: {np.mean(smoothed[smoothed > 0]):.1f}\n"

    return text


# Add information to text panels
for ax_text, distributions, samples, smoothed in [
    (ax1_text, lk_distributions, lk_samples, lk_smoothed),
    (ax2_text, lc_distributions, lc_samples, lc_smoothed)
]:
    ax_text.axis('off')
    info_text = create_info_text(distributions, samples, smoothed)
    ax_text.text(0, 1, info_text, fontsize=10, va='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import os
import pickle
from tabulate import tabulate

# Path to the data files
folder_path = 'cell_maps_train_data'
file_suffixes = ['0.0s', '0.5s', '1.0s', '1.5s', '2.0s']


# Function to extract stats from a single file
def get_file_stats(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        lk_distributions = data['lk_distributions']
        lc_distributions = data['lc_distributions']

        # Calculate statistics for each map
        def calculate_stats(distributions):
            if not distributions:
                return {'avg_sample_count': 0, 'max_sample_count': 0, 'num_cells': 0}

            return {
                'avg_sample_count': np.mean([d['sample_count'] for d in distributions.values()]),
                'max_sample_count': max(d['sample_count'] for d in distributions.values()),
                'num_cells': len(distributions)
            }

        lk_stats = calculate_stats(lk_distributions)
        lc_stats = calculate_stats(lc_distributions)

        return {'lk': lk_stats, 'lc': lc_stats}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Collect stats for all files
all_stats = {}
for suffix in file_suffixes:
    file_path = os.path.join(folder_path, f'separate_cell_maps_{suffix}.pkl')
    print(f"Processing: {file_path}")
    stats = get_file_stats(file_path)
    if stats:
        all_stats[suffix] = stats
    else:
        print(f"Failed to process {file_path}")

# Create comprehensive table
metrics = ['Average Samples/Cell', 'Max Samples/Cell', 'Number of Cells']
comprehensive_rows = []

for metric_index, metric in enumerate(metrics):
    row = [metric]

    for suffix in file_suffixes:
        if suffix not in all_stats:
            row.append("N/A")
            row.append("N/A")
            continue

        if metric_index == 0:  # Average Samples/Cell
            row.append(f"{all_stats[suffix]['lk']['avg_sample_count']:.2f}")
            row.append(f"{all_stats[suffix]['lc']['avg_sample_count']:.2f}")
        elif metric_index == 1:  # Max Samples/Cell
            row.append(str(all_stats[suffix]['lk']['max_sample_count']))
            row.append(str(all_stats[suffix]['lc']['max_sample_count']))
        else:  # Number of Cells
            row.append(str(all_stats[suffix]['lk']['num_cells']))
            row.append(str(all_stats[suffix]['lc']['num_cells']))

    comprehensive_rows.append(row)

# Create column headers for comprehensive table
headers = ['Metric']
for suffix in file_suffixes:
    headers.append(f"LK ({suffix})")
    headers.append(f"LC ({suffix})")

comprehensive_df = pd.DataFrame(comprehensive_rows, columns=headers)
print("\nComprehensive Statistics Table:")
print(tabulate(comprehensive_df, headers='keys', tablefmt='grid', showindex=False))

# Save to CSV
csv_path = 'cell_maps_statistics.csv'
comprehensive_df.to_csv(csv_path, index=False)
print(f"\nTable saved to {csv_path}")

# Also create individual tables for easier viewing
print("\nLK Map Statistics Across Time Horizons:")
lk_data = []
for suffix in file_suffixes:
    if suffix in all_stats:
        stats = all_stats[suffix]['lk']
        lk_data.append([
            suffix,
            f"{stats['avg_sample_count']:.2f}",
            f"{stats['max_sample_count']}",
            f"{stats['num_cells']}"
        ])
    else:
        lk_data.append([suffix, "N/A", "N/A", "N/A"])

lk_table = pd.DataFrame(lk_data, columns=['Time', 'Average Samples/Cell', 'Max Samples/Cell', 'Number of Cells'])
print(tabulate(lk_table, headers='keys', tablefmt='grid', showindex=False))

print("\nLC Map Statistics Across Time Horizons:")
lc_data = []
for suffix in file_suffixes:
    if suffix in all_stats:
        stats = all_stats[suffix]['lc']
        lc_data.append([
            suffix,
            f"{stats['avg_sample_count']:.2f}",
            f"{stats['max_sample_count']}",
            f"{stats['num_cells']}"
        ])
    else:
        lc_data.append([suffix, "N/A", "N/A", "N/A"])

lc_table = pd.DataFrame(lc_data, columns=['Time', 'Average Samples/Cell', 'Max Samples/Cell', 'Number of Cells'])
print(tabulate(lc_table, headers='keys', tablefmt='grid', showindex=False))

# Add this code to your existing visualization script after loading the data
# (after the line with "lc_distributions = data['lc_distributions']")

print("\n--- LC Map Diagnosis ---")

# 1. Check dominant states in the LC map
lk_dominated = 0
lcl_dominated = 0
lcr_dominated = 0
total_cells = len(lc_distributions)

# Store cells with high LK probability for examination
high_lk_cells = []

# Examine each cell in the LC map
for cell, info in lc_distributions.items():
    # Get the probabilities for each class
    lk_prob = info['LK']
    lcl_prob = info['LCL']
    lcr_prob = info['LCR']

    # Determine the dominant class
    if lk_prob > lcl_prob and lk_prob > lcr_prob:
        lk_dominated += 1
        if lk_prob > 0.6:  # High confidence LK cells
            high_lk_cells.append((cell, info))
    elif lcl_prob > lk_prob and lcl_prob > lcr_prob:
        lcl_dominated += 1
    else:
        lcr_dominated += 1

# Print summary statistics
print(f"Total cells in LC map: {total_cells}")
print(f"LK-dominated cells: {lk_dominated} ({lk_dominated / total_cells:.1%})")
print(f"LCL-dominated cells: {lcl_dominated} ({lcl_dominated / total_cells:.1%})")
print(f"LCR-dominated cells: {lcr_dominated} ({lcr_dominated / total_cells:.1%})")

# 2. Examine a few LK-dominated cells with high LK probability
print("\nExamining LK-dominated cells:")
high_lk_cells.sort(key=lambda x: x[1]['LK'], reverse=True)  # Sort by LK probability

for i, (cell, info) in enumerate(high_lk_cells[:5]):  # Look at top 5 cells
    n1, n2 = cell  # Cell coordinates
    resolution = 0.02  # Assuming resolution is 0.02 as in your code

    # Calculate the probability ranges for this cell
    lcl_range = (n1 * resolution, (n1 + 1) * resolution)
    lcr_range = (n2 * resolution, (n2 + 1) * resolution)

    print(f"\nCell {i + 1}: ({n1}, {n2})")
    print(f"  Probability ranges: LCL={lcl_range[0]:.2f}-{lcl_range[1]:.2f}, LCR={lcr_range[0]:.2f}-{lcr_range[1]:.2f}")
    print(f"  Class probabilities: LK={info['LK']:.3f}, LCL={info['LCL']:.3f}, LCR={info['LCR']:.3f}")
    print(f"  Sample count: {info['sample_count']}")


# 3. Fix the visualization issue (if it's a visualization problem)

# If you suspect the issue is with the +1 in the dominant state calculation, try this:
# Modify the create_heatmap_data function to properly map class indices to visualization values
def create_fixed_heatmap_data(distributions, size=100):
    """Convert sparse cell distributions to a dense numpy array for heatmap."""
    dominant_states = np.zeros((size, size))
    dominant_probs = np.zeros((size, size))

    for cell, info in distributions.items():
        x, y = cell
        if x < size and y < size:
            probs = [info['LK'], info['LCL'], info['LCR']]
            max_prob = max(probs)
            max_index = probs.index(max_prob)

            # Map the class indices directly to visualization values:
            # 0->1 (LK), 1->2 (LCL), 2->3 (LCR)
            # This ensures proper color mapping in the visualization
            vis_value = max_index + 1  # +1 because visualization uses 1-based indexing

            dominant_states[y, x] = vis_value
            dominant_probs[y, x] = max_prob

    return dominant_states, dominant_probs


# Try creating a new visualization with the corrected mapping
print("\nCreating a new visualization with corrected class mapping...")

# # Uncomment this to create and display a fixed visualization
# lc_states_fixed, lc_probs_fixed = create_fixed_heatmap_data(lc_distributions)
#
# plt.figure(figsize=(8, 6))
# colors = ['white', 'royalblue', 'limegreen', 'crimson']
# state_cmap = ListedColormap(colors)
#
# plt.imshow(lc_states_fixed, cmap=state_cmap, vmin=0, vmax=3)
# plt.title('LC Trajectories - Dominant States (Fixed)', fontsize=14)
# plt.xlabel('LCL Probability', fontsize=12)
# plt.ylabel('LCR Probability', fontsize=12)
#
# # Create tick labels
# tick_positions = np.arange(0, 101, 10)
# tick_labels = [f'{(x / 100):.2f}' for x in tick_positions]
# plt.xticks(tick_positions, tick_labels, rotation=45)
# plt.yticks(tick_positions, tick_labels)
#
# # Add legend
# legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors[1:]]
# legend_labels = ['LK', 'LCL', 'LCR']
# plt.legend(legend_elements, legend_labels, loc='upper right')
#
# plt.grid(True, color='gray', linestyle='-', alpha=0.2)
# plt.tight_layout()
# plt.show()