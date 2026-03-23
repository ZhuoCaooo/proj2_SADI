"""
Section 4.4.2 - Reliability Score Analysis for 1.0s, 1.5s, and 2.0s Prediction Windows
Visualizes ROC curves for selected prediction windows in academic journal style
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
import matplotlib
from matplotlib.ticker import MultipleLocator
import os

# Set up matplotlib for publication quality
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (8, 5)
})

# Selected prediction windows to analyze
prediction_windows = [1.0, 1.5, 2.0]

# Define colors for different reliability thresholds
threshold_colors = ['#56B4E9', '#E69F00', '#009E73', '#F0E442', '#0072B2']

# Define reliability thresholds
reliability_thresholds = [0.2, 0.4, 0.6, 0.8]

# Dictionary to store AUC values for the summary table
auc_values = {
    'LCL': {},
    'LCR': {}
}

# Function to load data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to prepare data for ROC curves
def prepare_roc_data(loaded_data):
    model_pred_list_all = loaded_data['model_predictions']
    reliability_scores_all = loaded_data['reliability_scores']
    ground_truth_each_timestep = loaded_data['ground_truth']
    probability_outputs = loaded_data['probability_outputs']

    # Prepare data for ROC curves
    y_true_onehot = []  # One-hot encoded ground truth (only classes 1 and 2)
    pred_probs = []  # Prediction probabilities for each class
    reliabilities = []  # Store reliability scores

    for traj_idx, (probs, ground_truths, reliability) in enumerate(
            zip(probability_outputs, ground_truth_each_timestep, reliability_scores_all)):

        final_label = ground_truths[-1]

        # Skip trajectories with final label 0
        if final_label == 0:
            continue

        # For each timestep in the trajectory
        for timestep in range(len(probs)):
            # Skip the first 24 timesteps as they don't have probability predictions
            if timestep < 24:
                continue

            # Get the ground truth at this timestep
            true_label = ground_truths[timestep]

            # Create one-hot encoding for true label (only for classes 1 and 2)
            true_onehot = [0, 0]
            if true_label > 0:  # if label is 1 or 2
                true_onehot[true_label - 1] = 1

            # Get probabilities for classes 1 and 2
            prob_timestep = probs[timestep - 24]  # adjust index since probs start at timestep 24
            class_probs = [prob_timestep[1], prob_timestep[2]]  # get probs for classes 1 and 2

            y_true_onehot.append(true_onehot)
            pred_probs.append(class_probs)
            reliabilities.append(reliability[timestep])

    # Convert to numpy arrays
    return np.array(y_true_onehot), np.array(pred_probs), np.array(reliabilities)

# Create plots for each of the selected prediction windows
for window in prediction_windows:
    # Construct file path
    file_path = f'../model_testing_paper_2/cnn_results_density_based_{window}s.pkl'

    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found, skipping...")
        continue

    # Load data
    loaded_data = load_data(file_path)

    # Prepare ROC data
    y_true_onehot, pred_probs, reliabilities = prepare_roc_data(loaded_data)

    # Ensure we have data
    if len(y_true_onehot) == 0:
        print(f"Warning: No data processed for {window}s window")
        continue

    # Create figure for normal ROC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Process LCL (class 0)
    fpr, tpr, _ = roc_curve(y_true_onehot[:, 0], pred_probs[:, 0])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='black', linestyle='-',
             label=f'All samples (AUC = {roc_auc:.3f})')

    # Process LCR (class 1)
    fpr, tpr, _ = roc_curve(y_true_onehot[:, 1], pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='black', linestyle='-',
             label=f'All samples (AUC = {roc_auc:.3f})')

    # Store AUC values for the summary table
    if f'{window}s' not in auc_values['LCL']:
        auc_values['LCL'][f'{window}s'] = {}
    if f'{window}s' not in auc_values['LCR']:
        auc_values['LCR'][f'{window}s'] = {}

    auc_values['LCL'][f'{window}s']['All'] = roc_auc
    auc_values['LCR'][f'{window}s']['All'] = roc_auc

    # Add reliability thresholds
    for j, threshold in enumerate(reliability_thresholds):
        mask = reliabilities >= threshold
        if np.sum(mask) > 0:
            # Calculate what percentage of data remains
            data_percentage = np.sum(mask) / len(reliabilities) * 100

            # Process LCL with threshold
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, 0], pred_probs[mask, 0])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            ax1.plot(fpr_thresh, tpr_thresh, color=threshold_colors[j], linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f}, {data_percentage:.1f}%)')
            auc_values['LCL'][f'{window}s'][f'Rel≥{threshold}'] = roc_auc_thresh

            # Process LCR with threshold
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, 1], pred_probs[mask, 1])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            ax2.plot(fpr_thresh, tpr_thresh, color=threshold_colors[j], linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f}, {data_percentage:.1f}%)')
            auc_values['LCR'][f'{window}s'][f'Rel≥{threshold}'] = roc_auc_thresh

    # Format the LCL subplot
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curves for LCL - {window}s Prediction Horizon')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Format the LCR subplot
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curves for LCR - {window}s Prediction Horizon')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Save the normal figure
    plt.tight_layout()
    plt.savefig(f'roc_curves_{window}s_window.pdf')
    plt.savefig(f'roc_curves_{window}s_window.png')

    # Create figure for zoomed ROC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Process LCL (class 0) - Zoomed
    fpr, tpr, _ = roc_curve(y_true_onehot[:, 0], pred_probs[:, 0])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='black', linestyle='-',
             label=f'All samples (AUC = {roc_auc:.3f})')

    # Process LCR (class 1) - Zoomed
    fpr, tpr, _ = roc_curve(y_true_onehot[:, 1], pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='black', linestyle='-',
             label=f'All samples (AUC = {roc_auc:.3f})')

    # Add reliability thresholds for zoomed view
    for j, threshold in enumerate(reliability_thresholds):
        mask = reliabilities >= threshold
        if np.sum(mask) > 0:
            # Calculate what percentage of data remains
            data_percentage = np.sum(mask) / len(reliabilities) * 100

            # Process LCL with threshold
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, 0], pred_probs[mask, 0])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            ax1.plot(fpr_thresh, tpr_thresh, color=threshold_colors[j], linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f}, {data_percentage:.1f}%)')

            # Process LCR with threshold
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, 1], pred_probs[mask, 1])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            ax2.plot(fpr_thresh, tpr_thresh, color=threshold_colors[j], linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f}, {data_percentage:.1f}%)')

    # Format the LCL subplot - Zoomed
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curves for LCL - {window}s Prediction Window (Zoomed)')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 0.3])  # Zoomed in x-axis
    ax1.set_ylim([0.7, 1.02])  # Zoomed in y-axis
    ax1.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.02))

    # Format the LCR subplot - Zoomed
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curves for LCR - {window}s Prediction Window (Zoomed)')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 0.3])  # Zoomed in x-axis
    ax2.set_ylim([0.7, 1.02])  # Zoomed in y-axis
    ax2.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.02))

    # Save the zoomed figure
    plt.tight_layout()
    plt.savefig(f'roc_curves_{window}s_window_zoomed.pdf')
    plt.savefig(f'roc_curves_{window}s_window_zoomed.png')

# Create a summary table of AUC values
print("\nSummary of AUC Values:")
print("\nLane Change Left (LCL):")
header = "Threshold"
for window in prediction_windows:
    header += f" | {window}s"
print(header)
print("-" * len(header))

# All samples row
row = "All samples"
for window in prediction_windows:
    key = f"{window}s"
    if key in auc_values['LCL'] and 'All' in auc_values['LCL'][key]:
        row += f" | {auc_values['LCL'][key]['All']:.3f}"
    else:
        row += " | -"
print(row)

# Threshold rows
for threshold in reliability_thresholds:
    row = f"Rel ≥ {threshold:.1f}"
    for window in prediction_windows:
        key = f"{window}s"
        thresh_key = f"Rel≥{threshold}"
        if key in auc_values['LCL'] and thresh_key in auc_values['LCL'][key]:
            row += f" | {auc_values['LCL'][key][thresh_key]:.3f}"
        else:
            row += " | -"
    print(row)

print("\nLane Change Right (LCR):")
header = "Threshold"
for window in prediction_windows:
    header += f" | {window}s"
print(header)
print("-" * len(header))

# All samples row
row = "All samples"
for window in prediction_windows:
    key = f"{window}s"
    if key in auc_values['LCR'] and 'All' in auc_values['LCR'][key]:
        row += f" | {auc_values['LCR'][key]['All']:.3f}"
    else:
        row += " | -"
print(row)

# Threshold rows
for threshold in reliability_thresholds:
    row = f"Rel ≥ {threshold:.1f}"
    for window in prediction_windows:
        key = f"{window}s"
        thresh_key = f"Rel≥{threshold}"
        if key in auc_values['LCR'] and thresh_key in auc_values['LCR'][key]:
            row += f" | {auc_values['LCR'][key][thresh_key]:.3f}"
        else:
            row += " | -"
    print(row)

# Generate LaTeX table code for the paper
print("\nLaTeX Table for Paper:")
print(r"""\begin{table}[!t]
\centering
\caption{AUC values for different prediction windows and reliability thresholds}
\label{tab:auc_values}
\begin{tabular}{l|ccc|ccc}
\hline
\multirow{2}{*}{Threshold} & \multicolumn{3}{c|}{Lane Change Left (LCL)} & \multicolumn{3}{c}{Lane Change Right (LCR)} \\
 & 1.0s & 1.5s & 2.0s & 1.0s & 1.5s & 2.0s \\
\hline""")

# All samples row
lcl_values = []
lcr_values = []
for window in prediction_windows:
    key = f"{window}s"
    if key in auc_values['LCL'] and 'All' in auc_values['LCL'][key]:
        lcl_values.append(f"{auc_values['LCL'][key]['All']:.3f}")
    else:
        lcl_values.append("-")

    if key in auc_values['LCR'] and 'All' in auc_values['LCR'][key]:
        lcr_values.append(f"{auc_values['LCR'][key]['All']:.3f}")
    else:
        lcr_values.append("-")

print(f"All samples & {' & '.join(lcl_values)} & {' & '.join(lcr_values)} \\\\")

# Reliability threshold rows
for threshold in reliability_thresholds:
    lcl_values = []
    lcr_values = []
    for window in prediction_windows:
        key = f"{window}s"
        thresh_key = f"Rel≥{threshold}"
        if key in auc_values['LCL'] and thresh_key in auc_values['LCL'][key]:
            lcl_values.append(f"{auc_values['LCL'][key][thresh_key]:.3f}")
        else:
            lcl_values.append("-")

        if key in auc_values['LCR'] and thresh_key in auc_values['LCR'][key]:
            lcr_values.append(f"{auc_values['LCR'][key][thresh_key]:.3f}")
        else:
            lcr_values.append("-")

    print(f"Rel $\\geq$ {threshold:.1f} & {' & '.join(lcl_values)} & {' & '.join(lcr_values)} \\\\")

print(r"""\hline
\end{tabular}
\end{table}""")