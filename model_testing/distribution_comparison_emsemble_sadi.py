"""
Combined Analysis of SADI and Ensemble CNN Results for Lane Change Prediction
This script compares reliability score distributions between SADI and Ensemble methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import seaborn as sns
import pandas as pd
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

# File paths for both models
SADI_FILE_PATH = '../model_testing_paper_2/cnn_results_density_based_0.0s_new.pkl'
ENSEMBLE_FILE_PATH = 'ensemble_sequential_test_results.pkl'  # Assuming parallel folder structure


def load_sadi_data(file_path):
    """Load SADI pickle data containing model predictions and reliability scores"""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"SADI data loaded successfully from {file_path}")
    print("Available keys:", loaded_data.keys())

    return {
        'model_predictions': loaded_data['model_predictions'],
        'reliability_scores': loaded_data['reliability_scores'],
        'ground_truth': loaded_data['ground_truth'],
        'probability_outputs': loaded_data['probability_outputs'],
        'ratio_scores': loaded_data.get('ratio_scores', []),
        'absolute_scores': loaded_data.get('absolute_scores', [])
    }


def load_ensemble_data(file_path):
    """Load ensemble pickle data containing model predictions and reliability scores"""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"Ensemble data loaded successfully from {file_path}")
    print("Available keys:", loaded_data.keys())

    return loaded_data


def get_final_behavior(predictions):
    """Get the final behavior (last non-zero prediction)"""
    final_pred = 0
    for pred in reversed(predictions):
        if pred != 0:
            final_pred = pred
            break
    return final_pred


def prepare_combined_dataset(data, model_name):
    """
    Create a combined dataset for all predictions across all trajectories
    with corresponding ground truth, reliability scores, and error indicators
    using the final behavior of each trajectory as the target
    """
    # Initialize lists to store combined data
    all_preds = []
    all_ground_truth = []
    all_reliability = []
    all_trajectory_ids = []
    all_timesteps = []
    all_final_behaviors = []
    all_pred_categories = []

    for traj_idx, (preds, gt, reliability) in enumerate(zip(
            data['model_predictions'],
            data['ground_truth'],
            data['reliability_scores'])):

        # Get the final behavior (last non-zero prediction)
        final_behavior = get_final_behavior(gt)
        trajectory_length = len(preds)

        for t in range(len(preds)):
            # Skip if any data is missing
            if t >= len(gt) or t >= len(reliability):
                continue

            pred_value = preds[t]
            gt_value = gt[t]
            rel_value = reliability[t]

            # Determine the prediction category based on definitions
            if pred_value != 0 and pred_value == final_behavior:
                # True Positive - prediction matches final behavior
                pred_category = "TP"
            elif pred_value != 0 and pred_value != final_behavior:
                # False Positive - predicted lane change differs from final behavior
                pred_category = "FP"
            elif pred_value == 0 and final_behavior != 0 and t > 120:
                # False Negative - still predicting LK after timestep 120 when final behavior is LC
                pred_category = "FN"
            elif pred_value == 0 and final_behavior == 0:
                # True Negative - correctly predicting LK when final behavior is also LK
                pred_category = "TN"
            else:
                # Other cases (early LK predictions or non-LC trajectories)
                pred_category = "Other"

            # Store all values
            all_preds.append(pred_value)
            all_ground_truth.append(gt_value)
            all_reliability.append(rel_value)
            all_trajectory_ids.append(traj_idx)
            all_timesteps.append(t)
            all_final_behaviors.append(final_behavior)
            all_pred_categories.append(pred_category)

    # Convert to numpy arrays for easier manipulation
    return {
        'predictions': np.array(all_preds),
        'ground_truth': np.array(all_ground_truth),
        'reliability': np.array(all_reliability),
        'trajectory_ids': np.array(all_trajectory_ids),
        'timesteps': np.array(all_timesteps),
        'final_behaviors': np.array(all_final_behaviors),
        'pred_categories': np.array(all_pred_categories),
        'model_name': model_name
    }


def plot_combined_reliability_distributions(sadi_data, ensemble_data, output_prefix='combined_comparison'):
    """
    Create a publication-quality plot showing reliability score distributions
    for different prediction categories comparing SADI and Ensemble methods
    """
    # Extract data for both methods
    sadi_reliability = sadi_data['reliability']
    sadi_pred_categories = sadi_data['pred_categories']

    ensemble_reliability = ensemble_data['reliability']
    ensemble_pred_categories = ensemble_data['pred_categories']

    # Set up figure - 2x2 subplots for TP, FP, FN, TN
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    # Define colors for each method
    sadi_color = '#2A9D8F'  # Teal for SADI
    ensemble_color = '#E63946'  # Red for Ensemble

    # Categories to plot
    categories = ['TP', 'FP', 'FN', 'TN']
    category_titles = {
        'TP': 'True Positive (TP) Reliability Scores',
        'FP': 'False Positive (FP) Reliability Scores',
        'FN': 'False Negative (FN) Reliability Scores',
        'TN': 'True Negative (TN) Reliability Scores'
    }

    # Plot each prediction category
    for i, category in enumerate(categories):
        ax = axes[i]

        # Get SADI data for this category
        sadi_mask = sadi_pred_categories == category
        sadi_values = sadi_reliability[sadi_mask]

        # Get Ensemble data for this category
        ensemble_mask = ensemble_pred_categories == category
        ensemble_values = ensemble_reliability[ensemble_mask]

        # Only plot if we have enough samples for both methods
        if len(sadi_values) > 10 and len(ensemble_values) > 10:
            # Create KDE plots for both methods
            sns.kdeplot(
                sadi_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=sadi_color,
                label=f'SADI (n={len(sadi_values)})',
                cut=0
            )

            sns.kdeplot(
                ensemble_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=ensemble_color,
                label=f'Ensemble (n={len(ensemble_values)})',
                cut=0
            )

            # Add mean and median lines for SADI
            sadi_mean = np.mean(sadi_values)
            sadi_median = np.median(sadi_values)
            ax.axvline(sadi_mean, color=sadi_color, linestyle='--', alpha=0.8,
                       label=f'SADI Mean: {sadi_mean:.2f}')
            ax.axvline(sadi_median, color=sadi_color, linestyle=':', alpha=0.8,
                       label=f'SADI Median: {sadi_median:.2f}')

            # Add mean and median lines for Ensemble
            ensemble_mean = np.mean(ensemble_values)
            ensemble_median = np.median(ensemble_values)
            ax.axvline(ensemble_mean, color=ensemble_color, linestyle='--', alpha=0.8,
                       label=f'Ensemble Mean: {ensemble_mean:.2f}')
            ax.axvline(ensemble_median, color=ensemble_color, linestyle=':', alpha=0.8,
                       label=f'Ensemble Median: {ensemble_median:.2f}')

        elif len(sadi_values) > 10:
            # Only SADI has enough samples
            sns.kdeplot(
                sadi_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=sadi_color,
                label=f'SADI (n={len(sadi_values)})',
                cut=0
            )
            ax.text(0.5, 0.3, f"Ensemble: Insufficient samples\n(n={len(ensemble_values)})",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color=ensemble_color)

        elif len(ensemble_values) > 10:
            # Only Ensemble has enough samples
            sns.kdeplot(
                ensemble_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=ensemble_color,
                label=f'Ensemble (n={len(ensemble_values)})',
                cut=0
            )
            ax.text(0.5, 0.3, f"SADI: Insufficient samples\n(n={len(sadi_values)})",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color=sadi_color)

        else:
            # Both have insufficient samples
            ax.text(0.5, 0.5, f"Insufficient samples\nSADI: n={len(sadi_values)}\nEnsemble: n={len(ensemble_values)}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)

        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_title(category_titles[category])
        ax.set_xlabel('Reliability Score')
        if i % 2 == 0:  # Left column
            ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    # Add a main title
    plt.suptitle('Reliability Score Distributions: SADI vs Ensemble Comparison', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust for the suptitle

    # Save the figure
    filename = f'{output_prefix}_reliability_distributions_comparison.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined reliability distributions figure to {filename}")

    return fig


def print_comparison_statistics(sadi_data, ensemble_data):
    """
    Print comparison statistics between SADI and Ensemble methods
    """
    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS: SADI vs ENSEMBLE")
    print("=" * 80)

    categories = ['TP', 'FP', 'FN', 'TN']

    for category in categories:
        print(f"\n{category} Category Analysis:")
        print("-" * 40)

        # SADI statistics
        sadi_mask = sadi_data['pred_categories'] == category
        sadi_values = sadi_data['reliability'][sadi_mask]

        # Ensemble statistics
        ensemble_mask = ensemble_data['pred_categories'] == category
        ensemble_values = ensemble_data['reliability'][ensemble_mask]

        if len(sadi_values) > 0:
            print(f"SADI     - Count: {len(sadi_values):6d}, Mean: {np.mean(sadi_values):.3f}, "
                  f"Median: {np.median(sadi_values):.3f}, Std: {np.std(sadi_values):.3f}")
        else:
            print(f"SADI     - Count: {len(sadi_values):6d}, No data available")

        if len(ensemble_values) > 0:
            print(f"Ensemble - Count: {len(ensemble_values):6d}, Mean: {np.mean(ensemble_values):.3f}, "
                  f"Median: {np.median(ensemble_values):.3f}, Std: {np.std(ensemble_values):.3f}")
        else:
            print(f"Ensemble - Count: {len(ensemble_values):6d}, No data available")

        # Calculate difference if both have data
        if len(sadi_values) > 0 and len(ensemble_values) > 0:
            mean_diff = np.mean(sadi_values) - np.mean(ensemble_values)
            median_diff = np.median(sadi_values) - np.median(ensemble_values)
            print(f"Difference (SADI - Ensemble): Mean = {mean_diff:+.3f}, Median = {median_diff:+.3f}")


def main():
    """Main comparison function"""
    print("Starting combined SADI vs Ensemble reliability analysis...")

    # Check if files exist
    if not os.path.exists(SADI_FILE_PATH):
        print(f"SADI file not found: {SADI_FILE_PATH}")
        return

    if not os.path.exists(ENSEMBLE_FILE_PATH):
        print(f"Ensemble file not found: {ENSEMBLE_FILE_PATH}")
        return

    try:
        # Load both datasets
        print("\nLoading SADI data...")
        sadi_raw_data = load_sadi_data(SADI_FILE_PATH)

        print("\nLoading Ensemble data...")
        ensemble_raw_data = load_ensemble_data(ENSEMBLE_FILE_PATH)

        # Prepare combined datasets for both methods
        print("\nPreparing SADI dataset...")
        sadi_combined_data = prepare_combined_dataset(sadi_raw_data, "SADI")

        print("\nPreparing Ensemble dataset...")
        ensemble_combined_data = prepare_combined_dataset(ensemble_raw_data, "Ensemble")

        # Print category breakdowns for both methods
        print("\nSADI Prediction Category Breakdown:")
        print("-" * 50)
        sadi_categories, sadi_counts = np.unique(sadi_combined_data['pred_categories'], return_counts=True)
        sadi_percentages = sadi_counts / len(sadi_combined_data['pred_categories']) * 100
        for cat, count, percentage in zip(sadi_categories, sadi_counts, sadi_percentages):
            print(f"{cat}: {count} ({percentage:.2f}%)")

        print("\nEnsemble Prediction Category Breakdown:")
        print("-" * 50)
        ensemble_categories, ensemble_counts = np.unique(ensemble_combined_data['pred_categories'], return_counts=True)
        ensemble_percentages = ensemble_counts / len(ensemble_combined_data['pred_categories']) * 100
        for cat, count, percentage in zip(ensemble_categories, ensemble_counts, ensemble_percentages):
            print(f"{cat}: {count} ({percentage:.2f}%)")

        # Generate combined comparison plots
        print("\nGenerating combined comparison plots...")
        plot_combined_reliability_distributions(sadi_combined_data, ensemble_combined_data)

        # Print detailed comparison statistics
        print_comparison_statistics(sadi_combined_data, ensemble_combined_data)

        print("\nCombined analysis complete. Results saved to files.")

        return sadi_combined_data, ensemble_combined_data

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()