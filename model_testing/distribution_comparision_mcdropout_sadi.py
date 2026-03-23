"""
Combined Analysis of SADI and Monte Carlo Dropout Results for Lane Change Prediction
This script compares reliability score distributions between SADI and MC Dropout methods
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
MC_DROPOUT_FILE_PATH = 'real_time_mc_dropout_results.pkl'  # Monte Carlo dropout results


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


def load_mc_dropout_data(file_path):
    """Load Monte Carlo dropout pickle data containing model predictions and reliability scores"""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"Monte Carlo dropout data loaded successfully from {file_path}")
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


def plot_combined_reliability_distributions(sadi_data, mc_dropout_data, output_prefix='sadi_mc_comparison'):
    """
    Create a publication-quality plot showing reliability score distributions
    for different prediction categories comparing SADI and Monte Carlo Dropout methods
    """
    # Extract data for both methods
    sadi_reliability = sadi_data['reliability']
    sadi_pred_categories = sadi_data['pred_categories']

    mc_reliability = mc_dropout_data['reliability']
    mc_pred_categories = mc_dropout_data['pred_categories']

    # Set up figure - 2x2 subplots for TP, FP, FN, TN
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    # Define colors for each method
    sadi_color = '#2A9D8F'  # Teal for SADI
    mc_color = '#F77F00'  # Orange for Monte Carlo Dropout

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

        # Get MC Dropout data for this category
        mc_mask = mc_pred_categories == category
        mc_values = mc_reliability[mc_mask]

        # Only plot if we have enough samples for both methods
        if len(sadi_values) > 10 and len(mc_values) > 10:
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
                mc_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=mc_color,
                label=f'MC Dropout (n={len(mc_values)})',
                cut=0
            )

            # Add mean and median lines for SADI
            sadi_mean = np.mean(sadi_values)
            sadi_median = np.median(sadi_values)
            ax.axvline(sadi_mean, color=sadi_color, linestyle='--', alpha=0.8,
                       label=f'SADI Mean: {sadi_mean:.2f}')
            ax.axvline(sadi_median, color=sadi_color, linestyle=':', alpha=0.8,
                       label=f'SADI Median: {sadi_median:.2f}')

            # Add mean and median lines for MC Dropout
            mc_mean = np.mean(mc_values)
            mc_median = np.median(mc_values)
            ax.axvline(mc_mean, color=mc_color, linestyle='--', alpha=0.8,
                       label=f'MC Mean: {mc_mean:.2f}')
            ax.axvline(mc_median, color=mc_color, linestyle=':', alpha=0.8,
                       label=f'MC Median: {mc_median:.2f}')

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
            ax.text(0.5, 0.3, f"MC Dropout: Insufficient samples\n(n={len(mc_values)})",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color=mc_color)

        elif len(mc_values) > 10:
            # Only MC Dropout has enough samples
            sns.kdeplot(
                mc_values,
                ax=ax,
                fill=True,
                alpha=0.6,
                color=mc_color,
                label=f'MC Dropout (n={len(mc_values)})',
                cut=0
            )
            ax.text(0.5, 0.3, f"SADI: Insufficient samples\n(n={len(sadi_values)})",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color=sadi_color)

        else:
            # Both have insufficient samples
            ax.text(0.5, 0.5, f"Insufficient samples\nSADI: n={len(sadi_values)}\nMC Dropout: n={len(mc_values)}",
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
    plt.suptitle('Reliability Score Distributions: SADI vs Monte Carlo Dropout Comparison', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust for the suptitle

    # Save the figure
    filename = f'{output_prefix}_reliability_distributions_comparison.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined reliability distributions figure to {filename}")

    return fig


def plot_retention_curves_comparison(sadi_data, mc_dropout_data, output_prefix='sadi_mc_comparison'):
    """
    Create retention curves comparison between SADI and MC Dropout methods
    """
    # Define thresholds
    thresholds = np.linspace(0, 0.95, 20)

    # Calculate retention rates for both methods
    categories = ['TP', 'FP', 'FN', 'TN']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    # Define colors
    sadi_color = '#2A9D8F'  # Teal for SADI
    mc_color = '#F77F00'  # Orange for Monte Carlo Dropout

    category_titles = {
        'TP': 'True Positive (TP) Retention',
        'FP': 'False Positive (FP) Retention',
        'FN': 'False Negative (FN) Retention',
        'TN': 'True Negative (TN) Retention'
    }

    for i, category in enumerate(categories):
        ax = axes[i]

        # Calculate retention curves for SADI
        sadi_mask = sadi_data['pred_categories'] == category
        sadi_reliability = sadi_data['reliability'][sadi_mask]

        # Calculate retention curves for MC Dropout
        mc_mask = mc_dropout_data['pred_categories'] == category
        mc_reliability = mc_dropout_data['reliability'][mc_mask]

        if len(sadi_reliability) > 0:
            sadi_retention = [np.mean(sadi_reliability >= t) for t in thresholds]
            ax.plot(thresholds, sadi_retention, 'o-',
                    color=sadi_color, linewidth=2, label=f'SADI (n={len(sadi_reliability)})')

        if len(mc_reliability) > 0:
            mc_retention = [np.mean(mc_reliability >= t) for t in thresholds]
            ax.plot(thresholds, mc_retention, 's-',
                    color=mc_color, linewidth=2, label=f'MC Dropout (n={len(mc_reliability)})')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Reliability Threshold')
        ax.set_ylabel('Retention Rate')
        ax.set_title(category_titles[category])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Retention Rate Comparison: SADI vs Monte Carlo Dropout', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the figure
    filename = f'{output_prefix}_retention_curves_comparison.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved retention curves comparison figure to {filename}")

    return fig


def print_comparison_statistics(sadi_data, mc_dropout_data):
    """
    Print comparison statistics between SADI and Monte Carlo Dropout methods
    """
    print("\n" + "=" * 80)
    print("COMPARISON STATISTICS: SADI vs MONTE CARLO DROPOUT")
    print("=" * 80)

    categories = ['TP', 'FP', 'FN', 'TN']

    for category in categories:
        print(f"\n{category} Category Analysis:")
        print("-" * 40)

        # SADI statistics
        sadi_mask = sadi_data['pred_categories'] == category
        sadi_values = sadi_data['reliability'][sadi_mask]

        # MC Dropout statistics
        mc_mask = mc_dropout_data['pred_categories'] == category
        mc_values = mc_dropout_data['reliability'][mc_mask]

        if len(sadi_values) > 0:
            print(f"SADI       - Count: {len(sadi_values):6d}, Mean: {np.mean(sadi_values):.3f}, "
                  f"Median: {np.median(sadi_values):.3f}, Std: {np.std(sadi_values):.3f}")
        else:
            print(f"SADI       - Count: {len(sadi_values):6d}, No data available")

        if len(mc_values) > 0:
            print(f"MC Dropout - Count: {len(mc_values):6d}, Mean: {np.mean(mc_values):.3f}, "
                  f"Median: {np.median(mc_values):.3f}, Std: {np.std(mc_values):.3f}")
        else:
            print(f"MC Dropout - Count: {len(mc_values):6d}, No data available")

        # Calculate difference if both have data
        if len(sadi_values) > 0 and len(mc_values) > 0:
            mean_diff = np.mean(sadi_values) - np.mean(mc_values)
            median_diff = np.median(sadi_values) - np.median(mc_values)
            print(f"Difference (SADI - MC Dropout): Mean = {mean_diff:+.3f}, Median = {median_diff:+.3f}")


def create_threshold_comparison_table(sadi_data, mc_dropout_data, thresholds=[0.0, 0.4, 0.6, 0.8]):
    """
    Create a comparison table showing retention rates at different thresholds
    """
    results = []

    for threshold in thresholds:
        row = {'Threshold': threshold}

        for category in ['TP', 'FP', 'FN', 'TN']:
            # SADI retention
            sadi_mask = sadi_data['pred_categories'] == category
            sadi_reliability = sadi_data['reliability'][sadi_mask]
            if len(sadi_reliability) > 0:
                sadi_retention = np.mean(sadi_reliability >= threshold)
                row[f'SADI_{category}'] = sadi_retention
            else:
                row[f'SADI_{category}'] = np.nan

            # MC Dropout retention
            mc_mask = mc_dropout_data['pred_categories'] == category
            mc_reliability = mc_dropout_data['reliability'][mc_mask]
            if len(mc_reliability) > 0:
                mc_retention = np.mean(mc_reliability >= threshold)
                row[f'MC_{category}'] = mc_retention
            else:
                row[f'MC_{category}'] = np.nan

        results.append(row)

    df = pd.DataFrame(results)

    # Save to CSV
    filename = 'sadi_mc_dropout_threshold_comparison.csv'
    df.to_csv(filename, index=False)
    print(f"Saved threshold comparison table to {filename}")

    return df


def main():
    """Main comparison function"""
    print("Starting combined SADI vs Monte Carlo Dropout reliability analysis...")

    # Check if files exist
    if not os.path.exists(SADI_FILE_PATH):
        print(f"SADI file not found: {SADI_FILE_PATH}")
        return

    if not os.path.exists(MC_DROPOUT_FILE_PATH):
        print(f"Monte Carlo dropout file not found: {MC_DROPOUT_FILE_PATH}")
        return

    try:
        # Load both datasets
        print("\nLoading SADI data...")
        sadi_raw_data = load_sadi_data(SADI_FILE_PATH)

        print("\nLoading Monte Carlo dropout data...")
        mc_dropout_raw_data = load_mc_dropout_data(MC_DROPOUT_FILE_PATH)

        # Prepare combined datasets for both methods
        print("\nPreparing SADI dataset...")
        sadi_combined_data = prepare_combined_dataset(sadi_raw_data, "SADI")

        print("\nPreparing Monte Carlo dropout dataset...")
        mc_dropout_combined_data = prepare_combined_dataset(mc_dropout_raw_data, "MC_Dropout")

        # Print category breakdowns for both methods
        print("\nSADI Prediction Category Breakdown:")
        print("-" * 50)
        sadi_categories, sadi_counts = np.unique(sadi_combined_data['pred_categories'], return_counts=True)
        sadi_percentages = sadi_counts / len(sadi_combined_data['pred_categories']) * 100
        for cat, count, percentage in zip(sadi_categories, sadi_counts, sadi_percentages):
            print(f"{cat}: {count} ({percentage:.2f}%)")

        print("\nMonte Carlo Dropout Prediction Category Breakdown:")
        print("-" * 50)
        mc_categories, mc_counts = np.unique(mc_dropout_combined_data['pred_categories'], return_counts=True)
        mc_percentages = mc_counts / len(mc_dropout_combined_data['pred_categories']) * 100
        for cat, count, percentage in zip(mc_categories, mc_counts, mc_percentages):
            print(f"{cat}: {count} ({percentage:.2f}%)")

        # Generate combined comparison plots
        print("\nGenerating combined comparison plots...")
        plot_combined_reliability_distributions(sadi_combined_data, mc_dropout_combined_data)
        plot_retention_curves_comparison(sadi_combined_data, mc_dropout_combined_data)

        # Print detailed comparison statistics
        print_comparison_statistics(sadi_combined_data, mc_dropout_combined_data)

        # Create threshold comparison table
        print("\nCreating threshold comparison table...")
        threshold_table = create_threshold_comparison_table(sadi_combined_data, mc_dropout_combined_data)
        print("\nThreshold Comparison Table:")
        print(threshold_table)

        print("\nCombined analysis complete. Results saved to files.")

        return sadi_combined_data, mc_dropout_combined_data

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()