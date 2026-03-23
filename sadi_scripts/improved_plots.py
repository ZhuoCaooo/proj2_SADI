"""
Integration script to create improved retention plots using existing data
This script loads your existing pickle files and creates improved B&W readable plots
"""

import pickle
import os
import sys
import numpy as np
import pandas as pd

# Import the improved plotting functions
# (Make sure the improved plot functions are in the same directory or adjust the import)
from improved_plot_functions import (
    plot_threshold_metrics_improved,
    plot_threshold_metrics_color_version,
    create_improved_plots
)


def load_data(file_path):
    """Load pickle data containing model predictions and reliability scores"""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return {
        'model_predictions': loaded_data['model_predictions'],
        'reliability_scores': loaded_data['reliability_scores'],
        'ground_truth': loaded_data['ground_truth'],
        'probability_outputs': loaded_data['probability_outputs'],
        'ratio_scores': loaded_data.get('ratio_scores', []),
        'absolute_scores': loaded_data.get('absolute_scores', [])
    }


def get_final_behavior(predictions):
    """Get the final behavior (last non-zero prediction)"""
    final_pred = 0
    for pred in reversed(predictions):
        if pred != 0:
            final_pred = pred
            break
    return final_pred


def prepare_combined_dataset(data):
    """Create a combined dataset for all predictions across all trajectories"""
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

        final_behavior = get_final_behavior(gt)
        trajectory_length = len(preds)

        for t in range(len(preds)):
            if t >= len(gt) or t >= len(reliability):
                continue

            pred_value = preds[t]
            gt_value = gt[t]
            rel_value = reliability[t]

            # Determine prediction category
            if pred_value != 0 and pred_value == final_behavior:
                pred_category = "TP"
            elif pred_value != 0 and pred_value != final_behavior:
                pred_category = "FP"
            elif pred_value == 0 and final_behavior != 0 and t > 120:
                pred_category = "FN"
            elif pred_value == 0 and final_behavior == 0:
                pred_category = "TN"
            else:
                pred_category = "Other"

            all_preds.append(pred_value)
            all_ground_truth.append(gt_value)
            all_reliability.append(rel_value)
            all_trajectory_ids.append(traj_idx)
            all_timesteps.append(t)
            all_final_behaviors.append(final_behavior)
            all_pred_categories.append(pred_category)

    return {
        'predictions': np.array(all_preds),
        'ground_truth': np.array(all_ground_truth),
        'reliability': np.array(all_reliability),
        'trajectory_ids': np.array(all_trajectory_ids),
        'timesteps': np.array(all_timesteps),
        'final_behaviors': np.array(all_final_behaviors),
        'pred_categories': np.array(all_pred_categories)
    }


def analyze_threshold_effects(combined_data, thresholds):
    """Analyze the effect of different reliability thresholds on prediction metrics"""
    results = []
    reliability = combined_data['reliability']
    pred_categories = combined_data['pred_categories']
    total_samples = len(reliability)

    for threshold in thresholds:
        mask = reliability >= threshold
        if np.sum(mask) == 0:
            continue

        filtered_count = np.sum(mask)
        retention_rate = filtered_count / total_samples
        retention_by_category = {}

        for cat in np.unique(pred_categories):
            cat_mask = pred_categories == cat
            cat_count = np.sum(cat_mask)

            if cat_count > 0:
                cat_filtered_mask = np.logical_and(cat_mask, mask)
                cat_filtered_count = np.sum(cat_filtered_mask)
                retention_by_category[f'retention_{cat}'] = cat_filtered_count / cat_count
                retention_by_category[f'count_{cat}_total'] = cat_count
                retention_by_category[f'count_{cat}_retained'] = cat_filtered_count

        result = {
            'threshold': threshold,
            'samples_kept': filtered_count,
            'total_samples': total_samples,
            'retention_rate': retention_rate,
            **retention_by_category
        }
        results.append(result)

    return pd.DataFrame(results)


def main():
    """Main function to create improved plots"""

    # File paths for different prediction horizons
    file_paths = [
        '../model_testing_paper_2/cnn_results_density_based_0.0s_new.pkl',
        '../model_testing_paper_2/cnn_results_density_based_0.5s_new.pkl',
        '../model_testing_paper_2/cnn_results_density_based_1.0s_new.pkl',
        '../model_testing_paper_2/cnn_results_density_based_1.5s_new.pkl',
        '../model_testing_paper_2/cnn_results_density_based_2.0s_new.pkl'
    ]

    # Labels for each horizon
    horizon_labels = ["0.0s", "0.5s", "1.0s", "1.5s", "2.0s"]

    print("Creating improved retention plots...")
    print("=" * 60)

    # Process each horizon
    for file_path, horizon_label in zip(file_paths, horizon_labels):
        print(f"\nProcessing horizon: {horizon_label}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            # Load and process data
            data = load_data(file_path)
            combined_data = prepare_combined_dataset(data)

            # Analyze thresholds
            thresholds = np.linspace(0, 0.95, 20)
            threshold_results = analyze_threshold_effects(combined_data, thresholds)

            # Create improved plots
            print(f"Creating improved B&W plot for {horizon_label}...")
            fig1 = plot_threshold_metrics_improved(threshold_results, horizon_label, 'sadi')

            print(f"Creating color-improved plot for {horizon_label}...")
            fig2 = plot_threshold_metrics_color_version(threshold_results, horizon_label, 'sadi')

            # Close figures to free memory
            import matplotlib.pyplot as plt
            plt.close(fig1)
            plt.close(fig2)

            print(f"✓ Completed plots for {horizon_label}")

        except Exception as e:
            print(f"✗ Error processing {horizon_label}: {str(e)}")

    print("\n" + "=" * 60)
    print("Improved plot generation complete!")
    print("\nFiles created:")
    print("- sadi_retention_analysis_improved_[horizon].png (B&W optimized)")
    print("- sadi_retention_analysis_color_improved_[horizon].png (Color version)")


if __name__ == "__main__":
    main()