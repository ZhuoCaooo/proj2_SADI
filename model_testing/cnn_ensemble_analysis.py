"""
Revised Analysis of Ensemble CNN Results for Lane Change Prediction
This script analyzes reliability scores and retention rates with updated definitions
that focus on the final behavior of each trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pickle
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd

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

# File path to the ensemble results
FILE_PATH = 'ensemble_sequential_test_results.pkl'


def load_data(file_path):
    """Load pickle data containing model predictions and reliability scores"""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    print("Data loaded successfully")
    print("Available keys:", loaded_data.keys())

    return loaded_data


def prepare_combined_dataset(data):
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
    all_pred_categories = []  # New field for categorizing predictions

    for traj_idx, (preds, gt, reliability) in enumerate(zip(
            data['model_predictions'],
            data['ground_truth'],
            data['reliability_scores'])):

        # Get the final behavior (last non-zero prediction)
        final_behavior = 0
        for pred in reversed(gt):
            if pred != 0:
                final_behavior = pred
                break

        trajectory_length = len(preds)

        for t in range(len(preds)):
            # Skip if any data is missing
            if t >= len(gt) or t >= len(reliability):
                continue

            pred_value = preds[t]
            gt_value = gt[t]
            rel_value = reliability[t]

            # Determine the prediction category based on new definitions
            if pred_value != 0 and pred_value == final_behavior:
                # Definition 3: True Positive - prediction matches final behavior
                pred_category = "TP"
            elif pred_value != 0 and pred_value != final_behavior:
                # Definition 1: False Positive - predicted lane change differs from final behavior
                pred_category = "FP"
            elif pred_value == 0 and final_behavior != 0:
                # Definition 2: False Negative - still predicting LK after timestep 120 when final behavior is LC
                pred_category = "FN"
            elif final_behavior == 0 and pred_value == 0:
                # Definition 4: True Negative - correctly predicting LK when final behavior is LK
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
        'pred_categories': np.array(all_pred_categories)
    }


def analyze_threshold_effects(combined_data, thresholds):
    """
    Analyze the effect of different reliability thresholds on prediction metrics
    using the updated prediction categories
    """
    results = []

    # Extract arrays for easier access
    reliability = combined_data['reliability']
    pred_categories = combined_data['pred_categories']

    total_samples = len(reliability)

    # Count samples by prediction category
    category_counts = {}
    for cat in np.unique(pred_categories):
        category_counts[cat] = np.sum(pred_categories == cat)

    print(f"Total samples: {total_samples}")
    print(f"Category counts: {category_counts}")

    # For each threshold
    for threshold in thresholds:
        # Create mask for samples above threshold
        mask = reliability >= threshold

        # Skip if no samples meet the threshold
        if np.sum(mask) == 0:
            continue

        # Calculate metrics
        filtered_count = np.sum(mask)

        # Calculate retention rate (% of samples kept)
        retention_rate = filtered_count / total_samples

        # Calculate retention rate by prediction category
        retention_by_category = {}

        for cat in np.unique(pred_categories):
            cat_mask = pred_categories == cat
            cat_count = np.sum(cat_mask)

            if cat_count > 0:
                cat_filtered_mask = np.logical_and(cat_mask, mask)
                cat_filtered_count = np.sum(cat_filtered_mask)
                retention_by_category[f'retention_{cat}'] = cat_filtered_count / cat_count

                # Store additional information
                retention_by_category[f'count_{cat}_total'] = cat_count
                retention_by_category[f'count_{cat}_retained'] = cat_filtered_count

        # Store results
        result = {
            'threshold': threshold,
            'samples_kept': filtered_count,
            'total_samples': total_samples,
            'retention_rate': retention_rate,
            **retention_by_category
        }
        results.append(result)

    return pd.DataFrame(results)


def plot_threshold_metrics(threshold_results, output_prefix='ensemble'):
    """
    Create a publication-quality plot showing how different metrics
    change with reliability thresholds using the updated definitions
    """
    # Set up figure with two subplots
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot: Retention rate vs threshold
    ax1.plot(threshold_results['threshold'], threshold_results['retention_rate'],
             'o-', label='Overall', color='#023047', linewidth=2)

    # Add retention by prediction category with specific colors
    category_colors = {
        'TP': '#2A9D8F',  # Green
        'FP': '#E63946',  # Red
        'FN': '#457B9D',  # Blue
        'TN': '#6A329F',  # Purple
        'Other': '#A8DADC'  # Light blue
    }

    for category, color in category_colors.items():
        col = f'retention_{category}'
        if col in threshold_results.columns:
            ax1.plot(threshold_results['threshold'], threshold_results[col],
                     'o-', label=category, color=color, linewidth=2)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Reliability Threshold')
    ax1.set_ylabel('Retention Rate')
    ax1.set_title('Retention Rate vs. Reliability Threshold (Revised Definitions)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure
    filename = f'{output_prefix}_retention_analysis.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved retention analysis figure to {filename}")

    return fig


def plot_reliability_distributions(combined_data, output_prefix='ensemble'):
    """
    Create a publication-quality plot showing reliability score distributions
    for different prediction categories using the updated definitions
    """
    # Extract data
    reliability = combined_data['reliability']
    pred_categories = combined_data['pred_categories']

    # Set up figure - now with 4 subplots to include TN
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)

    # Define colors
    colors = {
        'TP': '#2A9D8F',  # green
        'FP': '#E63946',  # red
        'FN': '#457B9D',  # blue
        'TN': '#6A329F'   # purple
    }

    # Plot each prediction category
    for i, category in enumerate(['TP', 'FP', 'FN', 'TN']):
        ax = axes[i]
        mask = pred_categories == category

        if np.sum(mask) > 10:  # Only plot if we have enough samples
            values = reliability[mask]

            # Create KDE plot
            sns.kdeplot(
                values,
                ax=ax,
                fill=True,
                color=colors.get(category, 'gray'),
                cut=0,
                label=f'n={np.sum(mask)}'
            )

            # Add mean and median lines
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='k', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='k', linestyle=':', alpha=0.7,
                       label=f'Median: {median_val:.2f}')

        ax.set_xlim(0, 1)
        ax.set_title(f'{category} Reliability Scores')
        ax.set_xlabel('Reliability Score')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.tight_layout()

    # Save the figure
    filename = f'{output_prefix}_reliability_distributions.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved reliability distributions figure to {filename}")

    return fig


def generate_trajectory_confusion_matrix(data, output_prefix='ensemble'):
    """
    Generate a trajectory-wise confusion matrix comparing only the last prediction
    with the final ground truth for each trajectory.
    """
    print("Generating trajectory-wise confusion matrix...")

    # Initialize lists to store final predictions and ground truths
    final_predictions = []
    final_ground_truths = []

    # For each trajectory
    for traj_idx, (preds, gt) in enumerate(zip(
            data['model_predictions'],
            data['ground_truth'])):

        # Get the final prediction (last timestep)
        final_pred = preds[-1] if len(preds) > 0 else 0

        # Get the final ground truth (last non-zero value, or 0 if all zero)
        final_gt = 0
        for pred in reversed(gt):
            if pred != 0:
                final_gt = pred
                break

        final_predictions.append(final_pred)
        final_ground_truths.append(final_gt)

    # Convert to numpy arrays
    final_predictions = np.array(final_predictions)
    final_ground_truths = np.array(final_ground_truths)

    # Get unique classes (ensure 0, 1, 2 are all represented)
    classes = np.unique(np.concatenate([final_predictions, final_ground_truths, [0, 1, 2]]))
    classes.sort()  # Ensure order: 0 (LK), 1 (LCL), 2 (LCR)
    n_classes = len(classes)

    # Create confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for i in range(len(final_predictions)):
        pred_idx = np.where(classes == final_predictions[i])[0][0]
        gt_idx = np.where(classes == final_ground_truths[i])[0][0]
        conf_matrix[gt_idx, pred_idx] += 1

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Use a nice color scheme
    cmap = plt.cm.Blues

    # Plot heatmap
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)

    # Add title and labels
    ax.set_title('Trajectory-wise Confusion Matrix\n(Final Prediction vs Final Ground Truth)')

    # Set tick labels
    tick_labels = ['LK (0)', 'LCL (1)', 'LCR (2)'] if n_classes == 3 else [f'{c}' for c in classes]
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over the data and create annotations
    for i in range(n_classes):
        for j in range(n_classes):
            # Calculate percentage
            total_in_row = conf_matrix[i].sum()
            percentage = 100 * conf_matrix[i, j] / total_in_row if total_in_row > 0 else 0

            # Add text with count and percentage
            text = ax.text(j, i, f"{conf_matrix[i, j]}\n({percentage:.1f}%)",
                           ha="center", va="center",
                           color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
                           fontsize=9)

    # Add axis labels
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Prediction')

    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.2%}', ha='center')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    filename = f'{output_prefix}_trajectory_confusion_matrix.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {filename}")

    # Also calculate and return performance metrics
    metrics = {}

    # Calculate class-wise metrics
    for i, cls in enumerate(classes):
        class_name = tick_labels[i]

        # True positives, false positives, false negatives
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        # Calculate precision, recall, F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'class_{cls}_precision'] = precision
        metrics[f'class_{cls}_recall'] = recall
        metrics[f'class_{cls}_f1'] = f1

    # Calculate macro averages
    metrics['macro_precision'] = np.mean([metrics[f'class_{cls}_precision'] for cls in classes])
    metrics['macro_recall'] = np.mean([metrics[f'class_{cls}_recall'] for cls in classes])
    metrics['macro_f1'] = np.mean([metrics[f'class_{cls}_f1'] for cls in classes])
    metrics['accuracy'] = accuracy

    # Print metrics
    print("\nTrajectory-wise Classification Metrics:")
    print("-" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("-" * 50)
    print("Class-wise metrics:")
    for i, cls in enumerate(classes):
        class_name = tick_labels[i]
        print(f"{class_name}: Precision={metrics[f'class_{cls}_precision']:.4f}, " +
              f"Recall={metrics[f'class_{cls}_recall']:.4f}, " +
              f"F1={metrics[f'class_{cls}_f1']:.4f}")
    print("-" * 50)
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    return conf_matrix, metrics, fig

def create_summary_table(threshold_results):
    """
    Create a summary table of key metrics at different reliability thresholds
    using the updated definitions
    """
    # Select key thresholds
    thresholds = [0.0, 0.4, 0.6, 0.8, 0.9]

    # Get rows for these thresholds
    summary_rows = []

    for threshold in thresholds:
        threshold_df = threshold_results[threshold_results['threshold'] >= threshold]
        if not threshold_df.empty:
            row = threshold_df.iloc[0]

            # Create summary row
            summary_row = {
                'Reliability Threshold': threshold,
                'Retention Rate (Overall)': row['retention_rate'],
            }

            # Add retention by prediction category
            for category in ['TP', 'FP', 'FN', 'TN', 'Other']:
                col = f'retention_{category}'
                count_col = f'count_{category}_retained'
                total_col = f'count_{category}_total'

                if col in row:
                    summary_row[f'{category} Retention'] = row[col]

                if count_col in row and total_col in row:
                    summary_row[f'{category} Count'] = f"{int(row[count_col])}/{int(row[total_col])}"

            summary_rows.append(summary_row)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Format for display
    pd.set_option('display.float_format', '{:.2%}'.format)
    formatted_df = summary_df.copy()

    # Save to CSV
    summary_df.to_csv(f'ensemble_summary_table.csv', index=False)
    print(f"Saved summary table to ensemble_summary_table.csv")

    return formatted_df


def plot_threshold_metrics_improved(threshold_results, output_prefix='ensemble'):
    """
    Create a publication-quality plot showing how different metrics
    change with reliability thresholds using the updated definitions
    IMPROVED VERSION: Black & white compatible with distinct line styles and markers
    """
    # Set up figure
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Define line styles and markers for B&W compatibility
    line_configs = {
        'Overall': {
            'color': 'black',
            'linestyle': '-',
            'marker': 'o',
            'markersize': 6,
            'linewidth': 2.5,
            'markerfacecolor': 'black',
            'markeredgecolor': 'black'
        },
        'TP': {
            'color': 'black',
            'linestyle': '-',
            'marker': 's',
            'markersize': 5,
            'linewidth': 2,
            'markerfacecolor': 'black',
            'markeredgecolor': 'black'
        },
        'FP': {
            'color': 'black',
            'linestyle': '--',
            'marker': '^',
            'markersize': 6,
            'linewidth': 2,
            'markerfacecolor': 'black',
            'markeredgecolor': 'black'
        },
        'FN': {
            'color': 'black',
            'linestyle': '-.',
            'marker': 'D',
            'markersize': 5,
            'linewidth': 2,
            'markerfacecolor': 'black',
            'markeredgecolor': 'black'
        },
        'TN': {
            'color': 'black',
            'linestyle': ':',
            'marker': 'v',
            'markersize': 6,
            'linewidth': 2.5,
            'markerfacecolor': 'black',
            'markeredgecolor': 'black'
        },
        'Other': {
            'color': 'gray',
            'linestyle': '-',
            'marker': 'x',
            'markersize': 6,
            'linewidth': 1.5,
            'markerfacecolor': 'gray',
            'markeredgecolor': 'gray'
        }
    }

    # Plot: Overall retention rate first
    if 'retention_rate' in threshold_results.columns:
        config = line_configs['Overall']
        ax1.plot(threshold_results['threshold'], threshold_results['retention_rate'],
                 label='Overall', **config)

    # Add retention by prediction category with B&W compatible styling
    for category in ['TP', 'FP', 'FN', 'TN', 'Other']:
        col = f'retention_{category}'
        if col in threshold_results.columns:
            config = line_configs[category]
            ax1.plot(threshold_results['threshold'], threshold_results[col],
                     label=category, **config)

    # Formatting
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Reliability Threshold', fontweight='bold')
    ax1.set_ylabel('Retention Rate', fontweight='bold')
    ax1.set_title('Retention Rate vs. Reliability Threshold (Revised Definitions)',
                  fontweight='bold')

    # Improve legend - use 2 columns and better styling
    legend = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Grid styling
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    plt.tight_layout()

    # Save the figure with new filename to avoid overwrite
    filename = f'{output_prefix}_retention_analysis_bw_improved.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved improved B&W retention analysis figure to {filename}")

    return fig


def plot_threshold_metrics_color_improved(threshold_results, output_prefix='ensemble'):
    """
    Create a color version that's still B&W readable
    Color version with improved styling but maintaining B&W compatibility
    """
    # Set up figure
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Define line styles and colors that work in both color and B&W
    line_configs = {
        'Overall': {
            'color': '#000000',  # Black
            'linestyle': '-',
            'marker': 'o',
            'markersize': 6,
            'linewidth': 2.5
        },
        'TP': {
            'color': '#2E8B57',  # SeaGreen - distinct from others
            'linestyle': '-',
            'marker': 's',
            'markersize': 5,
            'linewidth': 2
        },
        'FP': {
            'color': '#DC143C',  # Crimson - distinct red
            'linestyle': '--',
            'marker': '^',
            'markersize': 6,
            'linewidth': 2
        },
        'FN': {
            'color': '#4169E1',  # RoyalBlue - distinct blue
            'linestyle': '-.',
            'marker': 'D',
            'markersize': 5,
            'linewidth': 2
        },
        'TN': {
            'color': '#9932CC',  # DarkOrchid - distinct purple
            'linestyle': ':',
            'marker': 'v',
            'markersize': 6,
            'linewidth': 2.5
        },
        'Other': {
            'color': '#708090',  # SlateGray
            'linestyle': '-',
            'marker': 'x',
            'markersize': 6,
            'linewidth': 1.5
        }
    }

    # Plot: Overall retention rate first
    if 'retention_rate' in threshold_results.columns:
        config = line_configs['Overall']
        ax1.plot(threshold_results['threshold'], threshold_results['retention_rate'],
                 label='Overall', **config)

    # Add retention by prediction category
    for category in ['TP', 'FP', 'FN', 'TN', 'Other']:
        col = f'retention_{category}'
        if col in threshold_results.columns:
            config = line_configs[category]
            ax1.plot(threshold_results['threshold'], threshold_results[col],
                     label=category, **config)

    # Formatting
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Reliability Threshold', fontweight='bold')
    ax1.set_ylabel('Retention Rate', fontweight='bold')
    ax1.set_title('Retention Rate vs. Reliability Threshold (Revised Definitions)',
                  fontweight='bold')

    # Improve legend
    legend = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Grid styling
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    plt.tight_layout()

    # Save the figure with new filename
    filename = f'{output_prefix}_retention_analysis_color_improved.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved color-improved retention analysis figure to {filename}")

    return fig


def main():
    """Main analysis function with revised definitions"""
    print("Starting revised reliability threshold analysis...")

    # Load the data
    data = load_data(FILE_PATH)

    # Generate trajectory-wise confusion matrix
    conf_matrix, metrics, _ = generate_trajectory_confusion_matrix(data)

    # Prepare combined dataset with revised definitions
    print("Preparing combined dataset with revised definitions...")
    combined_data = prepare_combined_dataset(data)

    # Count prediction categories
    categories, counts = np.unique(combined_data['pred_categories'], return_counts=True)
    category_percentages = counts / len(combined_data['pred_categories']) * 100

    print("\nPrediction Category Breakdown:")
    print("-" * 40)
    for cat, count, percentage in zip(categories, counts, category_percentages):
        print(f"{cat}: {count} ({percentage:.2f}%)")
    print("-" * 40)

    # Define reliability thresholds to analyze
    thresholds = np.linspace(0, 0.95, 20)

    # Analyze threshold effects with revised definitions
    print("Analyzing threshold effects...")
    threshold_results = analyze_threshold_effects(combined_data, thresholds)

    # Generate and save plots
    print("Generating plots...")
    plot_reliability_distributions(combined_data)
    plot_threshold_metrics(threshold_results)
    plot_threshold_metrics_color_improved(threshold_results, 'ensemble')  # Color version

    # Create summary table
    print("Creating summary table...")
    summary_table = create_summary_table(threshold_results)
    print("\nSummary Table:")
    print(summary_table)

    # Print key findings
    print("\nKEY FINDINGS:")
    print("-" * 50)

    # Get metrics at different thresholds
    baseline = threshold_results.iloc[0]
    threshold_80_series = threshold_results[threshold_results['threshold'] >= 0.8]

    print(f"Baseline retention (all data): 100%")

    if not threshold_80_series.empty:
        threshold_80_row = threshold_80_series.iloc[0]
        retention_80 = threshold_80_row['retention_rate']

        print(f"Overall retention rate at 0.8 threshold: {retention_80:.1%}")

        # Print retention by prediction category
        for category in ['TP', 'FP', 'FN', 'TN']:
            col = f'retention_{category}'
            if col in threshold_80_row:
                print(f"Retention of {category} at 0.8 threshold: {threshold_80_row[col]:.1%}")
    else:
        print("No data available for threshold >= 0.8")

    # Save complete results
    threshold_results.to_csv('ensemble_threshold_analysis_results.csv', index=False)

    print("\nAnalysis complete. Results saved to files.")
    return threshold_results

if __name__ == "__main__":
    main()