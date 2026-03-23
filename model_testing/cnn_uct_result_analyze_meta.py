# Additional analysis for multiple prediction time settings
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Define time settings and corresponding files
time_settings = [0.0, 0.5, 1.0, 1.5, 2.0]
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different color for each time setting

# Initialize storage for results
all_wrong_predictions = {}
all_correct_predictions = {}

# Load and process data for each time setting
for time in time_settings:
    filename = f'cnn_results_{time:.1f}s.pkl'
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)

    wrong_predictions = []
    correct_predictions = []

    # Extract predictions
    for model_preds, ground_truths, reliability in zip(
            loaded_data['model_predictions'],
            loaded_data['ground_truth'],
            loaded_data['reliability_scores']):

        final_label = ground_truths[-1]

        for pred, truth, rel_score in zip(model_preds, ground_truths, reliability):
            is_wrong = False
            if truth == 0:
                if pred != 0 and pred != final_label:
                    is_wrong = True
            else:
                if pred != truth and pred != 0:
                    is_wrong = True

            if is_wrong:
                wrong_predictions.append(rel_score)
            else:
                correct_predictions.append(rel_score)

    all_wrong_predictions[time] = np.array(wrong_predictions)
    all_correct_predictions[time] = np.array(correct_predictions)

# Create comparison plots
plt.figure(figsize=(15, 10))

# 1. Distribution Comparison
plt.subplot(2, 2, 1)
x_range = np.linspace(0, 1, 200)
for i, time in enumerate(time_settings):
    wrong_kde = gaussian_kde(all_wrong_predictions[time])
    correct_kde = gaussian_kde(all_correct_predictions[time])

    plt.plot(x_range, wrong_kde(x_range), '--', color=colors[i], alpha=0.7,
             label=f'{time}s Wrong')
    plt.plot(x_range, correct_kde(x_range), '-', color=colors[i], alpha=0.7,
             label=f'{time}s Correct')

plt.xlabel('Reliability Score')
plt.ylabel('Density')
plt.title('Distribution Comparison Across Time Settings')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 2. Error Rate Analysis

def create_normalized_error_plot(reliability_scores, prediction_errors, num_bins=50):
    bins = np.linspace(0, 1, num_bins + 1)
    reliability_binned = np.digitize(reliability_scores, bins) - 1

    total_wrong = np.sum(prediction_errors)
    total_correct = len(prediction_errors) - total_wrong

    normalized_rates = []
    bin_centers = []

    for i in range(num_bins):
        mask = reliability_binned == i
        if np.sum(mask) > 0:
            wrong_in_bin = np.sum(prediction_errors[mask])
            correct_in_bin = np.sum(mask) - wrong_in_bin

            wrong_rate = wrong_in_bin / total_wrong if total_wrong > 0 else 0
            correct_rate = correct_in_bin / total_correct if total_correct > 0 else 0

            normalized_rate = wrong_rate / (wrong_rate + correct_rate) if (wrong_rate + correct_rate) > 0 else 0

            normalized_rates.append(normalized_rate)
            bin_centers.append(np.mean([bins[i], bins[i + 1]]))

            # print(f"\nBin {i} ({bins[i]:.3f}-{bins[i + 1]:.3f}):")
            # print(f"  Wrong predictions: {wrong_in_bin} ({wrong_rate:.4f} of total wrong)")
            # print(f"  Correct predictions: {correct_in_bin} ({correct_rate:.4f} of total correct)")
            # print(f"  Normalized rate: {normalized_rate:.4f}")

    return np.array(bin_centers), np.array(normalized_rates)

plt.subplot(2, 2, 2)
for i, time in enumerate(time_settings):
    reliability_scores = np.concatenate([all_wrong_predictions[time],
                                         all_correct_predictions[time]])
    binary_errors = np.concatenate([np.ones(len(all_wrong_predictions[time])),
                                    np.zeros(len(all_correct_predictions[time]))])

    bin_centers, norm_rates = create_normalized_error_plot(reliability_scores,
                                                           binary_errors,
                                                           num_bins=50)

    # Smoothing
    smoothed_rates = gaussian_kde(bin_centers, weights=norm_rates)(bin_centers)
    smoothed_rates = smoothed_rates * np.max(norm_rates) / np.max(smoothed_rates)

    plt.plot(bin_centers, smoothed_rates, '-', color=colors[i],
             label=f'{time}s', alpha=0.7)

plt.xlabel('Reliability Score')
plt.ylabel('Normalized Error Rate')
plt.title('Error Rate Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Performance Metrics
plt.subplot(2, 2, 3)
metrics = {
    'Mean Wrong': [np.mean(all_wrong_predictions[t]) for t in time_settings],
    'Mean Correct': [np.mean(all_correct_predictions[t]) for t in time_settings]
}

x = np.arange(len(time_settings))
width = 0.35

plt.bar(x - width / 2, metrics['Mean Wrong'], width, label='Wrong Predictions',
        color='red', alpha=0.6)
plt.bar(x + width / 2, metrics['Mean Correct'], width, label='Correct Predictions',
        color='green', alpha=0.6)

plt.xlabel('Prediction Time (s)')
plt.ylabel('Mean Reliability Score')
plt.title('Mean Reliability Scores Comparison')
plt.xticks(x, [f'{t}s' for t in time_settings])
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Cumulative Analysis
plt.subplot(2, 2, 4)
thresholds = np.linspace(0, 1, 100)

for i, time in enumerate(time_settings):
    reliability_scores = np.concatenate([all_wrong_predictions[time],
                                         all_correct_predictions[time]])
    samples_included = [np.mean(reliability_scores >= threshold)
                        for threshold in thresholds]
    plt.plot(thresholds, samples_included, color=colors[i],
             label=f'{time}s', alpha=0.7)

plt.xlabel('Reliability Score Threshold')
plt.ylabel('Fraction of Samples Included')
plt.title('Cumulative Analysis Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics for each time setting
print("\nSummary Statistics Across Time Settings:")
print("\nTime Setting | Wrong Predictions | Correct Predictions")
print("-" * 60)
for time in time_settings:
    wrong_mean = np.mean(all_wrong_predictions[time])
    wrong_std = np.std(all_wrong_predictions[time])
    correct_mean = np.mean(all_correct_predictions[time])
    correct_std = np.std(all_correct_predictions[time])

    print(f"{time:4.1f}s     | {wrong_mean:.3f} ± {wrong_std:.3f} | {correct_mean:.3f} ± {correct_std:.3f}")




'''roc section'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define time settings and colors
time_settings = [0.0, 0.5, 1.0, 1.5, 2.0]
colors = ['blue', 'green', 'red', 'purple', 'orange']


def is_within_streak(predictions, current_idx, pred_value, window_size=25):
    """Check if a prediction is part of a streak of at least window_size steps."""
    predictions = np.array(predictions)
    n = len(predictions)
    start_idx = max(0, current_idx - window_size + 1)
    end_idx = min(n, current_idx + window_size)
    segment = predictions[start_idx:end_idx]
    streak_mask = (segment == pred_value)
    if not any(streak_mask):
        return False
    streak_starts = np.where(np.diff(np.concatenate(([False], streak_mask))))[0]
    streak_ends = np.where(np.diff(np.concatenate((streak_mask, [False]))))[0]
    for start, end in zip(streak_starts, streak_ends):
        streak_length = end - start + 1
        if streak_length >= window_size and start <= (current_idx - start_idx) <= end:
            return True
    return False


# Plot ROC curves for both classes
plt.figure(figsize=(15, 6))

# Plot settings
class_names = ['LCL', 'LCR']
line_styles = ['-', '--']  # Solid for all samples, dashed for reliability threshold

for class_idx, class_name in enumerate(['LCL', 'LCR']):
    plt.subplot(1, 2, class_idx + 1)

    for time_idx, time in enumerate(time_settings):
        filename = f'cnn_results_{time:.1f}s.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Prepare data for ROC curves
        y_true_onehot = []
        pred_scores = []
        all_reliability_scores = []

        for traj_idx, (model_preds, ground_truths, reliability) in enumerate(
                zip(data['model_predictions'], data['ground_truth'], data['reliability_scores'])):

            final_label = ground_truths[-1]
            if final_label == 0:
                continue

            model_preds = np.array(model_preds)
            for timestep, (pred, truth, rel_score) in enumerate(zip(model_preds, ground_truths, reliability)):
                if final_label > 0:
                    # For LCL (class 1) or LCR (class 2)
                    true_onehot = [1 if final_label == class_idx + 1 else 0]
                    y_true_onehot.append(true_onehot)
                    all_reliability_scores.append(rel_score)

                    pred_score = [0]
                    if pred == class_idx + 1:
                        pred_score[0] = rel_score
                    elif pred == 0 and is_within_streak(model_preds, timestep, 0, 25):
                        pred_score[0] = rel_score if final_label == class_idx + 1 else 0

                    pred_scores.append(pred_score)

        y_true_onehot = np.array(y_true_onehot)
        pred_scores = np.array(pred_scores)
        all_reliability_scores = np.array(all_reliability_scores)

        # Plot base ROC curve (all samples)
        fpr, tpr, _ = roc_curve(y_true_onehot, pred_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, line_styles[0], color=colors[time_idx],
                 label=f'{time}s all (AUC={roc_auc:.3f})')

        # Plot ROC curve with different thresholds for each class
        threshold = 0.2 if class_idx == 0 else 0.6  # 0.2 for LCL, 0.6 for LCR
        mask = all_reliability_scores >= threshold
        if np.sum(mask) > 0:
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask], pred_scores[mask])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            plt.plot(fpr_thresh, tpr_thresh, line_styles[1], color=colors[time_idx],
                     label=f'{time}s rel≥{threshold} (AUC={roc_auc_thresh:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {class_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()













































