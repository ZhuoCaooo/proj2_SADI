"""this is the analysis of uncertainty scores on trajectories"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the data
with open('cnn_results_0.0s_new.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Extract the variables
model_pred_list_all = loaded_data['model_predictions']
reliability_scores_all = loaded_data['reliability_scores']
ground_truth_each_timestep = loaded_data['ground_truth']

wrong_predictions_reliability = []
correct_predictions_reliability = []

# Iterate through each trajectory
for traj_idx, (model_preds, ground_truths, reliability) in enumerate(
        zip(model_pred_list_all, ground_truth_each_timestep, reliability_scores_all)):
    final_label = ground_truths[-1]

    for timestep, (pred, truth, rel_score) in enumerate(zip(model_preds, ground_truths, reliability)):
        is_wrong = False

        if truth == 0:
            if pred != 0 and pred != final_label:
                is_wrong = True

        else:
            if pred != truth and pred != 0:
                is_wrong = True


        if is_wrong:
            wrong_predictions_reliability.append(rel_score)
        else:
            correct_predictions_reliability.append(rel_score)


# Convert to numpy arrays
wrong_predictions_reliability = np.array(wrong_predictions_reliability)
correct_predictions_reliability = np.array(correct_predictions_reliability)

# Create the distribution plot
plt.figure(figsize=(10, 6))

# Create KDEs
kde_wrong = gaussian_kde(wrong_predictions_reliability)
kde_correct = gaussian_kde(correct_predictions_reliability)
x_range = np.linspace(0, 1, 200)
kde_wrong_values = kde_wrong(x_range)
kde_correct_values = kde_correct(x_range)

# Plot distributions
plt.plot(x_range, kde_wrong_values, 'r-', linewidth=2, label='Wrong Predictions')
plt.fill_between(x_range, kde_wrong_values, alpha=0.3, color='red')
plt.plot(x_range, kde_correct_values, 'g-', linewidth=2, label='Correct Predictions')
plt.fill_between(x_range, kde_correct_values, alpha=0.3, color='green')

plt.xlabel('Reliability Score', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Print basic statistics
print("Wrong Predictions:")
print(f"Number of wrong predictions: {len(wrong_predictions_reliability)}")
print(f"Mean reliability score: {np.mean(wrong_predictions_reliability):.4f}")
print(f"Median reliability score: {np.median(wrong_predictions_reliability):.4f}")
print(f"Standard deviation: {np.std(wrong_predictions_reliability):.4f}")

print("\nCorrect Predictions:")
print(f"Number of correct predictions: {len(correct_predictions_reliability)}")
print(f"Mean reliability score: {np.mean(correct_predictions_reliability):.4f}")
print(f"Median reliability score: {np.median(correct_predictions_reliability):.4f}")
print(f"Standard deviation: {np.std(correct_predictions_reliability):.4f}")

# Create combined arrays for all predictions
reliability_scores = np.concatenate([wrong_predictions_reliability, correct_predictions_reliability])
binary_errors = np.concatenate([np.ones(len(wrong_predictions_reliability)),
                              np.zeros(len(correct_predictions_reliability))])

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

# Create new figure for error analysis plots
plt.figure(figsize=(12, 5))

# Plot 1: Normalized Error Rate with 50 bins
plt.subplot(1, 2, 1)
bin_centers, norm_rates = create_normalized_error_plot(reliability_scores, binary_errors, num_bins=50)

# Add smoothing for better visualization
smoothed_rates = gaussian_kde(bin_centers, weights=norm_rates)(bin_centers)
smoothed_rates = smoothed_rates * np.max(norm_rates) / np.max(smoothed_rates)  # Rescale to match original scale

plt.plot(bin_centers, norm_rates, 'o', alpha=0.5, markersize=3, label='Raw Data')
plt.plot(bin_centers, smoothed_rates, '-', label='Smoothed Trend')
plt.xlabel('Reliability Score (binned)')
plt.ylabel('Normalized Error Rate')
plt.title('Normalized Error Rate vs Reliability Score (50 bins)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Cumulative Analysis
plt.subplot(1, 2, 2)
thresholds = np.linspace(0, 1, 100)
samples_included = []

for threshold in thresholds:
    mask = reliability_scores >= threshold
    samples_included.append(np.mean(mask))

plt.plot(thresholds, samples_included, label='Fraction of Samples')
plt.xlabel('Reliability Score Threshold')
plt.ylabel('Rate')
plt.title('Cumulative Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Print cumulative analysis at key thresholds
key_thresholds = [0.2, 0.4, 0.6, 0.8]
print("\nCumulative Analysis at Key Thresholds:")
for threshold in key_thresholds:
    mask = reliability_scores >= threshold
    if np.sum(mask) > 0:
        error_rate = np.mean(binary_errors[mask])
        samples_ratio = np.mean(mask)
        print(f"\nThreshold {threshold:.1f}:")
        print(f"Error rate: {error_rate:.4f}")
        print(f"Fraction of samples included: {samples_ratio:.4f}")




# Add these imports if not already present
from sklearn.metrics import roc_curve, auc

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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


# Prepare data for ROC curves (excluding class 0)
y_true_onehot = []  # One-hot encoded ground truth (only classes 1 and 2)
pred_scores = []  # Prediction scores for each class (only classes 1 and 2)
all_reliability_scores = []  # Store all reliability scores

for traj_idx, (model_preds, ground_truths, reliability) in enumerate(
        zip(model_pred_list_all, ground_truth_each_timestep, reliability_scores_all)):

    final_label = ground_truths[-1]

    # Skip trajectories with final label 0
    if final_label == 0:
        continue

    model_preds = np.array(model_preds)

    for timestep, (pred, truth, rel_score) in enumerate(zip(model_preds, ground_truths, reliability)):
        # Only include if final label is 1 or 2
        if final_label > 0:
            true_onehot = [0, 0]
            true_onehot[final_label - 1] = 1
            y_true_onehot.append(true_onehot)
            all_reliability_scores.append(rel_score)

            # Initialize prediction scores for both classes
            pred_score = [0, 0]

            if pred != 0:
                # Assign reliability score to the predicted class
                pred_score[pred - 1] = rel_score
            else:  # pred == 0
                # If it's a valid streak of zeros, assign some confidence to the final label
                if is_within_streak(model_preds, timestep, 0, 25):
                    pred_score[final_label - 1] = rel_score

            pred_scores.append(pred_score)

# Convert to numpy arrays
y_true_onehot = np.array(y_true_onehot)
pred_scores = np.array(pred_scores)
all_reliability_scores = np.array(all_reliability_scores)


# Create separate figures for LCL and LCR
class_names = ['LCL', 'LCR']
colors = ['g', 'r', 'c', 'm']
reliability_thresholds = [0.2, 0.4, 0.6, 0.8]

for i in range(2):
    plt.figure(figsize=(8, 6))

    # Plot base ROC curve
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], pred_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', label=f'All samples (AUC = {roc_auc:.3f})')

    # Plot ROC curves for different thresholds
    for threshold, color in zip(reliability_thresholds, colors):
        mask = all_reliability_scores >= threshold
        if np.sum(mask) > 0:
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, i],
                                                  pred_scores[mask, i])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            plt.plot(fpr_thresh, tpr_thresh, color=color, linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {class_names[i]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Print statistics
print("\nNumber of samples per class:")
for i in range(2):
    print(f"Class {i + 1}: {np.sum(y_true_onehot[:, i])}")

print("\nPrediction score statistics:")
for i in range(2):
    scores = pred_scores[:, i]
    print(f"\nClass {i + 1}:")
    print(f"Non-zero predictions: {np.sum(scores > 0)}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Max score: {np.max(scores):.4f}")




