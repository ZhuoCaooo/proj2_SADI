"""Additional analysis of uncertainty scores on LK trajectories"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the data
with open('cnn_results_density_based_0.0s.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Extract all the variables
model_pred_list_all = loaded_data['model_predictions']
reliability_scores_all = loaded_data['reliability_scores']
ground_truth_each_timestep = loaded_data['ground_truth']
probability_outputs = loaded_data['probability_outputs']


def get_final_ground_truth(ground_truth):
    """Get the final ground truth behavior"""
    return ground_truth[-1]


def find_fp_periods_in_lk(model_preds, ground_truths):
    """Find continuous periods of false positive predictions in LK trajectories

    Args:
        model_preds: List of model predictions
        ground_truths: List of corresponding ground truths

    Returns:
        List of tuples (start_idx, end_idx) for each continuous FP period
    """
    fp_periods = []

    start_idx = None
    for t in range(len(model_preds)):
        # False positive when prediction is LC (1 or 2) but ground truth is LK (0)
        is_fp = (model_preds[t] in [1, 2]) and (ground_truths[t] == 0)

        # Start of a new FP period
        if is_fp and start_idx is None:
            start_idx = t
        # End of current FP period
        elif not is_fp and start_idx is not None:
            fp_periods.append((start_idx, t - 1))
            start_idx = None

    # Handle case where FP period ends at sequence end
    if start_idx is not None:
        fp_periods.append((start_idx, len(model_preds) - 1))

    return fp_periods


def analyze_lk_false_positives(model_preds, ground_truths, reliability_scores):
    """Extract reliability scores around continuous false positive periods in LK trajectories

    Args:
        model_preds: List of model predictions
        ground_truths: List of corresponding ground truths
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from windows around FP periods
    """
    fp_reliability_scores = []
    sequence_length = len(model_preds)

    # Find continuous periods of false positives
    fp_periods = find_fp_periods_in_lk(model_preds, ground_truths)

    # Extract windows around each FP period
    for start_fp, end_fp in fp_periods:
        # Calculate window boundaries
        window_start = max(0, start_fp)
        window_end = min(sequence_length, end_fp)

        # Add reliability scores for this window
        fp_reliability_scores.extend(reliability_scores[window_start:window_end])

    return fp_reliability_scores


def analyze_lk_correct_predictions(model_preds, ground_truths, reliability_scores):
    """Extract reliability scores for periods of correct LK predictions

    Args:
        model_preds: List of model predictions
        ground_truths: List of corresponding ground truths
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from correct LK prediction periods
    """
    correct_reliability_scores = []

    # Loop through each timestep
    for t in range(len(model_preds)):
        # Correct LK prediction when both prediction and ground truth are 0
        if model_preds[t] == 0 and ground_truths[t] == 0:
            correct_reliability_scores.append(reliability_scores[t])

    return correct_reliability_scores


# Initialize collectors for LK reliability scores
lk_fp_scores = []
lk_correct_scores = []

# Statistics for LK analysis
lk_trajectory_count = 0
lk_fp_trajectory_count = 0

# Process each trajectory
for model_preds, ground_truths, reliability in zip(model_pred_list_all, ground_truth_each_timestep,
                                                   reliability_scores_all):
    final_ground_truth = get_final_ground_truth(ground_truths)
    final_prediction = model_preds[-1]

    # Only analyze LK trajectories where final ground truth is 0
    if final_ground_truth == 0:
        lk_trajectory_count += 1

        # Check if the final prediction matches the ground truth
        #if final_prediction == 0:
            # Analyze correct LK predictions
        correct_scores = analyze_lk_correct_predictions(model_preds, ground_truths, reliability)
        lk_correct_scores.extend(correct_scores)

            # Analyze false positives in correct LK trajectories
        fp_scores = analyze_lk_false_positives(model_preds, ground_truths, reliability)
        if len(fp_scores) > 0:
            lk_fp_trajectory_count += 1
            lk_fp_scores.extend(fp_scores)

# Convert to numpy arrays
lk_fp_scores = np.array(lk_fp_scores)
lk_correct_scores = np.array(lk_correct_scores)

# Create distribution plots for LK trajectories
plt.figure(figsize=(12, 6))

# Plot LK correct reliability scores
plt.subplot(121)
if len(lk_correct_scores) > 0:
    kde_correct = gaussian_kde(lk_correct_scores)
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_correct(x_range), 'g-', linewidth=2)
    plt.fill_between(x_range, kde_correct(x_range), alpha=0.3, color='green')
plt.title('Correct LK Prediction Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Plot LK false positive reliability scores
plt.subplot(122)
if len(lk_fp_scores) > 0:
    kde_fp = gaussian_kde(lk_fp_scores)
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
    plt.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='red')
plt.title('False Positive in LK Trajectories')
plt.xlabel('Reliability Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lk_trajectory_analysis.png', dpi=300)
plt.show()

# Print statistics for LK analysis
print("\nLK Trajectory Analysis Statistics:")
print(f"Total LK trajectories analyzed: {lk_trajectory_count}")
print(f"LK trajectories with false positives: {lk_fp_trajectory_count}")

if len(lk_fp_scores) > 0:
    print("\nLK False Positive Statistics:")
    print(f"Number of samples: {len(lk_fp_scores)}")
    print(f"Mean reliability score: {np.mean(lk_fp_scores):.4f}")
    print(f"Median reliability score: {np.median(lk_fp_scores):.4f}")
    print(f"Standard deviation: {np.std(lk_fp_scores):.4f}")

if len(lk_correct_scores) > 0:
    print("\nLK Correct Prediction Statistics:")
    print(f"Number of samples: {len(lk_correct_scores)}")
    print(f"Mean reliability score: {np.mean(lk_correct_scores):.4f}")
    print(f"Median reliability score: {np.median(lk_correct_scores):.4f}")
    print(f"Standard deviation: {np.std(lk_correct_scores):.4f}")

# Create comparative boxplot
plt.figure(figsize=(10, 6))
box_data = []
labels = []

if len(lk_correct_scores) > 0:
    box_data.append(lk_correct_scores)
    labels.append('LK Correct')

if len(lk_fp_scores) > 0:
    box_data.append(lk_fp_scores)
    labels.append('LK False Positives')

if box_data:
    plt.boxplot(box_data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Reliability Score')
    plt.title('Comparison of Reliability Scores in LK Trajectories')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('lk_reliability_boxplot.png', dpi=300)
    plt.show()

# Additional analysis: Length of false positive periods in LK trajectories
fp_lengths = []
for model_preds, ground_truths, _ in zip(model_pred_list_all, ground_truth_each_timestep, reliability_scores_all):
    final_ground_truth = get_final_ground_truth(ground_truths)
    final_prediction = model_preds[-1]

    # Only analyze LK trajectories with correct final prediction
    if final_ground_truth == 0 and final_prediction == 0:
        fp_periods = find_fp_periods_in_lk(model_preds, ground_truths)
        for start, end in fp_periods:
            fp_lengths.append(end - start + 1)

if fp_lengths:
    plt.figure(figsize=(8, 5))
    plt.hist(fp_lengths, bins=range(1, max(fp_lengths) + 2), alpha=0.7,
             color='orange', edgecolor='black')
    plt.xlabel('Length of False Positive Period (timesteps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of False Positive Period Lengths in LK Trajectories')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('lk_fp_period_lengths.png', dpi=300)
    plt.show()

    print("\nFalse Positive Period Lengths in LK Trajectories:")
    print(f"Number of false positive periods: {len(fp_lengths)}")
    print(f"Mean length: {np.mean(fp_lengths):.2f} timesteps")
    print(f"Median length: {np.median(fp_lengths):.2f} timesteps")
    print(f"Max length: {np.max(fp_lengths)} timesteps")