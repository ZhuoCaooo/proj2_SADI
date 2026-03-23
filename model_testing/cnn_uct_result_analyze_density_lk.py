"""Additional analysis of uncertainty scores on LK trajectories"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the data - update path to your file location
with open('../model_testing_paper_2/cnn_results_density_based_0.5s.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Extract all the variables including the new metrics
model_pred_list_all = loaded_data['model_predictions']
reliability_scores_all = loaded_data['reliability_scores']
ground_truth_each_timestep = loaded_data['ground_truth']
probability_outputs = loaded_data['probability_outputs']
ratio_scores_all = loaded_data.get('ratio_scores', [])  # Use get to handle if these keys don't exist
absolute_scores_all = loaded_data.get('absolute_scores', [])


def get_final_ground_truth(ground_truth):
    """Get the final ground truth behavior"""
    return ground_truth[-1]


def find_fp_periods_in_lk(model_preds, ground_truths):
    """Find continuous periods of false positive predictions in LK trajectories"""
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


def analyze_lk_false_positives(model_preds, ground_truths, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Extract reliability scores around continuous false positive periods in LK trajectories"""
    fp_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    sequence_length = len(model_preds)

    # Find continuous periods of false positives
    fp_periods = find_fp_periods_in_lk(model_preds, ground_truths)

    # Extract windows around each FP period
    for start_fp, end_fp in fp_periods:
        # Calculate window boundaries
        window_start = max(0, start_fp)
        window_end = min(sequence_length, end_fp)

        # Add reliability scores for this window
        fp_metrics['reliability_scores'].extend(reliability_scores[window_start:window_end])

        # Add other metrics if available
        if ratio_scores is not None:
            fp_metrics['ratio_scores'].extend(ratio_scores[window_start:window_end])
        if absolute_scores is not None:
            fp_metrics['absolute_scores'].extend(absolute_scores[window_start:window_end])

    return fp_metrics


def analyze_lk_correct_predictions(model_preds, ground_truths, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Extract reliability scores for periods of correct LK predictions"""
    correct_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    # Loop through each timestep
    for t in range(len(model_preds)):
        # Correct LK prediction when both prediction and ground truth are 0
        if model_preds[t] == 0 and ground_truths[t] == 0:
            correct_metrics['reliability_scores'].append(reliability_scores[t])

            # Add other metrics if available
            if ratio_scores is not None:
                correct_metrics['ratio_scores'].append(ratio_scores[t])
            if absolute_scores is not None:
                correct_metrics['absolute_scores'].append(absolute_scores[t])

    return correct_metrics


# Initialize collectors for LK reliability scores
lk_fp_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

lk_correct_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

# Statistics for LK analysis
lk_trajectory_count = 0
lk_fp_trajectory_count = 0

# Process each trajectory
for idx, (model_preds, ground_truths, reliability) in enumerate(zip(
        model_pred_list_all, ground_truth_each_timestep, reliability_scores_all)):

    # Get additional metrics if available
    ratio = ratio_scores_all[idx] if idx < len(ratio_scores_all) else None
    absolute = absolute_scores_all[idx] if idx < len(absolute_scores_all) else None

    final_ground_truth = get_final_ground_truth(ground_truths)

    # Only analyze LK trajectories where final ground truth is 0
    if final_ground_truth == 0:
        lk_trajectory_count += 1

        # Analyze correct LK predictions
        correct_metric_values = analyze_lk_correct_predictions(
            model_preds, ground_truths, reliability, ratio, absolute)

        for key in lk_correct_metrics:
            lk_correct_metrics[key].extend(correct_metric_values[key])

        # Analyze false positives in LK trajectories
        fp_metric_values = analyze_lk_false_positives(
            model_preds, ground_truths, reliability, ratio, absolute)

        if len(fp_metric_values['reliability_scores']) > 0:
            lk_fp_trajectory_count += 1
            for key in lk_fp_metrics:
                lk_fp_metrics[key].extend(fp_metric_values[key])

# Convert to numpy arrays
for key in lk_fp_metrics:
    lk_fp_metrics[key] = np.array(lk_fp_metrics[key])

for key in lk_correct_metrics:
    lk_correct_metrics[key] = np.array(lk_correct_metrics[key])

# Create distribution plots for LK trajectories - reliability scores
plt.figure(figsize=(12, 6))

# Plot LK correct reliability scores
plt.subplot(121)
if len(lk_correct_metrics['reliability_scores']) > 0:
    kde_correct = gaussian_kde(lk_correct_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_correct(x_range), 'g-', linewidth=2)
    plt.fill_between(x_range, kde_correct(x_range), alpha=0.3, color='green')
plt.title('Correct LK Prediction Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Plot LK false positive reliability scores
plt.subplot(122)
if len(lk_fp_metrics['reliability_scores']) > 0:
    kde_fp = gaussian_kde(lk_fp_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
    plt.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='red')
plt.title('False Positive in LK Trajectories')
plt.xlabel('Reliability Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics for LK analysis
print("\nLK Trajectory Analysis Statistics:")
print(f"Total LK trajectories analyzed: {lk_trajectory_count}")
print(f"LK trajectories with false positives: {lk_fp_trajectory_count}")

if len(lk_fp_metrics['reliability_scores']) > 0:
    print("\nLK False Positive Statistics:")
    print(f"Number of samples: {len(lk_fp_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(lk_fp_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(lk_fp_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(lk_fp_metrics['reliability_scores']):.4f}")

if len(lk_correct_metrics['reliability_scores']) > 0:
    print("\nLK Correct Prediction Statistics:")
    print(f"Number of samples: {len(lk_correct_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(lk_correct_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(lk_correct_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(lk_correct_metrics['reliability_scores']):.4f}")

# Create comparative boxplot for reliability scores
plt.figure(figsize=(10, 6))
box_data = []
labels = []

if len(lk_correct_metrics['reliability_scores']) > 0:
    box_data.append(lk_correct_metrics['reliability_scores'])
    labels.append('LK Correct')

if len(lk_fp_metrics['reliability_scores']) > 0:
    box_data.append(lk_fp_metrics['reliability_scores'])
    labels.append('LK False Positives')

if box_data:
    plt.boxplot(box_data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Reliability Score')
    plt.title('Comparison of Reliability Scores in LK Trajectories')
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

# If the new metrics exist, create similar plots for them
if len(ratio_scores_all) > 0 and len(absolute_scores_all) > 0:
    # Create distribution plots for ratio scores
    plt.figure(figsize=(12, 6))

    # Plot LK correct ratio scores
    plt.subplot(121)
    if len(lk_correct_metrics['ratio_scores']) > 0:
        kde_correct = gaussian_kde(lk_correct_metrics['ratio_scores'])
        x_range = np.linspace(0, 1, 200)
        plt.plot(x_range, kde_correct(x_range), 'g-', linewidth=2)
        plt.fill_between(x_range, kde_correct(x_range), alpha=0.3, color='green')
    plt.title('Correct LK Prediction Ratio Scores')
    plt.xlabel('Ratio Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Plot LK false positive ratio scores
    plt.subplot(122)
    if len(lk_fp_metrics['ratio_scores']) > 0:
        kde_fp = gaussian_kde(lk_fp_metrics['ratio_scores'])
        x_range = np.linspace(0, 1, 200)
        plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
        plt.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='red')
    plt.title('False Positive Ratio Scores in LK Trajectories')
    plt.xlabel('Ratio Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create distribution plots for absolute scores
    plt.figure(figsize=(12, 6))

    # Plot LK correct absolute scores
    plt.subplot(121)
    if len(lk_correct_metrics['absolute_scores']) > 0:
        kde_correct = gaussian_kde(lk_correct_metrics['absolute_scores'])
        x_range = np.linspace(0, 1, 200)
        plt.plot(x_range, kde_correct(x_range), 'g-', linewidth=2)
        plt.fill_between(x_range, kde_correct(x_range), alpha=0.3, color='green')
    plt.title('Correct LK Prediction Absolute Scores')
    plt.xlabel('Absolute Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Plot LK false positive absolute scores
    plt.subplot(122)
    if len(lk_fp_metrics['absolute_scores']) > 0:
        kde_fp = gaussian_kde(lk_fp_metrics['absolute_scores'])
        x_range = np.linspace(0, 1, 200)
        plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
        plt.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='red')
    plt.title('False Positive Absolute Scores in LK Trajectories')
    plt.xlabel('Absolute Score')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Additional analysis: Length of false positive periods in LK trajectories
fp_lengths = []
for model_preds, ground_truths, _ in zip(model_pred_list_all, ground_truth_each_timestep, reliability_scores_all):
    final_ground_truth = get_final_ground_truth(ground_truths)

    # Only analyze LK trajectories
    if final_ground_truth == 0:
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
    plt.show()

    print("\nFalse Positive Period Lengths in LK Trajectories:")
    print(f"Number of false positive periods: {len(fp_lengths)}")
    print(f"Mean length: {np.mean(fp_lengths):.2f} timesteps")
    print(f"Median length: {np.median(fp_lengths):.2f} timesteps")
    print(f"Max length: {np.max(fp_lengths)} timesteps")