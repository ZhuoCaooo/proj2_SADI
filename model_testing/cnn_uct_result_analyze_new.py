"""this is the analysis of uncertainty scores on trajectories"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Load the data

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


with open('cnn_results_density_based_0.0s.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Extract all the variables including probability outputs
model_pred_list_all = loaded_data['model_predictions']
reliability_scores_all = loaded_data['reliability_scores']
ground_truth_each_timestep = loaded_data['ground_truth']
probability_outputs = loaded_data['probability_outputs']





def find_continuous_fp_periods(model_preds):
    """Find continuous periods of false positive predictions

    Args:
        model_preds: List of model predictions

    Returns:
        List of tuples (start_idx, end_idx) for each continuous FP period
    """
    final_behavior = get_final_behavior(model_preds)
    fp_periods = []

    start_idx = None
    for t in range(len(model_preds)):
        is_fp = model_preds[t] in [1, 2] and model_preds[t] != final_behavior

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

def get_final_behavior(predictions):
    """Get the final behavior (last non-zero prediction)"""
    final_pred = 0
    for pred in reversed(predictions):
        if pred != 0:
            final_pred = pred
            break
    return final_pred


def analyze_false_positives(model_preds, reliability_scores):
    """Extract reliability scores around continuous false positive periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from windows around FP periods
    """
    fp_reliability_scores = []
    sequence_length = len(model_preds)
    window_size = int(25 * 1.0)  # 1.2 times the original window
    half_window = window_size // 2

    # Find continuous periods of false positives
    fp_periods = find_continuous_fp_periods(model_preds)

    # Extract windows around each FP period
    for start_fp, end_fp in fp_periods:
        # Calculate window boundaries
        # Start window before FP period starts
        window_start = max(0, start_fp - half_window)
        # End window after FP period ends
        window_end = min(sequence_length, end_fp + half_window + 1)

        # Add reliability scores for this window
        fp_reliability_scores.extend(reliability_scores[window_start:window_end])

    return fp_reliability_scores


def find_continuous_fn_periods(model_preds, start_time):
    """Find continuous periods of false negative predictions

    Args:
        model_preds: List of model predictions
        start_time: Minimum timestep to start analyzing (default: 100)

    Returns:
        List of tuples (start_idx, end_idx) for each continuous FN period
    """
    final_behavior = get_final_behavior(model_preds)
    fn_periods = []

    # Only analyze if trajectory ends with a lane change
    if final_behavior not in [1, 2]:
        return fn_periods

    start_idx = None
    for t in range(start_time, len(model_preds)):
        is_fn = model_preds[t] == 0  # False negative when prediction is LK (0)

        # Start of a new FN period
        if is_fn and start_idx is None:
            start_idx = t
        # End of current FN period
        elif not is_fn and start_idx is not None:
            fn_periods.append((start_idx, t - 1))
            start_idx = None

    # Handle case where FN period ends at sequence end
    if start_idx is not None:
        fn_periods.append((start_idx, len(model_preds) - 1))

    return fn_periods


def analyze_false_negatives(model_preds, reliability_scores):
    """Extract reliability scores around continuous false negative periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from windows around FN periods
    """
    fn_reliability_scores = []
    sequence_length = len(model_preds)
    window_size = int(5)  # Original window size
    half_window = window_size // 2

    # Find continuous periods of false negatives (starting from t=100)
    fn_periods = find_continuous_fn_periods(model_preds, start_time=125)

    # Extract windows around each FN period
    for start_fn, end_fn in fn_periods:
        # Calculate window boundaries
        window_start = max(0, start_fn - half_window)
        window_end = min(sequence_length, end_fn + half_window + 1)

        # Add reliability scores for this window
        fn_reliability_scores.extend(reliability_scores[window_start:window_end])

    return fn_reliability_scores


def find_last_transition(predictions, final_behavior):
    """Find the last transition point to correct prediction"""
    last_transition = None
    for i in range(len(predictions) - 1):
        if predictions[i] != final_behavior and predictions[i + 1] == final_behavior:
            last_transition = i + 1
    return last_transition


def find_all_transitions(predictions, final_behavior):
    """Find all transition points to and from final behavior

    Args:
        predictions: List of model predictions
        final_behavior: The final behavior class

    Returns:
        List of transition points (indices where prediction changes)
    """
    transitions = []
    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:
            transitions.append((i, i + 1, predictions[i], predictions[i + 1]))
    return transitions


def find_continuous_tp_periods(model_preds, min_start_idx):
    """Find continuous periods of true positive predictions

    Args:
        model_preds: List of model predictions
        min_start_idx: Minimum index to start analyzing

    Returns:
        List of tuples (start_idx, end_idx) for each continuous TP period
    """
    final_behavior = get_final_behavior(model_preds)
    tp_periods = []

    if final_behavior not in [1, 2]:
        return tp_periods

    start_idx = None
    for t in range(min_start_idx, len(model_preds)):
        is_tp = model_preds[t] == final_behavior

        # Start of a new TP period
        if is_tp and start_idx is None:
            start_idx = t
        # End of current TP period
        elif not is_tp and start_idx is not None:
            tp_periods.append((start_idx, t - 1))
            start_idx = None

    # Handle case where TP period ends at sequence end
    if start_idx is not None:
        tp_periods.append((start_idx, len(model_preds) - 1))

    return tp_periods


def analyze_transitions(model_preds, reliability_scores):
    """Analyze reliability scores during transition periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from windows around transitions
    """
    transition_reliability_scores = []
    window_size = 25  # n steps to be analyzed
    half_window = window_size // 2
    sequence_length = len(model_preds)
    final_behavior = get_final_behavior(model_preds)

    # Only analyze trajectories that end with a lane change
    if final_behavior in [1, 2]:
        # Get all transitions
        transitions = find_all_transitions(model_preds, final_behavior)

        # Find the last transition to final behavior
        last_transition = None
        for t_start, t_end, behavior_from, behavior_to in reversed(transitions):
            if behavior_to == final_behavior:
                last_transition = t_end
                break

        if last_transition is not None:
            # Extract window around last transition
            start_idx = max(0, last_transition - half_window)
            end_idx = min(sequence_length, last_transition + half_window)
            transition_reliability_scores.extend(reliability_scores[start_idx:end_idx])

    return transition_reliability_scores


def analyze_correct_predictions(model_preds, reliability_scores):
    """Extract reliability scores for continuous periods of correct predictions

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores

    Returns:
        List of reliability scores from correct prediction periods
    """
    correct_reliability_scores = []
    sequence_length = len(model_preds)
    final_behavior = get_final_behavior(model_preds)

    if final_behavior in [1, 2]:  # If it's a lane change sequence
        # Calculate start index based on both percentage and minimum steps
        percentage_start = int(sequence_length * 0.25)
        min_steps_start = max(0, sequence_length - 50)
        start_idx = max(percentage_start, min_steps_start)

        # Find continuous periods of correct predictions
        tp_periods = find_continuous_tp_periods(model_preds, start_idx)

        # Extract reliability scores for each period
        for start_tp, end_tp in tp_periods:
            correct_reliability_scores.extend(reliability_scores[start_tp:end_tp + 1])

    return correct_reliability_scores





# Initialize collectors for all types of reliability scores
all_fp_scores = []
all_fn_scores = []
all_transition_scores = []
all_correct_scores = []

# Process each trajectory
for model_preds, reliability in zip(model_pred_list_all, reliability_scores_all):
    # Analyze false positives
    fp_scores = analyze_false_positives(model_preds, reliability)
    all_fp_scores.extend(fp_scores)

    # Analyze false negatives
    fn_scores = analyze_false_negatives(model_preds, reliability)
    all_fn_scores.extend(fn_scores)

    # Analyze transitions
    transition_scores = analyze_transitions(model_preds, reliability)
    all_transition_scores.extend(transition_scores)

    # Analyze correct predictions
    correct_scores = analyze_correct_predictions(model_preds, reliability)
    all_correct_scores.extend(correct_scores)

# Convert to numpy arrays
all_fp_scores = np.array(all_fp_scores)
all_fn_scores = np.array(all_fn_scores)
all_transition_scores = np.array(all_transition_scores)
all_correct_scores = np.array(all_correct_scores)

# Create distribution plots
plt.figure(figsize=(20, 5))

# Plot FP reliability scores
plt.subplot(141)
if len(all_fp_scores) > 0:
    kde_fp = gaussian_kde(all_fp_scores)
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
    plt.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='red')
plt.title('False Positive Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Plot FN reliability scores
plt.subplot(142)
if len(all_fn_scores) > 0:
    kde_fn = gaussian_kde(all_fn_scores)
    plt.plot(x_range, kde_fn(x_range), 'b-', linewidth=2)
    plt.fill_between(x_range, kde_fn(x_range), alpha=0.3, color='blue')
plt.title('False Negative Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Plot transition reliability scores
plt.subplot(143)
if len(all_transition_scores) > 0:
    kde_trans = gaussian_kde(all_transition_scores)
    plt.plot(x_range, kde_trans(x_range), 'g-', linewidth=2)
    plt.fill_between(x_range, kde_trans(x_range), alpha=0.3, color='green')
plt.title('Transition Period Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Plot correct prediction reliability scores
plt.subplot(144)
if len(all_correct_scores) > 0:
    kde_correct = gaussian_kde(all_correct_scores)
    plt.plot(x_range, kde_correct(x_range), 'purple', linewidth=2)
    plt.fill_between(x_range, kde_correct(x_range), alpha=0.3, color='purple')
plt.title('Correct Prediction Reliability Scores')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Print statistics
print("\nFalse Positive Statistics:")
if len(all_fp_scores) > 0:
    print(f"Number of samples: {len(all_fp_scores)}")
    print(f"Mean reliability score: {np.mean(all_fp_scores):.4f}")
    print(f"Median reliability score: {np.median(all_fp_scores):.4f}")
    print(f"Standard deviation: {np.std(all_fp_scores):.4f}")

print("\nFalse Negative Statistics:")
if len(all_fn_scores) > 0:
    print(f"Number of samples: {len(all_fn_scores)}")
    print(f"Mean reliability score: {np.mean(all_fn_scores):.4f}")
    print(f"Median reliability score: {np.median(all_fn_scores):.4f}")
    print(f"Standard deviation: {np.std(all_fn_scores):.4f}")

print("\nTransition Period Statistics:")
if len(all_transition_scores) > 0:
    print(f"Number of samples: {len(all_transition_scores)}")
    print(f"Mean reliability score: {np.mean(all_transition_scores):.4f}")
    print(f"Median reliability score: {np.median(all_transition_scores):.4f}")
    print(f"Standard deviation: {np.std(all_transition_scores):.4f}")

print("\nCorrect Prediction Statistics:")
if len(all_correct_scores) > 0:
    print(f"Number of samples: {len(all_correct_scores)}")
    print(f"Mean reliability score: {np.mean(all_correct_scores):.4f}")
    print(f"Median reliability score: {np.median(all_correct_scores):.4f}")
    print(f"Standard deviation: {np.std(all_correct_scores):.4f}")


# After your existing KDE plots, add this code to examine the distribution more closely
plt.figure(figsize=(20, 10))

# Examine False Positive reliability scores with histogram
plt.subplot(241)
if len(all_fp_scores) > 0:
    plt.hist(all_fp_scores, bins=50, alpha=0.6, density=True, color='red')
    kde_fp = gaussian_kde(all_fp_scores)
    x_range = np.linspace(0, 1, 200)
    plt.plot(x_range, kde_fp(x_range), 'r-', linewidth=2)
plt.title('False Positive Scores (Histogram + KDE)')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Examine False Negative reliability scores
plt.subplot(242)
if len(all_fn_scores) > 0:
    plt.hist(all_fn_scores, bins=50, alpha=0.6, density=True, color='blue')
    kde_fn = gaussian_kde(all_fn_scores)
    plt.plot(x_range, kde_fn(x_range), 'b-', linewidth=2)
plt.title('False Negative Scores (Histogram + KDE)')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Examine Transition reliability scores
plt.subplot(243)
if len(all_transition_scores) > 0:
    plt.hist(all_transition_scores, bins=50, alpha=0.6, density=True, color='green')
    kde_trans = gaussian_kde(all_transition_scores)
    plt.plot(x_range, kde_trans(x_range), 'g-', linewidth=2)
plt.title('Transition Scores (Histogram + KDE)')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

# Examine Correct prediction reliability scores
plt.subplot(244)
if len(all_correct_scores) > 0:
    plt.hist(all_correct_scores, bins=50, alpha=0.6, density=True, color='purple')
    kde_correct = gaussian_kde(all_correct_scores)
    plt.plot(x_range, kde_correct(x_range), 'purple', linewidth=2)
plt.title('Correct Prediction Scores (Histogram + KDE)')
plt.xlabel('Reliability Score')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Create CDF plots
plt.figure(figsize=(12, 8))

# Plot CDFs for each category
x_values = np.linspace(0, 1, 1000)

# Function to compute CDF
def compute_cdf(data, x_values):
    return np.array([np.mean(data <= x) for x in x_values])

# Plot all CDFs on the same plot
if len(all_fp_scores) > 0:
    plt.plot(x_values, compute_cdf(all_fp_scores, x_values), 'r-',
             linewidth=2, label='False Positives')
if len(all_fn_scores) > 0:
    plt.plot(x_values, compute_cdf(all_fn_scores, x_values), 'b-',
             linewidth=2, label='False Negatives')
if len(all_transition_scores) > 0:
    plt.plot(x_values, compute_cdf(all_transition_scores, x_values), 'g-',
             linewidth=2, label='Transitions')
if len(all_correct_scores) > 0:
    plt.plot(x_values, compute_cdf(all_correct_scores, x_values), 'purple',
             linewidth=2, label='Correct Predictions')

# Add grid and labels
plt.grid(True, alpha=0.3)
plt.title('Cumulative Distribution Function (CDF) of Reliability Scores', fontsize=14)
plt.xlabel('Reliability Score', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()



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
y_true_onehot = np.array(y_true_onehot)
pred_probs = np.array(pred_probs)
reliabilities = np.array(reliabilities)

# Create separate figures for LCL and LCR
class_names = ['LCL', 'LCR']
colors = ['g', 'r', 'c', 'm']
reliability_thresholds = [0.2, 0.4, 0.6, 0.8]

for i in range(2):
    plt.figure(figsize=(8, 6))

    # Plot base ROC curve using probabilities
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', label=f'All samples (AUC = {roc_auc:.3f})')

    # Plot ROC curves for different reliability thresholds
    for threshold, color in zip(reliability_thresholds, colors):
        mask = reliabilities >= threshold
        if np.sum(mask) > 0:
            fpr_thresh, tpr_thresh, _ = roc_curve(y_true_onehot[mask, i],
                                                  pred_probs[mask, i])
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
            plt.plot(fpr_thresh, tpr_thresh, color=color, linestyle='--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {class_names[i]} using Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Print statistics
print("\nNumber of samples per class:")
for i in range(2):
    print(f"Class {i + 1}: {np.sum(y_true_onehot[:, i])}")

print("\nProbability statistics:")
for i in range(2):
    probs = pred_probs[:, i]
    print(f"\nClass {i + 1}:")
    print(f"Mean probability: {np.mean(probs):.4f}")
    print(f"Max probability: {np.max(probs):.4f}")
    print(f"Min probability: {np.min(probs):.4f}")