"""Analysis of uncertainty scores on trajectories"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc

# Load the data - adjust path to your file
with open('../model_testing_paper_2/cnn_results_density_based_0.0s_new.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Extract all the variables including the new metrics
model_pred_list_all = loaded_data['model_predictions']
reliability_scores_all = loaded_data['reliability_scores']
ground_truth_each_timestep = loaded_data['ground_truth']
probability_outputs = loaded_data['probability_outputs']
ratio_scores_all = loaded_data.get('ratio_scores', [])  # Use get to handle if these keys don't exist
absolute_scores_all = loaded_data.get('absolute_scores', [])


def get_final_behavior(predictions):
    """Get the final behavior (last non-zero prediction)"""
    final_pred = 0
    for pred in reversed(predictions):
        if pred != 0:
            final_pred = pred
            break
    return final_pred


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


def analyze_false_positives(model_preds, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Extract uncertainty metrics around continuous false positive periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores
        ratio_scores: List of corresponding ratio scores (optional)
        absolute_scores: List of corresponding absolute scores (optional)

    Returns:
        Dictionary of uncertainty metrics from windows around FP periods
    """
    fp_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    sequence_length = len(model_preds)
    window_size = int(10)

    # Find continuous periods of false positives
    fp_periods = find_continuous_fp_periods(model_preds)

    # Extract windows around each FP period
    for start_fp, end_fp in fp_periods:
        # Calculate window boundaries
        window_start = max(0, start_fp- window_size)
        window_end = min(sequence_length, end_fp + window_size + 1)

        # Add reliability scores for this window
        fp_metrics['reliability_scores'].extend(reliability_scores[window_start:window_end])

        # Add other metrics if available
        if ratio_scores is not None:
            fp_metrics['ratio_scores'].extend(ratio_scores[window_start:window_end])
        if absolute_scores is not None:
            fp_metrics['absolute_scores'].extend(absolute_scores[window_start:window_end])

    return fp_metrics


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


def analyze_false_negatives(model_preds, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Extract uncertainty metrics around continuous false negative periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores
        ratio_scores: List of corresponding ratio scores (optional)
        absolute_scores: List of corresponding absolute scores (optional)

    Returns:
        Dictionary of uncertainty metrics from windows around FN periods
    """
    fn_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    sequence_length = len(model_preds)
    window_size = int(0)  # Original window size
    half_window = window_size // 2

    # Find continuous periods of false negatives (starting from t=100)
    fn_periods = find_continuous_fn_periods(model_preds, start_time=100)

    # Extract windows around each FN period
    for start_fn, end_fn in fn_periods:
        # Calculate window boundaries
        window_start = max(0, start_fn - half_window)
        window_end = min(sequence_length, end_fn + half_window+ 1)

        # Add reliability scores for this window
        fn_metrics['reliability_scores'].extend(reliability_scores[window_start:window_end])

        # Add other metrics if available
        if ratio_scores is not None:
            fn_metrics['ratio_scores'].extend(ratio_scores[window_start:window_end])
        if absolute_scores is not None:
            fn_metrics['absolute_scores'].extend(absolute_scores[window_start:window_end])

    return fn_metrics


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


def analyze_transitions(model_preds, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Analyze uncertainty metrics during transition periods

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores
        ratio_scores: List of corresponding ratio scores (optional)
        absolute_scores: List of corresponding absolute scores (optional)

    Returns:
        Dictionary of uncertainty metrics from windows around transitions
    """
    transition_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    window_size = 10  # n steps to be analyzed
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
            transition_metrics['reliability_scores'].extend(reliability_scores[start_idx:end_idx])

            # Add other metrics if available
            if ratio_scores is not None:
                transition_metrics['ratio_scores'].extend(ratio_scores[start_idx:end_idx])
            if absolute_scores is not None:
                transition_metrics['absolute_scores'].extend(absolute_scores[start_idx:end_idx])

    return transition_metrics


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


def analyze_correct_predictions(model_preds, reliability_scores, ratio_scores=None, absolute_scores=None):
    """Extract uncertainty metrics for continuous periods of correct predictions

    Args:
        model_preds: List of model predictions
        reliability_scores: List of corresponding reliability scores
        ratio_scores: List of corresponding ratio scores (optional)
        absolute_scores: List of corresponding absolute scores (optional)

    Returns:
        Dictionary of uncertainty metrics from correct prediction periods
    """
    correct_metrics = {
        'reliability_scores': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    sequence_length = len(model_preds)
    final_behavior = get_final_behavior(model_preds)

    if final_behavior in [1, 2]:  # If it's a lane change sequence
        # Calculate start index based on both percentage and minimum steps
        percentage_start = int(sequence_length * 0.25)
        min_steps_start = max(0, sequence_length - 50)
        start_idx = max(percentage_start, min_steps_start)

        # Find continuous periods of correct predictions
        tp_periods = find_continuous_tp_periods(model_preds, start_idx)

        # Extract uncertainty metrics for each period
        for start_tp, end_tp in tp_periods:
            correct_metrics['reliability_scores'].extend(reliability_scores[start_tp:end_tp + 1])

            # Add other metrics if available
            if ratio_scores is not None:
                correct_metrics['ratio_scores'].extend(ratio_scores[start_tp:end_tp + 1])
            if absolute_scores is not None:
                correct_metrics['absolute_scores'].extend(absolute_scores[start_tp:end_tp + 1])

    return correct_metrics


# Initialize collectors for all types of metrics
all_fp_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

all_fn_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

all_transition_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

all_correct_metrics = {
    'reliability_scores': [],
    'ratio_scores': [],
    'absolute_scores': []
}

# Process each trajectory
for idx, (model_preds, reliability) in enumerate(zip(model_pred_list_all, reliability_scores_all)):
    # Get additional metrics if available
    ratio = ratio_scores_all[idx] if idx < len(ratio_scores_all) else None
    absolute = absolute_scores_all[idx] if idx < len(absolute_scores_all) else None

    # Analyze false positives
    fp_metrics = analyze_false_positives(model_preds, reliability, ratio, absolute)
    for key in all_fp_metrics:
        all_fp_metrics[key].extend(fp_metrics[key])

    # Analyze false negatives
    fn_metrics = analyze_false_negatives(model_preds, reliability, ratio, absolute)
    for key in all_fn_metrics:
        all_fn_metrics[key].extend(fn_metrics[key])

    # Analyze transitions
    transition_metrics = analyze_transitions(model_preds, reliability, ratio, absolute)
    for key in all_transition_metrics:
        all_transition_metrics[key].extend(transition_metrics[key])

    # Analyze correct predictions
    correct_metrics = analyze_correct_predictions(model_preds, reliability, ratio, absolute)
    for key in all_correct_metrics:
        all_correct_metrics[key].extend(correct_metrics[key])

# Convert to numpy arrays
for key in all_fp_metrics:
    all_fp_metrics[key] = np.array(all_fp_metrics[key])

for key in all_fn_metrics:
    all_fn_metrics[key] = np.array(all_fn_metrics[key])

for key in all_transition_metrics:
    all_transition_metrics[key] = np.array(all_transition_metrics[key])

for key in all_correct_metrics:
    all_correct_metrics[key] = np.array(all_correct_metrics[key])


# Create distribution plots for reliability scores with improved academic style
plt.figure(figsize=(30, 20), dpi=500)  # Higher DPI for publication quality

# Define a consistent color palette with less saturated colors
colors = {
    'fp': '#E63946',  # less saturated red
    'fn': '#457B9D',  # less saturated blue
    'trans': '#2A9D8F',  # less saturated green
    'correct': '#7B2CBF'  # less saturated purple
}

# Set more appropriate font sizes for academic publication
title_size = 7
axis_label_size = 7
tick_size = 10
font_family = 'Arial'

plt.rcParams.update({
    'font.family': font_family,
    'font.size': tick_size,
    'axes.titlesize': title_size,
    'axes.labelsize': axis_label_size,
    'xtick.labelsize': tick_size,
    'ytick.labelsize': tick_size
})

# Create subplots with proper spacing
fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=300)
plt.subplots_adjust(wspace=0.6, hspace=0.6)

# Plot FP reliability scores with improved style
if len(all_fp_metrics['reliability_scores']) > 0:
    kde_fp = gaussian_kde(all_fp_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 300)  # More points for smoother curve
    axes[0].plot(x_range, kde_fp(x_range), '-', color=colors['fp'], linewidth=2)
    axes[0].fill_between(x_range, kde_fp(x_range), alpha=0.3, color=colors['fp'])
axes[0].set_title('False Positive\nReliability Scores')
axes[0].set_xlabel('Reliability Score')
axes[0].set_ylabel('Density')
axes[0].set_xlim(0, 1)
axes[0].grid(True, linestyle='--', alpha=0.3)

# Plot FN reliability scores with improved style
if len(all_fn_metrics['reliability_scores']) > 0:
    kde_fn = gaussian_kde(all_fn_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 300)
    axes[1].plot(x_range, kde_fn(x_range), '-', color=colors['fn'], linewidth=2)
    axes[1].fill_between(x_range, kde_fn(x_range), alpha=0.3, color=colors['fn'])
axes[1].set_title('False Negative\nReliability Scores')
axes[1].set_xlabel('Reliability Score')
#axes[1].set_ylabel('Density')
axes[1].set_xlim(0, 1)
axes[1].grid(True, linestyle='--', alpha=0.3)

# Plot transition reliability scores with improved style
if len(all_transition_metrics['reliability_scores']) > 0:
    kde_trans = gaussian_kde(all_transition_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 300)
    axes[2].plot(x_range, kde_trans(x_range), '-', color=colors['trans'], linewidth=2)
    axes[2].fill_between(x_range, kde_trans(x_range), alpha=0.3, color=colors['trans'])
axes[2].set_title('Transition Period\nReliability Scores')
axes[2].set_xlabel('Reliability Score')
#axes[2].set_ylabel('Density')
axes[2].set_xlim(0, 1)
axes[2].grid(True, linestyle='--', alpha=0.3)

# Plot correct prediction reliability scores with improved style
if len(all_correct_metrics['reliability_scores']) > 0:
    kde_correct = gaussian_kde(all_correct_metrics['reliability_scores'])
    x_range = np.linspace(0, 1, 300)
    axes[3].plot(x_range, kde_correct(x_range), '-', color=colors['correct'], linewidth=2)
    axes[3].fill_between(x_range, kde_correct(x_range), alpha=0.3, color=colors['correct'])
axes[3].set_title('Correct Prediction\nReliability Scores')
axes[3].set_xlabel('Reliability Score')
#axes[3].set_ylabel('Density')
axes[3].set_xlim(0, 1)
axes[3].grid(True, linestyle='--', alpha=0.3)

# Set layout with more generous spacing
plt.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.15, wspace=0.25)

# Show the figure
plt.show()

# Print statistics for reliability scores
print("\nFalse Positive Statistics:")
if len(all_fp_metrics['reliability_scores']) > 0:
    print(f"Number of samples: {len(all_fp_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(all_fp_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(all_fp_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(all_fp_metrics['reliability_scores']):.4f}")

print("\nFalse Negative Statistics:")
if len(all_fn_metrics['reliability_scores']) > 0:
    print(f"Number of samples: {len(all_fn_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(all_fn_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(all_fn_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(all_fn_metrics['reliability_scores']):.4f}")

print("\nTransition Period Statistics:")
if len(all_transition_metrics['reliability_scores']) > 0:
    print(f"Number of samples: {len(all_transition_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(all_transition_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(all_transition_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(all_transition_metrics['reliability_scores']):.4f}")

print("\nCorrect Prediction Statistics:")
if len(all_correct_metrics['reliability_scores']) > 0:
    print(f"Number of samples: {len(all_correct_metrics['reliability_scores'])}")
    print(f"Mean reliability score: {np.mean(all_correct_metrics['reliability_scores']):.4f}")
    print(f"Median reliability score: {np.median(all_correct_metrics['reliability_scores']):.4f}")
    print(f"Standard deviation: {np.std(all_correct_metrics['reliability_scores']):.4f}")





