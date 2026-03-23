import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_reliability_roc_with_thresholds(model_pred_list_all, reliability_scores_all, ground_truth_each_timestep):
    """
    Plot ROC curves for reliability scores for each class (LCL and LCR)
    with threshold annotations.
    """
    # Collect all predictions, ground truths, and reliability scores
    all_preds = []
    all_true = []
    all_reliability = []

    for model_preds, reliability, ground_truth in zip(model_pred_list_all, reliability_scores_all,
                                                      ground_truth_each_timestep):
        for t in range(len(model_preds)):
            # Skip early predictions if needed
            if t < 24:  # Adjust if needed
                continue

            # Add this prediction to our dataset
            all_preds.append(model_preds[t])
            all_true.append(ground_truth[t])
            all_reliability.append(reliability[t])

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_reliability = np.array(all_reliability)

    # Create correct/incorrect binary indicator (1 if prediction is wrong)
    is_incorrect = (all_preds != all_true).astype(int)

    # Calculate ROC curve for detecting errors using reliability scores
    # Note: Lower reliability should predict errors, so we use 1-reliability
    fpr, tpr, thresholds = roc_curve(is_incorrect, 1 - all_reliability)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Reliability ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line

    # Add threshold annotations
    # Choose a few key thresholds to annotate
    threshold_indices = [0, len(thresholds) // 4, len(thresholds) // 2, 3 * len(thresholds) // 4, len(thresholds) - 1]
    for idx in threshold_indices:
        if idx < len(thresholds):
            plt.annotate(f'{1 - thresholds[idx]:.2f}',
                         xy=(fpr[idx], tpr[idx]),
                         xytext=(fpr[idx] + 0.05, tpr[idx] + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Reliability Score as Error Predictor')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Create class-specific ROC curves (LCL and LCR)
    plt.figure(figsize=(12, 5))

    class_names = ['LK', 'LCL', 'LCR']
    colors = ['blue', 'green', 'red']
    reliability_thresholds = [0.2, 0.4, 0.6, 0.8]

    # Create subplots for each lane change class (LCL and LCR)
    for i in range(1, 3):  # Classes 1 and 2 (LCL and LCR)
        plt.subplot(1, 2, i)

        # Filter data for this class
        class_mask = all_true == i

        # Skip if not enough samples
        if np.sum(class_mask) < 10:
            plt.text(0.5, 0.5, f"Not enough samples for {class_names[i]}",
                     ha='center', va='center')
            continue

        # Binary indicator: 1 if predicted correctly for this class
        y_true_binary = (all_preds[class_mask] == i).astype(int)

        # Reliability scores for this class
        reliability_class = all_reliability[class_mask]

        # Calculate base ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, reliability_class)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, '-', color=colors[i],
                 label=f'All (AUC = {roc_auc:.3f})')

        # Plot ROC curves for different reliability thresholds
        for threshold in reliability_thresholds:
            # Filter by reliability threshold
            threshold_mask = reliability_class >= threshold

            # Skip if not enough samples
            if np.sum(threshold_mask) < 10:
                continue

            # Calculate ROC curve for this threshold
            fpr_thresh, tpr_thresh, _ = roc_curve(
                y_true_binary[threshold_mask],
                reliability_class[threshold_mask]
            )
            roc_auc_thresh = auc(fpr_thresh, tpr_thresh)

            plt.plot(fpr_thresh, tpr_thresh, '--',
                     label=f'Rel ≥ {threshold:.1f} (AUC = {roc_auc_thresh:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {class_names[i]} Predictions')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nReliability Score Statistics for Error Prediction:")
    correct_mask = all_preds == all_true
    incorrect_mask = ~correct_mask

    print(f"Mean reliability (correct predictions): {np.mean(all_reliability[correct_mask]):.4f}")
    print(f"Mean reliability (incorrect predictions): {np.mean(all_reliability[incorrect_mask]):.4f}")

    # Calculate optimal threshold based on Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = 1 - thresholds[optimal_idx]
    print(f"Optimal reliability threshold: {optimal_threshold:.4f}")
    print(f"At optimal threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")


# Load your data
# Note: Replace the following with code to load your actual data if running this script separately
def load_example_data():
    # Create dummy data if running this separately
    import pickle

    try:
        # Try to load the actual data file
        with open('cnn_results_0.0s_new.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        model_pred_list_all = loaded_data['model_predictions']
        reliability_scores_all = loaded_data['reliability_scores']
        ground_truth_each_timestep = loaded_data['ground_truth']

    except FileNotFoundError:
        # Create dummy data if the file doesn't exist
        print("Data file not found, using dummy data for demonstration")
        num_trajectories = 10
        traj_length = 150

        model_pred_list_all = []
        reliability_scores_all = []
        ground_truth_each_timestep = []

        for _ in range(num_trajectories):
            # Generate random predictions (0, 1, 2)
            preds = np.random.randint(0, 3, traj_length)

            # Generate some ground truth (0, 1, 2) with 80% accuracy
            gt = preds.copy()
            error_indices = np.random.choice(traj_length, int(traj_length * 0.2), replace=False)
            for idx in error_indices:
                gt[idx] = np.random.choice([c for c in [0, 1, 2] if c != preds[idx]])

            # Generate reliability scores correlating with correctness
            reliability = np.random.uniform(0.6, 0.9, traj_length)
            for idx in error_indices:
                reliability[idx] = np.random.uniform(0.1, 0.4)

            model_pred_list_all.append(preds)
            reliability_scores_all.append(reliability)
            ground_truth_each_timestep.append(gt)

    return model_pred_list_all, reliability_scores_all, ground_truth_each_timestep


# Example usage
model_pred_list_all, reliability_scores_all, ground_truth_each_timestep = load_example_data()
plot_reliability_roc_with_thresholds(model_pred_list_all, reliability_scores_all, ground_truth_each_timestep)


