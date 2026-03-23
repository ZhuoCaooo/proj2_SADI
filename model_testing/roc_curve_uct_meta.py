import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle


def plot_roc_with_reliability_thresholds(filename):
    """
    Plot ROC curves with reliability thresholds similar to MC dropout analysis
    """
    # Load the data
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    model_predictions = data['model_predictions']
    ground_truth = data['ground_truth']
    reliability_scores = data['reliability_scores']

    # Prepare data for ROC analysis
    def prepare_data_for_class(class_idx):
        y_true = []
        reliability_values = []
        pred_scores = []

        for preds, truths, rel_scores in zip(model_predictions, ground_truth, reliability_scores):
            final_label = truths[-1]
            if final_label == 0:  # Skip LK trajectories
                continue

            for pred, truth, rel_score in zip(preds, truths, rel_scores):
                # Binary classification for current class
                true_label = 1 if final_label == class_idx else 0
                y_true.append(true_label)
                reliability_values.append(rel_score)

                # Prediction score based on reliability
                pred_score = rel_score if pred == class_idx else 0
                pred_scores.append(pred_score)

        return np.array(y_true), np.array(pred_scores), np.array(reliability_values)

    # Plot settings
    plt.figure(figsize=(15, 6))
    class_names = ['LCL', 'LCR']

    # Create ROC curves for each class
    for idx, class_name in enumerate(['1', '2']):  # 1 for LCL, 2 for LCR
        plt.subplot(1, 2, idx + 1)

        # Get data for current class
        y_true, pred_scores, reliability_values = prepare_data_for_class(int(class_name))

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, pred_scores)
        roc_auc = auc(fpr, tpr)

        # Plot base ROC curve
        plt.plot(fpr, tpr, '-', label=f'Base ROC (AUC={roc_auc:.3f})')

        # Add threshold annotations
        if len(thresholds) > 1:
            # Annotate start point
            plt.annotate(f'Max threshold: {thresholds[0]:.3f}',
                         xy=(fpr[0], tpr[0]),
                         xytext=(0.2, 0.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

            # Annotate end point
            plt.annotate(f'Min threshold: {thresholds[-1]:.3f}',
                         xy=(fpr[-1], tpr[-1]),
                         xytext=(0.6, 0.9),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        # Add threshold-based ROC curves
        thresholds_to_try = [0.2, 0.4, 0.6, 0.8]
        for thresh in thresholds_to_try:
            # Filter predictions based on reliability threshold
            mask = reliability_values >= thresh
            if np.sum(mask) > 0:
                fpr_thresh, tpr_thresh, _ = roc_curve(y_true[mask], pred_scores[mask])
                roc_auc_thresh = auc(fpr_thresh, tpr_thresh)
                plt.plot(fpr_thresh, tpr_thresh, '--',
                         label=f'Rel≥{thresh} (AUC={roc_auc_thresh:.3f})')

        # Plot settings
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {class_names[idx]}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt


# Example usage for different time settings
time_settings = [0.0, 0.5, 1.0, 1.5]

for time in time_settings:
    filename = f'cnn_results_{time:.1f}s.pkl'
    plt = plot_roc_with_reliability_thresholds(filename)
    plt.suptitle(f'ROC Analysis for {time}s Prediction Time')
    plt.show()