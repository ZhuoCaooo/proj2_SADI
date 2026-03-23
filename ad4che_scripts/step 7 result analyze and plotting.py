import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from improved_plot_functions import plot_threshold_metrics_improved, plot_threshold_metrics_color_version


def load_ad4che_results(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_ad4che_combined(data):
    all_preds = []
    all_gt = []
    all_reliability = []
    all_categories = []

    for preds, gt, reliability in zip(data['model_predictions'], data['ground_truth'], data['reliability_scores']):
        # Determine ground truth final behavior for this trajectory
        final_behavior = 0
        for val in reversed(gt):
            if val != 0:
                final_behavior = val
                break

        for t in range(len(preds)):
            p, g, r = preds[t], gt[t], reliability[t]

            # Classification Logic consistent with HighD script
            if p != 0 and p == final_behavior:
                cat = "TP"
            elif p != 0 and p != final_behavior:
                cat = "FP"
            elif p == 0 and final_behavior != 0:
                cat = "FN"
            elif p == 0 and final_behavior == 0:
                cat = "TN"
            else:
                cat = "Other"

            all_preds.append(p)
            all_gt.append(g)
            all_reliability.append(r)
            all_categories.append(cat)

    return pd.DataFrame({
        'reliability': all_reliability,
        'category': all_categories
    })


def main():
    # 1. Load the results you just generated
    results_file = 'ad4che_uncertainty_results.pkl'
    raw_data = load_ad4che_results(results_file)

    # 2. Prepare the dataset
    df = prepare_ad4che_combined(raw_data)

    # 3. Analyze Thresholds (using same 0 to 0.95 range as HighD)
    thresholds = np.linspace(0, 0.95, 20)
    analysis_results = []

    total_samples = len(df)
    for ts in thresholds:
        mask = df['reliability'] >= ts
        retained_df = df[mask]

        row = {
            'threshold': ts,
            'retention_rate': len(retained_df) / total_samples if total_samples > 0 else 0
        }

        # EDIT: Removed 'Other' from this list to exclude it from the plot data
        for cat in ['TP', 'FP', 'FN', 'TN']:
            cat_total = len(df[df['category'] == cat])
            cat_retained = len(retained_df[retained_df['category'] == cat])
            row[f'retention_{cat}'] = cat_retained / cat_total if cat_total > 0 else 0

        analysis_results.append(row)

    threshold_df = pd.DataFrame(analysis_results)

    # 4. Plot using your existing style functions
    # Using "AD4CHE" as the horizon label for the title
    plot_threshold_metrics_improved(threshold_df, "AD4CHE_Final", "ad4che")
    plot_threshold_metrics_color_version(threshold_df, "AD4CHE_Final", "ad4che")

    print("Analysis complete. Check for 'Retention_analysis_AD4CHE.png'")


if __name__ == "__main__":
    main()