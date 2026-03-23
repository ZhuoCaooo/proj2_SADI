'''
Step 6: Run Uncertainty Test on AD4CHE Dataset (Final Fix)
- Fix: Overrides DTW function to prevent "Index out of bounds" error.
- Architecture: 2 Conv Layers (Matches Step 3)
- Window Length: 15
'''

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import other necessary functions
from density_based_uncertainty import (
    extract_min_max_densities,
    StateManager,
    improved_assess_prediction_with_density,
    plot_trajectory_with_enhanced_metrics,
    compute_lateral_speed
)

# ============================================================================
# 1. FIXED DTW FUNCTION (Overrides the one in density_based_uncertainty.py)
# ============================================================================
def local_compute_multi_scale_dtw_similarities(X_test_windows, current_time, window_length):
    """
    Fixed version for AD4CHE (8 Features).
    Does NOT delete columns, keeping x, y, vx, vy, etc.
    """
    X_test_windows = np.array(X_test_windows)

    # --- FIX: Removed the np.delete() line that caused the crash ---
    # We keep all 8 features (x, y, w, h, vx, vy, ax, ay) for similarity calc

    # Define window sizes
    short_scale = 15
    medium_scale = 30
    long_scale = 60

    # Define weights
    short_weight = 0.2
    medium_weight = 0.5
    long_weight = 0.3

    padding_size = max(40, long_scale * 2)
    padded_windows = np.pad(X_test_windows,
                            pad_width=((padding_size, 0), (0, 0), (0, 0)),
                            mode='edge')

    total_length = len(padded_windows)
    dtw_results = {}

    for scale, scale_name, gap in zip(
            [short_scale, medium_scale, long_scale],
            ['short', 'medium', 'long'],
            [5, 10, 20]):

        if current_time < padding_size:
            index_1 = 0
            index_2 = min(current_time + padding_size, total_length - 1)
        else:
            index_1 = current_time
            desired_index_2 = current_time + padding_size
            index_2 = min(desired_index_2, total_length - 1)
            if index_2 - index_1 < gap:
                index_1 = max(0, index_2 - gap)

        if index_1 + scale <= total_length and index_2 + scale <= total_length:
            window_1 = padded_windows[index_1:index_1 + scale]
            window_2 = padded_windows[index_2:index_2 + scale]
            try:
                distance, _ = fastdtw(window_1, window_2, dist=euclidean)
                dtw_results[scale_name] = distance
            except ValueError:
                window_1_reshaped = window_1.reshape(window_1.shape[0], -1)
                window_2_reshaped = window_2.reshape(window_2.shape[0], -1)
                distance, _ = fastdtw(window_1_reshaped, window_2_reshaped, dist=euclidean)
                dtw_results[scale_name] = distance
        else:
            dtw_results[scale_name] = dtw_results.get('medium', 0) if 'medium' in dtw_results else 0

    combined_distance = (
            short_weight * dtw_results['short'] +
            medium_weight * dtw_results['medium'] +
            long_weight * dtw_results['long']
    )
    dtw_results['combined'] = combined_distance
    return dtw_results

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DENSITY_MAP_PATH = "ad4che_density_maps.pkl"
TEST_DATA_PATH = "ad4che_sequences/ad4che_sequences_test.pickle"
MODEL_PATH = "ad4che_backbone_cnn.pt"

WINDOW_LENGTH = 30
STRIDE = 1
STATE_DIM = 8
OUTPUT_SIZE = 3
LATERAL_IDX = 1
LATERAL_SPEED_IDX = 5

# ============================================================================
# 3. MODEL DEFINITION (2-Layer)
# ============================================================================
class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_input_features = self._calculate_fc_input_features(num_features)
        self.fc1 = nn.Linear(in_features=self.fc_input_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def _calculate_fc_input_features(self, num_features):
        with torch.no_grad():
            x = torch.zeros((1, num_features, WINDOW_LENGTH))
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.flatten(x)
            return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prepare_window_slices(feature_sequence, labels, window_length, stride):
    num_windows = len(feature_sequence) - window_length + 1
    if num_windows <= 0:
        return np.array([]), np.array([])
    X_test_windows = np.zeros((num_windows, window_length, feature_sequence.shape[1]))
    y_test_windows = np.zeros(num_windows)
    for i in range(num_windows):
        X_test_windows[i] = feature_sequence[i:i + window_length]
        y_test_windows[i] = labels[i]
    return X_test_windows, y_test_windows

def convert_density_map_to_dict(numpy_map):
    map_dict = {}
    rows, cols = numpy_map.shape
    for r in range(rows):
        for c in range(cols):
            val = numpy_map[r, c]
            if val > 0:
                map_dict[(r, c)] = {'count': val}
    return map_dict

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
def main():
    print(f"Running on device: {DEVICE}")

    # 1. Load Data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: File not found: {TEST_DATA_PATH}")
        return

    print(f"Loading Test Data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, "rb") as f:
        input_data = pickle.load(f)

    if isinstance(input_data, tuple) and len(input_data) == 2:
        X_test_all, y_test_all = input_data
        input_data = []
        for i in range(len(X_test_all)):
            input_data.append((X_test_all[i], [y_test_all[i]]))

    # 2. Load Maps
    if not os.path.exists(DENSITY_MAP_PATH):
        raise FileNotFoundError(f"Missing {DENSITY_MAP_PATH}.")

    print("Loading Density Maps...")
    with open(DENSITY_MAP_PATH, 'rb') as f:
        raw_maps = pickle.load(f)

    cell_maps = {
        'lk_density_map': convert_density_map_to_dict(raw_maps['lk_map']),
        'lc_density_map': convert_density_map_to_dict(raw_maps['lc_map']),
        'parameters': {'resolution': raw_maps.get('resolution', 0.01)}
    }

    lk_min, lk_max, lc_min, lc_max = extract_min_max_densities(cell_maps)

    # 3. Load Model
    model = CNN1D(num_features=STATE_DIM, num_classes=OUTPUT_SIZE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Successfully loaded model from {MODEL_PATH}")
        except:
            print("Trying strict=False...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    else:
        print(f"WARNING: {MODEL_PATH} not found. Running initialized.")

    model = model.to(DEVICE)
    model.eval()

    # 4. Processing Loop
    state_manager = StateManager(lc_commitment_period=5)

    results_storage = {
        'model_predictions': [],
        'reliability_scores': [],
        'ground_truth': [],
        'ratio_scores': [],
        'absolute_scores': []
    }

    print("Starting Uncertainty Quantification...")

    for idx, item in enumerate(input_data):
        if len(item) == 2:
            feature_sequence, labels = item
        else:
            feature_sequence = item[0]
            labels = item[1]

        if np.isscalar(labels) or (isinstance(labels, list) and len(labels) == 1):
            single_label = labels[0] if isinstance(labels, list) else labels
            labels = [single_label] * len(feature_sequence)

        if idx % 50 == 0:
            print(f"Processing trajectory {idx}/{len(input_data)}")

        X_test_windows, y_test_windows = prepare_window_slices(feature_sequence, labels, WINDOW_LENGTH, STRIDE)

        if len(X_test_windows) == 0:
            continue

        X_tensor = torch.Tensor(X_test_windows).to(DEVICE)

        with torch.no_grad():
            outputs = model(X_tensor)
            predicted_probs = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_labels = torch.argmax(outputs, dim=1).cpu().tolist()

        padding = [0] * (WINDOW_LENGTH - 1)
        CNN_output = padding + predicted_labels

        lateral_position = feature_sequence[:, LATERAL_IDX]
        lateral_speeds = feature_sequence[:, LATERAL_SPEED_IDX]

        traj_reliability = []
        traj_ratio = []
        traj_absolute = []
        similarity_history = []

        lk_weights, lc_weights = [], []
        lk_density_scores, lc_density_scores = [], []
        dtw_short, dtw_medium, dtw_long, dtw_combined, dtw_scaled = [], [], [], [], []
        states = []

        numpy_windows = X_test_windows

        for current_step in range(len(feature_sequence)):
            if current_step < WINDOW_LENGTH - 1:
                traj_reliability.append(0.5)
                traj_ratio.append(0.5)
                traj_absolute.append(0.5)
                lk_weights.append(0.5); lc_weights.append(0.5)
                lk_density_scores.append(0); lc_density_scores.append(0)
                dtw_short.append(0); dtw_medium.append(0); dtw_long.append(0)
                dtw_combined.append(0); dtw_scaled.append(0)
                states.append('LK')
                continue

            window_idx = current_step - (WINDOW_LENGTH - 1)

            # --- USE LOCAL FIXED FUNCTION ---
            dtw_results = local_compute_multi_scale_dtw_similarities(
                numpy_windows, window_idx, WINDOW_LENGTH
            )

            current_probs = predicted_probs[window_idx]

            result = improved_assess_prediction_with_density(
                pred_probs=current_probs,
                classification_output=CNN_output,
                similarity_history=similarity_history,
                dtw_results=dtw_results,
                feature_windows=feature_sequence,
                current_time=current_step,
                state_manager=state_manager,
                cell_maps=cell_maps,
                current_step=current_step,
                total_steps=len(feature_sequence),
                lk_min=lk_min, lk_max=lk_max,
                lc_min=lc_min, lc_max=lc_max,
                lateral_speeds=lateral_speeds
            )

            traj_reliability.append(result['reliability_score'])
            traj_ratio.append(result['ratio_score'])
            traj_absolute.append(result['absolute_score'])

            lk_weights.append(result['lk_weight'])
            lc_weights.append(result['lc_weight'])
            lk_density_scores.append(result['lk_density_score'])
            lc_density_scores.append(result['lc_density_score'])
            dtw_short.append(result['dtw_short'])
            dtw_medium.append(result['dtw_medium'])
            dtw_long.append(result['dtw_long'])
            dtw_combined.append(result['dtw_combined'])
            dtw_scaled.append(result['dtw_scaled'])
            states.append(result['state'])

        results_storage['model_predictions'].append(CNN_output)
        results_storage['reliability_scores'].append(traj_reliability)
        results_storage['ground_truth'].append(labels)
        results_storage['ratio_scores'].append(traj_ratio)
        results_storage['absolute_scores'].append(traj_absolute)

        if idx < 1500 :
             plot_trajectory_with_enhanced_metrics(
                selected_feature=lateral_position,
                reliability_scores=traj_reliability,
                lk_density_scores=lk_density_scores,
                lc_density_scores=lc_density_scores,
                lk_weights=lk_weights,
                lc_weights=lc_weights,
                dtw_values={
                    'short': dtw_short, 'medium': dtw_medium, 'long': dtw_long,
                    'combined': dtw_combined, 'scaled': dtw_scaled
                },
                states=states,
                lateral_speeds=lateral_speeds,
                CNN_output=CNN_output,
                labels=labels,
                title=f'AD4CHE Trajectory {idx} Reliability'
            )

    with open('ad4che_uncertainty_results.pkl', 'wb') as f:
        pickle.dump(results_storage, f)
    print("Results saved to 'ad4che_uncertainty_results.pkl'")

if __name__ == "__main__":
    main()