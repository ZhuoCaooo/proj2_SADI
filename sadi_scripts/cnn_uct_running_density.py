'''this script is running cnn models to get the uncertainty value at each trajectory'''
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import os
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import our density-based uncertainty functions
from density_based_uncertainty import (
    extract_min_max_densities,
    StateManager,
    find_consecutive_ones,
    find_consecutive_twos,
    compute_multi_scale_dtw_similarities,
    improved_assess_prediction_with_density,
    plot_trajectory_with_enhanced_metrics,
    compute_lateral_speed
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# load data into pytorch tensor
input_data = []
for i in range(45, 61):
    idx_str = '{0:02}'.format(i)
    pickle_in = open("../output_normalized_exe_labeled/result" + idx_str + ".pickle", "rb")
    temp_data = pickle.load(pickle_in)
    print("Loaded " + idx_str + " data pack")
    input_data.extend(temp_data)

size = len(input_data)
print('total number of trajectories', size)

'''This block is designed to relabel the trajectory using the labelling results from matlab wtmm-he scripts'''
# Extract the first 1000 left change(maneuver1) traj.s
maneuver_0_traj = []
maneuver_1_traj = []
maneuver_2_traj = []

for traj_index in range(0, len(input_data)):

    maneuver_info = input_data[traj_index][1][-1]

    if maneuver_info == 1 and len(maneuver_1_traj) < 500:
        maneuver_1_traj.append(input_data[traj_index])

    if maneuver_info == 2 and len(maneuver_2_traj) < 500:
        maneuver_2_traj.append(input_data[traj_index])

    if maneuver_info == 0 and len(maneuver_0_traj) < 500:
        maneuver_0_traj.append(input_data[traj_index])

print(len(maneuver_1_traj), 'len(maneuver_1_traj)')
print(len(maneuver_2_traj), 'len(maneuver_2_traj)')
# print(len(maneuver_1_traj[0][0][149]))
# now read the matlab labelling results to change the maneuver info
mat_results_m1 = pd.read_csv('../mat_scripts/500_tra_peaks_m1_test.csv', header=None)
mat_results_m1 = np.array(mat_results_m1)[0]
mat_results_m2 = pd.read_csv('../mat_scripts/500_tra_peaks_m2_test.csv', header=None)
mat_results_m2 = np.array(mat_results_m2)[0]
relabeled_traj = []

# Load cell maps
with open('cell_maps_train_data/density_cell_maps_2.0s_march.pkl', 'rb') as f:
    cell_maps = pickle.load(f)
# Calculate the separate density ranges once from LK and LC maps
lk_min, lk_max, lc_min, lc_max = extract_min_max_densities(cell_maps)
print(f"LK density range: min={lk_min}, max={lk_max}")
print(f"LC density range: min={lc_min}, max={lc_max}")


# this is the prediction advance time, in unit of seconds
advanced_time = 1.5
label_adjustment = int(30 + advanced_time * 25)

for traj in range(0, len(maneuver_0_traj)):
    veh_info = maneuver_0_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_0_traj[traj][1]  # Maneuver info, should be a list or array of 150 length
    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

for traj in range(0, len(maneuver_1_traj)):
    veh_info = maneuver_1_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_1_traj[traj][1]  # Maneuver info, should be a list or array of 150 length
    # 15 is due to the signal padding in Matlab scripts
    label_index = min(149, max(int(mat_results_m1[traj]) - label_adjustment, 0))  # Ensure it's a valid integer

    # this is for 2s constant length labeling
    # label_index = 100

    # Create the iterables to assign to slices
    zeros = [0] * (label_index + 1)  # List of zeros for indices 0 to label_index
    ones = [1] * (150 - (label_index + 1))  # List of ones for indices from label_index + 1 to 149

    # Assign the iterables to the maneuver_info
    maneuver_info[:label_index + 1] = zeros
    maneuver_info[label_index + 1:] = ones

    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

for traj in range(0, len(maneuver_2_traj)):
    veh_info = maneuver_2_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_2_traj[traj][1]  # Maneuver info, should be a list or array of 150 length

    # 15 is due to the signal padding in Matlab scripts
    label_index = min(149, max(int(mat_results_m2[traj]) - label_adjustment, 0))  # Ensure it's a valid integer

    # this is for 2s constant length labeling
    # label_index = 100

    # Create the iterables to assign to slices
    zeros = [0] * (label_index + 1)  # List of zeros for indices 0 to label_index
    twos = [2] * (150 - (label_index + 1))  # List of ones for indices from label_index + 1 to 149

    # Assign the iterables to the maneuver_info
    maneuver_info[:label_index + 1] = zeros
    maneuver_info[label_index + 1:] = twos
    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

input_data = relabeled_traj

size = len(input_data)
random.shuffle(input_data)
print('total number of trajectories after relabeling', size)

window_length = 25
stride = 1


def prepare_window_slices(feature_sequence, labels, window_length, stride):
    num_windows = len(feature_sequence) - window_length + 1
    # Pre-allocate arrays
    X_test_windows = np.zeros((num_windows, window_length, feature_sequence.shape[1]))
    y_test_windows = np.zeros(num_windows)

    # Fill arrays
    for i in range(num_windows):
        X_test_windows[i] = feature_sequence[i:i + window_length]
        y_test_windows[i] = labels[i]

    return X_test_windows, y_test_windows


'''the structure of CNN model'''
# Set hyper-parameters
state_dim = 16
sequence_length = window_length
output_size = 3  # Adjusted to match the number of classes (0, 1, 2)

batch_size = 1


class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self._calculate_fc_input_features(num_features), out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=num_classes)  # Adjusted to match the output size

    def forward(self, x):
        x = x.transpose(1, 2)  # swap the temporal and feature dimensions
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _calculate_fc_input_features(self, num_features):
        with torch.no_grad():
            x = torch.zeros((1, num_features, sequence_length))
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.flatten(x)
            return x.numel()


# Load the pre-trained model
model = CNN1D(num_features=state_dim, num_classes=output_size)
#model.load_state_dict(torch.load("../auto_label_models/cnn_model_exe_time_he_0.5s.pt"))
model.load_state_dict(torch.load("../auto_label_models/cnn_model_prediction_horizon_2.0s.pt"))
model = model.to(device)
model.eval()

current_processing = 0
# The iteration on testing trajectories
cnn_pred_list = []
ground_truth = []
LK_FP_prediction_list = []
LC_root_list = []
correct_predictions = 0
total_examples = 0
LC_sample_num = 0
LK_sample_num = 0

prediction_prob = []
predictions_prob_meta = []
ground_truth_each_timestep = []

model_pred_list_all = []
reliability_scores_all = []
prob_output_list = []

# Lists for storing execution time data
all_exe_time_pred = []
all_exe_time_truth = []

# Add these containers for the new metrics
ratio_scores_all = []
absolute_scores_all = []

# Initialize the state manager (add before your loop)
state_manager = StateManager(lc_commitment_period=5)  # Adjust parameter as needed

for feature_sequence, labels in input_data:
    print('current_processing', current_processing)
    current_processing += 1

    # Record ground truth for all trajectories
    ground_truth_each_timestep.append(labels)
    end_maneuver = labels[-1]
    ground_truth.append(labels[-1])

    # Process windows and get predictions (unchanged)
    X_test_windows, y_test_windows = prepare_window_slices(feature_sequence, labels, window_length, stride)
    X_test_windows = torch.Tensor(X_test_windows)
    y_test_windows = torch.LongTensor(y_test_windows)
    test_dataset = TensorDataset(X_test_windows, y_test_windows)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    CNN_output = []
    predicted_probs_list_single = []

    # Get predictions for each window (unchanged)
    for inputs_2, _ in test_dataloader:
        inputs_2 = inputs_2.to(device)
        outputs = model(inputs_2)
        predicted_probs = F.softmax(outputs, dim=1)
        predicted_probs_list_single.append(predicted_probs.cpu().detach().numpy())
        predicted_labels = torch.argmax(outputs, dim=1).item()
        CNN_output.append(predicted_labels)

    # Record probability metadata (unchanged)
    predictions_prob_meta.append(np.concatenate(predicted_probs_list_single, axis=0))

    # Add padding for early timesteps (unchanged)
    CNN_output = [0] * 24 + CNN_output
    cnn_pred_list.append(CNN_output[-1])

    # Update accuracy metrics (unchanged)
    correct_predictions += np.sum(CNN_output[-1] == labels[-1])
    total_examples += 1

    # Calculate execution times for LC trajectories only (unchanged)
    if end_maneuver in [1, 2]:  # LC trajectories
        if end_maneuver == 1:
            exe_time_truth = find_consecutive_ones(labels, 10)
            exe_time_pred = find_consecutive_ones(CNN_output, 5)
        else:  # end_maneuver == 2
            exe_time_truth = find_consecutive_twos(labels, 10)
            exe_time_pred = find_consecutive_twos(CNN_output, 5)
        all_exe_time_truth.append(exe_time_truth)
        all_exe_time_pred.append(exe_time_pred)

    # Process uncertainty for all trajectories (modified section)
    similarity_history = []
    uct_result = []

    # Containers for multi-scale DTW results
    dtw_short_values = []
    dtw_medium_values = []
    dtw_long_values = []
    dtw_combined_values = []
    dtw_scaled_values = []
    state_values = []

    # Containers for density-related information (unchanged)
    lk_densities = []
    lc_densities = []
    lk_density_scores = []
    lc_density_scores = []
    lk_weights = []
    lc_weights = []

    # Extract lateral position
    lateral_position = feature_sequence[:, 2]  # lateral position (3rd feature)

    # Compute lateral speed from position instead of using feature_sequence[:, 3]
    lateral_speeds = compute_lateral_speed(lateral_position)

    # Prepare numpy array version of windows for DTW calculations
    numpy_windows = X_test_windows.cpu().numpy()

    # Containers for the new reliability metrics
    ratio_scores = []
    absolute_scores = []

    for current_step in range(0, len(feature_sequence)):
        # Get multi-scale DTW results
        dtw_results = compute_multi_scale_dtw_similarities(
            numpy_windows, current_step, window_length)

        # Add to result collections
        dtw_short_values.append(dtw_results['short'])
        dtw_medium_values.append(dtw_results['medium'])
        dtw_long_values.append(dtw_results['long'])
        dtw_combined_values.append(dtw_results['combined'])

        # Get predictions
        pred_probs = predicted_probs_list_single[max(0, current_step - 24)]

        # Use the improved assessment function with lateral_speeds as a parameter
        result = improved_assess_prediction_with_density(
            pred_probs, CNN_output, similarity_history, dtw_results,
            feature_sequence, current_step, state_manager,
            cell_maps, current_step, total_steps=150,
            lk_min=lk_min, lk_max=lk_max,
            lc_min=lc_min, lc_max=lc_max,
            lateral_speeds=lateral_speeds
        )
        # Store the new metrics
        ratio_scores.append(result['ratio_score'])
        absolute_scores.append(result['absolute_score'])

        # Store results (unchanged + new fields)
        lk_densities.append(result['lk_density'])
        lc_densities.append(result['lc_density'])
        lk_density_scores.append(result['lk_density_score'])
        lc_density_scores.append(result['lc_density_score'])
        lk_weights.append(result['lk_weight'])
        lc_weights.append(result['lc_weight'])
        dtw_scaled_values.append(result['dtw_scaled'])
        state_values.append(result['state'])

        uct_result.append(result)

    # Extract and store reliability scores
    reliability_scores = [result['reliability_score'] for result in uct_result]

    # Use the enhanced visualization
    selected_feature = feature_sequence[:, 2]  # lateral position (3rd feature)

    plot_trajectory_with_enhanced_metrics(
        selected_feature=selected_feature,
        reliability_scores=reliability_scores,
        lk_density_scores=lk_density_scores,
        lc_density_scores=lc_density_scores,
        lk_weights=lk_weights,
        lc_weights=lc_weights,
        dtw_values={
            'short': dtw_short_values,
            'medium': dtw_medium_values,
            'long': dtw_long_values,
            'combined': dtw_combined_values,
            'scaled': dtw_scaled_values
        },
        states=state_values,
        lateral_speeds=lateral_speeds,
        CNN_output=CNN_output,
        labels=labels,
        title=f'Trajectory {current_processing} with Physics-informed Reliability'
    )

    # Store results (unchanged)
    model_pred_list_all.append(CNN_output)
    reliability_scores_all.append(reliability_scores)
    prob_output_list.append(np.concatenate(predicted_probs_list_single, axis=0))
    # Store results for all metrics
    ratio_scores_all.append(ratio_scores)
    absolute_scores_all.append(absolute_scores)


# Update your output data structure
output_data = {
    'model_predictions': model_pred_list_all,
    'reliability_scores': reliability_scores_all,
    'ground_truth': ground_truth_each_timestep,
    'probability_outputs': prob_output_list,
    'execution_times': {
        'truth': all_exe_time_truth,
        'predicted': all_exe_time_pred
    },
    # Add new metadata and metrics
    'ratio_scores': ratio_scores_all,
    'absolute_scores': absolute_scores_all,
    'uncertainty_metadata': {
        'method': 'enhanced-density-based-with-ratio',
        'description': 'Reliability scores using multi-scale DTW, lateral speed scaling, state persistence, and ratio-based density comparison with weight-based boosting'
    }
}

# Save results
# if not os.path.exists('../model_testing_paper_2'):
#     os.makedirs('../model_testing_paper_2')
#
# with open('../model_testing_paper_2/cnn_results_density_based_2.0s_new.pkl', 'wb') as f:
#     pickle.dump(output_data, f)
# print("Results saved successfully!")
