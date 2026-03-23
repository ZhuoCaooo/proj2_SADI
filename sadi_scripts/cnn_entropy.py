import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
import csv
import os
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# load data into pytorch tensor
input_data = []
for i in range(50, 61):
    idx_str = '{0:02}'.format(i)
    pickle_in = open("../output_normalized_exe_labeled/result" + idx_str + ".pickle", "rb")
    temp_data = pickle.load(pickle_in)
    print("Loaded " + idx_str + " data pack")
    input_data.extend(temp_data)

# pickle_in = open("result.pickle","rb")
# input_data = pickle.load(pickle_in)
size = len(input_data)
print('total number of trajectories', size)

#training_set = input_data[:int(size * 0.8)]
#print('total number of training trajectories', len(training_set))
#testing_set = input_data[int(size * 0.8):]
#print('total number of testing trajectories', len(testing_set))


from sklearn.metrics.pairwise import cosine_similarity

'''This block is designed to relabel the trajectory using the labelling results from matlab wtmm-he scripts'''
# Extract the first 1000 left change(maneuver1) traj.s
maneuver_0_traj = []
maneuver_1_traj = []
maneuver_2_traj = []

for traj_index in range(0, len(input_data)):

    maneuver_info = input_data[traj_index][1][-1]

    if maneuver_info == 1 and len(maneuver_1_traj) < 400:
        maneuver_1_traj.append(input_data[traj_index])

    if maneuver_info == 2 and len(maneuver_2_traj) < 400:
        maneuver_2_traj.append(input_data[traj_index])

    if maneuver_info == 0 and len(maneuver_0_traj) < 50:
        maneuver_0_traj.append(input_data[traj_index])

# print(len(maneuver_1_traj[0][0][149]))
# now read the matlab labelling results to change the maneuver info
mat_results_m1 = pd.read_csv('../mat_scripts/400_tra_he_m1_test.csv', header=None)
mat_results_m1 = np.array(mat_results_m1)[0]
mat_results_m2 = pd.read_csv('../mat_scripts/400_tra_he_m2_test.csv', header=None)
mat_results_m2 = np.array(mat_results_m2)[0]
relabeled_traj = []

for traj in range(0, len(maneuver_0_traj)):
    veh_info = maneuver_0_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_0_traj[traj][1]  # Maneuver info, should be a list or array of 150 length
    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

for traj in range(0, len(maneuver_1_traj)):
    veh_info = maneuver_1_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_1_traj[traj][1]  # Maneuver info, should be a list or array of 150 length
    # 15 is due to the signal padding in Matlab scripts
    label_index = min(149, max(int(mat_results_m1[traj]) - 15, 0))  # Ensure it's a valid integer

    #this is for 2s constant length labeling
    #label_index = 100

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
    label_index = min(149, max(int(mat_results_m2[traj]) - 15, 0))  # Ensure it's a valid integer

    # this is for 2s constant length labeling
    #label_index = 100

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
    # Convert the training set tensors to NumPy arrays
    X_test = np.array([feature_sequence])
    y_test = np.array([labels])

    # print('X_test.shape------', X_test.shape)
    # print('y_test.shape------', y_test.shape)

    # Initialize lists to store the windowed data for testing
    X_test_windows = []
    y_test_windows = []

    # Iterate through each trajectory in X_test and y_test
    for trajectory, labels in zip(X_test, y_test):
        num_timesteps = trajectory.shape[0]

        # Slide the window over the trajectory
        for start in range(0, num_timesteps - window_length + 1, stride):
            end = start + window_length

            # Extract the windowed data
            windowed_trajectory = trajectory[start:end, :]
            windowed_labels = labels[start:end]

            # Append to the lists
            X_test_windows.append(windowed_trajectory)
            y_test_windows.append(windowed_labels)

    # Convert the lists to NumPy arrays
    X_test_windows = np.array(X_test_windows)
    y_test_windows = np.array(y_test_windows)
    # Convert the shape to (samples, features_num, length)
    # X_test_windows = np.transpose(X_test_windows, (0, 2, 1))
    # Extract the first number from each row and create a new array with shape (744)
    y_test_windows = y_test_windows[:, 0]
    # print('X_test_windows.shape:', X_test_windows.shape)
    # print('y_test_windows.shape:', y_test_windows.shape)

    return X_test_windows, y_test_windows


def find_consecutive_ones(arr, n):
    """
    Find the index of the first occurrence of n consecutive 1s in the list.

    Parameters:
    arr (list): The input list of integers.
    n (int): The number of consecutive 1s to find.

    Returns:
    int: The starting index of the first occurrence of n consecutive 1s, or -1 if not found.
    """
    length = len(arr)

    if length < n:
        return -1  # Not enough elements to find the pattern

    for i in range(length - n + 1):
        if all(x == 1 for x in arr[i:i + n]):
            return i

    return len(arr)  # Pattern not found


def find_consecutive_twos(arr, n):
    """
    Find the index of the first occurrence of exactly n consecutive 2s in the list.

    Parameters:
    arr (list): The input list of integers.
    n (int): The number of consecutive 2s to find.

    Returns:
    int: The starting index of the first occurrence of n consecutive 2s, or -1 if not found.
    """
    length = len(arr)

    if length < n:
        return -1  # Not enough elements to find the pattern

    for i in range(length - n + 1):
        if all(arr[i + j] == 2 for j in range(n)):
            return i

    return len(arr)  # Pattern not found


def sum_false_positives(arr, min_length):
    """
    Find and sum the lengths of all sequences of more than `min_length` consecutive non-zero values in the list.

    Parameters:
    arr (list): The input list of integers.
    min_length (int): The minimum length of consecutive non-zero values to consider.

    Returns:
    int: The sum of lengths of all sequences of more than `min_length` consecutive non-zero values.
    """
    total_length = 0
    current_length = 0

    for value in arr:
        if value != 0:
            current_length += 1
        else:
            if current_length > min_length:
                total_length += current_length
            current_length = 0

    # Check the last sequence
    if current_length > min_length:
        total_length += current_length

    return total_length


def compute_dtw_similarities(X_test_windows, exe_time_pred, window_length):
    """
    Compute DTW (Dynamic Time Warping) similarities between specific windows in the test data.

    Parameters:
    X_test_windows (numpy array): The test data array with shape (num_samples, num_timesteps, num_features).
    exe_time_pred (int): The predicted execution time for window selection.
    window_length (int): The length of the window used for slicing.

    Returns:
    tuple: A tuple containing the DTW distance values (distance_value_p, distance_value_n).
    """
    # Scale the data
    X_test_windows = np.array(X_test_windows)

    # Remove the features that don't have much difference
    X_test_windows = np.delete(X_test_windows, [0, 1, 7], axis=2)  # axis=2 refers to the feature dimension

    # Determine the indices for the windows
    index_1 = max(window_length, exe_time_pred - window_length - 1 - 25)  # no too early windows
    index_2 = min(exe_time_pred - window_length - 1, len(X_test_windows) - 1)
    index_3 = min(exe_time_pred - window_length - 1 + 5, len(X_test_windows) - 1)

    # Extract the windows (not flattening, as DTW works on sequences)
    window_1 = X_test_windows[index_1][:3]
    window_2 = X_test_windows[index_2][:3]
    window_3 = X_test_windows[index_3][:3]

    # Compute the DTW distance
    distance_n, _ = fastdtw(window_1, window_2, dist=euclidean)
    distance_p, _ = fastdtw(window_2, window_3, dist=euclidean)

    # Print results (optional)
    #print(f"DTW distance_n between the two windows: {distance_n}")
    #print(f"DTW distance_p between the two windows: {distance_p}")

    return distance_p, distance_n



def normalize_similarity_lists(list1, list2):
    """
    Normalize two lists of similarity values based on their combined z-scores.

    Parameters:
    list1 (list of float): The first list of similarity values.
    list2 (list of float): The second list of similarity values.

    Returns:
    tuple: A tuple containing the z-score normalized versions of both lists.
    """
    # Combine both lists
    combined_list = list1 + list2

    # Calculate the mean and standard deviation of the combined list
    mean = sum(combined_list) / len(combined_list)
    std_dev = (sum((x - mean) ** 2 for x in combined_list) / len(combined_list)) ** 0.5

    # Define a function for z-score normalization
    def zscore_normalize(values, mean_val, std_val):
        return [(v - mean_val) / std_val for v in values]

    # Normalize each list based on the z-scores
    z_scores_list1 = zscore_normalize(list1, mean, std_dev)
    z_scores_list2 = zscore_normalize(list2, mean, std_dev)

    # Combine the z-scores for min-max scaling
    combined_z_scores = z_scores_list1 + z_scores_list2

    # Find the min and max z-scores
    min_z = min(combined_z_scores)
    max_z = max(combined_z_scores)

    # Define a function for min-max scaling
    def min_max_scale(values, min_val, max_val):
        return [(v - min_val) / (max_val - min_val) for v in values]

    # Rescale the z-scores between 0 and 1
    normalized_list1 = min_max_scale(z_scores_list1, min_z, max_z)
    normalized_list2 = min_max_scale(z_scores_list2, min_z, max_z)

    return normalized_list1, normalized_list2

def calculate_uct(normalized_p, normalized_n):
    """
    Calculate the uncertainty UCT for each timestep.

    Parameters:
    normalized_p (list or np.array): List of normalized Sim(u, up) values.
    normalized_n (list or np.array): List of normalized Sim(u, un) values.

    Returns:
    np.array: Array of UCT values for each timestep.
    """
    # Convert lists to numpy arrays if they aren't already
    normalized_p = np.array(normalized_p)
    normalized_n = np.array(normalized_n)

    # Calculate Pu and Nu for each timestep
    Pu = normalized_p / (normalized_p + normalized_n)
    Nu = normalized_n / (normalized_p + normalized_n)

    # Avoid log of zero by adding a small epsilon
    epsilon = 1e-10
    Pu = np.clip(Pu, epsilon, 1 - epsilon)
    Nu = np.clip(Nu, epsilon, 1 - epsilon)

    # Calculate UCT for each timestep
    UCT = -(Pu * np.log10(Pu) + Nu * np.log10(Nu))
    return UCT

def extend_uct_with_initial_values(UCT_values, extension_length=25):
    """
    Extend the UCT values by adding a specified number of initial values equal to the first UCT value.

    Parameters:
    UCT_values (np.array): Array of UCT values.
    extension_length (int): Number of values to add at the beginning.

    Returns:
    np.array: Extended array of UCT values.
    """
    # Get the first value of the UCT values
    first_value = (UCT_values[0]+UCT_values[1]+ UCT_values[2])/3

    # Create an array of the first value repeated extension_length times
    initial_values = np.full(extension_length, first_value)

    # Concatenate the initial values with the original UCT values
    extended_UCT_values = np.concatenate((initial_values, UCT_values))

    return extended_UCT_values

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
model.load_state_dict(torch.load("../auto_label_models/cnn_model_exe_time_he_0.0s.pt"))
model.eval()
model.to(device)

# The iteration on testing trajectories
cnn_pred_list = []
ground_truth = []
earliness_list = []
LK_FP_prediction_list = []
pre_quality_list = []
LC_root_list = []
correct_predictions = 0
total_examples = 0
LC_sample_num = 0
LK_sample_num = 0

timeliness_list = []
prediction_prob = []
predictions_prob_meta = []
ground_truth_each_timestep = []
similarity_value_p_m1 = []
similarity_value_n_m1 = []
similarity_value_p_m2 = []
similarity_value_n_m2 = []
for feature_sequence, labels in input_data:

    ground_truth_each_timestep.append(labels)
    # The end_maneuver
    end_maneuver = labels[-1]

    # the interested ground truth with index of timesteps
    ground_truth.append(labels[-1])

    X_test_windows, y_test_windows = prepare_window_slices(feature_sequence, labels, window_length, stride)

    # Convert NumPy arrays to PyTorch tensors
    X_test_windows = torch.Tensor(X_test_windows)
    y_test_windows = torch.LongTensor(y_test_windows)

    # Create TensorDatasets
    test_dataset = TensorDataset(X_test_windows, y_test_windows)

    # Create DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print(maneuvre_index)
    # print('sentence_in.shape', X_sample.shape)
    # print('state_sequence.shape', state_sequence.shape)

    CNN_output = []
    predicted_probs_list_single = []
    # CNN_output_proba = []
    # iterate through each window, therefore generate prediction results on every time step
    for inputs_2, _ in test_dataloader:
        inputs_2 = inputs_2.to(device)
        # Forward pass
        outputs = model(inputs_2)

        # Apply softmax activation to get class probabilities
        predicted_probs = F.softmax(outputs, dim=1)
        # Append the predicted probabilities for each window
        predicted_probs_list_single.append(predicted_probs.cpu().detach().numpy())  # Convert to NumPy array and move to CPU
        # Get predicted labels by taking the class with the highest probability
        predicted_labels = torch.argmax(outputs, dim=1).item()
        CNN_output.append(predicted_labels)

    predictions_prob_meta.append(np.concatenate(predicted_probs_list_single, axis=0))
    #print('len(predicted_probs_list_single)', predicted_probs_list_single)
    # print('len(CNN_output)', len(CNN_output))
    # print('len(labels)', len(labels))
    #print('predicted_probs_list_single', predicted_probs_list_single[0])
    # Convert the list of arrays into a single NumPy array for easier plotting

    # Convert list of predicted probabilities to a NumPy array and remove the singleton dimension
    predicted_probs_array_1 = np.array(predicted_probs_list_single).squeeze(axis=1)  # Shape becomes (126, 3)

    # Calculate Shannon entropy for each timestep
    shannon_entropy = entropy(predicted_probs_array_1, axis=1)  # Entropy along the class probabilities axis

    # Pad 24 zeros at the beginning to make the length 150
    shannon_entropy = np.pad(shannon_entropy, (24, 0), 'constant', constant_values=0)

    predicted_probs_array = np.vstack(predicted_probs_list_single)

    # Extract the probabilities for each class
    lk_probs = predicted_probs_array[:, 0]  # Probabilities for LK (Lane Keep)
    lcl_probs = predicted_probs_array[:, 1]  # Probabilities for LCL (Lane Change Left)
    lcr_probs = predicted_probs_array[:, 2]  # Probabilities for LCR (Lane Change Right)

    # Pad the beginning of the probabilities with NaNs to shift them to start at timestep 24
    padding_length = 24
    padded_lk_probs = np.concatenate((np.full(padding_length, np.nan), lk_probs))
    padded_lcl_probs = np.concatenate((np.full(padding_length, np.nan), lcl_probs))
    padded_lcr_probs = np.concatenate((np.full(padding_length, np.nan), lcr_probs))

    # Generate time steps (0 to 149 for 150 timesteps)
    time_steps = np.arange(150)

    # Plotting the probabilities over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, padded_lk_probs, label='LK Probability', color='blue')
    plt.plot(time_steps, padded_lcl_probs, label='LCL Probability', color='green')
    plt.plot(time_steps, padded_lcr_probs, label='LCR Probability', color='red')
    plt.plot(range(len(shannon_entropy)), shannon_entropy, marker='o', linestyle='-')

    plt.title('Class Probabilities Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 150)  # Set x-axis limits to cover all 150 timesteps
    plt.show()

    # The prediction begins at 50th timestep
    CNN_output = [0] * 24 + CNN_output
    # Extract the cnn output at each trajectory at the interested timestep
    cnn_pred_list.append(CNN_output[-1])

    correct_predictions += np.sum(CNN_output[-1] == labels[-1])
    total_examples += 1

    # print(CNN_output)
    # print('ground truth labels', labels)
    # find the ture positive continuous prediction

    # print_single_results(CNN_output, labels)

    '''This is the earliness part'''
    model_pred_list = CNN_output
    if end_maneuver == 1:
        exe_time_truth = find_consecutive_ones(labels, 10)
        exe_time_pred = find_consecutive_ones(model_pred_list, 5)
        #if prediction is later than truth
        if exe_time_pred >= exe_time_truth:
            earliness = (len(feature_sequence) - exe_time_pred)/(len(feature_sequence) - exe_time_truth+1)
        elif exe_time_pred in range(exe_time_truth-50, exe_time_truth):
            earliness = 1
        else:
            earliness = max(0, (100 - exe_time_truth + exe_time_pred)/50)

        earliness_list.append(earliness)

        '''This is the uncertainty part'''
        similarity_value_p_continues_list = []
        similarity_value_n_continues_list = []
        if exe_time_pred != 150:
            for current_step in range(25, len(feature_sequence)):
                similarity_value_p_continues, similarity_value_n_continues= compute_dtw_similarities(X_test_windows, current_step, window_length)

                similarity_value_p_continues_list.append(similarity_value_p_continues)
                similarity_value_n_continues_list.append(similarity_value_n_continues)
            #print(" ".join(["{:.4f}".format(value) for value in similarity_value_p_continues_list]))
            #print(" ".join(["{:.4f}".format(value) for value in similarity_value_n_continues_list]))
            normalized_p_continues, normalized_n_continues = normalize_similarity_lists(similarity_value_p_continues_list, similarity_value_n_continues_list)

            #print(" ".join(["{:.4f}".format(value) for value in normalized_p_continues]))
            #print(" ".join(["{:.4f}".format(value) for value in normalized_n_continues]))
            UCT_values = calculate_uct(normalized_p_continues, normalized_n_continues)
            #print(" ".join(["{:.4f}".format(value) for value in UCT_values]))

            # Extend UCT values to 150 length by adding 25 values at the beginning
            extended_UCT_values = extend_uct_with_initial_values(UCT_values)

            # Calculate the mean and standard deviation
            mean_val = np.mean(extended_UCT_values)
            std_val = np.std(extended_UCT_values)

            # Standardize the values (z-score)
            standardized_values = [(v - mean_val) / std_val for v in extended_UCT_values]

            # Rescale the standardized values to [0, 1] using min-max scaling
            min_val = min(standardized_values)
            max_val = max(standardized_values)

            extended_UCT_values = [(v - min_val) / (max_val - min_val) for v in standardized_values]
            #print(" ".join(["{:.4f}".format(value) for value in extended_UCT_values]))

            # Define the time window range
            window_start = exe_time_pred
            window_end = exe_time_pred + 12

            # Create a copy of extended_UCT_values to modify
            modified_UCT_values = extended_UCT_values.copy()

            # Apply the changes based on the conditions
            for i in range(len(extended_UCT_values)):
                if i < window_start:
                    modified_UCT_values[i] = 1 - extended_UCT_values[i]
                elif i > window_end:
                    modified_UCT_values[i] = 1 - extended_UCT_values[i]
                # else: keep the value as it is

            # Select the feature with index 2 (the 3rd feature)
            selected_feature = feature_sequence[:, 2]

            # Create a time array for the x-axis (125 time steps)
            time_steps = np.arange(len(selected_feature))
            print('exe_time_truth', exe_time_truth)
            print('exe_time_pred',   exe_time_pred)

            # Use the 'RdYlGn' colormap directly with extended_UCT_values without normalization
            cmap = plt.colormaps['RdYlGn'].reversed()

            # Ensure extended_UCT_values are scaled appropriately for the colormap
            min_val = np.min(extended_UCT_values)
            max_val = np.max(extended_UCT_values)

            # Plot each segment with the corresponding color
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(len(time_steps) - 1):
                # Directly map extended_UCT_values to colors without normalization
                #color_value = extended_UCT_values[i]
                color_value = modified_UCT_values[i]
                ax.plot(time_steps[i:i + 2], selected_feature[i:i + 2],
                        color=cmap((color_value - min_val) / (max_val - min_val)),
                        linewidth=5.5)

            ax.set_title('Trajectory Profile Over Time with UCT-based Coloring')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Lateral Position')
            ax.grid(True)

            # Create a colorbar with direct mapping of extended_UCT_values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_val, vmax=max_val))
            sm.set_array([])  # Note: set_array is typically needed to work with colorbar
            fig.colorbar(sm, ax=ax, label='UCT Value')

            # Mark the specific timestep (exe_time_pred) on the plot
            ax.axvline(x=exe_time_pred, color='blue', linestyle='--', linewidth=2, label=f'Exe Time Pred ({exe_time_pred})')
            ax.text(exe_time_pred, ax.get_ylim()[1] * 0.9, f'{exe_time_pred}', color='blue', fontsize=12,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

            plt.legend()
            plt.show()

            # Define a custom color map for the three labels
            cmap_labels = mcolors.ListedColormap(
                ['blue', 'green', 'red'])  # Assigning blue for 0, green for 1, red for 2

            # Plot each segment with the color based on CNN_output label
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(len(time_steps) - 1):
                # Map CNN_output to colors using the custom colormap
                color_value = CNN_output[i]  # Get label (0, 1, or 2)
                ax.plot(time_steps[i:i + 2], selected_feature[i:i + 2],
                        color=cmap_labels(color_value),
                        linewidth=5.5)

            ax.set_title('Trajectory Profile Over Time with CNN Label Coloring')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Lateral Position')
            ax.grid(True)

            # Create a colorbar to indicate lane-keeping and lane-changing labels
            sm = plt.cm.ScalarMappable(cmap=cmap_labels, norm=mcolors.Normalize(vmin=0, vmax=2))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Lane Keeping', 'Lane Change Left', 'Lane Change Right'])
            cbar.set_label('CNN Output Label')

            plt.show()




    if end_maneuver == 2:
        exe_time_truth = find_consecutive_twos(labels, 10)
        exe_time_pred = find_consecutive_twos(model_pred_list, 10)

        #if prediction is later than truth
        if exe_time_pred >= exe_time_truth:
            earliness = (len(feature_sequence) - exe_time_pred)/(len(feature_sequence) - exe_time_truth+1)
        elif exe_time_pred in range (exe_time_truth-50, exe_time_truth):
            earliness = 1
        else:
            earliness = max(0, (100 - exe_time_truth + exe_time_pred)/50)
        earliness_list.append(earliness)


        similarity_value_p_continues_list = []
        similarity_value_n_continues_list = []
        if exe_time_pred != 150:
            for current_step in range(25, len(feature_sequence)):
                similarity_value_p_continues, similarity_value_n_continues= compute_dtw_similarities(X_test_windows, current_step, window_length)

                similarity_value_p_continues_list.append(similarity_value_p_continues)
                similarity_value_n_continues_list.append(similarity_value_n_continues)
            #print(" ".join(["{:.4f}".format(value) for value in similarity_value_p_continues_list]))
            #print(" ".join(["{:.4f}".format(value) for value in similarity_value_n_continues_list]))
            normalized_p_continues, normalized_n_continues = normalize_similarity_lists(similarity_value_p_continues_list, similarity_value_n_continues_list)

            #print(" ".join(["{:.4f}".format(value) for value in normalized_p_continues]))
            #print(" ".join(["{:.4f}".format(value) for value in normalized_n_continues]))
            UCT_values = calculate_uct(normalized_p_continues, normalized_n_continues)
            #print(" ".join(["{:.4f}".format(value) for value in UCT_values]))

            # Extend UCT values to 150 length by adding 25 values at the beginning
            extended_UCT_values = extend_uct_with_initial_values(UCT_values)

            # Calculate the mean and standard deviation
            mean_val = np.mean(extended_UCT_values)
            std_val = np.std(extended_UCT_values)

            # Standardize the values (z-score)
            standardized_values = [(v - mean_val) / std_val for v in extended_UCT_values]

            # Rescale the standardized values to [0, 1] using min-max scaling
            min_val = min(standardized_values)
            max_val = max(standardized_values)

            extended_UCT_values = [(v - min_val) / (max_val - min_val) for v in standardized_values]
            #print(" ".join(["{:.4f}".format(value) for value in extended_UCT_values]))

            # Define the time window range
            window_start = exe_time_pred
            window_end = exe_time_pred + 12

            # Create a copy of extended_UCT_values to modify
            modified_UCT_values = extended_UCT_values.copy()

            # Apply the changes based on the conditions
            for i in range(len(extended_UCT_values)):
                if i < window_start:
                    modified_UCT_values[i] = 1 - extended_UCT_values[i]
                elif i > window_end:
                    modified_UCT_values[i] = 1 - extended_UCT_values[i]
                # else: keep the value as it is

            # Select the feature with index 2 (the 3rd feature)
            selected_feature = feature_sequence[:, 2]

            # Create a time array for the x-axis (125 time steps)
            time_steps = np.arange(len(selected_feature))
            print('exe_time_truth', exe_time_truth)
            print('exe_time_pred',   exe_time_pred)

            # Use the 'RdYlGn' colormap directly with extended_UCT_values without normalization
            cmap = plt.colormaps['RdYlGn'].reversed()

            # Ensure extended_UCT_values are scaled appropriately for the colormap
            min_val = np.min(extended_UCT_values)
            max_val = np.max(extended_UCT_values)

            # Plot each segment with the corresponding color
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(len(time_steps) - 1):
                # Directly map extended_UCT_values to colors without normalization
                #color_value = extended_UCT_values[i]
                color_value = modified_UCT_values[i]
                ax.plot(time_steps[i:i + 2], selected_feature[i:i + 2],
                        color=cmap((color_value - min_val) / (max_val - min_val)),
                        linewidth=5.5)

            ax.set_title('Trajectory Profile Over Time with UCT-based Coloring')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Lateral Position')
            ax.grid(True)

            # Create a colorbar with direct mapping of extended_UCT_values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_val, vmax=max_val))
            sm.set_array([])  # Note: set_array is typically needed to work with colorbar
            fig.colorbar(sm, ax=ax, label='UCT Value')

            # Mark the specific timestep (exe_time_pred) on the plot
            ax.axvline(x=exe_time_pred, color='blue', linestyle='--', linewidth=2, label=f'Exe Time Pred ({exe_time_pred})')
            ax.text(exe_time_pred, ax.get_ylim()[1] * 0.9, f'{exe_time_pred}', color='blue', fontsize=12,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

            plt.legend()
            plt.show()


            # Define a custom color map for the three labels
            cmap_labels = mcolors.ListedColormap(
                ['blue', 'green', 'red'])  # Assigning blue for 0, green for 1, red for 2

            # Plot each segment with the color based on CNN_output label
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(len(time_steps) - 1):
                # Map CNN_output to colors using the custom colormap
                color_value = CNN_output[i]  # Get label (0, 1, or 2)
                ax.plot(time_steps[i:i + 2], selected_feature[i:i + 2],
                        color=cmap_labels(color_value),
                        linewidth=5.5)

            ax.set_title('Trajectory Profile Over Time with CNN Label Coloring')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Lateral Position')
            ax.grid(True)

            # Create a colorbar to indicate lane-keeping and lane-changing labels
            sm = plt.cm.ScalarMappable(cmap=cmap_labels, norm=mcolors.Normalize(vmin=0, vmax=2))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Lane Keeping', 'Lane Change Left', 'Lane Change Right'])
            cbar.set_label('CNN Output Label')

            plt.show()


    if end_maneuver == 0:
        fp_length = sum_false_positives(model_pred_list, 10)
        earliness = max(0, (50-fp_length)/50)
        earliness_list.append(earliness)

    if end_maneuver == model_pred_list[-1]:
        ture_end = 1
    else:
        ture_end = 0
    if ture_end + earliness ==0:
        pre_quality = 0
    else:
        pre_quality = (2*ture_end*earliness)/(ture_end + earliness)

    pre_quality_list.append(pre_quality)

# Calculate the accuracy
overall_accuracy = correct_predictions / total_examples
print(f"Test Overall Accuracy: {100 * overall_accuracy:.2f}%")

overall_pre_quality = sum(pre_quality_list) / total_examples
print('Test Overall Prediction Quality', overall_pre_quality)


cm_all = confusion_matrix(ground_truth, cnn_pred_list)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues")


# Create line plot
plt.figure(figsize=(10, 6))
plt.plot(earliness_list, marker='o', linestyle='-', color='b', alpha=0.7)
plt.title('Line Plot of Earliness')
plt.xlabel('Index')
plt.ylabel('Earliness')
plt.grid(True)
plt.show()

