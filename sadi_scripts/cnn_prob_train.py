'''this script is to get the cnn model probability output on training trajectories. the output is used to train the cell map in other script.'''

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.utils.data import TensorDataset, DataLoader


start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def relabel_trajectory(trajectory_data):
    feature_sequence, labels = trajectory_data
    end_label = labels[-1]

    # Create a copy of the features to keep them unchanged
    new_features = feature_sequence.copy()

    # Create a new label array
    new_labels = np.array(labels.copy())  # Ensure it's a NumPy array

    if end_label in [1, 2]:  # Lane Change trajectories
        # Set first 100 steps as Lane Keeping (0)
        new_labels[0:100] = np.zeros(100)  # Create an array of zeros

        # Set last 50 steps as the end_label (either 1 or 2)
        new_labels[100:150] = np.full(50, end_label)  # Create an array filled with end_label

    # For Lane Keeping trajectories (end_label == 0), keep original labels

    return (new_features, new_labels)


# load data into pytorch tensor
input_data = []
for i in range(25, 45):
    idx_str = '{0:02}'.format(i)
    pickle_in = open("../output_normalized_exe_labeled/result" + idx_str + ".pickle", "rb")
    temp_data = pickle.load(pickle_in)
    print("Loaded " + idx_str + " data pack")

    # Apply relabeling to each trajectory in temp_data before extending
    relabeled_temp_data = [relabel_trajectory(traj) for traj in temp_data]
    input_data.extend(relabeled_temp_data)

size = len(input_data)
print('total number of trajectories', size)


size = len(input_data)
#random.shuffle(input_data)
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
model.load_state_dict(torch.load("../auto_label_models/cnn_model_prediction_horizon_2.0s.pt"))
model.eval()
model.to(device)

current_processing = 0
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

# Append results to the lists
all_modified_UCT_values = []
all_exe_time_pred = []
all_exe_time_truth = []
prob_output_list = []
for feature_sequence, labels in input_data:
    print('current_processing', current_processing)
    current_processing+=1
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
    with torch.no_grad():
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



    '''block for probability plotting'''
    predictions_prob_meta.append(np.concatenate(predicted_probs_list_single, axis=0))

    # #print('len(predicted_probs_list_single)', predicted_probs_list_single)
    # # print('len(CNN_output)', len(CNN_output))
    # # print('len(labels)', len(labels))
    #
    # # Convert the list of arrays into a single NumPy array for easier plotting
    # predicted_probs_array = np.vstack(predicted_probs_list_single)
    #
    # # Extract the probabilities for each class
    # lk_probs = predicted_probs_array[:, 0]  # Probabilities for LK (Lane Keep)
    # lcl_probs = predicted_probs_array[:, 1]  # Probabilities for LCL (Lane Change Left)
    # lcr_probs = predicted_probs_array[:, 2]  # Probabilities for LCR (Lane Change Right)
    #
    # # Pad the beginning of the probabilities with NaNs to shift them to start at timestep 24
    # padding_length = 24
    # padded_lk_probs = np.concatenate((np.full(padding_length, np.nan), lk_probs))
    # padded_lcl_probs = np.concatenate((np.full(padding_length, np.nan), lcl_probs))
    # padded_lcr_probs = np.concatenate((np.full(padding_length, np.nan), lcr_probs))
    #
    # # Generate time steps (0 to 149 for 150 timesteps)
    # time_steps = np.arange(150)
    #
    # # Plotting the probabilities over time
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_steps, padded_lk_probs, label='LK Probability', color='blue')
    # plt.plot(time_steps, padded_lcl_probs, label='LCL Probability', color='green')
    # plt.plot(time_steps, padded_lcr_probs, label='LCR Probability', color='red')
    #
    # plt.title('Class Probabilities Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.grid(True)
    # plt.xlim(0, 150)  # Set x-axis limits to cover all 150 timesteps
    # plt.show()



    # The prediction begins at 50th timestep
    CNN_output = [0] * 24 + CNN_output
    # Extract the cnn output at each trajectory at the interested timestep
    cnn_pred_list.append(CNN_output[-1])

    correct_predictions += np.sum(CNN_output[-1] == labels[-1])
    total_examples += 1



# Calculate the accuracy
overall_accuracy = correct_predictions / total_examples
print(f"Test Overall Accuracy: {100 * overall_accuracy:.2f}%")


# Define the folder name where you want to save the data
output_folder = "cell_maps_train_data"  # You can change this name as needed

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the full path for saving the file
save_path = os.path.join(output_folder, 'cnn_probs_and_truth_2.0s_march.pkl')

# Save the data
data_to_save = {
    'predictions_probabilities': predictions_prob_meta,
    'ground_truth': ground_truth_each_timestep
}

with open(save_path, 'wb') as f:
    pickle.dump(data_to_save, f)

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")



