import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data into pytorch tensor
input_data = []
for i in range(45, 61):
    idx_str = '{0:02}'.format(i)
    pickle_in = open("../output_normalized_exe_labeled/result" + idx_str + ".pickle", "rb")
    temp_data = pickle.load(pickle_in)
    print("Loaded " + idx_str + " data pack")
    input_data.extend(temp_data)

size = len(input_data)
print('total number of trajectories', size)

# Extract trajectories by maneuver type
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

# Read the MATLAB labeling results
mat_results_m1 = pd.read_csv('../mat_scripts/500_tra_peaks_m1_test.csv', header=None)
mat_results_m1 = np.array(mat_results_m1)[0]
mat_results_m2 = pd.read_csv('../mat_scripts/500_tra_peaks_m2_test.csv', header=None)
mat_results_m2 = np.array(mat_results_m2)[0]
relabeled_traj = []

# Prediction advance time, in seconds
advanced_time = 1.0
label_adjustment = int(30 + advanced_time * 25)
print(len(maneuver_1_traj), 'len(maneuver_1_traj)')
print(len(maneuver_2_traj), 'len(maneuver_2_traj)')

# Process lane keeping trajectories
for traj in range(0, len(maneuver_0_traj)):
    veh_info = maneuver_0_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_0_traj[traj][1]  # Maneuver info
    # Append the trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

# Process left lane change trajectories
for traj in range(0, len(maneuver_1_traj)):
    veh_info = maneuver_1_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_1_traj[traj][1]  # Maneuver info

    # Find label transition point
    label_index = min(149, max(int(mat_results_m1[traj]) - label_adjustment, 0))

    # Create the iterables to assign to slices
    zeros = [0] * (label_index + 1)  # List of zeros for indices 0 to label_index
    ones = [1] * (150 - (label_index + 1))  # List of ones for indices from label_index + 1 to 149

    # Assign the iterables to the maneuver_info
    maneuver_info[:label_index + 1] = zeros
    maneuver_info[label_index + 1:] = ones

    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

# Process right lane change trajectories
for traj in range(0, len(maneuver_2_traj)):
    veh_info = maneuver_2_traj[traj][0]  # Vehicle info
    maneuver_info = maneuver_2_traj[traj][1]  # Maneuver info

    # Find label transition point
    label_index = min(149, max(int(mat_results_m2[traj]) - label_adjustment, 0))

    # Create the iterables to assign to slices
    zeros = [0] * (label_index + 1)  # List of zeros for indices 0 to label_index
    twos = [2] * (150 - (label_index + 1))  # List of twos for indices from label_index + 1 to 149

    # Assign the iterables to the maneuver_info
    maneuver_info[:label_index + 1] = zeros
    maneuver_info[label_index + 1:] = twos

    # Append the updated trajectory information to relabeled_traj
    relabeled_traj.append((veh_info, maneuver_info))

input_data = relabeled_traj

size = len(input_data)
print('total number of trajectories after relabeling', size)

window_length = 25
stride = 1


# Define the CNN model architecture
class CNN1D(nn.Module):
    def __init__(self, num_features, sequence_len, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self._calculate_fc_input_features(num_features, sequence_len), out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # swap the temporal and feature dimensions
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _calculate_fc_input_features(self, num_features, sequence_len):
        with torch.no_grad():
            x = torch.zeros((1, num_features, sequence_len))
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.flatten(x)
            return x.numel()

# Ensemble class for sequential prediction
class EnsembleCNN:
    def __init__(self, num_models, num_features, sequence_len, num_classes):
        self.num_models = num_models
        self.models = []

        # Create multiple models with different initializations
        for i in range(num_models):
            model = CNN1D(num_features, sequence_len, num_classes)
            model.to(device)
            self.models.append(model)

    def predict_single_window(self, window):
        """
        Predict a single window using all models in the ensemble
        Returns: Ensemble prediction probabilities and uncertainty
        """
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # Add batch dimension
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(window_tensor)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        # Stack model predictions (shape: [num_models, 1, num_classes])
        all_probs = np.stack(all_probs)

        # Calculate mean predictions (shape: [1, num_classes])
        mean_probs = np.mean(all_probs, axis=0)

        # Calculate variance across models for each class
        model_variance = np.var(all_probs, axis=0)  # [1, num_classes]

        # Calculate entropy of mean prediction
        epsilon = 1e-10  # Small value to avoid log(0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)

        # Calculate average variance across classes
        avg_variance = np.mean(model_variance, axis=1)

        # Combine entropy and variance as uncertainty measure
        uncertainty = (entropy + avg_variance) / 2

        return mean_probs[0], uncertainty[0]  # Remove batch dimension

    def load_models(self, directory):
        """Load all models in the ensemble"""
        # Load ensemble metadata
        ensemble_info = torch.load(os.path.join(directory, "ensemble_info.pt"))
        self.num_models = ensemble_info['num_models']

        # Load each model
        self.models = []
        for i in range(self.num_models):
            model = CNN1D(num_features=16, sequence_len=25, num_classes=3)
            model.load_state_dict(torch.load(os.path.join(directory, f"ensemble_model_{i}.pt")))
            model.to(device)
            self.models.append(model)

        print(f"Loaded {self.num_models} models from {directory}")


def sequential_trajectory_inference(ensemble, trajectory_data, window_length=25):
    """
    Process each trajectory sequentially, one timestep at a time,
    simulating real-world online processing
    """
    # Lists to store results for all trajectories
    all_true_labels = []
    all_predicted_classes = []
    all_reliabilities = []
    all_probabilities = []

    for traj_idx, (feature_sequence, labels) in enumerate(trajectory_data):
        print(f"Processing trajectory {traj_idx + 1}/{len(trajectory_data)}")

        # Convert to numpy if not already
        feature_sequence = np.array(feature_sequence)
        labels = np.array(labels)

        # Lists to store results for current trajectory
        traj_true_labels = []
        traj_predicted_classes = []
        traj_probabilities = []
        traj_uncertainties = []

        # For each timestep (starting at window_length-1 to have enough history)
        for t in range(window_length - 1, len(feature_sequence)):
            # Extract the window ending at current timestep
            window = feature_sequence[t - window_length + 1:t + 1]

            # Get prediction for this window
            probs, uncertainty = ensemble.predict_single_window(window)

            # Calculate reliability score
            confidence = np.max(probs)
            reliability = confidence * (1 - uncertainty)

            # Store results
            traj_true_labels.append(labels[t])
            traj_predicted_classes.append(np.argmax(probs))
            traj_probabilities.append(probs)
            traj_uncertainties.append(reliability)

        # Store trajectory results
        all_true_labels.append(np.array(traj_true_labels))
        all_predicted_classes.append(np.array(traj_predicted_classes))
        all_reliabilities.append(np.array(traj_uncertainties))
        all_probabilities.append(np.array(traj_probabilities))

    # Format results
    formatted_results = {
        'model_predictions': all_predicted_classes,
        'reliability_scores': all_reliabilities,
        'ground_truth': all_true_labels,
        'probability_outputs': all_probabilities,
    }

    return formatted_results


def analyze_results(formatted_results):
    """
    Analyze results to get confusion matrix and accuracy
    """
    # Flatten all predictions and ground truth
    all_preds = np.concatenate(formatted_results['model_predictions'])
    all_true = np.concatenate(formatted_results['ground_truth'])

    # Calculate confusion matrix
    cm = confusion_matrix(all_true, all_preds)

    # Calculate overall accuracy
    acc = accuracy_score(all_true, all_preds)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nOverall Accuracy: {acc:.4f}")

    # Return metrics
    return {
        'confusion_matrix': cm,
        'accuracy': acc
    }


def save_results(formatted_results, metrics, file_path='sequential_results.pkl'):
    """
    Save formatted results and metrics to a pickle file
    """
    # Add metrics to the results
    results_with_metrics = formatted_results.copy()
    results_with_metrics['metrics'] = metrics

    with open(file_path, 'wb') as f:
        pickle.dump(results_with_metrics, f)
    print(f"Results saved to {file_path}")


# Load the ensemble model
print("Loading ensemble models...")
ensemble_dir = "../auto_label_models/ensemble_prediction_horizon_1.0s"
ensemble = EnsembleCNN(num_models=5, num_features=16, sequence_len=25, num_classes=3)
ensemble.load_models(ensemble_dir)

# Process trajectories sequentially
print("Starting sequential trajectory inference...")
results = sequential_trajectory_inference(ensemble, input_data, window_length=window_length)

# Analyze results
metrics = analyze_results(results)

# Save results
save_results(results, metrics, 'ensemble_sequential_test_results.pkl')

print("Sequential analysis completed!")