import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time
import pandas as pd
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# CNN Model with Monte Carlo Dropout
class MCDropoutCNN1D(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.5):
        super(MCDropoutCNN1D, self).__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._calculate_fc_input_features(num_features, 25), 8)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(8, num_classes)

    def _calculate_fc_input_features(self, num_features, window_length):
        with torch.no_grad():
            x = torch.zeros((1, num_features, window_length))
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.flatten(x)
            return x.numel()

    def forward(self, x, enable_dropout=False):
        if enable_dropout:
            self.train()
        else:
            self.eval()

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x) if enable_dropout else x
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x) if enable_dropout else x
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x) if enable_dropout else x
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x) if enable_dropout else x
        x = self.fc2(x)

        return x


def load_trajectory_data(start_idx=45, end_idx=61, max_trajectories=None):
    """
    Load trajectory data from pickle files
    """
    input_data = []
    for i in range(start_idx, end_idx):
        idx_str = '{0:02}'.format(i)
        pickle_in = open(f"../output_normalized_exe_labeled/result{idx_str}.pickle", "rb")
        temp_data = pickle.load(pickle_in)
        print(f"Loaded {idx_str} data pack with {len(temp_data)} trajectories")
        input_data.extend(temp_data)

    print(f'Total number of trajectories: {len(input_data)}')

    # If max_trajectories is specified, limit the number of trajectories
    if max_trajectories and len(input_data) > max_trajectories:
        input_data = input_data[:max_trajectories]
        print(f'Limited to {max_trajectories} trajectories')

    return input_data


def relabel_trajectories(input_data, advanced_time=0):
    """
    Relabel the trajectories based on MATLAB WTMM-HE script results
    """
    # Extract trajectories by maneuver type
    maneuver_0_traj = []
    maneuver_1_traj = []
    maneuver_2_traj = []

    for traj_index in range(len(input_data)):
        maneuver_info = input_data[traj_index][1][-1]

        if maneuver_info == 1 and len(maneuver_1_traj) < 500:
            maneuver_1_traj.append(input_data[traj_index])

        if maneuver_info == 2 and len(maneuver_2_traj) < 500:
            maneuver_2_traj.append(input_data[traj_index])

        if maneuver_info == 0 and len(maneuver_0_traj) < 500:
            maneuver_0_traj.append(input_data[traj_index])

    # Read MATLAB labeling results
    mat_results_m1 = pd.read_csv('../mat_scripts/500_tra_peaks_m1_test.csv', header=None)
    mat_results_m1 = np.array(mat_results_m1)[0]
    mat_results_m2 = pd.read_csv('../mat_scripts/500_tra_peaks_m2_test.csv', header=None)
    mat_results_m2 = np.array(mat_results_m2)[0]

    # Label adjustment (prediction advance time in units of seconds)
    label_adjustment = int(30 + advanced_time * 25)
    print(len(maneuver_1_traj), 'len(maneuver_1_traj)')
    print(len(maneuver_2_traj), 'len(maneuver_2_traj)')
    relabeled_traj = []

    # Process lane-keeping (maneuver 0) trajectories
    for traj in range(len(maneuver_0_traj)):
        veh_info = maneuver_0_traj[traj][0]
        maneuver_info = maneuver_0_traj[traj][1]
        relabeled_traj.append((veh_info, maneuver_info))

    # Process left lane change (maneuver 1) trajectories
    for traj in range(len(maneuver_1_traj)):
        veh_info = maneuver_1_traj[traj][0]
        maneuver_info = maneuver_1_traj[traj][1].copy()  # Create a copy to avoid modifying the original

        # Determine label transition index
        label_index = min(149, max(int(mat_results_m1[traj]) - label_adjustment, 0))

        # Create arrays of zeros and ones
        zeros = [0] * (label_index + 1)
        ones = [1] * (150 - (label_index + 1))

        # Assign the labels
        maneuver_info[:label_index + 1] = zeros
        maneuver_info[label_index + 1:] = ones

        relabeled_traj.append((veh_info, maneuver_info))

    # Process right lane change (maneuver 2) trajectories
    for traj in range(len(maneuver_2_traj)):
        veh_info = maneuver_2_traj[traj][0]
        maneuver_info = maneuver_2_traj[traj][1].copy()  # Create a copy to avoid modifying the original

        # Determine label transition index
        label_index = min(149, max(int(mat_results_m2[traj]) - label_adjustment, 0))

        # Create arrays of zeros and twos
        zeros = [0] * (label_index + 1)
        twos = [2] * (150 - (label_index + 1))

        # Assign the labels
        maneuver_info[:label_index + 1] = zeros
        maneuver_info[label_index + 1:] = twos

        relabeled_traj.append((veh_info, maneuver_info))

    print(f'Total number of trajectories after relabeling: {len(relabeled_traj)}')
    return relabeled_traj


def real_time_mc_dropout_predict(model, input_data, window_length=25, num_samples= None,
                                 simulate_delay=False, device='cuda'):
    """
    Simulate real-time prediction by processing trajectories and timesteps sequentially
    """
    model.to(device)
    all_trajectory_predictions = []
    all_trajectory_uncertainties = []
    all_trajectory_classes = []
    all_trajectory_ground_truth = []
    all_trajectory_reliability = []

    total_processing_time = 0
    num_windows_processed = 0

    for traj_idx, (feature_sequence, labels) in enumerate(input_data):
        print(f"Processing trajectory {traj_idx + 1}/{len(input_data)}")

        # Convert to numpy if not already
        feature_sequence = np.array(feature_sequence)
        labels = np.array(labels)
        traj_length = len(feature_sequence)

        # Store predictions for this trajectory
        traj_predictions = []
        traj_uncertainties = []
        traj_classes = []
        traj_ground_truth = []
        traj_reliability = []

        # Process windows sequentially
        for t in range(traj_length - window_length + 1):
            start_time = time.time()

            # Extract current window
            current_window = feature_sequence[t:t + window_length]
            window_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(device)

            # Record ground truth for this window (using the last timestep's label)
            gt = labels[t + window_length - 1]
            traj_ground_truth.append(gt)

            # Run MC dropout samples
            mc_samples = []
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = model(window_tensor, enable_dropout=True)
                    probabilities = F.softmax(outputs, dim=1)
                    mc_samples.append(probabilities)

            # Stack all MC samples
            mc_samples = torch.cat(mc_samples, dim=0)

            # Calculate mean and uncertainty
            mean_pred = torch.mean(mc_samples, dim=0)
            uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10))

            # Get predicted class
            pred_class = torch.argmax(mean_pred).item()
            traj_classes.append(pred_class)

            # Store predictions
            traj_predictions.append(mean_pred.cpu().numpy()[0])
            traj_uncertainties.append(uncertainty.cpu().item())

            # Calculate reliability score
            confidence = torch.max(mean_pred).item()
            reliability = confidence * (1 - (uncertainty.item() / 4.0))  # Normalize by max uncertainty (log(3) ≈ 1.1)
            traj_reliability.append(reliability)

            # Track processing time
            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            num_windows_processed += 1

            # Simulate real-time delay if requested
            if simulate_delay:
                time.sleep(0.01)  # 10ms delay to simulate real-time processing

        all_trajectory_predictions.append(np.array(traj_predictions))
        all_trajectory_uncertainties.append(np.array(traj_uncertainties))
        all_trajectory_classes.append(np.array(traj_classes))
        all_trajectory_ground_truth.append(np.array(traj_ground_truth))
        all_trajectory_reliability.append(np.array(traj_reliability))

    avg_processing_time = total_processing_time / num_windows_processed if num_windows_processed > 0 else 0
    print(f"Average processing time per window: {avg_processing_time * 1000:.2f} ms")

    return {
        'predictions': all_trajectory_predictions,
        'uncertainties': all_trajectory_uncertainties,
        'classes': all_trajectory_classes,
        'ground_truth': all_trajectory_ground_truth,
        'reliability': all_trajectory_reliability,
    }


def analyze_prediction_results(results):
    """
    Analyze prediction results across all trajectories
    """
    # Get aggregated metrics
    all_gt = np.concatenate(results['ground_truth'])
    all_pred = np.concatenate(results['classes'])
    all_uncertainty = np.concatenate(results['uncertainties'])
    all_reliability = np.concatenate(results['reliability'])

    # Calculate accuracy
    accuracy = np.mean(all_gt == all_pred)

    # Calculate class-specific metrics
    class_names = ['Lane Keeping', 'Left Lane Change', 'Right Lane Change']
    class_accuracies = {}
    class_counts = {}

    for i, name in enumerate(class_names):
        mask = all_gt == i
        if np.sum(mask) > 0:
            class_accuracies[name] = np.mean(all_pred[mask] == all_gt[mask])
            class_counts[name] = np.sum(mask)

    # Calculate confusion matrix
    confusion = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            confusion[i, j] = np.sum((all_gt == i) & (all_pred == j))

    # Print results
    print("\nOverall Results:")
    print(f"Total windows analyzed: {len(all_gt)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print("\nClass-specific accuracy:")
    for name, acc in class_accuracies.items():
        print(f"  {name}: {acc:.4f} ({class_counts[name]} samples)")

    print("\nConfusion Matrix:")
    print("    Predicted class")
    print("      LK   LCL  LCR")
    for i, name in enumerate(['LK ', 'LCL', 'LCR']):
        print(f"True {name}", end=" ")
        for j in range(3):
            print(f"{confusion[i, j]:5.0f}", end="")
        print()

    # Calculate and print uncertainty metrics
    print("\nUncertainty Analysis:")
    correct_uncertainties = all_uncertainty[all_pred == all_gt]
    incorrect_uncertainties = all_uncertainty[all_pred != all_gt]

    print(f"Mean uncertainty for correct predictions: {np.mean(correct_uncertainties):.4f}")
    print(f"Mean uncertainty for incorrect predictions: {np.mean(incorrect_uncertainties):.4f}")

    # Return compiled results
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion': confusion,
        'all_gt': all_gt,
        'all_pred': all_pred,
        'all_uncertainty': all_uncertainty,
        'all_reliability': all_reliability
    }





def format_results_for_sadi_analysis(results):
    """
    Format MC dropout results to match the format expected by the SADI-UQ analysis
    """
    # Format data
    formatted_results = {
        'model_predictions': results['classes'],
        'reliability_scores': results['reliability'],
        'ground_truth': results['ground_truth'],
        'probability_outputs': results['predictions'],
    }

    return formatted_results


def save_results_for_analysis(formatted_results, file_path='real_time_mc_dropout_results.pkl'):
    """
    Save formatted results to a pickle file for external analysis
    """
    with open(file_path, 'wb') as f:
        pickle.dump(formatted_results, f)
    print(f"Results saved to {file_path}")


def main():
    # Load data
    print("Loading and preparing data...")
    input_data = load_trajectory_data(start_idx=45, end_idx=61, max_trajectories=9999)  # Limit for testing

    # Relabel trajectories
    print("Relabeling trajectories...")
    relabeled_data = relabel_trajectories(input_data, advanced_time=0)

    # Load the model
    print("Loading trained model...")
    model = MCDropoutCNN1D(num_features=16, num_classes=3, dropout_rate=0.5)
    checkpoint = torch.load('../auto_label_models/cnn_mc_dropout.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Perform real-time MC dropout prediction
    print("Starting real-time MC dropout prediction simulation...")
    window_length = 25
    results = real_time_mc_dropout_predict(
        model,
        relabeled_data,
        window_length=window_length,
        num_samples=20,  # Number of MC samples
        simulate_delay=False,  # Set to True to add a small delay between predictions
        device=device
    )

    # Format and save results for external analysis
    print("Formatting and saving results for SADI-UQ analysis...")
    formatted_results = format_results_for_sadi_analysis(results)
    save_results_for_analysis(formatted_results, 'real_time_mc_dropout_results.pkl')



if __name__ == "__main__":
    main()