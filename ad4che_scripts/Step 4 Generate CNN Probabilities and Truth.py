'''
Step 4: Generate CNN Probs and Truth for Cell Map Training
- Input: Trained backbone (ad4che_backbone_cnn.pt) and sequences
- Parameters: Window=30, Stride=15 (to match Step 3 training)
- Output: Pickle file containing probabilities and ground truth
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model from Step 3
MODEL_PATH = "ad4che_backbone_cnn.pt"
# Sequences from Step 2
INPUT_SEQUENCES = "ad4che_sequences/ad4che_sequences.pickle"
# Output file for next steps
OUTPUT_FILE = "ad4che_cnn_probs_and_truth.pkl"

# Parameters must match Step 3
WINDOW_LENGTH = 30
STRIDE = 10
STATE_DIM = 8
NUM_CLASSES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL DEFINITION (Same architecture as Step 3)
# ============================================================================
class AD4CHE_CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AD4CHE_CNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc_input_dim = 64 * (WINDOW_LENGTH // 4)
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# GENERATION LOGIC
# ============================================================================

def generate_probs():
    # 1. Load Model
    model = AD4CHE_CNN(STATE_DIM, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # 2. Load Sequences
    with open(INPUT_SEQUENCES, 'rb') as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} trajectories from {INPUT_SEQUENCES}")

    all_results = []

    # 3. Inference loop
    with torch.no_grad():
        for i, (features, labels) in enumerate(sequences):
            traj_probs = []
            traj_truth = []

            num_timesteps = features.shape[0]

            # Slide window
            for start in range(0, num_timesteps - WINDOW_LENGTH + 1, STRIDE):
                end = start + WINDOW_LENGTH

                # Prepare window
                window_feat = torch.FloatTensor(features[start:end, :]).unsqueeze(0).to(device)
                window_truth = labels[end - 1]

                # Get soft probabilities (Softmax)
                outputs = model(window_feat)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

                traj_probs.append(probs)
                traj_truth.append(window_truth)

            # Store results for this trajectory
            # Each entry: [ [prob_t1, prob_t2...], [truth_t1, truth_t2...] ]
            all_results.append([np.array(traj_probs), np.array(traj_truth)])

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(sequences)} trajectories...")

    # 4. Save
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nSaved probabilities and truth to {OUTPUT_FILE}")
    print(f"Format: List of {len(all_results)} trajectories.")
    print(f"Each trajectory has {len(traj_probs)} window steps.")


if __name__ == "__main__":
    generate_probs()