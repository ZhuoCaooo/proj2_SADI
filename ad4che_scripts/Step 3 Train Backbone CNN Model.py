'''
Step 3: Train Backbone CNN Model
- Input: ad4che_sequences.pickle (from Step 2)
- Window Length: 30
- Stride: 15
- Output: Trained PyTorch model (.pt)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "ad4che_sequences/ad4che_sequences.pickle"
MODEL_SAVE_PATH = "ad4che_backbone_cnn.pt"

# Per user request:
WINDOW_LENGTH = 30
STRIDE = 15

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
STATE_DIM = 8  # x, y, width, height, vx, vy, ax, ay
NUM_CLASSES = 3 # 0: LK, 1: LCL, 2: LCR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_windowed_dataset(pickle_file, window_len, stride):
    with open(pickle_file, 'rb') as f:
        sequences = pickle.load(f)

    X_windows = []
    y_labels = []

    for features, labels in sequences:
        num_timesteps = features.shape[0]

        # Slide the window across the 150-timestep trajectory
        for start in range(0, num_timesteps - window_len + 1, stride):
            end = start + window_len

            # Extract window
            window_feat = features[start:end, :]
            # Label for the window is the label at the end of the window
            window_label = labels[end-1]

            X_windows.append(window_feat)
            y_labels.append(window_label)

    X_train = torch.FloatTensor(np.array(X_windows))
    y_train = torch.LongTensor(np.array(y_labels))

    return TensorDataset(X_train, y_train)


def plot_confusion_matrix(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix: AD4CHE Backbone CNN')
    plt.show()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class AD4CHE_CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AD4CHE_CNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # Calculate flatten size based on WINDOW_LENGTH
        # After two MaxPool(2) operations, sequence length becomes window_len // 4
        self.fc_input_dim = 64 * (WINDOW_LENGTH // 4)

        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (Batch, Seq_Len, Feat) -> Transform to (Batch, Feat, Seq_Len)
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================================
# TRAINING LOOP
# ============================================================================

if __name__ == "__main__":
    print(f"Loading data and creating windows (Length: {WINDOW_LENGTH}, Stride: {STRIDE})...")
    dataset = create_windowed_dataset(INPUT_FILE, WINDOW_LENGTH, STRIDE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AD4CHE_CNN(STATE_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Plot Confusion Matrix
    class_names = ['LK', 'LCL', 'LCR']
    print("\nGenerating Confusion Matrix...")
    # Using the train_loader for demonstration; ideally, use a separate test_loader
    plot_confusion_matrix(model, train_loader, device, class_names)