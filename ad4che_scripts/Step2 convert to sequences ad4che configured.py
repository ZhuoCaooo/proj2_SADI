'''
Step 2: Convert AD4CHE CSV to sequence data for CNN
- Filters for 'ego' vehicle rows only
- Handles unique trajectory events (recording + vehicle + center_frame)
- Relabels sequences for training (LK=0, LCL=1, LCR=2)
'''

import pandas as pd
import numpy as np
import pickle
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output
CSV_FILE = "vehicle_states_LC_and_LK_test.csv"
OUTPUT_DIR = "ad4che_sequences"
OUTPUT_FILE = "ad4che_sequences_test.pickle"

# Unique trajectory identifiers from your Step 1 script
TRAJECTORY_ID_COLS = ['recording_id', 'ego_vehicle_id', 'center_frame']
VEHICLE_TYPE_COL = 'vehicle_type'
TIME_COL = 'frame'
LABEL_COL = 'lane_change_direction'  # 0=LK (will be filled), 1=LCL, 2=LCR

# Feature columns (8 features)
FEATURE_COLS = [
    'x', 'y',
    'width', 'height',
    'xVelocity', 'yVelocity',
    'xAcceleration', 'yAcceleration'
]

# Sequence parameters
SEQUENCE_LENGTH = 180  # 6 seconds (if 25Hz) or 5 seconds (if 30Hz)
MIN_SEQUENCE_LENGTH = 140

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def load_and_prepare_data(csv_file):
    """Load CSV and prepare for sequence extraction"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 1. CRITICAL: Filter for EGO vehicle only.
    # Neighbor vehicle rows are not used as sequence targets.
    df = df[df[VEHICLE_TYPE_COL] == 'ego'].copy()

    # 2. Fill LK labels: Lane Keeping entries have NaN in lane_change_direction
    df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)

    # 3. Sort to ensure chronological order within each event
    df = df.sort_values(TRAJECTORY_ID_COLS + [TIME_COL])

    print(f"Filtered to {len(df)} ego-vehicle rows.")
    return df

def extract_sequences(df):
    """Extract sequences for each unique trajectory event"""
    print("\nExtracting sequences...")
    sequences = []

    # Group by the unique event identifiers
    groups = df.groupby(TRAJECTORY_ID_COLS)

    total_groups = len(groups)
    processed = 0
    skipped_short = 0

    for group_id, group_df in groups:
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed}/{total_groups} trajectories...")

        # Check if sequence is long enough
        if len(group_df) < MIN_SEQUENCE_LENGTH:
            skipped_short += 1
            continue

        # Extract features
        features = group_df[FEATURE_COLS].values

        # Get the final label (intent) from the last frame
        final_label = int(group_df[LABEL_COL].iloc[-1])

        # Standardize length: Take the last SEQUENCE_LENGTH timesteps
        if len(features) > SEQUENCE_LENGTH:
            features = features[-SEQUENCE_LENGTH:]
            labels = np.full(SEQUENCE_LENGTH, final_label)
        else:
            labels = np.full(len(features), final_label)

        sequences.append((features, labels))

    print(f"\nExtraction summary:")
    print(f"  Total events found: {total_groups}")
    print(f"  Valid sequences extracted: {len(sequences)}")
    print(f"  Skipped (too short): {skipped_short}")

    return sequences

def relabel_for_sliding_window(trajectory_data, lc_timesteps=120):
    """
    Relabeling logic:
    - For LC (1, 2): The first part of the 6s window is LK (0),
      the last 'lc_timesteps' are the LC direction.
    """
    feature_sequence, labels = trajectory_data
    end_label = int(labels[-1])

    new_labels = labels.copy()

    if end_label in [1, 2]:
        # Calculate where the intention starts
        lk_zone = len(labels) - lc_timesteps
        new_labels[:lk_zone] = 0
        new_labels[lk_zone:] = end_label

    return (feature_sequence, new_labels)

def analyze_sequences(sequences):
    """Print distribution and stats"""
    print("\n" + "=" * 80)
    print("Sequence Analysis")
    print("=" * 80)

    label_counts = {0: 0, 1: 0, 2: 0}
    for _, labels in sequences:
        final_label = int(labels[-1])
        label_counts[final_label] += 1

    label_names = {0: 'Lane Keeping (LK)', 1: 'Left Lane Change (LCL)', 2: 'Right Lane Change (LCR)'}
    for label, name in label_names.items():
        count = label_counts[label]
        pct = (count / len(sequences)) * 100 if sequences else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"\nFeature Matrix Shape: {sequences[0][0].shape}")
    return label_counts

def save_sequences(sequences, output_dir, output_file):
    """Save to pickle"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"\nSaved to {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df = load_and_prepare_data(CSV_FILE)

    # Step A: Raw Extraction
    sequences = extract_sequences(df)

    if not sequences:
        print("ERROR: No sequences found. Check your CSV filters.")
        exit()

    # Step B: Relabeling (Simulating the 'transition' from LK to LC)
    print("\nApplying relabeling (transition logic)...")
    sequences = [relabel_for_sliding_window(s) for s in sequences]

    # Step C: Stats & Save
    analyze_sequences(sequences)
    save_sequences(sequences, OUTPUT_DIR, OUTPUT_FILE)
    print("\nDone! Ready for Step 3.")