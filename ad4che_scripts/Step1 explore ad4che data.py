'''Step 1: Explore AD4CHE dataset structure'''

import pandas as pd
import numpy as np

# Load the CSV file
csv_file = "vehicle_states_LC_and_LK_40_recordings.csv"
df = pd.read_csv(csv_file)

print("=" * 80)
print("Dataset Overview")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nDataset statistics:")
print(df.describe())

# Check for unique identifiers
print("\n" + "=" * 80)
print("Unique Identifiers")
print("=" * 80)
if 'vehicle_id' in df.columns or 'id' in df.columns:
    id_col = 'vehicle_id' if 'vehicle_id' in df.columns else 'id'
    print(f"Unique vehicles: {df[id_col].nunique()}")
    print(f"Average trajectory length: {len(df) / df[id_col].nunique():.1f} timesteps")

# Check for trajectory/recording IDs
if 'recording_id' in df.columns or 'trajectory_id' in df.columns:
    traj_col = 'recording_id' if 'recording_id' in df.columns else 'trajectory_id'
    print(f"Unique trajectories: {df[traj_col].nunique()}")

# Check for lane change labels
print("\n" + "=" * 80)
print("Lane Change Labels")
print("=" * 80)
label_candidates = ['label', 'maneuver', 'lane_change', 'direction', 'class']
for col in df.columns:
    if any(candidate in col.lower() for candidate in label_candidates):
        print(f"\n'{col}' values:")
        print(df[col].value_counts())

# Check for key features
print("\n" + "=" * 80)
print("Available Features")
print("=" * 80)
feature_candidates = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'velocity', 'acceleration',
                      'lateral', 'longitudinal', 'position', 'speed', 'heading']
available_features = []
for col in df.columns:
    if any(candidate in col.lower() for candidate in feature_candidates):
        available_features.append(col)

print(f"Detected feature columns: {available_features}")

# Save summary
print("\n" + "=" * 80)
print("Saving column information...")
print("=" * 80)
with open('ad4che_column_info.txt', 'w') as f:
    f.write("Column Information:\n")
    f.write("=" * 80 + "\n")
    for col in df.columns:
        f.write(f"{col}: {df[col].dtype}\n")

print("Column info saved to 'ad4che_column_info.txt'")
print("\nNext steps:")
print("1. Review the output above")
print("2. Identify which columns to use as features")
print("3. Identify the label column for lane change direction")
print("4. Run step2 script to convert to sequences")