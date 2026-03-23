"""
Extract ego and surrounding vehicle states from AD4CHE dataset - WITH LANE KEEPING
- Lane change extraction (same as before)
- Lane keeping extraction (2x the number of lane changes)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATA_PATH = Path("../AD4CHE/AD4CHE_Data_V1.0")


def load_recording_meta(recording_id):
    """Load recording metadata"""
    file_path = DATA_PATH / f"{recording_id}_recordingMeta.csv"
    return pd.read_csv(file_path)


def load_tracks_meta(recording_id):
    """Load vehicle summary statistics"""
    file_path = DATA_PATH / f"{recording_id}_tracksMeta.csv"
    return pd.read_csv(file_path)


def load_tracks(recording_id):
    """Load detailed trajectory data"""
    file_path = DATA_PATH / f"{recording_id}_tracks.csv"
    return pd.read_csv(file_path)


def get_surrounding_vehicle_ids(ego_data_row):
    """Get surrounding vehicle IDs from ego vehicle data row"""
    surrounding_cols = [
        'precedingId', 'followingId',
        'leftPrecedingId', 'leftAlongsideId', 'leftFollowingId',
        'rightPrecedingId', 'rightAlongsideId', 'rightFollowingId'
    ]

    surrounding_ids = []
    for col in surrounding_cols:
        vehicle_id = ego_data_row[col]
        if vehicle_id > 0:
            surrounding_ids.append(int(vehicle_id))

    return surrounding_ids


def detect_lane_change_start(ego_trajectory, frame_rate=30):
    """
    Detect the frame where lane change STARTS
    Returns frame number where laneId first changes, None if no valid lane change
    """
    ego_trajectory = ego_trajectory.sort_values('frame').reset_index(drop=True)

    # Check if any frame has intersection lane (laneId >= 100)
    if (ego_trajectory['laneId'] >= 100).any():
        return None

    # Find all lane changes
    lane_changes = []
    prev_lane = None

    for idx, row in ego_trajectory.iterrows():
        current_lane = row['laneId']
        if prev_lane is not None and current_lane != prev_lane:
            lane_changes.append(row['frame'])
        prev_lane = current_lane

    # Must have exactly 1 lane change
    if len(lane_changes) != 1:
        return None

    return lane_changes[0]


def is_valid_time_window(ego_trajectory, center_frame, frame_rate=30,
                         frames_before=120, frames_after=60):
    """Check if we have valid time window around center frame"""
    start_frame = center_frame - frames_before
    end_frame = center_frame + frames_after

    available_frames = ego_trajectory['frame'].values
    min_frame = available_frames.min()
    max_frame = available_frames.max()

    if start_frame < min_frame or end_frame > max_frame:
        return False, None, None

    return True, start_frame, end_frame


def extract_states_for_window(recording_id, ego_id, center_frame, start_frame, end_frame,
                              tracks_by_vehicle, tracks_by_frame, frame_rate,
                              trajectory_type, lane_change_direction=None):
    """Extract vehicle states for a given time window"""
    events = []

    ego_trajectory = tracks_by_vehicle[ego_id]
    window_trajectory = ego_trajectory[
        (ego_trajectory['frame'] >= start_frame) &
        (ego_trajectory['frame'] <= end_frame)
    ]

    for frame_idx, ego_row in window_trajectory.iterrows():
        frame = ego_row['frame']

        # Extract ego vehicle state
        ego_state = {
            'x': ego_row['x'],
            'y': ego_row['y'],
            'width': ego_row['width'],
            'height': ego_row['height'],
            'xVelocity': ego_row['xVelocity'],
            'yVelocity': ego_row['yVelocity'],
            'xAcceleration': ego_row['xAcceleration'],
            'yAcceleration': ego_row['yAcceleration'],
            'laneId': ego_row['laneId']
        }

        # Get surrounding vehicle IDs
        surrounding_ids = get_surrounding_vehicle_ids(ego_row)

        # Get vehicles in this frame
        if frame in tracks_by_frame:
            frame_data = tracks_by_frame[frame]
            frame_vehicles = {row['id']: row for _, row in frame_data.iterrows()}
        else:
            frame_vehicles = {}

        # Extract surrounding vehicles' states
        surrounding_vehicles = []
        for surr_id in surrounding_ids:
            if surr_id in frame_vehicles:
                surr_row = frame_vehicles[surr_id]
                surr_state = {
                    'id': surr_id,
                    'x': surr_row['x'],
                    'y': surr_row['y'],
                    'width': surr_row['width'],
                    'height': surr_row['height'],
                    'xVelocity': surr_row['xVelocity'],
                    'yVelocity': surr_row['yVelocity'],
                    'xAcceleration': surr_row['xAcceleration'],
                    'yAcceleration': surr_row['yAcceleration'],
                    'laneId': surr_row['laneId']
                }
                surrounding_vehicles.append(surr_state)

        # Store this frame's data
        event_data = {
            'recording_id': recording_id,
            'ego_vehicle_id': ego_id,
            'trajectory_type': trajectory_type,
            'center_frame': center_frame,
            'lane_change_direction': lane_change_direction,
            'frame': frame,
            'time_relative_to_center': (frame - center_frame) / frame_rate,
            'ego_state': ego_state,
            'surrounding_vehicles': surrounding_vehicles
        }
        events.append(event_data)

    return events


def extract_lane_change_trajectories(recording_id, max_lane_changes=None, frame_rate=30):
    """Extract lane change trajectories (same as before)"""
    print(f"\n{'=' * 80}")
    print(f"Processing Lane Changes - Recording {recording_id}")
    print(f"{'=' * 80}")

    # Load metadata
    recording_meta = load_recording_meta(recording_id)
    frame_rate = int(recording_meta['frameRate'].iloc[0])

    tracks_meta = load_tracks_meta(recording_id)

    # Filter for lane changers with drivingDirection == 2
    lane_changers = tracks_meta[
        (tracks_meta['numLaneChanges'] > 0) &
        (tracks_meta['drivingDirection'] == 2)
    ].copy()

    if max_lane_changes is not None and len(lane_changers) > max_lane_changes:
        lane_changers = lane_changers.head(max_lane_changes)

    print(f"Found {len(lane_changers)} vehicles with lane changes (drivingDirection=2)")

    # Load and pre-group tracks
    tracks = load_tracks(recording_id)
    print("Pre-grouping data...")
    tracks_by_vehicle = {vehicle_id: group for vehicle_id, group in tracks.groupby('id')}
    tracks_by_frame = {frame: group for frame, group in tracks.groupby('frame')}

    # Extract lane changes
    lane_change_events = []
    valid_count = 0
    rejected_intersection = 0
    rejected_multiple_lc = 0
    rejected_insufficient_data = 0

    for counter, (idx, vehicle) in enumerate(tqdm(lane_changers.iterrows(),
                                                   total=len(lane_changers),
                                                   desc=f"LC Recording {recording_id}"), start=1):
        ego_id = vehicle['id']

        if ego_id not in tracks_by_vehicle:
            continue

        ego_trajectory = tracks_by_vehicle[ego_id].sort_values('frame')

        # Detect lane change start
        lc_start_frame = detect_lane_change_start(ego_trajectory, frame_rate)

        if lc_start_frame is None:
            if (ego_trajectory['laneId'] >= 100).any():
                rejected_intersection += 1
            else:
                rejected_multiple_lc += 1
            continue

        # Determine lane change direction
        lc_frame_data = ego_trajectory[ego_trajectory['frame'] == lc_start_frame].iloc[0]
        prev_frame_data = ego_trajectory[ego_trajectory['frame'] < lc_start_frame].iloc[-1]
        lane_change_direction = 1 if lc_frame_data['laneId'] > prev_frame_data['laneId'] else 2

        # Check time window (4s before, 2s after)
        is_valid, start_frame, end_frame = is_valid_time_window(
            ego_trajectory, lc_start_frame, frame_rate,
            frames_before=4*frame_rate, frames_after=2*frame_rate
        )

        if not is_valid:
            rejected_insufficient_data += 1
            continue

        valid_count += 1

        # Extract states
        events = extract_states_for_window(
            recording_id, ego_id, lc_start_frame, start_frame, end_frame,
            tracks_by_vehicle, tracks_by_frame, frame_rate,
            trajectory_type='lane_change',
            lane_change_direction=lane_change_direction
        )
        lane_change_events.extend(events)

    print(f"\nLane Change Summary:")
    print(f"Valid: {valid_count}")
    print(f"Rejected - intersection: {rejected_intersection}")
    print(f"Rejected - multiple LC: {rejected_multiple_lc}")
    print(f"Rejected - insufficient data: {rejected_insufficient_data}")
    print(f"Total frames: {len(lane_change_events)}")

    return lane_change_events, valid_count


def extract_lane_keeping_trajectories(recording_id, target_count, frame_rate=30):
    """Extract lane keeping trajectories (6 seconds each)"""
    print(f"\n{'=' * 80}")
    print(f"Processing Lane Keeping - Recording {recording_id}")
    print(f"{'=' * 80}")
    print(f"Target: {target_count} lane keeping trajectories")

    # Load data
    tracks_meta = load_tracks_meta(recording_id)

    # Prioritize vehicles with NO lane changes and drivingDirection == 2
    lane_keepers = tracks_meta[
        (tracks_meta['numLaneChanges'] == 0) &
        (tracks_meta['drivingDirection'] == 2)
        ].copy()

    # Sort by numFrames to get longer trajectories first
    lane_keepers = lane_keepers.sort_values('numFrames', ascending=False)

    print(f"Found {len(lane_keepers)} vehicles with no lane changes (drivingDirection=2)")

    # Load and pre-group tracks
    tracks = load_tracks(recording_id)
    tracks_by_vehicle = {vehicle_id: group for vehicle_id, group in tracks.groupby('id')}
    tracks_by_frame = {frame: group for frame, group in tracks.groupby('frame')}

    lane_keeping_events = []
    valid_count = 0

    # 6 seconds total window (same as lane change)
    frames_before = 3 * frame_rate  # 3 seconds before
    frames_after = 3 * frame_rate  # 3 seconds after
    window_size = frames_before + frames_after  # 6 seconds total

    for idx, vehicle in tqdm(lane_keepers.iterrows(),
                             total=len(lane_keepers),
                             desc=f"LK Recording {recording_id}"):
        if valid_count >= target_count:
            break

        ego_id = vehicle['id']

        if ego_id not in tracks_by_vehicle:
            continue

        ego_trajectory = tracks_by_vehicle[ego_id].sort_values('frame')

        # Sample non-overlapping windows from this trajectory
        available_frames = ego_trajectory['frame'].values

        # Start from the beginning and sample windows
        # Center frame should be frames_before from the start
        current_center = available_frames[0] + frames_before

        while current_center + frames_after <= available_frames[-1] and valid_count < target_count:
            # Check if this window is valid
            is_valid, start_frame, end_frame = is_valid_time_window(
                ego_trajectory, current_center, frame_rate,
                frames_before=frames_before,
                frames_after=frames_after
            )

            if is_valid:
                # Verify lane stays constant in this window
                window_data = ego_trajectory[
                    (ego_trajectory['frame'] >= start_frame) &
                    (ego_trajectory['frame'] <= end_frame)
                    ]

                # Check: only one unique lane ID and no intersection lanes
                unique_lanes = window_data['laneId'].unique()
                if len(unique_lanes) == 1 and unique_lanes[0] < 100:
                    # Extract states
                    events = extract_states_for_window(
                        recording_id, ego_id, current_center, start_frame, end_frame,
                        tracks_by_vehicle, tracks_by_frame, frame_rate,
                        trajectory_type='lane_keeping',
                        lane_change_direction=None
                    )
                    lane_keeping_events.extend(events)
                    valid_count += 1

                    # Move to next non-overlapping window
                    current_center += window_size
                else:
                    # Skip this window and try next one (move by 1 second)
                    current_center += frame_rate
            else:
                # Move forward by 1 second
                current_center += frame_rate

    print(f"\nLane Keeping Summary:")
    print(f"Valid: {valid_count}")
    print(f"Total frames: {len(lane_keeping_events)}")

    return lane_keeping_events, valid_count


def save_to_csv(all_events, output_file):
    """Save extracted data to CSV format"""
    rows = []

    for event in all_events:
        recording_id = event['recording_id']
        ego_id = event['ego_vehicle_id']
        trajectory_type = event['trajectory_type']
        center_frame = event['center_frame']
        lane_change_direction = event.get('lane_change_direction', None)
        frame = event['frame']
        time_rel = event['time_relative_to_center']
        ego = event['ego_state']

        # Ego vehicle row
        rows.append({
            'recording_id': recording_id,
            'ego_vehicle_id': ego_id,
            'trajectory_type': trajectory_type,
            'center_frame': center_frame,
            'lane_change_direction': lane_change_direction,
            'frame': frame,
            'time_relative_to_center': time_rel,
            'vehicle_id': ego_id,
            'vehicle_type': 'ego',
            'x': ego['x'],
            'y': ego['y'],
            'width': ego['width'],
            'height': ego['height'],
            'xVelocity': ego['xVelocity'],
            'yVelocity': ego['yVelocity'],
            'xAcceleration': ego['xAcceleration'],
            'yAcceleration': ego['yAcceleration'],
            'laneId': ego['laneId']
        })

        # Surrounding vehicles rows
        for surr in event['surrounding_vehicles']:
            rows.append({
                'recording_id': recording_id,
                'ego_vehicle_id': ego_id,
                'trajectory_type': trajectory_type,
                'center_frame': center_frame,
                'lane_change_direction': lane_change_direction,
                'frame': frame,
                'time_relative_to_center': time_rel,
                'vehicle_id': surr['id'],
                'vehicle_type': 'surrounding',
                'x': surr['x'],
                'y': surr['y'],
                'width': surr['width'],
                'height': surr['height'],
                'xVelocity': surr['xVelocity'],
                'yVelocity': surr['yVelocity'],
                'xAcceleration': surr['xAcceleration'],
                'yAcceleration': surr['yAcceleration'],
                'laneId': surr['laneId']
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nSaved data to {output_file}")
    return df


if __name__ == "__main__":
    # Process recordings
    recording_ids = [f"{i:02d}" for i in range(1, 41)]  # First 40 recordings

    print("="*80)
    print("AD4CHE DATASET - LANE CHANGE + LANE KEEPING EXTRACTION")
    print("="*80)
    print(f"Recordings to process: {len(recording_ids)}")
    print("="*80)

    all_lc_events = []
    all_lk_events = []
    total_lc_count = 0
    total_lk_count = 0

    for recording_id in recording_ids:
        try:
            # Extract lane changes
            lc_events, lc_count = extract_lane_change_trajectories(
                recording_id,
                max_lane_changes=None
            )
            all_lc_events.extend(lc_events)
            total_lc_count += lc_count

            # Extract lane keeping (2x the lane changes)
            target_lk = lc_count * 2
            lk_events, lk_count = extract_lane_keeping_trajectories(
                recording_id,
                target_count=target_lk,
                frame_rate=30
            )
            all_lk_events.extend(lk_events)
            total_lk_count += lk_count

        except Exception as e:
            print(f"ERROR processing recording {recording_id}: {e}")
            continue

    # Combine both types
    all_events = all_lc_events + all_lk_events

    # Save to CSV
    df = save_to_csv(all_events, output_file="vehicle_states_LC_and_LK_40_recordings.csv")

    # Print final statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total recordings processed: {len(df['recording_id'].unique())}")
    print(f"\nLane Change Events: {total_lc_count}")
    print(f"Lane Keeping Events: {total_lk_count}")
    print(f"LK/LC Ratio: {total_lk_count/total_lc_count if total_lc_count > 0 else 0:.2f}")
    print(f"\nTotal trajectory events: {len(df[['recording_id', 'ego_vehicle_id', 'center_frame']].drop_duplicates())}")
    print(f"Total frames: {len(df[df['vehicle_type']=='ego'])}")
    print(f"Total rows in CSV: {len(df)}")

    # Breakdown by type
    lc_df = df[df['trajectory_type'] == 'lane_change']
    lk_df = df[df['trajectory_type'] == 'lane_keeping']

    print(f"\nLane Change frames: {len(lc_df[lc_df['vehicle_type']=='ego'])}")
    print(f"Lane Keeping frames: {len(lk_df[lk_df['vehicle_type']=='ego'])}")
    print("="*80)