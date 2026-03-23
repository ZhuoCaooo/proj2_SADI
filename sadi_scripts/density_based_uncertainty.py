'''these are the function used while doing the uncertainty testing'''
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


def compute_lateral_speed(lateral_position, time_step=1 / 25):
    """
    Compute lateral speed from lateral position using finite difference.

    Parameters:
    -----------
    lateral_position : array-like
        Array of lateral position values
    time_step : float
        Time interval between consecutive samples (default: 1/25 s)

    Returns:
    --------
    array-like
        Array of lateral speed values
    """
    # Calculate the difference between consecutive positions
    lateral_speed = np.diff(lateral_position)

    # Convert to speed by dividing by time step
    lateral_speed = lateral_speed / time_step

    # Add a zero at the beginning to maintain same array length
    lateral_speed = np.concatenate(([0], lateral_speed))

    # Optional: Apply smoothing (moving average) to reduce noise
    window_size = 3
    kernel = np.ones(window_size) / window_size
    lateral_speed_smooth = np.convolve(lateral_speed, kernel, mode='same')

    return lateral_speed_smooth



def get_cell_distribution_with_density(pred_probs, cell_maps, resolution=0.02):
    """
    Get empirical distributions and cell density from both maps, using raw counts.
    """
    # Handle array shapes
    if isinstance(pred_probs, np.ndarray):
        if len(pred_probs.shape) == 2 and pred_probs.shape[0] == 1:
            pred_probs = pred_probs[0]
    elif isinstance(pred_probs, list) and len(pred_probs) == 1:
        pred_probs = pred_probs[0]

    # Get resolution
    if 'parameters' in cell_maps and 'resolution' in cell_maps['parameters']:
        resolution = cell_maps['parameters']['resolution']

    # Map prediction to cell
    n1 = int(pred_probs[1] / resolution)  # LCL
    n2 = int(pred_probs[2] / resolution)  # LCR
    cell = (n1, n2)

    # Get maps
    lk_map = cell_maps.get('lk_density_map', {})
    lc_map = cell_maps.get('lc_density_map', {})

    # Initialize with different default values
    lk_dist = None
    lc_dist = None
    lk_density = int(max(10, pred_probs[0] * 200))  # Different scaling factor
    lc_density = int(max(10, (pred_probs[1] + pred_probs[2]) * 150))  # Different scaling factor

    # Get LK density - ONLY use the raw count, not pre-normalized values
    if cell in lk_map:
        lk_dist = lk_map[cell]
        if isinstance(lk_dist, dict) and 'count' in lk_dist:
            lk_density = lk_dist['count']
        elif isinstance(lk_dist, (int, float)):
            lk_density = lk_dist

    # Get LC density - ONLY use the raw count, not pre-normalized values
    if cell in lc_map:
        lc_dist = lc_map[cell]
        if isinstance(lc_dist, dict) and 'count' in lc_dist:
            lc_density = lc_dist['count']
        elif isinstance(lc_dist, (int, float)):
            lc_density = lc_dist

    return lk_dist, lc_dist, lk_density, lc_density, cell


def extract_min_max_densities(cell_maps):
    """
    Extract minimum and maximum density values separately for LK and LC maps.

    Parameters:
    -----------
    cell_maps : dict
        The cell maps containing lk_density_map and lc_density_map

    Returns:
    --------
    tuple
        (lk_min_density, lk_max_density, lc_min_density, lc_max_density)
    """
    # Get maps
    lk_map = cell_maps.get('lk_density_map', {})
    lc_map = cell_maps.get('lc_density_map', {})

    # Extract LK density values
    lk_densities = []
    for cell, data in lk_map.items():
        if isinstance(data, dict) and 'count' in data:
            lk_densities.append(data['count'])
        elif isinstance(data, (int, float)):
            lk_densities.append(data)

    # Extract LC density values (separate from LK)
    lc_densities = []
    for cell, data in lc_map.items():
        if isinstance(data, dict) and 'count' in data:
            lc_densities.append(data['count'])
        elif isinstance(data, (int, float)):
            lc_densities.append(data)

    # Filter out zero values and get min/max for each
    lk_positive = [d for d in lk_densities if d > 0]
    lc_positive = [d for d in lc_densities if d > 0]

    # Set defaults if needed
    if not lk_positive:
        lk_min, lk_max = 1, 1000
    else:
        lk_min, lk_max = min(lk_positive), max(lk_positive)

    if not lc_positive:
        lc_min, lc_max = 1, 1000
    else:
        lc_min, lc_max = min(lc_positive), max(lc_positive)

    print(f"LK density range: min={lk_min}, max={lk_max}")
    print(f"LC density range: min={lc_min}, max={lc_max}")

    return lk_min, lk_max, lc_min, lc_max


def assess_prediction_with_density(pred_probs, classification_output,
                                                          similarity_history, cell_maps,
                                                          current_step, total_steps,
                                                          lk_min=None, lk_max=None,
                                                          lc_min=None, lc_max=None):
    """
    Modified assessment using cell density for reliability scoring with separate
    normalization for LK and LC density values.

    Parameters:
    -----------
    pred_probs : array-like
        The prediction probabilities [LK, LCL, LCR]
    classification_output : array-like
        The model's classification output history
    similarity_history : array-like
        The DTW similarity history
    cell_maps : dict
        The cell maps containing density maps
    current_step : int
        The current timestep
    total_steps : int
        The total number of timesteps
    lk_min, lk_max : int, optional
        Min/max density values for LK map
    lc_min, lc_max : int, optional
        Min/max density values for LC map

    Returns:
    --------
    dict
        Assessment results including reliability score
    """
    # If min/max densities aren't provided, calculate them from cell_maps
    if lk_min is None or lk_max is None or lc_min is None or lc_max is None:
        lk_min, lk_max, lc_min, lc_max = extract_min_max_densities(cell_maps)
        # Print once for debugging
        if current_step == 0:
            print(f"Using separate normalization ranges:")
            print(f"  LK: min={lk_min}, max={lk_max}")
            print(f"  LC: min={lc_min}, max={lc_max}")

    # Get base similarity with smoother normalization
    recent_start_idx = max(0, current_step - 25)
    if recent_start_idx < current_step and 0 in classification_output[recent_start_idx:current_step]:
        norm_similarity = normalize_similarity_window(similarity_history)
    else:
        norm_similarity = 1 - normalize_similarity_window(similarity_history)

    # After (without temporal factor)
    base_weight = 1.0  # Adjust to 1.0 since we're removing the temporal component

    lk_weight = max(0.05, base_weight * norm_similarity)
    lc_weight = max(0.05, base_weight * (1 - norm_similarity))

    # Normalize weights
    total = lk_weight + lc_weight
    lk_weight /= total
    lc_weight /= total

    # Handle batch-style predictions
    if isinstance(pred_probs, np.ndarray) and len(pred_probs.shape) == 2:
        pred_probs_to_use = pred_probs[0]
    else:
        pred_probs_to_use = pred_probs

    # Get distributions and densities
    lk_dist, lc_dist, lk_density, lc_density, cell = get_cell_distribution_with_density(
        pred_probs_to_use, cell_maps)

    # Normalize densities using SEPARATE min/max values for LK and LC
    lk_density_score = normalize_density(lk_density, lk_min, lk_max)
    lc_density_score = normalize_density(lc_density, lc_min, lc_max)

    # Calculate weighted reliability score
    reliability_score = (lk_weight * lk_density_score) + (lc_weight * lc_density_score)


    return {
        'cell': cell,
        'lk_weight': float(lk_weight),
        'lc_weight': float(lc_weight),
        'lk_density': int(lk_density),
        'lc_density': int(lc_density),
        'lk_density_score': float(lk_density_score),
        'lc_density_score': float(lc_density_score),
        'reliability_score': float(reliability_score)
    }


def normalize_density(density, min_density, max_density):
    """
    Normalize density with better sensitivity in the middle range.
    Uses a square root transformation instead of logarithmic.
    """
    if density <= 0:
        return 0.0

    # Square root transformation - less compression than log
    # Provides more sensitivity in middle ranges while still handling large ranges
    sqrt_density = np.sqrt(density)
    sqrt_min = np.sqrt(min_density)
    sqrt_max = np.sqrt(max_density)

    # Basic normalization with square root values
    normalized = (sqrt_density - sqrt_min) / (sqrt_max - sqrt_min)

    # Optional: Apply a power function to reshape the curve
    # Values < 1 increase sensitivity in lower ranges
    # Values > 1 increase sensitivity in higher ranges
    power = 0.7  # Less than 1 to give more weight to middle values
    normalized = normalized ** power

    # Ensure the value is clipped to [0, 1]
    normalized = float(np.clip(normalized, 0.0, 1.0))

    return normalized

# Helper function from your original code
def normalize_similarity_window(similarity_values, window_size=50):
    """
    Normalize similarity values with smoother distribution.
    """
    if len(similarity_values) < window_size:
        window_size = len(similarity_values)

    window = similarity_values[-window_size:]

    # Use percentile-based normalization instead of min-max
    p10 = np.percentile(window, 10)
    p90 = np.percentile(window, 90)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    normalized = (similarity_values[-1] - p10) / (p90 - p10 + epsilon)

    # Apply sigmoid transformation for smoother distribution
    normalized = 1 / (1 + np.exp(-3 * (normalized - 0.5)))

    return np.clip(normalized, 0.1, 0.9)



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


# New functions to implement the improved DTW approach
def compute_multi_scale_dtw_similarities(X_test_windows, current_time, window_length):
    """
    Compute DTW similarities at multiple time scales with proper bounds handling.

    Parameters:
    -----------
    X_test_windows : array-like
        The input windows of features for DTW calculation
    current_time : int
        The current timestep
    window_length : int
        The base window length

    Returns:
    --------
    dict
        Dictionary containing DTW dissimilarities at different time scales
        and a combined weighted score
    """
    X_test_windows = np.array(X_test_windows)
    X_test_windows = np.delete(X_test_windows, [0, 1, 7, 10, 11, 12, 13, 14, 15], axis=2)

    # Define window sizes for different time scales
    short_scale = 15  # For immediate changes
    medium_scale = window_length  # Your current window_length (25)
    long_scale = 45  # For longer-term patterns

    # Define weights for combining scales (more weight to longer scales to reduce oscillation sensitivity)
    short_weight = 0.2
    medium_weight = 0.3
    long_weight = 0.5

    padding_size = max(40, long_scale * 2)  # Ensure enough padding for the largest window
    padded_windows = np.pad(X_test_windows,
                            pad_width=((padding_size, 0), (0, 0), (0, 0)),
                            mode='edge')

    total_length = len(padded_windows)

    # Dictionary to store results for each scale
    dtw_results = {}

    # Calculate DTW at each scale
    for scale, scale_name, gap in zip(
            [short_scale, medium_scale, long_scale],
            ['short', 'medium', 'long'],
            [5, 10, 20]  # Different gaps for different scales
    ):
        # Modified index calculation with proper bounds checking for this scale
        if current_time < padding_size:
            index_1 = 0
            index_2 = min(current_time + padding_size, total_length - 1)
        else:
            index_1 = current_time
            desired_index_2 = current_time + padding_size

            # Make sure index_2 doesn't exceed array bounds
            index_2 = min(desired_index_2, total_length - 1)

            # If we're near the end and need to maintain minimum gap
            if index_2 - index_1 < gap:
                # Adjust index_1 backwards to maintain gap if possible
                index_1 = max(0, index_2 - gap)

                # Extract windows of appropriate size
        if index_1 + scale <= total_length and index_2 + scale <= total_length:
            window_1 = padded_windows[index_1:index_1 + scale]
            window_2 = padded_windows[index_2:index_2 + scale]

            # The fastdtw function expects each time point to be a 1D vector
            # Each window is shape (time_steps, features)
            # Calculate DTW distance for this scale
            try:
                distance, _ = fastdtw(window_1, window_2, dist=euclidean)
                dtw_results[scale_name] = distance
            except ValueError:
                # Fallback: Flatten each time step into a 1D vector if needed
                # This happens if each time step is multi-dimensional
                window_1_reshaped = window_1.reshape(window_1.shape[0], -1)
                window_2_reshaped = window_2.reshape(window_2.shape[0], -1)
                distance, _ = fastdtw(window_1_reshaped, window_2_reshaped, dist=euclidean)
                dtw_results[scale_name] = distance
        else:
            # Fallback if windows exceed array bounds
            dtw_results[scale_name] = dtw_results.get('medium', 0) if 'medium' in dtw_results else 0

    # Calculate weighted combination of DTW results
    combined_distance = (
            short_weight * dtw_results['short'] +
            medium_weight * dtw_results['medium'] +
            long_weight * dtw_results['long']
    )

    dtw_results['combined'] = combined_distance

    return dtw_results


def lateral_speed_scaling(feature_windows, current_time, dtw_dissimilarity, lateral_speeds):
    """
    Scale DTW dissimilarity based on lateral speed with stricter thresholds
    """
    # Get recent lateral speeds
    recent_speeds = lateral_speeds[max(0, current_time - 5):current_time + 1]

    if len(recent_speeds) > 0:
        # Use absolute values of lateral speeds
        abs_recent_speeds = np.abs(recent_speeds)
        avg_lateral_speed = np.mean(abs_recent_speeds)
        max_lateral_speed = np.max(abs_recent_speeds)

        # New threshold for significant lateral movement
        lateral_speed_threshold = 0.06  # Increased from 0.05

        # More aggressive scaling for low speeds
        if max_lateral_speed < lateral_speed_threshold:
            # Much stronger penalty for speeds below threshold
            scaling_factor = max(0.1, max_lateral_speed / lateral_speed_threshold)
            # More aggressive sigmoid curve (steeper drop-off)
            smooth_scaling = 0.1 + 0.9 * (1 / (1 + np.exp(-15 * (scaling_factor - 0.7))))
        else:
            # Normal scaling for speeds above threshold
            scaling_factor = 1.0
            smooth_scaling = 1.0

        # Scale the dissimilarity
        scaled_dissimilarity = dtw_dissimilarity * smooth_scaling

        return scaled_dissimilarity, max_lateral_speed < lateral_speed_threshold

    # Default case: return the original dissimilarity and low speed flag
    return dtw_dissimilarity, True


class StateManager:
    """
    Manages state transitions between LK and LC with lateral speed consideration
    """

    def __init__(self, lc_commitment_period=5):
        self.current_state = 'LK'  # Start in Lane Keeping state
        self.consecutive_opposite_frames = 0
        self.lc_commitment_period = lc_commitment_period
        self.state_duration = 0

    def update_state(self, predicted_class, similarity, is_low_speed=False):
        """
        Update the internal state with lateral speed considerations

        Parameters:
        -----------
        predicted_class : int
            The predicted class (0 for LK, 1/2 for LC)
        similarity : float
            The current normalized similarity value
        is_low_speed : bool
            Flag indicating if lateral speed is below threshold

        Returns:
        --------
        dict
            Updated state information and adjusted weights
        """
        is_lc_prediction = predicted_class > 0  # 1 or 2 means LC
        self.state_duration += 1

        # If prediction matches current state, reset counter
        if (is_lc_prediction and self.current_state == 'LC') or \
                (not is_lc_prediction and self.current_state == 'LK'):
            self.consecutive_opposite_frames = 0
        else:
            # If prediction opposes current state, increment counter
            # But increment less if lateral speed is low and trying to switch to LC
            if is_low_speed and is_lc_prediction and self.current_state == 'LK':
                # Increment by a smaller value (making it harder to switch to LC)
                self.consecutive_opposite_frames += 0.3  # Reduced increment
            else:
                # Normal increment
                self.consecutive_opposite_frames += 1

        # Determine threshold for state change
        if self.current_state == 'LK':
            # Make it harder to go from LK to LC when lateral speed is low
            if is_low_speed:
                threshold = 7  # Increased threshold for low speed
            else:
                threshold = 3  # Normal threshold
        else:  # self.current_state == 'LC'
            # Easier to go back to LK
            base_threshold = self.lc_commitment_period
            duration_factor = min(1.0, self.state_duration / 10)
            threshold = int(base_threshold * (1 + duration_factor))

        # Check for state transition
        if self.consecutive_opposite_frames >= threshold:
            # Change state
            self.current_state = 'LC' if self.current_state == 'LK' else 'LK'
            self.consecutive_opposite_frames = 0
            self.state_duration = 0

        # Adjust map weights based on state, similarity, and lateral speed
        if self.current_state == 'LK':
            # In LK state, higher similarity means more weight to LK map
            if is_low_speed:
                # Much higher LK weight when speed is low
                lk_weight = max(0.8, similarity)
                lc_weight = 1 - lk_weight
            else:
                # Normal weighting
                lk_weight = max(0.6, similarity)
                lc_weight = 1 - lk_weight
        else:  # LC state
            # In LC state, lower similarity means more weight to LC map
            lc_weight = max(0.6, 1 - similarity)
            lk_weight = 1 - lc_weight

        # Normalize weights
        total = lk_weight + lc_weight
        lk_weight /= total
        lc_weight /= total

        return {
            'state': self.current_state,
            'duration': self.state_duration,
            'consecutive_opposite': self.consecutive_opposite_frames,
            'threshold_for_change': threshold,
            'lk_weight': float(lk_weight),
            'lc_weight': float(lc_weight),
            'is_low_speed': is_low_speed
        }


def improved_assess_prediction_with_density(pred_probs, classification_output,
                                            similarity_history, dtw_results,
                                            feature_windows, current_time,
                                            state_manager, cell_maps,
                                            current_step, total_steps,
                                            lk_min=None, lk_max=None,
                                            lc_min=None, lc_max=None,
                                            lateral_speeds=None):
    """
    Modified assessment using multi-scale DTW, lateral speed scaling, and state persistence
    for improved reliability scoring with ratio-based comparison and weight-based reliability boost.
    """
    # If min/max densities aren't provided, calculate them from cell_maps
    if lk_min is None or lk_max is None or lc_min is None or lc_max is None:
        lk_min, lk_max, lc_min, lc_max = extract_min_max_densities(cell_maps)
        # Print once for debugging
        if current_step == 0:
            print(f"Using separate normalization ranges:")
            print(f"  LK: min={lk_min}, max={lk_max}")
            print(f"  LC: min={lc_min}, max={lc_max}")

    # Get the combined DTW dissimilarity and apply lateral speed scaling
    combined_dissimilarity = dtw_results['combined']
    scaled_dissimilarity, is_low_speed = lateral_speed_scaling(
        feature_windows, current_time, combined_dissimilarity, lateral_speeds)

    # Add to similarity history
    similarity_history.append(scaled_dissimilarity)

    # Normalize similarity with the enhanced approach
    norm_similarity = normalize_similarity_window(similarity_history)

    # Get current prediction class (0 for LK, 1/2 for LC)
    current_prediction = classification_output[current_step] if current_step < len(classification_output) else 0

    # Update state manager and get adjusted weights - pass is_low_speed flag
    state_info = state_manager.update_state(current_prediction, norm_similarity, is_low_speed)
    lk_weight = state_info['lk_weight']
    lc_weight = state_info['lc_weight']

    # Direct lateral speed threshold enforcement
    recent_speeds = lateral_speeds[max(0, current_time - 5):current_time + 1]
    if len(recent_speeds) > 0:
        max_lateral_speed = np.max(np.abs(recent_speeds))

        # If lateral speed is below threshold AND we're in LC state
        if max_lateral_speed < 0.06 and state_info['state'] == 'LC':
            # Apply a direct penalty to LC weight
            penalty_factor = 0.5
            lc_weight = lc_weight * penalty_factor
            lk_weight = 1 - lc_weight

    # Handle batch-style predictions
    if isinstance(pred_probs, np.ndarray) and len(pred_probs.shape) == 2:
        pred_probs_to_use = pred_probs[0]
    else:
        pred_probs_to_use = pred_probs

    # Get distributions and densities
    lk_dist, lc_dist, lk_density, lc_density, cell = get_cell_distribution_with_density(
        pred_probs_to_use, cell_maps)

    # Normalize densities using SEPARATE min/max values for LK and LC
    lk_density_score = normalize_density(lk_density, lk_min, lk_max)
    lc_density_score = normalize_density(lc_density, lc_min, lc_max)

    # Calculate ratio-based reliability score
    epsilon = 1e-6

    # Calculate the ratio between the densities (with smoothing)
    # Use raw density values for more contrast
    density_ratio = max(lk_density, epsilon) / max(lc_density, epsilon) if lk_weight > lc_weight else max(lc_density,
                                                                                                          epsilon) / max(
        lk_density, epsilon)

    # Apply sigmoid transformation to bound the ratio between 0 and 1
    ratio_score = 1 / (1 + np.exp(-0.5 * (np.log(density_ratio) - 1)))

    # Calculate a weighted combination of absolute density scores and ratio score
    alpha = 0.4  # Weight for absolute scores
    beta = 0.6  # Weight for ratio score

    absolute_score = (lk_weight * lk_density_score) + (lc_weight * lc_density_score)

    # Final reliability score combines both approaches
    reliability_score = (alpha * absolute_score) + (beta * ratio_score)

    # MODIFIED: Directly tie reliability boost to LK/LC weights when they match the prediction
    # Check if prediction and dominant weight align
    is_lk_prediction = current_prediction == 0
    is_lc_prediction = current_prediction > 0  # 1 or 2

    # Define weight threshold for boost (as per your request)
    weight_threshold = 0.6

    # Apply boost when prediction and dominant weight align
    if (is_lk_prediction and lk_weight > weight_threshold) or (is_lc_prediction and lc_weight > weight_threshold):
        # Get the relevant weight (LK or LC) based on the prediction
        dominant_weight = lk_weight if is_lk_prediction else lc_weight

        # Calculate boost factor (larger when weight is higher)
        # This scales the boost based on how much the weight exceeds the threshold
        weight_excess = (dominant_weight - weight_threshold) / (1.0 - weight_threshold)
        boost_factor = 0.3 * weight_excess  # Adjust the 0.3 multiplier as needed

        # Apply the boost
        reliability_score = min(1.0, reliability_score + boost_factor)

    return {
        'cell': cell,
        'lk_weight': float(lk_weight),
        'lc_weight': float(lc_weight),
        'lk_density': int(lk_density),
        'lc_density': int(lc_density),
        'lk_density_score': float(lk_density_score),
        'lc_density_score': float(lc_density_score),
        'reliability_score': float(reliability_score),
        'ratio_score': float(ratio_score),
        'absolute_score': float(absolute_score),
        'state': state_info['state'],
        'state_duration': state_info['duration'],
        'dtw_short': float(dtw_results['short']),
        'dtw_medium': float(dtw_results['medium']),
        'dtw_long': float(dtw_results['long']),
        'dtw_combined': float(dtw_results['combined']),
        'dtw_scaled': float(scaled_dissimilarity),
        'normalized_similarity': float(norm_similarity),
        'is_low_speed': is_low_speed
    }


def plot_trajectory_with_enhanced_metrics(selected_feature, reliability_scores,
                                          lk_density_scores, lc_density_scores,
                                          lk_weights, lc_weights,
                                          dtw_values, states, lateral_speeds,
                                          CNN_output, labels, title=None):
    """
    Plot trajectory with enhanced metrics including multi-scale DTW and state information.

    Parameters:
    -----------
    selected_feature : array-like
        The trajectory data to plot (e.g., lateral position)
    reliability_scores : array-like
        Overall reliability scores for each timestep
    lk_density_scores, lc_density_scores : array-like
        Normalized density scores
    lk_weights, lc_weights : array-like
        Weights applied to scores
    dtw_values : dict of array-like
        DTW values at different scales
    states : array-like
        State information for each timestep
    lateral_speeds : array-like
        Lateral speed values
    CNN_output : array-like
        Model predictions for each timestep
    labels : array-like
        Ground truth labels for each timestep
    title : str, optional
        Plot title
    """
    # Create time steps array
    time_steps = np.arange(len(selected_feature))

    # Use the 'RdYlGn' colormap - red for low reliability, green for high
    cmap = plt.colormaps['RdYlGn']

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot trajectory with reliability coloring on the first subplot
    min_val, max_val = 0, 1
    for i in range(len(time_steps) - 1):
        color_value = reliability_scores[i]
        ax1.plot(time_steps[i:i + 2], selected_feature[i:i + 2],
                 color=cmap((color_value - min_val) / (max_val - min_val)),
                 linewidth=5.5)

    # Get y-axis limits for marker positioning
    y_min, y_max = ax1.get_ylim()
    y_range = y_max - y_min

    # Calculate positions for marker rows
    pred_marker_y = y_max + 0.15 * y_range  # Predictions row
    truth_marker_y = y_max + 0.25 * y_range  # Ground truth row
    state_marker_y = y_max + 0.05 * y_range  # State row

    # Define colors and labels for classes
    class_colors = {0: 'gray', 1: 'blue', 2: 'orange'}
    class_labels = {0: 'LK', 1: 'LCL', 2: 'LCR'}
    state_colors = {'LK': 'gray', 'LC': 'red'}

    # Plot CNN prediction markers
    for i, cls in enumerate(CNN_output):
        ax1.plot(time_steps[i], pred_marker_y, 'o',
                 color=class_colors[cls],
                 markersize=10,
                 clip_on=False)

    # Plot ground truth markers
    for i, cls in enumerate(labels):
        ax1.plot(time_steps[i], truth_marker_y, 'o',
                 color=class_colors[cls],
                 markersize=10,
                 clip_on=False)

    # Plot state markers
    for i, state in enumerate(states):
        ax1.plot(time_steps[i], state_marker_y, 'o',
                 color=state_colors[state],
                 markersize=7,
                 clip_on=False)

    # Add text labels for rows
    ax1.text(-5, truth_marker_y, 'Ground Truth',
             verticalalignment='center',
             fontsize=12,  # Larger font size
             fontweight='bold',  # Bold text
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))

    ax1.text(-5, pred_marker_y, 'Prediction',
             verticalalignment='center',
             fontsize=12,  # Larger font size
             fontweight='bold',  # Bold text
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))

    ax1.text(-5, state_marker_y, 'State',
             verticalalignment='center',
             fontsize=12,  # Larger font size
             fontweight='bold',  # Bold text
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))

    # Create legend for classes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, label=label, markersize=10)
                       for cls, (color, label) in enumerate(zip(class_colors.values(),
                                                                class_labels.values()))]

    # Add state to legend
    for state, color in state_colors.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, label=f'State: {state}', markersize=7)
        )

    # Add legend to first subplot
    ax1.legend(handles=legend_elements, loc='lower right')

    # Update y-axis limits
    ax1.set_ylim(y_min, truth_marker_y + 0.1 * y_range)

    # Set labels and grid for first subplot
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Normalized Lateral Position')
    ax1.grid(True)
    if title:
        ax1.set_title(title)

    # Add colorbar for first subplot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, label='Reliability Score')

    # Plot density scores and weights on the second subplot
    ax2.plot(time_steps, lk_weights, 'b-', label='LK Weight', alpha=0.7)
    ax2.plot(time_steps, lc_weights, 'r-', label='LC Weight', alpha=0.7)
    ax2.set_ylim(0, 1.1)

    # Create a twin y-axis for normalized density scores
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_steps, lk_density_scores, 'b--', label='LK Density Score', alpha=0.5)
    ax2_twin.plot(time_steps, lc_density_scores, 'r--', label='LC Density Score', alpha=0.5)
    ax2_twin.set_ylim(0, 1.1)

    # Set labels for second subplot
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Weights')
    ax2_twin.set_ylabel('Normalized Density Scores')
    ax2.grid(True)
    ax2.set_title('Component Analysis')

    # Add legends to second subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot DTW values on third subplot
    ax3.plot(time_steps, dtw_values['short'], 'g-', label='Short DTW', alpha=0.7)
    ax3.plot(time_steps, dtw_values['medium'], 'b-', label='Medium DTW', alpha=0.7)
    ax3.plot(time_steps, dtw_values['long'], 'r-', label='Long DTW', alpha=0.7)
    ax3.plot(time_steps, dtw_values['combined'], 'k-', label='Combined DTW', alpha=0.7)
    ax3.plot(time_steps, dtw_values['scaled'], 'k--', label='Scaled DTW', alpha=0.7)

    # Create a twin y-axis for lateral speed
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_steps, np.abs(lateral_speeds), 'c-', label='|Lateral Speed|', alpha=0.5)

    # Set labels for third subplot
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('DTW Dissimilarity')
    ax3_twin.set_ylabel('|Lateral Speed|')
    ax3.grid(True)
    ax3.set_title('DTW Analysis and Lateral Speed')

    # Add legends to third subplot
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')

    plt.tight_layout()
    plt.show()