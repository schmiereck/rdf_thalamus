"""
ITAG (Input-Level Temporal AutoCorrelation Gating) and ISAG (Input-Level Spatial AutoCorrelation Gating)

These metrics operate on raw pixel values at spatial positions identified as surprising by the pre-trained encoder.
They avoid all three cold-start pathologies (no encoder cold-start, no predictor cold-start, no optimization transient).
"""

import numpy as np
import torch


def identify_surprising_positions(prediction_error_map, top_k=16):
    """
    Identify the top-K spatial positions with highest per-position prediction error.
    
    Args:
        prediction_error_map: torch.Tensor of shape (128,) or (B, 128) or (B, d_max, 128)
                              If multi-dimensional, the norm along the channel/feature dimension is used.
        top_k (int): Number of top positions to return.
    
    Returns:
        surprising_positions: numpy array of shape (top_k,) containing sorted spatial indices.
    """
    if prediction_error_map.ndim == 3:
        # (B, d_max, 128) -> compute L2 norm across d_max dimension
        error_map = torch.norm(prediction_error_map, dim=1)  # (B, 128)
    elif prediction_error_map.ndim == 2:
        # (B, 128) -> use first batch element or mean across batch
        error_map = prediction_error_map[0] if prediction_error_map.shape[0] > 0 else prediction_error_map
    else:
        error_map = prediction_error_map
    
    # Ensure 1D
    error_map = error_map.reshape(-1)
    
    # Get top-k indices using torch
    _, flat_indices = torch.topk(error_map, top_k, largest=True, sorted=False)
    flat_indices = flat_indices.cpu().numpy()
    
    # Sort indices before returning
    return np.sort(flat_indices)


def compute_itag(pixel_array, surprising_positions, window=20):
    """
    Compute Input-Level Temporal Autocorrelation Gating (ITAG).
    
    For each position x in surprising_positions, compute the lag-1 temporal
    autocorrelation of pixel values (averaged across RGB channels) over the last
    `window` frames. Return the mean across all surprising positions.
    
    Args:
        pixel_array: list or array of raw pixel frames, each of shape (3, 128).
                     Typically the last `window` frames from the pixel history.
        surprising_positions: numpy array of spatial indices (int), shape (top_k,).
        window (int): Number of consecutive timesteps to use for temporal autocorrelation.
    
    Returns:
        itag_score (float): Mean lag-1 temporal autocorrelation, clipped to [0, 1].
    """
    if len(pixel_array) < 2:
        return 0.0
    
    # Use at most `window` most recent frames
    frames_used = pixel_array[-window:] if len(pixel_array) >= window else pixel_array
    
    # Average across RGB channels -> (T, 128)
    frames_avg = []
    for frame in frames_used:
        if frame.ndim == 2 and frame.shape[0] == 3:
            frames_avg.append(np.mean(frame, axis=0))
        else:
            frames_avg.append(frame.reshape(-1))
    frames_avg = np.stack(frames_avg, axis=0)
    
    itag_scores = []
    for pos in surprising_positions:
        seq = frames_avg[:, pos]  # (T,)
        mean = seq.mean()
        std = seq.std(ddof=0)
        if std < 1e-8:
            itag_scores.append(0.0)
        else:
            seq_norm = (seq - mean) / std
            # Lag-1 temporal autocorrelation
            corr = np.mean(seq_norm[:-1] * seq_norm[1:])
            # Clip negative autocorrelation to 0 (noise should not contribute)
            itag_scores.append(max(0.0, corr))
    
    return float(np.mean(itag_scores)) if itag_scores else 0.0


def compute_isag(pixel_array, surprising_positions, window=20):
    """
    Compute Input-Level Spatial Autocorrelation Gating (ISAG).
    
    For each frame, consider the pixel values (averaged across RGB channels) at the
    surprising positions. Compute the lag-1 spatial autocorrelation between adjacent
    surprising positions within that frame, normalized by the frame-level variance.
    Then average across frames.
    
    Uses at most `window` most recent frames.
    
    Args:
        pixel_array: list or array of raw pixel frames, each of shape (3, 128).
        surprising_positions: numpy array of spatial indices, shape (top_k,).
        window (int): Number of consecutive timesteps to use.
    
    Returns:
        isag_score (float): Mean spatial autocorrelation, clipped to [0, 1].
    """
    if len(pixel_array) < 1:
        return 0.0
    
    # Use at most `window` most recent frames
    frames_used = pixel_array[-window:] if len(pixel_array) >= window else pixel_array
    
    sorted_positions = np.sort(surprising_positions)
    if len(sorted_positions) < 2:
        return 0.0
    
    frame_scores = []
    for frame in frames_used:
        if frame.ndim == 2 and frame.shape[0] == 3:
            pixel_vals = np.mean(frame, axis=0)
        else:
            pixel_vals = frame.reshape(-1)
        
        values = pixel_vals[sorted_positions]
        mean = values.mean()
        std = values.std(ddof=0)
        if std < 1e-8:
            frame_scores.append(0.0)
            continue
        
        # Lag-1 spatial autocorrelation: correlation between adjacent values
        corrs = []
        for i in range(len(values) - 1):
            corr = (values[i] - mean) * (values[i + 1] - mean) / (std ** 2)
            corrs.append(corr)
        
        frame_scores.append(np.mean(corrs))
    
    # Clip negative values to 0
    frame_scores = [max(0.0, s) for s in frame_scores]
    return float(np.mean(frame_scores)) if frame_scores else 0.0
