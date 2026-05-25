import numpy as np
import torch

def identify_surprising_positions(prediction_error_map, top_k=16):
    """
    Identify top-K spatial positions with highest prediction error.
    Args:
        prediction_error_map: (B, 128) or (128,) - per-position prediction error from the encoder
        top_k: number of positions to select
    Returns:
        positions: list of indices into the 128-pixel array
    """
    # If batched, take mean across batch
    if prediction_error_map.dim() == 2:
        prediction_error_map = prediction_error_map.mean(dim=0)  # (128,)
    elif prediction_error_map.dim() == 3:
        # (B, C, 128) -> take norm across C, mean across B
        prediction_error_map = prediction_error_map.norm(dim=1).mean(dim=0)  # (128,)
    
    _, top_indices = torch.topk(prediction_error_map, min(top_k, prediction_error_map.shape[0]))
    return top_indices.sort()[0].tolist()

def compute_itag(pixel_buffer, surprising_positions, window=20):
    """
    Compute ITAG score: mean lag-1 temporal autocorrelation of raw pixel values 
    at surprising positions over a sliding window.
    
    Args:
        pixel_buffer: list of length W_t, each element is (3, 128) numpy array
        surprising_positions: list of spatial indices
        window: number of timesteps to use (uses last `window` from buffer)
    Returns:
        itag_score: float in [-1, 1]
    """
    if len(pixel_buffer) < 2 or len(surprising_positions) == 0:
        return 0.0
    
    # Use last `window` timesteps
    buffer_slice = pixel_buffer[-window:]
    if len(buffer_slice) < 2:
        return 0.0
    
    # Stack into array: (T, 3, 128)
    pixel_array = np.stack(buffer_slice, axis=0)
    T = pixel_array.shape[0]
    
    # Compute per-position temporal autocorrelation
    autocorrs = []
    for pos in surprising_positions:
        if pos < 0 or pos >= 128:
            continue
        # Get pixel intensity at this position across time (sum across RGB channels for robustness)
        # Shape: (T,)
        pixel_vals = pixel_array[:, :, pos].sum(axis=1)  # (T,)
        
        # Compute lag-1 autocorrelation
        if T < 2:
            autocorrs.append(0.0)
            continue
        
        x_t = pixel_vals[:-1]  # (T-1,)
        x_t1 = pixel_vals[1:]  # (T-1,)
        
        # Pearson correlation between x_t and x_{t+1}
        std_t = np.std(x_t)
        std_t1 = np.std(x_t1)
        
        if std_t < 1e-8 or std_t1 < 1e-8:
            # Constant signal: high autocorrelation
            autocorrs.append(1.0 if np.mean(np.abs(x_t - x_t[0])) < 1e-6 else 0.0)
            continue
        
        corr = np.corrcoef(x_t, x_t1)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        autocorrs.append(corr)
    
    if len(autocorrs) == 0:
        return 0.0
    
    return float(np.mean(autocorrs))

def compute_isag(pixel_buffer, surprising_positions):
    """
    Compute ISAG score: mean lag-1 spatial autocorrelation of raw pixel values 
    at adjacent surprising positions, computed per frame.
    
    Args:
        pixel_buffer: list of (3, 128) numpy arrays (single frame used from last)
        surprising_positions: list of spatial indices (sorted)
    Returns:
        isag_score: float in [-1, 1]
    """
    if len(pixel_buffer) == 0 or len(surprising_positions) < 2:
        return 0.0
    
    # Use last frame
    frame = pixel_buffer[-1]  # (3, 128)
    
    # Sort positions
    sorted_pos = sorted(surprising_positions)
    
    # Compute spatial autocorrelation between adjacent surprising positions
    spatial_corrs = []
    for i in range(len(sorted_pos) - 1):
        pos_a = sorted_pos[i]
        pos_b = sorted_pos[i + 1]
        
        if pos_a >= 128 or pos_b >= 128:
            continue
        
        # Pixel intensity (sum across RGB)
        val_a = frame[:, pos_a].sum()
        val_b = frame[:, pos_b].sum()
        
        # For spatial autocorrelation we need multiple adjacent pairs
        # Since we only have one frame, compute correlation across the spatial positions
        # using a simpler approach: check if adjacent surprising positions have similar intensity
        # This is a proxy for spatial smoothness
        spatial_corrs.append(val_a * val_b)  # unnormalized similarity
    
    if len(spatial_corrs) == 0:
        return 0.0
    
    # Normalize: compare actual similarity to expected if positions were independent
    all_vals = [frame[:, p].sum() for p in sorted_pos if p < 128]
    if len(all_vals) == 0:
        return 0.0
    
    mean_val = np.mean(all_vals)
    var_val = np.var(all_vals)
    
    if var_val < 1e-8:
        return 1.0  # All positions same intensity -> perfectly smooth
    
    # Spatial autocorrelation (Moran's I style)
    n = len(sorted_pos)
    w_sum = n - 1  # number of adjacent pairs
    numerator = sum((all_vals[i] - mean_val) * (all_vals[i+1] - mean_val) 
                    for i in range(len(all_vals)-1))
    denominator = var_val * w_sum
    
    if abs(denominator) < 1e-8:
        return 0.0
    
    isag = numerator / denominator
    return float(np.clip(isag, -1.0, 1.0))
