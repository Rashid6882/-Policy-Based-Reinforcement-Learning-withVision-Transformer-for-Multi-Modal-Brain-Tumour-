import numpy as np

def zscore_normalize(volume: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply per-modality z-score normalization on non-zero brain voxels.
    """
    if mask is None:
        mask = volume > 0

    if np.any(mask):
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        if std < 1e-8:
            std = 1.0
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[mask] = (volume[mask] - mean) / std
        return normalized
    else:
        std = np.std(volume)
        if std < 1e-8:
            return np.zeros_like(volume, dtype=np.float32)
        return (volume - np.mean(volume)) / std
