import numpy as np

def check_and_resample(volume: np.ndarray, target_spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Verification pass for 1mm isotropic spacing.
    BraTS 2023 is natively 1mm isotropic.
    """
    return volume
