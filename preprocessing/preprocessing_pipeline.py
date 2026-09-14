import numpy as np
import cv2
from .nifti_loader import BraTSNiftiLoader
from .normalization import zscore_normalize
from .resampling import check_and_resample

class BraTSDataPipeline:
    def __init__(self, data_dir):
        self.loader = BraTSNiftiLoader(data_dir)

    def process_case(self, case_id, subregion="WT"):
        modalities_raw, seg_raw = self.loader.load_case(case_id)
        
        normalized_modalities = {}
        for mod, vol in modalities_raw.items():
            resampled = check_and_resample(vol)
            normalized_modalities[mod] = zscore_normalize(resampled)

        stacked_volume = np.stack([
            normalized_modalities['t1n'],
            normalized_modalities['t1c'],
            normalized_modalities['t2w'],
            normalized_modalities['t2f']
        ], axis=0) # [4, D, H, W] or [4, H, W, D]

        # Find axial slice with max lesion area (or middle slice if no lesion)
        if seg_raw is not None:
            if subregion == "WT":
                mask_3d = (seg_raw > 0)
            elif subregion == "TC":
                mask_3d = np.isin(seg_raw, [1, 3])
            elif subregion == "ET":
                mask_3d = (seg_raw == 3)
            else:
                mask_3d = (seg_raw > 0)

            slice_sums = mask_3d.sum(axis=(0, 1))
            best_slice_idx = int(np.argmax(slice_sums))
            if slice_sums[best_slice_idx] == 0:
                best_slice_idx = seg_raw.shape[2] // 2
        else:
            best_slice_idx = stacked_volume.shape[3] // 2

        slice_2d = stacked_volume[:, :, :, best_slice_idx] # [4, H, W]
        seg_2d = seg_raw[:, :, best_slice_idx] if seg_raw is not None else None

        return slice_2d, seg_2d, best_slice_idx

def extract_crop(image_2d: np.ndarray, bbox: list, crop_size=(64, 64)) -> np.ndarray:
    """
    Extract crop tensor [4, crop_h, crop_w] given normalized bbox [x_min, y_min, x_max, y_max] in [0, 1].
    """
    c, H, W = image_2d.shape
    xmin, ymin, xmax, ymax = bbox
    
    px_xmin = int(np.clip(xmin * W, 0, W - 1))
    px_ymin = int(np.clip(ymin * H, 0, H - 1))
    px_xmax = int(np.clip(xmax * W, px_xmin + 1, W))
    px_ymax = int(np.clip(ymax * H, px_ymin + 1, H))

    cropped_channels = []
    for ch in range(c):
        patch = image_2d[ch, px_ymin:px_ymax, px_xmin:px_xmax]
        if patch.size == 0:
            patch = np.zeros((crop_size[0], crop_size[1]), dtype=np.float32)
        else:
            patch = cv2.resize(patch, crop_size, interpolation=cv2.INTER_LINEAR)
        cropped_channels.append(patch)

    return np.stack(cropped_channels, axis=0) # [4, 64, 64]
