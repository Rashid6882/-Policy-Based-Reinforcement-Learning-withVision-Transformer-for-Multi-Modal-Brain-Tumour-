import os
import glob
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BraTS2DDataset(Dataset):
    """
    Dataset loader for BraTS 2D slice HDF5 files.
    Each file contains:
    - 'image': (240, 240, 4) representing 4 MRI modalities (T1, T1ce, T2, FLAIR)
    - 'mask': (240, 240, 3) representing tumor sub-regions
    """
    def __init__(self, data_dir, filter_tumors_only=True, max_samples=None):
        self.data_dir = data_dir
        self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        self.file_paths.sort()
        
        if os.path.exists("tumor_slices_index.json") and filter_tumors_only:
            with open("tumor_slices_index.json", "r") as f:
                valid_files = json.load(f)
            if max_samples:
                valid_files = valid_files[:max_samples]
            self.file_paths = valid_files
        elif filter_tumors_only:
            valid_files = []
            max_search = max_samples * 8 if max_samples else len(self.file_paths)
            search_paths = self.file_paths[:max_search]
            for path in search_paths:
                try:
                    with h5py.File(path, 'r') as f:
                        if 'mask' in f and f['mask'][:].sum() > 0:
                            valid_files.append(path)
                except Exception:
                    continue
                if max_samples and len(valid_files) >= max_samples:
                    break
            self.file_paths = valid_files
        elif max_samples:
            self.file_paths = self.file_paths[:max_samples]
            
        print(f"Loaded {len(self.file_paths)} slice samples from {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with h5py.File(path, 'r') as f:
            image = f['image'][:] # (240, 240, 4)
            mask = f['mask'][:]   # (240, 240, 3)

        # Transpose image to PyTorch (Channels, H, W) -> (4, 240, 240)
        image_t = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Combine mask channels to get overall tumor mask (240, 240)
        combined_mask = (mask.sum(axis=-1) > 0).astype(np.uint8)

        # Calculate ground truth bounding box [xmin, ymin, xmax, ymax]
        y_indices, x_indices = np.where(combined_mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            gt_bbox = np.array([
                np.min(x_indices),
                np.min(y_indices),
                np.max(x_indices),
                np.max(y_indices)
            ], dtype=np.float32)
        else:
            gt_bbox = np.array([0, 0, 240, 240], dtype=np.float32)

        return {
            'image': image_t,                  # (4, 240, 240)
            'mask': combined_mask,              # (240, 240)
            'sub_masks': np.transpose(mask, (2, 0, 1)), # (3, 240, 240)
            'gt_bbox': gt_bbox,                 # [x1, y1, x2, y2]
            'file_name': os.path.basename(path)
        }

def get_dataloader(data_dir, batch_size=1, filter_tumors_only=True, max_samples=None):
    dataset = BraTS2DDataset(data_dir, filter_tumors_only=filter_tumors_only, max_samples=max_samples)
    return dataset
