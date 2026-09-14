import os
import glob
import numpy as np
import nibabel as nib

class BraTSNiftiLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_case(self, case_id):
        case_folder = os.path.join(self.base_dir, case_id)
        if not os.path.exists(case_folder):
            raise FileNotFoundError(f"Case folder {case_folder} does not exist.")

        modalities = {}
        for mod in ['t1n', 't1c', 't2w', 't2f']:
            matches = glob.glob(os.path.join(case_folder, f"*{mod}.nii.gz"))
            if not matches:
                raise FileNotFoundError(f"Modality {mod} not found in {case_folder}")
            img = nib.load(matches[0])
            data = img.get_fdata().astype(np.float32)
            modalities[mod] = data

        seg_matches = glob.glob(os.path.join(case_folder, "*seg.nii.gz"))
        seg_data = None
        if seg_matches:
            seg_img = nib.load(seg_matches[0])
            seg_data = seg_img.get_fdata().astype(np.uint8)

        return modalities, seg_data
