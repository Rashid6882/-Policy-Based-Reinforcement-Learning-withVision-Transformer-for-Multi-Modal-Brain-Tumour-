import numpy as np

def compute_subregion_metrics(bbox: list, seg_2d: np.ndarray):
    """
    Computes overlap metrics (Dice, IoU, Sensitivity, Precision) for Whole Tumor (WT),
    Tumor Core (TC), and Enhancing Tumor (ET) subregions from raw 2D multi-label mask.
    """
    H, W = seg_2d.shape
    xmin, ymin, xmax, ymax = bbox
    
    px_xmin = int(np.clip(xmin * W, 0, W - 1))
    px_ymin = int(np.clip(ymin * H, 0, H - 1))
    px_xmax = int(np.clip(xmax * W, px_xmin + 1, W))
    px_ymax = int(np.clip(ymax * H, px_ymin + 1, H))

    pred_mask = np.zeros((H, W), dtype=bool)
    pred_mask[px_ymin:px_ymax, px_xmin:px_xmax] = True

    subregion_masks = {
        "WT": (seg_2d > 0),
        "TC": np.isin(seg_2d, [1, 3]),
        "ET": (seg_2d == 3)
    }

    metrics = {}
    for region_name, gt_mask in subregion_masks.items():
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, ~gt_mask).sum()
        fn = np.logical_and(~pred_mask, gt_mask).sum()
        
        dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0

        metrics[region_name] = {
            "dice": dice,
            "iou": iou,
            "sensitivity": sens,
            "precision": prec
        }

    return metrics
