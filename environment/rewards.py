import numpy as np

def compute_dice(bbox: list, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient between normalized bbox [xmin, ymin, xmax, ymax] and 2D ground truth binary mask.
    """
    H, W = gt_mask.shape
    xmin, ymin, xmax, ymax = bbox
    
    px_xmin = int(np.clip(xmin * W, 0, W - 1))
    px_ymin = int(np.clip(ymin * H, 0, H - 1))
    px_xmax = int(np.clip(xmax * W, px_xmin + 1, W))
    px_ymax = int(np.clip(ymax * H, px_ymin + 1, H))

    pred_mask = np.zeros((H, W), dtype=bool)
    pred_mask[px_ymin:px_ymax, px_xmin:px_xmax] = True
    
    gt_bool = (gt_mask > 0)
    
    intersection = np.logical_and(pred_mask, gt_bool).sum()
    total = pred_mask.sum() + gt_bool.sum()
    
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)

def calculate_reward(prev_dice: float, curr_dice: float, action: int, terminate_action_idx: int = 6,
                     iou_threshold: float = 0.3, goal_bonus: float = 2.0, premature_penalty: float = -0.1,
                     step_penalty: float = -0.01, reward_scale: float = 10.0) -> float:
    if action == terminate_action_idx:
        if curr_dice >= iou_threshold:
            return goal_bonus
        else:
            return premature_penalty
            
    diff_gain = reward_scale * (curr_dice - prev_dice)
    return diff_gain + step_penalty
