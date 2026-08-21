import numpy as np
import torch
import torch.nn.functional as F

def compute_iou_bbox(box1, box2):
    """
    Compute IoU between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def compute_dice_mask(pred_bbox, gt_mask, image_size=(240, 240)):
    """
    Compute Dice coefficient between predicted bounding box mask and ground truth binary mask.
    """
    pred_mask = np.zeros(image_size, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, pred_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_size[1], x2), min(image_size[0], y2)
    
    if x2 > x1 and y2 > y1:
        pred_mask[y1:y2, x1:x2] = 1

    intersection = np.logical_and(pred_mask, gt_mask > 0).sum()
    total_area = pred_mask.sum() + (gt_mask > 0).sum()

    if total_area == 0:
        return 1.0
    return (2.0 * intersection) / total_area

def compute_metrics(pred_bbox, gt_mask, gt_bbox, image_size=(240, 240)):
    """
    Compute Dice, IoU, Sensitivity, and Precision.
    """
    pred_mask = np.zeros(image_size, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, pred_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_size[1], x2), min(image_size[0], y2)
    
    if x2 > x1 and y2 > y1:
        pred_mask[y1:y2, x1:x2] = 1

    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    tp = np.logical_and(pred_mask == 1, gt_binary == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_binary == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_binary == 1).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = compute_iou_bbox(pred_bbox, gt_bbox)
    sensitivity = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'precision': float(precision)
    }

class BrainLesionEnv:
    """
    Gym-like Environment for Sequential Brain Lesion Localization via Bounding Box Movement.
    """
    def __init__(self, roi_size=(64, 64), max_steps=20, move_factor=0.1, scale_factor=0.15):
        self.roi_size = roi_size
        self.max_steps = max_steps
        self.move_factor = move_factor
        self.scale_factor = scale_factor
        self.action_space_dim = 7  # 0: Left, 1: Right, 2: Up, 3: Down, 4: Zoom In, 5: Zoom Out, 6: Trigger
        self.img_h = 240
        self.img_w = 240
        
        self.image = None       # (4, 240, 240)
        self.gt_mask = None     # (240, 240)
        self.gt_bbox = None     # [x1, y1, x2, y2]
        self.current_bbox = None
        self.current_step = 0
        self.prev_iou = 0.0
        self.trajectory = []

    def reset(self, sample):
        """
        Reset environment with a new slice sample.
        """
        self.image = sample['image']        # numpy array (4, 240, 240)
        self.gt_mask = sample['mask']      # numpy array (240, 240)
        self.gt_bbox = sample['gt_bbox']    # numpy array [x1, y1, x2, y2]
        self.current_step = 0
        
        # Initialize agent view window to cover full scan or central area
        self.current_bbox = np.array([10.0, 10.0, 230.0, 230.0], dtype=np.float32)
        self.prev_iou = compute_iou_bbox(self.current_bbox, self.gt_bbox)
        self.trajectory = [self.current_bbox.copy().tolist()]

        return self._get_observation()

    def _get_observation(self):
        """
        Extract cropped 4-channel ROI patch and normalized bounding box features.
        """
        x1, y1, x2, y2 = map(int, self.current_bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_w, max(x1 + 10, x2)), min(self.img_h, max(y1 + 10, y2))

        # Crop ROI across all 4 MRI modalities (4, H_crop, W_crop)
        roi_crop = self.image[:, y1:y2, x1:x2]

        # Convert to Torch tensor and resize to fixed size (4, 64, 64)
        roi_tensor = torch.from_numpy(roi_crop).unsqueeze(0).float() # (1, 4, H_crop, W_crop)
        roi_resized = F.interpolate(roi_tensor, size=self.roi_size, mode='bilinear', align_corners=False).squeeze(0) # (4, 64, 64)

        # Vector of normalized bounding box & step features
        box_vector = np.array([
            self.current_bbox[0] / self.img_w,
            self.current_bbox[1] / self.img_h,
            self.current_bbox[2] / self.img_w,
            self.current_bbox[3] / self.img_h,
            self.current_step / float(self.max_steps)
        ], dtype=np.float32)

        return {
            'roi': roi_resized,                          # Torch tensor (4, 64, 64)
            'box_feat': torch.from_numpy(box_vector)     # Torch tensor (5,)
        }

    def step(self, action):
        """
        Execute one action step.
        """
        self.current_step += 1
        x1, y1, x2, y2 = self.current_bbox
        w = x2 - x1
        h = y2 - y1
        
        dx = self.move_factor * w
        dy = self.move_factor * h

        done = False
        reward = 0.0

        if action == 0:   # Move Left
            x1 -= dx
            x2 -= dx
        elif action == 1: # Move Right
            x1 += dx
            x2 += dx
        elif action == 2: # Move Up
            y1 -= dy
            y2 -= dy
        elif action == 3: # Move Down
            y1 += dy
            y2 += dy
        elif action == 4: # Zoom In (Scale Down Box)
            x1 += self.scale_factor * w
            y1 += self.scale_factor * h
            x2 -= self.scale_factor * w
            y2 -= self.scale_factor * h
        elif action == 5: # Zoom Out (Scale Up Box)
            x1 -= self.scale_factor * w
            y1 -= self.scale_factor * h
            x2 += self.scale_factor * w
            y2 += self.scale_factor * h
        elif action == 6: # Trigger / Stop
            done = True

        # Clip box bounds and maintain minimum size of 15x15
        x1 = max(0.0, min(float(self.img_w - 15), x1))
        y1 = max(0.0, min(float(self.img_h - 15), y1))
        x2 = max(x1 + 15.0, min(float(self.img_w), x2))
        y2 = max(y1 + 15.0, min(float(self.img_h), y2))

        self.current_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.trajectory.append(self.current_bbox.copy().tolist())

        current_iou = compute_iou_bbox(self.current_bbox, self.gt_bbox)

        # Reward formulation
        step_penalty = 0.02
        reward = (current_iou - self.prev_iou) - step_penalty
        self.prev_iou = current_iou

        if action == 6:
            # Terminal Trigger reward
            if current_iou > 0.5:
                reward += 10.0 * current_iou + 5.0
            else:
                reward += 10.0 * current_iou - 2.0

        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_observation()
        info = compute_metrics(self.current_bbox, self.gt_mask, self.gt_bbox)
        info['trajectory'] = self.trajectory

        return obs, reward, done, info
