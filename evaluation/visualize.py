import os
import matplotlib.pyplot as plt
import numpy as np

def generate_trajectory_plot(case_id, slice_2d, gt_mask, trajectory, output_path):
    # slice_2d: [4, H, W], modality 1 is t1c or 3 is t2f
    img = slice_2d[1] # T1c modality for clear anatomical visualization
    H, W = img.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    
    # Overlay ground truth contour
    if gt_mask is not None and gt_mask.max() > 0:
        ax.contour(gt_mask > 0, colors='red', linewidths=1.5, levels=[0.5])

    # Draw step boxes
    best_step = max(trajectory, key=lambda x: x["dice"])
    for step_info in trajectory:
        bbox = step_info["bbox"]
        step = step_info["step"]
        xmin, ymin, xmax, ymax = bbox
        
        px_xmin = xmin * W
        px_ymin = ymin * H
        px_xmax = xmax * W
        px_ymax = ymax * H
        
        is_best = (step == best_step["step"])
        color = 'gold' if is_best else 'cyan'
        linewidth = 2.5 if is_best else 1.0
        alpha = 1.0 if is_best else 0.5
        
        rect = plt.Rectangle((px_xmin, px_ymin), px_xmax - px_xmin, px_ymax - px_ymin,
                             fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)
        if is_best:
            ax.text(px_xmin, px_ymin - 3, f"Best Step {step} (Dice: {step_info['dice']:.2f})",
                    color='gold', fontweight='bold', fontsize=10, bbox=dict(facecolor='black', alpha=0.6))

    ax.set_title(f"Case {case_id} Trajectory (Gold = Best Window)")
    ax.axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
