import os
import json
import numpy as np
import matplotlib.pyplot as plt
from .evaluate import run_eval

def compare_cnn_vs_vit(output_dir="outputs/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Running Evaluation for Non-ViT (CNN) Model ---")
    cnn_summary = run_eval("cnn", output_json=os.path.join(output_dir, "cnn_eval_summary.json"))

    print("--- Running Evaluation for ViT (Vision Transformer) Model ---")
    vit_summary = run_eval("vit", output_json=os.path.join(output_dir, "vit_eval_summary.json"))

    comparison_report = {
        "cnn": cnn_summary,
        "vit": vit_summary
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison_report, f, indent=4)

    # Plot Comparison Bar Chart
    regions = ["WT", "TC", "ET"]
    cnn_dices = [cnn_summary[r]["dice_mean"] for r in regions]
    vit_dices = [vit_summary[r]["dice_mean"] for r in regions]

    cnn_ious = [cnn_summary[r]["iou_mean"] for r in regions]
    vit_ious = [vit_summary[r]["iou_mean"] for r in regions]

    x = np.arange(len(regions))
    width = 0.35

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Dice comparison
    rects1 = ax[0].bar(x - width/2, cnn_dices, width, label='Non-ViT (CNN)', color='steelblue')
    rects2 = ax[0].bar(x + width/2, vit_dices, width, label='ViT (Transformer)', color='darkorange')
    ax[0].set_ylabel('Dice Score (DSC)')
    ax[0].set_title('Subregion Dice Score: Non-ViT vs ViT')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(regions)
    ax[0].legend()
    ax[0].set_ylim(0, 1.0)
    for r in rects1:
        h = r.get_height()
        ax[0].annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    for r in rects2:
        h = r.get_height()
        ax[0].annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

    # IoU comparison
    rects3 = ax[1].bar(x - width/2, cnn_ious, width, label='Non-ViT (CNN)', color='steelblue')
    rects4 = ax[1].bar(x + width/2, vit_ious, width, label='ViT (Transformer)', color='darkorange')
    ax[1].set_ylabel('Jaccard Index (IoU)')
    ax[1].set_title('Subregion IoU: Non-ViT vs ViT')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(regions)
    ax[1].legend()
    ax[1].set_ylim(0, 1.0)
    for r in rects3:
        h = r.get_height()
        ax[1].annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    for r in rects4:
        h = r.get_height()
        ax[1].annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Write Markdown Report
    md_content = f"""# Quantitative Comparison: Non-ViT (CNN) vs Vision Transformer (ViT)

## Evaluation Results Summary

| Subregion | Metric | Non-ViT (CNN) | ViT (Vision Transformer) | Improvement (ViT vs CNN) |
|---|---|---|---|---|
| **WT (Whole Tumor)** | Dice Score | {cnn_summary['WT']['dice_mean']:.4f} ± {cnn_summary['WT']['dice_std']:.4f} | **{vit_summary['WT']['dice_mean']:.4f} ± {vit_summary['WT']['dice_std']:.4f}** | **+{(vit_summary['WT']['dice_mean'] - cnn_summary['WT']['dice_mean']):+.4f}** |
| | Jaccard IoU | {cnn_summary['WT']['iou_mean']:.4f} ± {cnn_summary['WT']['iou_std']:.4f} | **{vit_summary['WT']['iou_mean']:.4f} ± {vit_summary['WT']['iou_std']:.4f}** | **+{(vit_summary['WT']['iou_mean'] - cnn_summary['WT']['iou_mean']):+.4f}** |
| | Sensitivity | {cnn_summary['WT']['sens_mean']:.4f} | **{vit_summary['WT']['sens_mean']:.4f}** | **+{(vit_summary['WT']['sens_mean'] - cnn_summary['WT']['sens_mean']):+.4f}** |
| | Precision | {cnn_summary['WT']['prec_mean']:.4f} | **{vit_summary['WT']['prec_mean']:.4f}** | **+{(vit_summary['WT']['prec_mean'] - cnn_summary['WT']['prec_mean']):+.4f}** |
| **TC (Tumor Core)** | Dice Score | {cnn_summary['TC']['dice_mean']:.4f} | **{vit_summary['TC']['dice_mean']:.4f}** | **+{(vit_summary['TC']['dice_mean'] - cnn_summary['TC']['dice_mean']):+.4f}** |
| | Jaccard IoU | {cnn_summary['TC']['iou_mean']:.4f} | **{vit_summary['TC']['iou_mean']:.4f}** | **+{(vit_summary['TC']['iou_mean'] - cnn_summary['TC']['iou_mean']):+.4f}** |
| **ET (Enhancing Tumor)** | Dice Score | {cnn_summary['ET']['dice_mean']:.4f} | **{vit_summary['ET']['dice_mean']:.4f}** | **+{(vit_summary['ET']['dice_mean'] - cnn_summary['ET']['dice_mean']):+.4f}** |
| | Jaccard IoU | {cnn_summary['ET']['iou_mean']:.4f} | **{vit_summary['ET']['iou_mean']:.4f}** | **+{(vit_summary['ET']['iou_mean'] - cnn_summary['ET']['iou_mean']):+.4f}** |

- **Mean Best-Step Fraction**:
  - Non-ViT (CNN): {cnn_summary['mean_best_step_frac']*100:.1f}% of episode
  - ViT (Transformer): {vit_summary['mean_best_step_frac']*100:.1f}% of episode

![Metrics Comparison](metrics_comparison.png)
"""
    with open(os.path.join(output_dir, "model_comparison.md"), "w") as f:
        f.write(md_content)

    print(f"Comparison report saved to {os.path.join(output_dir, 'model_comparison.md')}")
    return comparison_report

if __name__ == "__main__":
    compare_cnn_vs_vit()
