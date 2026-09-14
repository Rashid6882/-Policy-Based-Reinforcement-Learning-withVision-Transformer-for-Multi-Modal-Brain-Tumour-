import os
import json
import argparse
import numpy as np
import torch

from environment.brain_lesion_env import BrainLesionEnv
from agents.dqn_agent import DQNAgent
from .metrics import compute_subregion_metrics

def evaluate_agent(agent, env, case_ids, bbox_selection="best_dice"):
    results = []
    
    for case_id in case_ids:
        obs, info = env.reset(options={"case_id": case_id})
        done = False
        truncated = False

        best_dice = -1.0
        best_bbox = None
        best_step_idx = 0

        step_idx = 0
        while not (done or truncated):
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            step_idx += 1
            
            curr_dice = info["dice"]
            if curr_dice > best_dice:
                best_dice = curr_dice
                best_bbox = list(info["bbox"])
                best_step_idx = step_idx

        eval_bbox = best_bbox if (bbox_selection == "best_dice" and best_bbox is not None) else info["bbox"]
        
        # Subregion metrics
        metrics = compute_subregion_metrics(eval_bbox, env.gt_mask)
        results.append({
            "case_id": case_id,
            "best_step_fraction": float(best_step_idx / max(1, step_idx)),
            "best_dice": float(best_dice),
            "final_dice": float(info["dice"]),
            "subregions": metrics
        })

    # Summary
    summary = {}
    for region in ["WT", "TC", "ET"]:
        summary[region] = {
            "dice_mean": float(np.mean([r["subregions"][region]["dice"] for r in results])),
            "dice_std": float(np.std([r["subregions"][region]["dice"] for r in results])),
            "iou_mean": float(np.mean([r["subregions"][region]["iou"] for r in results])),
            "iou_std": float(np.std([r["subregions"][region]["iou"] for r in results])),
            "sens_mean": float(np.mean([r["subregions"][region]["sensitivity"] for r in results])),
            "prec_mean": float(np.mean([r["subregions"][region]["precision"] for r in results]))
        }
        
    summary["mean_best_step_frac"] = float(np.mean([r["best_step_fraction"] for r in results]))
    return summary, results

def run_eval(encoder_type="cnn", checkpoint_path=None, output_json=None):
    data_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    split_file = "data/splits/train_val_split.json"
    
    with open(split_file, "r") as f:
        split_data = json.load(f)
    val_cases = split_data["val"]

    env = BrainLesionEnv(
        data_dir=data_dir,
        case_ids=val_cases,
        max_steps=20,
        subregion="WT",
        randomize_initial_window=False,
        initial_scale=0.5
    )

    agent = DQNAgent(
        encoder_type=encoder_type,
        in_channels=4,
        num_actions=7,
        feature_dim=256
    )

    if checkpoint_path is None:
        checkpoint_path = f"outputs/{encoder_type}_dqn/checkpoints/model_best.pt"

    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found! Evaluating randomly initialized agent.")

    summary, results = evaluate_agent(agent, env, val_cases, bbox_selection="best_dice")

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"summary": summary, "per_case": results}, f, indent=4)
        print(f"Saved eval summary to {output_json}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_eval(args.model_type, args.checkpoint, args.output)
