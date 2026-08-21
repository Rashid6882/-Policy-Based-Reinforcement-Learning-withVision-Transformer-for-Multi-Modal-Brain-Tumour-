import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import get_dataloader
from environment import BrainLesionEnv, compute_metrics
from agent_dqn import DQNAgent, DoubleDQNAgent

def run_testing_and_visualization(num_test_samples=10, save_dir='test_results'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Standalone Model Testing & Bounding Box Visualization on {device} ---")

    data_dir = r"BraTS2020_training_data/content/data"
    # Load dataset slices for testing
    dataset = get_dataloader(data_dir, filter_tumors_only=True, max_samples=num_test_samples + 30)
    test_samples = [dataset[i] for i in range(len(dataset) - num_test_samples, len(dataset))]
    
    env = BrainLesionEnv(max_steps=20)

    # Load Trained Agents
    checkpoints_dir = 'checkpoints'
    dqn_path = os.path.join(checkpoints_dir, 'dqn_model.pt')
    ddqn_path = os.path.join(checkpoints_dir, 'double_dqn_model.pt')

    dqn_agent = None
    if os.path.exists(dqn_path):
        dqn_agent = DQNAgent(device=device)
        dqn_agent.q_net.load_state_dict(torch.load(dqn_path, map_location=device))
        dqn_agent.q_net.eval()

    ddqn_agent = None
    if os.path.exists(ddqn_path):
        ddqn_agent = DoubleDQNAgent(device=device)
        ddqn_agent.q_net.load_state_dict(torch.load(ddqn_path, map_location=device))
        ddqn_agent.q_net.eval()

    test_metrics = {
        'DQN': {'dice': [], 'iou': [], 'sensitivity': [], 'precision': [], 'steps': []},
        'Double-DQN': {'dice': [], 'iou': [], 'sensitivity': [], 'precision': [], 'steps': []}
    }

    visualization_results = []

    for idx, sample in enumerate(test_samples):
        # 1. Run Standard DQN Inference
        dqn_box = [10.0, 10.0, 230.0, 230.0]
        dqn_traj = []
        dqn_info = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'precision': 0.0}
        
        if dqn_agent:
            obs = env.reset(sample)
            done = False
            step_c = 0
            while not done:
                action = dqn_agent.select_action(obs, epsilon=0.0)
                obs, reward, done, info = env.step(action)
                step_c += 1
            dqn_box = env.current_bbox.tolist()
            dqn_traj = info['trajectory']
            dqn_info = info
            test_metrics['DQN']['dice'].append(info['dice'])
            test_metrics['DQN']['iou'].append(info['iou'])
            test_metrics['DQN']['sensitivity'].append(info['sensitivity'])
            test_metrics['DQN']['precision'].append(info['precision'])
            test_metrics['DQN']['steps'].append(step_c)

        # 2. Run Double-DQN Inference
        ddqn_box = [10.0, 10.0, 230.0, 230.0]
        ddqn_traj = []
        ddqn_info = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'precision': 0.0}

        if ddqn_agent:
            obs = env.reset(sample)
            done = False
            step_c = 0
            while not done:
                action = ddqn_agent.select_action(obs, epsilon=0.0)
                obs, reward, done, info = env.step(action)
                step_c += 1
            ddqn_box = env.current_bbox.tolist()
            ddqn_traj = info['trajectory']
            ddqn_info = info
            test_metrics['Double-DQN']['dice'].append(info['dice'])
            test_metrics['Double-DQN']['iou'].append(info['iou'])
            test_metrics['Double-DQN']['sensitivity'].append(info['sensitivity'])
            test_metrics['Double-DQN']['precision'].append(info['precision'])
            test_metrics['Double-DQN']['steps'].append(step_c)

        # 3. Create 4-Modality Plot with Overlaid Bounding Boxes
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
        image_np = sample['image'] # (4, 240, 240)
        gt_box = sample['gt_bbox'] # [x1, y1, x2, y2]

        for m_idx in range(4):
            ax = axes[m_idx]
            mod_img = image_np[m_idx]
            ax.imshow(mod_img, cmap='gray')
            ax.set_title(f"Modality: {modalities[m_idx]}", fontsize=12, fontweight='bold', color='navy')
            ax.axis('off')

            # Ground Truth Box (Green Solid)
            gt_w = gt_box[2] - gt_box[0]
            gt_h = gt_box[3] - gt_box[1]
            rect_gt = patches.Rectangle((gt_box[0], gt_box[1]), gt_w, gt_h, linewidth=2.5, edgecolor='#00e676', facecolor='none', label='Ground Truth')
            ax.add_patch(rect_gt)

            # DQN Box (Red Dashed)
            dqn_w = dqn_box[2] - dqn_box[0]
            dqn_h = dqn_box[3] - dqn_box[1]
            rect_dqn = patches.Rectangle((dqn_box[0], dqn_box[1]), dqn_w, dqn_h, linewidth=2.0, edgecolor='#ff1744', linestyle='--', facecolor='none', label='DQN Pred')
            ax.add_patch(rect_dqn)

            # Double-DQN Box (Cyan Solid)
            ddqn_w = ddqn_box[2] - ddqn_box[0]
            ddqn_h = ddqn_box[3] - ddqn_box[1]
            rect_ddqn = patches.Rectangle((ddqn_box[0], ddqn_box[1]), ddqn_w, ddqn_h, linewidth=2.5, edgecolor='#00e5ff', facecolor='none', label='Double-DQN Pred')
            ax.add_patch(rect_ddqn)

            if m_idx == 0:
                ax.legend(loc='upper right', fontsize=9)

        plt.suptitle(f"Test Sample #{idx+1} ({sample['file_name']})\nDDQN Dice: {ddqn_info['dice']:.4f} | IoU: {ddqn_info['iou']:.4f}  vs  DQN Dice: {dqn_info['dice']:.4f} | IoU: {dqn_info['iou']:.4f}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        save_fig_path = os.path.join(save_dir, f"test_result_sample_{idx+1}.png")
        plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        visualization_results.append({
            'sample_id': idx + 1,
            'file_name': sample['file_name'],
            'image_path': save_fig_path,
            'gt_bbox': gt_box.tolist(),
            'dqn_bbox': dqn_box,
            'ddqn_bbox': ddqn_box,
            'dqn_metrics': dqn_info,
            'ddqn_metrics': ddqn_info
        })

    # Summary Statistics Calculation
    summary = {}
    for agent_name in ['DQN', 'Double-DQN']:
        m = test_metrics[agent_name]
        summary[agent_name] = {
            'test_mean_dice': float(np.mean(m['dice'])) if m['dice'] else 0.0,
            'test_std_dice': float(np.std(m['dice'])) if m['dice'] else 0.0,
            'test_mean_iou': float(np.mean(m['iou'])) if m['iou'] else 0.0,
            'test_std_iou': float(np.std(m['iou'])) if m['iou'] else 0.0,
            'test_mean_sensitivity': float(np.mean(m['sensitivity'])) if m['sensitivity'] else 0.0,
            'test_mean_precision': float(np.mean(m['precision'])) if m['precision'] else 0.0,
            'test_mean_steps': float(np.mean(m['steps'])) if m['steps'] else 0.0
        }

    summary_path = os.path.join(save_dir, "test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary, 'samples': visualization_results}, f, indent=4)

    print("\n================ STANDALONE TEST RESULTS SUMMARY ================")
    for agent_name, res in summary.items():
        print(f"[{agent_name}]")
        print(f"  Test Dice Score:       {res['test_mean_dice']:.4f} ± {res['test_std_dice']:.4f}")
        print(f"  Test IoU Score:        {res['test_mean_iou']:.4f} ± {res['test_std_iou']:.4f}")
        print(f"  Test Sensitivity:      {res['test_mean_sensitivity']:.4f}")
        print(f"  Test Precision:        {res['test_mean_precision']:.4f}")
        print(f"  Test Avg Steps:        {res['test_mean_steps']:.2f}")
        print("---------------------------------------------------------------")

    print(f"Generated {len(visualization_results)} bounding box visual result figures in '{save_dir}/'")
    return summary, visualization_results

if __name__ == '__main__':
    run_testing_and_visualization(num_test_samples=5)
