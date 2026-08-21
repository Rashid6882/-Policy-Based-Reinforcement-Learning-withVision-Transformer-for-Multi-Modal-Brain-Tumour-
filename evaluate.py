import os
import json
import numpy as np
import torch
from dataset import get_dataloader
from environment import BrainLesionEnv
from agent_dqn import DQNAgent, DoubleDQNAgent

def evaluate_models(num_eval_samples=20, checkpoints_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r"BraTS2020_training_data/content/data"
    dataset = get_dataloader(data_dir, filter_tumors_only=True, max_samples=num_eval_samples + 50)
    
    # Select evaluation samples from the end of dataset list
    eval_samples = [dataset[i] for i in range(len(dataset) - num_eval_samples, len(dataset))]
    env = BrainLesionEnv(max_steps=20)

    models = {}
    
    dqn_path = os.path.join(checkpoints_dir, 'dqn_model.pt')
    if os.path.exists(dqn_path):
        dqn_agent = DQNAgent(device=device)
        dqn_agent.q_net.load_state_dict(torch.load(dqn_path, map_location=device))
        dqn_agent.q_net.eval()
        models['DQN'] = dqn_agent

    ddqn_path = os.path.join(checkpoints_dir, 'double_dqn_model.pt')
    if os.path.exists(ddqn_path):
        ddqn_agent = DoubleDQNAgent(device=device)
        ddqn_agent.q_net.load_state_dict(torch.load(ddqn_path, map_location=device))
        ddqn_agent.q_net.eval()
        models['Double-DQN'] = ddqn_agent

    results = {}
    trajectories_data = []

    for model_name, agent in models.items():
        metrics_list = {'dice': [], 'iou': [], 'sensitivity': [], 'precision': [], 'steps': []}
        
        for idx, sample in enumerate(eval_samples):
            obs = env.reset(sample)
            done = False
            step_count = 0
            
            while not done:
                action = agent.select_action(obs, epsilon=0.0) # Greedy action
                obs, reward, done, info = env.step(action)
                step_count += 1

            metrics_list['dice'].append(info['dice'])
            metrics_list['iou'].append(info['iou'])
            metrics_list['sensitivity'].append(info['sensitivity'])
            metrics_list['precision'].append(info['precision'])
            metrics_list['steps'].append(step_count)

            if model_name == 'Double-DQN' or len(trajectories_data) < num_eval_samples:
                trajectories_data.append({
                    'sample_id': idx,
                    'file_name': sample['file_name'],
                    'gt_bbox': sample['gt_bbox'].tolist(),
                    'pred_bbox': env.current_bbox.tolist(),
                    'model': model_name,
                    'trajectory': info['trajectory'],
                    'metrics': info
                })

        results[model_name] = {
            'mean_dice': float(np.mean(metrics_list['dice'])),
            'std_dice': float(np.std(metrics_list['dice'])),
            'mean_iou': float(np.mean(metrics_list['iou'])),
            'std_iou': float(np.std(metrics_list['iou'])),
            'mean_sensitivity': float(np.mean(metrics_list['sensitivity'])),
            'mean_precision': float(np.mean(metrics_list['precision'])),
            'mean_steps_to_converge': float(np.mean(metrics_list['steps']))
        }

    eval_out_path = os.path.join(checkpoints_dir, 'evaluation_summary.json')
    with open(eval_out_path, 'w') as f:
        json.dump(results, f, indent=4)

    traj_out_path = os.path.join(checkpoints_dir, 'trajectories.json')
    with open(traj_out_path, 'w') as f:
        json.dump(trajectories_data, f, indent=4)

    print("\n================ EVALUATION SUMMARY ================")
    for model_name, res in results.items():
        print(f"[{model_name}]")
        print(f"  Dice Score:       {res['mean_dice']:.4f} ± {res['std_dice']:.4f}")
        print(f"  IoU Score:        {res['mean_iou']:.4f} ± {res['std_iou']:.4f}")
        print(f"  Sensitivity:      {res['mean_sensitivity']:.4f}")
        print(f"  Precision:        {res['mean_precision']:.4f}")
        print(f"  Avg Steps:        {res['mean_steps_to_converge']:.2f}")
        print("--------------------------------------------------")

    return results

if __name__ == '__main__':
    evaluate_models()
