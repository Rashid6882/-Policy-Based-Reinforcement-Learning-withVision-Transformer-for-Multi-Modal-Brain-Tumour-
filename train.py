import os
import argparse
import json
import numpy as np
import torch
from dataset import get_dataloader
from environment import BrainLesionEnv
from agent_dqn import DQNAgent, DoubleDQNAgent

def train_agent(agent_type='dqn', num_episodes=50, num_samples=100, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training {agent_type.upper()} Agent on {device} ---")

    data_dir = r"BraTS2020_training_data/content/data"
    dataset = get_dataloader(data_dir, filter_tumors_only=True, max_samples=num_samples)
    env = BrainLesionEnv(max_steps=20)

    if agent_type == 'dqn':
        agent = DQNAgent(device=device)
    else:
        agent = DoubleDQNAgent(device=device)

    history = {
        'episode': [],
        'reward': [],
        'loss': [],
        'dice': [],
        'iou': [],
        'sensitivity': [],
        'precision': [],
        'q_overestimation': []
    }

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.95

    epsilon = eps_start
    target_update_freq = 5

    for ep in range(1, num_episodes + 1):
        sample = dataset[ep % len(dataset)]
        obs = env.reset(sample)
        
        ep_reward = 0.0
        ep_loss = []
        done = False

        while not done:
            action = agent.select_action(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action)

            agent.memory.push(
                obs['roi'], obs['box_feat'],
                action, reward,
                next_obs['roi'], next_obs['box_feat'],
                done
            )

            loss = agent.train_step(batch_size=32)
            if loss > 0:
                ep_loss.append(loss)

            obs = next_obs
            ep_reward += reward

        epsilon = max(eps_end, epsilon * eps_decay)

        if ep % target_update_freq == 0:
            agent.update_target_network()

        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        
        history['episode'].append(ep)
        history['reward'].append(float(ep_reward))
        history['loss'].append(mean_loss)
        history['dice'].append(info['dice'])
        history['iou'].append(info['iou'])
        history['sensitivity'].append(info['sensitivity'])
        history['precision'].append(info['precision'])

        # Compute Q-value estimation metric
        with torch.no_grad():
            roi_t = obs['roi'].unsqueeze(0).to(device)
            box_t = obs['box_feat'].unsqueeze(0).to(device)
            q_val = float(agent.q_net(roi_t, box_t).max().item())
            history['q_overestimation'].append(q_val)

        if ep % 10 == 0 or ep == num_episodes:
            print(f"Ep {ep}/{num_episodes} | Reward: {ep_reward:.2f} | Loss: {mean_loss:.4f} | Dice: {info['dice']:.4f} | IoU: {info['iou']:.4f}", flush=True)

    # Save model weights and training statistics
    model_path = os.path.join(save_dir, f"{agent_type}_model.pt")
    torch.save(agent.q_net.state_dict(), model_path)
    
    stats_path = os.path.join(save_dir, f"{agent_type}_history.json")
    with open(stats_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Saved {agent_type.upper()} model to {model_path} and stats to {stats_path}", flush=True)
    return agent, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=60, help='Number of training episodes')
    parser.add_argument('--samples', type=int, default=100, help='Number of dataset slice samples')
    args = parser.parse_args()

    print("=== Training DQN Baseline ===")
    train_agent(agent_type='dqn', num_episodes=args.episodes, num_samples=args.samples)

    print("\n=== Training Double-DQN Baseline ===")
    train_agent(agent_type='double_dqn', num_episodes=args.episodes, num_samples=args.samples)
