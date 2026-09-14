import os
import json
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.split_dataset import create_split
from environment.brain_lesion_env import BrainLesionEnv
from agents.dqn_agent import DQNAgent

def evaluate_during_training(agent, env, num_eval_episodes=20):
    val_dices = []
    val_rewards = []
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        best_dice = info["dice"]
        
        while not (done or truncated):
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if info["dice"] > best_dice:
                best_dice = info["dice"]
                
        val_dices.append(best_dice)
        val_rewards.append(ep_reward)
        
    return float(np.mean(val_dices)), float(np.mean(val_rewards))

def train(config_path, episodes_override=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if episodes_override:
        cfg['training']['episodes'] = episodes_override

    data_dir = cfg['data']['data_dir']
    split_file = cfg['data']['split_file']

    if not os.path.exists(split_file):
        create_split(data_dir, split_file, sample_ratio=0.5, seed=42)

    with open(split_file, 'r') as f:
        split_data = json.load(f)
    train_cases = split_data['train']
    val_cases = split_data['val']

    output_dir = cfg['training']['output_dir']
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    train_env = BrainLesionEnv(
        data_dir=data_dir,
        case_ids=train_cases,
        max_steps=cfg['environment']['max_steps'],
        subregion=cfg['data']['subregion'],
        randomize_initial_window=cfg['environment']['randomize_initial_window'],
        initial_scale=cfg['environment']['initial_scale']
    )

    val_env = BrainLesionEnv(
        data_dir=data_dir,
        case_ids=val_cases,
        max_steps=cfg['environment']['max_steps'],
        subregion=cfg['data']['subregion'],
        randomize_initial_window=False,
        initial_scale=cfg['environment']['initial_scale']
    )

    agent = DQNAgent(
        encoder_type=cfg['model']['encoder_type'],
        in_channels=cfg['model']['in_channels'],
        num_actions=cfg['model']['num_actions'],
        feature_dim=cfg['model']['feature_dim'],
        lr=cfg['agent']['lr'],
        gamma=cfg['agent']['gamma'],
        buffer_capacity=cfg['agent']['buffer_capacity'],
        batch_size=cfg['agent']['batch_size'],
        epsilon_start=cfg['agent']['epsilon_start'],
        epsilon_end=cfg['agent']['epsilon_end'],
        epsilon_decay=cfg['agent']['epsilon_decay'],
        target_update_freq=cfg['agent']['target_update_freq'],
        use_double_dqn=cfg['agent']['use_double_dqn']
    )

    total_episodes = cfg['training']['episodes']
    eval_freq = cfg['training']['eval_freq']
    save_freq = cfg['training']['save_freq']

    history = {
        "episode_rewards": [],
        "episode_losses": [],
        "val_dices": [],
        "val_episodes": [],
        "best_val_dice": -1.0
    }

    print(f"--- Starting Training ({cfg['model']['encoder_type'].upper()} Encoder) for {total_episodes} episodes ---")

    for ep in range(1, total_episodes + 1):
        obs, info = train_env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_losses = []

        while not (done or truncated):
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, done, truncated, info = train_env.step(action)
            agent.memory.push(obs, action, reward, next_obs, float(done or truncated))

            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            ep_reward += reward

        history["episode_rewards"].append(ep_reward)
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["episode_losses"].append(mean_loss)

        if ep % eval_freq == 0 or ep == total_episodes:
            val_dice, val_reward = evaluate_during_training(agent, val_env)
            history["val_dices"].append(val_dice)
            history["val_episodes"].append(ep)

            print(f"Episode {ep}/{total_episodes} | Train Reward: {ep_reward:.2f} | Loss: {mean_loss:.4f} | Val Dice: {val_dice:.4f} | Epsilon: {agent.epsilon:.3f}")

            if val_dice > history["best_val_dice"]:
                history["best_val_dice"] = val_dice
                best_ckpt = os.path.join(output_dir, "checkpoints", "model_best.pt")
                agent.save(best_ckpt)

        if ep % save_freq == 0:
            latest_ckpt = os.path.join(output_dir, "checkpoints", "model_latest.pt")
            agent.save(latest_ckpt)

    final_ckpt = os.path.join(output_dir, "checkpoints", "model_final.pt")
    agent.save(final_ckpt)

    with open(os.path.join(output_dir, "logs", "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Plot curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["episode_rewards"], label="Train Reward", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward Curve ({cfg['model']['encoder_type'].upper()})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_episodes"], history["val_dices"], label="Val Dice (Best-Step)", color="orange", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Dice Score")
    plt.title(f"Validation Dice ({cfg['model']['encoder_type'].upper()})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_curves.png"), dpi=200)
    plt.close()

    print(f"Training finished for {cfg['model']['encoder_type'].upper()}. Best Val Dice: {history['best_val_dice']:.4f}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.episodes)
