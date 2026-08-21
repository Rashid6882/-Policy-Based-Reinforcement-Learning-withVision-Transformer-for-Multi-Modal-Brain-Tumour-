import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQNNetwork

class ReplayBuffer:
    """
    Experience Replay Memory for DQN & Double-DQN agents.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, roi, box_feat, action, reward, next_roi, next_box_feat, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            roi.cpu(), box_feat.cpu(), action, reward, next_roi.cpu(), next_box_feat.cpu(), done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        rois, box_feats, actions, rewards, next_rois, next_box_feats, dones = zip(*batch)

        rois_t = torch.stack(rois).to(device)
        box_feats_t = torch.stack(box_feats).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_rois_t = torch.stack(next_rois).to(device)
        next_box_feats_t = torch.stack(next_box_feats).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        return rois_t, box_feats_t, actions_t, rewards_t, next_rois_t, next_box_feats_t, dones_t

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Standard Deep Q-Network Agent.
    """
    def __init__(self, action_dim=7, lr=1e-4, gamma=0.99, buffer_capacity=10000, device='cpu'):
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        self.q_net = DQNNetwork(action_dim=action_dim).to(device)
        self.target_net = DQNNetwork(action_dim=action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            roi = obs['roi'].unsqueeze(0).to(self.device)
            box_feat = obs['box_feat'].unsqueeze(0).to(self.device)
            q_values = self.q_net(roi, box_feat)
            return torch.argmax(q_values, dim=1).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0

        rois, box_feats, actions, rewards, next_rois, next_box_feats, dones = self.memory.sample(batch_size, self.device)

        # Current Q-values
        q_values = self.q_net(rois, box_feats)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Vanilla DQN max action over target net)
        with torch.no_grad():
            next_q_values = self.target_net(next_rois, next_box_feats)
            max_next_q = next_q_values.max(1)[0]
            expected_state_action_values = rewards + (self.gamma * max_next_q * (1.0 - dones))

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

class DoubleDQNAgent(DQNAgent):
    """
    Double Deep Q-Network Agent (DDQN).
    Decouples action selection from target value evaluation to reduce overestimation bias.
    """
    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0

        rois, box_feats, actions, rewards, next_rois, next_box_feats, dones = self.memory.sample(batch_size, self.device)

        # Current Q-values
        q_values = self.q_net(rois, box_feats)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target:
        # 1. Select best action using ONLINE network: a* = argmax Q_online(s', a)
        # 2. Evaluate target value using TARGET network: Q_target(s', a*)
        with torch.no_grad():
            next_q_online = self.q_net(next_rois, next_box_feats)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_net(next_rois, next_box_feats)
            double_q_values = next_q_target.gather(1, best_actions).squeeze(1)

            expected_state_action_values = rewards + (self.gamma * double_q_values * (1.0 - dones))

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())
