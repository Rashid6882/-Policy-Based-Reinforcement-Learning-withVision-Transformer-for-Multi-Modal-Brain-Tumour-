import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.q_network import QNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, encoder_type="cnn", in_channels=4, num_actions=7, feature_dim=256,
                 lr=1e-4, gamma=0.95, buffer_capacity=40000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9999,
                 target_update_freq=500, use_double_dqn=True, device=None):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder_type = encoder_type
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn

        self.online_net = QNetwork(encoder_type=encoder_type, in_channels=in_channels,
                                   num_actions=num_actions, feature_dim=feature_dim).to(self.device)
        self.target_net = QNetwork(encoder_type=encoder_type, in_channels=in_channels,
                                   num_actions=num_actions, feature_dim=feature_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0

    def select_action(self, obs, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        img = torch.tensor(obs["image"], dtype=torch.float32, device=self.device).unsqueeze(0)
        coords = torch.tensor(obs["coords"], dtype=torch.float32, device=self.device).unsqueeze(0)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(img, coords)
            action = q_values.argmax(dim=1).item()
        self.online_net.train()

        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        (images, coords), actions, rewards, (next_images, next_coords), dones = self.memory.sample(self.batch_size, self.device)

        # Current Q-values
        curr_q = self.online_net(images, coords).gather(1, actions)

        # Target Q-values calculation
        with torch.no_grad():
            if self.use_double_dqn:
                next_online_actions = self.online_net(next_images, next_coords).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_images, next_coords).gather(1, next_online_actions)
            else:
                next_q = self.target_net(next_images, next_coords).max(dim=1, keepdim=True)[0]

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, filepath):
        torch.save({
            'encoder_type': self.encoder_type,
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_steps = checkpoint.get('train_steps', 0)
