import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=40000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state: {"image": [4, 64, 64], "coords": [4]}
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        
        images = torch.tensor(np.array([b[0]["image"] for b in batch]), dtype=torch.float32, device=device)
        coords = torch.tensor(np.array([b[0]["coords"] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        
        next_images = torch.tensor(np.array([b[3]["image"] for b in batch]), dtype=torch.float32, device=device)
        next_coords = torch.tensor(np.array([b[3]["coords"] for b in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

        return (images, coords), actions, rewards, (next_images, next_coords), dones

    def __len__(self):
        return len(self.buffer)
