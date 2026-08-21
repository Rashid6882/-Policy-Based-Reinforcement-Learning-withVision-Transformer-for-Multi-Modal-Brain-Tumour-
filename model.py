import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN Feature Extractor for 4-channel (T1, T1ce, T2, FLAIR) ROI patches.
    """
    def __init__(self, in_channels=4, feature_dim=512):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, feature_dim)

    def forward(self, x):
        # x: (N, 4, 64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x) # (N, 32, 32, 32)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x) # (N, 64, 16, 16)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # (N, 128, 8, 8)
        
        x = x.view(x.size(0), -1) # (N, 128*8*8 = 8192)
        features = F.relu(self.fc(x)) # (N, 512)
        return features

class DQNNetwork(nn.Module):
    """
    Deep Q-Network combining CNN ROI features and normalized spatial box features.
    """
    def __init__(self, in_channels=4, box_dim=5, action_dim=7):
        super(DQNNetwork, self).__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=256)
        
        self.fc_head = nn.Sequential(
            nn.Linear(256 + box_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, roi, box_feat):
        # roi: (N, 4, 64, 64)
        # box_feat: (N, 5)
        roi_feats = self.encoder(roi) # (N, 256)
        combined = torch.cat([roi_feats, box_feat], dim=1) # (N, 261)
        q_values = self.fc_head(combined) # (N, action_dim)
        return q_values
