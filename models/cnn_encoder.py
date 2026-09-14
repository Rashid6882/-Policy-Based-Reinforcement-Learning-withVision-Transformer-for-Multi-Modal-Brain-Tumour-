import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class CNNEncoder(nn.Module):
    """
    Non-ViT Version: ResNet-style 2D CNN Feature Extractor.
    Processes multi-modal crop patch tensor [B, 4, 64, 64] -> [B, feature_dim].
    """
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), # -> [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # -> [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(64)
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # -> [128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResBlock(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
