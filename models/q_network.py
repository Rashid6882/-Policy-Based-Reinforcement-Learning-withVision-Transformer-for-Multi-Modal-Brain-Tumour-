import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
from .vit_encoder import ViTEncoder

class QNetwork(nn.Module):
    """
    Unified Q-Network supporting both Non-ViT (CNN) and ViT feature encoders.
    Fuses spatial visual representation [B, feature_dim] with bounding box coordinates [B, 4].
    """
    def __init__(self, encoder_type="cnn", in_channels=4, num_actions=7, feature_dim=256):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "cnn":
            self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=feature_dim)
        elif self.encoder_type == "vit":
            self.encoder = ViTEncoder(in_channels=in_channels, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'cnn' or 'vit'.")

        self.coord_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True)
        )

        self.q_head = nn.Sequential(
            nn.Linear(feature_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

    def forward(self, image_crop, coords):
        # image_crop: [B, 4, 64, 64], coords: [B, 4]
        feat = self.encoder(image_crop)
        coord_feat = self.coord_fc(coords)
        fused = torch.cat([feat, coord_feat], dim=1)
        q_values = self.q_head(fused)
        return q_values
