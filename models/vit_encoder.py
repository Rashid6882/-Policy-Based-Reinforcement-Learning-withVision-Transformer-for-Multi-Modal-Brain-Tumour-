import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, patch_size=8, embed_dim=128, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 4, 64, 64] -> [B, embed_dim, 8, 8]
        x = self.proj(x)
        # flatten spatial dims: [B, embed_dim, 64] -> transpose to [B, 64, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTEncoder(nn.Module):
    """
    ViT Version: Vision Transformer Spatial Feature Encoder with Multi-Head Self-Attention.
    Processes multi-modal crop patch tensor [B, 4, 64, 64] -> [B, feature_dim].
    """
    def __init__(self, in_channels=4, img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4, feature_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, feature_dim)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x) # [B, num_patches + 1, embed_dim]
        x = self.norm(x)

        cls_out = x[:, 0] # Extract CLS token representation [B, embed_dim]
        out = self.head(cls_out) # [B, feature_dim]
        return out
