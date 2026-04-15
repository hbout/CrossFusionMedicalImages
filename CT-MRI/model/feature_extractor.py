import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.encoder(x)
