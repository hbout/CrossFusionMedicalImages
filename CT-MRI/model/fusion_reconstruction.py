import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class FusionReconstruction(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()  # sortie normalisée 0-1
        )

    def forward(self, x):
        return self.final(x)
