import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .cross_attention import CrossAttention
from .fusion_reconstruction import FusionReconstruction

class FusionModel(nn.Module):
    def __init__(self, in_channels=1, feat_channels=32, patch_size=64):
        super().__init__()
        self.irm_branch = FeatureExtractor(in_channels, feat_channels)
        self.ct_branch = FeatureExtractor(in_channels, feat_channels)
        self.cross_att = CrossAttention(feat_channels, patch_size=patch_size, fusion_mode="concat")
        self.reconstruction = FusionReconstruction(feat_channels, out_channels=1)

    def forward(self, irm, ct):
        irm_feat = self.irm_branch(irm)
        ct_feat = self.ct_branch(ct)
        fused_feat = self.cross_att(irm_feat, ct_feat)
        fused_Y = self.reconstruction(fused_feat)
        return fused_Y, irm_feat, ct_feat
