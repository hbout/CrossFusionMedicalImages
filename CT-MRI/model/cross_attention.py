import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class CrossAttention(nn.Module):
    def __init__(self, channels=32, patch_size=64, fusion_mode="concat"):
        super().__init__()
        self.patch_size = patch_size
        self.fusion_mode = fusion_mode

        self.query_conv_irm = nn.Conv2d(channels, channels, 1)
        self.key_conv_ct = nn.Conv2d(channels, channels, 1)
        self.value_conv_ct = nn.Conv2d(channels, channels, 1)

        self.query_conv_ct = nn.Conv2d(channels, channels, 1)
        self.key_conv_irm = nn.Conv2d(channels, channels, 1)
        self.value_conv_irm = nn.Conv2d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=-1)
        
        if fusion_mode == "concat":
            self.reduce = nn.Sequential(
                nn.Conv2d(channels*2, channels*2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels*2, channels, 1)
            )

        self.post_residual = nn.Sequential(
            ResidualBlock(channels)
        )

    def _attend(self, query_feat, key_feat, value_feat, ref_patch):
        B, C, H, W = query_feat.size()
        
        query = query_feat.view(B, C, -1).permute(0, 2, 1)
        key = key_feat.view(B, C, -1)
        value = value_feat.view(B, C, -1)

        d_k = key.size(1)
        att = self.softmax(torch.bmm(query, key) / (d_k ** 0.5))
        patch_out = torch.bmm(value, att.transpose(1, 2))
        
        return patch_out.view(B, C, H, W) + ref_patch

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        P = self.patch_size

        out_a2b = torch.zeros_like(x1)
        out_b2a = torch.zeros_like(x2)

        h_steps = list(range(0, H, P))
        w_steps = list(range(0, W, P))

        for i in h_steps:
            for j in w_steps:
                x1_patch = x1[:, :, i:i+P, j:j+P]
                x2_patch = x2[:, :, i:i+P, j:j+P]

                attended_a2b = self._attend(
                    self.query_conv_irm(x1_patch),
                    self.key_conv_ct(x2_patch),
                    self.value_conv_ct(x2_patch),
                    x1_patch
                )
                out_a2b[:, :, i:i+P, j:j+P] = attended_a2b

                attended_b2a = self._attend(
                    self.query_conv_ct(x2_patch),
                    self.key_conv_irm(x1_patch),
                    self.value_conv_irm(x1_patch),
                    x2_patch
                )
                out_b2a[:, :, i:i+P, j:j+P] = attended_b2a

        out = torch.cat([out_a2b, out_b2a], dim=1)
        out = self.reduce(out)
        return self.post_residual(out)