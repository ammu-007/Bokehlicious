"""
training/loss.py — Bokehlicious training loss (L1 + λ·LPIPS_VGG).
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity as lpips_fn,
)


class BokehliciousLoss(nn.Module):
    """
    Paper Eq. 5:  L = L1(output, target) + lambda * LPIPS_VGG(output, target)
    lambda = 0.6 by default.
    """
    def __init__(self, lambda_lpips: float = 0.6):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.l1 = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l1_loss = self.l1(output, target)
        lpips_loss = lpips_fn(output.clamp(0, 1), target.clamp(0, 1),
                              normalize=True, net_type='vgg')
        total = l1_loss + self.lambda_lpips * lpips_loss
        return total, {
            'l1':    l1_loss.item(),
            'lpips': lpips_loss.item(),
            'total': total.item(),
        }
