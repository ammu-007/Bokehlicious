"""
training/validation.py — Full-image validation loop.
"""

import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr_fn,
    structural_similarity_index_measure as ssim_fn,
    learned_perceptual_image_patch_similarity as lpips_fn,
)


@torch.no_grad()
def validate(model: nn.Module, val_dataset, device: str,
             logger: logging.Logger) -> dict:
    """Run validation on rank 0. Returns metric averages."""
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for idx in tqdm(range(len(val_dataset)), desc='Validating', leave=False):
        batch = val_dataset[idx]
        inputs = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}

        output = model(
            source=inputs['source'],
            bokeh_strength=inputs['bokeh_strength'],
            pos_map=inputs['pos_map'],
            bokeh_strength_map=inputs['bokeh_strength_map'],
        ).clamp(0, 1)

        target = inputs['target']
        psnr_vals.append(psnr_fn(output, target, data_range=1.0).item())
        ssim_vals.append(ssim_fn(output, target, data_range=1.0).item())
        lpips_vals.append(lpips_fn(output, target, normalize=True).item())

    model.train()
    n = len(psnr_vals)
    return {
        'psnr':  sum(psnr_vals)  / n,
        'ssim':  sum(ssim_vals)  / n,
        'lpips': sum(lpips_vals) / n,
    }
