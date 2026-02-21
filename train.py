"""
train.py — Bokehlicious Training Script
Reproduces paper results: L1 + 0.6*LPIPS_VGG loss, Adam lr=5e-4, batch=4, patch=512x512.

Usage:
    # Train from parquet dataset (recommended):
    .venv\\Scripts\\python.exe train.py -size small -data_path ./dataset/RealBokeh_Parquet --parquet

    # Train from raw imagefolder dataset:
    .venv\\Scripts\\python.exe train.py -size small -data_path ./dataset/RealBokeh_3MP

    # Resume from checkpoint:
    .venv\\Scripts\\python.exe train.py -size small -data_path ./dataset/RealBokeh_Parquet --parquet -resume ./checkpoints/small_epoch10.pt

    # Smoke test (CPU, no data needed beyond a tiny parquet file):
    .venv\\Scripts\\python.exe train.py -size small -data_path ./dataset/RealBokeh_Parquet --parquet -epochs 1 -batch_size 1 -patch_size 128 -num_workers 0 -device cpu
"""

from typing import Tuple
import yaml
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr_fn,
    structural_similarity_index_measure as ssim_fn,
    learned_perceptual_image_patch_similarity as lpips_fn,
)

from dataset.loader import RealBokehTrain, RealBokeh, RealBokehParquet, RealBokehExtracted
from dataset.util import Mode
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious
from util.parser import get_train_parser


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

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
        # lpips_fn returns a scalar Tensor
        lpips_loss = lpips_fn(output.clamp(0, 1), target.clamp(0, 1),
                              normalize=True, net_type='vgg')
        total = l1_loss + self.lambda_lpips * lpips_loss
        return total, {
            'l1':    l1_loss.item(),
            'lpips': lpips_loss.item(),
            'total': total.item(),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: Bokehlicious, val_dataset, device: str) -> dict:
    """Run validation. Accepts any dataset that returns per-sample dicts."""
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for idx in tqdm(range(len(val_dataset)), desc='Validating', leave=False):
        batch = val_dataset[idx]
        # Move tensors to device and add batch dim
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    # Parse config values
    exp_name = config['experiment']['name']
    
    size   = config['model']['size']
    device = config['model']['device']
    
    data_path    = Path(config['data']['path'])
    dataset_type = config['data'].get('dataset_type', 'imagefolder')
    num_workers  = config['data']['num_workers']
    
    epochs       = config['training']['epochs']
    batch_size   = config['training']['batch_size']
    patch_size   = config['training']['patch_size']
    lr           = float(config['training']['lr'])
    lambda_lpips = float(config['training']['lambda_lpips'])
    val_freq     = config['training']['val_freq']
    save_freq    = config['training']['save_freq']
    resume       = config['training'].get('resume')
    
    checkpoint_dir = Path(config['logging']['checkpoint_dir']) / exp_name
    log_dir        = Path(config['logging']['log_dir']) / exp_name

    # ---- DDP Setup ----
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank, global_rank, world_size = 0, 0, 1
        device = config['model']['device']

    if global_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    model_config = bokehlicious_size_builder(size)
    model = Bokehlicious(**model_config).to(device)
    
    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Initialized Bokehlicious-{size} ({total_params:,} parameters) on {device}")
        print(f"Experiment: {exp_name} | DDP: {is_ddp} (World Size: {world_size})")

    # ---- Resume ----
    start_epoch = 0
    best_psnr = 0.0
    if resume is not None and resume != "null" and resume != "":
        if global_rank == 0:
            print(f"Resuming from checkpoint: {resume}")
        state = torch.load(resume, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
            start_epoch = state.get('epoch', 0) + 1
            best_psnr = state.get('best_psnr', 0.0)
            if global_rank == 0:
                print(f"  Resumed at epoch {start_epoch}, best PSNR: {best_psnr:.4f}")
        else:
            model.load_state_dict(state)
            if global_rank == 0:
                print("  Loaded plain weights dict.")

    # Wrap model in DDP after loading checkpoint
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ---- Datasets ----
    if dataset_type == 'extracted':
        train_dir = data_path / 'train'
        val_dir   = data_path / 'validation'
        train_dataset = RealBokehExtracted(train_dir, patch_size=patch_size, augment=True)
        val_dataset   = RealBokehExtracted(val_dir,   patch_size=None,       augment=False)
    elif dataset_type == 'parquet':
        train_dir = data_path / 'train'
        val_dir   = data_path / 'validation'
        train_dataset = RealBokehParquet(train_dir, patch_size=patch_size, augment=True)
        val_dataset   = RealBokehParquet(val_dir,   patch_size=None,       augment=False)
    else:
        train_dataset = RealBokehTrain(data_path, patch_size=patch_size)
        val_dataset   = RealBokeh(data_path, mode=Mode.VAL, device='cpu')

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.startswith('cuda') or device == 'cuda'),
        drop_last=True if not is_ddp else False,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    # ---- Loss ----
    criterion = BokehliciousLoss(lambda_lpips=lambda_lpips)

    # ---- TensorBoard ----
    if global_rank == 0:
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: tensorboard --logdir {log_dir.parent}\n")
    else:
        writer = None

    # ---- Training ----
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0.0
        
        if global_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = train_loader

        for batch in pbar:
            source             = batch['source'].to(device)
            target             = batch['target'].to(device)
            bokeh_strength     = batch['bokeh_strength'].to(device)
            pos_map            = batch['pos_map'].to(device)
            bokeh_strength_map = batch['bokeh_strength_map'].to(device)

            optimizer.zero_grad()
            output = model(
                source=source,
                bokeh_strength=bokeh_strength,
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
            )
            loss, loss_dict = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_dict['total']
            global_step += 1

            if global_rank == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['total']:.4f}",
                    l1=f"{loss_dict['l1']:.4f}",
                    lpips=f"{loss_dict['lpips']:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % 100 == 0:
                    writer.add_scalar('train/loss_total', loss_dict['total'], global_step)
                    writer.add_scalar('train/loss_l1',    loss_dict['l1'],    global_step)
                    writer.add_scalar('train/loss_lpips', loss_dict['lpips'], global_step)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        if is_ddp:
            # Sync epoch loss across GPUs
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item()
            
        scheduler.step()
        avg_loss = epoch_loss / (len(train_loader) * world_size)
        
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}")
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # Periodic checkpoint
        if global_rank == 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = checkpoint_dir / f"{size}_epoch{epoch+1}.pt"
            torch.save({
                'model':     model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':     epoch,
                'best_psnr': best_psnr,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Validation
        if (epoch + 1) % val_freq == 0:
            if global_rank == 0:
                print("  Running validation...")
                metrics = validate(model.module if is_ddp else model, val_dataset, device)
                print(f"  Val PSNR: {metrics['psnr']:.4f} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}")
                writer.add_scalar('val/psnr',  metrics['psnr'],  epoch)
                writer.add_scalar('val/ssim',  metrics['ssim'],  epoch)
                writer.add_scalar('val/lpips', metrics['lpips'], epoch)

                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    best_path = checkpoint_dir / f"{size}_best.pt"
                    torch.save(model.module.state_dict() if is_ddp else model.state_dict(), best_path)
                    print(f"  New best PSNR {best_psnr:.4f} -> saved to {best_path}")
            
            # Synchronize so other processes don't race ahead
            if is_ddp:
                dist.barrier()

    if global_rank == 0:
        writer.close()
        print(f"\nTraining complete. Best PSNR: {best_psnr:.4f}")
        print(f"Best model: {checkpoint_dir / f'{size}_best.pt'}")
        
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train(config)
