"""
train.py — Bokehlicious Training Script (DDP-Ready)

Reproduces paper results: L1 + 0.6*LPIPS_VGG loss, Adam lr=5e-4, batch=4, patch=512x512.

Usage:
    # Single-GPU training:
    .venv\\Scripts\\python.exe train.py --config configs/default.yaml

    # Multi-GPU training (e.g. 8× P40):
    .venv\\Scripts\\torchrun.exe --nproc_per_node=8 train.py --config configs/default.yaml

    # Smoke test (CPU):
    .venv\\Scripts\\python.exe train.py --config configs/smoke_test.yaml
"""

import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
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
# Logging Setup
# ---------------------------------------------------------------------------

class RankFilter(logging.Filter):
    """Injects `rank` into every log record."""
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def setup_logging(rank: int, log_dir: Path, log_file: str) -> logging.Logger:
    """
    Configure Python logging with rank-aware formatting.
    - All ranks get a StreamHandler (console) for WARNING+
    - Rank 0 gets an additional StreamHandler for INFO+ and a FileHandler.
    """
    logger = logging.getLogger("bokehlicious")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # prevent duplicate handlers on re-calls

    fmt = logging.Formatter(
        "[%(asctime)s] [Rank %(rank)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rank_filter = RankFilter(rank)

    if rank == 0:
        # Console — INFO level
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        console.addFilter(rank_filter)
        logger.addHandler(console)

        # File — DEBUG level (captures everything)
        fh = logging.FileHandler(str(log_dir / log_file), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.addFilter(rank_filter)
        logger.addHandler(fh)
    else:
        # Non-rank-0 processes: only WARNING+ to console
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(fmt)
        console.addFilter(rank_filter)
        logger.addHandler(console)

    return logger


def get_git_hash() -> Optional[str]:
    """Return the short git commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return None


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    # ---- Parse config ----
    exp_name     = config['experiment']['name']
    size         = config['model']['size']
    device       = config['model']['device']

    data_path    = Path(config['data']['path'])
    dataset_type = config['data'].get('dataset_type', 'imagefolder')
    num_workers  = config['data']['num_workers']

    epochs          = config['training']['epochs']
    batch_size      = config['training']['batch_size']
    patch_size      = config['training']['patch_size']
    lr              = float(config['training']['lr'])
    lambda_lpips    = float(config['training']['lambda_lpips'])
    val_freq        = config['training']['val_freq']
    save_freq       = config['training']['save_freq']
    resume          = config['training'].get('resume')
    seed            = config['training'].get('seed', 42)
    log_interval    = config['training'].get('log_interval', 50)
    grad_clip_norm  = float(config['training'].get('grad_clip_norm', 1.0))
    sample_log_freq = config['training'].get('sample_log_freq', 5)

    checkpoint_dir = Path(config['logging']['checkpoint_dir']) / exp_name
    log_dir        = Path(config['logging']['log_dir']) / exp_name
    log_file       = config['logging'].get('log_file', 'train.log')

    # ---- DDP Setup ----
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank, global_rank, world_size = 0, 0, 1
        device = config['model']['device']

    # ---- Directories (rank 0 creates, then barrier) ----
    if global_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()  # wait for rank 0 to create dirs

    # ---- Logging ----
    logger = setup_logging(global_rank, log_dir, log_file)

    # ---- Reproducibility ----
    set_seeds(seed + global_rank)  # offset per rank for data diversity

    git_hash = get_git_hash()
    logger.info("=" * 70)
    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Git commit : {git_hash or 'N/A'}")
    logger.info(f"DDP        : {is_ddp} | World Size: {world_size} | Rank: {global_rank}")
    logger.info(f"Seed       : {seed} (per-rank offset: +{global_rank})")
    logger.info("=" * 70)

    # ---- Model ----
    model_config = bokehlicious_size_builder(size)
    model = Bokehlicious(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model      : Bokehlicious-{size} ({total_params:,} params) on {device}")

    # ---- Resume ----
    start_epoch = 0
    best_psnr = 0.0
    if resume is not None and resume != "null" and resume != "":
        logger.info(f"Resuming from checkpoint: {resume}")
        state = torch.load(resume, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
            start_epoch = state.get('epoch', 0) + 1
            best_psnr = state.get('best_psnr', 0.0)
            logger.info(f"  -> Resumed at epoch {start_epoch}, best PSNR: {best_psnr:.4f}")
        else:
            model.load_state_dict(state)
            logger.info("  -> Loaded plain state dict (no optimizer/epoch info).")

    # ---- DDP Wrap (after checkpoint load, before optimizer) ----
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Helper to get the unwrapped model for saving / validation
    raw_model = model.module if is_ddp else model

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

    logger.info(f"Dataset    : {dataset_type} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank,
        shuffle=True, drop_last=True
    ) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
        drop_last=(not is_ddp),
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    # Restore optimizer + scheduler state on resume
    if resume is not None and resume != "null" and resume != "":
        if isinstance(state, dict) and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
            logger.info("  -> Restored optimizer state.")
        if isinstance(state, dict) and 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
            logger.info("  -> Restored scheduler state.")
        elif start_epoch > 0:
            # No saved scheduler — fast-forward it
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"  -> Fast-forwarded scheduler to epoch {start_epoch}.")

    # ---- Loss ----
    criterion = BokehliciousLoss(lambda_lpips=lambda_lpips)
    logger.info(f"Loss       : L1 + {lambda_lpips}×LPIPS_VGG")

    # ---- TensorBoard (rank 0 only) ----
    writer: Optional[SummaryWriter] = None
    if global_rank == 0:
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard: tensorboard --logdir {log_dir.parent}")

        # Log config as text
        import json
        writer.add_text('Config', f"```json\n{json.dumps(config, indent=2)}\n```", 0)

        # Log model graph
        try:
            sample = train_dataset[0]
            dummy_input = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            writer.add_graph(raw_model, (
                dummy_input['source'],
                dummy_input['bokeh_strength'],
                dummy_input['pos_map'],
                dummy_input['bokeh_strength_map'],
            ))
            logger.info("  -> Logged model graph to TensorBoard.")
        except Exception as e:
            logger.warning(f"  -> Could not log model graph: {e}")

    logger.info("")

    # ---- Training ----
    global_step = start_epoch * len(train_loader)
    effective_batch = batch_size * world_size

    for epoch in range(start_epoch, epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_l1   = 0.0
        epoch_lpips = 0.0
        iter_count = 0
        epoch_start = time.time()

        if global_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = train_loader

        for batch in pbar:
            step_start = time.time()
            source             = batch['source'].to(device, non_blocking=True)
            target             = batch['target'].to(device, non_blocking=True)
            bokeh_strength     = batch['bokeh_strength'].to(device, non_blocking=True)
            pos_map            = batch['pos_map'].to(device, non_blocking=True)
            bokeh_strength_map = batch['bokeh_strength_map'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                source=source,
                bokeh_strength=bokeh_strength,
                pos_map=pos_map,
                bokeh_strength_map=bokeh_strength_map,
            )
            loss, loss_dict = criterion(output, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            ).item()
            optimizer.step()

            epoch_loss  += loss_dict['total']
            epoch_l1    += loss_dict['l1']
            epoch_lpips += loss_dict['lpips']
            iter_count  += 1
            global_step += 1

            step_time = time.time() - step_start
            throughput = source.shape[0] / step_time if step_time > 0 else 0.0

            # ---- Per-iteration logging (rank 0, every log_interval) ----
            if global_rank == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['total']:.4f}",
                    l1=f"{loss_dict['l1']:.4f}",
                    lpips=f"{loss_dict['lpips']:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    gpu_mem_mb = (torch.cuda.memory_reserved(device) / 1024**2) if device.startswith('cuda') else 0.0

                    logger.info(
                        f"[Epoch {epoch+1}] Step {global_step} | "
                        f"loss={loss_dict['total']:.4f} (L1={loss_dict['l1']:.4f}, LPIPS={loss_dict['lpips']:.4f}) | "
                        f"LR={current_lr:.2e} | grad_norm={grad_norm:.4f} | "
                        f"GPU={gpu_mem_mb:.0f}MB | {throughput:.1f} img/s"
                    )

                    writer.add_scalar('Loss/train_total', loss_dict['total'], global_step)
                    writer.add_scalar('Loss/train_l1',    loss_dict['l1'],    global_step)
                    writer.add_scalar('Loss/train_lpips', loss_dict['lpips'], global_step)
                    writer.add_scalar('LR/lr',            current_lr,         global_step)
                    writer.add_scalar('Gradients/norm',   grad_norm,          global_step)
                    writer.add_scalar('GPU/memory_MB',    gpu_mem_mb,         global_step)
                    writer.add_scalar('Perf/throughput_img_s', throughput,     global_step)

        # ---- End of epoch ----
        if is_ddp:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item()

        scheduler.step()
        avg_loss = epoch_loss / (len(train_loader) * world_size)
        epoch_time = time.time() - epoch_start

        if global_rank == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} complete — "
                f"avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s | "
                f"effective_batch={effective_batch}"
            )
            writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)

            # ---- Weight & gradient histograms ----
            # TensorBoard's add_histogram uses NumPy internally and only
            # supports float32. Cast everything to float32 on CPU first to
            # avoid "No loop matching signature" TypeError with bfloat16/fp16.
            for name, param in raw_model.named_parameters():
                if param.requires_grad:
                    try:
                        writer.add_histogram(
                            f'Weights/{name}',
                            param.data.detach().cpu().float(),
                            epoch,
                        )
                        if param.grad is not None:
                            writer.add_histogram(
                                f'Gradients/{name}',
                                param.grad.data.detach().cpu().float(),
                                epoch,
                            )
                    except Exception as hist_e:
                        logger.warning(f"  add_histogram failed for '{name}': {hist_e}")

            writer.flush()

        # ---- Sample images to TensorBoard ----
        if global_rank == 0 and (epoch + 1) % sample_log_freq == 0:
            try:
                raw_model.eval()
                with torch.no_grad():
                    sample = val_dataset[0]
                    s_in = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample.items()}
                    s_out = raw_model(
                        source=s_in['source'],
                        bokeh_strength=s_in['bokeh_strength'],
                        pos_map=s_in['pos_map'],
                        bokeh_strength_map=s_in['bokeh_strength_map'],
                    ).clamp(0, 1)

                    writer.add_images('Samples/input',  s_in['source'], epoch)
                    writer.add_images('Samples/output', s_out,          epoch)
                    writer.add_images('Samples/target', s_in['target'], epoch)
                raw_model.train()
                logger.debug(f"  Logged sample images to TensorBoard (epoch {epoch+1}).")
            except Exception as e:
                logger.warning(f"  Could not log sample images: {e}")

        # ---- Periodic checkpoint ----
        if global_rank == 0 and (epoch + 1) % save_freq == 0:
            ckpt_path = checkpoint_dir / f"{size}_epoch{epoch+1}.pt"
            torch.save({
                'model':     raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch':     epoch,
                'best_psnr': best_psnr,
                'config':    config,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")
        if is_ddp:
            dist.barrier()  # all ranks wait for rank 0 to finish saving

        # ---- Validation ----
        if (epoch + 1) % val_freq == 0:
            if global_rank == 0:
                logger.info("  Running validation...")
                metrics = validate(raw_model, val_dataset, device, logger)
                logger.info(
                    f"  Val — PSNR: {metrics['psnr']:.4f} | "
                    f"SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}"
                )
                writer.add_scalar('Metrics/PSNR',  metrics['psnr'],  epoch)
                writer.add_scalar('Metrics/SSIM',  metrics['ssim'],  epoch)
                writer.add_scalar('Metrics/LPIPS', metrics['lpips'], epoch)

                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    best_path = checkpoint_dir / f"{size}_best.pt"
                    torch.save(raw_model.state_dict(), best_path)
                    logger.info(f"  * New best PSNR {best_psnr:.4f} → saved to {best_path}")

            if is_ddp:
                dist.barrier()  # sync after validation

    # ---- Cleanup ----
    if global_rank == 0:
        writer.close()
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Training complete. Best PSNR: {best_psnr:.4f}")
        logger.info(f"Best model: {checkpoint_dir / f'{size}_best.pt'}")
        logger.info("=" * 70)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
