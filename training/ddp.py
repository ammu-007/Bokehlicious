"""
training/ddp.py — Distributed Data Parallel setup and teardown helpers.
"""

import os
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist


def setup_ddp(config_device: str) -> Tuple[int, int, int, str, bool]:
    """
    Initialise DDP if LOCAL_RANK is set, otherwise return single-GPU defaults.

    Returns:
        (local_rank, global_rank, world_size, device, is_ddp)
    """
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        # Increase timeout to 3 hours to accommodate long validation runs.
        # DDP's forward() broadcasts buffers as a collective op (BROADCAST),
        # so ALL ranks must participate.  If rank 0 is validating for 30+ min
        # while other ranks are waiting, the default 30-min timeout kills them.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
        local_rank  = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank, global_rank, world_size = 0, 0, 1
        device = config_device

    return local_rank, global_rank, world_size, device, is_ddp


def cleanup_ddp(is_ddp: bool):
    """Destroy the process group if DDP is active."""
    if is_ddp:
        dist.destroy_process_group()
