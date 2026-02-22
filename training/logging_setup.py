"""
training/logging_setup.py — Rank-aware logging, IST timestamps, and reproducibility helpers.
"""

import datetime
import logging
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rank-aware Logging
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
    All timestamps are in IST (UTC+5:30).
    """
    logger = logging.getLogger("bokehlicious")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # prevent duplicate handlers on re-calls

    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

    def _ist_time(*args):
        return datetime.datetime.now(IST).timetuple()

    fmt = logging.Formatter(
        "[%(asctime)s IST] [Rank %(rank)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = _ist_time

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


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

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
