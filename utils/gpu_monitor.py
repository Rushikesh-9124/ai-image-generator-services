"""
utils/gpu_monitor.py
Upgraded:
  - Accurate VRAM tracking (allocated + reserved)
  - Peak memory stats
  - Multi-GPU safe
  - Better utilization parsing
  - Fragmentation awareness
"""

import logging
from typing import Optional
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────

@dataclass
class GPUStats:
    available: bool
    device_name: Optional[str]
    total_gb: Optional[float]
    used_gb: Optional[float]        # allocated
    reserved_gb: Optional[float]    # reserved (important)
    free_gb: Optional[float]
    used_pct: Optional[float]
    utilization_pct: Optional[str]
    peak_gb: Optional[float]


# ─────────────────────────────────────────
# Core
# ─────────────────────────────────────────

def get_gpu_stats(device_index: int = 0) -> GPUStats:
    if not torch.cuda.is_available():
        return GPUStats(False, None, None, None, None, None, None, None, None)

    try:
        props = torch.cuda.get_device_properties(device_index)

        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        peak = torch.cuda.max_memory_allocated(device_index)

        free = total - reserved   # more realistic than allocated

        total_gb = total / 1024**3
        used_gb = allocated / 1024**3
        reserved_gb = reserved / 1024**3
        free_gb = free / 1024**3
        peak_gb = peak / 1024**3

        used_pct = (reserved / total) * 100

        util = _get_gpu_utilization(device_index)

        return GPUStats(
            available=True,
            device_name=props.name,
            total_gb=round(total_gb, 2),
            used_gb=round(used_gb, 2),
            reserved_gb=round(reserved_gb, 2),
            free_gb=round(free_gb, 2),
            used_pct=round(used_pct, 1),
            utilization_pct=util,
            peak_gb=round(peak_gb, 2),
        )

    except Exception as e:
        logger.warning(f"GPU stats error: {e}")
        return GPUStats(False, None, None, None, None, None, None, None, None)


# ─────────────────────────────────────────
# Utilization helper
# ─────────────────────────────────────────

def _get_gpu_utilization(device_index: int) -> Optional[str]:
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )

        lines = result.stdout.strip().split("\n")
        if len(lines) > device_index:
            return f"{lines[device_index].strip()} %"

        return "unknown"

    except Exception:
        return "unavailable"


# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────

def log_gpu_stats(prefix: str = "", device_index: int = 0) -> None:
    s = get_gpu_stats(device_index)

    if not s.available:
        logger.info(f"{prefix}GPU: not available")
        return

    logger.info(
        f"{prefix}GPU {s.device_name} | "
        f"VRAM: {s.reserved_gb:.1f}/{s.total_gb:.1f} GB "
        f"({s.used_pct:.0f}%) | "
        f"alloc={s.used_gb:.1f}GB peak={s.peak_gb:.1f}GB | "
        f"util: {s.utilization_pct}"
    )


# ─────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────

def warn_if_low_vram(threshold_gb: float = 2.0) -> None:
    s = get_gpu_stats()

    if s.available and s.free_gb is not None and s.free_gb < threshold_gb:
        logger.warning(
            f"⚠️ Low VRAM: {s.free_gb:.2f} GB free | "
            f"Used: {s.reserved_gb:.2f}/{s.total_gb:.2f} GB"
        )


# ─────────────────────────────────────────
# Extra (NEW but optional)
# ─────────────────────────────────────────

def reset_peak_memory() -> None:
    """Reset peak VRAM stats (use before inference benchmarking)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_peak_memory(prefix: str = "") -> None:
    """Log peak VRAM usage."""
    if not torch.cuda.is_available():
        return

    peak = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"{prefix}Peak VRAM usage: {peak:.2f} GB")
