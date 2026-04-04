"""
model/pipeline_loader.py
========================
Responsible for loading, configuring, and caching SDXL pipelines.

Handles:
  - SDXL Base pipeline (txt2img + img2img)
  - SDXL Refiner pipeline (optional high-quality final pass)
  - LoRA weight loading
  - Scheduler swapping (DPMSolver++, Euler, DDIM, etc.)
  - GPU memory optimizations (float16, xformers, attention slicing, tiled VAE)
  - Singleton pattern — model loaded once, reused across all requests
"""

import logging
from typing import Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL,
)

logger = logging.getLogger(__name__)

# ── Model identifiers ─────────────────────────────────────────────────────────
SDXL_BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

# High-quality VAE that fixes colour saturation issues in SDXL
SDXL_VAE_ID     = "madebyollin/sdxl-vae-fp16-fix"

# Scheduler name → class mapping
SCHEDULER_MAP = {
    "dpm++":  DPMSolverMultistepScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim":   DDIMScheduler,
}

# ── Singleton storage ─────────────────────────────────────────────────────────
_base_pipe:     Optional[StableDiffusionXLPipeline]         = None
_img2img_pipe:  Optional[StableDiffusionXLImg2ImgPipeline]  = None
_refiner_pipe:  Optional[StableDiffusionXLImg2ImgPipeline]  = None


# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype() -> torch.dtype:
    return torch.float16 if get_device() == "cuda" else torch.float32


def _apply_memory_optimizations(pipe, device: str) -> None:
    """
    Apply all available memory and speed optimizations to a pipeline.
    Order matters — xformers must come after .to(device).
    """
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("  ✓ xformers memory-efficient attention enabled")
        except Exception:
            logger.info("  ✗ xformers not available — using PyTorch SDPA fallback")
            # torch >= 2.0 provides scaled_dot_product_attention natively
            pipe.unet.set_attn_processor(
                __import__(
                    "diffusers.models.attention_processor",
                    fromlist=["AttnProcessor2_0"]
                ).AttnProcessor2_0()
            )

    logger.info("  ✓ attention slicing, VAE slicing + tiling enabled")


def _load_vae(dtype: torch.dtype) -> AutoencoderKL:
    """Load the fp16-fixed SDXL VAE separately to avoid colour drift."""
    logger.info(f"  Loading fixed VAE ({SDXL_VAE_ID}) ...")
    vae = AutoencoderKL.from_pretrained(SDXL_VAE_ID, torch_dtype=dtype)
    logger.info("  ✓ VAE loaded")
    return vae


def _configure_scheduler(pipe, scheduler_name: str) -> None:
    """Swap the pipeline's scheduler in-place."""
    cls = SCHEDULER_MAP.get(scheduler_name, DPMSolverMultistepScheduler)
    pipe.scheduler = cls.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,          # sharper results
        algorithm_type="dpmsolver++",    # ignored by non-DPM schedulers
    )
    logger.info(f"  ✓ Scheduler set to {cls.__name__} (karras sigmas)")


# ─────────────────────────────────────────────────────────────────────────────
def load_pipelines(
    scheduler_name:   str  = "dpm++",
    load_refiner:     bool = True,
    enable_safety:    bool = False,
) -> None:
    """
    Load SDXL Base (txt2img + img2img share weights) and optionally the Refiner.
    Idempotent — calling twice is a no-op.
    """
    global _base_pipe, _img2img_pipe, _refiner_pipe

    if _base_pipe is not None:
        logger.info("Pipelines already loaded — skipping.")
        return

    device = get_device()
    dtype  = get_dtype()

    logger.info(f"Loading SDXL pipelines | device={device} | dtype={dtype}")

    # ── 1. Shared fixed VAE ───────────────────────────────────────────────────
    vae = _load_vae(dtype)

    # ── 2. SDXL Base (txt2img) ────────────────────────────────────────────────
    logger.info(f"  Loading SDXL Base ({SDXL_BASE_ID}) ...")
    safety_kwargs = {} if enable_safety else {
        "safety_checker":         None,
        "requires_safety_checker": False,
    }
    _base_pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE_ID,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
        add_watermarker=False,
        **safety_kwargs,
    ).to(device)

    _configure_scheduler(_base_pipe, scheduler_name)
    _apply_memory_optimizations(_base_pipe, device)
    logger.info("  ✓ SDXL Base loaded")

    # ── 3. SDXL img2img (shares VAE + UNet — near-zero extra VRAM) ───────────
    logger.info("  Building img2img pipeline from base weights ...")
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=_base_pipe.vae,
        text_encoder=_base_pipe.text_encoder,
        text_encoder_2=_base_pipe.text_encoder_2,
        tokenizer=_base_pipe.tokenizer,
        tokenizer_2=_base_pipe.tokenizer_2,
        unet=_base_pipe.unet,
        scheduler=_base_pipe.scheduler,
    ).to(device)
  logger.info("  ✓ img2img pipeline ready (shared weights)")

    # ── 4. Optional SDXL Refiner ─────────────────────────────────────────────
    if load_refiner:
        logger.info(f"  Loading SDXL Refiner ({SDXL_REFINER_ID}) ...")
        _refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL_REFINER_ID,
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            add_watermarker=False,
        ).to(device)
        _configure_scheduler(_refiner_pipe, scheduler_name)
        _apply_memory_optimizations(_refiner_pipe, device)
        logger.info("  ✓ Refiner loaded")
    else:
        logger.info("  ✗ Refiner skipped (load_refiner=False)")

    logger.info("All pipelines ready.")


def unload_pipelines() -> None:
    """Release all pipeline memory (used during shutdown)."""
    global _base_pipe, _img2img_pipe, _refiner_pipe
    for name, pipe in [
        ("base", _base_pipe),
        ("img2img", _img2img_pipe),
        ("refiner", _refiner_pipe),
    ]:
        if pipe is not None:
            del pipe
            logger.info(f"  Released {name} pipeline")
    _base_pipe = _img2img_pipe = _refiner_pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("All pipelines unloaded.")


# ── Public accessors ──────────────────────────────────────────────────────────
def get_base_pipe()     -> Optional[StableDiffusionXLPipeline]:
    return _base_pipe

def get_img2img_pipe()  -> Optional[StableDiffusionXLImg2ImgPipeline]:
    return _img2img_pipe

def get_refiner_pipe()  -> Optional[StableDiffusionXLImg2ImgPipeline]:
    return _refiner_pipe

def pipelines_loaded()  -> bool:
    return _base_pipe is not None


# ── LoRA loader ───────────────────────────────────────────────────────────────
def load_lora_weights(
    lora_path:  str,
    lora_scale: float = 0.9,
    adapter_name: str = "default",
) -> None:
    """
    Load a LoRA checkpoint into the base + img2img pipelines.

    Args:
        lora_path:    Local path or HuggingFace repo ID (e.g. 'civitai/...')
        lora_scale:   Blend weight 0.0–1.0
        adapter_name: Name used to identify the adapter (for multi-LoRA stacking)
    """
    if _base_pipe is None:
        raise RuntimeError("Pipelines not loaded. Call load_pipelines() first.")

    logger.info(f"Loading LoRA: {lora_path} (scale={lora_scale})")

    for pipe_name, pipe in [("base", _base_pipe), ("img2img", _img2img_pipe)]:
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        logger.info(f"  ✓ LoRA applied to {pipe_name}")


def unload_lora_weights(adapter_name: str = "default") -> None:
    """Remove a previously loaded LoRA from both pipelines."""
    if _base_pipe is None:
        return
    for pipe in [_base_pipe, _img2img_pipe]:
        if pipe is not None:
            pipe.delete_adapters([adapter_name])
    logger.info(f"LoRA '{adapter_name}' unloaded.")


def swap_scheduler(scheduler_name: str) -> None:
    """Hot-swap scheduler on all loaded pipelines without reloading models."""
    for pipe in [_base_pipe, _img2img_pipe, _refiner_pipe]:
        if pipe is not None:
            _configure_scheduler(pipe, scheduler_name)
    logger.info(f"Scheduler swapped to: {scheduler_name}")
