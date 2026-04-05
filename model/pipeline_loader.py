# (FULL FILE — REPLACE ENTIRE FILE)

"""
model/pipeline_loader.py
Upgraded:
  - Proper LoRA stacking (no overwrite)
  - Better scheduler tuning (prompt adherence)
  - Improved memory optimizations
  - Optional CPU offload
  - Safer refiner handling
"""

import logging
from typing import Optional, List

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

# ── Model IDs ─────────────────────────────────────────
SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
SDXL_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# ── Scheduler Map ─────────────────────────────────────
SCHEDULER_MAP = {
    "dpm++": DPMSolverMultistepScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
}

# ── Singleton storage ─────────────────────────────────
_base_pipe: Optional[StableDiffusionXLPipeline] = None
_img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
_refiner_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None

# Track loaded LoRAs
_loaded_loras: List[str] = []


# ─────────────────────────────────────────────────────
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype() -> torch.dtype:
    return torch.float16 if get_device() == "cuda" else torch.float32


# ─────────────────────────────────────────────────────
def _apply_memory_optimizations(pipe, device: str):
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("✓ xformers enabled")
        except Exception:
            logger.info("xformers not available → using SDPA")

    # Optional CPU offload (safe fallback)
    try:
        pipe.enable_model_cpu_offload()
        logger.info("✓ CPU offload enabled")
    except Exception:
        pass


# ─────────────────────────────────────────────────────
def _load_vae(dtype):
    logger.info("Loading fixed VAE...")
    return AutoencoderKL.from_pretrained(SDXL_VAE_ID, torch_dtype=dtype)


# ─────────────────────────────────────────────────────
def _configure_scheduler(pipe, scheduler_name: str):
    cls = SCHEDULER_MAP.get(scheduler_name, DPMSolverMultistepScheduler)

    pipe.scheduler = cls.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )

    logger.info(f"✓ Scheduler: {scheduler_name}")


# ─────────────────────────────────────────────────────
def load_pipelines(
    scheduler_name: str = "dpm++",
    load_refiner: bool = True,
    enable_safety: bool = False,
):
    global _base_pipe, _img2img_pipe, _refiner_pipe

    if _base_pipe is not None:
        logger.info("Pipelines already loaded")
        return

    device = get_device()
    dtype = get_dtype()

    logger.info(f"Loading pipelines | {device} | {dtype}")

    vae = _load_vae(dtype)

    # ── Base ──────────────────────────────────────────
    safety_kwargs = {} if enable_safety else {
        "safety_checker": None,
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

    # ── img2img (shared weights) ──────────────────────
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=_base_pipe.vae,
        text_encoder=_base_pipe.text_encoder,
        text_encoder_2=_base_pipe.text_encoder_2,
        tokenizer=_base_pipe.tokenizer,
        tokenizer_2=_base_pipe.tokenizer_2,
        unet=_base_pipe.unet,
        scheduler=_base_pipe.scheduler,
    ).to(device)

    # ── Refiner ───────────────────────────────────────
    if load_refiner:
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

    logger.info("✓ Pipelines ready")


# ─────────────────────────────────────────────────────
def unload_pipelines():
    global _base_pipe, _img2img_pipe, _refiner_pipe

    _base_pipe = None
    _img2img_pipe = None
    _refiner_pipe = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("✓ Pipelines unloaded")


# ─────────────────────────────────────────────────────
def get_base_pipe():
    return _base_pipe


def get_img2img_pipe():
    return _img2img_pipe


def get_refiner_pipe():
    return _refiner_pipe


def pipelines_loaded():
    return _base_pipe is not None


# ─────────────────────────────────────────────────────
# 🔥 FIXED LoRA STACKING (IMPORTANT)
# ─────────────────────────────────────────────────────
def load_lora_weights(
    lora_path: str,
    lora_scale: float = 0.8,
):
    global _loaded_loras

    if _base_pipe is None:
        raise RuntimeError("Load pipelines first")

    adapter_name = f"lora_{len(_loaded_loras)}"

    for pipe in [_base_pipe, _img2img_pipe]:
        if pipe is None:
            continue

        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

        _loaded_loras.append(adapter_name)

        pipe.set_adapters(
            _loaded_loras,
            adapter_weights=[lora_scale] * len(_loaded_loras),
        )

    logger.info(f"✓ LoRA stacked: {lora_path}")


def unload_lora_weights():
    global _loaded_loras

    for pipe in [_base_pipe, _img2img_pipe]:
        if pipe is not None:
            pipe.delete_adapters(_loaded_loras)

    _loaded_loras = []
    logger.info("✓ All LoRAs unloaded")


# ─────────────────────────────────────────────────────
def swap_scheduler(scheduler_name: str):
    for pipe in [_base_pipe, _img2img_pipe, _refiner_pipe]:
        if pipe:
            _configure_scheduler(pipe, scheduler_name)
