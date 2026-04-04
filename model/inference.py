"""
model/inference.py
==================
Core inference engine.

Handles:
  - txt2img via SDXL Base
  - img2img via SDXL Base img2img pipeline
  - Optional refiner pass (base → refiner hand-off)
  - Reproducible seed control
  - PIL image pre/post processing
  - Async-wrapped sync inference (non-blocking event loop)
"""

import asyncio
import logging
import time
from typing import Optional

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from model.pipeline_loader import (
    get_base_pipe,
    get_img2img_pipe,
    get_refiner_pipe,
    get_device,
    pipelines_loaded,
)

logger = logging.getLogger(__name__)

# ── Refiner settings ──────────────────────────────────────────────────────────
# The base handles the first (1 - REFINER_HANDOFF_FRACTION) of denoising steps.
# The refiner handles the final REFINER_HANDOFF_FRACTION to add detail.
REFINER_HANDOFF_FRACTION = 0.2    # last 20 % of steps → refiner


# ─────────────────────────────────────────────────────────────────────────────
#  Image pre-processing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_generator(seed: int) -> torch.Generator:
    device = get_device()
    return torch.Generator(device=device).manual_seed(seed)


def _preprocess_init_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """
    Resize + convert input image for img2img.

    We use LANCZOS for quality downscaling and ensure the image is RGB
    (handles RGBA / palette PNG inputs gracefully).
    """
    image = image.convert("RGB")
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def _postprocess_output(image: Image.Image, sharpen: bool = True) -> Image.Image:
    """
    Optional post-processing to boost perceptual quality:
      - Slight sharpening (UnsharpMask)
      - Mild contrast enhancement
    Both are very subtle — the goal is to match DSLR output, not over-process.
    """
    if sharpen:
        image = image.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3)
        )
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
    return image


def _snap_to_multiple(value: int, multiple: int = 8) -> int:
    """SDXL UNet requires dimensions divisible by 8."""
    return (value // multiple) * multiple


# ─────────────────────────────────────────────────────────────────────────────
#  Core synchronous inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_txt2img(
    prompt:          str,
    negative_prompt: str,
    steps:           int,
    guidance_scale:  float,
    width:           int,
    height:          int,
    seed:            int,
    use_refiner:     bool,
    refiner_steps:   int,
) -> Image.Image:
    """
    SDXL txt2img with optional refiner.

    When use_refiner=True:
      1. Base runs (1 - REFINER_HANDOFF_FRACTION) of steps, outputs latents
      2. Refiner polishes with the remaining fraction for crisp detail
    """
    base    = get_base_pipe()
    refiner = get_refiner_pipe()

    width  = _snap_to_multiple(width)
    height = _snap_to_multiple(height)
    gen    = _make_generator(seed)

    if use_refiner and refiner is not None:
        # ── Two-pass: base → latents → refiner ───────────────────────────────
        denoising_end = 1.0 - REFINER_HANDOFF_FRACTION

        with torch.inference_mode():
            latents = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
                denoising_end=denoising_end,
                output_type="latent",
            ).images                      # still latent tensors here

        gen_refiner = _make_generator(seed)
        with torch.inference_mode():
            result = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=latents,
                num_inference_steps=refiner_steps,
                denoising_start=denoising_end,
                guidance_scale=guidance_scale,
                generator=gen_refiner,
            )
        image = result.images[0]
    else:
        # ── Single-pass base only ─────────────────────────────────────────────
        with torch.inference_mode():
            result = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
            )
        image = result.images[0]

    return _postprocess_output(image)


def _run_img2img(
    init_image:      Image.Image,
    prompt:          str,
    negative_prompt: str,
    steps:           int,
    guidance_scale:  float,
    width:           int,
    height:          int,
    strength:        float,
    seed:            int,
    use_refiner:     bool,
    refiner_steps:   int,
) -> Image.Image:
    """
    SDXL img2img with optional refiner polish.

    strength controls how much the model deviates from the source image:
      0.0 → identical to input
      1.0 → completely reimagined (effectively txt2img)
      0.4–0.7 → ideal for realistic edits while preserving structure
    """
    pipe    = get_img2img_pipe()
    refiner = get_refiner_pipe()

    width  = _snap_to_multiple(width)
    height = _snap_to_multiple(height)

    init_image = _preprocess_init_image(init_image, width, height)
    gen        = _make_generator(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
    image = result.images[0]

    # Optional refiner pass on the img2img output for extra detail
    if use_refiner and refiner is not None:
        gen_refiner = _make_generator(seed)
        with torch.inference_mode():
            result = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=0.35,          # light refinement pass — preserve edits
                num_inference_steps=refiner_steps,
                guidance_scale=guidance_scale,
                generator=gen_refiner,
            )
        image = result.images[0]

    return _postprocess_output(image)


# ─────────────────────────────────────────────────────────────────────────────
#  Public async interface
# ─────────────────────────────────────────────────────────────────────────────

async def generate_txt2img(
    prompt:          str,
    negative_prompt: str,
    steps:           int,
    guidance_scale:  float,
    width:           int,
    height:          int,
    seed:            int,
    use_refiner:     bool   = True,
    refiner_steps:   int    = 20,
) -> tuple[Image.Image, float]:
    """
    Async wrapper around txt2img inference.
    Returns (PIL image, elapsed_seconds).
    """
    if not pipelines_loaded():
        raise RuntimeError("Pipelines not loaded.")

    loop  = asyncio.get_event_loop()
    start = time.time()

    image = await loop.run_in_executor(
        None,
        _run_txt2img,
        prompt, negative_prompt, steps, guidance_scale,
        width, height, seed, use_refiner, refiner_steps,
    )

    return image, round(time.time() - start, 2)


async def generate_img2img(
    init_image:      Image.Image,
    prompt:          str,
    negative_prompt: str,
    steps:           int,
    guidance_scale:  float,
    width:           int,
    height:          int,
    strength:        float,
    seed:            int,
    use_refiner:     bool = True,
    refiner_steps:   int  = 20,
) -> tuple[Image.Image, float]:
    """
    Async wrapper around img2img inference.
    Returns (PIL image, elapsed_seconds).
    """
    if not pipelines_loaded():
        raise RuntimeError("Pipelines not loaded.")

    loop  = asyncio.get_event_loop()
    start = time.time()

    image = await loop.run_in_executor(
        None,
        _run_img2img,
        init_image, prompt, negative_prompt, steps, guidance_scale,
        width, height, strength, seed, use_refiner, refiner_steps,
    )

    return image, round(time.time() - start, 2)