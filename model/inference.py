# (FULL FILE — REPLACE ENTIRE FILE)

import asyncio
import logging
import time
from typing import Optional

import torch
from PIL import Image, ImageFilter, ImageEnhance

from model.pipeline_loader import (
    get_base_pipe,
    get_img2img_pipe,
    get_refiner_pipe,
    get_device,
    pipelines_loaded,
)

logger = logging.getLogger(__name__)

REFINER_HANDOFF_FRACTION = 0.25   # 🔥 slightly higher → better faces


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device=get_device()).manual_seed(seed)


def _snap_to_multiple(v: int, m: int = 8) -> int:
    return (v // m) * m


def _postprocess_output(image: Image.Image) -> Image.Image:
    # 🔥 More natural enhancement (less aggressive)
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=90, threshold=3))
    image = ImageEnhance.Contrast(image).enhance(1.03)
    return image


def _preprocess_init_image(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


# ─────────────────────────────────────────
# txt2img
# ─────────────────────────────────────────

def _run_txt2img(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    use_refiner: bool,
    refiner_steps: int,
) -> Image.Image:

    base = get_base_pipe()
    refiner = get_refiner_pipe()

    width = _snap_to_multiple(width)
    height = _snap_to_multiple(height)

    gen = _make_generator(seed)

    # 🔥 autocast improves performance + VRAM
    autocast_ctx = torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()

    if use_refiner and refiner is not None:

        denoise_split = 1.0 - REFINER_HANDOFF_FRACTION

        with torch.inference_mode(), autocast_ctx:
            latents = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
                denoising_end=denoise_split,
                output_type="latent",
            ).images

        gen_refiner = _make_generator(seed)

        with torch.inference_mode(), autocast_ctx:
            result = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=latents,
                num_inference_steps=refiner_steps,
                denoising_start=denoise_split,
                guidance_scale=guidance_scale * 0.9,   # 🔥 stabilize refinement
                generator=gen_refiner,
            )

        image = result.images[0]

    else:
        with torch.inference_mode(), autocast_ctx:
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


# ─────────────────────────────────────────
# img2img
# ─────────────────────────────────────────

def _run_img2img(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    strength: float,
    seed: int,
    use_refiner: bool,
    refiner_steps: int,
) -> Image.Image:

    pipe = get_img2img_pipe()
    refiner = get_refiner_pipe()

    width = _snap_to_multiple(width)
    height = _snap_to_multiple(height)

    init_image = _preprocess_init_image(init_image, width, height)
    gen = _make_generator(seed)

    autocast_ctx = torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()

    with torch.inference_mode(), autocast_ctx:
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

    # 🔥 Better refiner pass (more realistic humans)
    if use_refiner and refiner is not None:
        gen_refiner = _make_generator(seed)

        with torch.inference_mode(), autocast_ctx:
            result = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=0.25,   # 🔥 lower → preserve identity better
                num_inference_steps=refiner_steps,
                guidance_scale=guidance_scale * 0.85,
                generator=gen_refiner,
            )

        image = result.images[0]

    return _postprocess_output(image)


# ─────────────────────────────────────────
# Async wrappers
# ─────────────────────────────────────────

async def generate_txt2img(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    use_refiner: bool = True,
    refiner_steps: int = 20,
):
    if not pipelines_loaded():
        raise RuntimeError("Pipelines not loaded.")

    loop = asyncio.get_event_loop()
    start = time.time()

    image = await loop.run_in_executor(
        None,
        _run_txt2img,
        prompt, negative_prompt, steps, guidance_scale,
        width, height, seed, use_refiner, refiner_steps,
    )

    return image, round(time.time() - start, 2)


async def generate_img2img(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    strength: float,
    seed: int,
    use_refiner: bool = True,
    refiner_steps: int = 20,
):
    if not pipelines_loaded():
        raise RuntimeError("Pipelines not loaded.")

    loop = asyncio.get_event_loop()
    start = time.time()

    image = await loop.run_in_executor(
        None,
        _run_img2img,
        init_image, prompt, negative_prompt, steps, guidance_scale,
        width, height, strength, seed, use_refiner, refiner_steps,
    )

    return image, round(time.time() - start, 2)
