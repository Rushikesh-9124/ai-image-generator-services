"""
services/image_service.py
==========================
Orchestration layer that sits between API routes and the model layer.

Responsibilities:
  - Decode / validate base64 init images
  - Auto-enhance prompts via prompt_service
  - Dispatch to txt2img or img2img inference
  - Encode PIL output to base64 PNG
  - Seed generation (random or user-supplied)
  - Structured result packaging
"""

import io
import base64
import logging
from typing import Optional
from dataclasses import dataclass

import torch
from PIL import Image

from model.inference import generate_txt2img, generate_img2img
from services.prompt_service import enhance_prompt, EnhancedPrompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    image_base64:     str
    seed:             int
    generation_time:  float
    width:            int
    height:           int
    enhanced_prompt:  str
    negative_prompt:  str
    mode:             str      # "txt2img" | "img2img"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed)
    return int(torch.randint(0, 2 ** 32, (1,)).item())


def _decode_image(b64_string: Optional[str]) -> Optional[Image.Image]:
    """
    Safely decode a base64-encoded image string → PIL Image.
    Returns None if the input is empty or malformed.
    Fixes missing padding automatically.
    """
    if not b64_string or not b64_string.strip():
        return None
    try:
        data = b64_string.strip()
        # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in data:
            data = data.split(",", 1)[1]
        # Fix padding
        missing = len(data) % 4
        if missing:
            data += "=" * (4 - missing)
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning(f"Failed to decode init image: {exc}")
        return None


def _encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=6)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Public service methods
# ─────────────────────────────────────────────────────────────────────────────

async def run_txt2img(
    prompt:               str,
    negative_prompt:      str       = "",
    style:                Optional[str] = None,
    steps:                int       = 35,
    guidance_scale:       float     = 7.5,
    width:                int       = 1024,
    height:               int       = 1024,
    seed:                 Optional[int] = None,
    use_refiner:          bool      = True,
    refiner_steps:        int       = 20,
    auto_enhance_prompt:  bool      = True,
) -> GenerationResult:
    """
    Full txt2img pipeline:
      1. Optionally enhance the prompt
      2. Resolve seed
      3. Run SDXL inference (+ optional refiner)
      4. Encode result and return structured response
    """
    seed = _resolve_seed(seed)

    if auto_enhance_prompt:
        enhanced: EnhancedPrompt = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
        )
        final_positive = enhanced.positive
        final_negative = enhanced.negative
        # Respect style-recommended parameters if caller used defaults
        if steps == 35 and enhanced.recommended_steps != 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 7.5 and enhanced.recommended_cfg != 7.5:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_positive = prompt
        final_negative = negative_prompt

    logger.info(
        f"[txt2img] seed={seed} steps={steps} cfg={guidance_scale} "
        f"{width}×{height} refiner={use_refiner}"
    )

    image, elapsed = await generate_txt2img(
        prompt=final_positive,
        negative_prompt=final_negative,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        use_refiner=use_refiner,
        refiner_steps=refiner_steps,
    )

    logger.info(f"[txt2img] done in {elapsed}s")

    return GenerationResult(
        image_base64=_encode_image(image),
        seed=seed,
        generation_time=elapsed,
        width=image.width,
        height=image.height,
        enhanced_prompt=final_positive,
        negative_prompt=final_negative,
        mode="txt2img",
    )


async def run_img2img(
    init_image_b64:       str,
    prompt:               str,
    negative_prompt:      str       = "",
    style:                Optional[str] = None,
    steps:                int       = 35,
    guidance_scale:       float     = 7.5,
    width:                int       = 1024,
    height:               int       = 1024,
    strength:             float     = 0.55,
    seed:                 Optional[int] = None,
    use_refiner:          bool      = True,
    refiner_steps:        int       = 20,
    auto_enhance_prompt:  bool      = True,
) -> GenerationResult:
    """
    Full img2img pipeline:
      1. Decode init image
      2. Optionally enhance prompt
      3. Resolve seed
      4. Run SDXL img2img (+ optional refiner)
      5. Encode and return result

    strength guide:
      0.30 – 0.45 → subtle edits (lighting, texture, colour)
      0.45 – 0.65 → moderate changes (style transfer, outfit)
      0.65 – 0.85 → heavy reinterpretation (background, mood)
      0.85 – 1.00 → near txt2img (ignores original)
    """
    init_image = _decode_image(init_image_b64)
    if init_image is None:
        raise ValueError(
            "init_image_base64 is missing or invalid. "
            "Provide a valid base64-encoded PNG/JPEG for img2img."
        )

    seed = _resolve_seed(seed)

    if auto_enhance_prompt:
        enhanced = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
        )
        final_positive = enhanced.positive
        final_negative = enhanced.negative
        if steps == 35 and enhanced.recommended_steps != 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 7.5 and enhanced.recommended_cfg != 7.5:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_positive = prompt
        final_negative = negative_prompt

    logger.info(
        f"[img2img] seed={seed} strength={strength} steps={steps} "
        f"cfg={guidance_scale} {width}×{height} refiner={use_refiner}"
    )

    image, elapsed = await generate_img2img(
        init_image=init_image,
        prompt=final_positive,
        negative_prompt=final_negative,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        strength=strength,
        seed=seed,
        use_refiner=use_refiner,
        refiner_steps=refiner_steps,
    )

    logger.info(f"[img2img] done in {elapsed}s")

    return GenerationResult(
        image_base64=_encode_image(image),
        seed=seed,
        generation_time=elapsed,
        width=image.width,
        height=image.height,
        enhanced_prompt=final_positive,
        negative_prompt=final_negative,
        mode="img2img",
    )