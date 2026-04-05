# (FULL FILE — REPLACE ENTIRE FILE)

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


# ─────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────
@dataclass
class GenerationResult:
    image_base64: str
    seed: int
    generation_time: float
    width: int
    height: int
    enhanced_prompt: str
    negative_prompt: str
    mode: str


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

DEFAULT_NEGATIVE = (
    "blurry, deformed face, bad anatomy, extra fingers, distorted eyes, "
    "low quality, worst quality, jpeg artifacts, cartoon, painting"
)


def _resolve_seed(seed: Optional[int]) -> int:
    return int(seed) if seed is not None else int(torch.randint(0, 2**32, (1,)).item())


def _fix_negative(neg: str) -> str:
    if not neg or len(neg.strip()) < 5:
        return DEFAULT_NEGATIVE
    return neg


def _fix_cfg(cfg: float) -> float:
    # SDXL sweet spot
    if cfg > 8:
        return 7.0
    if cfg < 4:
        return 5.0
    return cfg


def _cleanup_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_image(b64_string: Optional[str]) -> Optional[Image.Image]:
    if not b64_string or not b64_string.strip():
        return None
    try:
        data = b64_string.strip()
        if "," in data:
            data = data.split(",", 1)[1]
        missing = len(data) % 4
        if missing:
            data += "=" * (4 - missing)
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning(f"Decode failed: {exc}")
        return None


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=6)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────
# txt2img
# ─────────────────────────────────────────────────────
async def run_txt2img(
    prompt: str,
    negative_prompt: str = "",
    style: Optional[str] = None,
    steps: int = 35,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    use_refiner: bool = True,
    refiner_steps: int = 20,
    auto_enhance_prompt: bool = True,
) -> GenerationResult:

    seed = _resolve_seed(seed)

    # 🔥 FIX INPUT QUALITY
    negative_prompt = _fix_negative(negative_prompt)
    guidance_scale = _fix_cfg(guidance_scale)

    # 🔥 PROMPT ENHANCEMENT (SAFE)
    if auto_enhance_prompt:
        enhanced: EnhancedPrompt = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
        )
        final_positive = enhanced.positive
        final_negative = enhanced.negative

        if steps == 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 7.5:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_positive = prompt
        final_negative = negative_prompt

    logger.info(
        f"[txt2img] seed={seed} steps={steps} cfg={guidance_scale} "
        f"{width}x{height} refiner={use_refiner}"
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

    _cleanup_vram()

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


# ─────────────────────────────────────────────────────
# img2img
# ─────────────────────────────────────────────────────
async def run_img2img(
    init_image_b64: str,
    prompt: str,
    negative_prompt: str = "",
    style: Optional[str] = None,
    steps: int = 35,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.55,
    seed: Optional[int] = None,
    use_refiner: bool = True,
    refiner_steps: int = 20,
    auto_enhance_prompt: bool = True,
) -> GenerationResult:

    init_image = _decode_image(init_image_b64)
    if init_image is None:
        raise ValueError("Invalid base64 image")

    seed = _resolve_seed(seed)

    # 🔥 FIX INPUT QUALITY
    negative_prompt = _fix_negative(negative_prompt)
    guidance_scale = _fix_cfg(guidance_scale)

    if auto_enhance_prompt:
        enhanced = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
        )
        final_positive = enhanced.positive
        final_negative = enhanced.negative

        if steps == 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 7.5:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_positive = prompt
        final_negative = negative_prompt

    logger.info(
        f"[img2img] seed={seed} strength={strength} steps={steps} "
        f"cfg={guidance_scale} {width}x{height} refiner={use_refiner}"
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

    _cleanup_vram()

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
