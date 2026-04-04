"""
services/prompt_service.py
===========================
Prompt engineering layer.

Responsibilities:
  - Automatic prompt enrichment (adds photorealism keywords)
  - Style template library (photorealistic, cinematic, editorial, etc.)
  - Negative prompt builder with sensible defaults
  - Prompt validation and sanitisation
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Quality keyword banks
# ─────────────────────────────────────────────────────────────────────────────

# Always appended to every positive prompt
BASE_QUALITY_TOKENS = [
    "photorealistic",
    "ultra detailed",
    "8k uhd",
    "sharp focus",
    "high dynamic range",
    "professional photography",
]

# Appended when the subject is a human / portrait
HUMAN_QUALITY_TOKENS = [
    "realistic skin texture",
    "natural skin pores",
    "detailed face",
    "accurate anatomy",
    "real person",
    "lifelike eyes",
    "natural hair strands",
]

# Core negative prompt — applied to every generation
BASE_NEGATIVE_TOKENS = [
    "blurry", "out of focus", "low quality", "low resolution",
    "jpeg artifacts", "compression artifacts", "noise", "grain",
    "pixelated", "watermark", "signature", "logo", "text",
    "deformed", "distorted", "disfigured",
    "bad anatomy", "wrong anatomy", "extra limbs", "missing limbs",
    "extra fingers", "missing fingers", "fused fingers", "mutated hands",
    "poorly drawn hands", "poorly drawn face",
    "clone face", "duplicate", "multiple heads",
    "cartoon", "anime", "illustration", "painting", "drawing", "sketch",
    "3d render", "cgi", "unrealistic", "oversaturated",
    "ugly", "gross", "disgusting",
]

# Extra negative tokens for human portraits
HUMAN_NEGATIVE_TOKENS = [
    "bad face", "distorted face", "unnatural face",
    "plastic skin", "doll-like", "mannequin",
    "asymmetric eyes", "cross-eyed", "wall-eyed",
    "unnatural skin color", "orange skin", "grey skin",
    "makeup clown", "over-retouched",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Style templates
# ─────────────────────────────────────────────────────────────────────────────

STYLE_TEMPLATES: dict[str, dict] = {
    "photorealistic": {
        "positive": [
            "DSLR photograph", "50mm lens", "f/1.8 aperture",
            "natural lighting", "cinematic lighting", "soft rim light",
            "shallow depth of field", "bokeh background",
        ],
        "negative": [],
        "steps": 35,
        "guidance_scale": 7.5,
    },
    "cinematic": {
        "positive": [
            "cinematic shot", "movie still", "35mm film", "anamorphic lens",
            "dramatic lighting", "chiaroscuro", "deep shadows", "lens flare",
            "film grain (subtle)", "color grading",
        ],
        "negative": ["amateur", "snapshot"],
        "steps": 40,
        "guidance_scale": 8.0,
    },
    "studio_portrait": {
        "positive": [
            "professional studio portrait", "softbox lighting",
            "three-point lighting setup", "white seamless background",
            "85mm portrait lens", "catchlights in eyes", "editorial quality",
        ],
        "negative": ["harsh shadows", "red-eye"],
        "steps": 35,
        "guidance_scale": 7.5,
    },
    "editorial": {
        "positive": [
            "editorial fashion photography", "magazine cover quality",
            "high-key lighting", "professional model", "vogue aesthetic",
            "stylized composition", "striking pose",
        ],
        "negative": ["casual", "snapshot", "amateur"],
        "steps": 38,
        "guidance_scale": 8.0,
    },
    "street_photography": {
        "positive": [
            "candid street photography", "natural ambient light",
            "urban environment", "35mm prime lens", "documentary style",
            "authentic moment", "photojournalism",
        ],
        "negative": ["posed", "studio"],
        "steps": 30,
        "guidance_scale": 7.0,
    },
    "oil_painting": {
        "positive": [
            "oil painting", "impressionist brushwork", "textured canvas",
            "old masters style", "rich colour palette", "painterly",
        ],
        "negative": ["photo", "photograph", "realistic"],
        "steps": 35,
        "guidance_scale": 8.5,
    },
    "cyberpunk": {
        "positive": [
            "cyberpunk aesthetic", "neon lights", "dystopian future",
            "rain-slicked streets", "holographic displays",
            "techno-noir atmosphere", "Blade Runner inspired",
        ],
        "negative": ["bright daylight", "natural", "pastoral"],
        "steps": 35,
        "guidance_scale": 8.0,
    },
    "fantasy": {
        "positive": [
            "epic fantasy setting", "magical atmosphere", "mystical lighting",
            "otherworldly environment", "concept art quality",
            "highly detailed environment", "dramatic sky",
        ],
        "negative": ["mundane", "modern", "contemporary"],
        "steps": 40,
        "guidance_scale": 8.5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_HUMAN_KEYWORDS = re.compile(
    r"\b(person|woman|man|girl|boy|human|face|portrait|model|people|"
    r"female|male|lady|gentleman|child|baby|teenager|adult|elderly|"
    r"celebrity|actor|actress|athlete|student|professional)\b",
    re.IGNORECASE,
)


def _is_human_subject(prompt: str) -> bool:
    return bool(_HUMAN_KEYWORDS.search(prompt))


def _deduplicate_tokens(tokens: list[str]) -> list[str]:
    """Preserve order while removing case-insensitive duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(token.strip())
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancedPrompt:
    positive:        str
    negative:        str
    style_applied:   Optional[str]
    is_human_subject: bool
    recommended_steps: int          = 35
    recommended_cfg:   float        = 7.5


def enhance_prompt(
    raw_prompt:          str,
    style:               Optional[str] = None,
    raw_negative_prompt: str = "",
    force_human_tokens:  bool = False,
) -> EnhancedPrompt:
    """
    Enrich a user prompt with quality and style tokens.

    Args:
        raw_prompt:           The user's original positive prompt
        style:                Optional style key from STYLE_TEMPLATES
        raw_negative_prompt:  User-supplied negative prompt (merged with defaults)
        force_human_tokens:   Add human-specific quality tokens even if not detected

    Returns:
        EnhancedPrompt dataclass with enriched positive / negative strings
    """
    raw_prompt = raw_prompt.strip()
    is_human   = force_human_tokens or _is_human_subject(raw_prompt)

    # ── Positive tokens ───────────────────────────────────────────────────────
    positive_parts = [raw_prompt]
    positive_parts.extend(BASE_QUALITY_TOKENS)
    if is_human:
        positive_parts.extend(HUMAN_QUALITY_TOKENS)

    recommended_steps = 35
    recommended_cfg   = 7.5
    style_applied     = None

    if style and style in STYLE_TEMPLATES:
        template       = STYLE_TEMPLATES[style]
        positive_parts.extend(template["positive"])
        recommended_steps = template.get("steps", recommended_steps)
        recommended_cfg   = template.get("guidance_scale", recommended_cfg)
        style_applied     = style

    # ── Negative tokens ───────────────────────────────────────────────────────
    negative_parts = list(BASE_NEGATIVE_TOKENS)
    if is_human:
        negative_parts.extend(HUMAN_NEGATIVE_TOKENS)
    if style and style in STYLE_TEMPLATES:
        negative_parts.extend(STYLE_TEMPLATES[style].get("negative", []))

    # Merge user-supplied negative tokens
    if raw_negative_prompt.strip():
        user_neg = [t.strip() for t in raw_negative_prompt.split(",")]
        negative_parts.extend(user_neg)

    positive_str = ", ".join(_deduplicate_tokens(positive_parts))
    negative_str = ", ".join(_deduplicate_tokens(negative_parts))

    logger.debug(
        f"Prompt enhanced | human={is_human} | style={style_applied} "
        f"| pos_tokens={len(positive_parts)} | neg_tokens={len(negative_parts)}"
    )

    return EnhancedPrompt(
        positive=positive_str,
        negative=negative_str,
        style_applied=style_applied,
        is_human_subject=is_human,
        recommended_steps=recommended_steps,
        recommended_cfg=recommended_cfg,
    )


def list_styles() -> list[dict]:
    """Return all available style templates with metadata."""
    return [
        {
            "id":          key,
            "name":        key.replace("_", " ").title(),
            "steps":       tmpl.get("steps", 35),
            "guidance_scale": tmpl.get("guidance_scale", 7.5),
            "preview_keywords": tmpl["positive"][:3],
        }
        for key, tmpl in STYLE_TEMPLATES.items()
    ]


def build_negative_prompt(
    user_negative: str = "",
    is_human: bool     = False,
) -> str:
    """
    Build a final negative prompt string, useful when the caller
    already has an enhanced positive but wants a fresh negative.
    """
    parts = list(BASE_NEGATIVE_TOKENS)
    if is_human:
        parts.extend(HUMAN_NEGATIVE_TOKENS)
    if user_negative.strip():
        parts.extend([t.strip() for t in user_negative.split(",")])
    return ", ".join(_deduplicate_tokens(parts))