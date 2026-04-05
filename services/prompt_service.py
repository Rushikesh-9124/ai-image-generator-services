# (FULL FILE — REPLACE ENTIRE FILE)

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# QUALITY TOKENS (BALANCED)
# ─────────────────────────────────────────

BASE_QUALITY_TOKENS = [
    "photorealistic",
    "high detail",
    "sharp focus",
    "natural lighting",
    "professional photography",
]

# 🔥 IMPROVED HUMAN TOKENS (VERY IMPORTANT)
HUMAN_QUALITY_TOKENS = [
    "realistic skin texture",
    "natural skin pores",
    "detailed face",
    "symmetrical facial features",
    "lifelike eyes",
    "natural eye reflections",
    "sharp eyes focus",
    "natural hair strands",
    "fine skin details",
    "real person",
]

# 🔥 STRONG NEGATIVE PROMPT (SDXL optimized)
BASE_NEGATIVE_TOKENS = [
    "blurry", "low quality", "low resolution",
    "jpeg artifacts", "noise", "grain",
    "pixelated", "watermark", "text",
    "deformed", "distorted", "disfigured",
    "bad anatomy", "extra limbs", "missing limbs",
    "extra fingers", "missing fingers", "mutated hands",
    "poorly drawn face", "duplicate face",
    "multiple heads", "long neck",
    "cartoon", "anime", "painting", "drawing",
    "cgi", "unrealistic",
]

# 🔥 HUMAN NEGATIVE BOOST
HUMAN_NEGATIVE_TOKENS = [
    "bad face", "distorted face",
    "unnatural face", "asymmetrical face",
    "cross-eyed", "lazy eye",
    "plastic skin", "doll face",
    "overexposed skin", "underexposed face",
]


# ─────────────────────────────────────────
# STYLES (TUNED CFG)
# ─────────────────────────────────────────

STYLE_TEMPLATES = {
    "photorealistic": {
        "positive": [
            "DSLR photo",
            "85mm lens",
            "shallow depth of field",
            "bokeh background",
        ],
        "negative": [],
        "steps": 35,
        "guidance_scale": 6.5,   # 🔥 FIXED (better than 7.5)
    },
    "cinematic": {
        "positive": [
            "cinematic lighting",
            "movie still",
            "dramatic shadows",
        ],
        "negative": [],
        "steps": 38,
        "guidance_scale": 7.0,
    },
    "studio_portrait": {
        "positive": [
            "studio lighting",
            "softbox lighting",
            "professional portrait",
        ],
        "negative": [],
        "steps": 35,
        "guidance_scale": 6.5,
    },
}


# ─────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────

_HUMAN_KEYWORDS = re.compile(
    r"\b(person|woman|man|girl|boy|face|portrait|model|people|human)\b",
    re.IGNORECASE,
)


def _is_human(prompt: str) -> bool:
    return bool(_HUMAN_KEYWORDS.search(prompt))


def _dedupe(tokens: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in tokens:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


# ─────────────────────────────────────────
# OUTPUT STRUCT
# ─────────────────────────────────────────

@dataclass
class EnhancedPrompt:
    positive: str
    negative: str
    style_applied: Optional[str]
    is_human_subject: bool
    recommended_steps: int = 35
    recommended_cfg: float = 6.5


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────

def enhance_prompt(
    raw_prompt: str,
    style: Optional[str] = None,
    raw_negative_prompt: str = "",
    force_human_tokens: bool = False,
) -> EnhancedPrompt:

    raw_prompt = raw_prompt.strip()

    is_human = force_human_tokens or _is_human(raw_prompt)

    # ── Positive ─────────────────────────
    positive_parts = [raw_prompt]
    positive_parts += BASE_QUALITY_TOKENS

    if is_human:
        positive_parts += HUMAN_QUALITY_TOKENS

    recommended_steps = 35
    recommended_cfg = 6.5
    style_applied = None

    if style and style in STYLE_TEMPLATES:
        tmpl = STYLE_TEMPLATES[style]
        positive_parts += tmpl["positive"]
        recommended_steps = tmpl["steps"]
        recommended_cfg = tmpl["guidance_scale"]
        style_applied = style

    # ── Negative ─────────────────────────
    negative_parts = list(BASE_NEGATIVE_TOKENS)

    if is_human:
        negative_parts += HUMAN_NEGATIVE_TOKENS

    if raw_negative_prompt.strip():
        negative_parts += [x.strip() for x in raw_negative_prompt.split(",")]

    positive = ", ".join(_dedupe(positive_parts))
    negative = ", ".join(_dedupe(negative_parts))

    logger.debug(f"Enhanced prompt | human={is_human} style={style_applied}")

    return EnhancedPrompt(
        positive=positive,
        negative=negative,
        style_applied=style_applied,
        is_human_subject=is_human,
        recommended_steps=recommended_steps,
        recommended_cfg=recommended_cfg,
    )


# ─────────────────────────────────────────
# STYLES LIST
# ─────────────────────────────────────────

def list_styles():
    return [
        {
            "id": k,
            "name": k.replace("_", " ").title(),
            "steps": v["steps"],
            "guidance_scale": v["guidance_scale"],
        }
        for k, v in STYLE_TEMPLATES.items()
    ]


# ─────────────────────────────────────────
# NEGATIVE BUILDER
# ─────────────────────────────────────────

def build_negative_prompt(user_negative: str = "", is_human: bool = False) -> str:
    parts = list(BASE_NEGATIVE_TOKENS)
    if is_human:
        parts += HUMAN_NEGATIVE_TOKENS
    if user_negative.strip():
        parts += [x.strip() for x in user_negative.split(",")]
    return ", ".join(_dedupe(parts))
