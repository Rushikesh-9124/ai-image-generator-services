"""
app.py
======
FastAPI application entrypoint.

Upgrades included:
  - GPU performance optimizations (TF32)
  - Multi-LoRA support
  - Default negative prompt (for better quality)
  - Model warmup (reduces first request latency)
  - Improved logging for debugging
"""

import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.pipeline_loader import (
    load_pipelines,
    unload_pipelines,
    load_lora_weights,
    get_txt2img_pipeline
)
from api.routes import router


# ─────────────────────────────────────────────────────────────────────────────
#  Performance Optimization (GPU)
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Env Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
#  Startup Configuration
# ─────────────────────────────────────────────────────────────────────────────
LOAD_REFINER    = _env_bool("LOAD_REFINER", True)
ENABLE_SAFETY   = _env_bool("ENABLE_SAFETY_CHECKER", False)
SCHEDULER       = os.getenv("SCHEDULER", "dpm++")

# 🔥 Multi-LoRA support
LORA_PATHS      = os.getenv("LORA_PATHS", "")  # comma-separated
LORA_SCALE      = float(os.getenv("LORA_SCALE", "0.8"))

# 🔥 Default Negative Prompt (global quality control)
DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "DEFAULT_NEGATIVE_PROMPT",
    "blurry, deformed face, bad anatomy, extra fingers, distorted eyes, "
    "low quality, cartoon, painting, worst quality, jpeg artifacts"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan (Startup / Shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  AI Image Generation Service — starting up")
    logger.info(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  Refiner    : {LOAD_REFINER}")
    logger.info(f"  Scheduler  : {SCHEDULER}")
    logger.info(f"  Safety     : {ENABLE_SAFETY}")
    logger.info(f"  LoRAs      : {LORA_PATHS if LORA_PATHS else 'None'}")
    logger.info("=" * 60)

    # Load pipelines
    load_pipelines(
        scheduler_name=SCHEDULER,
        load_refiner=LOAD_REFINER,
        enable_safety=ENABLE_SAFETY,
    )

    # 🔥 Load multiple LoRAs
    if LORA_PATHS:
        for path in LORA_PATHS.split(","):
            path = path.strip()
            if not path:
                continue
            try:
                logger.info(f"Loading LoRA: {path} (scale={LORA_SCALE})")
                load_lora_weights(lora_path=path, lora_scale=LORA_SCALE)
            except Exception as exc:
                logger.warning(f"LoRA load failed for {path}: {exc}")

    # 🔥 Warmup (important for first request latency)
    try:
        logger.info("Warming up model...")
        pipe = get_txt2img_pipeline()

        _ = pipe(
            prompt="warmup",
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=5,
            guidance_scale=1.0,
        )

        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    logger.info("Service ready — accepting requests.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing GPU memory ...")
    unload_pipelines()
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Image Generation Service",
    description=(
        "Production-grade SDXL-based image generation API. "
        "Supports txt2img, img2img, prompt enhancement, LoRA stacking, "
        "scheduler control, and high-quality photorealistic outputs."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────────────────────────────────────
#  CORS
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(router)
