"""Unified Configure Video Generation node for LLM Toolkit."""

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from video_generation_capabilities import (
    DEFAULT_MODEL,
    MANAGED_KEYS,
    normalize_generation_config,
    resolve_canonical_model,
)

logger = logging.getLogger(__name__)

ASPECT_CHOICES = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]
RESOLUTION_CHOICES = ["720p", "1080p"]
DURATION_CHOICES = ["5", "8", "10"]
DEFAULT_RESOLUTION = RESOLUTION_CHOICES[-1]
DEFAULT_DURATION_SECONDS = int(DURATION_CHOICES[1])
NEGATIVE_PROMPT_DEFAULT = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy"


class ConfigGenerateVideo:
    """Collect video generation parameters for supported providers/models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "aspect_ratio": (ASPECT_CHOICES, {"default": "16:9"}),
                "video_resolution": (RESOLUTION_CHOICES, {"default": DEFAULT_RESOLUTION}),
                "duration_seconds": (DURATION_CHOICES, {"default": str(DEFAULT_DURATION_SECONDS)}),
                "guidance_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "negative_prompt": ("STRING", {"default": NEGATIVE_PROMPT_DEFAULT, "multiline": True}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config"

    def configure(
        self,
        context: Optional[Any] = None,
        aspect_ratio: str = "16:9",
        video_resolution: str = DEFAULT_RESOLUTION,
        duration_seconds: str = str(DEFAULT_DURATION_SECONDS),
        guidance_scale: float = 0.5,
        negative_prompt: str = NEGATIVE_PROMPT_DEFAULT,
        enhance_prompt: bool = True,
        enable_prompt_expansion: bool = True,
        generate_audio: bool = True,
        seed: int = -1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideo executing.")

        if context is None:
            out_ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            out_ctx = context.copy()
        else:
            out_ctx = extract_context(context)
            if not isinstance(out_ctx, dict):
                out_ctx = {"passthrough_data": context}

        provider_cfg = out_ctx.get("provider_config", {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}

        gen_cfg = out_ctx.get("generation_config", {})
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}

        model_hint = (
            provider_cfg.get("llm_model")
            or gen_cfg.get("model_id")
            or DEFAULT_MODEL
        )
        canonical_model = resolve_canonical_model(model_hint)

        try:
            duration_value = int(duration_seconds)
        except (TypeError, ValueError):
            duration_value = DEFAULT_DURATION_SECONDS

        requested_values: Dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "video_resolution": video_resolution,
            "duration_seconds": duration_value,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "enhance_prompt": enhance_prompt,
            "enable_prompt_expansion": enable_prompt_expansion,
            "generate_audio": generate_audio,
            "seed": seed,
        }

        _, sanitized_payload = normalize_generation_config(
            canonical_model,
            requested=requested_values,
            existing=gen_cfg,
        )

        for key in list(gen_cfg.keys()):
            if key in MANAGED_KEYS and key not in sanitized_payload:
                gen_cfg.pop(key)

        gen_cfg.update(sanitized_payload)
        out_ctx["generation_config"] = gen_cfg

        logger.info(
            "ConfigGenerateVideo: model=%s payload=%s",
            canonical_model,
            sanitized_payload,
        )
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {"ConfigGenerateVideo": ConfigGenerateVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"ConfigGenerateVideo": "Configure Video Generation (🔗LLMToolkit)"}
