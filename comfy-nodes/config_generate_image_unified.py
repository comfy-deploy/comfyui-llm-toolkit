"""Unified Configure Image Generation node leveraging capability mappings."""

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from image_generation_capabilities import (
    DEFAULT_MODEL,
    MANAGED_KEYS,
    normalize_generation_config,
    resolve_canonical_model,
)

logger = logging.getLogger(__name__)

SIZE_CHOICES = [
    "auto",
    "256x256",
    "512x512",
    "640x832",
    "720x1280",
    "768x1024",
    "832x640",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1792x1024",
    "1024x1792",
    "2048x2048",
    "2048x3072",
    "3072x2048",
]
ASPECT_RATIO_CHOICES = [
    "auto",
    "1:1",
    "3:4",
    "4:3",
    "2:3",
    "3:2",
    "9:16",
    "16:9",
    "21:9",
    "9:21",
]


def _automatic_defaults_for_model(canonical_model: str) -> Dict[str, Any]:
    """Return automatic parameter defaults tailored to the resolved model."""

    lowered = (canonical_model or "").lower()
    defaults: Dict[str, Any] = {}

    if lowered in {"openai/dall-e-2", "openai/dall-e-3"}:
        defaults["response_format"] = "auto"

    if lowered == "openai/dall-e-3":
        defaults.update({
            "quality": "hd",
            "style": "auto",
        })

    if lowered == "openai/gpt-image-1":
        defaults.update({
            "quality": "high",
            "background": "auto",
            "output_format": "auto",
            "output_compression": 100,
            "moderation": "low",
        })

    if lowered in {
        "openai/gpt-image-1",
        "wavespeed/hunyuan-image-3",
        "wavespeed/qwen-image-edit-plus",
        "bfl/flux",
    }:
        defaults.setdefault("output_format", "auto")

    if lowered in {"google/gemini-image", "google/imagen"}:
        defaults.update({
            "person_policy": "allow_all",
            "safety_setting": "relaxed",
            "language_hint": "auto",
        })

    if lowered == "bfl/flux":
        defaults["safety_setting"] = "relaxed"

    if lowered in {"wavespeed/imagen4", "wavespeed/imagen4-ultra"}:
        defaults.setdefault("size", "2048x2048")
    elif lowered == "wavespeed/imagen4-fast":
        defaults.setdefault("size", "1024x1024")

    if lowered == "wavespeed/dreamina-v3.1":
        defaults.setdefault("size", "1328x1328")

    if lowered == "wavespeed/nano-banana-edit":
        defaults.setdefault("output_format", "png")

    if lowered in {
        "google/gemini-image",
        "google/imagen",
    }:
        defaults.setdefault("temperature", 0.7)
        defaults.setdefault("max_tokens", 512)

    if lowered == "bfl/flux":
        defaults.setdefault("safety_tolerance", 6)

    return defaults


class ConfigGenerateImageUnified:
    """Collect image generation parameters for supported providers/models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "size": (SIZE_CHOICES, {"default": "auto"}),
                "aspect_ratio": (ASPECT_RATIO_CHOICES, {"default": "auto"}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": 28, "min": 1, "max": 50, "step": 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "prompt_enhancement": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
                "user_tag": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config"

    def configure(
        self,
        context: Optional[Any] = None,
        image_count: int = 1,
        size: str = "auto",
        aspect_ratio: str = "auto",
        guidance_scale: float = 2.5,
        inference_steps: int = 28,
        enable_safety_checker: bool = True,
        prompt_enhancement: bool = False,
        seed: int = -1,
        user_tag: str = "",
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageUnified executing.")

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
            or gen_cfg.get("model")
            or DEFAULT_MODEL
        )
        canonical_model = resolve_canonical_model(model_hint)

        requested_values: Dict[str, Any] = {
            "provider": provider_cfg.get("provider_name", ""),
            "image_count": image_count,
            "size": size,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "enable_safety_checker": enable_safety_checker,
            "prompt_enhancement": prompt_enhancement,
            "seed": seed,
            "user_tag": user_tag,
        }

        requested_values.update(_automatic_defaults_for_model(canonical_model))

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
            "ConfigGenerateImageUnified: model=%s payload=%s",
            canonical_model,
            sanitized_payload,
        )
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {"ConfigGenerateImageUnified": ConfigGenerateImageUnified}
NODE_DISPLAY_NAME_MAPPINGS = {"ConfigGenerateImageUnified": "Configure Image Generation (🔗llm_toolkit)"}
