# transformers_provider.py
"""Provider wrapper that connects *send_request.py* to the local HuggingFace
Transformers backend.

This module lives inside the *comfy-nodes* package so it gets picked up by the
ComfyUI auto-import logic just like the other provider modules (e.g.
`openai_provider.py`).  The public coroutine `send_transformers_request` mirrors
other provider-specific functions such as `openai_api.send_openai_request` so
that *send_request.py* can treat it transparently.
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging
from pathlib import Path

import folder_paths  # Provided by ComfyUI runtime

# The heavy lifting (tokenisation, model loading, etc.) is delegated to
# *transformers_api.py* located at the repository root.  We simply re-export its
# main coroutine so callers can `from transformers_provider import
# send_transformers_request`.
from transformers_api import send_transformers_request  # type: ignore
from transformers_api import send_transformers_request as _hf_send_transformers_request  # type: ignore
from transformers_api import send_transformers_request_stream as _hf_send_transformers_request_stream  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model discovery helpers – surf *models/transformers* and *models/LLM*
# -----------------------------------------------------------------------------

def _discover_models() -> List[str]:
    """Return a list of local transformer model directories (relative names)."""

    model_parent_dirs: List[Path] = []

    # Explicitly look inside the common folders used for HF style checkpoints.
    # Users often place models under either  `models/transformers/<repo>` or
    # `models/LLM/<repo>` so we iterate over both when they exist.
    for fld in ("transformers", "LLM"):
        try:
            for p in folder_paths.get_folder_paths(fld):
                model_parent_dirs.append(Path(p))
        except Exception:
            # The folder name might not be registered in `folder_paths` – that is
            # fine, we will fall back to the generic `models` directory anyway.
            continue

    # Fallback: root *models* dir itself (ComfyUI/models)
    model_parent_dirs.append(Path(folder_paths.models_dir))

    seen: set[str] = set()
    out: List[str] = []
    for parent in model_parent_dirs:
        if not parent.exists():
            continue
        # We inspect *one* level below the parent because HF repos are usually
        # stored as `<parent>/<repo-name>/*` with the `config.json` at the root
        # of the repo directory.
        for child in parent.iterdir():
            if (child / "config.json").exists():
                if child.name not in seen:
                    out.append(child.as_posix())
                    seen.add(child.name)
            # Some users keep an extra nesting level, e.g. `models/LLM/<group>/<repo>`.
            # Check one more level deep for such cases.
            elif child.is_dir():
                for grand in child.iterdir():
                    if (grand / "config.json").exists() and grand.name not in seen:
                        out.append(grand.as_posix())
                        seen.add(grand.name)
    return sorted(out)


# -----------------------------------------------------------------------------
# PUBLIC: async send_transformers_request – thin wrapper with logging
# -----------------------------------------------------------------------------

async def send_transformers_request_provider(
    *,
    base64_images: List[str],
    base64_audio: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]] | None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    precision: str = "fp16",
    **kwargs,
):
    """Thin wrapper that forwards to *transformers_api* but adds helpful logging."""
    logger.info("[Transformers-Provider] model=%s | precision=%s", model, precision)
    return await _hf_send_transformers_request(
        base64_images=base64_images,
        base64_audio=base64_audio,
        model=model,
        system_message=system_message,
        user_message=user_message,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        precision=precision,
    )


# Expose the coroutine under the canonical name so callers can simply do:
# `from transformers_provider import send_transformers_request`
send_transformers_request = send_transformers_request_provider 


# -----------------------------------------------------------------------------
# Streaming wrapper – exposes *send_transformers_request_stream* to callers
# -----------------------------------------------------------------------------

async def send_transformers_request_stream_provider(
    *,
    base64_images: List[str],
    base64_audio: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]] | None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    precision: str = "fp16",
    **kwargs,
):
    """Streaming variant that yields text chunks."""
    logger.info("[Transformers-Provider-Stream] model=%s | precision=%s", model, precision)
    async for chunk in _hf_send_transformers_request_stream(
        base64_images=base64_images,
        base64_audio=base64_audio,
        model=model,
        system_message=system_message,
        user_message=user_message,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        precision=precision,
    ):
        yield chunk


# Convenience alias
send_transformers_request_stream = send_transformers_request_stream_provider


# -----------------------------------------------------------------------------
# ComfyUI Node: Local Transformers Provider
# -----------------------------------------------------------------------------

try:
    from server import PromptServer  # Optional – only available inside ComfyUI runtime
    from aiohttp import web

    @PromptServer.instance.routes.get("/ComfyLLMToolkit/get_transformer_models")
    async def get_transformer_models_endpoint(request):
        """Frontend helper endpoint that returns the list of discovered local HF models."""
        try:
            models = _discover_models()
            if not models:
                models = ["No local models found"]
            return web.json_response(models)
        except Exception as e:
            logger.error("Error in get_transformer_models_endpoint: %s", e, exc_info=True)
            return web.json_response(["Error fetching models"], status=500)

    logger.info("TransformersProviderNode: /ComfyLLMToolkit/get_transformer_models endpoint registered")
except (ImportError, AttributeError):
    # Running outside the ComfyUI webserver; skip endpoint registration.
    pass


class LocalTransformersProviderNode:
    """Node that lets the user pick a local HF model and outputs provider_config."""

    PROVIDER_NAME = "transformers"

    @classmethod
    def INPUT_TYPES(cls):
        models = _discover_models()
        if not models:
            models = ["No local models found"]
        default_model = models[0] if models else ""
        return {
            "required": {
                "llm_model": (
                    models,
                    {
                        "default": default_model,
                        "tooltip": "Select a local HuggingFace model directory.",
                    },
                ),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_transformers"
    CATEGORY = "llm_toolkit"

    @classmethod
    def IS_CHANGED(cls, llm_model: str, context=None):
        # Recompute if model path changes
        return llm_model

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def configure_transformers(self, llm_model: str, context=None):
        logger.info("LocalTransformersProviderNode: configuring model '%s'", llm_model)

        # Build provider configuration dictionary
        provider_config = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip(),
            "api_key": "1234",  # placeholder so downstream nodes treat as local provider (no key).
            "base_ip": None,
            "port": None,
        }

        # Merge with incoming context to retain pipeline data
        if context is not None and isinstance(context, dict):
            context["provider_config"] = provider_config
            output = context
        else:
            output = {"provider_config": provider_config}

        return (output,)


# Explicit node registration (optional, auto-discovery already handled by __init__)
NODE_CLASS_MAPPINGS = {
    "LocalTransformersProviderNode": LocalTransformersProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalTransformersProviderNode": "Local Transformers Provider (LLMToolkit)",
} 