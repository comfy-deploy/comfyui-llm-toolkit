# transformers_api.py
"""Utilities for loading & running local HuggingFace Transformer models in ComfyUI LLM-Toolkit.

The helper focuses on *offline* (local) inference and therefore does not require any Internet
connectivity once the model weights have been downloaded.  It provides a thin wrapper that mirrors
our other provider_* modules (openai_api, gemini_api, ollama_api …) so that *send_request* can call
it with a familiar signature.

Key design goals
----------------
• *Precision* – Load models in half-precision (float16 / bfloat16) by default to minimise VRAM
  footprint.  Users may override via the *precision* kwarg.
• *AWQ* – If the model repo or directory name contains the substring "AWQ" we will attempt to load
  it with the *autoawq* loader (if available).  Otherwise we fallback to standard HF loading.
• *Multi-modal* – Basic support for Alibaba Qwen2.5-Omni (text-image-audio) and Qwen-VL/VL-like
  models is included.  The caller can pass *base64_images* (List[str]) and *base64_audio* (List[str])
  which will be converted to PIL images / raw audio arrays via *qwen_omni_utils* when available.
• *Model cache* – Keep models & tokenisers in a global dict so subsequent calls are instant.
• *Thread safety* – The cache is guarded by a simple asyncio lock.

The public coroutine *send_transformers_request* follows the same signature as other providers so
that integration inside *send_request.py* is trivial.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import base64
import logging
import os
from pathlib import Path
import textwrap

import torch

# Transformers is an optional dependency for the toolkit – guard the import
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
    )

    # Optional – might not exist in every transformers release.  We import them
    # in a nested try so failure does *not* disable the whole provider.
    try:
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,  # type: ignore
            Qwen2_5OmniProcessor,  # type: ignore
        )
    except (ImportError, AttributeError):  # pragma: no cover
        Qwen2_5OmniForConditionalGeneration = None  # type: ignore
        Qwen2_5OmniProcessor = None  # type: ignore

    TRANSFORMERS_AVAILABLE = True
except Exception as _e:  # pragma: no cover – keep ComfyUI boot-time lightweight
    TRANSFORMERS_AVAILABLE = False  # Will raise later when used

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#  Model cache (global across requests)
# -----------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = asyncio.Lock()


# -----------------------------------------------------------------------------
#  Helper – determine torch dtype from precision string
# -----------------------------------------------------------------------------

_PRECISION_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

def _to_dtype(precision: str | None) -> torch.dtype:
    if not precision:
        return torch.float16
    precision = precision.lower()
    return _PRECISION_MAP.get(precision, torch.float16)


# -----------------------------------------------------------------------------
#  Helper – decide which device to place the model on
# -----------------------------------------------------------------------------

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # macOS Metal
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
#  Loader – main entry used by *send_transformers_request*
# -----------------------------------------------------------------------------

def _is_awq_model(model_name: str) -> bool:
    return "awq" in model_name.lower()

async def _ensure_model_loaded(
    model_name_or_path: str,
    *,
    precision: str = "fp16",
    device: str | None = None,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    """Load *model_name_or_path* into memory once and cache it.

    Returns a dict with keys *model*, *tokenizer*, and optionally *processor* (for Qwen-Omni).
    Calls are thread-safe (asyncio).
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers_api – HuggingFace transformers not installed.  Please `pip install transformers`"
        )

    async with _CACHE_LOCK:
        if model_name_or_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_name_or_path]

        device = device or _best_device()
        dtype = _to_dtype(precision)

        logger.info(
            "[Transformers-API] Loading model %s | device=%s | dtype=%s",
            model_name_or_path,
            device,
            dtype,
        )

        # ------------------------------------------------------------------
        #  Special-case Qwen2.5-Omni multimodal
        # ------------------------------------------------------------------
        if (
            "qwen2.5-omni" in model_name_or_path.lower()
            and Qwen2_5OmniForConditionalGeneration is not None
            and Qwen2_5OmniProcessor is not None
        ):
            try:
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,  # CPU / MPS handled by HF
                    trust_remote_code=trust_remote_code,
                )
                processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
                tokenizer = None  # Processor handles tokenisation
                _MODEL_CACHE[model_name_or_path] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "processor": processor,
                    "modal": "omni",
                }
                return _MODEL_CACHE[model_name_or_path]
            except Exception as exc:
                logger.error("Failed to load Qwen2.5-Omni model %s – %s", model_name_or_path, exc)
                # Fall through to generic loader below instead of raising

        # If the user requested an Omni model but the specialised classes are unavailable, log a hint.
        if "qwen2.5-omni" in model_name_or_path.lower() and Qwen2_5OmniForConditionalGeneration is None:
            logger.warning(
                "Qwen2.5-Omni support not available in your transformers build; "
                "loading as a plain text model.  Install a newer transformers pre-release "
                "to enable multimodal features."
            )

        # ------------------------------------------------------------------
        #  AWQ (AutoAWQ) support – try specialised loader first
        # ------------------------------------------------------------------
        if _is_awq_model(model_name_or_path):
            try:
                from awq import AutoAWQForCausalLM  # type: ignore

                model = AutoAWQForCausalLM.from_pretrained(  # noqa: F401 – optional import
                    model_name_or_path,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=trust_remote_code,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
                _MODEL_CACHE[model_name_or_path] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "modal": "text",
                }
                return _MODEL_CACHE[model_name_or_path]
            except Exception as exc:
                logger.warning(
                    "AWQ specific loader failed for %s (continuing with normal HF loader): %s",
                    model_name_or_path,
                    exc,
                )
                # Fall through to normal loader

        # ------------------------------------------------------------------
        #  Generic text models (AutoModel)
        # ------------------------------------------------------------------
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            _MODEL_CACHE[model_name_or_path] = {
                "model": model,
                "tokenizer": tokenizer,
                "modal": "text",
            }
            return _MODEL_CACHE[model_name_or_path]
        except Exception as exc:
            logger.error("Could not load model %s via transformers – %s", model_name_or_path, exc)
            raise


# -----------------------------------------------------------------------------
#  Utility – multimodal helpers (images / audio) for Qwen-Omni
# -----------------------------------------------------------------------------

try:
    from qwen_omni_utils import process_mm_info  # provided by qwen-omni-utils package
except Exception:  # pragma: no cover – optional dependency
    def process_mm_info(*_args, **_kwargs):  # type: ignore
        raise NotImplementedError(
            "qwen_omni_utils not installed – required for image/audio inputs with Qwen-Omni"
        )


# -----------------------------------------------------------------------------
#  Public coroutine – mirror *send_openai_request* API
# -----------------------------------------------------------------------------
async def send_transformers_request(
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
) -> Dict[str, Any]:
    """Run local inference with a HF transformer checkpoint.

    Returns an *OpenAI-style* dict so the rest of the toolkit can remain provider-agnostic.
    """
    if not TRANSFORMERS_AVAILABLE:
        return {
            "choices": [
                {"message": {"content": "Transformers not installed.  Please `pip install transformers`."}}]
        }

    try:
        artefacts = await _ensure_model_loaded(model, precision=precision)
        mdl = artefacts["model"]
        tokenizer = artefacts.get("tokenizer")
        processor = artefacts.get("processor")
        modal_type = artefacts.get("modal", "text")

        device = mdl.device if hasattr(mdl, "device") else torch.device("cpu")
        mdl.eval()

        # Build the conversation list for chat-style models
        if messages is None:
            messages = []
        if system_message:
            messages = [{"role": "system", "content": system_message}] + messages
        if user_message:
            messages = messages + [{"role": "user", "content": user_message}]

        # ------------------------------------------------------------------
        #  Qwen2.5-Omni – multimodal path
        # ------------------------------------------------------------------
        if modal_type == "omni":
            if processor is None:
                raise RuntimeError("Qwen-Omni processor not initialised")

            # Convert base64 images / audio to the expected format via helper
            audios, images, videos = process_mm_info(
                messages,  # uses the conversation to pull URLs but we feed empty lists next
                use_audio_in_video=False,
            )
            # Replace the image/audio placeholders explicitly – images list expects raw bytes or PIL
            # For simplicity we treat base64_images as url-style – processor can handle raw bytes too.
            for b64 in base64_images or []:
                images.append({"type": "image", "image": f"data:image/png;base64,{b64}"})
            for b64 in base64_audio or []:
                audios.append({"type": "audio", "audio": base64.b64decode(b64)})

            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text_prompt, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
            inputs = inputs.to(device)

            out_ids, out_audio = mdl.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                return_audio=False,  # We'll stick to text for now
            )
            output_texts = processor.batch_decode(out_ids, skip_special_tokens=True)
            content = output_texts[0]
        else:
            # ------------------------------------------------------------------
            #  Generic text-only generation
            # ------------------------------------------------------------------
            if tokenizer is None:
                raise RuntimeError("Tokenizer missing for text model")

            # Compose single prompt from messages for simple models
            prompt_parts: List[str] = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"[INST] {msg['content']} [/INST]")
                elif msg["role"] == "user":
                    prompt_parts.append(f"<|user|>: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<|assistant|>: {msg['content']}")
            prompt = "\n".join(prompt_parts) + "\n<|assistant|>:"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen_ids = mdl.generate(
                    input_ids,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                )
            content = tokenizer.decode(gen_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)

        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
    except Exception as exc:
        logger.error("transformers_api – inference error: %s", exc, exc_info=True)
        return {"choices": [{"message": {"content": f"Error: {exc}"}}]}


# -----------------------------------------------------------------------------
#  Public coroutine – streaming version (yields chunks)
# -----------------------------------------------------------------------------
async def send_transformers_request_stream(
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
    """Async generator that yields text chunks as they are generated."""
    if not TRANSFORMERS_AVAILABLE:
        yield "[Transformers not installed]"
        return

    try:
        artefacts = await _ensure_model_loaded(model, precision=precision)
        mdl = artefacts["model"]
        tokenizer = artefacts.get("tokenizer")
        modal_type = artefacts.get("modal", "text")

        if modal_type != "text":
            # For now fall back to full generation for multimodal models.
            full_resp = await send_transformers_request(
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
            content = full_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                for _chunk in textwrap.wrap(content, 120):
                    yield _chunk
            return

        if tokenizer is None:
            raise RuntimeError("Tokenizer missing for text model (streaming)")

        device = mdl.device if hasattr(mdl, "device") else torch.device("cpu")
        mdl.eval()

        # Build prompt
        if messages is None:
            messages = []
        if system_message:
            messages = [{"role": "system", "content": system_message}] + messages
        if user_message:
            messages = messages + [{"role": "user", "content": user_message}]

        prompt_parts: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg["role"] == "user":
                prompt_parts.append(f"<|user|>: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>: {msg['content']}")
        prompt = "\n".join(prompt_parts) + "\n<|assistant|>:"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        from threading import Thread
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
        )

        # Run generate in background thread so we can iterate streamer concurrently
        gen_thread = Thread(target=mdl.generate, kwargs=generation_kwargs, daemon=True)
        gen_thread.start()

        loop = asyncio.get_running_loop()

        stop = False
        while not stop:
            def _next_chunk():
                try:
                    return next(streamer)
                except StopIteration:
                    return None

            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                stop = True
            elif chunk:
                yield chunk

        gen_thread.join(timeout=0)
    except Exception as exc:
        logger.warning(
            "transformers_api – streaming failed (%s). Falling back to blocking generation.",
            exc,
        )
        try:
            full_resp = await send_transformers_request(
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
            content = full_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                for _chunk in textwrap.wrap(content, 120):
                    yield _chunk
        except Exception as e2:
            logger.error("transformers_api – fallback blocking generation also failed: %s", e2, exc_info=True)
            yield f"[Error: {e2}]" 