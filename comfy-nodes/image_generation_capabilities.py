"""Image model capability helpers for LLM Toolkit image generation nodes."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

DEFAULT_MODEL = "wavespeed-ai/flux-kontext-dev-ultra-fast"

# Keys managed by the unified Configure Image node.
MANAGED_KEYS = {
    "model_id",
    "n",
    "size",
    "response_format",
    "user",
    "quality_dalle3",
    "style_dalle3",
    "quality_gpt",
    "background_gpt",
    "output_format_gpt",
    "output_format",
    "output_compression_gpt",
    "moderation_gpt",
    "aspect_ratio",
    "person_generation",
    "safety_filter_level",
    "language",
    "temperature_gemini",
    "max_tokens_gemini",
    "seed",
    "prompt_upsampling",
    "safety_tolerance",
    "output_format_bfl",
    "guidance_scale",
    "num_inference_steps",
    "enable_safety_checker",
    "resolution",
    "prompt_expansion",
}

logger = logging.getLogger(__name__)

HandlerType = Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

_MODEL_HANDLERS: Dict[str, HandlerType] = {}
_ALIAS_LOOKUP: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _register_model(name: str, handler: HandlerType, aliases: Optional[Sequence[str]] = None) -> None:
    key = name.lower()
    _MODEL_HANDLERS[key] = handler
    _ALIAS_LOOKUP[key] = name
    if aliases:
        for alias in aliases:
            _ALIAS_LOOKUP[alias.lower()] = name


def resolve_canonical_model(model_id: Optional[str]) -> str:
    if not model_id:
        return DEFAULT_MODEL
    lookup = model_id.strip().lower()
    return _ALIAS_LOOKUP.get(lookup, model_id)


def _parse_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        stripped = stripped.replace("seconds", "").replace("second", "")
        if stripped.endswith("s"):
            stripped = stripped[:-1]
        stripped = stripped.strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _parse_int(value: Any) -> Optional[int]:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    try:
        return int(round(parsed))
    except (TypeError, ValueError):
        return None


def _clamp_int(value: Optional[int], minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, value))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _sanitize_seed(value: Any, fallback: Any) -> Optional[int]:
    candidate = _parse_int(value)
    if candidate is None:
        candidate = _parse_int(fallback)
    if candidate is None:
        return None
    if candidate < 0:
        return -1
    max_seed = (1 << 63) - 1
    return max(0, min(max_seed, candidate))


def _normalize_response_format(requested: Any, existing: Any, default: str = "b64_json") -> str:
    allowed = {"url", "b64_json"}
    value = _clean_string(requested)
    if value:
        lowered = value.lower()
        if lowered == "auto":
            return default
        if lowered in allowed:
            return lowered
    value = _clean_string(existing)
    if value and value.lower() in allowed:
        return value.lower()
    return default


def _size_to_aspect(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    tokens = size.lower().replace("*", "x").split("x")
    if len(tokens) != 2:
        return None
    try:
        width = int(tokens[0])
        height = int(tokens[1])
        if width <= 0 or height <= 0:
            return None
        import math

        g = math.gcd(width, height)
        return f"{width // g}:{height // g}"
    except (ValueError, TypeError):
        return None


def _sanitize_size(value: Any, existing: Any, allowed: Optional[Sequence[str]] = None, default: Optional[str] = None) -> Optional[str]:
    def _pick(candidate: Any) -> Optional[str]:
        cleaned = _clean_string(candidate)
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered == "auto":
            return None
        normalized = lowered.replace("*", "x")
        return normalized

    for candidate in (value, existing, default):
        normalized = _pick(candidate)
        if normalized:
            if allowed is None or normalized in {opt.lower() for opt in allowed}:
                return normalized
    return None


def _convert_size_for_seedream(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    return size.replace("x", "*")


def _match_choice(value: Any, existing: Any, allowed: Sequence[str], default: str) -> str:
    allowed_set = {opt.lower(): opt for opt in allowed}
    candidate = _clean_string(value)
    if candidate and candidate.lower() in allowed_set:
        return allowed_set[candidate.lower()]
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed_set:
        return allowed_set[candidate.lower()]
    return default


def _map_quality_for_dalle3(value: Any, existing: Any) -> str:
    mapping = {
        "hd": "hd",
        "high": "hd",
        "ultra": "hd",
        "standard": "standard",
        "medium": "standard",
        "low": "standard",
    }
    candidate = _clean_string(value)
    if candidate:
        mapped = mapping.get(candidate.lower())
        if mapped:
            return mapped
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in {"hd", "standard"}:
        return candidate.lower()
    return "standard"


def _map_style_for_dalle3(value: Any, existing: Any) -> str:
    allowed = {"vivid", "natural"}
    candidate = _clean_string(value)
    if candidate:
        lowered = candidate.lower()
        if lowered == "auto":
            return "vivid"
        if lowered in allowed:
            return lowered
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "vivid"


def _map_quality_for_gpt(value: Any, existing: Any) -> str:
    allowed = {"auto", "low", "medium", "high"}
    candidate = _clean_string(value)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "auto"


def _map_background(value: Any, existing: Any) -> str:
    allowed = {"auto", "opaque", "transparent"}
    candidate = _clean_string(value)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "auto"


def _map_output_format_generic(value: Any, existing: Any, allowed: Sequence[str], default: str) -> str:
    allowed_set = {opt.lower(): opt for opt in allowed}
    candidate = _clean_string(value)
    if candidate:
        lowered = candidate.lower()
        if lowered == "auto":
            return default
        if lowered in allowed_set:
            return allowed_set[lowered]
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed_set:
        return allowed_set[candidate.lower()]
    return default

def _map_output_format_gpt(value: Any, existing: Any) -> str:
    allowed = {"png", "jpeg", "webp"}
    candidate = _clean_string(value)
    if candidate:
        lowered = candidate.lower()
        if lowered == "auto":
            return "png"
        if lowered in allowed:
            return lowered
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "png"


def _map_output_format_bfl(value: Any, existing: Any) -> str:
    allowed = {"png", "jpeg"}
    candidate = _clean_string(value)
    if candidate:
        lowered = candidate.lower()
        if lowered == "auto":
            return "png"
        if lowered in allowed:
            return lowered
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "png"


def _map_moderation_gpt(value: Any, existing: Any) -> str:
    allowed = {"auto", "low"}
    candidate = _clean_string(value)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "auto"


def _map_person_policy(value: Any, existing: Any) -> str:
    allowed = {"dont_allow", "allow_adult", "allow_all"}
    candidate = _clean_string(value)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    candidate = _clean_string(existing)
    if candidate and candidate.lower() in allowed:
        return candidate.lower()
    return "allow_adult"


def _map_imagen4_aspect_ratio(value: Any, existing: Any) -> str:
    allowed = {opt.lower(): opt for opt in IMAGEN4_ASPECTS}

    def _resolve(candidate: Any) -> Optional[str]:
        cleaned = _clean_string(candidate)
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered == "auto":
            return None
        return allowed.get(lowered)

    ratio = _resolve(value)
    if ratio:
        return ratio
    ratio = _resolve(existing)
    if ratio:
        return ratio
    return IMAGEN4_ASPECTS[0]


def _map_imagen4_resolution(
    size_value: Any,
    resolution_value: Any,
    existing_resolution: Any,
    default: str = "1k",
) -> str:
    def _resolve(candidate: Any) -> Optional[str]:
        cleaned = _clean_string(candidate)
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered == "auto":
            return None
        if lowered in {"1k", "2k"}:
            return lowered
        normalized = lowered.replace("*", "x")
        if "x" in normalized:
            tokens = normalized.split("x")
            if len(tokens) == 2:
                try:
                    width = int(tokens[0])
                    height = int(tokens[1])
                    max_dim = max(width, height)
                    if max_dim >= 2000:
                        return "2k"
                    if max_dim >= 1200:
                        return "1k"
                except (ValueError, TypeError):
                    return None
        return None

    for source in (resolution_value, size_value, existing_resolution):
        resolved = _resolve(source)
        if resolved:
            return resolved
    return default


def _map_safety_for_gemini(setting: Any, existing: Any) -> str:
    mapping = {
        "strict": "block_high_and_above",
        "balanced": "block_medium_and_above",
        "default": "block_medium_and_above",
        "relaxed": "block_low_and_above",
    }
    candidate = _clean_string(setting)
    if candidate:
        mapped = mapping.get(candidate.lower())
        if mapped:
            return mapped
    candidate = _clean_string(existing)
    if candidate:
        lowered = candidate.lower()
        if lowered in {"block_low_and_above", "block_medium_and_above", "block_high_and_above"}:
            return lowered
    return "block_medium_and_above"


def _map_safety_for_bfl(setting: Any, override: Any, existing: Any) -> int:
    override_val = _parse_int(override)
    if override_val is not None and 0 <= override_val <= 6:
        return override_val
    mapping = {
        "strict": 0,
        "balanced": 2,
        "default": 2,
        "relaxed": 4,
    }
    candidate = _clean_string(setting)
    if candidate:
        mapped = mapping.get(candidate.lower())
        if mapped is not None:
            return mapped
    fallback = _parse_int(existing)
    if fallback is not None and 0 <= fallback <= 6:
        return fallback
    return 2


def _sanitize_temperature(value: Any, existing: Any, default: float) -> float:
    candidate = _parse_float(value)
    if candidate is None:
        candidate = _parse_float(existing)
    if candidate is None:
        candidate = default
    return max(0.0, min(2.0, candidate))


def _sanitize_max_tokens(value: Any, existing: Any, default: int) -> int:
    candidate = _parse_int(value)
    if candidate is None:
        candidate = _parse_int(existing)
    if candidate is None:
        candidate = default
    return _clamp_int(candidate, 1, 32768, default)


def _sanitize_guidance(value: Any, existing: Any, default: float, minimum: float, maximum: float) -> float:
    candidate = _parse_float(value)
    if candidate is None:
        candidate = _parse_float(existing)
    if candidate is None:
        candidate = default
    return max(minimum, min(maximum, candidate))


def _sanitize_compression(value: Any, existing: Any) -> Optional[int]:
    candidate = _parse_int(value)
    if candidate is None:
        candidate = _parse_int(existing)
    if candidate is None:
        return None
    return _clamp_int(candidate, 0, 100, 100)


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------

OPENAI_DALLE3_SIZES = ["1024x1024", "1792x1024", "1024x1792"]
OPENAI_DALLE2_SIZES = ["256x256", "512x512", "1024x1024"]
OPENAI_GPT_SIZES = ["1024x1024", "1024x1536", "1536x1024"]
GEMINI_ASPECTS = ["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
BFL_ASPECTS = ["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "21:9", "9:21"]
IMAGEN4_ASPECTS = ["1:1", "16:9", "9:16", "4:3", "3:4"]


def _handle_dalle3(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    size = _sanitize_size(requested.get("size"), existing.get("size"), OPENAI_DALLE3_SIZES, OPENAI_DALLE3_SIZES[0])
    response_format = _normalize_response_format(requested.get("response_format"), existing.get("response_format"))
    quality = _map_quality_for_dalle3(requested.get("quality"), existing.get("quality_dalle3"))
    style = _map_style_for_dalle3(requested.get("style"), existing.get("style_dalle3"))
    user = _clean_string(requested.get("user_tag")) or _clean_string(existing.get("user"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": 1,
        "size": size,
        "response_format": response_format,
        "quality_dalle3": quality,
        "style_dalle3": style,
    }
    if user:
        payload["user"] = user
    return payload


def _handle_dalle2(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, 10, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), OPENAI_DALLE2_SIZES, OPENAI_DALLE2_SIZES[2])
    response_format = _normalize_response_format(requested.get("response_format"), existing.get("response_format"))
    user = _clean_string(requested.get("user_tag")) or _clean_string(existing.get("user"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": n,
        "size": size,
        "response_format": response_format,
    }
    if user:
        payload["user"] = user
    return payload


def _handle_gpt_image(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, 10, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), OPENAI_GPT_SIZES, OPENAI_GPT_SIZES[0])
    quality = _map_quality_for_gpt(requested.get("quality"), existing.get("quality_gpt"))
    background = _map_background(requested.get("background"), existing.get("background_gpt"))
    output_format = _map_output_format_gpt(requested.get("output_format"), existing.get("output_format_gpt"))
    moderation = _map_moderation_gpt(requested.get("moderation"), existing.get("moderation_gpt"))
    compression = _sanitize_compression(requested.get("output_compression"), existing.get("output_compression_gpt"))
    user = _clean_string(requested.get("user_tag")) or _clean_string(existing.get("user"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": n,
        "size": size,
        "response_format": "b64_json",
        "quality_gpt": quality,
        "background_gpt": background,
        "output_format_gpt": output_format,
        "moderation_gpt": moderation,
    }
    if compression is not None:
        payload["output_compression_gpt"] = compression
    if user:
        payload["user"] = user
    return payload


def _common_gemini_payload(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any], *, max_images: int) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, max_images, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, None)
    ratio = _clean_string(requested.get("aspect_ratio")) or _clean_string(existing.get("aspect_ratio")) or _size_to_aspect(size)
    if ratio and ratio not in GEMINI_ASPECTS:
        ratio = _size_to_aspect(size)
    person = _map_person_policy(requested.get("person_policy"), existing.get("person_generation"))
    safety = _map_safety_for_gemini(requested.get("safety_setting"), existing.get("safety_filter_level"))
    language = _clean_string(requested.get("language_hint"))
    if language and language.lower() in {"auto", "default"}:
        language = None
    if not language:
        language = _clean_string(existing.get("language"))
    temperature = _sanitize_temperature(requested.get("temperature"), existing.get("temperature_gemini"), 0.7)
    max_tokens = _sanitize_max_tokens(requested.get("max_tokens"), existing.get("max_tokens_gemini"), 8192)
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": n,
        "temperature_gemini": temperature,
        "max_tokens_gemini": max_tokens,
        "person_generation": person,
        "safety_filter_level": safety,
    }
    if size:
        payload["size"] = size
    if ratio and ratio in GEMINI_ASPECTS:
        payload["aspect_ratio"] = ratio
    if language:
        payload["language"] = language
    if seed is not None:
        payload["seed"] = seed
    return payload


def _handle_gemini_image(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    return _common_gemini_payload(canonical, requested, existing, max_images=4)


def _handle_imagen(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    # Imagen Ultra models typically only allow a single image per request.
    return _common_gemini_payload(canonical, requested, existing, max_images=1)


def _handle_wavespeed_flux(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, 8, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, "1024x1024")
    guidance = _sanitize_guidance(requested.get("guidance_scale"), existing.get("guidance_scale"), 2.5, 0.5, 20.0)
    steps = _clamp_int(_parse_int(requested.get("inference_steps")) or _parse_int(existing.get("num_inference_steps")), 1, 50, 28)
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    safety_checker = _coerce_bool(requested.get("enable_safety_checker"), _coerce_bool(existing.get("enable_safety_checker"), True))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": n,
        "size": size,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "enable_safety_checker": safety_checker,
    }
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload




def _handle_wavespeed_hunyuan(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, "1024x1024")
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    output_format = _map_output_format_generic(
        requested.get("output_format"),
        existing.get("output_format"),
        ["png", "jpeg"],
        "png",
    )

    payload: Dict[str, Any] = {"model_id": canonical, "n": 1, "output_format": output_format}
    if size:
        payload["size"] = _convert_size_for_seedream(size)
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_qwen_edit_plus(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, None)
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    output_format = _map_output_format_generic(
        requested.get("output_format"),
        existing.get("output_format"),
        ["jpeg", "png", "webp"],
        "jpeg",
    )

    payload: Dict[str, Any] = {"model_id": canonical, "n": 1, "output_format": output_format}
    if size:
        payload["size"] = _convert_size_for_seedream(size)
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_seededit(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    guidance = _sanitize_guidance(requested.get("guidance_scale"), existing.get("guidance_scale"), 0.5, 0.0, 1.0)

    payload: Dict[str, Any] = {"model_id": canonical, "n": 1, "guidance_scale": guidance}
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_portrait(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    payload: Dict[str, Any] = {"model_id": canonical, "n": 1}
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_seedream(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, 16, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, "2048x2048")
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {"model_id": canonical, "n": n}
    if size:
        payload["size"] = _convert_size_for_seedream(size)
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_bfl_flux(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    ratio = _clean_string(requested.get("aspect_ratio")) or _clean_string(existing.get("aspect_ratio"))
    if not ratio or ratio not in BFL_ASPECTS:
        ratio = "1:1"
    safety = _map_safety_for_bfl(requested.get("safety_setting"), requested.get("safety_tolerance"), existing.get("safety_tolerance"))
    output_format = _map_output_format_bfl(requested.get("output_format"), existing.get("output_format_bfl"))
    prompt_upsampling = _coerce_bool(requested.get("prompt_enhancement"), _coerce_bool(existing.get("prompt_upsampling"), False))
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": 1,
        "aspect_ratio": ratio,
        "prompt_upsampling": prompt_upsampling,
        "safety_tolerance": safety,
        "output_format_bfl": output_format,
    }
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_openrouter_generic(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    n = _clamp_int(_parse_int(requested.get("image_count")) or _parse_int(existing.get("n")), 1, 10, 1)
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, None)
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {"model_id": canonical, "n": n}
    if size:
        payload["size"] = size
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_imagen4(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    default_resolution = "1k"
    lowered = canonical.lower()
    if "ultra" in lowered:
        default_resolution = "2k"
    elif lowered.endswith("imagen4"):
        default_resolution = "2k"

    n = _clamp_int(
        _parse_int(requested.get("image_count")) or _parse_int(existing.get("n")),
        1,
        4,
        1,
    )
    aspect_ratio = _map_imagen4_aspect_ratio(requested.get("aspect_ratio"), existing.get("aspect_ratio"))
    resolution = _map_imagen4_resolution(
        requested.get("size"),
        requested.get("resolution"),
        existing.get("resolution"),
        default=default_resolution,
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": n,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_dreamina(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    size = _sanitize_size(requested.get("size"), existing.get("size"), None, "1328x1328")
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    prompt_expansion = _coerce_bool(
        requested.get("prompt_enhancement"),
        _coerce_bool(existing.get("prompt_expansion"), True),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": 1,
        "prompt_expansion": prompt_expansion,
    }
    if size:
        payload["size"] = size
    if seed is not None and seed != -1:
        payload["seed"] = seed
    return payload


def _handle_wavespeed_nano_banana_edit(canonical: str, requested: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    output_format = _map_output_format_generic(
        requested.get("output_format"),
        existing.get("output_format"),
        ["png", "jpeg", "webp"],
        "png",
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "n": 1,
        "output_format": output_format,
    }
    return payload


def _infer_handler_from_provider(provider: str, model: str) -> Optional[str]:
    provider = (provider or "").lower()
    lowered_model = (model or "").lower()
    if provider == "openai":
        if "gpt-image" in lowered_model:
            return "openai/gpt-image-1"
        if "dall-e-2" in lowered_model:
            return "openai/dall-e-2"
        if "dall-e" in lowered_model:
            return "openai/dall-e-3"
    elif provider in {"gemini", "google"}:
        if lowered_model.startswith("imagen") or "imagen" in lowered_model:
            return "google/imagen"
        return "google/gemini-image"
    elif provider == "wavespeed":
        if "hunyuan" in lowered_model:
            return "wavespeed/hunyuan-image-3"
        if "qwen" in lowered_model and "edit" in lowered_model:
            return "wavespeed/qwen-image-edit-plus"
        if "imagen4" in lowered_model:
            if "fast" in lowered_model:
                return "wavespeed/imagen4-fast"
            if "ultra" in lowered_model:
                return "wavespeed/imagen4-ultra"
            return "wavespeed/imagen4"
        if "nano" in lowered_model and "banana" in lowered_model:
            return "wavespeed/nano-banana-edit"
        if "dreamina" in lowered_model:
            return "wavespeed/dreamina-v3.1"
        if "seededit" in lowered_model:
            return "wavespeed/seededit"
        if "portrait" in lowered_model:
            return "wavespeed/portrait"
        if "seedream" in lowered_model:
            return "wavespeed/seedream"
        return "wavespeed/flux"
    elif provider == "bfl":
        return "bfl/flux"
    elif provider == "openrouter":
        return "openrouter/generic"
    return None


def normalize_generation_config(
    model_id: Optional[str],
    requested: Optional[Dict[str, Any]] = None,
    existing: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    requested = requested or {}
    existing = existing.copy() if existing else {}

    canonical = resolve_canonical_model(model_id)
    handler = _MODEL_HANDLERS.get(canonical.lower())

    provider_hint = _clean_string(requested.get("provider")) or _clean_string(existing.get("provider")) or ""
    if handler is None:
        inferred = _infer_handler_from_provider(provider_hint, canonical)
        if inferred:
            canonical = inferred
            handler = _MODEL_HANDLERS.get(canonical.lower())

    if handler is None:
        logger.debug("No image capability handler for model '%s'; using default flux handler.", model_id)
        canonical = DEFAULT_MODEL
        handler = _MODEL_HANDLERS[canonical.lower()]

    payload = handler(canonical, requested, existing)
    return canonical, payload


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------
_register_model(
    "openai/dall-e-3",
    _handle_dalle3,
    aliases=["dall-e-3", "openai/dall-e-3", "dalle-3", "openai:dall-e-3"],
)
_register_model(
    "openai/dall-e-2",
    _handle_dalle2,
    aliases=["dall-e-2", "openai/dall-e-2", "dalle-2", "openai:dall-e-2"],
)
_register_model(
    "openai/gpt-image-1",
    _handle_gpt_image,
    aliases=["gpt-image-1", "openai/gpt-image-1", "openai:gpt-image-1"],
)
_register_model(
    "google/gemini-image",
    _handle_gemini_image,
    aliases=[
        "gemini-2.5-flash-image-preview",
        "gemini-2.0-flash-preview-image-generation",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-2.0-flash-preview-image-generation",
        "gemini-2.5-flash-image-preview:free",
    ],
)
_register_model(
    "google/imagen",
    _handle_imagen,
    aliases=[
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-generate-preview-06-06",
        "imagen-4.0-ultra-generate-preview-06-06",
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3-light-alpha",
        "google/imagen-4.0-generate-001",
        "google/imagen-4.0-ultra-generate-001",
    ],
)
_register_model(
    "wavespeed/flux",
    _handle_wavespeed_flux,
    aliases=[
        "wavespeed-ai/flux-kontext-dev-ultra-fast",
        "wavespeed-ai/flux-kontext-dev/multi-ultra-fast",
        "flux-kontext-dev-ultra-fast",
    ],
)
_register_model(
    "wavespeed/hunyuan-image-3",
    _handle_wavespeed_hunyuan,
    aliases=[
        "wavespeed-ai/hunyuan-image-3",
        "hunyuan-image-3",
        "hunyuanimage-3.0",
    ],
)
_register_model(
    "wavespeed/seededit",
    _handle_wavespeed_seededit,
    aliases=["bytedance/seededit-v3"],
)
_register_model(
    "wavespeed/portrait",
    _handle_wavespeed_portrait,
    aliases=["bytedance/portrait"],
)
_register_model(
    "wavespeed/qwen-image-edit-plus",
    _handle_wavespeed_qwen_edit_plus,
    aliases=["wavespeed-ai/qwen-image/edit-plus", "qwen-image-edit-plus"],
)
_register_model(
    "wavespeed/seedream",
    _handle_wavespeed_seedream,
    aliases=[
        "bytedance/seedream-v3.1",
        "bytedance/seedream-v4",
        "bytedance/seedream-v4-edit",
        "bytedance/seedream-v4-edit-sequential",
        "bytedance/seedream-v4-sequential",
    ],
)
_register_model(
    "wavespeed/imagen4",
    _handle_wavespeed_imagen4,
    aliases=[
        "google/imagen4",
        "imagen4",
    ],
)
_register_model(
    "wavespeed/imagen4-fast",
    _handle_wavespeed_imagen4,
    aliases=[
        "google/imagen4-fast",
        "imagen4-fast",
    ],
)
_register_model(
    "wavespeed/imagen4-ultra",
    _handle_wavespeed_imagen4,
    aliases=[
        "google/imagen4-ultra",
        "imagen4-ultra",
    ],
)
_register_model(
    "bfl/flux",
    _handle_bfl_flux,
    aliases=["flux-kontext-max", "bfl/flux-kontext-max"],
)
_register_model(
    "wavespeed/dreamina-v3.1",
    _handle_wavespeed_dreamina,
    aliases=[
        "bytedance/dreamina-v3.1/text-to-image",
        "dreamina-v3.1",
    ],
)
_register_model(
    "wavespeed/nano-banana-edit",
    _handle_wavespeed_nano_banana_edit,
    aliases=[
        "google/nano-banana/edit",
        "nano-banana/edit",
    ],
)
_register_model(
    "openrouter/generic",
    _handle_openrouter_generic,
    aliases=["openrouter/image"],
)

__all__ = [
    "DEFAULT_MODEL",
    "MANAGED_KEYS",
    "normalize_generation_config",
    "resolve_canonical_model",
]
