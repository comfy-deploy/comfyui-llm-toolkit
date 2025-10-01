"""Video model capability helpers for LLM Toolkit video generation nodes."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

DEFAULT_MODEL = "veo-2.0-generate-001"

# Keys we manage inside generation_config. Anything outside this set is left intact.
MANAGED_KEYS = {
    "model_id",
    "aspect_ratio",
    "person_generation",
    "number_of_videos",
    "duration_seconds",
    "duration",
    "negative_prompt",
    "enhance_prompt",
    "enable_prompt_expansion",
    "guidance_scale",
    "image",
    "resolution",
    "size",
    "audio",
    "video",
    "generate_audio",
    "seed",
    "voice_id",
    "voice_language",
    "voice_speed",
    "avatar_id",
    "character_id",
    "effect_id",
    "effect_type",
    "effect_strength",
}

logger = logging.getLogger(__name__)

HandlerType = Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

_MODEL_HANDLERS: Dict[str, HandlerType] = {}
_ALIAS_LOOKUP: Dict[str, str] = {}


def resolve_canonical_model(model_id: Optional[str]) -> str:
    """Return the canonical model string used by capability handlers."""
    if not model_id:
        return DEFAULT_MODEL
    return _ALIAS_LOOKUP.get(model_id.lower(), model_id)


def normalize_generation_config(
    model_id: Optional[str],
    requested: Optional[Dict[str, Any]] = None,
    existing: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return sanitized generation settings for the given model.

    Parameters
    ----------
    model_id: Optional[str]
        Model identifier from provider or prior context.
    requested: Optional[Dict[str, Any]]
        Values requested by the configuration node (may be empty when called
        during runtime validation).
    existing: Optional[Dict[str, Any]]
        Current ``generation_config`` values. These are used as fallbacks when
        the node did not request an override.
    """

    requested = requested or {}
    existing = existing.copy() if existing else {}

    canonical = resolve_canonical_model(model_id)
    handler = _MODEL_HANDLERS.get(canonical.lower())
    if handler is None:
        logger.debug("No handler registered for model %s; using default.", canonical)
        canonical = DEFAULT_MODEL
        handler = _MODEL_HANDLERS[canonical.lower()]

    payload = handler(canonical, requested, existing)
    return canonical, payload


def _register_model(name: str, handler: HandlerType, aliases: Optional[Sequence[str]] = None) -> None:
    _MODEL_HANDLERS[name.lower()] = handler
    _ALIAS_LOOKUP[name.lower()] = name
    if aliases:
        for alias in aliases:
            _ALIAS_LOOKUP[alias.lower()] = name


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


def _nearest_value(target: int, allowed: Sequence[int]) -> int:
    return min(allowed, key=lambda candidate: abs(candidate - target))


def _select_choice(
    requested_value: Any,
    existing_value: Any,
    allowed: Sequence[str],
    default: str,
) -> str:
    lookup = {opt.lower(): opt for opt in allowed}
    for candidate in (requested_value, existing_value):
        if isinstance(candidate, str):
            mapped = lookup.get(candidate.strip().lower())
            if mapped:
                return mapped
    return lookup.get(default.lower(), allowed[0])


def _sanitize_negative_prompt(requested_value: Any, existing_value: Any) -> Optional[str]:
    for candidate in (requested_value, existing_value):
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned
    return None


def _sanitize_image(requested_value: Any, existing_value: Any) -> Optional[str]:
    candidates = []
    if isinstance(requested_value, str):
        candidates.append(requested_value)
    if isinstance(existing_value, str):
        candidates.append(existing_value)
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned:
            return cleaned
    return None


def _sanitize_seed(requested_value: Any, existing_value: Any, default: int = -1) -> int:
    seed = _parse_int(requested_value)
    if seed is None:
        seed = _parse_int(existing_value)
    if seed is None:
        seed = default
    if seed < -1:
        seed = -1
    return seed

def _sanitize_string_field(requested_value: Any, existing_value: Any) -> Optional[str]:
    for candidate in (requested_value, existing_value):
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned
    return None


def _sanitize_audio(requested_value: Any, existing_value: Any) -> Optional[str]:
    return _sanitize_string_field(requested_value, existing_value)




def _compute_duration(
    requested_value: Any,
    existing_value: Any,
    allowed: Sequence[int],
    default_bias: Any = "max",
) -> int:
    if not allowed:
        candidate = _parse_int(requested_value)
        if candidate is None:
            candidate = _parse_int(existing_value)
        return candidate if candidate is not None else 0

    allowed_sorted = sorted({int(x) for x in allowed})
    if not allowed_sorted:
        return 0

    if isinstance(default_bias, int) and default_bias in allowed_sorted:
        fallback = default_bias
    elif isinstance(default_bias, str) and default_bias.lower() == "min":
        fallback = allowed_sorted[0]
    else:
        fallback = allowed_sorted[-1]

    candidate = _parse_int(requested_value)
    if candidate is None:
        candidate = _parse_int(existing_value)
    if candidate is None:
        candidate = fallback

    candidate = max(allowed_sorted[0], min(allowed_sorted[-1], candidate))
    return _nearest_value(candidate, allowed_sorted)



# ---------------------------------------------------------------------------
# Model handlers
# ---------------------------------------------------------------------------
_GEMINI_ASPECTS = ["16:9", "9:16"]
_WAVESPEED_ASPECTS = ["16:9", "9:16", "1:1", "4:3", "3:4"]

_VEO3_IMAGE_ASPECTS = ["16:9", "9:16"]
_VEO3_IMAGE_RESOLUTIONS = ["720p", "1080p"]


_SEEDANCE_ASPECTS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]


_WAN_RESOLUTIONS = ["480p", "720p", "1080p"]
_WAN_SIZE_BY_RESOLUTION = {
    "480p": "854*480",
    "720p": "1280*720",
    "1080p": "1920*1080",
}


def _handle_gemini_veo2(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _GEMINI_ASPECTS,
        "16:9",
    )

    person_req = requested.get("person_generation")
    if not isinstance(person_req, str) or not person_req.strip():
        person_req = "allow_adult"
    if person_req.strip().lower() == "allow_all":
        person_req = "allow_adult"
    person = _select_choice(
        person_req,
        existing.get("person_generation"),
        ["dont_allow", "allow_adult"],
        "allow_adult",
    )

    videos_value = requested.get("number_of_videos", existing.get("number_of_videos"))
    videos_int = _parse_int(videos_value)
    if videos_int is None:
        videos_int = 1
    videos_int = max(1, min(2, videos_int))

    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration_seconds"),
        [5, 6, 7, 8],
    )

    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    enhance_prompt = _coerce_bool(
        requested.get("enhance_prompt"),
        _coerce_bool(existing.get("enhance_prompt"), True),
    )

    payload: Dict[str, Any] = {
        "aspect_ratio": aspect,
        "person_generation": person,
        "number_of_videos": videos_int,
        "duration_seconds": duration,
        "enhance_prompt": enhance_prompt,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload



def _handle_wavespeed_veo2_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _WAVESPEED_ASPECTS,
        "16:9",
    )
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 6, 7, 8],
    )
    return {
        "model_id": canonical,
        "aspect_ratio": aspect,
        "duration": f"{duration}s",
    }



def _handle_wavespeed_veo2_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    payload = _handle_wavespeed_veo2_t2v(canonical, requested, existing)
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )
    if image:
        payload["image"] = image
    return payload


def _handle_veo3_fast(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _WAVESPEED_ASPECTS,
        "16:9",
    )
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [8],
    )
    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    enable_expansion = _coerce_bool(
        requested.get("enable_prompt_expansion"),
        _coerce_bool(existing.get("enable_prompt_expansion"), True),
    )
    generate_audio = _coerce_bool(
        requested.get("generate_audio"),
        _coerce_bool(existing.get("generate_audio"), True),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "aspect_ratio": aspect,
        "duration": duration,
        "enable_prompt_expansion": enable_expansion,
        "generate_audio": generate_audio,
        "seed": seed,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload



def _handle_veo3_fast_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _VEO3_IMAGE_ASPECTS,
        "16:9",
    )
    resolution = _select_choice(
        requested.get("video_resolution") or requested.get("resolution"),
        existing.get("resolution"),
        _VEO3_IMAGE_RESOLUTIONS,
        "720p",
    )
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [8],
    )
    generate_audio = _coerce_bool(
        requested.get("generate_audio"),
        _coerce_bool(existing.get("generate_audio"), False),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "aspect_ratio": aspect,
        "duration": duration,
        "resolution": resolution,
        "generate_audio": generate_audio,
        "seed": seed,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if image:
        payload["image"] = image
    return payload



def _handle_veo3_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    return _handle_veo3_fast_i2v(canonical, requested, existing)



def _handle_veo3(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    return {
        "model_id": canonical,
        "seed": seed,
    }


def _handle_seedance_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _SEEDANCE_ASPECTS,
        "16:9",
    )
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        list(range(5, 11)),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    return {
        "model_id": canonical,
        "aspect_ratio": aspect,
        "duration": duration,
        "seed": seed,
    }



def _handle_seedance_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        list(range(5, 11)),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )
    payload: Dict[str, Any] = {
        "model_id": canonical,
        "duration": duration,
        "seed": seed,
    }
    if image:
        payload["image"] = image
    return payload



def _resolve_wan_resolution(requested: Dict[str, Any], existing: Dict[str, Any]) -> str:
    existing_resolution = existing.get("resolution")
    if not isinstance(existing_resolution, str):
        size_val = existing.get("size")
        if isinstance(size_val, str):
            for res, code in _WAN_SIZE_BY_RESOLUTION.items():
                if size_val.strip().lower() == code.lower():
                    existing_resolution = res
                    break

    requested_resolution = None
    for key in ("resolution", "video_resolution"):
        value = requested.get(key)
        if isinstance(value, str) and value.strip():
            requested_resolution = value
            break

    default_resolution = _WAN_RESOLUTIONS[-1]
    return _select_choice(
        requested_resolution,
        existing_resolution,
        _WAN_RESOLUTIONS,
        default_resolution,
    )



def _handle_wan25_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    resolution = _resolve_wan_resolution(requested, existing)
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 8, 10],
    )
    enable_expansion = _coerce_bool(
        requested.get("enable_prompt_expansion"),
        _coerce_bool(existing.get("enable_prompt_expansion"), True),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )
    audio = _sanitize_audio(
        requested.get("audio") or requested.get("audio_url"),
        existing.get("audio"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "resolution": resolution,
        "duration": duration,
        "enable_prompt_expansion": enable_expansion,
        "seed": seed,
    }
    if image:
        payload["image"] = image
    if audio:
        payload["audio"] = audio
    return payload



def _handle_wan25_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    resolution = _resolve_wan_resolution(requested, existing)
    size = existing.get("size")
    size_value = None
    if isinstance(size, str):
        normalized = size.strip().lower()
        for res, code in _WAN_SIZE_BY_RESOLUTION.items():
            if normalized == code.lower():
                size_value = code
                break
    if not size_value:
        size_value = _WAN_SIZE_BY_RESOLUTION.get(resolution, _WAN_SIZE_BY_RESOLUTION["1080p"])

    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 8, 10],
    )
    enable_expansion = _coerce_bool(
        requested.get("enable_prompt_expansion"),
        _coerce_bool(existing.get("enable_prompt_expansion"), True),
    )
    seed = _sanitize_seed(requested.get("seed"), existing.get("seed"))
    audio = _sanitize_audio(
        requested.get("audio") or requested.get("audio_url"),
        existing.get("audio"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "resolution": resolution,
        "size": size_value,
        "duration": duration,
        "enable_prompt_expansion": enable_expansion,
        "seed": seed,
    }
    if audio:
        payload["audio"] = audio
    return payload



def _handle_kling_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 8, 10],
    )
    guidance = _parse_float(requested.get("guidance_scale"))
    if guidance is None:
        guidance = _parse_float(existing.get("guidance_scale"))
    if guidance is None:
        guidance = 0.5
    guidance = max(0.0, min(1.0, guidance))

    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "guidance_scale": guidance,
        "duration": str(duration),
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if image:
        payload["image"] = image
    return payload





def _handle_kling25_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 10],
        default_bias=5,
    )
    guidance = _parse_float(requested.get("guidance_scale"))
    if guidance is None:
        guidance = _parse_float(existing.get("guidance_scale"))
    if guidance is None:
        guidance = 0.5
    guidance = max(0.0, min(1.0, guidance))

    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "guidance_scale": round(guidance, 4),
        "duration": str(duration or 5),
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if image:
        payload["image"] = image
    return payload



def _handle_kling25_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    aspect = _select_choice(
        requested.get("aspect_ratio"),
        existing.get("aspect_ratio"),
        _WAVESPEED_ASPECTS,
        "16:9",
    )
    duration = _compute_duration(
        requested.get("duration_seconds"),
        existing.get("duration"),
        [5, 10],
        default_bias=5,
    )
    guidance = _parse_float(requested.get("guidance_scale"))
    if guidance is None:
        guidance = _parse_float(existing.get("guidance_scale"))
    if guidance is None:
        guidance = 0.5
    guidance = max(0.0, min(1.0, guidance))

    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "aspect_ratio": aspect,
        "guidance_scale": round(guidance, 4),
        "duration": str(duration or 5),
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload



def _handle_kling_lipsync_a2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    audio = _sanitize_audio(
        requested.get("audio") or requested.get("audio_url"),
        existing.get("audio"),
    )
    video = _sanitize_string_field(
        requested.get("video") or requested.get("video_url"),
        existing.get("video"),
    )

    payload: Dict[str, Any] = {"model_id": canonical}
    if audio:
        payload["audio"] = audio
    if video:
        payload["video"] = video
    return payload



def _handle_kling_lipsync_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    voice_id = _sanitize_string_field(
        requested.get("voice_id"),
        existing.get("voice_id"),
    )
    voice_language = _sanitize_string_field(
        requested.get("voice_language"),
        existing.get("voice_language"),
    )

    voice_speed_val = _parse_float(requested.get("voice_speed"))
    if voice_speed_val is None:
        voice_speed_val = _parse_float(existing.get("voice_speed"))
    voice_speed = None
    if voice_speed_val is not None:
        voice_speed = max(0.1, min(4.0, round(voice_speed_val, 4)))
    else:
        raw_speed = requested.get("voice_speed") or existing.get("voice_speed")
        if isinstance(raw_speed, str) and raw_speed.strip():
            voice_speed = raw_speed.strip()

    guidance = _parse_float(requested.get("guidance_scale"))
    if guidance is None:
        guidance = _parse_float(existing.get("guidance_scale"))
    if guidance is not None:
        guidance = max(0.0, min(1.0, guidance))

    payload: Dict[str, Any] = {"model_id": canonical}
    if voice_id:
        payload["voice_id"] = voice_id
    if voice_language:
        payload["voice_language"] = voice_language
    if voice_speed is not None:
        payload["voice_speed"] = voice_speed
    if guidance is not None:
        payload["guidance_scale"] = round(guidance, 4)
    return payload



def _handle_kling_avatar_pro(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    guidance = _parse_float(requested.get("guidance_scale"))
    if guidance is None:
        guidance = _parse_float(existing.get("guidance_scale"))
    if guidance is not None:
        guidance = max(0.0, min(1.0, guidance))

    negative_prompt = _sanitize_negative_prompt(
        requested.get("negative_prompt"),
        existing.get("negative_prompt"),
    )
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )
    audio = _sanitize_audio(
        requested.get("audio") or requested.get("audio_url"),
        existing.get("audio"),
    )
    video = _sanitize_string_field(
        requested.get("video") or requested.get("video_url"),
        existing.get("video"),
    )
    voice_id = _sanitize_string_field(
        requested.get("voice_id"),
        existing.get("voice_id"),
    )
    voice_language = _sanitize_string_field(
        requested.get("voice_language"),
        existing.get("voice_language"),
    )

    voice_speed_val = _parse_float(requested.get("voice_speed"))
    if voice_speed_val is None:
        voice_speed_val = _parse_float(existing.get("voice_speed"))
    voice_speed = None
    if voice_speed_val is not None:
        voice_speed = max(0.1, min(4.0, round(voice_speed_val, 4)))
    else:
        raw_speed = requested.get("voice_speed") or existing.get("voice_speed")
        if isinstance(raw_speed, str) and raw_speed.strip():
            voice_speed = raw_speed.strip()

    avatar_id = _sanitize_string_field(
        requested.get("avatar_id"),
        existing.get("avatar_id"),
    )
    character_id = _sanitize_string_field(
        requested.get("character_id"),
        existing.get("character_id"),
    )

    duration_override = requested.get("duration") or existing.get("duration")
    if duration_override is None:
        duration_seconds = _parse_int(requested.get("duration_seconds"))
        if duration_seconds is None:
            duration_seconds = _parse_int(existing.get("duration_seconds"))
        duration_override = duration_seconds
    duration_value = None
    if isinstance(duration_override, (int, float)):
        duration_value = str(int(duration_override))
    elif isinstance(duration_override, str) and duration_override.strip():
        duration_value = duration_override.strip()

    payload: Dict[str, Any] = {"model_id": canonical}
    if guidance is not None:
        payload["guidance_scale"] = round(guidance, 4)
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if image:
        payload["image"] = image
    if audio:
        payload["audio"] = audio
    if video:
        payload["video"] = video
    if voice_id:
        payload["voice_id"] = voice_id
    if voice_language:
        payload["voice_language"] = voice_language
    if voice_speed is not None:
        payload["voice_speed"] = voice_speed
    if avatar_id:
        payload["avatar_id"] = avatar_id
    if character_id:
        payload["character_id"] = character_id
    if duration_value:
        payload["duration"] = duration_value
    return payload



def _handle_kling_effects(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    video = _sanitize_string_field(
        requested.get("video") or requested.get("video_url"),
        existing.get("video"),
    )
    effect_id = _sanitize_string_field(
        requested.get("effect_id"),
        existing.get("effect_id"),
    )
    effect_type = _sanitize_string_field(
        requested.get("effect_type"),
        existing.get("effect_type"),
    )

    effect_strength_val = _parse_float(requested.get("effect_strength"))
    if effect_strength_val is None:
        effect_strength_val = _parse_float(existing.get("effect_strength"))
    effect_strength = None
    if effect_strength_val is not None:
        effect_strength = max(0.0, min(10.0, round(effect_strength_val, 4)))
    else:
        raw_strength = requested.get("effect_strength") or existing.get("effect_strength")
        if isinstance(raw_strength, str) and raw_strength.strip():
            effect_strength = raw_strength.strip()

    duration_override = requested.get("duration") or existing.get("duration")
    duration_value = None
    if isinstance(duration_override, (int, float)):
        duration_value = str(int(duration_override))
    elif isinstance(duration_override, str) and duration_override.strip():
        duration_value = duration_override.strip()

    payload: Dict[str, Any] = {"model_id": canonical}
    if video:
        payload["video"] = video
    if effect_id:
        payload["effect_id"] = effect_id
    if effect_type:
        payload["effect_type"] = effect_type
    if effect_strength is not None:
        payload["effect_strength"] = effect_strength
    if duration_value:
        payload["duration"] = duration_value
    return payload

def _handle_hailuo_i2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    enable_expansion = _coerce_bool(
        requested.get("enable_prompt_expansion"),
        _coerce_bool(existing.get("enable_prompt_expansion"), True),
    )
    image = _sanitize_image(
        requested.get("image") or requested.get("image_url"),
        existing.get("image"),
    )

    payload: Dict[str, Any] = {
        "model_id": canonical,
        "enable_prompt_expansion": enable_expansion,
    }
    if canonical.endswith("i2v-standard"):
        duration = _compute_duration(
            requested.get("duration_seconds"),
            existing.get("duration"),
            [6, 10],
        )
        payload["duration"] = duration
    if image:
        payload["image"] = image
    return payload



def _handle_hailuo_t2v(
    canonical: str,
    requested: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    enable_expansion = _coerce_bool(
        requested.get("enable_prompt_expansion"),
        _coerce_bool(existing.get("enable_prompt_expansion"), True),
    )
    payload: Dict[str, Any] = {
        "model_id": canonical,
        "enable_prompt_expansion": enable_expansion,
    }
    if canonical.endswith("t2v-standard"):
        duration = _compute_duration(
            requested.get("duration_seconds"),
            existing.get("duration"),
            [6, 10],
        )
        payload["duration"] = duration
    return payload



# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------
_register_model("alibaba/wan-2.5/image-to-video", _handle_wan25_i2v, aliases=["wan-2.5-image-to-video", "wan25-i2v", "alibaba-wan-2.5-i2v"])
_register_model("alibaba/wan-2.5/image-to-video-fast", _handle_wan25_i2v, aliases=["wan-2.5-image-to-video-fast", "wan25-i2v-fast", "alibaba-wan-2.5-i2v-fast"])
_register_model("alibaba/wan-2.5/text-to-video", _handle_wan25_t2v, aliases=["wan-2.5-text-to-video", "wan25-t2v", "alibaba-wan-2.5-t2v"])
_register_model("alibaba/wan-2.5/text-to-video-fast", _handle_wan25_t2v, aliases=["wan-2.5-text-to-video-fast", "wan25-t2v-fast", "alibaba-wan-2.5-t2v-fast"])
_register_model("veo-2.0-generate-001", _handle_gemini_veo2, aliases=["gemini-veo2", "veo2", "veo-2"])
_register_model("wavespeed-ai/veo2-t2v", _handle_wavespeed_veo2_t2v, aliases=["veo2-t2v", "wavespeed-veo2-t2v"])
_register_model("wavespeed-ai/veo2-i2v", _handle_wavespeed_veo2_i2v, aliases=["veo2-i2v", "wavespeed-veo2-i2v"])
_register_model("google/veo3-fast/image-to-video", _handle_veo3_fast_i2v, aliases=["veo3-fast-image-to-video", "google-veo3-fast-i2v", "veo3-fast-i2v"])
_register_model("google/veo3/image-to-video", _handle_veo3_i2v, aliases=["veo3-image-to-video", "google-veo3-i2v", "veo3-i2v"])
_register_model("google/veo3-fast", _handle_veo3_fast, aliases=["veo3-fast", "google-veo3-fast"])
_register_model("google-veo3", _handle_veo3, aliases=["google/veo3", "veo3"])
_register_model("bytedance-seedance-v1-pro-t2v-720p", _handle_seedance_t2v, aliases=["seedance-t2v-720p", "seedance-pro-t2v"])
_register_model("bytedance-seedance-v1-pro-i2v-720p", _handle_seedance_i2v, aliases=["seedance-i2v-720p", "seedance-pro-i2v"])
_register_model("kwaivgi/kling-v2.1-i2v-pro", _handle_kling_i2v, aliases=["kling-v2.1-i2v-pro", "kling-pro"])
_register_model("kwaivgi/kling-v2.1-i2v-standard", _handle_kling_i2v, aliases=["kling-v2.1-i2v-standard", "kling-standard"])
_register_model("kwaivgi/kling-v2.1-i2v-master", _handle_kling_i2v, aliases=["kling-v2.1-i2v-master", "kling-master"])
_register_model("kwaivgi/kling-v2.5-turbo-pro/image-to-video", _handle_kling25_i2v, aliases=["kling-v2.5-turbo-pro-i2v", "kling-v2.5-i2v", "kling-25-i2v"])
_register_model("kwaivgi/kling-v2.5-turbo-pro/text-to-video", _handle_kling25_t2v, aliases=["kling-v2.5-turbo-pro-t2v", "kling-v2.5-t2v", "kling-25-t2v"])
_register_model("kwaivgi/kling-lipsync/audio-to-video", _handle_kling_lipsync_a2v, aliases=["kling-lipsync-a2v", "kling-lipsync-audio"])
_register_model("kwaivgi/kling-lipsync/text-to-video", _handle_kling_lipsync_t2v, aliases=["kling-lipsync-t2v", "kling-lipsync-text"])
_register_model("kwaivgi/kling-v1-ai-avatar-pro", _handle_kling_avatar_pro, aliases=["kling-ai-avatar-pro", "kling-avatar-pro"])
_register_model("kwaivgi/kling-effects", _handle_kling_effects, aliases=["kling-vfx", "kling-effects"])
_register_model("minimax/hailuo-02/i2v-pro", _handle_hailuo_i2v, aliases=["hailuo-02-i2v-pro"])
_register_model("minimax/hailuo-02/i2v-standard", _handle_hailuo_i2v, aliases=["hailuo-02-i2v-standard"])
_register_model("minimax/hailuo-02/t2v-pro", _handle_hailuo_t2v, aliases=["hailuo-02-t2v-pro"])
_register_model("minimax/hailuo-02/t2v-standard", _handle_hailuo_t2v, aliases=["hailuo-02-t2v-standard"])

__all__ = ["DEFAULT_MODEL", "MANAGED_KEYS", "normalize_generation_config", "resolve_canonical_model"]
