# comfy-nodes/generate_video.py
"""Generate Video using Veo 2 via Gemini API.

The node supports text-to-video for now. It polls the long-running
operation until completion and downloads the resulting MP4(s).
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from video_generation_capabilities import (
    MANAGED_KEYS,
    normalize_generation_config,
    resolve_canonical_model,
)

try:
    from llmtoolkit_utils import get_api_key
except ImportError:
    get_api_key = None  # type: ignore

try:
    import folder_paths
except ImportError:
    folder_paths = None  # type: ignore

logger = logging.getLogger(__name__)


class GenerateVideo:
    DEFAULT_MODEL = "veo-2.0-generate-001"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic shot of sunrise over mountains"}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "video_paths")
    FUNCTION = "generate"
    CATEGORY = "🔗llm_toolkit/generators"

    # ------------------------------------------------------------------
    def _save_videos(self, client, videos: List[Any], out_dir: Path, base_name: str) -> List[str]:
        paths: List[str] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, vid in enumerate(videos):
            fname = f"{base_name}_{idx}.mp4"
            tgt = out_dir / fname
            try:
                client.files.download(file=vid.video)  # ensures local cache
                vid.video.save(str(tgt))
                paths.append(str(tgt))
            except Exception as dl_exc:
                logger.error("Failed saving video %s – %s", fname, dl_exc, exc_info=True)
        return paths

    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: Optional[Any] = None):
        logger.info("GenerateVideo node executing…")

        # ------------ Context handling -------------
        if context is None:
            ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            ctx = context.copy()
        else:
            ctx = extract_context(context)
            if not isinstance(ctx, dict):
                ctx = {"passthrough_data": context}

        provider_cfg = ctx.get("provider_config", {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}

        gen_cfg = ctx.get("generation_config", {})
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}

        model_hint = resolve_canonical_model(
            provider_cfg.get("llm_model")
            or gen_cfg.get("model_id")
            or self.DEFAULT_MODEL
        )
        canonical_model, sanitized_cfg = normalize_generation_config(
            model_hint,
            existing=gen_cfg,
        )

        for key in list(gen_cfg.keys()):
            if key in MANAGED_KEYS and key not in sanitized_cfg:
                gen_cfg.pop(key)

        gen_cfg.update(sanitized_cfg)
        ctx["generation_config"] = gen_cfg
        logger.debug("GenerateVideo normalized config for %s: %s", canonical_model, sanitized_cfg)

        # Determine provider name (default gemini)
        llm_provider = provider_cfg.get("provider_name", "gemini").lower()

        # -------------------------------------------------------------
        # WaveSpeed branch
        # -------------------------------------------------------------
        if llm_provider == "wavespeed":
            # Resolve API key
            api_key = provider_cfg.get("api_key", "").strip()
            if (not api_key or api_key == "1234") and get_api_key is not None:
                try:
                    api_key = get_api_key("WAVESPEED_API_KEY", "wavespeed")
                except ValueError:
                    api_key = ""

            if not api_key:
                err = "GenerateVideo: missing WaveSpeed API key"
                logger.error(err)
                ctx["error"] = err
                return (ctx, "")

            # Determine model / endpoint
            model_id = gen_cfg.get("model_id") or provider_cfg.get("llm_model")
            if not model_id:
                err = "GenerateVideo: WaveSpeed model_id not specified"
                logger.error(err)
                ctx["error"] = err
                return (ctx, "")

            # Build endpoint path: if a slash is present, use the model_id
            # directly. Otherwise, assume "provider-model" format.
            if "/" in model_id:
                endpoint_path = model_id
            elif "-" in model_id:
                provider_seg, remainder = model_id.split("-", 1)
                endpoint_path = f"{provider_seg}/{remainder}"
            else:
                endpoint_path = model_id  # fallback

            base_url = "https://api.wavespeed.ai/api/v3/"
            post_url = base_url + endpoint_path

            # Build payload from generation_config (except model_id)
            payload: Dict[str, Any] = {k: v for k, v in gen_cfg.items() if k != "model_id"}

            # ---------------- Prompt precedence -----------------
            # 1) Check prompt_config from PromptManager (highest priority)
            prompt_cfg = ctx.get("prompt_config", {})
            prompt_text: str = ""
            if isinstance(prompt_cfg, dict) and prompt_cfg.get("text"):
                prompt_text = str(prompt_cfg["text"]).strip()

            # 2) Fallback to prompt string from node parameter
            if not prompt_text and prompt.strip():
                prompt_text = prompt.strip()

            if prompt_text:
                payload["prompt"] = prompt_text

            # ---------------- Image handling for I2V models -----------------
            # If the model expects an image (i2v) and the user supplied an
            # image tensor via PromptManager but didn't provide an image URL
            # manually, upload the first image to WaveSpeed and insert the
            # resulting download_url.

            if ("image" not in payload or not str(payload["image"]).strip()) and isinstance(prompt_cfg, dict):
                img_b64 = prompt_cfg.get("image_base64")
                if isinstance(img_b64, list):
                    img_b64 = img_b64[0] if img_b64 else None
                
                # Check for a non-empty base64 string.
                if isinstance(img_b64, str) and img_b64:
                    try:
                        from llmtoolkit_utils import base64_to_pil
                        import io, requests
                        from PIL import Image

                        pil_img = base64_to_pil(img_b64)
                        if pil_img:
                            buffered = io.BytesIO()
                            pil_img.save(buffered, format="PNG")
                            buffered.seek(0)

                            upload_url = "https://api.wavespeed.ai/api/v3/media/upload/binary"
                            files = {"file": ("image.png", buffered, "image/png")}
                            up_resp = requests.post(upload_url, headers={"Authorization": f"Bearer {api_key}"}, files=files, timeout=60)
                            if up_resp.status_code == 200:
                                up_data = up_resp.json()
                                if isinstance(up_data, dict) and up_data.get("code") == 200:
                                    dl_url = up_data["data"]["download_url"]
                                    payload["image"] = dl_url
                                    logger.info("Uploaded image to WaveSpeed – using returned URL: %s", dl_url)
                    except Exception as up_exc:
                        logger.warning("Image upload to WaveSpeed failed: %s", up_exc)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            try:
                import requests
                resp = requests.post(post_url, headers=headers, json=payload, timeout=60)
                if resp.status_code != 200:
                    err = f"WaveSpeed POST failed {resp.status_code}: {resp.text[:120]}"
                    logger.error(err)
                    ctx["error"] = err
                    return (ctx, "")

                result = resp.json().get("data", {})
                request_id = result.get("id")
                if not request_id:
                    err = "WaveSpeed API response missing request id"
                    logger.error(err)
                    ctx["error"] = err
                    return (ctx, "")

                # Poll loop
                poll_url = base_url + f"predictions/{request_id}/result"
                poll_headers = {"Authorization": f"Bearer {api_key}"}
                poll_interval = 5
                max_wait = 600
                waited = 0
                while waited < max_wait:
                    pr = requests.get(poll_url, headers=poll_headers, timeout=30)
                    if pr.status_code != 200:
                        err = f"WaveSpeed poll failed {pr.status_code}: {pr.text[:120]}"
                        logger.error(err)
                        ctx["error"] = err
                        return (ctx, "")
                    pr_data = pr.json().get("data", {})
                    status = pr_data.get("status")
                    if status == "completed":
                        outputs = pr_data.get("outputs", [])
                        if not outputs:
                            err = "WaveSpeed generation completed but no outputs returned"
                            logger.error(err)
                            ctx["error"] = err
                            return (ctx, "")

                        # Download videos to output folder
                        if folder_paths and hasattr(folder_paths, "get_output_directory"):
                            base_output_dir = folder_paths.get_output_directory()
                            out_dir = Path(base_output_dir) / "wavespeed_videos"
                        else:
                            out_dir = Path(os.getcwd()) / "output" / "wavespeed_videos"
                        out_dir.mkdir(parents=True, exist_ok=True)

                        paths: List[str] = []
                        for idx, url in enumerate(outputs):
                            try:
                                video_resp = requests.get(url, stream=True, timeout=120)
                                video_resp.raise_for_status()
                                fname = f"{model_id.replace('/', '_')}_{request_id}_{idx}.mp4"
                                tgt = out_dir / fname
                                with open(tgt, "wb") as f_out:
                                    for chunk in video_resp.iter_content(chunk_size=8192):
                                        if chunk:
                                            f_out.write(chunk)
                                paths.append(str(tgt))
                            except Exception as dl_exc:
                                logger.error("Failed downloading video %s – %s", url, dl_exc, exc_info=True)

                        ctx["generated_video_paths"] = paths if paths else outputs
                        logger.info("GenerateVideo (🔗LLMToolkit): saved %d video(s)", len(paths))
                        return (
                            ctx,
                            paths[0] if len(paths) == 1 and paths else "|".join(paths if paths else outputs),
                        )

                    elif status == "failed":
                        err = f"WaveSpeed generation failed: {pr_data.get('error', 'unknown error')}"
                        logger.error(err)
                        ctx["error"] = err
                        return (ctx, "")

                    time.sleep(poll_interval)
                    waited += poll_interval

                err = "WaveSpeed generation timed out"
                logger.error(err)
                ctx["error"] = err
                return (ctx, "")

            except Exception as exc:
                err = f"WaveSpeed API error: {exc}"
                logger.error(err, exc_info=True)
                ctx["error"] = err
                return (ctx, "")

        # -------------------------------------------------------------
        # Gemini / Veo branch (default)
        # -------------------------------------------------------------
        if llm_provider not in {"gemini", "google"}:
            err = f"GenerateVideo supports Gemini/Veo or WaveSpeed, provider '{llm_provider}' unsupported."
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        llm_model = provider_cfg.get("llm_model", self.DEFAULT_MODEL) or self.DEFAULT_MODEL

        # API key
        api_key = provider_cfg.get("api_key", "").strip()
        if (not api_key or api_key == "1234") and get_api_key is not None:
            try:
                api_key = get_api_key("GEMINI_API_KEY", "gemini")
            except ValueError:
                pass
        if not api_key:
            err = "GenerateVideo: missing Gemini API key"
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        # Import SDK
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            err = "google-generativeai not installed – run 'pip install google-generativeai'"
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        aspect_ratio = gen_cfg.get("aspect_ratio", "16:9")
        person_generation = gen_cfg.get("person_generation", "allow_adult")
        
        # Map unsupported "allow_all" to "allow_adult" for backward compatibility
        if person_generation == "allow_all":
            logger.warning("person_generation 'allow_all' is not supported by Veo API, using 'allow_adult' instead")
            person_generation = "allow_adult"
            
        number_of_videos = int(gen_cfg.get("number_of_videos", 1))
        duration_seconds = int(gen_cfg.get("duration_seconds", 8))
        negative_prompt = gen_cfg.get("negative_prompt", "")
        enhance_prompt = bool(gen_cfg.get("enhance_prompt", True))

        video_cfg = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt if negative_prompt else None,
            enhance_prompt=enhance_prompt,
        )

        try:
            client = genai.Client(api_key=api_key)
            operation = client.models.generate_videos(
                model=llm_model,
                prompt=prompt,
                config=video_cfg,
            )

            # Poll
            poll_sec = 20
            while not operation.done:
                logger.info("GenerateVideo: operation not done yet – sleeping %s sec", poll_sec)
                time.sleep(poll_sec)
                operation = client.operations.get(operation)

            videos = operation.response.generated_videos  # type: ignore
            if not videos:
                raise RuntimeError("Veo generation returned no videos")

            # Use ComfyUI's standard output directory
            if folder_paths and hasattr(folder_paths, 'get_output_directory'):
                base_output_dir = folder_paths.get_output_directory()
                out_dir = Path(base_output_dir) / "veo_videos"
            else:
                # Fallback to current directory if folder_paths not available
                out_dir = Path(os.getcwd()) / "output" / "veo_videos"
            
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving videos to: {out_dir}")
            
            base_name = f"veo_{os.getpid()}"
            paths = self._save_videos(client, videos, out_dir, base_name)
            if not paths:
                raise RuntimeError("Failed saving any generated videos")

            ctx["generated_video_paths"] = paths
            logger.info("GenerateVideo: saved %d video(s)", len(paths))
            # Return first path for convenience
            return (ctx, paths[0] if len(paths) == 1 else "|".join(paths))

        except Exception as exc:
            err = f"Veo API error: {exc}"
            logger.error(err, exc_info=True)
            ctx["error"] = err
            return (ctx, "")


NODE_CLASS_MAPPINGS = {"GenerateVideo": GenerateVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerateVideo": "Generate Video (🔗LLMToolkit)"} 