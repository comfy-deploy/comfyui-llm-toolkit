# generate_image.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Relative imports
try:
    from llmtoolkit_utils import tensor_to_base64, process_images_for_comfy, TENSOR_SUPPORT, get_api_key
    # Import helpers from send_request
    from send_request import run_async, send_request  # Added send_request for Gemini image generation
    # Import the specific API call function from root directory
    from api.openai_api import send_openai_image_generation_request
    # Import new Gemini image generation functions
    from api.gemini_image_api import send_gemini_image_generation_unified
    # Import new WaveSpeed image generation functions
    from api.wavespeed_image_api import (
        send_wavespeed_image_edit_request,
        send_wavespeed_seedream_request,
        send_wavespeed_hunyuan_request,
        send_wavespeed_qwen_edit_plus_request,
        send_wavespeed_imagen4_request,
        send_wavespeed_dreamina_request,
        send_wavespeed_nano_banana_edit_request,
    )
    # Import OpenRouter image generation
    from api.openrouter_api import send_openrouter_image_generation_request
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed relative imports in generate_image.py. Check file structure and __init__.py.")
    # Fallback to absolute imports if run standalone or structure differs
    try:
        from llmtoolkit_utils import tensor_to_base64, process_images_for_comfy, TENSOR_SUPPORT, get_api_key
        from send_request import run_async
        from api.openai_api import send_openai_image_generation_request
        from api.gemini_image_api import send_gemini_image_generation_unified
        from api.wavespeed_image_api import (
            send_wavespeed_image_edit_request,
            send_wavespeed_seedream_request,
            send_wavespeed_hunyuan_request,
            send_wavespeed_qwen_edit_plus_request,
            send_wavespeed_imagen4_request,
            send_wavespeed_dreamina_request,
            send_wavespeed_nano_banana_edit_request,
        )
    except ImportError:
        logging.error("Failed to import required modules for generate_image.py")
        TENSOR_SUPPORT = False

logger = logging.getLogger(__name__)

# Attempt to import ComfyUI's PreviewImage helper for temporary previews. Fallback gracefully if not available.
try:
    from nodes import PreviewImage  # ComfyUI builtin located at ComfyUI/nodes.py
except Exception:
    PreviewImage = None

# Helper to extract dict from payload
from context_payload import extract_context

class GenerateImage:
    """
    Generates an image using configuration and prompt details from the context.
    Currently supports OpenAI DALL-E/GPT-Image providers.
    Outputs the generated image(s) as a ComfyUI IMAGE tensor and updates the context.
    """

    DEFAULT_PROVIDER = "wavespeed"
    DEFAULT_MODEL = "wavespeed-ai/flux-kontext-dev-ultra-fast"
    # Default prompt shown in the node UI when no context is provided
    DEFAULT_PROMPT = (
        """
A stunning, professional-quality portrait of a character with rainbow-colored short curly hair.
"""
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Direct prompt input (overridden if context provides one)
                "prompt": ("STRING", {"multiline": True, "default": cls.DEFAULT_PROMPT}),
                "mode": (["generate", "edit", "variation"], {"default": "generate", "tooltip": "Choose the type of image request."}),
            },
            "optional": {
                "context": ("*", {}),  # Combined config and prompt info (optional)
            }
        }

    RETURN_TYPES = ("*", "IMAGE",) # Keep IMAGE for direct use/preview
    RETURN_NAMES = ("context", "image",)
    FUNCTION = "generate"
    CATEGORY = "🔗llm_toolkit/generators"
    OUTPUT_NODE = True

    def generate(self, prompt: str, mode: str = "generate", context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Any]:
        """
        Extracts config, calls the appropriate API, processes the response.
        """
        logger.info("GenerateImage node executing...")

        # Allow context to be optional; unwrap from ContextPayload if provided
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = extract_context(context)
            if not isinstance(context, dict):
                context = {}

        # --- Extract Configurations ---
        provider_config = context.get("provider_config", {})
        prompt_config = context.get("prompt_config", {})
        generation_config = context.get("generation_config", {})
        banana_config = context.get("banana_config", {}) # Extract banana_config

        # Debug logging
        logger.debug(f"Provider config: {provider_config}")
        logger.debug(f"Generation config keys: {list(generation_config.keys())}")

        # --- Determine Provider and Model ---
        llm_provider = provider_config.get("provider_name", self.DEFAULT_PROVIDER).lower()
        llm_model = provider_config.get("llm_model", "")

        # If no model in provider_config, check generation_config or use defaults
        if not llm_model:
            llm_model = generation_config.get("model", "")
            
        # If still no model, use provider-specific defaults
        if not llm_model:
            if llm_provider == "openai":
                llm_model = "dall-e-3" 
            elif llm_provider == "bfl":
                llm_model = "flux-kontext-max"
            elif llm_provider in {"gemini", "google"}:
                llm_model = "gemini-2.5-flash-image-preview"  # Latest Gemini image generation model
            elif llm_provider == "wavespeed":
                llm_model = self.DEFAULT_MODEL # Use the class default

        logger.info(f"Using provider: {llm_provider}, model: {llm_model}")

        # --- Get API Key ---
        api_key = provider_config.get("api_key", "")
        
        # Check for context-based API keys (higher priority than provider config)
        context_api_keys = context.get("api_keys", {})
        if context_api_keys:
            # Try exact provider match first
            context_key = context_api_keys.get(llm_provider)
            # Handle google/gemini alias
            if not context_key and llm_provider == "google":
                context_key = context_api_keys.get("gemini")
            elif not context_key and llm_provider == "gemini":
                context_key = context_api_keys.get("google")
            
            if context_key and context_key.strip():
                api_key = context_key.strip()
                # Secure logging: only show first 5 characters
                masked_key = api_key[:5] + "..." if len(api_key) > 5 else "..."
                logger.info(f"GenerateImage: Using context API key for {llm_provider} ({masked_key})")

        # If API key is missing or placeholder, attempt automatic resolution via utils.get_api_key
        if (
            not api_key or api_key in {"1234", "", None}
        ) and llm_provider in {"openai", "bfl", "gemini", "google", "wavespeed", "openrouter"}:
            env_var_name = {
                "openai": "OPENAI_API_KEY",
                "bfl": "BFL_API_KEY", 
                "gemini": "GEMINI_API_KEY",
                "google": "GEMINI_API_KEY",
                "wavespeed": "WAVESPEED_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }.get(llm_provider, "")
            
            if env_var_name:
                try:
                    api_key = get_api_key(env_var_name, llm_provider)
                    logger.info(
                        "GenerateImage: Retrieved API key for %s via get_api_key helper.",
                        llm_provider,
                    )
                except ValueError as _e:
                    logger.warning("GenerateImage: get_api_key failed – %s", _e)

        # After retries, ensure we have a usable key for providers that need one
        if llm_provider in {"openai", "bfl", "gemini", "google", "wavespeed", "openrouter"} and (not api_key or api_key == "1234"):
            logger.error(f"GenerateImage: Missing API key for provider '{llm_provider}'.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = f"Missing API key for {llm_provider}"

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            # Ensure UI dict is properly formatted
            if not isinstance(ui_dict, dict):
                ui_dict = {}
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Get Prompt Details ---
        # Get system message from banana_config
        system_message = banana_config.get("system_message", "")

        # Choose the prompt text: prefer context's prompt_config if present, otherwise use the direct node input
        user_prompt = prompt_config.get("text") or prompt

        # Combine system message and user prompt
        if system_message:
            prompt_text = f"{system_message.strip()}\\n\\n{user_prompt.strip()}"
            logger.info("GenerateImage: Combined system message from banana_config with user prompt.")
        else:
            prompt_text = user_prompt
        
        image_b64 = prompt_config.get("image_base64", None) # From PromptManager
        mask_b64 = prompt_config.get("mask_base64", None)   # From PromptManager

        # Handle multiple images for batch or list inputs
        if isinstance(image_b64, list) and len(image_b64) > 0:
            # For providers that support only single image, use the first one
            primary_image_b64 = image_b64[0]
            all_images_b64 = image_b64
            logger.info(f"GenerateImage: Found {len(image_b64)} input images from context")
        else:
            primary_image_b64 = image_b64
            all_images_b64 = [image_b64] if image_b64 else []
            if primary_image_b64:
                logger.info("GenerateImage: Found 1 input image from context")
            else:
                logger.info("GenerateImage: No input images found in context")

        if not prompt_text and not primary_image_b64:
            logger.error("GenerateImage: Requires at least a text prompt or an input image.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Missing text prompt or input image"

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            # Ensure UI dict is properly formatted
            if not isinstance(ui_dict, dict):
                ui_dict = {}
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Get Generation Settings (with defaults) ---
        n = generation_config.get("n", 1)
        size = generation_config.get("size")  # could be None
        response_format = generation_config.get("response_format", "b64_json")
        user = generation_config.get("user", None)

        # Model-specific settings
        quality = None
        style = None
        background = None
        output_format_gpt = None
        output_compression_gpt = None
        moderation_gpt = None

        if llm_model == "dall-e-3":
            quality = generation_config.get("quality_dalle3", "standard")
            style = generation_config.get("style_dalle3", "vivid")
        elif llm_model == "gpt-image-1":
            quality = generation_config.get("quality_gpt", "auto")
            background = generation_config.get("background_gpt", "auto")
            output_format_gpt = generation_config.get("output_format_gpt", "png")
            output_compression_gpt = generation_config.get("output_compression_gpt", None) # Default handled by API
            moderation_gpt = generation_config.get("moderation_gpt", "auto")
            response_format = "b64_json" # Override for gpt-image-1
        elif llm_model == "dall-e-2":
            quality = "standard" # Only option

        # --- Determine Mode (Generate, Edit, Variation) from dropdown ---
        mode = str(mode).lower().strip()
        if mode not in {"generate", "edit", "variation"}:
            logger.warning("GenerateImage: Unknown mode '%s', defaulting to 'generate'.", mode)
            mode = "generate"

        edit_mode = mode == "edit"
        variation_mode = mode == "variation"

        # Provider-specific mode validation
        if llm_provider in {"gemini", "google"} and llm_model.startswith("imagen"):
            # Imagen models only support generation
            if edit_mode or variation_mode:
                logger.warning(f"Imagen models only support 'generate' mode. Switching from '{mode}' to 'generate'.")
                mode = "generate"
                edit_mode = False
                variation_mode = False
        
        if llm_provider == "wavespeed" and llm_model in {"bytedance/seededit-v3", "bytedance/portrait", "wavespeed-ai/qwen-image/edit-plus"}:
            if mode != "edit":
                logger.warning(f"{llm_model} only supports 'edit' mode. Forcing 'edit' mode.")
                mode = "edit"
                edit_mode = True
                variation_mode = False

        if llm_provider == "wavespeed" and llm_model == "wavespeed-ai/hunyuan-image-3" and (edit_mode or variation_mode):
            logger.warning("wavespeed-ai/hunyuan-image-3 only supports 'generate' mode. Forcing 'generate'.")
            mode = "generate"
            edit_mode = False
            variation_mode = False

        # If in edit mode with image input and no explicit size passed, choose size based on image dims
        if edit_mode and primary_image_b64 and not size:
            from llmtoolkit_utils import get_dims_from_base64, choose_openai_size
            dims = get_dims_from_base64(primary_image_b64)
            if dims:
                w, h = dims
                size = choose_openai_size(w, h, llm_model)
                logger.info(f"GenerateImage: Auto-selected size '{size}' for edit request based on input image {w}x{h}.")

        # Fallback default if still None
        if not size:
            size = "1024x1024"

        # Validate requirements for chosen mode
        if edit_mode and not primary_image_b64:
            logger.error("GenerateImage: 'edit' mode requires at least one reference image.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Edit mode requires both image and mask"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # Mask requirement specific for DALL·E 2 edits
        if edit_mode and llm_model == "dall-e-2" and not mask_b64:
            logger.error("GenerateImage: 'edit' mode with dall-e-2 requires a mask.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "DALL·E 2 edit requires a mask"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        if variation_mode and not primary_image_b64:
            logger.error("GenerateImage: 'variation' mode selected but input image missing.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Variation mode requires an input image"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Call API ---
        output_image_tensor = None
        raw_api_response = None
        error_message = None

        try:
            if llm_provider == "openai":
                logger.info(f"Calling OpenAI Image API (model: {llm_model}, mode: {'edit' if edit_mode else 'variation' if variation_mode else 'generate'})")
                # Use run_async to call the async API function from sync context
                raw_api_response = run_async(
                    send_openai_image_generation_request(
                        api_key=api_key,
                        model=llm_model,
                        prompt=prompt_text,
                        n=n,
                        size=size,
                        quality=quality,
                        style=style,
                        response_format=response_format,
                        user=user,
                        background=background,
                        output_format_gpt=output_format_gpt,
                        output_compression_gpt=output_compression_gpt,
                        moderation_gpt=moderation_gpt,
                        image_base64=primary_image_b64,
                        mask_base64=mask_b64,
                        edit_mode=edit_mode,
                        variation_mode=variation_mode
                    )
                )

                # --- Process Response ---
                if raw_api_response and raw_api_response.get("data"):
                    logger.info(f"Received {len(raw_api_response['data'])} image(s) from API.")
                    # Use the utility to convert API response (list of {'b64_json':...} or {'url':...}) to ComfyUI tensor
                    output_image_tensor, _ = process_images_for_comfy(raw_api_response) # Discard mask from util
                else:
                    error_message = "API response did not contain expected image data."
                    logger.error(f"{error_message} Response: {raw_api_response}")

            elif llm_provider in {"gemini", "google"}:
                # ------------------------------------------------------------------
                #  Gemini / Imagen – unified image generation
                # ------------------------------------------------------------------
                logger.info(f"Calling Gemini/Imagen API (model: {llm_model}, mode: {mode})")
                
                # Collect all relevant parameters from generation_config
                kwargs = {
                    "temperature": generation_config.get("temperature_gemini", 0.7),
                    "max_tokens": generation_config.get("max_tokens_gemini", 8192),
                    "person_generation": generation_config.get("person_generation", "allow_adult"),
                    "safety_filter_level": generation_config.get("safety_filter_level", "block_medium_and_above"),
                    "language": generation_config.get("language"),
                }
                
                # Get seed
                seed = generation_config.get("seed", None)
                
                # Get aspect ratio (prefer explicit over size conversion)
                aspect_ratio = generation_config.get("aspect_ratio", None)
                
                # Pass input images to Gemini if they exist (for any mode including generate)
                input_images_to_pass = all_images_b64 if all_images_b64 and all_images_b64[0] else None
                
                logger.info(f"GenerateImage: Passing {len(all_images_b64) if all_images_b64 else 0} images to Gemini API")
                
                raw_api_response = run_async(
                    send_gemini_image_generation_unified(
                        api_key=api_key,
                        model=llm_model,
                        prompt=prompt_text,
                        n=n,
                        size=size,
                        aspect_ratio=aspect_ratio,
                        seed=seed,
                        edit_mode=edit_mode,
                        variation_mode=variation_mode,
                        input_image_base64=input_images_to_pass,
                        mask_base64=mask_b64,
                        **kwargs
                    )
                )
                
                if raw_api_response and raw_api_response.get("data"):
                    logger.info(f"GenerateImage: Processing Gemini API response with {len(raw_api_response['data'])} image(s)")
                    output_image_tensor, _ = process_images_for_comfy(raw_api_response)
                    
                    # Validate the processed tensor
                    if output_image_tensor is not None:
                        logger.info(f"GenerateImage: Successfully processed tensor shape: {output_image_tensor.shape}")
                    else:
                        error_message = "Failed to process Gemini image data into valid tensor"
                        logger.error(error_message)
                else:
                    error_message = "Gemini/Imagen API response did not contain expected image data."
                    logger.error("%s Response: %s", error_message, raw_api_response)

            elif llm_provider == "openrouter":
                # ------------------------------------------------------------------
                #  OpenRouter Provider (via chat completions)
                # ------------------------------------------------------------------
                logger.info(f"Calling OpenRouter API for image generation (model: {llm_model})")

                # Pass input images to OpenRouter if they exist
                input_images_to_pass = all_images_b64 if all_images_b64 and all_images_b64[0] else None

                # Collect relevant params from generation_config
                kwargs = {
                    "n": generation_config.get("n", 1),
                    "size": generation_config.get("size"),
                    "seed": generation_config.get("seed"),
                }
                # Filter out None values
                kwargs = {k: v for k, v in kwargs.items() if v is not None}

                raw_api_response = run_async(
                    send_openrouter_image_generation_request(
                        api_key=api_key,
                        model=llm_model,
                        prompt=prompt_text,
                        input_image_base64=input_images_to_pass,
                        **kwargs
                    )
                )

                if raw_api_response and raw_api_response.get("data"):
                    logger.info(f"GenerateImage: Processing OpenRouter API response with {len(raw_api_response['data'])} image(s)")
                    output_image_tensor, _ = process_images_for_comfy(raw_api_response)
                else:
                    error_message = "OpenRouter API response did not contain expected image data."
                    logger.error("%s Response: %s", error_message, raw_api_response)

            elif llm_provider == "wavespeed":
                # ------------------------------------------------------------------
                #  WaveSpeed Provider (various models)
                # ------------------------------------------------------------------
                if llm_model in {"bytedance/seededit-v3", "bytedance/portrait"}:
                    from api.wavespeed_image_api import send_wavespeed_image_edit_request
                    # This model is specifically for editing
                    if not edit_mode:
                        logger.error(f"WaveSpeed model {llm_model} only supports 'edit' mode.")
                        error_message = f"WaveSpeed model {llm_model} only supports 'edit' mode."
                    elif not primary_image_b64:
                        error_message = "Edit mode with WaveSpeed requires an input image."
                    else:
                        seed = generation_config.get("seed", -1)
                        guidance_scale = None
                        if llm_model == "bytedance/seededit-v3":
                            guidance_scale = generation_config.get("guidance_scale", 0.5)

                        logger.info(f"Calling WaveSpeed model {llm_model}...")
                        raw_api_response = run_async(
                            send_wavespeed_image_edit_request(
                                api_key=api_key,
                                model=llm_model,
                                prompt=prompt_text,
                                image_base64=primary_image_b64,
                                guidance_scale=guidance_scale,
                                seed=seed,
                            )
                        )
                elif llm_model == "wavespeed-ai/qwen-image/edit-plus":
                    images_to_pass = [img for img in all_images_b64 if img]
                    if not images_to_pass:
                        error_message = "Qwen Image Edit Plus requires at least one input image."
                    else:
                        if len(images_to_pass) > 3:
                            logger.warning("Qwen Image Edit Plus supports up to 3 reference images. Truncating list.")
                            images_to_pass = images_to_pass[:3]
                        logger.info("Calling WaveSpeed Qwen Image Edit Plus...")
                        raw_api_response = run_async(
                            send_wavespeed_qwen_edit_plus_request(
                                api_key=api_key,
                                model=llm_model,
                                prompt=prompt_text,
                                images_base64=images_to_pass,
                                size=generation_config.get("size"),
                                seed=generation_config.get("seed", -1),
                                output_format=generation_config.get("output_format")
                            )
                        )

                elif llm_model in {
                    "wavespeed-ai/flux-kontext-dev/multi-ultra-fast",
                    "wavespeed-ai/flux-kontext-dev-ultra-fast"
                }:
                    from api.wavespeed_image_api import send_wavespeed_flux_request

                    # This model supports generate, edit, and variation based on inputs
                    # Determine which image parameter to use
                    image_param = None
                    images_param = None
                    
                    is_multi = "multi" in llm_model
                    if is_multi:
                        images_param = all_images_b64 if (edit_mode or variation_mode) and all_images_b64 else None
                    else:
                        image_param = primary_image_b64 if (edit_mode or variation_mode) and primary_image_b64 else None

                    if (edit_mode or variation_mode) and not image_param and not images_param:
                        error_message = f"Mode '{mode}' requires an input image for the Flux model."
                    else:
                        params = {
                            "api_key": api_key,
                            "model": llm_model,
                            "prompt": prompt_text,
                            "image_base64": image_param,
                            "images_base64": images_param,
                            "size": generation_config.get("size"),
                            "num_inference_steps": generation_config.get("num_inference_steps", 28),
                            "guidance_scale": generation_config.get("guidance_scale", 2.5),
                            "num_images": generation_config.get("n", 1),
                            "seed": generation_config.get("seed", -1),
                            "enable_safety_checker": generation_config.get("enable_safety_checker", True),
                        }
                        logger.info(f"Calling WaveSpeed Flux model ({llm_model}) with mode: {mode}...")
                        raw_api_response = run_async(send_wavespeed_flux_request(**params))
                elif llm_model == "wavespeed-ai/hunyuan-image-3":
                    if edit_mode or variation_mode:
                        error_message = "Hunyuan Image 3.0 only supports generate mode."
                    else:
                        raw_api_response = run_async(
                            send_wavespeed_hunyuan_request(
                                api_key=api_key,
                                model=llm_model,
                                prompt=prompt_text,
                                size=generation_config.get("size"),
                                seed=generation_config.get("seed", -1),
                                output_format=generation_config.get("output_format")
                            )
                        )

                elif llm_model.startswith("bytedance/seedream"):
                    logger.info(f"Calling WaveSpeed Seedream model {llm_model}...")

                    is_edit = "edit" in llm_model
                    if is_edit and not all_images_b64:
                        error_message = f"WaveSpeed model {llm_model} requires an input image."
                    else:
                        raw_api_response = run_async(
                            send_wavespeed_seedream_request(
                                api_key=api_key,
                                model=llm_model,
                                prompt=prompt_text,
                                images_base64=all_images_b64 if is_edit else None,
                                size=generation_config.get("size"),
                                seed=generation_config.get("seed", -1),
                                max_images=generation_config.get("n", 1),
                            )
                        )
                elif llm_model in {
                    "google/imagen4",
                    "google/imagen4-fast",
                    "google/imagen4-ultra",
                    "wavespeed/imagen4",
                    "wavespeed/imagen4-fast",
                    "wavespeed/imagen4-ultra",
                }:
                    if edit_mode or variation_mode:
                        error_message = "Imagen 4 models only support generate mode."
                    else:
                        api_model = llm_model
                        if api_model.startswith("wavespeed/imagen4"):
                            suffix = api_model.replace("wavespeed/", "")
                            api_model = f"google/{suffix}"

                        raw_api_response = run_async(
                            send_wavespeed_imagen4_request(
                                api_key=api_key,
                                model=api_model,
                                prompt=prompt_text,
                                aspect_ratio=generation_config.get("aspect_ratio"),
                                resolution=generation_config.get("resolution"),
                                num_images=generation_config.get("n", 1),
                                seed=generation_config.get("seed", -1),
                                negative_prompt=None,
                            )
                        )

                elif llm_model in {
                    "bytedance/dreamina-v3.1/text-to-image",
                    "wavespeed/dreamina-v3.1",
                }:
                    if edit_mode or variation_mode:
                        error_message = "Dreamina V3.1 only supports generate mode."
                    else:
                        api_model = (
                            "bytedance/dreamina-v3.1/text-to-image"
                            if llm_model == "wavespeed/dreamina-v3.1"
                            else llm_model
                        )
                        raw_api_response = run_async(
                            send_wavespeed_dreamina_request(
                                api_key=api_key,
                                model=api_model,
                                prompt=prompt_text,
                                size=generation_config.get("size"),
                                seed=generation_config.get("seed", -1),
                                prompt_expansion=generation_config.get(
                                    "prompt_expansion",
                                    generation_config.get("prompt_enhancement", True),
                                ),
                            )
                        )

                elif llm_model in {
                    "google/nano-banana/edit",
                    "wavespeed/nano-banana-edit",
                }:
                    if not edit_mode:
                        error_message = "Nano Banana edit requires 'edit' mode."
                    elif not primary_image_b64:
                        error_message = "Nano Banana edit requires an input image."
                    else:
                        api_model = (
                            "google/nano-banana/edit"
                            if llm_model == "wavespeed/nano-banana-edit"
                            else llm_model
                        )
                        raw_api_response = run_async(
                            send_wavespeed_nano_banana_edit_request(
                                api_key=api_key,
                                model=api_model,
                                prompt=prompt_text.strip() or None,
                                image_base64=primary_image_b64,
                                output_format=generation_config.get("output_format", "png"),
                            )
                        )
                else:
                    error_message = f"The WaveSpeed model '{llm_model}' is not supported by the GenerateImage node yet."

                # Common response processing for all wavespeed models
                if not error_message:
                    if raw_api_response and raw_api_response.get("data"):
                        output_image_tensor, _ = process_images_for_comfy(raw_api_response)
                    else:
                        error_message = "WaveSpeed API response did not contain expected image data."
                        if raw_api_response and raw_api_response.get("error"):
                            error_message += f" Details: {raw_api_response.get('details', raw_api_response['error'])}"
                        logger.error(f"{error_message} Response: {raw_api_response}")

            elif llm_provider == "bfl":
                # ------------------------------------------------------------------
                #  BFL (Flux Kontext MAX) provider
                # ------------------------------------------------------------------
                from api.bfl_api import send_bfl_image_generation_request

                # Convert OpenAI-style size (e.g. "1024x1024") to aspect ratio "1:1"
                def _size_to_aspect(sz: str) -> str:
                    try:
                        w, h = [int(x) for x in sz.lower().split("x")]
                        # Reduce fraction to smallest integers (approx.)
                        import math

                        g = math.gcd(w, h)
                        return f"{w//g}:{h//g}"
                    except Exception:
                        return "1:1"

                # Allow explicit aspect_ratio override from generation_config
                aspect_ratio = generation_config.get("aspect_ratio") or _size_to_aspect(size or "1024x1024")

                # Optional advanced params
                seed_bfl = generation_config.get("seed")
                prompt_upsampling = generation_config.get("prompt_upsampling", False)
                safety_tolerance = generation_config.get("safety_tolerance", 2)
                output_format_bfl = generation_config.get("output_format_bfl")  # 'jpeg' or 'png'

                logger.info(
                    "Calling BFL Flux Kontext MAX (aspect_ratio=%s, edit=%s)",
                    aspect_ratio,
                    edit_mode,
                )

                raw_api_response = run_async(
                    send_bfl_image_generation_request(
                        api_key=api_key,
                        prompt=prompt_text,
                        aspect_ratio=aspect_ratio,
                        input_image_base64=primary_image_b64 if edit_mode else None,
                        seed=seed_bfl,
                        prompt_upsampling=prompt_upsampling,
                        safety_tolerance=safety_tolerance,
                        output_format=output_format_bfl,
                    )
                )

                if raw_api_response and raw_api_response.get("data"):
                    output_image_tensor, _ = process_images_for_comfy(raw_api_response)
                else:
                    error_message = "BFL API response did not contain expected image data."
                    logger.error(f"{error_message} Response: {raw_api_response}")

            else:
                error_message = (
                    f"Provider '{llm_provider}' not currently supported by GenerateImage node."
                )
                logger.error(error_message)

        except Exception as e:
            error_message = f"Error during image generation API call: {str(e)}"
            logger.error(error_message, exc_info=True)

        # --- Update Context and Return ---
        output_context = context.copy() # Work on a copy

        if error_message:
            output_context["error"] = error_message
        if raw_api_response:
            output_context["image_generation_response"] = raw_api_response # Store raw response

        if output_image_tensor is not None:
            output_context["generated_image_tensor"] = output_image_tensor # Store tensor in context
            logger.info("GenerateImage node finished successfully.")

            # ---------------------------------------------
            # Prepare preview image for ComfyUI UI panel
            # ---------------------------------------------
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    # Validate tensor before passing to preview
                    if output_image_tensor is not None and hasattr(output_image_tensor, 'shape'):
                        logger.info(f"GenerateImage: Creating preview for tensor shape: {output_image_tensor.shape}")
                        preview_node = PreviewImage()
                        preview_res = preview_node.save_images(output_image_tensor, filename_prefix="GenerateImage")
                        ui_dict = preview_res.get("ui", {})
                    else:
                        logger.warning("GenerateImage: Invalid tensor for preview, skipping UI preview")
                except Exception as e:
                    logger.warning("GenerateImage: Failed to create preview image – %s", e, exc_info=True)
                    # Clear UI dict to prevent invalid data from being passed
                    ui_dict = {}

            # Ensure UI dict is properly formatted
            if not isinstance(ui_dict, dict):
                ui_dict = {}
            
            return {"ui": ui_dict, "result": (output_context, output_image_tensor,)}
        else:
            # Return placeholder image if generation failed
            logger.warning("GenerateImage node failed, returning placeholder image.")
            placeholder_img, _ = process_images_for_comfy(None)

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            return {"ui": ui_dict, "result": (output_context, placeholder_img,)}

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "GenerateImage": GenerateImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateImage": "Generate Image (🔗LLMToolkit)"
} 
