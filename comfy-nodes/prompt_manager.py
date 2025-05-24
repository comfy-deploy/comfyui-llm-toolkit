# prompt_manager.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple
import torch
from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT, ensure_rgba_mask, resize_mask_to_match_image

# Ensure parent directory is in path if running standalone for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Relative import of utilities
try:
    from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT
except ImportError:
    # Fallback for standalone execution or environment issues
    try:
        from ..llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT
    except ImportError:
        # Final fallback
        from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT

logger = logging.getLogger(__name__)

# Lazy loading for optional dependencies
cv2 = None
def get_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            logger.warning("cv2 not available. Video frame extraction disabled.")
    return cv2

class PromptManager:
    """
    Manages and structures prompt components (text, image, mask, paths)
    into a prompt_config dictionary within the main context object.
    Accepts various optional inputs and adds them to the context if provided.
    """
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "context": ("*", {}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        # Conditionally add image/mask inputs if torch is available
        if TENSOR_SUPPORT:
            inputs["optional"]["image"] = ("IMAGE",)
            inputs["optional"]["mask"] = ("MASK",)
            # Allow connecting video tensors (e.g., from LoadVideo nodes)
            inputs["optional"]["video"] = ("IMAGE",)
        else:
             logger.warning("PromptManager: Torch/Numpy/PIL not found. IMAGE and MASK inputs disabled.")

        # Add path inputs (always available)
        inputs["optional"]["audio_path"] = ("STRING", {"multiline": False, "default": ""})
        inputs["optional"]["file_path"] = ("STRING", {"multiline": False, "default": "", "placeholder": "/path/to/file1.pdf, /path/to/video.mp4"})
        inputs["optional"]["url"]       = ("STRING", {"multiline": False, "default": "", "placeholder": "https://example.com, https://foo.bar/audio.mp3"})

        return inputs

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "manage_prompt"
    CATEGORY = "llm_toolkit"

    def manage_prompt(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Assembles prompt components into a dictionary within the context.
        """
        logger.info("PromptManager executing...")

        # Initialize or copy the context
        if context is None:
            output_context = {}
            logger.debug("PromptManager: Initializing new context.")
        elif isinstance(context, dict):
            output_context = context.copy()
            logger.debug("PromptManager: Copied input context.")
        else:
            # Handle non-dict context input gracefully
            output_context = {"passthrough_data": context}
            logger.warning("PromptManager: Received non-dict context input. Wrapping it.")

        # Initialize prompt_config dictionary (get existing if present, else create)
        prompt_config = output_context.get("prompt_config", {})
        if not isinstance(prompt_config, dict):
            logger.warning("PromptManager: Existing 'prompt_config' in context is not a dict. Overwriting.")
            prompt_config = {}

        # Process optional inputs using kwargs
        text_prompt = kwargs.get("text_prompt", "").strip()
        image_tensor = kwargs.get("image", None)
        mask_tensor = kwargs.get("mask", None)
        video_tensor = kwargs.get("video", None)
        audio_path = kwargs.get("audio_path", "").strip()
        file_path_str = kwargs.get("file_path", "").strip()
        url_str       = kwargs.get("url", "").strip()

        if text_prompt:
            prompt_config["text"] = text_prompt
            logger.debug(f"PromptManager: Added text_prompt (length: {len(text_prompt)}).")

        # --- Helper to process tensor or list of tensors into b64 list ---
        def _tensor_or_list_to_b64(tensor_or_list, max_items: int = 16):
            """Return list of base64 strings given tensor or list of tensors."""
            b64_list = []
            if torch.is_tensor(tensor_or_list):
                if tensor_or_list.dim() == 4 and tensor_or_list.shape[0] > 1:
                    sample_count = min(tensor_or_list.shape[0], max_items)
                    for idx in range(sample_count):
                        b64 = tensor_to_base64(tensor_or_list[idx:idx+1], image_format="PNG")
                        if b64:
                            b64_list.append(b64)
                else:
                    b64 = tensor_to_base64(tensor_or_list, image_format="PNG")
                    if b64:
                        b64_list.append(b64)
            elif isinstance(tensor_or_list, list):
                sample_items = tensor_or_list[:max_items]
                for t in sample_items:
                    if torch.is_tensor(t):
                        # Ensure has batch dim
                        if t.dim() == 3:
                            t = t.unsqueeze(0)
                        b64 = tensor_to_base64(t, image_format="PNG")
                        if b64:
                            b64_list.append(b64)
            return b64_list

        # --- Handle IMAGE input (single, list, or batch tensor) ---
        if image_tensor is not None and TENSOR_SUPPORT:
            imgs_b64 = _tensor_or_list_to_b64(image_tensor)
            if imgs_b64:
                prompt_config["image_base64"] = imgs_b64 if len(imgs_b64) > 1 else imgs_b64[0]
                logger.debug(f"PromptManager: Added {len(imgs_b64)} image(s) to prompt_config.")
            else:
                logger.warning("PromptManager: Failed to convert provided image(s) to base64.")

        # --- Handle MASK tensor or batch ---
        if mask_tensor is not None and TENSOR_SUPPORT:
            logger.debug(f"PromptManager: Processing mask tensor with shape: {mask_tensor.shape}")

            def _prep_mask(t):
                # Ensure grayscale channel dim ==1
                if t.dim() == 4 and t.shape[-1] in [3, 4]:
                    t = t.mean(dim=-1, keepdim=True)
                elif t.dim() == 3:  # B H W
                    t = t.unsqueeze(-1)
                return t

            if mask_tensor.dim() == 4 and mask_tensor.shape[0] > 1:
                mask_list = []
                for idx in range(mask_tensor.shape[0]):
                    single_mask = _prep_mask(mask_tensor[idx:idx+1])
                    b64 = tensor_to_base64(single_mask, image_format="PNG")
                    if b64:
                        mask_list.append(b64)
                if mask_list:
                    prompt_config["mask_base64"] = mask_list
                    logger.debug(f"PromptManager: Added list of {len(mask_list)} base64 masks.")
                else:
                    logger.warning("PromptManager: Failed to convert batch masks to base64.")
            else:
                # Resize mask if needed to match image dimensions
                if image_tensor is not None and isinstance(image_tensor, torch.Tensor):
                    mask_tensor = resize_mask_to_match_image(mask_tensor, image_tensor)

                try:
                    # For OpenAI edits transparent areas (alpha=0) are replaced. In many
                    # ComfyUI masks the area to *edit* is white (1).  Therefore invert
                    # the mask so edited pixels get alpha=0.
                    inv_mask_tensor = 1.0 - mask_tensor.clamp(0, 1)
                    mask_tensor_rgba = ensure_rgba_mask(inv_mask_tensor)
                except Exception:
                    mask_tensor_rgba = mask_tensor  # Fallback

                mask_base64 = tensor_to_base64(mask_tensor_rgba, image_format="PNG")
                if mask_base64:
                    prompt_config["mask_base64"] = mask_base64
                    logger.debug("PromptManager: Added mask_base64.")
                else:
                    logger.warning("PromptManager: Failed to convert mask tensor to base64.")

        # --- Handle VIDEO tensor/batch/list ---
        # Video tensors from nodes (e.g., LoadVideo) are extracted to frames
        # for APIs like OpenAI that only support images.
        # This is different from video file paths which are kept intact.
        if video_tensor is not None and TENSOR_SUPPORT:
            logger.debug("PromptManager: Processing video tensor for frame extraction...")
            # Extract frames at intervals (every 16th frame, max 5 frames)
            extracted_frames = []
            
            if torch.is_tensor(video_tensor):
                # Video tensor shape is typically [frames, H, W, C]
                if video_tensor.dim() == 4:
                    frame_count = video_tensor.shape[0]
                    interval = 16
                    max_frames = 5
                    
                    for i in range(0, min(frame_count, interval * max_frames), interval):
                        frame = video_tensor[i:i+1]  # Keep batch dim
                        b64 = tensor_to_base64(frame, image_format="JPEG")
                        if b64:
                            extracted_frames.append(b64)
                    
                    logger.debug(f"PromptManager: Extracted {len(extracted_frames)} frames from video tensor (every {interval}th frame).")
                else:
                    # Single frame or unexpected shape - treat as image
                    b64 = tensor_to_base64(video_tensor, image_format="JPEG")
                    if b64:
                        extracted_frames.append(b64)
            
            if extracted_frames:
                prompt_config["video_frames_base64"] = extracted_frames
                logger.debug(f"PromptManager: Added {len(extracted_frames)} extracted video frame(s).")
            else:
                logger.warning("PromptManager: Failed to convert video input to base64.")

        if audio_path:
            prompt_config["audio_path"] = audio_path
            logger.debug(f"PromptManager: Added audio_path: {audio_path}")

        # --- Handle FILE PATHS (comma-separated) ---
        # Note: Video files (.mp4) in file_paths are NOT extracted to frames
        # because some APIs (Gemini, etc.) can process video files directly.
        # Only video tensors from nodes get frame extraction.
        if file_path_str:
            paths = [p.strip() for p in file_path_str.split(",") if p.strip()]
            if paths:
                prompt_config["file_paths"] = paths if len(paths) > 1 else paths[0]
                logger.debug(f"PromptManager: Added {len(paths)} file_path(s).")

        # --- Handle URLS (comma-separated) ---
        if url_str:
            urls = [u.strip() for u in url_str.split(",") if u.strip()]
            if urls:
                prompt_config["urls"] = urls if len(urls) > 1 else urls[0]
                logger.debug(f"PromptManager: Added {len(urls)} url(s).")

        # Update the main context with the (potentially updated) prompt_config
        if prompt_config: # Only add if not empty
             output_context["prompt_config"] = prompt_config
             logger.info("PromptManager: Updated context with prompt_config.")
        else:
             logger.info("PromptManager: No prompt components provided.")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager (LLMToolkit)"
} 