#send_request.py
import asyncio
import json
import logging
import base64
import os, sys  # Needed for dynamic path adjustment before importing transformers_provider
from typing import List, Union, Optional, Dict, Any

# Minimal imports for Ollama & OpenAI only
from ollama_api import send_ollama_request, create_ollama_embedding
from openai_api import (
    send_openai_request,
    generate_image,
    generate_image_variations,
    edit_image,
    create_openai_compatible_embedding,
)
from llmtoolkit_utils import convert_images_for_api, ensure_ollama_server, ensure_ollama_model

# Gemini helpers (OpenAI-compat layer)
from gemini_api import (
    send_gemini_request,
    send_gemini_image_generation_request,
    create_gemini_compatible_embedding,
)

# Optional: folder_paths may be used elsewhere but isn't necessary here —
# leave a harmless import to keep previous behaviour for callers that expect
# it to exist.
try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None

# Ensure comfy-nodes directory is on sys.path so we can import transformers_provider
_root_dir = os.path.dirname(os.path.abspath(__file__))
_comfy_nodes_dir = os.path.join(_root_dir, "comfy-nodes")
if _comfy_nodes_dir not in sys.path:
    sys.path.insert(0, _comfy_nodes_dir)

from transformers_provider import send_transformers_request  # NEW: local HF models

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_async(coroutine):
    """Helper function to run coroutines in a new event loop if necessary"""
    try:
        # Check if we received a valid coroutine
        if not asyncio.iscoroutine(coroutine):
            logger.error(f"run_async received non-coroutine object: {type(coroutine)}")
            return None
            
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the coroutine with proper error handling
        try:
            return loop.run_until_complete(coroutine)
        except Exception as e:
            logger.error(f"Error in run_async while executing coroutine: {str(e)}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in run_async: {str(e)}", exc_info=True)
        return None

async def send_request(
    llm_provider: str,
    base_ip: str,
    port: str,
    images: Optional[List] = None,
    llm_model: str = "",
    system_message: str = "",
    user_message: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    random: bool = False,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    stop: Optional[List[str]] = None,
    keep_alive: bool = False,
    llm_api_key: Optional[str] = None,
    strategy: str = "normal",
    batch_count: int = 1,
    mask: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Sends a request to the specified LLM provider and returns a unified response.
    
    Args:
        llm_provider (str): The LLM provider to use.
        base_ip (str): Base IP address for the API.
        port (int): Port number for the API.
        base64_images (List[str]): List of images encoded in base64.
        llm_model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        seed (Optional[int]): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        random (bool): Whether to use randomness.
        top_k (int): Top K for sampling.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        stop (Optional[List[str]]): Stop sequences.
        keep_alive (bool): Whether to keep the session alive.
        llm_api_key (Optional[str], optional): API key for the LLM provider.
        strategy (str): Strategy for image generation.
        batch_count (int): Number of images to generate.
        mask (Optional[str], optional): Mask for image editing.

    Returns:
        Union[str, Dict[str, Any]]: Unified response format.
    """
    # Added entry logging to track function execution
    logger.info(f"send_request started for provider: {llm_provider}, model: {llm_model}")
    
    # Validate essential parameters
    if not llm_provider:
        error_msg = "Missing required parameter: llm_provider"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": f"Error: {error_msg}"}}]}
    
    if not llm_model:
        error_msg = "Missing required parameter: llm_model"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": f"Error: {error_msg}"}}]}

    try:
        # Basic aspect‑ratio sizes for DALL·E endpoints
        aspect_ratio_mapping = {
            "1:1": "1024x1024",
            "4:5": "1024x1280",
            "3:4": "1024x1365",
            "5:4": "1280x1024",
            "16:9": "1600x900",
            "9:16": "900x1600",
        }
        size = aspect_ratio_mapping.get("1:1")

        formatted_images = []
        if images:
            formatted_images = convert_images_for_api(images, target_format="base64")

        # ------------------------------------------------------------------
        #  Ollama
        # ------------------------------------------------------------------
        if llm_provider == "ollama":
            # Lazily start local Ollama daemon and pull model if necessary
            if not ensure_ollama_server(base_ip, port):
                err = "Ollama daemon is not running and could not be started."
                logger.error(err)
                return {"choices": [{"message": {"content": err}}]}

            # Ensure the requested model is present locally
            ensure_ollama_model(llm_model, base_ip, port)

            api_url = f"http://{base_ip}:{port}/api/chat"  
            logger.info(f"Constructed Ollama API URL: {api_url}")
            kwargs = dict(
                base64_images=formatted_images,  
                model=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages or [],
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
                random=random,  
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop,
                keep_alive=keep_alive,
            )
            return await send_ollama_request(api_url, **kwargs)

        # ------------------------------------------------------------------
        #  Gemini (OpenAI-compat) – chat & Imagen 3 image generation
        # ------------------------------------------------------------------
        if llm_provider == "gemini":
            # Distinguish image requests (Imagen-3) versus text chat
            if llm_model.startswith("imagen") or llm_model.startswith("image"):
                # For now use same aspect ratio → size mapping as DALL·E
                size = aspect_ratio_mapping.get("1:1")
                try:
                    response = await send_gemini_image_generation_request(
                        api_key=llm_api_key,
                        model=llm_model,
                        prompt=user_message,
                        n=batch_count,
                        size=size,
                        response_format="b64_json",
                    )
                    return response  # structure matches OpenAI image response
                except Exception as exc:
                    logger.error(f"Gemini image generation error: {exc}", exc_info=True)
                    return {"error": str(exc)}
            else:
                # Text chat/completions path
                return await send_gemini_request(
                    api_url=None,
                    base64_images=formatted_images,
                    model=llm_model,
                    system_message=system_message,
                    user_message=user_message,
                    messages=messages or [],
                    api_key=llm_api_key or "",
                    seed=seed if random else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    tools=None,
                    tool_choice=None,
                )

        # ------------------------------------------------------------------
        #  OpenAI (chat + DALL·E)
        # ------------------------------------------------------------------
        if llm_provider == "openai":
            if llm_model.startswith("dall-e"):
                # Image‑generation branches
                if strategy == "create":
                    result_imgs = await generate_image(
                        prompt=user_message,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key,
                    )
                elif strategy == "edit":
                    base_img = formatted_images[0] if formatted_images else None
                    mask_b64 = convert_images_for_api(mask, target_format="base64")[0] if mask else None
                    result_imgs = await edit_image(
                        image_base64=base_img,
                        mask_base64=mask_b64,
                        prompt=user_message,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key,
                    )
                elif strategy == "variations":
                    base_img = formatted_images[0] if formatted_images else None
                    result_imgs = await generate_image_variations(
                        image_base64=base_img,
                        model=llm_model,
                        n=batch_count,
                        size=size,
                        api_key=llm_api_key,
                    )
                else:
                    return {"error": f"Unsupported strategy {strategy} for DALL·E"}

                return {"images": result_imgs}

            # Regular chat models
            api_url = "https://api.openai.com/v1/chat/completions"
            return await send_openai_request(
                api_url=api_url,
                base64_images=formatted_images,
                model=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages or [],
                api_key=llm_api_key,
                seed=seed if random else None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                tools=None,
                tool_choice=None,
            )

        # ------------------------------------------------------------------
        #  Local HuggingFace Transformers (offline)
        # ------------------------------------------------------------------
        if llm_provider in {"transformers", "hf", "local"}:
            return await send_transformers_request(
                base64_images=formatted_images,
                base64_audio=[],  # TODO: support audio if needed
                model=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages or [],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                precision="fp16",  # Could be param
            )

        return {"error": f"Unsupported llm_provider '{llm_provider}'"}

    except Exception as e:
        logger.error(f"Exception in send_request: {e}", exc_info=True)
        return {"error": str(e)}

def format_response(response, tools):
    """Helper function to format the response consistently"""
    if tools:
        return response
    try:
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return response
    except (KeyError, IndexError, TypeError) as e:
        error_msg = f"Error formatting response: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

async def create_embedding(embedding_provider: str, api_base: str, embedding_model: str, input: Union[str, List[str]], embedding_api_key: Optional[str] = None) -> Union[List[float], None]: # Correct return type hint
    if embedding_provider == "ollama":
        return await create_ollama_embedding(api_base, embedding_model, input)
    
    
    elif embedding_provider in ["openai", "lmstudio", "llamacpp", "textgen", "mistral", "xai"]:
        try:
            return await create_openai_compatible_embedding(api_base, embedding_model, input, embedding_api_key)
        except ValueError as e:
            print(f"Error creating embedding: {e}")
            return None
    
    elif embedding_provider == "gemini":
        try:
            return await create_gemini_compatible_embedding(api_base, embedding_model, input, embedding_api_key)
        except ValueError as e:
            print(f"Error creating embedding: {e}")
            return None
    
    else:
        raise ValueError(f"Unsupported embedding_provider: {embedding_provider}")