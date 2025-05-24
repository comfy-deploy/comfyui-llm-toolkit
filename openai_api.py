#openai_api.py
import aiohttp
import json
import logging
from typing import List, Union, Optional, Dict, Any
import asyncio
import requests 
import base64
import os
logger = logging.getLogger(__name__)

async def create_openai_compatible_embedding(api_base: str, model: str, input: Union[str, List[str]], api_key: Optional[str] = None) -> List[float]:
    """
    Create embeddings using an OpenAI-compatible API asynchronously.
    
    :param api_base: The base URL for the API
    :param model: The name of the model to use for embeddings
    :param input: A string or list of strings to embed
    :param api_key: The API key (if required)
    :return: A list of embeddings
    """
    # Normalize the API base URL
    api_base = api_base.rstrip('/')
    if not api_base.endswith('/v1'):
        api_base += '/v1'
    
    url = f"{api_base}/embeddings"
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "input": input,
        "encoding_format": "float"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                    return result["data"][0]["embedding"] # Return the embedding directly as a list
                elif "data" in result and len(result["data"]) == 0: # handle no data in embedding result from API
                    raise ValueError("No embedding generated for the input text.")
                else:
                    raise ValueError("Unexpected response format: 'embedding' data not found")
    except aiohttp.ClientError as e:
        raise RuntimeError(f"Error calling embedding API: {str(e)}")

async def send_openai_request(api_url, base64_images, model, system_message, user_message, messages, api_key, 
                        seed, temperature, max_tokens, top_p, repeat_penalty, tools=None, tool_choice=None):
    """
    Sends a request to the OpenAI API and returns a unified response format.

    Args:
        api_url (str): The OpenAI API endpoint URL.
        base64_images (List[str]): List of images encoded in base64.
        model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        api_key (str): API key for OpenAI.
        seed (Optional[int]): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        tools (Optional[Any], optional): Tools to be used.
        tool_choice (Optional[Any], optional): Tool choice.

    Returns:
        Union[str, Dict[str, Any]]: Standardized response.
    """
    try:
        # Validate required parameters
        if not api_key:
            error_msg = "API key is required for OpenAI requests"
            logger.error(error_msg)
            return {"choices": [{"message": {"content": f"Error: {error_msg}"}}]}
            
        if not model:
            error_msg = "Model parameter is required for OpenAI requests"
            logger.error(error_msg)
            return {"choices": [{"message": {"content": f"Error: {error_msg}"}}]}
            
        if not api_url:
            api_url = "https://api.openai.com/v1/chat/completions"
            logger.info(f"No API URL provided, using default: {api_url}")
            
        openai_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare messages
        openai_messages = prepare_openai_messages(base64_images, system_message, user_message, messages)
        
        logger.info(f"Making OpenAI API request with model: {model}")

        data = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        if seed is not None:
            data["seed"] = seed
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        # --- Sanitize debug output to avoid clogging logs with large base64 blobs ---
        if logger.isEnabledFor(logging.DEBUG):
            # Clone payload so we do not mutate the real one
            _san = {**data}
            # Replace potentially huge fields with short placeholders
            if "messages" in _san:
                _san["messages"] = f"[{len(_san['messages'])} messages]"
            if base64_images:
                _san["base64_images"] = f"[{len(base64_images)} base64 image(s) omitted]"
            logger.debug(f"Request headers: {openai_headers}")
            logger.debug(f"Request data (sanitized): {_san}")

        logger.info(f"Sending OpenAI request to {api_url} for model {model}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=openai_headers, json=data) as response:
                    # Check status code before raising for status
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: Status {response.status}, Response: {error_text}")
                        return {"choices": [{"message": {"content": f"OpenAI API error: {response.status}. {error_text}"}}]}
                    
                    # Process successful response
                    logger.info(f"OpenAI API request successful with status {response.status}")
                    response_data = await response.json()
                    logger.debug(f"Response data: {response_data}")
                    
                    if tools:
                        return response_data
                    else:
                        choices = response_data.get('choices', [])
                        if choices:
                            choice = choices[0]
                            message = choice.get('message', {})
                            generated_text = message.get('content', '')
                            
                            # Check if content is None or empty
                            if generated_text is None:
                                error_msg = "Error: OpenAI returned null content"
                                logger.error(f"{error_msg}, full response: {response_data}")
                                return {"choices": [{"message": {"content": error_msg}}]}
                                
                            return {
                                "choices": [{
                                    "message": {
                                        "content": generated_text
                                    }
                                }]
                            }
                        else:
                            error_msg = "Error: No valid choices in the OpenAI response."
                            logger.error(f"{error_msg}, full response: {response_data}")
                            return {"choices": [{"message": {"content": error_msg}}]}
        except Exception as e:
            error_msg = f"Exception during OpenAI request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"choices": [{"message": {"content": error_msg}}]}
    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP error occurred: {e.status}, message='{e.message}', url={e.request_info.real_url}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except asyncio.CancelledError:
        # Handle task cancellation if needed
        raise
    except Exception as e:
        error_msg = f"Exception during OpenAI API call: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def prepare_openai_messages(base64_images, system_message, user_message, messages):
    openai_messages = []
    
    if system_message:
        openai_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            openai_messages.append({"role": "system", "content": content})
        elif role == "user":
            openai_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            openai_messages.append({"role": "assistant", "content": content})
    
    # Add the current user message with all images if provided
    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
        openai_messages.append({
            "role": "user",
            "content": content
        })
        print(f"Number of images sent: {len(base64_images)}")
    else:
        openai_messages.append({"role": "user", "content": user_message})
    
    return openai_messages

async def generate_image(
    prompt: str,
    model: str = "dall-e-3",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Generate images from a text prompt using DALL·E.

    :param prompt: The text prompt to generate images from.
    :param model: The model to use ("dall-e-3" or "dall-e-2").
    :param n: Number of images to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenAI generate_image error {response.status}: {error_text}")
                response.raise_for_status()
            data = await response.json()
            
            # Handle different response structures
            if "data" in data and isinstance(data["data"], list):
                images = []
                for item in data["data"]:
                    if "b64_json" in item:
                        images.append(item["b64_json"])
                    elif "url" in item:
                        # If response_format was changed or API returned URLs
                        images.append(item["url"])
                return images
            else:
                logger.error("Unexpected response format in generate_image")
                raise ValueError("Unexpected response format")

async def edit_image(
    image_base64: str,
    mask_base64: str,
    prompt: str,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Edit an existing image by replacing areas defined by a mask using DALL·E.

    :param image_base64: Base64-encoded original image.
    :param mask_base64: Base64-encoded mask image.
    :param prompt: The text prompt describing the desired edits.
    :param model: The model to use ("dall-e-2").
    :param n: Number of edited images to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of edited image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/edits"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }
    files = {
        "image": image_base64,
        "mask": mask_base64
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            images = [item["b64_json"] for item in data.get("data", [])]
            return images

async def generate_image_variations(
    image_base64: str,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Generate variations of an existing image using DALL·E.

    :param image_base64: Base64-encoded original image.
    :param model: The model to use ("dall-e-2").
    :param n: Number of variations to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of variation image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/variations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }
    files = {
        "image": image_base64
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            images = [item["b64_json"] for item in data.get("data", [])]
            return images

async def text_to_speech(text: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", output_path: str = "speech.mp3", api_key: Optional[str] = None) -> None:
    """
    Convert text to spoken audio using OpenAI's TTS API.

    :param text: The text to be converted to speech.
    :param model: The TTS model to use ("tts-1" or "tts-1-hd").
    :param voice: The voice to use for audio generation.
    :param response_format: The format of the output audio ("mp3", "opus", "aac", etc.).
    :param output_path: The file path to save the generated audio.
    :param api_key: The OpenAI API key.
    """
    api_url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            if response_format == "mp3":
                audio_data = await response.read()
                with open(output_path, "wb") as audio_file:
                    audio_file.write(audio_data)
            else:
                # Handle other formats if necessary
                pass

async def transcribe_audio(file_path: str, model: str = "whisper-1", response_format: str = "text", language: Optional[str] = None, api_key: Optional[str] = None) -> Union[str, dict]:
    """
    Transcribe audio into text using OpenAI's Whisper API.

    :param file_path: Path to the audio file to transcribe.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param language: (Optional) The language of the audio.
    :param api_key: The OpenAI API key.
    :return: Transcribed text or detailed JSON based on response_format.
    """
    api_url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(file_path), audio_file, "audio/mpeg"),
            "model": (None, model),
            "response_format": (None, response_format)
        }
        if language:
            files["language"] = (None, language)

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, data=files) as response:
                response.raise_for_status()
                if response_format == "text":
                    data = await response.text()
                else:
                    data = await response.json()
                return data

async def translate_audio(file_path: str, model: str = "whisper-1", response_format: str = "text", api_key: Optional[str] = None) -> Union[str, dict]:
    """
    Translate audio into English text using OpenAI's Whisper API.

    :param file_path: Path to the audio file to translate.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param api_key: The OpenAI API key.
    :return: Translated text or detailed JSON based on response_format.
    """
    api_url = "https://api.openai.com/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(file_path), audio_file, "audio/mpeg"),
            "model": (None, model),
            "response_format": (None, response_format)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, data=files) as response:
                response.raise_for_status()
                if response_format == "text":
                    data = await response.text()
                else:
                    data = await response.json()
                return data

# --- Streaming helper --------------------------------------------------
async def send_openai_request_stream(
    api_url: str,
    base64_images: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
):
    """Yield chunks from OpenAI chat completion stream (text-only).

    This mirrors *send_openai_request* but sets `stream=True` and yields each
    delta content piece as soon as it arrives.
    """
    if not api_key:
        raise RuntimeError("send_openai_request_stream – api_key is required")

    if not api_url:
        api_url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prepare message list
    openai_messages: List[Dict[str, Any]] = []
    if system_message:
        openai_messages.append({"role": "system", "content": system_message})
    if messages:
        openai_messages.extend(messages)
    if user_message:
        if base64_images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
            for img_b64 in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            openai_messages.append({"role": "user", "content": content})
        else:
            openai_messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.content:
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                    else:
                        data_str = line
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        choices = data_json.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content_piece = delta.get("content")
                        if content_piece:
                            yield content_piece
                    except json.JSONDecodeError:
                        logger.debug(f"Ignoring non-JSON line from stream: {data_str}")
    except aiohttp.ClientError as e:
        logger.error(f"OpenAI stream error: {e}")
        yield f"[OpenAI stream error: {e}]"

async def send_openai_image_generation_request(
    api_key: str,
    model: str,
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    quality: Optional[str] = None, # Combined quality
    style: Optional[str] = None,   # DALL-E 3 specific
    response_format: str = "b64_json",
    user: Optional[str] = None,
    # GPT-Image specific
    background: Optional[str] = None,
    output_format_gpt: Optional[str] = None,
    output_compression_gpt: Optional[int] = None,
    moderation_gpt: Optional[str] = None,
    # Edit/Variation specific (requires more parameters)
    image_base64: Optional[str] = None,
    mask_base64: Optional[str] = None,
    edit_mode: bool = False, # Flag to distinguish generation vs edit
    variation_mode: bool = False # Flag for variations
) -> Dict[str, Any]:
    """
    Sends an image generation or edit request to the OpenAI API.
    Handles different models (dall-e-2, dall-e-3, gpt-image-1) and modes.
    Returns the raw API response dictionary.
    """
    if not api_key:
        raise ValueError("API key is required for OpenAI image generation.")

    if edit_mode and variation_mode:
         raise ValueError("Cannot perform edit and variation simultaneously.")
    if edit_mode and not image_base64:
         raise ValueError("Image edit mode requires an input image.")
    if variation_mode and not image_base64:
         raise ValueError("Image variation mode requires an input image.")
    if mask_base64 and not edit_mode:
         logger.warning("Mask provided but not in edit mode. Mask will be ignored.")
         mask_base64 = None # Ignore mask if not editing

    headers = {"Authorization": f"Bearer {api_key}"}
    api_url = None
    payload = {}
    files = []

    # --- Determine API Endpoint and Mode ---
    if edit_mode:
        logger.info(f"OpenAI Image Edit Request: model={model}, prompt={prompt[:50]}...")
        api_url = "https://api.openai.com/v1/images/edits"
        headers.pop("Content-Type", None) # Use multipart/form-data
        if image_base64:
            if isinstance(image_base64, list):
                for idx, b64 in enumerate(image_base64):
                    try:
                        file_data = base64.b64decode(b64)
                    except Exception as e:
                        logger.error("Invalid base64 string for reference image %s: %s", idx, e)
                        continue
                    # Use the OpenAI array syntax "image[]" when multiple images are provided
                    # to avoid the "Duplicate parameter: 'image'" error.
                    files.append(("image[]", file_data, f"image_{idx}.png", "image/png"))
            else:
                try:
                    data_dec = base64.b64decode(image_base64)
                    files.append(("image", data_dec, "input_image.png", "image/png"))
                except Exception as e:
                    logger.error("Invalid base64 string for reference image: %s", e)
        if mask_base64:
             try:
                 mask_dec = base64.b64decode(mask_base64)
                 files.append(("mask", mask_dec, "mask_image.png", "image/png"))
             except Exception as e:
                 logger.error("Invalid base64 string for mask: %s", e)
        payload = { # Form data, not JSON
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        # Only include response_format if model allows it (dall-e-*). gpt-image-1 currently rejects it.
        if model.startswith("dall-e") and response_format:
            payload["response_format"] = response_format
        if user: payload["user"] = user
        # Quality/Style not applicable to DALL-E 2 edits
        if model == "gpt-image-1" and quality: payload["quality"] = quality # GPT edit uses quality


    elif variation_mode:
        logger.info(f"OpenAI Image Variation Request: model={model}...")
        if model != "dall-e-2":
             raise ValueError("Image variations are only supported for dall-e-2.")
        api_url = "https://api.openai.com/v1/images/variations"
        headers.pop("Content-Type", None) # Use multipart/form-data
        if image_base64:
             if isinstance(image_base64, list):
                 for idx, b64 in enumerate(image_base64):
                     try:
                         file_data = base64.b64decode(b64)
                     except Exception as e:
                         logger.error("Invalid base64 string for reference image %s: %s", idx, e)
                         continue
                     # Use the OpenAI array syntax "image[]" when multiple images are provided
                     # to avoid the "Duplicate parameter: 'image'" error.
                     files.append(("image[]", file_data, f"image_{idx}.png", "image/png"))
             else:
                 try:
                     data_dec = base64.b64decode(image_base64)
                     files.append(("image", data_dec, "input_image.png", "image/png"))
                 except Exception as e:
                     logger.error("Invalid base64 string for reference image: %s", e)
        payload = { # Form data
            "model": model,
            "n": n,
            "size": size,
        }
        if model.startswith("dall-e") and response_format:
            payload["response_format"] = response_format
        if user: payload["user"] = user

    else: # Generation mode
        logger.info(f"OpenAI Image Generation Request: model={model}, prompt={prompt[:50]}...")
        api_url = "https://api.openai.com/v1/images/generations"
        headers["Content-Type"] = "application/json"
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        # For DALL-E models include response_format; gpt-image-1 omits it (defaults to b64_json)
        if model.startswith("dall-e"):
            payload["response_format"] = response_format
        else:  # gpt-image-1 specific extras
            if quality: payload["quality"] = quality
            if background: payload["background"] = background
            if output_format_gpt: payload["output_format"] = output_format_gpt
            if output_compression_gpt is not None: payload["output_compression"] = output_compression_gpt
            if moderation_gpt: payload["moderation"] = moderation_gpt


        # DALL-E 3 specific
        if model == "dall-e-3":
             if quality: payload["quality"] = quality # standard or hd
             if style: payload["style"] = style     # vivid or natural

        if user: payload["user"] = user
        # Seed is documented but might not be universally supported yet
        # if seed is not None and seed != -1: payload["seed"] = seed

    # --- Make API Call ---
    try:
        async with aiohttp.ClientSession() as session:
            request_params = {"headers": headers}
            if files:
                # Use multipart/form-data
                form_data = aiohttp.FormData()
                for key, value in payload.items():
                    if value is not None:
                        form_data.add_field(key, str(value))
                for field_name, binary, filename, ctype in files:
                    form_data.add_field(field_name, binary, filename=filename, content_type=ctype)
                request_params["data"] = form_data
            else:
                # For JSON requests (generations) attach payload via 'json' key so aiohttp sets body
                request_params["json"] = payload

            async with session.post(api_url, **request_params) as response:
                response_text = await response.text() # Read text first for logging
                logger.debug(f"OpenAI Image API Raw Response ({response.status}): {response_text[:500]}...")
                if response.status != 200:
                    logger.error(f"OpenAI Image API error: Status {response.status}, Response: {response_text}")
                    # Try to parse error details if JSON
                    try:
                        error_data = json.loads(response_text)
                        error_msg = error_data.get("error", {}).get("message", response_text)
                    except json.JSONDecodeError:
                        error_msg = response_text
                    raise aiohttp.ClientResponseError(
                         response.request_info, response.history, status=response.status, message=error_msg, headers=response.headers
                    )

                # Parse successful JSON response
                response_data = json.loads(response_text)
                logger.info(f"OpenAI Image API request successful.")
                logger.debug(f"Response data structure: { {k: type(v).__name__ for k, v in response_data.items()} }")
                return response_data # Return the raw JSON response

    except aiohttp.ClientResponseError as e:
        # Re-raise with more context if possible
        raise RuntimeError(f"OpenAI Image API HTTP error: {e.status} - {e.message}") from e
    except asyncio.TimeoutError:
        raise TimeoutError("OpenAI Image API request timed out.")
    except Exception as e:
        logger.error(f"Exception during OpenAI Image API call: {str(e)}", exc_info=True)
        raise RuntimeError(f"Exception during OpenAI Image API call: {str(e)}") from e
