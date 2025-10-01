# wavespeed_image_api.py
import asyncio
import logging
import httpx
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

async def send_wavespeed_image_edit_request(
    api_key: str,
    model: str,
    prompt: str,
    image_base64: str,
    guidance_scale: Optional[float] = None,
    seed: int = -1,
) -> Dict[str, Any]:
    """
    Sends a request to the WaveSpeedAI Image Edit/Enhancement APIs and polls for the result.
    Handles models like SeedEdit V3 and Portrait.
    """
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not image_base64:
        raise ValueError("Input image is required for this model.")

    # Always request base64 output for direct use in ComfyUI
    payload = {
        "prompt": prompt,
        "image": f"data:image/png;base64,{image_base64}",
        "enable_base64_output": True,
    }
    if seed != -1:
        payload["seed"] = seed
    
    # Conditionally add guidance_scale for models that support it
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Submitting image edit task to WaveSpeedAI model: {model}")
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get("data") and result["data"].get("id"):
                request_id = result["data"]["id"]
                logger.info(f"Task submitted successfully. Request ID: {request_id}")
            else:
                logger.error(f"WaveSpeedAI API Error: Unexpected submission response format. {result}")
                return {"error": "Unexpected submission response format.", "details": result}

        except httpx.HTTPStatusError as e:
            logger.error(f"WaveSpeedAI API Error on submission: {e.response.status_code} - {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            logger.error(f"Error submitting to WaveSpeedAI: {e}", exc_info=True)
            return {"error": str(e)}

        if not request_id:
            return {"error": "Failed to get a request ID from WaveSpeedAI."}

        # --- Polling for result ---
        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}
        
        for attempt in range(60): # Poll for up to 60 seconds (60 * 1s)
            try:
                await asyncio.sleep(1) # 1-second polling interval
                logger.debug(f"Polling for result... (Attempt {attempt + 1})")
                
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    logger.info("WaveSpeedAI task completed.")
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    
                    # Expecting a list with one base64 string
                    b64_data = outputs[0]
                    # The API returns the full data URI, strip the prefix
                    if "base64," in b64_data:
                        b64_json = b64_data.split("base64,")[1]
                    else:
                        b64_json = b64_data
                    
                    return {"data": [{"b64_json": b64_json}]}

                elif status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    logger.error(f"WaveSpeedAI task failed: {error_msg}")
                    return {"error": f"Task failed: {error_msg}"}
                else:
                    logger.info(f"Task status: {status}. Continuing to poll.")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"WaveSpeedAI API Error during polling: {e.response.status_code} - {e.response.text}")
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as e:
                logger.error(f"Error polling WaveSpeedAI: {e}", exc_info=True)
                return {"error": f"Polling failed: {str(e)}"}

    return {"error": "Polling timed out after 60 seconds."}


async def send_wavespeed_flux_request(
    api_key: str,
    model: str,
    prompt: str,
    image_base64: Optional[str] = None, # For single image models
    images_base64: Optional[list[str]] = None, # For multi-image models
    size: Optional[str] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    num_images: int = 1,
    seed: int = -1,
    enable_safety_checker: bool = True,
) -> Dict[str, Any]:
    """
    Sends a request to the WaveSpeedAI FLUX Kontext Dev API and polls for the result.
    """
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")

    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "enable_base64_output": True,
    }
    if seed != -1:
        payload["seed"] = seed
    if size:
        payload["size"] = size
    if images_base64:
        payload["images"] = [f"data:image/png;base64,{b64}" for b64 in images_base64]
    elif image_base64:
        payload["image"] = f"data:image/png;base64,{image_base64}"
    
    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Submitting task to WaveSpeedAI Flux model: {model}")
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get("data") and result["data"].get("id"):
                request_id = result["data"]["id"]
                logger.info(f"Task submitted successfully. Request ID: {request_id}")
            else:
                logger.error(f"WaveSpeedAI API Error: Unexpected submission response. {result}")
                return {"error": "Unexpected submission response format.", "details": result}

        except httpx.HTTPStatusError as e:
            logger.error(f"WaveSpeedAI API Error on submission: {e.response.status_code} - {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            logger.error(f"Error submitting to WaveSpeedAI: {e}", exc_info=True)
            return {"error": str(e)}

        if not request_id:
            return {"error": "Failed to get a request ID from WaveSpeedAI."}

        # --- Polling for result ---
        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}
        
        for attempt in range(120): # Poll for up to 120 seconds
            try:
                await asyncio.sleep(1)
                logger.debug(f"Polling for result... (Attempt {attempt + 1})")
                
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    logger.info("WaveSpeedAI Flux task completed.")
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    
                    # Process multiple potential outputs
                    b64_jsons = []
                    for out in outputs:
                        if "base64," in out:
                            b64_jsons.append(out.split("base64,")[1])
                        else:
                            b64_jsons.append(out)
                    
                    return {"data": [{"b64_json": b64} for b64 in b64_jsons]}

                elif status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    logger.error(f"WaveSpeedAI Flux task failed: {error_msg}")
                    return {"error": f"Task failed: {error_msg}"}
                else:
                    logger.info(f"Task status: {status}. Continuing to poll.")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"WaveSpeedAI API Error during polling: {e.response.status_code} - {e.response.text}")
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as e:
                logger.error(f"Error polling WaveSpeedAI: {e}", exc_info=True)
                return {"error": f"Polling failed: {str(e)}"}

    return {"error": "Polling timed out after 120 seconds."} 

async def send_wavespeed_seedream_request(
    api_key: str,
    model: str,
    prompt: str,
    size: Optional[str] = "2048*2048",
    seed: Optional[int] = -1,
    images_base64: Optional[list[str]] = None,
    max_images: Optional[int] = 1,
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """
    Generic function for Bytedance Seedream V4 models on WaveSpeed.
    Handles task submission and polling for results.
    """
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")

    # Correct the model path for the API endpoint
    api_model_path = model
    if model.startswith("bytedance/seedream-v4-"):
        suffix = model.replace("bytedance/seedream-v4-", "")
        api_model_path = f"bytedance/seedream-v4/{suffix}"

    submit_url = f"https://api.wavespeed.ai/api/v3/{api_model_path}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "enable_sync_mode": enable_sync_mode,
        "enable_base64_output": True,
    }

    if "edit" in model:
        if not images_base64:
            return {"error": "Edit mode requires at least one image."}
        payload["images"] = [f"data:image/png;base64,{b64}" for b64 in images_base64]
        payload["prompt"] = prompt
    else:
        payload["prompt"] = prompt

    if size:
        payload["size"] = size
        
    if seed is not None and seed != -1:
        payload["seed"] = seed

    if "sequential" in model:
        payload["max_images"] = max_images

    logger.info(f"WaveSpeed Seedream Request: URL={submit_url}, Model={model}")
    
    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if result.get("data") and result["data"].get("id"):
                request_id = result["data"]["id"]
                logger.info(f"Task submitted successfully. Request ID: {request_id}")
            else:
                return {"error": "Unexpected submission response format.", "details": result}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            return {"error": str(e)}

        if not request_id:
            return {"error": "Failed to get a request ID from WaveSpeedAI."}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}
        
        for attempt in range(120):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    logger.info("WaveSpeedAI Seedream task completed.")
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    
                    b64_jsons = []
                    for out in outputs:
                        if "base64," in out:
                            b64_jsons.append(out.split("base64,")[1])
                        else:
                            b64_jsons.append(out)
                    
                    return {"data": [{"b64_json": b64} for b64 in b64_jsons]}

                elif status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as e:
                return {"error": f"Polling failed: {str(e)}"}

    return {"error": "Polling timed out after 120 seconds."}


async def send_wavespeed_hunyuan_request(
    api_key: str,
    model: str,
    prompt: str,
    size: Optional[str] = None,
    seed: int = -1,
    output_format: Optional[str] = None,
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """Submit and poll a WaveSpeed Hunyuan Image 3 task."""
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not prompt:
        raise ValueError("Prompt is required for Hunyuan Image generation.")

    def _normalize_size(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = value.replace('x', '*').replace('X', '*')
        return normalized

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "enable_base64_output": True,
        "enable_sync_mode": enable_sync_mode,
    }
    normalized_size = _normalize_size(size)
    if normalized_size:
        payload["size"] = normalized_size
    if seed is not None and seed != -1:
        payload["seed"] = seed
    if output_format:
        payload["output_format"] = output_format

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data", {})
            request_id = data.get("id")
            if not request_id:
                return {"error": "Unexpected submission response format.", "details": result}
            logger.info("WaveSpeed Hunyuan task submitted. Request ID: %s", request_id)
        except httpx.HTTPStatusError as e:
            logger.error("WaveSpeed Hunyuan submission failed: %s - %s", e.response.status_code, e.response.text)
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as exc:
            logger.error("Failed to submit WaveSpeed Hunyuan request", exc_info=True)
            return {"error": str(exc)}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(120):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    b64_values = []
                    for output in outputs:
                        if "base64," in output:
                            b64_values.append(output.split("base64,")[1])
                        else:
                            b64_values.append(output)
                    return {"data": [{"b64_json": value} for value in b64_values]}
                if status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as exc:
                logger.error("Error polling WaveSpeed Hunyuan request", exc_info=True)
                return {"error": f"Polling failed: {str(exc)}"}

    return {"error": "Polling timed out after 120 seconds."}


async def send_wavespeed_qwen_edit_plus_request(
    api_key: str,
    model: str,
    prompt: str,
    images_base64: list[str],
    size: Optional[str] = None,
    seed: int = -1,
    output_format: Optional[str] = None,
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """Submit and poll a WaveSpeed Qwen Image Edit Plus task."""
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not images_base64:
        raise ValueError("At least one reference image is required for Qwen Image Edit Plus.")

    def _normalize_size(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.replace('x', '*').replace('X', '*')

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "images": [f"data:image/png;base64,{b64}" for b64 in images_base64],
        "enable_base64_output": True,
        "enable_sync_mode": enable_sync_mode,
    }
    normalized_size = _normalize_size(size)
    if normalized_size:
        payload["size"] = normalized_size
    if seed is not None and seed != -1:
        payload["seed"] = seed
    if output_format:
        payload["output_format"] = output_format

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data", {})
            request_id = data.get("id")
            if not request_id:
                return {"error": "Unexpected submission response format.", "details": result}
            logger.info("WaveSpeed Qwen Edit Plus task submitted. Request ID: %s", request_id)
        except httpx.HTTPStatusError as e:
            logger.error("WaveSpeed Qwen submission failed: %s - %s", e.response.status_code, e.response.text)
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as exc:
            logger.error("Failed to submit WaveSpeed Qwen request", exc_info=True)
            return {"error": str(exc)}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(120):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    b64_values = []
                    for output in outputs:
                        if "base64," in output:
                            b64_values.append(output.split("base64,")[1])
                        else:
                            b64_values.append(output)
                    return {"data": [{"b64_json": value} for value in b64_values]}
                if status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as exc:
                logger.error("Error polling WaveSpeed Qwen request", exc_info=True)
                return {"error": f"Polling failed: {str(exc)}"}

    return {"error": "Polling timed out after 120 seconds."}


async def send_wavespeed_imagen4_request(
    api_key: str,
    model: str,
    prompt: str,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    num_images: int = 1,
    seed: int = -1,
    negative_prompt: Optional[str] = None,
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """Submit and poll a WaveSpeed Imagen 4 generation task."""

    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not prompt:
        raise ValueError("Prompt is required for Imagen 4 generation.")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "num_images": max(1, min(4, num_images)),
        "enable_base64_output": True,
        "enable_sync_mode": enable_sync_mode,
    }

    if aspect_ratio and aspect_ratio.lower() != "auto":
        payload["aspect_ratio"] = aspect_ratio
    if resolution and resolution.lower() != "auto":
        payload["resolution"] = resolution.lower()
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None and seed != -1:
        payload["seed"] = seed

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data", {})
            request_id = data.get("id")
            if not request_id:
                return {"error": "Unexpected submission response format.", "details": result}
            logger.info("WaveSpeed Imagen4 task submitted. Request ID: %s", request_id)
        except httpx.HTTPStatusError as exc:
            logger.error("WaveSpeed Imagen4 submission failed: %s - %s", exc.response.status_code, exc.response.text)
            return {"error": f"HTTP {exc.response.status_code}", "details": exc.response.text}
        except Exception as exc:
            logger.error("Failed to submit WaveSpeed Imagen4 request", exc_info=True)
            return {"error": str(exc)}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(150):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    images = []
                    for output in outputs:
                        if isinstance(output, str) and "base64," in output:
                            images.append({"b64_json": output.split("base64,")[1]})
                        elif isinstance(output, str) and output.strip():
                            images.append({"url": output})
                    if images:
                        return {"data": images}
                    return {"error": "Completed task but no usable outputs returned."}

                if status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code} while polling", "details": exc.response.text}
            except Exception as exc:
                logger.error("Error polling WaveSpeed Imagen4 request", exc_info=True)
                return {"error": f"Polling failed: {str(exc)}"}

    return {"error": "Polling timed out after 150 seconds."}


async def send_wavespeed_dreamina_request(
    api_key: str,
    model: str,
    prompt: str,
    size: Optional[str] = None,
    seed: int = -1,
    prompt_expansion: bool = True,
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """Submit and poll a WaveSpeed Dreamina V3.1 generation task."""

    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not prompt:
        raise ValueError("Prompt is required for Dreamina generation.")

    def _normalize_size(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.replace("x", "*").replace("X", "*")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "enable_prompt_expansion": bool(prompt_expansion),
        "enable_base64_output": True,
        "enable_sync_mode": enable_sync_mode,
    }
    normalized_size = _normalize_size(size)
    if normalized_size:
        payload["size"] = normalized_size
    if seed is not None and seed != -1:
        payload["seed"] = seed

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data", {})
            request_id = data.get("id")
            if not request_id:
                return {"error": "Unexpected submission response format.", "details": result}
            logger.info("WaveSpeed Dreamina task submitted. Request ID: %s", request_id)
        except httpx.HTTPStatusError as exc:
            logger.error("WaveSpeed Dreamina submission failed: %s - %s", exc.response.status_code, exc.response.text)
            return {"error": f"HTTP {exc.response.status_code}", "details": exc.response.text}
        except Exception as exc:
            logger.error("Failed to submit WaveSpeed Dreamina request", exc_info=True)
            return {"error": str(exc)}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(150):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    images = []
                    for output in outputs:
                        if isinstance(output, str) and "base64," in output:
                            images.append({"b64_json": output.split("base64,")[1]})
                        elif isinstance(output, str) and output.strip():
                            images.append({"url": output})
                    if images:
                        return {"data": images}
                    return {"error": "Completed task but no usable outputs returned."}

                if status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code} while polling", "details": exc.response.text}
            except Exception as exc:
                logger.error("Error polling WaveSpeed Dreamina request", exc_info=True)
                return {"error": f"Polling failed: {str(exc)}"}

    return {"error": "Polling timed out after 150 seconds."}


async def send_wavespeed_nano_banana_edit_request(
    api_key: str,
    model: str,
    prompt: Optional[str],
    image_base64: str,
    output_format: str = "png",
    enable_sync_mode: bool = False,
) -> Dict[str, Any]:
    """Submit and poll a WaveSpeed Nano Banana edit task."""

    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not image_base64:
        raise ValueError("An input image is required for Nano Banana edit requests.")

    payload: Dict[str, Any] = {
        "image": f"data:image/png;base64,{image_base64}",
        "output_format": (output_format or "png").lower(),
        "enable_base64_output": True,
        "enable_sync_mode": enable_sync_mode,
    }
    if prompt:
        payload["prompt"] = prompt

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data", {})
            request_id = data.get("id")
            if not request_id:
                return {"error": "Unexpected submission response format.", "details": result}
            logger.info("WaveSpeed Nano Banana task submitted. Request ID: %s", request_id)
        except httpx.HTTPStatusError as exc:
            logger.error("WaveSpeed Nano Banana submission failed: %s - %s", exc.response.status_code, exc.response.text)
            return {"error": f"HTTP {exc.response.status_code}", "details": exc.response.text}
        except Exception as exc:
            logger.error("Failed to submit WaveSpeed Nano Banana request", exc_info=True)
            return {"error": str(exc)}

        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(150):
            try:
                await asyncio.sleep(1)
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    images = []
                    for output in outputs:
                        if isinstance(output, str) and "base64," in output:
                            images.append({"b64_json": output.split("base64,")[1]})
                        elif isinstance(output, str) and output.strip():
                            images.append({"url": output})
                    if images:
                        return {"data": images}
                    return {"error": "Completed task but no usable outputs returned."}

                if status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    return {"error": f"Task failed: {error_msg}"}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code} while polling", "details": exc.response.text}
            except Exception as exc:
                logger.error("Error polling WaveSpeed Nano Banana request", exc_info=True)
                return {"error": f"Polling failed: {str(exc)}"}

    return {"error": "Polling timed out after 150 seconds."}
