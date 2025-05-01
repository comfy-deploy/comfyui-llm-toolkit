# generate_text.py
import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import ComfyUI-specific modules
try:
    import folder_paths
    from server import PromptServer # <-- Import PromptServer
except ImportError:
    logger.warning("Could not import folder_paths or server. Make sure ComfyUI environment is set up.")
    folder_paths = None
    PromptServer = None

# Check for required dependencies
missing_deps = []
try:
    import aiohttp
except ImportError:
    missing_deps.append("aiohttp")

if missing_deps:
    logger.warning(f"Missing dependencies: {', '.join(missing_deps)}. Some functionality may not work.")
    logger.warning("Please install missing dependencies: pip install " + " ".join(missing_deps))

# Import utility functions (assuming they exist)
from send_request import send_request, run_async # Keep non-streaming version if needed
from llmtoolkit_utils import query_local_ollama_models, ensure_ollama_server, ensure_ollama_model, get_api_key

# Local transformers streaming (optional)
try:
    from transformers_provider import send_transformers_request_stream
except ImportError:
    send_transformers_request_stream = None  # type: ignore

# --- NEW STREAMING REQUEST FUNCTION (Example for Ollama) ---
# IMPORTANT: This needs to be adapted based on the actual API structure of the provider!
async def send_request_stream(
    llm_provider: str,
    base_ip: str,
    port: str,
    llm_model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, str]],
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    random: bool = False,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    stop: Optional[List[str]] = None,
    keep_alive: Union[bool, str] = True,
    llm_api_key: Optional[str] = None,
    timeout: int = 120 # Add a timeout for the connection
) -> AsyncGenerator[str, None]:
    """
    Sends a streaming request to an LLM provider (Example for Ollama).
    Yields text chunks as they are received.
    """
    if llm_provider.lower() == "openai":
        # --- OpenAI Specific Streaming Logic ---
        if not llm_api_key:
            logger.error("OpenAI streaming requested but no API key supplied.")
            async for chunk in send_request_stream(
                llm_provider="openai_fallback_nonstream",
                base_ip=base_ip,
                port=port,
                llm_model=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
                random=random,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop,
                keep_alive=keep_alive,
                llm_api_key=llm_api_key,
            ):
                yield chunk
            return

        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json",
        }
        # Build message list if not provided
        if not messages:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if user_message:
                messages.append({"role": "user", "content": user_message})

        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Streaming request to OpenAI: model={llm_model}")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(openai_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for raw_line in response.content:
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        # OpenAI streams multiple lines that may begin with 'data:'; join those if needed.
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :].strip()
                        else:
                            data_str = line
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    yield content_piece
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line from OpenAI stream: {data_str}")
        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
            yield f"[OpenAI streaming error: {e}]"
        return

    if llm_provider.lower() == "ollama":
        # --- Ollama Specific Streaming Logic ---
        if not ensure_ollama_server(base_ip, port):
            logger.error("Ollama daemon unavailable and could not be started – aborting stream.")
            yield "[Error: Ollama daemon unavailable]"
            return

        # Ensure requested model is present locally (will pull if missing)
        ensure_ollama_model(llm_model, base_ip, port)

        url = f"http://{base_ip}:{port}/api/generate"
        headers = {"Content-Type": "application/json"}
        if llm_api_key: # Ollama doesn't typically use API keys this way, but include for consistency
            headers["Authorization"] = f"Bearer {llm_api_key}"

        # Construct messages list if not provided directly
        if not messages:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if user_message:
                messages.append({"role": "user", "content": user_message})

        # Prepare payload for Ollama /api/generate
        payload = {
            "model": llm_model,
            "prompt": user_message, # Ollama uses 'prompt' for the main user message
            "system": system_message if system_message else None,
            "stream": True, # Explicitly request streaming
            "options": {
                "seed": seed,
                "temperature": temperature,
                "num_predict": max_tokens, # Ollama uses num_predict for max_tokens
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "stop": stop,
            }
        }
        # Clean up None values Ollama might not like
        if not system_message: del payload["system"]
        payload["options"] = {k: v for k, v in payload["options"].items() if v is not None}

        logger.info(f"Streaming request to Ollama: {url} with payload: { {**payload, 'options': {**payload.get('options', {}), 'api_key': '****' if llm_api_key else None} } }")

        try:
            # Use a single session if possible, manage timeouts
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                    # Process the streaming response line by line
                    async for line in response.content:
                        if line:
                            try:
                                decoded_line = line.decode('utf-8').strip()
                                if decoded_line:
                                    data = json.loads(decoded_line)
                                    chunk = data.get("response", "")
                                    if chunk:
                                        yield chunk
                                    # Check if generation is done (Ollama specific)
                                    if data.get("done", False):
                                        logger.info("Ollama stream finished.")
                                        break
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON line: {line.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                logger.error(f"Error processing stream line: {e}", exc_info=True)
                                yield f"[Error processing stream: {e}]"
                                break # Stop streaming on error

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error to {url}: {e}")
            yield f"[Connection Error: {e}]"
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error {e.status} from {url}: {e.message}")
            # Attempt to read error details from response body
            error_body = await e.response.text() if hasattr(e, 'response') else 'No details'
            logger.error(f"Error Body: {error_body}")
            yield f"[HTTP Error {e.status}: {e.message} - {error_body[:100]}]"
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout} seconds to {url}")
            yield f"[Timeout Error]"
        except Exception as e:
            logger.error(f"An unexpected error occurred during streaming request: {e}", exc_info=True)
            yield f"[Unexpected Error: {e}]"

    # ------------------------------------------------------------------
    #  Local HuggingFace Transformers – streaming
    # ------------------------------------------------------------------
    if llm_provider.lower() in {"transformers", "hf", "local"} and send_transformers_request_stream is not None:
        try:
            async for chunk in send_transformers_request_stream(
                base64_images=[],
                base64_audio=[],
                model=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                precision="fp16",
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Transformers streaming error: {e}", exc_info=True)
            yield f"[Error: {e}]"
        return

    # Existing fallback logic for other providers
    if llm_provider.lower() not in ["ollama", "openai", "transformers", "hf", "local"]:
        logger.warning(f"Streaming not implemented for provider '{llm_provider}'. Falling back to non-streaming.")
        try:
            full_response_data = await send_request(
                llm_provider=llm_provider, base_ip=base_ip, port=port, images=None, llm_model=llm_model,
                system_message=system_message, user_message=user_message, messages=messages, seed=seed,
                temperature=temperature, max_tokens=max_tokens, random=random, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty, stop=stop, keep_alive=keep_alive, llm_api_key=llm_api_key
            )
            if isinstance(full_response_data, dict):
                 if "choices" in full_response_data and full_response_data["choices"]:
                     message = full_response_data["choices"][0].get("message", {})
                     content = message.get("content", "")
                     if content: yield content
                 elif "response" in full_response_data:
                     if full_response_data["response"]: yield full_response_data["response"]
                 else:
                    logger.error(f"Unexpected non-streaming response format: {full_response_data}")
            elif isinstance(full_response_data, str):
                 if full_response_data: yield full_response_data
            else:
                 logger.error(f"Unexpected non-streaming response type: {type(full_response_data)}")

        except Exception as e:
            logger.error(f"Error in fallback non-streaming request for {llm_provider}: {e}", exc_info=True)
            yield f"[Error: {e}]"
        return # Stop generation after yielding the fallback response

# --- Original Node (for reference or non-streaming use) ---
class LLMToolkitTextGenerator:
    DEFAULT_PROVIDER = "openai"
    # For default OpenAI we will use GPT-4o mini (or 4o-mini) – alias may differ
    DEFAULT_MODEL: str = "gpt-4o-mini"

    MODEL_LIST: List[str] = [DEFAULT_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (cls.MODEL_LIST, {"default": cls.DEFAULT_MODEL}),
                "prompt": ("STRING", {"multiline": False, "default": "Write a short story about a robot learning to paint."})
            },
            "optional": {
                "context": ("*", {})
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit"
    OUTPUT_NODE = True # Keeps the text widget for non-streaming version

    def generate(self, llm_model, prompt, context=None):
        # ... (original generate logic using run_async(send_request(...))) ...
        # This function remains mostly the same as the user provided,
        # calling the original non-streaming send_request.
        # We'll copy the parameter processing logic from the streaming version
        # for consistency, but it will call the non-streaming send_request.
        try:
            params = {
                "llm_provider": self.DEFAULT_PROVIDER,
                "llm_model": llm_model,
                "system_message": "You are a helpful, creative, and concise assistant.",
                "user_message": prompt,
                "base_ip": "localhost",
                "port": "11434",
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": None,
                "keep_alive": True,
                "messages": []
            }

            provider_config = None
            if context is not None:
                if isinstance(context, dict) and "provider_name" in context:
                    provider_config = context
                elif isinstance(context, dict) and "provider_config" in context:
                    provider_config = context["provider_config"]

            if provider_config and isinstance(provider_config, dict):
                if "provider_name" in provider_config: params["llm_provider"] = provider_config["provider_name"]
                if "api_key" in provider_config: params["llm_api_key"] = provider_config["api_key"]
                if "base_ip" in provider_config: params["base_ip"] = provider_config["base_ip"]
                if "port" in provider_config: params["port"] = provider_config["port"]
                for key in provider_config:
                    if key not in ["provider_name", "llm_model", "api_key", "base_ip", "port", "user_prompt"]:
                        params[key] = provider_config[key]
                if "user_prompt" in provider_config: params["user_message"] = provider_config["user_prompt"]
                provider_model = provider_config.get("llm_model", "")
                if provider_model:
                    params["llm_model"] = provider_model
                else:
                    params["llm_model"] = "" # Will be handled below

            if params.get("llm_provider"):
                provider = str(params["llm_provider"]).lower()
                if not params.get("llm_model"):
                    PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-3-opus-20240229"}
                    fallback = PROVIDER_DEFAULTS.get(provider)
                    if fallback: params["llm_model"] = fallback

            # Auto-fetch API key for OpenAI if missing/placeholder
            if provider == "openai" and (not params.get("llm_api_key") or params["llm_api_key"] in {"", "1234", None}):
                try:
                    params["llm_api_key"] = get_api_key("OPENAI_API_KEY", "openai")
                    logger.info("generate: Retrieved OpenAI API key via get_api_key helper.")
                except ValueError as _e:
                    logger.warning(f"generate: get_api_key failed – {_e}")

            log_params = {**params}
            if "llm_api_key" in log_params: log_params["llm_api_key"] = "****" if log_params["llm_api_key"] else "None"
            logger.info(f"[Non-Streaming] Making LLM request with params: {log_params}")

            try:
                # --- CALL NON-STREAMING VERSION ---
                response_data = run_async(
                    send_request( # Original non-streaming call
                        llm_provider=params["llm_provider"],
                        base_ip=params.get("base_ip", "localhost"),
                        port=params.get("port", "11434"),
                        images=None,
                        llm_model=params["llm_model"],
                        system_message=params["system_message"],
                        user_message=params["user_message"],
                        messages=params["messages"],
                        seed=params.get("seed"),
                        temperature=params["temperature"],
                        max_tokens=params["max_tokens"],
                        random=params.get("random", False),
                        top_k=params["top_k"],
                        top_p=params["top_p"],
                        repeat_penalty=params["repeat_penalty"],
                        stop=params.get("stop"),
                        keep_alive=params.get("keep_alive", True),
                        llm_api_key=params.get("llm_api_key"),
                    )
                )
                # --- END NON-STREAMING CALL ---
            except Exception as e:
                 logger.error(f"Error in non-streaming send_request call: {e}", exc_info=True)
                 response_data = {"choices": [{"message": {"content": f"Error calling send_request: {str(e)}"}}]}

            if response_data is None: content = "Error: Received None response"
            elif isinstance(response_data, dict):
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    if content is None: content = "Error: Null content in response"
                elif "response" in response_data: content = response_data["response"]
                else: content = f"Error: Unexpected format: {str(response_data)}"
            elif isinstance(response_data, str): content = response_data
            else: content = f"Error: Unexpected type: {type(response_data)}"

            if context is not None and isinstance(context, dict):
                context_out = context.copy()
                context_out["llm_response"] = content
                context_out["llm_raw_response"] = response_data
            else:
                context_out = {"llm_response": content, "llm_raw_response": response_data, "passthrough_data": context}

            return {"ui": {"string": [content]}, "result": (context_out,)}

        except Exception as e:
            error_message = f"Error generating text: {str(e)}"
            logger.error(error_message, exc_info=True)
            error_output = {"error": error_message, "original_input": context}
            return {"ui": {"string": [error_message]}, "result": (error_output,)}


# --- NEW STREAMING NODE ---
class LLMToolkitTextGeneratorStream:
    DEFAULT_PROVIDER = "openai"
    # For default OpenAI we will use GPT-4o mini (or 4o-mini) – alias may differ
    DEFAULT_MODEL: str = "gpt-4o-mini"

    MODEL_LIST: List[str] = [DEFAULT_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (cls.MODEL_LIST, {"default": cls.DEFAULT_MODEL}),
                "prompt": ("STRING", {"multiline": False, "default": "Write a detailed description of a futuristic city."})
            },
            "optional": {
                "context": ("*", {})
            },
            "hidden": { # <-- Add hidden inputs
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "generate_stream" # <-- Use new function name
    CATEGORY = "llm_toolkit"
    OUTPUT_NODE = True # Keep the JS widget logic

    def generate_stream(self, llm_model, prompt, unique_id, context=None, **kwargs):
        """
        Generates text using the specified provider and streams the response back
        to the UI via websocket messages.  (synchronous wrapper)
        """
        # Wrap the previous async implementation inside an inner coroutine
        async def _async_generate():
            # Previous async body START
            if PromptServer is None:
                logger.error("PromptServer not available. Cannot stream.")
                error_msg = "Streaming requires PromptServer, which is not available."
                error_output = {"error": error_msg, "original_input": context}
                return {"ui": {"string": [error_msg]}, "result": (error_output,)}

            server = PromptServer.instance
            full_response_text = ""

            try:
                # --- Parameter processing logic (same as non-streaming) ---
                params = {
                    "llm_provider": self.DEFAULT_PROVIDER,
                    "llm_model": llm_model,
                    "system_message": "You are a helpful, creative, and concise assistant.",
                    "user_message": prompt,
                    "base_ip": "localhost",
                    "port": "11434",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "stop": None,
                    "keep_alive": "5m",  # Keep ollama model loaded for 5 mins
                    "messages": []
                }

                provider_config = None
                if context is not None:
                    if isinstance(context, dict) and "provider_name" in context:
                        provider_config = context
                    elif isinstance(context, dict) and "provider_config" in context:
                        provider_config = context["provider_config"]

                if provider_config and isinstance(provider_config, dict):
                    if "provider_name" in provider_config:
                        params["llm_provider"] = provider_config["provider_name"]
                    if "api_key" in provider_config:
                        params["llm_api_key"] = provider_config["api_key"]
                    if "base_ip" in provider_config:
                        params["base_ip"] = provider_config["base_ip"]
                    if "port" in provider_config:
                        params["port"] = provider_config["port"]
                    # Merge additional keys carefully
                    for key, value in provider_config.items():
                        if key not in [
                            "provider_name",
                            "llm_model",
                            "api_key",
                            "base_ip",
                            "port",
                            "user_prompt",
                            "system_message",
                        ]:
                            params[key] = value
                        elif key in ["system_message"]:
                            params[key] = value

                    if "user_prompt" in provider_config:
                        params["user_message"] = provider_config["user_prompt"]

                    provider_model = provider_config.get("llm_model", "")
                    params["llm_model"] = provider_model or ""
                elif provider_config:
                    logger.warning(
                        f"provider_config is not a dictionary, it's a {type(provider_config)}. Using defaults."
                    )

                # Finalize model name fallback
                if params.get("llm_provider"):
                    provider = str(params["llm_provider"]).lower()
                    if not params.get("llm_model"):
                        PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-3-opus-20240229"}
                        fallback = PROVIDER_DEFAULTS.get(provider)
                        params["llm_model"] = fallback or self.DEFAULT_MODEL

                # Auto-fetch API key for OpenAI if missing/placeholder
                if provider == "openai" and (not params.get("llm_api_key") or params["llm_api_key"] in {"", "1234", None}):
                    try:
                        params["llm_api_key"] = get_api_key("OPENAI_API_KEY", "openai")
                        logger.info("generate_stream: Retrieved OpenAI API key via get_api_key helper.")
                    except ValueError as _e:
                        logger.warning(f"generate_stream: get_api_key failed – {_e}")

                # --- End Parameter Processing ---

                log_params = {**params}
                if "llm_api_key" in log_params:
                    log_params["llm_api_key"] = "****" if log_params["llm_api_key"] else "None"
                logger.info(
                    f"[Streaming] Initiating LLM stream with params: {log_params} for node {unique_id}"
                )

                # --- Send START message ---
                server.send_sync("llmtoolkit.stream.start", {"node": unique_id}, sid=server.client_id)

                # --- Initiate and process the stream ---
                stream_generator = send_request_stream(
                    llm_provider=params["llm_provider"],
                    base_ip=params.get("base_ip", "localhost"),
                    port=params.get("port", "11434"),
                    llm_model=params["llm_model"],
                    system_message=params["system_message"],
                    user_message=params["user_message"],
                    messages=params["messages"],
                    seed=params.get("seed"),
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    random=params.get("random", False),
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop=params.get("stop"),
                    keep_alive=params.get("keep_alive", True),
                    llm_api_key=params.get("llm_api_key"),
                )

                async for chunk in stream_generator:
                    if chunk:
                        full_response_text += chunk
                        # Send chunk to frontend via websocket
                        server.send_sync(
                            "llmtoolkit.stream.chunk",
                            {"node": unique_id, "text": chunk},
                            sid=server.client_id,
                        )
                    await asyncio.sleep(0.001)

                logger.info(
                    f"[Streaming] Finished for node {unique_id}. Total length: {len(full_response_text)}"
                )
                server.send_sync(
                    "llmtoolkit.stream.end",
                    {"node": unique_id, "final_text": full_response_text},
                    sid=server.client_id,
                )

                # --- Prepare final context output ---
                if context is not None and isinstance(context, dict):
                    context_out = context.copy()
                    context_out["llm_response"] = full_response_text
                    context_out["llm_raw_response"] = {
                        "status": "Streamed successfully",
                        "final_length": len(full_response_text),
                    }
                else:
                    context_out = {
                        "llm_response": full_response_text,
                        "llm_raw_response": {
                            "status": "Streamed successfully",
                            "final_length": len(full_response_text),
                        },
                        "passthrough_data": context,
                    }

                return {"ui": {"string": [full_response_text]}, "result": (context_out,)}

            except Exception as e:
                error_message = f"Error during streaming generation: {str(e)}"
                logger.error(error_message, exc_info=True)
                if server and unique_id:
                    server.send_sync(
                        "llmtoolkit.stream.error",
                        {"node": unique_id, "error": error_message},
                        sid=server.client_id,
                    )
                error_output = {
                    "error": error_message,
                    "partial_response": full_response_text,
                    "original_input": context,
                }
                return {
                    "ui": {"string": [f"Error: {error_message}\nPartial: {full_response_text}"]},
                    "result": (error_output,),
                }
            # Previous async body END

        # Execute the inner coroutine and return its result synchronously
        return run_async(_async_generate())


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LLMToolkitTextGenerator": LLMToolkitTextGenerator, # Keep original
    "LLMToolkitTextGeneratorStream": LLMToolkitTextGeneratorStream # Add streaming version
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMToolkitTextGenerator": "Generate Text (LLMToolkit)",
    "LLMToolkitTextGeneratorStream": "Generate Text Stream (LLMToolkit)" # New display name
}