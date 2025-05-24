"""
MCP (Modular Cooperative Protocol) integration utilities for the ComfyUI-LLM-Toolkit.

This module becomes the bridge between ComfyUI workflows and the MCP world.
It offers three key building blocks:

  • MCPManager – a *singleton* keeping
        – ComfyUI workflows exposed as MCP tools (internal tools)
        – Configured external MCP servers (via `mcp_servers.json`)
  • A user-friendly global alias `mcp_manager` (importable directly)
  • Two aiohttp endpoints (`/llmtoolkit/mcp/tools/list`, `/llmtoolkit/mcp/tools/call`)
    registered on ComfyUI's `PromptServer` so external clients can discover and
    call our tools using raw HTTP + JSON-RPC 2.

The code is defensive: if either `mcp` (the python SDK) *or* ComfyUI's
`PromptServer`/`aiohttp` is missing, the rest of the toolkit still loads –
only the MCP functionality is disabled and a warning is logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

# ────────────────────────────────────────────────────────────────────────────
#  Optional dependencies – MCP SDK
# ────────────────────────────────────────────────────────────────────────────
try:
    from mcp import ClientSession  # type: ignore
    from mcp.client.sse import sse_client  # type: ignore
    from mcp.client.websocket import websocket_client  # type: ignore
    from mcp.errors import McpError  # type: ignore
    from mcp.types import (
        Tool as MCPToolFormat,
        TextContent,
        CallToolRequest,
        CallToolResult,
        ListToolsResult,
        JSONRPCError,
        JSONRPCRequest,
        JSONRPCResponse,
        ServerResult,
        ErrorData,
    )

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover – allow import without MCP installed
    MCP_AVAILABLE = False

    class ClientSession:  # type: ignore
        pass

    class MCPToolFormat:  # type: ignore
        pass

    class TextContent:  # type: ignore
        pass

    def sse_client(*_a, **_kw):  # type: ignore
        raise NotImplementedError("mcp SDK not installed")

    def websocket_client(*_a, **_kw):  # type: ignore
        raise NotImplementedError("mcp SDK not installed")

    class McpError(Exception):
        """Fallback error when MCP SDK is missing."""


# ────────────────────────────────────────────────────────────────────────────
#  Optional dependencies – aiohttp / PromptServer (ComfyUI runtime only)
# ────────────────────────────────────────────────────────────────────────────
try:
    from aiohttp import web  # type: ignore
    from server import PromptServer  # type: ignore

    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover – running outside ComfyUI
    AIOHTTP_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────
#  Logger setup
# ────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("LLMToolkit.MCPManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ────────────────────────────────────────────────────────────────────────────
#  Paths / constants
# ────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MCP_CONFIG_FILE = os.path.join(_THIS_DIR, "mcp_servers.json")


# ────────────────────────────────────────────────────────────────────────────
#  MCPManager – singleton
# ────────────────────────────────────────────────────────────────────────────
class MCPManager:
    """Central registry, discovery and execution router for MCP tools."""

    _instance: Optional["MCPManager"] = None

    # ------------------------------------------------------------------
    def __init__(self) -> None:  # noqa: D401 – simple
        if MCPManager._instance is not None:
            raise RuntimeError("MCPManager already instantiated – use .get_instance().")

        # Internal ComfyUI workflows – {name: info_dict}
        self.comfy_tools: Dict[str, Dict[str, Any]] = {}
        # External MCP servers – {server_name: cfg_dict}
        self.external_servers: Dict[str, Dict[str, Any]] = {}

        self._lock = asyncio.Lock()
        self._load_external_server_config()

        MCPManager._instance = self
        logger.info("MCPManager initialised – %d external servers configured", len(self.external_servers))

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "MCPManager":
        if cls._instance is None:
            cls._instance = MCPManager()
        return cls._instance

    # ------------------------------------------------------------------
    #  External server config
    # ------------------------------------------------------------------
    def _load_external_server_config(self, cfg_path: str = DEFAULT_MCP_CONFIG_FILE) -> None:
        if not os.path.isfile(cfg_path):
            logger.debug("mcp_servers.json not found – no external servers")
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s – %s", cfg_path, exc, exc_info=True)
            return

        servers = payload.get("mcpServers")
        if not isinstance(servers, dict):
            logger.warning("mcp_servers.json missing 'mcpServers' dict – ignoring file")
            return

        self.external_servers = servers
        logger.info("Loaded %d external MCP servers from config", len(self.external_servers))

    # ------------------------------------------------------------------
    #  Public API – registration / querying / execution
    # ------------------------------------------------------------------
    async def register_comfy_workflow_tool(
        self,
        *,
        name: str,
        description: str,
        workflow_json: str,
        input_map: Dict[str, Any],
        output_map: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a ComfyUI workflow to the internal MCP registry."""
        async with self._lock:
            if schema is None:
                schema = {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in input_map},
                    "required": list(input_map.keys()),
                }
            self.comfy_tools[name] = {
                "description": description,
                "workflow_json": workflow_json,
                "input_map": input_map,
                "output_map": output_map,
                "schema": schema,
            }
            logger.info("Registered internal MCP tool '%s'", name)

    # .................................................................
    async def list_available_tools(self, *, target_format: str = "openai") -> List[Dict[str, Any]]:
        """Return internal + external tools formatted for the requested LLM provider."""
        out: List[Dict[str, Any]] = []

        # -------- internal tools --------
        async with self._lock:
            for name, info in self.comfy_tools.items():
                if target_format == "openai":
                    out.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": info.get("description", ""),
                                "parameters": info.get("schema", {"type": "object", "properties": {}}),
                            },
                        }
                    )
                elif target_format == "mcp" and MCP_AVAILABLE:
                    out.append(
                        MCPToolFormat(
                            name=name,
                            description=info.get("description", ""),
                            inputSchema=info.get("schema", {"type": "object", "properties": {}}),
                        ).model_dump(exclude_none=True)
                    )
                else:
                    out.append({"name": name, "description": info.get("description", "")})

        # -------- external tools --------
        if MCP_AVAILABLE:
            for srv_name in self.external_servers:
                try:
                    ext_tools = await self._list_tools_external(srv_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Listing tools on '%s' failed – %s", srv_name, exc)
                    continue

                prefix = f"{srv_name}_"
                for t in ext_tools:
                    t.name = prefix + t.name
                    if target_format == "openai":
                        out.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": t.name,
                                    "description": t.description or "",
                                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                                },
                            }
                        )
                    elif target_format == "mcp":
                        out.append(t.model_dump(exclude_none=True))
                    else:
                        out.append({"name": t.name, "description": t.description or ""})

        logger.info("list_available_tools(format=%s) → %d tools", target_format, len(out))
        return out

    # .................................................................
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute *tool_name* (internal or external)."""
        # Internal?
        async with self._lock:
            if tool_name in self.comfy_tools:
                return await self._execute_comfy_workflow(tool_name, arguments)

        # External?
        for srv_name in self.external_servers:
            prefix = f"{srv_name}_"
            if tool_name.startswith(prefix):
                real = tool_name[len(prefix) :]
                return await self._execute_external_tool(srv_name, real, arguments)

        logger.error("Tool '%s' not found in registry", tool_name)
        return {"error": f"Tool '{tool_name}' not found."}

    # ------------------------------------------------------------------
    #  Internal workflow execution – placeholder for Phase-1
    # ------------------------------------------------------------------
    async def _execute_comfy_workflow(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        info = self.comfy_tools[tool_name]
        logger.info("[MCP] Running ComfyUI workflow '%s' (placeholder)", tool_name)
        # TODO: Implement actual execution by calling /prompt etc.
        await asyncio.sleep(0.1)
        return {
            "tool": tool_name,
            "status": "success",
            "inputs": arguments,
            "note": "This is a placeholder result. Real workflow execution coming soon.",
        }

    # ------------------------------------------------------------------
    #  External helpers
    # ------------------------------------------------------------------
    async def _get_external_client_session(self, server_name: str) -> ClientSession:
        cfg = self.external_servers[server_name]
        url = cfg["url"]
        transport = cfg.get("transport", "sse").lower()
        auth_token = cfg.get("auth_token")

        if transport == "sse":
            ctx = sse_client(url, headers={"Authorization": f"Bearer {auth_token}"} if auth_token else None)
        elif transport in {"ws", "websocket"}:
            ctx = websocket_client(url, headers={"Authorization": f"Bearer {auth_token}"} if auth_token else None)
        else:
            raise RuntimeError(f"Unsupported transport '{transport}' for server '{server_name}'.")

        streams = await ctx.__aenter__()
        return ClientSession(*streams)

    # .................................................................
    async def _list_tools_external(self, server_name: str) -> List[MCPToolFormat]:
        if not MCP_AVAILABLE:
            return []
        async with await self._get_external_client_session(server_name) as sess:
            await sess.initialize()
            res = await sess.list_tools()
            return res.tools

    # .................................................................
    async def _execute_external_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not MCP_AVAILABLE:
            return {"error": "MCP SDK not installed."}

        try:
            async with await self._get_external_client_session(server_name) as sess:
                await sess.initialize()
                res = await sess.call_tool(tool_name, arguments)

                if res.isError:
                    txt = "Unknown error"
                    if res.content and isinstance(res.content[0], TextContent):
                        txt = res.content[0].text
                    return {"error": txt}

                out: List[str] = []
                for item in res.content:
                    if isinstance(item, TextContent):
                        out.append(item.text)
                return "\n".join(out) if out else ""
        except McpError as exc:
            logger.error("MCP error: %s", exc)
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.error("Exception calling external tool: %s", exc, exc_info=True)
            return {"error": str(exc)}


# ────────────────────────────────────────────────────────────────────────────
#  Convenience global alias
# ────────────────────────────────────────────────────────────────────────────
try:
    mcp_manager: MCPManager = MCPManager.get_instance()
except Exception:  # pragma: no cover
    mcp_manager = None  # type: ignore

# ────────────────────────────────────────────────────────────────────────────
#  aiohttp routes (only when inside ComfyUI runtime with PromptServer)
# ────────────────────────────────────────────────────────────────────────────
if AIOHTTP_AVAILABLE and MCP_AVAILABLE:

    def _register_routes() -> None:
        srv = PromptServer.instance
        if srv is None:
            logger.warning("PromptServer not ready yet – delaying MCP route registration")
            return

        routes = srv.routes

        # /llmtoolkit/mcp/tools/list (GET)
        @routes.get("/llmtoolkit/mcp/tools/list")
        async def _list(request: web.Request) -> web.Response:  # noqa: D401
            req_id = request.query.get("id", "list_tools")
            try:
                tools = await mcp_manager.list_available_tools(target_format="mcp")
                res = ListToolsResult(tools=tools)
                payload = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=req_id,
                    result=ServerResult(root=res).model_dump(mode="json", exclude_none=True),
                )
                return web.json_response(payload.model_dump(mode="json", exclude_none=True))
            except Exception as exc:  # noqa: BLE001
                logger.error("/tools/list failed – %s", exc, exc_info=True)
                err = JSONRPCError(jsonrpc="2.0", id=req_id, error=ErrorData(code=-32000, message=str(exc)))
                return web.json_response(err.model_dump(mode="json", exclude_none=True), status=500)

        # /llmtoolkit/mcp/tools/call (POST)
        @routes.post("/llmtoolkit/mcp/tools/call")
        async def _call(request: web.Request) -> web.Response:  # noqa: D401
            req_id = "unknown"
            try:
                data = await request.json()
                rpc_req = JSONRPCRequest.model_validate(data)
                req_id = rpc_req.id

                call_req = CallToolRequest.model_validate(data)
                name = call_req.params.name
                args = call_req.params.arguments or {}

                result_content = await mcp_manager.execute_tool(name, args)
                is_err = isinstance(result_content, dict) and "error" in result_content

                if is_err:
                    res = CallToolResult(content=[TextContent(type="text", text=str(result_content["error"]))], isError=True)
                else:
                    res = CallToolResult(content=[TextContent(type="text", text=json.dumps(result_content) if not isinstance(result_content, str) else result_content)], isError=False)

                payload = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=req_id,
                    result=ServerResult(root=res).model_dump(mode="json", exclude_none=True),
                )
                return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200 if not is_err else 500)
            except Exception as exc:  # noqa: BLE001
                logger.error("/tools/call failed – %s", exc, exc_info=True)
                err = JSONRPCError(jsonrpc="2.0", id=req_id, error=ErrorData(code=-32000, message=str(exc)))
                return web.json_response(err.model_dump(mode="json", exclude_none=True), status=500)

        logger.info("MCP HTTP routes registered under /llmtoolkit/mcp/*")

    # PromptServer may not yet be ready when this module is imported (ComfyUI loads
    # nodes first). We therefore try now and, on failure, schedule a background task.
    try:
        _register_routes()
    except Exception:  # pragma: no cover
        async def _later():
            while PromptServer.instance is None:
                await asyncio.sleep(0.5)
            _register_routes()

        asyncio.ensure_future(_later())
else:
    if not MCP_AVAILABLE:
        logger.info("MCP SDK not installed – MCP endpoints disabled.")
    elif not AIOHTTP_AVAILABLE:
        logger.info("aiohttp/PromptServer not available – MCP endpoints disabled.") 