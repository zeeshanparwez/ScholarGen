import asyncio
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from contextlib import AsyncExitStack
from typing import Any, Dict, List
from dotenv import load_dotenv

# LangGraph imports for memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

from backend.core.course_retriever import CourseRetriever, CourseTool

# Project root is 1 level up from this file (backend/ → ScholarGen/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, "Config", ".env"))

class MCPToolLogger(BaseTool):
    """MCP tool wrapper with detailed logging."""

    def __init__(self, tool_name: str, tool_description: str, session: ClientSession):
        super().__init__(
            name=tool_name,
            description=tool_description
        )
        self._session = session
        self._tool_name = tool_name

    def _run(self, **kwargs) -> str:
        logger.debug("TOOL CALLED: %s | params: %s", self._tool_name, kwargs)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._arun(**kwargs))
            logger.debug("TOOL RESULT (%s): %s", self._tool_name, str(result)[:200])
            return result
        except Exception as e:
            error_msg = f"Error executing {self._tool_name}: {str(e)}"
            logger.error("TOOL ERROR (%s): %s", self._tool_name, error_msg)
            return error_msg

    async def _arun(self, **kwargs) -> str:
        try:
            clean_kwargs = self._prepare_arguments(kwargs)
            logger.debug("TOOL FINAL PARAMS (%s): %s", self._tool_name, clean_kwargs)
            result = await self._session.call_tool(self._tool_name, arguments=clean_kwargs)

            if result and result.content:
                if hasattr(result.content, 'text'):
                    return result.content.text
                else:
                    return str(result.content)
            return "Tool executed but returned no content"

        except Exception as e:
            raise Exception(f"MCP call failed: {str(e)}")

    def _prepare_arguments(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Unwrap nested kwargs
        if isinstance(kwargs, dict) and 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            kwargs = kwargs['kwargs']

        if self._tool_name == "fetch":
            if 'url' not in kwargs:
                for key, value in kwargs.items():
                    if isinstance(value, str) and ('http' in value.lower()):
                        return {'url': value}
            return kwargs
        elif self._tool_name == "search_papers":
            if 'topic' not in kwargs:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        return {'topic': value, 'max_results': kwargs.get('max_results', 5)}
            return kwargs
        elif self._tool_name == "get_transcript":
            if 'url' not in kwargs:
                for key, value in kwargs.items():
                    if isinstance(value, str) and ('youtu' in value.lower()):
                        return {'url': value}
            return kwargs
        elif self._tool_name == "extract_info":
            if 'paper_id' not in kwargs:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        return {'paper_id': value}
            return kwargs

        return kwargs

class MCPSessionManager:
    """Manages persistent MCP sessions."""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.mcp_tools = []

    async def connect_to_servers(self):
        """Connect to MCP servers and create tools. Cleans up on partial failure."""
        _uv = os.environ.get("UV_PATH", "uv")
        _uvx = os.environ.get("UVX_PATH", "uvx")
        server_configs = [
            ("research", StdioServerParameters(command=_uv, args=["run", "backend/mcp/research_mcp.py"])),
            ("youtube",  StdioServerParameters(command=_uv, args=["run", "backend/mcp/youtube_mcp.py"])),
            ("fetch",    StdioServerParameters(command=_uvx, args=["mcp-server-fetch"])),
        ]

        logger.info("Connecting to MCP servers...")

        try:
            for server_name, server_params in server_configs:
                try:
                    stdio_transport = await self.exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()

                    self.sessions[server_name] = session
                    tools_response = await session.list_tools()

                    for tool in tools_response.tools:
                        wrapped_tool = MCPToolLogger(
                            tool_name=tool.name,
                            tool_description=tool.description or f"MCP tool: {tool.name}",
                            session=session,
                        )
                        self.mcp_tools.append(wrapped_tool)

                    logger.info("Connected to %s: %s", server_name, [t.name for t in tools_response.tools])

                except Exception as e:
                    logger.warning("Failed to connect to %s: %s", server_name, e)

        except Exception:
            await self.exit_stack.aclose()
            raise

        return self.mcp_tools

    async def cleanup(self):
        await self.exit_stack.aclose()
