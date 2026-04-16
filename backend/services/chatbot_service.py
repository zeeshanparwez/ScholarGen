"""
Singleton async chatbot service.
Initialised once at app startup via FastAPI lifespan, then reused for all requests.
Each user gets their own LangGraph thread_id so memory is isolated per user.
Supports two providers: 'gemini' (default) and 'groq'.
"""

import json
import logging
import os
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from groq import AsyncGroq
from openai import AsyncOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.agent_orchestrator import MCPSessionManager
from backend.core.course_retriever import CourseRetriever, CourseTool
from backend.core.key_manager import get_all_keys
from backend.core.azure_llm import get_azure_chat_llm, is_azure_configured

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_BASE, "Config", ".env"))

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are UpskillOS, an AI-powered upskilling assistant designed to help professionals grow their careers and close skill gaps.

Your Mission: Help professionals identify skill gaps, build personalised learning paths, discover industry insights, and connect with their talent network.

Your Tools:
- find_nptel_courses: Recommend courses for any skill or technology
- search_papers: Find relevant research papers and industry insights
- fetch: Retrieve and analyse content from websites and articles
- get_transcript: Get transcripts from YouTube talks, lectures, and demos
- extract_info: Get detailed information about specific research papers
- search_cached_papers: Semantically search all papers fetched in this session

Your Capabilities:
- Build personalised learning paths based on current role and career goals
- Identify skill gaps between where someone is and where they want to be
- Recommend the most relevant courses for any technology or domain
- Surface the latest industry trends and research
- Generate skill assessments and practice questions on any topic
- Summarise articles, papers, and video content
- Guide career transitions and technology adoption decisions

CRITICAL FORMATTING RULES — follow these exactly:
1. **Always include links.** When tool results contain URLs, you MUST include them as clickable markdown links in your response. Format: [Course Title](https://...) or [Paper Title](https://...). NEVER drop a URL from your response.
2. **Course format:** Present each course as: **[Course Name](URL)** — one-line description.
3. **Paper format:** Present each paper as: **[Paper Title](URL)** — one-line summary.
4. If a tool returns courses or papers, list ALL of them with their links — do not summarise them without links.

Your Personality:
- Professional, direct, and results-oriented
- Speak to professionals, not students — assume competence
- Always ground recommendations in practical, career-relevant outcomes
- Use markdown formatting: **bold** for emphasis, bullet lists for steps, headers for sections
- Be concise — professionals value their time

Do not ask questions before answering. Provide value immediately, then ask for clarification if needed."""

# Shorter prompt for Groq (free tier has 8K TPM limit — tool schemas are large)
GROQ_SYSTEM_PROMPT = """You are UpskillOS, an AI upskilling assistant for professionals.
Help with: skill gaps, learning paths, course recommendations, career transitions, industry trends.
Use find_nptel_courses to recommend courses. Be concise and professional."""


def _build_gemini_llm():
    """Create Gemini LLM, rotating through backup keys on failure."""
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    keys = get_all_keys()
    for i, key in enumerate(keys):
        try:
            llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=key,
                temperature=0.7,
                max_retries=0,
                model_kwargs={
                    "generation_config": {
                        "thinking_config": {"thinking_budget": 0}
                    }
                },
            )
            logger.info("Gemini LLM ready (key %d, model: %s)", i + 1, gemini_model)
            return llm
        except Exception as exc:
            logger.warning("Gemini key %d failed: %s", i + 1, exc)
            if i == len(keys) - 1:
                raise
    raise RuntimeError("All Gemini API keys failed")


def _build_groq_client() -> AsyncGroq:
    """Create async Groq client."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return AsyncGroq(api_key=groq_key)


class ChatbotService:
    def __init__(self):
        self._session_manager: MCPSessionManager | None = None
        self._agents: dict = {}                       # provider → LangGraph ReAct agent
        self._groq_client: AsyncGroq | None = None    # direct Groq client (8K TPM, no tools)
        self._nim_client: AsyncOpenAI | None = None   # NVIDIA NIM direct client
        self._nim_chat_models: dict[str, str] = {}    # provider_id → model name (no-tool NIM models)
        self._groq_history: dict[str, list] = {}      # username → message list
        self._nim_history: dict[str, list] = {}       # username → message list
        self._memory = MemorySaver()
        self._user_threads: dict[str, str] = {}
        self._ready = False
        self._available_providers: list[str] = []

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def available_providers(self) -> list[str]:
        return self._available_providers

    async def initialize(self):
        """Called once at app startup. Builds one agent per available provider."""
        logger.info("Initializing ChatbotService...")
        self._session_manager = MCPSessionManager()
        mcp_tools = await self._session_manager.connect_to_servers()

        try:
            course_tool = CourseTool(CourseRetriever()).tool()
            all_tools = mcp_tools + [course_tool]
            logger.info("CourseRetriever ready")
        except Exception as e:
            logger.warning("CourseRetriever failed, continuing without it: %s", e)
            all_tools = mcp_tools

        gemini_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # ── Azure OpenAI agent (PRIMARY — LangGraph ReAct with full tool support) ──
        azure_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        if is_azure_configured():
            try:
                azure_llm = get_azure_chat_llm(temperature=0.0)
                self._agents["azure"] = create_react_agent(
                    azure_llm, all_tools, prompt=azure_prompt, checkpointer=self._memory,
                )
                self._available_providers.append("azure")
                logger.info("Azure OpenAI agent ready (deployment: %s)", os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""))
            except Exception as e:
                logger.warning("Azure OpenAI agent skipped: %s", e)
        else:
            logger.info("Azure OpenAI not configured — skipping (set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)")

        # ── Gemini agent (LangGraph ReAct with all tools) ─────────────────────
        try:
            gemini_llm = _build_gemini_llm()
            self._agents["gemini"] = create_react_agent(
                gemini_llm, all_tools, prompt=gemini_prompt, checkpointer=self._memory,
            )
            self._available_providers.append("gemini")
            logger.info("Gemini agent ready")
        except Exception as e:
            logger.warning("Gemini agent skipped: %s", e)

        # ── Groq (direct AsyncGroq client)
        try:
            self._groq_client = _build_groq_client()
            self._available_providers.append("groq")
            logger.info("Groq client ready (model: %s)", os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
        except Exception as e:
            logger.warning("Groq client skipped: %s", e)

        # ── NVIDIA NIM — tool-capable models as LangGraph agents, others as direct chat
        nim_key = os.environ.get("NIM_API_KEY", "")
        nim_base = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if nim_key:
            # These models support function/tool calling via NIM
            NIM_TOOL_MODELS = {
                "nim":        os.environ.get("NIM_MODEL",        "meta/llama-3.3-70b-instruct"),
                "nim_llama4": os.environ.get("NIM_LLAMA4_MODEL", "meta/llama-4-maverick-17b-128e-instruct"),
            }
            # These are reasoning models that do NOT support tool use on NIM
            NIM_CHAT_ONLY = {
                "nim_deepseek": os.environ.get("NIM_DEEPSEEK_MODEL", "deepseek-ai/deepseek-r1-distill-qwen-32b"),
                "nim_qwq":      os.environ.get("NIM_QWQ_MODEL",      "qwen/qwq-32b"),
            }

            nim_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ])
            for nim_id, nim_model in NIM_TOOL_MODELS.items():
                try:
                    nim_llm = ChatOpenAI(
                        base_url=nim_base,
                        api_key=nim_key,
                        model=nim_model,
                        temperature=0.7,
                        max_retries=1,
                    )
                    self._agents[nim_id] = create_react_agent(
                        nim_llm, all_tools, prompt=nim_prompt, checkpointer=self._memory,
                    )
                    self._available_providers.append(nim_id)
                    logger.info("NIM agent (with tools): %s → %s", nim_id, nim_model)
                except Exception as e:
                    logger.warning("NIM agent %s skipped: %s", nim_id, e)

            # Chat-only reasoning models — direct stream, no tools
            for nim_id, nim_model in NIM_CHAT_ONLY.items():
                self._nim_chat_models[nim_id] = nim_model
                self._available_providers.append(nim_id)
                logger.info("NIM chat-only (no tools): %s → %s", nim_id, nim_model)

            # Direct async client for chat-only models and fallback
            try:
                self._nim_client = AsyncOpenAI(base_url=nim_base, api_key=nim_key)
            except Exception as e:
                logger.warning("NIM async client skipped: %s", e)
        else:
            logger.warning("NIM_API_KEY not set — NIM agents skipped")

        if not self._agents and not self._groq_client and not self._nim_client:
            raise RuntimeError("No LLM providers could be initialised")

        self._ready = True
        logger.info(
            "ChatbotService ready | providers=%s | tools=%d: %s",
            self._available_providers,
            len(all_tools),
            [t.name for t in all_tools],
        )

    def _get_config(self, username: str, provider: str) -> dict:
        key = f"{provider}:{username}"
        if key not in self._user_threads:
            self._user_threads[key] = str(uuid.uuid4())
        return {
            "configurable": {"thread_id": self._user_threads[key]},
            "recursion_limit": 80,
        }

    async def stream_chat(
        self, username: str, message: str, provider: str = "gemini"
    ) -> AsyncGenerator[str, None]:
        """
        Yields SSE-formatted data lines.
        Format: data: <json>\\n\\n
        Types: token | tool_call | done | error
        provider: 'gemini' (LangGraph ReAct agent) or 'groq' (direct AsyncGroq)
        """
        if not self._ready:
            yield 'data: {"type":"error","content":"Service not ready yet"}\n\n'
            return

        if provider == "groq":
            async for chunk in self._stream_groq(username, message):
                yield chunk
        elif provider in self._nim_chat_models and self._nim_client:
            # Reasoning models (DeepSeek R1, QwQ) — no tool support, direct chat
            async for chunk in self._stream_nim(username, message, model=self._nim_chat_models[provider]):
                yield chunk
        elif provider in self._agents:
            # Gemini + tool-capable NIM models (Llama 3.3, Llama 4) via LangGraph
            async for chunk in self._stream_gemini(username, message, provider):
                yield chunk
        else:
            err = json.dumps({"type": "error", "content": f"Provider '{provider}' is not available."})
            yield f"data: {err}\n\n"
            yield f'data: {json.dumps({"type": "done"})}\n\n'

    async def _stream_gemini(
        self, username: str, message: str, provider: str
    ) -> AsyncGenerator[str, None]:
        """Stream via LangGraph ReAct agent (Azure / Gemini / NIM). Hard 60s timeout."""
        if provider not in self._agents:
            # Prefer azure → gemini → whatever is available
            fallback_order = ["azure", "gemini"] + list(self._agents.keys())
            provider = next((p for p in fallback_order if p in self._agents), None)
            if provider is None:
                yield 'data: {"type":"error","content":"No LLM provider available"}\n\n'
                yield 'data: {"type":"done"}\n\n'
                return
            logger.warning("Provider not available, using %s", provider)

        agent = self._agents[provider]
        config = self._get_config(username, provider)

        import asyncio

        async def _run():
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    if content:
                        yield json.dumps({'type': 'token', 'content': content})
                elif kind == "on_tool_start":
                    yield json.dumps({'type': 'tool_call', 'tool': event.get('name', 'tool'), 'status': 'start'})
                elif kind == "on_tool_end":
                    yield json.dumps({'type': 'tool_call', 'tool': event.get('name', 'tool'), 'status': 'end'})

        try:
            # Collect with a per-event 60s watchdog
            async def with_timeout():
                async for payload in _run():
                    yield payload

            async for payload in with_timeout():
                yield f"data: {payload}\n\n"

        except asyncio.TimeoutError:
            logger.warning("Gemini stream timed out for user %s", username)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out (60s). The model may be overloaded — try switching to Groq.'})}\n\n"
        except Exception as e:
            logger.error("Stream error for user %s (provider=%s): %s", username, provider, e)
            err_msg = str(e)
            if "503" in err_msg:
                err_msg = "Gemini is currently overloaded (503). Please switch to Groq using the model selector."
            yield f"data: {json.dumps({'type': 'error', 'content': err_msg})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def _stream_groq(
        self, username: str, message: str
    ) -> AsyncGenerator[str, None]:
        """Stream directly via AsyncGroq (no LangGraph overhead). Hard 60s timeout."""
        import asyncio

        if not self._groq_client:
            yield 'data: {"type":"error","content":"Groq not configured"}\n\n'
            yield 'data: {"type":"done"}\n\n'
            return

        # Maintain per-user conversation history (last 20 messages)
        history = self._groq_history.setdefault(username, [])
        history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": GROQ_SYSTEM_PROMPT}] + history[-20:]

        try:
            groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            stream = await asyncio.wait_for(
                self._groq_client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=4096,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=True,
                ),
                timeout=60,
            )
            full_reply = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_reply += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            if full_reply:
                history.append({"role": "assistant", "content": full_reply})
            if len(history) > 20:
                self._groq_history[username] = history[-20:]

        except asyncio.TimeoutError:
            logger.warning("Groq stream timed out for user %s", username)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out (60s). Please try again.'})}\n\n"
        except Exception as e:
            logger.error("Groq stream error for user %s: %s", username, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def _stream_nim(
        self, username: str, message: str, model: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream via NVIDIA NIM (OpenAI-compatible). Hard 60s timeout."""
        import asyncio

        if not self._nim_client:
            yield 'data: {"type":"error","content":"NVIDIA NIM not configured"}\n\n'
            yield 'data: {"type":"done"}\n\n'
            return

        history = self._nim_history.setdefault(username, [])
        history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": GROQ_SYSTEM_PROMPT}] + history[-20:]

        try:
            nim_model = model or os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")
            stream = await asyncio.wait_for(
                self._nim_client.chat.completions.create(
                    model=nim_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                    stream=True,
                ),
                timeout=60,
            )
            full_reply = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_reply += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            if full_reply:
                history.append({"role": "assistant", "content": full_reply})
            if len(history) > 20:
                self._nim_history[username] = history[-20:]

        except asyncio.TimeoutError:
            logger.warning("NIM stream timed out for user %s", username)
            yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out (60s). Please try again.'})}\n\n"
        except Exception as e:
            logger.error("NIM stream error for user %s: %s", username, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def clear_memory(self, username: str):
        """Start fresh conversations for the user across all providers."""
        for provider in self._agents:
            key = f"{provider}:{username}"
            self._user_threads[key] = str(uuid.uuid4())
        self._groq_history.pop(username, None)
        self._nim_history.pop(username, None)

    async def cleanup(self):
        if self._session_manager:
            await self._session_manager.cleanup()


# Singleton — imported by routers
chatbot_service = ChatbotService()
