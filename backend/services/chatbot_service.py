"""
Singleton async chatbot service.
Initialised once at app startup via FastAPI lifespan, then reused for all requests.
Each user gets their own LangGraph thread_id so memory is isolated per user.
"""

import logging
import os
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.agent_orchestrator import MCPSessionManager
from backend.core.course_retriever import CourseRetriever, CourseTool

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_BASE, "Config", ".env"))

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are EduAssist, an AI personal assistant designed to make students' lives easier and enhance their learning experience.

Your Mission: Help students succeed academically by providing personalized assistance, educational resources, and study support.

Your Tools:
- find_nptel_courses: Recommend NPTEL courses for any learning topic
- search_papers: Find relevant research papers and academic content
- fetch: Retrieve and analyze content from websites and articles
- get_transcript: Get transcripts from educational YouTube videos
- extract_info: Get detailed information about specific research papers
- search_cached_papers: Semantically search all papers fetched in this session

Your Capabilities:
- Find and recommend courses, papers, and learning resources
- Clarify complex concepts with step-by-step explanations
- Help with academic research and finding credible sources
- Generate quizzes and practice questions on any topic
- Summarize articles, papers, and video content
- Guide career and course selection decisions

Your Personality:
- Friendly, encouraging, and patient
- Adapt explanations to the student's level
- Always provide practical examples
- Ask clarifying questions only when absolutely necessary

Do not ask questions before answering. Provide value immediately, then ask for clarification if needed."""


class ChatbotService:
    def __init__(self):
        self._session_manager: MCPSessionManager | None = None
        self._agent = None
        self._memory = MemorySaver()
        self._user_threads: dict[str, str] = {}
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def initialize(self):
        """Called once at app startup. Blocks until all MCP servers are connected."""
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

        llm = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.7,
            max_retries=2,
            model_kwargs={
                "generation_config": {
                    "thinking_config": {"thinking_budget": 0}
                }
            },
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self._agent = create_react_agent(
            llm,
            all_tools,
            prompt=prompt,
            checkpointer=self._memory,
        )

        self._ready = True
        logger.info("ChatbotService ready with %d tools: %s", len(all_tools), [t.name for t in all_tools])

    def _get_config(self, username: str) -> dict:
        if username not in self._user_threads:
            self._user_threads[username] = str(uuid.uuid4())
        return {"configurable": {"thread_id": self._user_threads[username]}}

    async def stream_chat(self, username: str, message: str) -> AsyncGenerator[str, None]:
        """
        Yields SSE-formatted data lines.
        Format: data: <json>\n\n
        Types: token | tool_call | done | error
        """
        if not self._ready or not self._agent:
            yield 'data: {"type":"error","content":"Service not ready yet"}\n\n'
            return

        config = self._get_config(username)

        try:
            async for event in self._agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content
                    # Gemini may return list chunks
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    if content:
                        import json
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                elif kind == "on_tool_start":
                    import json
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.get('name', 'tool'), 'status': 'start'})}\n\n"

                elif kind == "on_tool_end":
                    import json
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.get('name', 'tool'), 'status': 'end'})}\n\n"

        except Exception as e:
            import json
            logger.error("Stream error for user %s: %s", username, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        import json
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def clear_memory(self, username: str):
        """Start a fresh conversation for the user."""
        self._user_threads[username] = str(uuid.uuid4())

    async def cleanup(self):
        if self._session_manager:
            await self._session_manager.cleanup()


# Singleton — imported by routers
chatbot_service = ChatbotService()
