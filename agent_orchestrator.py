import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Import your CourseRetriever classes
from course_retriever import CourseRetriever, CourseTool

load_dotenv("./Config/.env")

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
        print(f"\n🔧 TOOL CALLED: {self._tool_name}")
        print(f"📝 RAW PARAMETERS: {kwargs}")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._arun(**kwargs))
            print(f"✅ TOOL RESULT: {result[:200]}{'...' if len(str(result)) > 200 else ''}")
            return result
        except Exception as e:
            error_msg = f"Error executing {self._tool_name}: {str(e)}"
            print(f"❌ TOOL ERROR: {error_msg}")
            return error_msg
    
    async def _arun(self, **kwargs) -> str:
        try:
            clean_kwargs = self._prepare_arguments(kwargs)
            print(f"🔄 FINAL PARAMETERS: {clean_kwargs}")
            
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
        
        # Handle tool-specific parameter mapping (same as before)
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
        """Connect to MCP servers and create tools."""
        server_configs = [
            ("research", StdioServerParameters(command="uv", args=["run", "research_mcp.py"])),
            ("youtube", StdioServerParameters(command="uv", args=["run", "youtube_mcp.py"])),
            ("fetch", StdioServerParameters(command="uvx", args=["mcp-server-fetch"]))
        ]
        
        print("🚀 Connecting to MCP servers...")
        
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
                        session=session
                    )
                    self.mcp_tools.append(wrapped_tool)
                
                print(f"✅ Connected to {server_name}: {[t.name for t in tools_response.tools]}")
                
            except Exception as e:
                print(f"❌ Failed to connect to {server_name}: {e}")
        
        return self.mcp_tools
    
    async def cleanup(self):
        await self.exit_stack.aclose()

class ChatbotWithMemory:
    """Chatbot with persistent memory using LangGraph."""
    
    def __init__(self, tools: List[BaseTool]):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_retries=2
        )
        self.tools = tools
        
        # Create memory checkpoint
        self.memory = MemorySaver()
        
        system_prompt = """You are EduAssist, an AI personal assistant specifically designed to make students' lives easier and enhance their learning experience.

🎯 **Your Mission**: Help students succeed academically by providing personalized assistance, educational resources, and study support.

🛠️ **Your Tools**:
- **find_nptel_courses**: Recommend NPTEL courses for any learning topic
- **search_papers**: Find relevant research papers and academic content  
- **fetch**: Retrieve and analyze content from websites, articles, and resources
- **get_transcript**: Get transcripts from educational videos for study
- **extract_info**: Get detailed information about specific research papers

👨‍🎓 **Your Capabilities**:
- Create custom quizzes and practice questions on any topic
- Clarify complex concepts and provide detailed explanations
- Help with academic research and finding credible sources
- Suggest study schedules and learning paths
- Answer doubts with patience and multiple explanation approaches
- Provide real-world examples to make concepts clearer
- Help with daily academic tasks like note-taking, summarization
- Guide career and course selection decisions

💡 **Your Personality**:
- Friendly, encouraging, and patient
- Always provide step-by-step explanations
- Adapt your communication style to the student's level
- Celebrate learning progress and achievements
- Ask clarifying questions when needed
- Provide multiple learning resources when possible

📚 **How You Help**:
1. **For learning**: Find courses, papers, and create study materials
2. **For doubts**: Break down complex topics into simple explanations
3. **For research**: Search academic papers and summarize key findings
4. **For practice**: Generate quizzes, flashcards, and exercises
5. **For planning**: Suggest learning paths and study schedules

Remember: Every student learns differently. Do not ask quesions before answering the current questions. Your goal is to make learning enjoyable, accessible, and effective."""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        # Create the agent with memory
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self.prompt,
            checkpointer=self.memory
        )
        
        # Thread configuration for memory
        self.thread_config = {"configurable": {"thread_id": "main_conversation"}}
        
        print("🧠 Memory system initialized with persistent checkpointing")
    
    async def chat(self, user_input: str) -> str:
        """Send a message and get response with memory context."""
        try:
            print(f"\n💭 Processing with memory context...")
            
            # Invoke the agent with memory
            response = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=self.thread_config
            )
            
            # Extract the final message
            final_message = response["messages"][-1]
            return final_message.content
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history."""
        try:
            # Get the current state
            state = self.agent.get_state(self.thread_config)
            messages = state.values.get("messages", [])
            
            # Format messages for display
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    def clear_memory(self):
        """Clear the conversation memory."""
        try:
            # Create a new thread ID to effectively clear memory
            import uuid
            new_thread_id = str(uuid.uuid4())
            self.thread_config = {"configurable": {"thread_id": new_thread_id}}
            print("🗑️ Memory cleared - starting fresh conversation")
        except Exception as e:
            print(f"Error clearing memory: {e}")

async def setup_all_tools():
    """Setup both MCP tools and CourseRetriever tool."""
    session_manager = MCPSessionManager()
    mcp_tools = await session_manager.connect_to_servers()
    
    # Setup CourseRetriever tool
    print("🎓 Initializing NPTEL CourseRetriever...")
    try:
        course_retriever = CourseRetriever()
        course_tool_wrapper = CourseTool(course_retriever)
        course_tool = course_tool_wrapper.tool()
        print("✅ CourseRetriever initialized successfully")
        
        all_tools = mcp_tools + [course_tool]
        print(f"🛠️ Total tools available: {[tool.name for tool in all_tools]}")
        
        return all_tools, session_manager
        
    except Exception as e:
        print(f"❌ Failed to initialize CourseRetriever: {e}")
        return mcp_tools, session_manager

async def main():
    """Main function with memory-enabled chatbot."""
    session_manager = None
    
    try:
        print("🤖 Initializing Personalized Assistant with Memory...")
        
        # Setup all tools
        tools, session_manager = await setup_all_tools()
        
        if not tools:
            print("⚠️ No tools available.")
            return
        
        # Create chatbot with memory
        chatbot = ChatbotWithMemory(tools)
        
        # Interactive chat loop with memory
        print(f"\n🎯 Personalized Assistant with Memory started!")
        print(f"🛠️ Available tools: {[tool.name for tool in tools]}")
        print("🧠 Memory: Remembers conversation context across messages")
        print("💡 Try:")
        print("   - 'My name is John' then 'What's my name?'")
        print("   - 'Find courses about ML' then 'Show me more on that topic'")
        print("   - 'fetch https://example.com' then 'What did you just fetch?'")
        print("\nCommands:")
        print("   - 'quit' to exit")
        print("   - 'clear' to clear memory")
        print("   - 'history' to see conversation history")
        print()
        
        while True:
            try:
                user_input = input("🗨️ You: ").strip()
                
                if user_input.lower() == "quit":
                    break
                elif user_input.lower() == "clear":
                    chatbot.clear_memory()
                    continue
                elif user_input.lower() == "history":
                    history = chatbot.get_conversation_history()
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
                        role_emoji = "🗨️" if msg["role"] == "user" else "🤖"
                        print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
                    print()
                    continue
                elif not user_input:
                    continue
                
                print(f"\n🤔 Processing: {user_input}")
                print("=" * 60)
                
                # Get response with memory context
                response = await chatbot.chat(user_input)
                print(f"\n🤖 Assistant: {response}")
                
                print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("\n" + "=" * 60 + "\n")
    
    except Exception as e:
        print(f"💥 Setup error: {e}")
        
    finally:
        if session_manager:
            await session_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
