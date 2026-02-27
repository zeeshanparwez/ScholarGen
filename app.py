import streamlit as st
import threading
import asyncio
import queue
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import tempfile
import shutil
import pandas as pd
import os
import hashlib
import json
import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_orchestrator import setup_all_tools, ChatbotWithMemory
from collaboration import (
    match_similar_users,
    suggest_collaboration_topics,
    update_user_profile_with_timestamp,
)
from flashcards import show_flashcard_modal

warnings.filterwarnings("ignore")

CREDENTIALS_PATH = "Data/user_credentials.xlsx"
PROFILE_PATH = "Data/user_profiles.xlsx"

# ---------------- AUTH HELPERS ----------------


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_credentials():
    if not os.path.exists(CREDENTIALS_PATH):
        return pd.DataFrame(columns=["username", "password_hash"])
    return pd.read_excel(CREDENTIALS_PATH)


def save_credentials(df: pd.DataFrame):
    os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
    df.to_excel(CREDENTIALS_PATH, index=False)


def signup(username: str, password: str):
    df = load_credentials()
    if username in df["username"].values:
        return False, "Username already exists"
    new_row = pd.DataFrame(
        [{"username": username, "password_hash": hash_password(password)}]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    save_credentials(df)
    return True, "Signup successful"


def login(username: str, password: str):
    df = load_credentials()
    if username not in df["username"].values:
        return False, "Username not found"
    stored_hash = df.loc[df["username"] == username, "password_hash"].values[0]
    if hash_password(password) == stored_hash:
        return True, "Login successful"
    return False, "Incorrect password"


# ---------------- LLM FOR PROFILE EXTRACTION ----------------


@st.cache_resource
def get_llm():
    load_dotenv("./Config/.env")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_retries=2)


def extract_profile_from_text(text: str) -> dict:
    """
    Use Gemini via LangChain to extract {interests, skills} as JSON.
    """
    prompt = (
        "Extract user's interests and skills from the following text. "
        "Return ONLY a valid JSON dictionary with keys 'interests' and 'skills'.\n\n"
        f"Text:\n{text}\n\n"
        'Respond with ONLY this format:\n'
        '{"interests": ["AI", "Machine Learning"], "skills": ["Python", "Data Analysis"]}'
    )

    try:
        llm = get_llm()
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])

        response_text = response.content.strip()
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            profile = json.loads(json_str)
        else:
            profile = json.loads(response_text)

        profile.setdefault("interests", [])
        profile.setdefault("skills", [])
        return profile

    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Raw response: {response_text[:200] if 'response_text' in locals() else ''}...")
        return {"interests": [], "skills": []}
    except Exception as e:
        print(f"Profile extraction failed: {e}")
        return {"interests": [], "skills": []}


def save_profiles(df: pd.DataFrame) -> bool:
    """
    Streamlit-safe Excel save with temp file (prevents partial writes).
    """
    directory = os.path.dirname(PROFILE_PATH)
    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xlsx", delete=False, dir=directory
    ) as tmp:
        df.to_excel(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        shutil.move(tmp_path, PROFILE_PATH)
        return True
    except Exception:
        os.unlink(tmp_path)
        return False


# ---------------- ROBUST CHATBOT SERVICE ----------------


class RobustChatbotService:
    """Enhanced chatbot service with timeout handling and debugging."""

    def __init__(self):
        self.chatbot = None
        self.session_manager = None
        self.initialized = False
        self.error = None
        self.initialization_logs = []

        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self._run_service, daemon=True)
        self.thread.start()

    def _run_service(self):
        async def service_loop():
            try:
                self.initialization_logs.append("🚀 Starting initialization...")
                stdout_capture = StringIO()
                stderr_capture = StringIO()

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    tools, session_manager = await asyncio.wait_for(
                        setup_all_tools(), timeout=60
                    )

                    chatbot = ChatbotWithMemory(tools)
                    self.chatbot = chatbot
                    self.session_manager = session_manager
                    self.initialized = True
                    self.initialization_logs.append("✅ Initialization complete")

                while self.running:
                    try:
                        request = self.request_queue.get(timeout=1)
                        if request["type"] == "chat":
                            try:
                                response = await asyncio.wait_for(
                                    self.chatbot.chat(request["message"]), timeout=45
                                )
                                self.response_queue.put(
                                    {"success": True, "response": response}
                                )
                            except asyncio.TimeoutError:
                                self.response_queue.put(
                                    {
                                        "success": False,
                                        "error": "Chat request timed out after 45 seconds",
                                    }
                                )
                            except Exception as e:
                                self.response_queue.put(
                                    {"success": False, "error": f"Chat error: {str(e)}"}
                                )
                        elif request["type"] == "clear":
                            try:
                                self.chatbot.clear_memory()
                                self.response_queue.put(
                                    {"success": True, "response": "Memory cleared"}
                                )
                            except Exception as e:
                                self.response_queue.put(
                                    {
                                        "success": False,
                                        "error": f"Clear error: {str(e)}",
                                    }
                                )
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.response_queue.put(
                            {"success": False, "error": f"Service error: {str(e)}"}
                        )
            except asyncio.TimeoutError:
                self.error = "Initialization timed out after 60 seconds"
            except Exception as e:
                self.error = f"Service initialization failed: {str(e)}"

        try:
            asyncio.run(service_loop())
        except Exception as e:
            self.error = f"Event loop error: {str(e)}"

    def is_ready(self):
        return self.initialized

    def get_error(self):
        return self.error

    def get_logs(self):
        return self.initialization_logs

    def chat(self, message, timeout=50):
        self.request_queue.put({"type": "chat", "message": message})
        try:
            response = self.response_queue.get(timeout=timeout)
            if response["success"]:
                return response["response"], None
            else:
                return None, response["error"]
        except queue.Empty:
            return None, f"Request timed out after {timeout} seconds"

    def clear_memory(self):
        self.request_queue.put({"type": "clear"})
        try:
            response = self.response_queue.get(timeout=10)
            return response.get("success", False)
        except queue.Empty:
            return False

    def get_tools(self):
        if self.chatbot and self.chatbot.tools:
            return [tool.name for tool in self.chatbot.tools]
        return []

    def stop(self):
        self.running = False


@st.cache_resource(show_spinner=False)
def init_chatbot_service():
    return RobustChatbotService()


# ---------------- STREAMLIT APP ----------------


st.set_page_config(
    page_title="🎓 LearnSphere Chatbot", page_icon="🎓", layout="wide"
)

# state flags
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "show_flashcard_dialog" not in st.session_state:
    st.session_state.show_flashcard_dialog = False

# -------- LOGIN / SIGNUP --------
if not st.session_state.logged_in:
    st.title("LearnSphere - Login or Signup")
    option = st.radio("Select option", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            success, message = signup(username.strip(), password)
            if success:
                st.success(message)
            else:
                st.error(message)
    else:
        if st.button("Login"):
            success, message = login(username.strip(), password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error(message)
    st.stop()

st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

st.title("🎓 LearnSphere - Collaborative AI Learning Hub")
st.markdown("*Powered by LangGraph, MCP & Google Gemini*")

# -------- INIT CHATBOT SERVICE --------
service = init_chatbot_service()

if not service.is_ready():
    error = service.get_error()
    if error:
        st.error(f"❌ **Initialization Failed**: {error}")
        with st.expander("🔍 Debug Information"):
            st.write("**Initialization Logs:**")
            for log in service.get_logs():
                st.write(f"• {log}")
            st.write("**Possible Solutions:**")
            st.write(
                """
1. Check your network connection  
2. Verify MCP servers are running  
3. Check Google API key is valid  
4. Restart the application  
5. Check console for detailed error logs
"""
            )
        if st.button("🔄 Retry Initialization"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(60):
            if service.is_ready():
                progress_bar.progress(100)
                status_text.success("✅ Initialization complete!")
                time.sleep(1)
                st.rerun()
                break
            elif service.get_error():
                progress_bar.empty()
                status_text.error(f"❌ Initialization failed: {service.get_error()}")
                break
            progress = min((i + 1) * 100 // 60, 95)
            progress_bar.progress(progress)
            status_text.text(
                f"🚀 Initializing LearnSphere... ({i+1}/60 seconds)"
            )
            time.sleep(1)
        else:
            st.error("❌ Initialization timed out")
            st.stop()

# -------- SIDEBAR --------
st.sidebar.header("📚 GATE Practice")
if st.sidebar.button("🎯 Open Flashcards", use_container_width=True):
    st.session_state.show_flashcard_dialog = True
st.sidebar.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    timeout_setting = st.slider(
        "Request Timeout (seconds)",
        min_value=10,
        max_value=120,
        value=50,
        help="How long to wait for responses",
    )

    if st.button("🔄 Clear Conversation", use_container_width=True):
        if service.clear_memory():
            st.session_state.messages = []
            st.success("✅ Conversation cleared!")
            st.rerun()
        else:
            st.error("❌ Failed to clear memory")

    st.subheader("⚡ Quick Example Queries")
    example_queries = [
        "Find me courses on Machine Learning",
        "List recent research papers on transformers",
        "Explain convolutional neural networks in simple terms",
        "Provide me a summary of the articles from https://www.ibm.com/think/topics/ai-agent-memory",
        "Create a quiz based on the lessons from https://youtu.be/I2wURDqiXdM?si=ytma61TmMeP2YPZX",
    ]
    for query in example_queries:
        if st.button(query, use_container_width=True):
            st.session_state.prompt_override = query
            st.rerun()

    # -------- COLLABORATE BUTTON --------
    if st.sidebar.button("Collaborate"):
        matched_users = match_similar_users(st.session_state.username)
        collaboration_topics = suggest_collaboration_topics(matched_users)

        st.sidebar.markdown("### Matched Users")
        if matched_users:
            for user in matched_users:
                st.sidebar.write(f"- {user}")
        else:
            st.sidebar.write("No matches found yet.")

        st.sidebar.markdown("### Collaboration Topics")
        if collaboration_topics:
            for topic in collaboration_topics:
                st.sidebar.write(f"- {topic}")
        else:
            st.sidebar.write("No topics suggested yet.")

    st.header("🛠️ Available Tools")
    tools = service.get_tools()
    if tools:
        for tool in tools:
            st.write(f"• **{tool}**")
    else:
        st.write("Loading tools...")

    st.header("💡 Tips for Success")
    st.info(
        "**If requests time out:**\n"
        "- Try simpler queries first\n"
        "- Increase timeout in settings\n"
        "- Check your internet connection\n"
        "- Restart the app if needed\n\n"
        "**Best practices:**\n"
        "- Ask specific questions\n"
        "- One topic at a time\n"
        "- Be patient with tool calls"
    )

# -------- FLASHCARD MODAL --------
if st.session_state.show_flashcard_dialog:
    show_flashcard_modal()

# -------- CHAT HISTORY --------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm LearnSphere, your collaborative AI learning hub. "
            "I can help you find courses, research papers, and answer academic questions.\n\n"
            "**Available tools:** "
            + ", ".join(service.get_tools()),
        }
    ]

prompt = st.chat_input("Ask me anything about learning...")

if "prompt_override" in st.session_state:
    prompt = st.session_state.prompt_override
    del st.session_state.prompt_override

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -------- HANDLE CHAT PROMPT --------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        start_time = time.time()
        with progress_placeholder:
            st.info(f"🤔 Processing your request... (timeout: {timeout_setting}s)")
        response, error = service.chat(prompt, timeout=timeout_setting)
        progress_placeholder.empty()

        if error:
            elapsed = time.time() - start_time
            error_msg = f"❌ **Request failed after {elapsed:.1f}s**: {error}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
        else:
            elapsed = time.time() - start_time
            if response and response.strip():
                st.write(response)
                st.caption(f"*Response generated in {elapsed:.1f} seconds*")
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            else:
                empty_msg = (
                    "⚠️ I generated a response but it appears to be empty. "
                    "This might be a tool execution issue."
                )
                st.warning(empty_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": empty_msg}
                )

    # ----- PROFILE UPDATE + COLLABORATION -----
    print("Background storage started ...")
    print(prompt + "\n" + (response or ""))

    profile = extract_profile_from_text((prompt or "") + "\n" + (response or ""))
    timestamp = datetime.datetime.now().isoformat()

    update_user_profile_with_timestamp(
        username=st.session_state.username,
        profile=profile,
        timestamp=timestamp,
    )
    print("Updated User Profile")
    st.rerun()

# -------- FOOTER METRICS --------
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("🛠️ Tools Available", len(service.get_tools()))
with c2:
    st.metric("💬 Messages", len(st.session_state.get("messages", [])))
with c3:
    status = "🟢 Ready" if service.is_ready() else "🔴 Not Ready"
    st.metric("📡 Service Status", status)