"""
Azure OpenAI helper — single source of truth for all backend modules.

Reads from env vars (loaded from Config/.env):
  AZURE_OPENAI_API_KEY      — required
  AZURE_OPENAI_ENDPOINT     — required (e.g. https://xxx.openai.azure.com/)
  AZURE_OPENAI_DEPLOYMENT   — deployment name (default: synapt-dev-gpt-4o-mini)
  AZURE_OPENAI_API_VERSION  — API version  (default: 2025-01-01-preview)

When both key and endpoint are set, Azure OpenAI is the PRIMARY LLM across
the entire application; NIM and Gemini serve as cascading fallbacks.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — read once at import time (env already loaded by each module's own
# load_dotenv call before importing this helper).
# ---------------------------------------------------------------------------
AZURE_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "synapt-dev-gpt-4o-mini")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")


def is_azure_configured() -> bool:
    """Return True when both API key and endpoint are present in the environment."""
    return bool(AZURE_API_KEY and AZURE_ENDPOINT)


def get_azure_chat_llm(temperature: float = 0.0):
    """
    Return a LangChain AzureChatOpenAI instance for this deployment.
    Returns None if Azure credentials are not configured.

    This LLM supports tool calling and can be used directly with
    LangGraph create_react_agent.
    """
    if not is_azure_configured():
        return None
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        openai_api_version=AZURE_API_VERSION,
        api_key=AZURE_API_KEY,
        temperature=temperature,
        max_retries=2,
    )


def invoke_azure(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Synchronous single-turn call to Azure OpenAI via the raw openai SDK.
    Raises RuntimeError if Azure is not configured.
    Propagates openai exceptions on API failure (caller should catch and fallback).
    """
    if not is_azure_configured():
        raise RuntimeError(
            "Azure OpenAI not configured — set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT"
        )
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT + "/",
        api_version=AZURE_API_VERSION,
    )
    r = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()
