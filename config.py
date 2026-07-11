"""
config.py
---------
Central configuration for the RAG PDF bot.

Responsibilities:
- Load secrets (API keys) and settings from a local ``.env`` file.
- Choose between Google Gemini (default) and OpenAI via a single env variable.
- NEVER hardcode API keys -- everything sensitive comes from the environment.

Switching providers is as easy as editing one line in ``.env``:
    LLM_PROVIDER=gemini   # or: openai
"""

import os

from dotenv import load_dotenv

# Load key=value pairs from a ".env" file (if present) into the environment.
load_dotenv()

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
# LLM_PROVIDER controls which backend is used for BOTH the chat model and the
# embedding model. Supported values: "gemini" (default) or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# ---------------------------------------------------------------------------
# API keys (read from environment / .env -- never hardcoded)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
# We deliberately pick each provider's CURRENT recommended models (not old,
# deprecated ones). All are overridable from .env if you want to experiment.

# --- Google Gemini ---
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
# "gemini-embedding-001" is Google's current, generally-available embedding
# model. It supersedes the older/deprecated "embedding-001" and
# "text-embedding-004". The "models/" prefix is what the API expects.
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

# --- OpenAI ---
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# "text-embedding-3-small" is OpenAI's current, cost-effective embedding model.
# It supersedes the deprecated "text-embedding-ada-002".
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Vector store + retrieval settings
# ---------------------------------------------------------------------------
# Folder where the persisted Chroma database lives (added to .gitignore).
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# Chunking knobs (see rag_pipeline.py).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many chunks to retrieve per question.
TOP_K = 4


class ConfigError(Exception):
    """Raised when required configuration (e.g. an API key) is missing."""


def validate_config() -> None:
    """
    Ensure the selected provider has everything it needs to run.

    Raises:
        ConfigError: with a friendly, user-facing message the UI can display.
    """
    if LLM_PROVIDER not in ("gemini", "openai"):
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Set LLM_PROVIDER=gemini or LLM_PROVIDER=openai in your .env file."
        )

    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        raise ConfigError(
            "GOOGLE_API_KEY is missing. Add it to your .env file. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ConfigError(
            "OPENAI_API_KEY is missing. Add it to your .env file. "
            "Get a key at https://platform.openai.com/api-keys"
        )
