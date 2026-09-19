"""Configuration module for Mini AI Knowledge Assistant.

Handles loading of environment variables and exposes application settings
in a centralized, readable manner.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load variables from .env file if present (optional in cloud environments)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=BASE_DIR / ".env")
except Exception:
    pass

def get_gemini_api_key() -> str:
    """Retrieves the Gemini API key from environment or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for candidate in ["GEMINI_API_KEY", "gemini_api_key", "GOOGLE_API_KEY", "google_api_key"]:
                    if candidate in st.secrets:
                        key = str(st.secrets[candidate]).strip()
                        break
        except Exception:
            pass
    return key.strip("'\"")


GEMINI_API_KEY = get_gemini_api_key()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()

try:
    import streamlit as st
    if hasattr(st, "secrets"):
        if "GEMINI_MODEL" in st.secrets:
            GEMINI_MODEL = str(st.secrets["GEMINI_MODEL"]).strip()
        if "GEMINI_EMBEDDING_MODEL" in st.secrets:
            GEMINI_EMBEDDING_MODEL = str(st.secrets["GEMINI_EMBEDDING_MODEL"]).strip()
except Exception:
    pass

# Chunking Parameters (Configurable for Phase 2)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))

# Retrieval & Storage Settings (Configurable for Phase 3 & 4)
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(DATA_DIR / "uploads")))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.2"))



def get_config_summary() -> dict:
    """Returns a dictionary summarizing the current environment configuration."""
    api_key_configured = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "your_gemini_api_key_here"

    return {
        "base_directory": str(BASE_DIR),
        "data_directory": str(DATA_DIR),
        "vector_store_directory": str(VECTOR_STORE_DIR),
        "gemini_api_key_set": api_key_configured,
        "gemini_model": GEMINI_MODEL,
        "gemini_embedding_model": GEMINI_EMBEDDING_MODEL,
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "default_top_k": DEFAULT_TOP_K,
        "retrieval_score_threshold": RETRIEVAL_SCORE_THRESHOLD,
    }


