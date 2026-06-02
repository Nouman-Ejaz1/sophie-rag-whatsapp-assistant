import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (utf-8-sig strips BOM if present — Windows editors often add it)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", encoding="utf-8-sig")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
SOPHIE_TOOL_MODE = os.getenv("SOPHIE_TOOL_MODE", "gateway").strip().lower()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_TOOL_MODEL = os.getenv("OPENROUTER_TOOL_MODEL", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_FREE_MODEL_POOL = [
    model.strip()
    for model in os.getenv(
        "OPENROUTER_FREE_MODEL_POOL",
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "qwen/qwen3-coder:free,"
        "deepseek/deepseek-v4-flash:free,"
        "openai/gpt-oss-120b:free,"
        "openrouter/free",
    ).split(",")
    if model.strip()
]
OPENROUTER_MAX_MODEL_ATTEMPTS = int(os.getenv("OPENROUTER_MAX_MODEL_ATTEMPTS", "4"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sentinel-ai")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_data/sentinel_ai.db")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")
VOICE_TRANSCRIPTION_PROVIDER = os.getenv("VOICE_TRANSCRIPTION_PROVIDER", "local_whisper").strip().lower()
VOICE_TRANSCRIPTION_FALLBACK = os.getenv("VOICE_TRANSCRIPTION_FALLBACK", "none").strip().lower()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small").strip()
MAX_VOICE_SECONDS = int(os.getenv("MAX_VOICE_SECONDS", "180"))
MAX_MEDIA_MB = int(os.getenv("MAX_MEDIA_MB", "25"))
SEARCH_PROVIDER_ORDER = os.getenv(
    "SEARCH_PROVIDER_ORDER",
    "searxng,google_cse,google_news,bing_news,jina_search,duckduckgo"
)
SEARXNG_URL = os.getenv("SEARXNG_URL", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
SEARCH_MAX_SOURCES = int(os.getenv("SEARCH_MAX_SOURCES", "20"))
SEARCH_REQUIRE_TRUSTED_FOR_LATEST = os.getenv(
    "SEARCH_REQUIRE_TRUSTED_FOR_LATEST",
    "true"
).strip().lower() in {"1", "true", "yes", "on"}

# Parse Whitelist Allowed Numbers
ALLOWED_NUMBERS_RAW = os.getenv("ALLOWED_NUMBERS", "")
ALLOWED_NUMBERS = [num.strip() for num in ALLOWED_NUMBERS_RAW.split(",") if num.strip()]

# Parse and configure dynamic GEMINI_MODEL
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")
if GEMINI_MODEL and not GEMINI_MODEL.startswith("models/"):
    GEMINI_MODEL = f"models/{GEMINI_MODEL}"

# Ensure required environment variables are set
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")
if LLM_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment or .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment or .env file.")

openrouter_mode = "free_auto" if not OPENROUTER_MODEL or OPENROUTER_MODEL.lower() == "none" else "fixed"
active_model = (
    f"{openrouter_mode}:{OPENROUTER_MODEL or 'free_pool'}"
    if LLM_PROVIDER == "openrouter"
    else GEMINI_MODEL
)
print(f"Loaded config: Host={HOST}, Port={PORT}, IndexName={PINECONE_INDEX_NAME}, AllowedNumbers={ALLOWED_NUMBERS}, Provider={LLM_PROVIDER}, Model={active_model}")
