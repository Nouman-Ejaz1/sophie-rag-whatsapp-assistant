# Sophie AI: Self-Hosted WhatsApp Assistant (RAG & Tool Calling)

A self-hosted WhatsApp AI assistant named Sophie. She has a warm personality, remembers what you tell her, searches through documents you give her, can run system commands and tool calls, and can text you on a schedule.

I built this because I wanted a custom AI assistant that actually lives on my phone inside my personal chat, rather than a boring, clinical web interface.

---

## 🎙️ What makes Sophie special?

*   **She has a personality:** Sophie is warm, supportive, and polite. When you send her a message, she triggers a typing indicator (`typing...`) on WhatsApp before replying, so it feels like a real conversation.
*   **She can run tools and system commands:** Sophie isn't just a text box. She can trigger backend operations, fetch files, look up the weather, and execute commands directly on the machine she is hosted on.
*   **She asks before deleting:** Destructive delete actions are queued for WhatsApp approval and only run after you reply with the matching approval ID.
*   **She searches your documents (Pinecone RAG):** You can feed books, coding manuals, or text files into her knowledge base. When you ask a question, she pulls context from Pinecone and gives you a precise response.
*   **She can ingest files and voice notes:** Sophie can accept WhatsApp documents, PDF URLs, local files, pasted text, and free-first local Whisper voice transcription.
*   **She has a human-like memory (SQLite Decay):** Instead of remembering everything forever or forgetting everything instantly, Sophie uses an episodic memory decay system. Old memories naturally fade over time, mimicking how a human brain works.
*   **She can text you first:** An autonomous background scheduler runs inside her core. If you tell her to monitor a website, release news, or give you a morning weather report, she will execute the task on time and message your phone.
*   **Easy to test (Self-Chat):** You can scan the QR code and message your own number (self-chat) to test her out instantly.

---

## 🛠️ The Tech Stack

*   **Backend:** FastAPI (Python) running the orchestrator, memory OS, and background scheduler.
*   **WhatsApp Gateway:** Node.js (with `whatsapp-web.js` & Puppeteer) acting as a self-hosted messaging client.
*   **Database:** SQLite (local memory and logs) + Pinecone (vector database for document search).
*   **AI Model:** OpenRouter via LangGraph + LangChain tool calling. Gemini can remain configured only as an optional legacy fallback.

---

## 🚀 Setting Up Sophie

### 1. Configure the Environment
Create a `.env` file inside the `backend` folder:
```env
# Optional legacy Gemini key. Sophie uses OpenRouter when LLM_PROVIDER=openrouter.
GEMINI_API_KEY=
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
DATABASE_URL=sqlite:///./sentinel_ai.db
PORT=8000
HOST=127.0.0.1

# Model provider. OpenRouter is the default path.
LLM_PROVIDER=openrouter
SOPHIE_TOOL_MODE=gateway
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_FREE_MODEL_POOL=nvidia/nemotron-3-super-120b-a12b:free,qwen/qwen3-coder:free,deepseek/deepseek-v4-flash:free,openai/gpt-oss-120b:free,openrouter/free
OPENROUTER_MAX_MODEL_ATTEMPTS=4

# Leave OPENROUTER_MODEL empty or set it to none for free-auto mode.
# Paste a specific model ID here for fixed mode, for example:
# OPENROUTER_MODEL=anthropic/claude-sonnet-4.6

# Optional: Add comma-separated phone numbers to whitelist. Keep empty to reply to everyone.
ALLOWED_NUMBERS=

# Optional voice/media settings. local_whisper is free-first and does not call Gemini for voice.
VOICE_TRANSCRIPTION_PROVIDER=local_whisper
WHISPER_MODEL_SIZE=small
MAX_VOICE_SECONDS=180
MAX_MEDIA_MB=25
VOICE_TRANSCRIPTION_FALLBACK=none

# Optional trusted research/search settings. Defaults are free-first.
SEARCH_PROVIDER_ORDER=searxng,google_cse,google_news,bing_news,jina_search,duckduckgo
SEARXNG_URL=
GOOGLE_CSE_API_KEY=
GOOGLE_CSE_ID=
JINA_API_KEY=
SEARCH_MAX_SOURCES=20
SEARCH_REQUIRE_TRUSTED_FOR_LATEST=true
```

### 2. Start the Backend (FastAPI)
Navigate to the `backend` folder, install dependencies, and start the server:
```bash
pip install -r requirements.txt
python run.py
```
This runs the Python server on `http://127.0.0.1:8000`.

For WhatsApp voice notes, install `ffmpeg` and make sure it is available on your PATH. The first local Whisper transcription may download the configured model.

### 3. Start the WhatsApp Gateway
In a new terminal window inside the `backend` folder, install the Node dependencies and start the gateway:
```bash
npm install
npm start
```
*   The gateway will initialize and output a **QR Code** in your terminal.
*   Open WhatsApp on your phone, go to **Linked Devices**, and scan the QR code.
*   Once it prints `Sophie WhatsApp Client is READY and CONNECTED!`, you're good to go!

---


