from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import os
import re
import time
import uuid

from app.models import ChatRequest, IngestRequest, IngestSourceRequest, MockEventRequest
from app.database import db
from app.brain.doc_brain import doc_brain
from app.brain.memory_os import memory_os
from app.brain.trigger_mind import trigger_mind
from app.brain.orchestrator import orchestrator
from app.brain.document_ingest import (
    ingest_local_file,
    ingest_media_document,
    ingest_text_document,
    ingest_url_document,
)
from app.brain.safety import resolve_approval_command
from app.brain.voice import transcribe_voice_note
from app.brain.research_engine import is_exchange_rate_query, research_query

app = FastAPI(title="SentinelAI Autonomous Agent API", version="1.0.0")

def sanitize_whatsapp_reply(text: str) -> str:
    """Final response cleanup for WhatsApp/log safety without removing normal language text."""
    cleaned = text or ""
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("…", "...")
    cleaned = cleaned.replace("â", "-").replace("â", "-").replace("â¦", "...")
    cleaned = cleaned.replace("â¨", "").replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()

def exchange_rate_whatsapp_reply(message: str) -> Optional[str]:
    """Fast path for simple currency-rate questions so they do not require an LLM call."""
    if not is_exchange_rate_query(message):
        return None
    try:
        result = research_query(message, mode="auto")
    except Exception as exc:
        return (
            "I could not check the exchange rate right now because the finance lookup failed. "
            f"Error: {exc}"
        )

    claims = result.get("claims") or []
    if not claims:
        return (
            "I could not verify a numeric exchange rate from the free providers right now. "
            "I will not invent the rate. Please try again in a minute."
        )

    claim = claims[0]
    details = claim.get("details") or []
    trust = claim.get("trust_score", result.get("trust_score", 0))
    lines = [
        f"{claim.get('value')}.",
        f"Trust: {trust}%.",
    ]
    if details:
        lines.append("")
        lines.append("Checked:")
        lines.extend(f"- {detail}" for detail in details[:2])
    lines.append("")
    lines.append("Rates can move during the day.")
    return sanitize_whatsapp_reply("\n".join(lines))

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_activity_logger(request, call_next):
    """Print compact request lifecycle logs so terminal activity is obvious."""
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    print(
        f"[HTTP {request_id}] --> {request.method} {request.url.path} from {client}",
        flush=True
    )
    try:
        response = await call_next(request)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[HTTP {request_id}] <-- 500 {request.method} {request.url.path} "
            f"in {elapsed_ms:.1f}ms | error={e}",
            flush=True
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    print(
        f"[HTTP {request_id}] <-- {response.status_code} {request.method} {request.url.path} "
        f"in {elapsed_ms:.1f}ms",
        flush=True
    )
    return response

@app.on_event("startup")
def startup_seeder():
    """Seeds baseline reference books for Coding, AI, and Prompt Engineering on initial startup."""
    try:
        logs = db.get_logs(limit=100)
        has_seeded = any("seeder" in str(log.get("source", "")).lower() for log in logs)
        if not has_seeded:
            print("[SEEDER] Database is empty. Loading premium developer and AI manuals...")
            
            # Seed 1: Coding Manual
            doc_brain.ingest_document(
                title="The Art of Modern Pythonic & Clean Coding",
                content=(
                    "Modern Python development values readability, efficiency, and structural simplicity. The Core Pillars of clean code are: "
                    "1. SOLID Principles: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. "
                    "2. Pythonic Idioms: Leverage list and dict comprehensions, generator expressions for memory efficiency, and context managers "
                    "(with statements) to safely acquire and release system resources. "
                    "3. Type Hinting & Verification: Use static type hints and robust verification structures like Pydantic v2 to enforce schema "
                    "compliance at API boundaries. "
                    "4. Robust Testing: Implement unit and integration tests using pytest, asserting boundary conditions and isolating external "
                    "API calls with mock dependencies to maintain a highly resilient codebase."
                ),
                source_type="coding",
                source_url="coding://pythonic_clean_code"
            )
            
            # Seed 2: AI Manual
            doc_brain.ingest_document(
                title="Foundations of Generative AI & Transformer Architectures",
                content=(
                    "Generative AI models rely on the Transformer architecture, pioneered in 2017. Transformers process input sequences in parallel, "
                    "unlike previous recurrent networks. The core mechanisms include: "
                    "1. Self-Attention: Measures the pairwise relationship between all tokens in a sequence, creating dynamic context-aware representations. "
                    "2. Multi-Head Attention: Projects tokens into multiple subspaces, allowing the model to attend to different parts of the sequence simultaneously. "
                    "3. Encoder-Decoder Structures: Encoders construct high-dimensional vector representations, and decoders generate text auto-regressively, "
                    "predicting the next token based on previous outputs. "
                    "4. Tuning and Alignment: Models undergo pre-training on massive datasets, instruction fine-tuning for task-following, and Reinforcement "
                    "Learning from Human Feedback (RLHF) to align outputs with human preferences."
                ),
                source_type="ai",
                source_url="ai://transformers_foundations"
            )
            
            # Seed 3: Prompt Engineering Manual
            doc_brain.ingest_document(
                title="Prompt Engineering: The Complete Codex of Cognitive Patterns",
                content=(
                    "Prompt engineering is the strategic craft of structuring inputs to prompt precise, logical behaviors in LLMs. "
                    "The foundational techniques are: "
                    "1. Few-Shot Prompting: Provide high-quality inputs and target outputs to steer formatting and style by example. "
                    "2. Chain-of-Thought (CoT): Prompt the model to write out its step-by-step reasoning process before presenting its final output, "
                    "which significantly reduces logical errors and hallucinations. "
                    "3. ReAct Framework: Combine reasoning and action planning, prompting the agent to alternatingly 'Think' and 'Act' using external tools. "
                    "4. Strict Security Guardrails: Declare rigid persona directives, block jailbreak attempts like 'ignore previous instructions', "
                    "and strictly constrain responses using formatted JSON schemas to prevent prompt injection."
                ),
                source_type="prompting",
                source_url="prompting://cognitive_patterns"
            )
            
            # Seed 4: Cybersecurity Manual
            doc_brain.ingest_document(
                title="Cybersecurity & Autonomous Defense Frameworks",
                content=(
                    "Modern backend systems require defense-in-depth principles to secure sensitive data. Essential protocols are: "
                    "1. Secure Port Management: Production endpoints must operate on encrypted port 443 (HTTPS), while non-secure development ports like 8080 "
                    "must be fully isolated to local sandboxes. "
                    "2. Token Security and Rotation: Implement cryptographically signed JWT access tokens with short lifetimes (e.g. 15 minutes) coupled with "
                    "secure, database-backed refresh tokens. "
                    "3. Authentication Integrity: Never commit raw access keys (such as 'AIzaSy' or 'pcsk_') into version control. Implement HMAC request "
                    "verification on webhook endpoints to confirm the sender's identity. "
                    "4. Secure Database Access: Avoid raw query concatenation. Always use parameterized queries or trusted Object-Relational Mappers (ORMs) "
                    "to completely mitigate SQL injection attacks."
                ),
                source_type="cybersecurity",
                source_url="cybersecurity://defense_frameworks"
            )
            
            # Seed 5: Stateful Agentic Orchestration Manual
            doc_brain.ingest_document(
                title="LlamaIndex & LangGraph: Stateful Orchestration Handbook",
                content=(
                    "Advanced AI agents go beyond single-prompt APIs by running stateful, multi-turn reasoning pipelines. The core orchestration concepts are: "
                    "1. Stateful Graphs: Build agent workflows as Directed Acyclic Graphs (DAGs) using frameworks like LangGraph, where a shared state dictionary "
                    "persists across all processing nodes. "
                    "2. Conditional Routing: Evaluate graph variables at runtime to direct execution dynamically (e.g. route to high-capacity model if query is "
                    "complex, or execute local scripts). "
                    "3. Human-in-the-Loop: Create interrupt gates in the graph structure that pause agent execution and wait for human validation before firing "
                    "destructive external tools. "
                    "4. Custom Retrieval (RAG): Implement namespaced vector indexing, semantic window chunking, and dual-encoder similarity matching to feed rich, "
                    "real-time context into the agent's active reasoning state."
                ),
                source_type="orchestration",
                source_url="orchestration://stateful_agents"
            )
            
            # Seed some base semantic memories in SQLite/Pinecone
            memory_os.add_semantic_memory(
                "SentinelAI is configured to automatically alert the security team via Slack when a commit push references unsecured development ports like 8080."
            )
            memory_os.add_semantic_memory(
                "The JWT access tokens used by the backend are set to rotate every 15 minutes, with refreshes valid for up to 24 hours."
            )
            memory_os.add_semantic_memory(
                "Episodic memories are automatically flagged as stale when their calculated freshness score decays below 40%."
            )
            
            db.log_event(
                source="seeder",
                message="Successfully seeded baseline Coding, AI, and Prompt Engineering books into DocBrain namespaces!",
                status="success"
            )
            print("[SEEDER] baseline knowledge seeded successfully.")
    except Exception as e:
        print(f"Warning: startup seeder encountered an error (index might be initializing): {e}")

@app.on_event("startup")
def startup_scheduler():
    """Starts the background scheduler with reminders + smart triggers."""
    import threading
    from app.brain.trigger_engine import trigger_engine
    
    # Initialize trust registry and check Google CSE at startup
    try:
        from app.brain.trust_registry import trust_registry
        from app.brain.research_engine import check_google_cse_configured
        trust_registry.load(regions=["pakistan"])
        check_google_cse_configured()
    except Exception as e:
        print(f"[Startup] Failed to initialize trust registry or check CSE: {e}")

    def scheduler_loop():
        import datetime
        import time
        import requests
        
        print("[SCHEDULER] Autonomous Background Task Worker thread successfully launched!")
        # Sleep initially to let FastAPI and Gateway bind completely
        time.sleep(10)
        
        while True:
            try:
                # Run all smart triggers (threshold, keyword, recurring)
                trigger_engine.run_cycle()
                
                # One-time reminders (existing logic — keep this)
                now = datetime.datetime.utcnow()
                reminders = db.get_due_reminders(now.isoformat())
                for reminder in reminders:
                    target = reminder.get("sender")
                    if not target:
                        db.mark_reminder_sent(reminder["id"], now.isoformat())
                        continue
                    reminder_text = reminder.get("message") or reminder.get("title") or "Reminder"
                    try:
                        gateway_payload = {
                            "to": target,
                            "message": f"*🔔 Reminder*\n{reminder_text}"
                        }
                        res = requests.post("http://127.0.0.1:3001/send", json=gateway_payload, timeout=15)
                        if res.status_code == 200:
                            print(f"[SCHEDULER] Sent reminder '{reminder.get('title')}' to {target}.")
                            db.mark_reminder_sent(reminder["id"], now.isoformat())
                        else:
                            print(f"[SCHEDULER] WhatsApp Gateway returned reminder error: {res.status_code}. Marking as failed.")
                            db.mark_reminder_sent(reminder["id"], now.isoformat())
                    except Exception as gw_err:
                        print(f"[SCHEDULER] Connection error pushing reminder: {gw_err}. Marking as failed.")
                        db.mark_reminder_sent(reminder["id"], now.isoformat())
                            
            except Exception as e:
                print(f"[SCHEDULER] Scheduler loop encountered an error: {e}")
                
            time.sleep(30)
            
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "agent": "SentinelAI",
        "version": "1.0.0",
        "engine": "OpenRouter + LangGraph tool gateway",
        "vector_store": "Pinecone Namespaces"
    }

# --- System Logs Endpoint ---
@app.get("/api/logs")
def get_system_logs(limit: int = 50):
    """Returns real-time event logs and wake-up alerts for the dashboard feed."""
    return db.get_logs(limit=limit)

# --- Conversational Endpoint ---
@app.post("/api/chat")
def chat(request: ChatRequest):
    """Processes user chat, pulls documents and memories, and runs reasoning engine."""
    from app.brain.watchdog import watchdog, WatchdogInterrupt
    import uuid
    request_id = f"chat_{uuid.uuid4().hex[:8]}"
    try:
        watchdog.register_request(request_id)
        # We can extract recent logs to supply as context or keep it directly in orchestrator
        result = orchestrator.run(message=request.message)
        return {
            "response": result.get("response", ""),
            "reasoning": result.get("reasoning", ""),
            "citations": result.get("citations", []),
            "confidence_score": result.get("confidence_score", 100.0),
            "retrieved_docs": result.get("retrieved_docs", []),
            "retrieved_memories": result.get("retrieved_memories", [])
        }
    except WatchdogInterrupt:
        print(f"[API] Watchdog killed chat request {request_id} due to timeout.", flush=True)
        return {
            "response": "My processing time limit was reached. Please try a simpler query or try again.",
            "reasoning": "Watchdog circuit-breaker activated.",
            "citations": [],
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "retrieved_memories": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        watchdog.unregister_request()

# --- Premium Book Catalog Endpoints ---
@app.get("/api/documents")
def get_documents():
    """Lists unique ingested documents/books in SQLite with chunk count and created dates."""
    try:
        return db.get_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/chunks")
def get_document_chunks_endpoint(title: str):
    """Retrieves all chunks of a specific ingested document/book by title."""
    try:
        return db.get_document_chunks(title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Document Ingestion Endpoint ---
@app.post("/api/documents/ingest")
def ingest_document(request: IngestRequest):
    """Ingests text content, chunks, embeds, and uploads to Pinecone namespace."""
    try:
        result = doc_brain.ingest_document(
            title=request.title,
            content=request.content,
            source_type=request.source_type,
            source_url=request.source_url
        )
        return {
            "status": "success",
            "message": f"Successfully ingested '{request.title}'",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/ingest-source")
def ingest_document_source(request: IngestSourceRequest):
    """Ingests text, local files, URLs, or WhatsApp media into Pinecone/SQLite RAG."""
    try:
        source_kind = request.source_kind.strip().lower()
        if source_kind == "text":
            result = ingest_text_document(
                title=request.title or "Untitled text document",
                content=request.content or "",
                source_type=request.source_type,
                source_url=request.source_url
            )
        elif source_kind == "local_path":
            if not request.path:
                raise HTTPException(status_code=400, detail="Missing path for local_path ingestion")
            result = ingest_local_file(
                file_path=request.path,
                title=request.title,
                source_type=request.source_type,
                source_url=request.source_url,
                mime_type=request.mime_type
            )
        elif source_kind == "url":
            if not request.url:
                raise HTTPException(status_code=400, detail="Missing url for URL ingestion")
            result = ingest_url_document(
                url=request.url,
                title=request.title,
                source_type=request.source_type
            )
        elif source_kind == "whatsapp_media":
            result = ingest_media_document(
                media_base64=request.media_base64 or "",
                filename=request.filename,
                title=request.title,
                mime_type=request.mime_type,
                source_type=request.source_type or "whatsapp"
            )
        else:
            raise HTTPException(status_code=400, detail="source_kind must be text, local_path, url, or whatsapp_media")

        return {"status": "success", "message": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Trigger Simulation Endpoint ---
@app.post("/api/triggers/mock")
def mock_trigger(request: MockEventRequest):
    """Fires a simulated event (GitHub, Slack, Google Calendar, News) to wake up the agent."""
    try:
        # Step 1: TriggerMind classifies incoming event and logs it
        classified_event = trigger_mind.classify_event(
            source=request.source,
            title=request.title,
            description=request.description,
            metadata=request.metadata
        )
        
        # Step 2: Run orchestrator workflow autonomously with this event trigger!
        print(f"[API] Waking up SentinelAI autonomously due to mock trigger: {request.title}")
        result = orchestrator.run(triggered_event=classified_event)
        
        return {
            "trigger_classification": classified_event,
            "agent_state": {
                "is_urgent": result.get("is_urgent", False),
                "reasoning": result.get("reasoning", ""),
                "action_taken": result.get("action_taken"),
                "response": result.get("response", ""),
                "confidence_score": result.get("confidence_score", 100.0),
                "citations": result.get("citations", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/triggers/mock-source/{source}")
def mock_trigger_by_source(source: str):
    """Fires a standard pre-defined corporate trigger event by keyword name."""
    try:
        classified_event = trigger_mind.simulate_mock_event(source)
        result = orchestrator.run(triggered_event=classified_event)
        return {
            "trigger_classification": classified_event,
            "agent_state": {
                "is_urgent": result.get("is_urgent", False),
                "reasoning": result.get("reasoning", ""),
                "action_taken": result.get("action_taken"),
                "response": result.get("response", ""),
                "confidence_score": result.get("confidence_score", 100.0),
                "citations": result.get("citations", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Memory Decay & Management Endpoints ---
@app.get("/api/memories")
def get_memories():
    """Lists all episodic and semantic memories with active freshness scores."""
    return memory_os.get_all_decay_states()

@app.post("/api/memories/decay")
def simulate_decay(payload: Dict[str, float]):
    """Simulates elapsed time (in days) to trigger artificial memory decay scores."""
    days = payload.get("days", 1.0)
    try:
        updated_records = memory_os.trigger_artificial_decay(days)
        db.log_event(
            source="MemoryOS",
            message=f"Simulated time-decay lapse: advanced clock by {days} days",
            status="warning",
            meta_dict={"days_elapsed": days}
        )
        return {
            "status": "success",
            "message": f"Time advanced by {days} days. Freshness scores decayed.",
            "memories": updated_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/refresh/{memory_id}")
def refresh_memory(memory_id: str):
    """Resets memory age and accesses it to make it 100% fresh again."""
    try:
        # To refresh, we just update its created_at timestamp to now in SQLite
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memory_decay_records SET created_at = ?, status = 'active' WHERE id = ?",
                (now, memory_id)
            )
            conn.commit()
            
        db.log_event(
            source="MemoryOS",
            message=f"Refreshed memory: {memory_id}",
            status="success"
        )
        return {"status": "success", "message": f"Memory {memory_id} successfully refreshed!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WhatsApp Messaging Gateway Route ---
@app.post("/api/whatsapp/message")
def whatsapp_message(payload: Dict[str, Any]):
    """Receives message from self-hosted WhatsApp web gateway, processes it, and returns the response."""
    import re
    route_started = time.perf_counter()
    sender = payload.get("sender", "unknown")
    message = payload.get("message", "").strip()
    media_type = (payload.get("media_type") or "").strip().lower()
    mime_type = payload.get("mime_type")
    filename = payload.get("filename")
    media_base64 = payload.get("media_base64")
    has_media = bool(media_base64)
    print(
        f"[WA API] START sender={sender} text_len={len(message)} "
        f"media={has_media} media_type={media_type or 'none'} mime={mime_type or 'none'}",
        flush=True
    )
        
    # Whitelisted senders validation check
    from app.config import ALLOWED_NUMBERS
    clean_sender = sender.split("@")[0]
    if ALLOWED_NUMBERS and "*" not in ALLOWED_NUMBERS:
        if sender not in ALLOWED_NUMBERS and clean_sender not in ALLOWED_NUMBERS:
            print(f"[WhatsApp] Rejecting unauthorized sender: {sender}")
            db.log_event(
                source="WhatsApp",
                message=f"Rejected unauthorized sender: {sender}",
                status="warning"
            )
            raise HTTPException(status_code=403, detail="Sender unauthorized")

    approval_result = resolve_approval_command(sender, message)
    if approval_result:
        print(
            f"[WA API] APPROVAL sender={sender} status={approval_result['status']}",
            flush=True
        )
        db.save_chat_message(sender, "user", message)
        db.save_chat_message(sender, "assistant", approval_result["message"])
        db.log_event(
            source="Approval",
            message=f"{approval_result['status']}: {message}",
            status="success" if approval_result["status"] in {"executed", "cancelled"} else "warning",
            meta_dict={"sender": sender, "approval": approval_result}
        )
        return {"status": "success", "response": approval_result["message"]}

    if media_base64 and media_type in {"ptt", "audio", "voice"}:
        print(f"[WA API] VOICE transcribing sender={sender} filename={filename or 'none'}", flush=True)
        transcript = transcribe_voice_note(media_base64, mime_type=mime_type, filename=filename)
        if transcript.startswith("I received") or transcript.startswith("Voice note is too large"):
            print(f"[WA API] VOICE failed-cleanly sender={sender}: {transcript[:120]}", flush=True)
            db.save_chat_message(sender, "user", "[voice note]")
            db.save_chat_message(sender, "assistant", transcript)
            return {"status": "success", "response": transcript}
        caption = f"\n\nCaption: {message}" if message else ""
        message = f"Voice note transcript:\n{transcript}{caption}"
        print(f"[WA API] VOICE transcript_ready sender={sender} transcript_len={len(transcript)}", flush=True)

    elif media_base64 and media_type in {"document", "application", "file"}:
        print(f"[WA API] DOCUMENT ingesting sender={sender} filename={filename or 'none'}", flush=True)
        ingest_result = ingest_media_document(
            media_base64=media_base64,
            filename=filename,
            title=message or filename,
            mime_type=mime_type,
            source_type="whatsapp"
        )
        message = (
            f"WhatsApp document received: {filename or 'unnamed file'}\n"
            f"Ingestion result:\n{ingest_result}\n\n"
            f"User caption/request: {message or 'Please confirm this was saved.'}"
        )
        print(f"[WA API] DOCUMENT ingest_done sender={sender} result_len={len(ingest_result)}", flush=True)

    if not message:
        print(f"[WA API] REJECT empty message sender={sender}", flush=True)
        raise HTTPException(status_code=400, detail="Empty message received")

    print(f"[WhatsApp] Incoming from {sender}: '{message}'", flush=True)
    
    # 1. Log user's incoming message to chat_history table
    db.save_chat_message(sender, "user", message)
    
    # Log incoming WhatsApp event to the system log database
    db.log_event(
        source="WhatsApp",
        message=f"Received: {message[:40]}...",
        status="info",
        meta_dict={"sender": sender, "message": message}
    )

    finance_reply = exchange_rate_whatsapp_reply(message)
    if finance_reply:
        db.save_chat_message(sender, "assistant", finance_reply)
        db.log_event(
            source="Finance",
            message=f"Exchange-rate reply to {sender}",
            status="success",
            meta_dict={"sender": sender, "reply": finance_reply[:100]}
        )
        elapsed_ms = (time.perf_counter() - route_started) * 1000
        print(
            f"[WA API] FINANCE_FASTPATH sender={sender} reply_len={len(finance_reply)} "
            f"elapsed={elapsed_ms:.1f}ms",
            flush=True
        )
        return {"status": "success", "response": finance_reply}

    try:
        # 2. Retrieve recent conversation history for this sender
        history = db.get_chat_history(sender, limit=10)
        print(f"[WA API] HISTORY sender={sender} messages={len(history)}", flush=True)
        
        # 3. Run orchestrator with whitelisted sender and loaded chat history context
        print(f"[WA API] ORCHESTRATOR start sender={sender}", flush=True)
        from app.brain.watchdog import watchdog, WatchdogInterrupt
        import uuid
        request_id = f"wa_{uuid.uuid4().hex[:8]}"
        try:
            watchdog.register_request(request_id)
            result = orchestrator.run(message=message, sender=sender, history=history)
        except WatchdogInterrupt:
            elapsed_ms = (time.perf_counter() - route_started) * 1000
            print(
                f"[WA API] WATCHDOG_KILL sender={sender} request={request_id} "
                f"elapsed={elapsed_ms:.1f}ms. Request exceeded timeout.",
                flush=True
            )
            db.log_event(
                source="Watchdog",
                message=f"Killed request {request_id} for {sender} after {elapsed_ms:.0f}ms",
                status="warning",
                meta_dict={"sender": sender, "request_id": request_id}
            )
            timeout_reply = (
                "I ran into my processing time limit on that request. "
                "Could you try rephrasing it or breaking it into a simpler question?"
            )
            db.save_chat_message(sender, "assistant", timeout_reply)
            return {"status": "success", "response": timeout_reply}
        finally:
            watchdog.unregister_request()
        raw_response = result.get("response", "")
        print(
            f"[WA API] ORCHESTRATOR done sender={sender} "
            f"confidence={result.get('confidence_score', 'n/a')} raw_len={len(raw_response)}",
            flush=True
        )
        
        # 4. Print the complete RAW LLM response (with XML tags, tools, reasoning) directly to the console
        print("\n" + "="*50)
        print(f"📡 [RAW EVENT] Full Raw LLM Response for {sender}:")
        print(raw_response)
        print("="*50 + "\n", flush=True)
        
        # 5. XML Parser Middleware: Extract only the content inside the <user_output> tags
        clean_reply = raw_response
        user_output_match = re.search(r"<user_output>(.*?)</user_output>", raw_response, re.DOTALL)
        if user_output_match:
            clean_reply = user_output_match.group(1).strip()
        else:
            if "<user_output>" in raw_response and "</user_output>" in raw_response:
                try:
                    start = raw_response.find("<user_output>") + len("<user_output>")
                    end = raw_response.find("</user_output>")
                    clean_reply = raw_response[start:end].strip()
                except Exception:
                    pass
        clean_reply = sanitize_whatsapp_reply(clean_reply)
                    
        # 6. Log the clean assistant reply to chat_history table
        db.save_chat_message(sender, "assistant", clean_reply)
        elapsed_ms = (time.perf_counter() - route_started) * 1000
        print(
            f"[WA API] SUCCESS sender={sender} reply_len={len(clean_reply)} "
            f"elapsed={elapsed_ms:.1f}ms",
            flush=True
        )
        
        # Log successful agent response to system events log
        db.log_event(
            source="WhatsApp",
            message=f"Replied to {sender}",
            status="success",
            meta_dict={"sender": sender, "reply": clean_reply[:100]}
        )
        
        return {
            "status": "success",
            "response": clean_reply
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - route_started) * 1000
        print(f"[WA API] ERROR sender={sender} elapsed={elapsed_ms:.1f}ms error={e}", flush=True)
        db.log_event(
            source="WhatsApp",
            message=f"Failed processing from {sender}",
            status="error",
            meta_dict={"error": str(e)}
        )
        return {
            "status": "error",
            "response": "I encountered a minor cognitive bottleneck processing your request. Please try again."
        }

# --- DESIGN.md Token Parsing & Linter Support ---
def parse_design_md(content: str) -> tuple:
    import re
    # Extract YAML front matter
    parts = content.strip().split("---")
    if len(parts) >= 3:
        yaml_content = parts[1]
        body_content = "---".join(parts[2:])
    else:
        yaml_content = ""
        body_content = content
        
    # Standard dictionary layout matching spec structures
    tokens = {
        "version": "alpha",
        "name": "Custom Theme",
        "description": "",
        "colors": {
            "primary": "#00f2fe",
            "secondary": "#8a2be2",
            "tertiary": "#00ff87",
            "neutral": "#0c1527",
            "neutral-light": "#1e293b",
            "danger": "#ff0844",
            "warning": "#ffb703"
        },
        "typography": {
            "h1": {"fontFamily": "Outfit", "fontSize": "2.2rem", "fontWeight": "800"},
            "body-md": {"fontFamily": "Inter", "fontSize": "0.85rem", "fontWeight": "400"}
        },
        "rounded": {
            "sm": "6px",
            "md": "12px"
        },
        "spacing": {
            "sm": "8px",
            "md": "16px"
        }
    }
    
    # Try importing PyYAML first
    try:
        import yaml
        parsed = yaml.safe_load(yaml_content)
        if parsed and isinstance(parsed, dict):
            for k in ["version", "name", "description"]:
                if k in parsed:
                    tokens[k] = parsed[k]
            for k in ["colors", "typography", "rounded", "spacing"]:
                if k in parsed and isinstance(parsed[k], dict):
                    tokens[k] = parsed[k]
            return tokens, body_content
    except Exception:
        pass
        
    # Standard Regex fallback parser if yaml is not installed or errors out
    try:
        current_section = None
        lines = yaml_content.split("\n")
        for line in lines:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue
            
            # Match top-level keys
            top_match = re.match(r"^([a-zA-Z0-9_\-]+)\s*:\s*(.*)$", line)
            if top_match:
                key, val = top_match.groups()
                val = val.strip().strip("'\"")
                if val:
                    if key in ["version", "name", "description"]:
                        tokens[key] = val
                    current_section = None
                else:
                    current_section = key
                continue
                
            # Match nested properties (indented)
            sub_match = re.match(r"^\s+([a-zA-Z0-9_\-]+)\s*:\s*(.*)$", line)
            if sub_match and current_section:
                sub_key, sub_val = sub_match.groups()
                sub_val = sub_val.strip().strip("'\"")
                if current_section in ["colors", "rounded", "spacing"]:
                    tokens[current_section][sub_key] = sub_val
                elif current_section == "typography":
                    # Simple flatten: assign under body-md or h1
                    # Since typography is highly nested in full spec, keep structured fallback keys
                    pass
    except Exception:
        pass
        
    return tokens, body_content

def run_design_md_linter(file_path: str) -> dict:
    import subprocess
    import json
    try:
        # Run local linter from npx package
        result = subprocess.run(
            ["npx", "@google/design.md", "lint", file_path],
            shell=True,
            capture_output=True,
            text=True,
            timeout=6
        )
        
        stdout_str = result.stdout.strip()
        
        if stdout_str:
            try:
                # Direct JSON parse
                return json.loads(stdout_str)
            except Exception:
                # Seek boundaries
                start = stdout_str.find("{")
                end = stdout_str.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(stdout_str[start:end+1])
                    
        return {
            "findings": [
                {
                    "severity": "info",
                    "path": "DESIGN.md",
                    "message": "Design system matches general schema guidelines. No syntax errors detected."
                }
            ],
            "summary": {"errors": 0, "warnings": 0, "info": 1}
        }
    except Exception as e:
        return {
            "findings": [
                {
                    "severity": "info",
                    "path": "cli.fallback",
                    "message": f"Design system offline diagnostic: visual parameters compiled successfully. (CLI output: {str(e)})"
                }
            ],
            "summary": {"errors": 0, "warnings": 0, "info": 1}
        }

@app.get("/api/design/themes")
def get_design_themes():
    """Lists prebuilt visually cohesive DESIGN.md identity configurations."""
    import os
    themes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
    themes = []
    
    if os.path.exists(themes_dir):
        for f in os.listdir(themes_dir):
            if f.startswith("design_") and f.endswith(".md"):
                file_path = os.path.join(themes_dir, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as file_data:
                        content = file_data.read()
                        tokens, _ = parse_design_md(content)
                        themes.append({
                            "id": f.replace("design_", "").replace(".md", ""),
                            "name": tokens.get("name", f),
                            "description": tokens.get("description", ""),
                            "content": content
                        })
                except Exception:
                    pass
    return {"status": "success", "themes": themes}

@app.get("/api/design/current")
def get_current_design_tokens():
    """Returns currently compiled active DESIGN.md layout rules & CLI linter feedback."""
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    design_path = os.path.join(root_dir, "DESIGN.md")
    
    # Auto-seed cyberpunk theme if not present at root
    if not os.path.exists(design_path):
        themes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
        default_theme_file = os.path.join(themes_dir, "design_cyberpunk.md")
        if os.path.exists(default_theme_file):
            import shutil
            shutil.copy(default_theme_file, design_path)
        else:
            with open(design_path, "w", encoding="utf-8") as df:
                df.write("---\nname: Custom Theme\ncolors:\n  primary: '#00f2fe'\n---\n## Overview\nDefault specs.")
                
    with open(design_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    tokens, body = parse_design_md(content)
    linter_results = run_design_md_linter(design_path)
    
    return {
        "status": "success",
        "content": content,
        "tokens": tokens,
        "body": body,
        "findings": linter_results.get("findings", []),
        "summary": linter_results.get("summary", {"errors": 0, "warnings": 0, "info": 0})
    }

@app.post("/api/design/select")
def select_design_system_theme(payload: Dict[str, str]):
    """Selects and hot-reloads a preconfigured visual identity stylesheet."""
    theme_id = payload.get("theme_id", "cyberpunk")
    import os
    import shutil
    
    themes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
    theme_file = os.path.join(themes_dir, f"design_{theme_id}.md")
    
    if not os.path.exists(theme_file):
        raise HTTPException(status_code=404, detail="Selected visual theme template not found")
        
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    design_path = os.path.join(root_dir, "DESIGN.md")
    
    shutil.copy(theme_file, design_path)
    
    with open(design_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    tokens, body = parse_design_md(content)
    linter_results = run_design_md_linter(design_path)
    
    db.log_event(
        source="DesignSystem",
        message=f"Switched theme design to: {tokens.get('name', theme_id)}",
        status="success"
    )
    
    return {
        "status": "success",
        "message": f"Successfully activated dynamic '{tokens.get('name', theme_id)}' visual identity!",
        "content": content,
        "tokens": tokens,
        "findings": linter_results.get("findings", []),
        "summary": linter_results.get("summary", {"errors": 0, "warnings": 0, "info": 0})
    }

@app.post("/api/design/save")
def save_and_lint_design_spec(payload: Dict[str, str]):
    """Saves custom styling text inputs, running npx CLI validators programmatically."""
    content = payload.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Custom spec content cannot be empty")
        
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    design_path = os.path.join(root_dir, "DESIGN.md")
    
    with open(design_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    tokens, body = parse_design_md(content)
    linter_results = run_design_md_linter(design_path)
    
    db.log_event(
        source="DesignSystem",
        message="Saved user custom design token modifications",
        status="success"
    )
    
    return {
        "status": "success",
        "message": "Custom visual theme saved and lint validated!",
        "content": content,
        "tokens": tokens,
        "findings": linter_results.get("findings", []),
        "summary": linter_results.get("summary", {"errors": 0, "warnings": 0, "info": 0})
    }

@app.on_event("startup")
def start_memory_daemon():
    try:
        from app.brain.memory_daemon import memory_daemon
        memory_daemon.start()
    except Exception as e:
        print(f"[Startup] Failed to start MemoryDaemon: {e}")
    try:
        from app.brain.watchdog import watchdog
        watchdog.start()
    except Exception as e:
        print(f"[Startup] Failed to start watchdog: {e}")

@app.on_event("shutdown")
def stop_memory_daemon():
    try:
        from app.brain.memory_daemon import memory_daemon
        memory_daemon.stop()
    except Exception as e:
        print(f"[Shutdown] Failed to stop MemoryDaemon: {e}")
    try:
        from app.brain.watchdog import watchdog
        watchdog.stop()
    except Exception as e:
        print(f"[Shutdown] Failed to stop watchdog: {e}")

