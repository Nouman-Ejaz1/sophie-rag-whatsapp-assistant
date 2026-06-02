from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- API Models ---
class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    title: str
    content: str
    source_type: str = Field(default="url", description="url, notion, pdf, gmail")
    source_url: Optional[str] = None

class IngestSourceRequest(BaseModel):
    source_kind: str = Field(..., description="text, local_path, url, whatsapp_media")
    title: Optional[str] = None
    content: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    media_base64: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    source_type: str = Field(default="pdf", description="url, notion, pdf, gmail, whatsapp")
    source_url: Optional[str] = None

class MockEventRequest(BaseModel):
    source: str = Field(..., description="github, slack, calendar, news")
    title: str
    description: str
    metadata: Optional[Dict[str, Any]] = None

# --- Data Schemas ---
class Citation(BaseModel):
    source: str
    text: str
    confidence: float

class MemoryItem(BaseModel):
    id: str
    type: str  # episodic, semantic, procedural
    content: str
    freshness: float
    status: str

# --- LangGraph State Schema ---
class AgentState(BaseModel):
    # Incoming triggers / event
    triggered_event: Optional[Dict[str, Any]] = None
    
    # Conversations
    messages: List[Dict[str, str]] = []  # format: {"role": "user"|"assistant", "content": "..."}
    
    # Retrieved contexts
    retrieved_docs: List[Dict[str, Any]] = []
    retrieved_memories: List[Dict[str, Any]] = []
    
    # Decisions / Actions
    is_urgent: bool = False
    reasoning: str = ""
    citations: List[Citation] = []
    confidence_score: float = 100.0
    action_taken: Optional[str] = None
    
    # Final Response
    response: str = ""
