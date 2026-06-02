from __future__ import annotations

import re
from typing import Dict, List, Optional


SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9_-]{12,}\b",
    r"\bsk-or-[A-Za-z0-9_-]{12,}\b",
    r"\b(api[_ -]?key|password|passwd|secret|token)\b\s*[:=]",
    r"\b[A-Za-z0-9_-]{32,}\b",
]

MEMORY_PATTERNS = [
    r"\bmy name is\s+(.+)",
    r"\bcall me\s+(.+)",
    r"\bi prefer\s+(.+)",
    r"\bi like\s+(.+)",
    r"\bi use\s+(.+)",
    r"\bi am using\s+(.+)",
    r"\bmy project\s+(.+)",
    r"\bremember that\s+(.+)",
    r"\bremember this\s+(.+)",
    r"\bfrom now on\s+(.+)",
]


def has_sensitive_secret(text: str) -> bool:
    value = text or ""
    return any(re.search(pattern, value, flags=re.I) for pattern in SECRET_PATTERNS)


def clean_memory_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = cleaned.strip(" .,:;")
    return cleaned[:500]


def extract_memory_candidates(user_message: str, assistant_response: str = "") -> List[str]:
    message = clean_memory_text(user_message)
    if not message or has_sensitive_secret(message):
        return []

    candidates: List[str] = []
    lowered = message.lower()

    for pattern in MEMORY_PATTERNS:
        match = re.search(pattern, message, flags=re.I)
        if match:
            fact = clean_memory_text(match.group(0))
            if fact:
                candidates.append(fact)

    durable_markers = [
        "always",
        "usually",
        "my goal",
        "i want sophie",
        "i want you",
        "my timezone",
        "my work",
        "my business",
        "my backend",
        "my whatsapp",
        "i don't like",
        "i do not like",
    ]
    if any(marker in lowered for marker in durable_markers):
        candidates.append(message)

    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate.lower() not in {item.lower() for item in unique}:
            unique.append(candidate)
    return unique[:3]


def review_and_save_memories(
    sender: str,
    user_message: str,
    assistant_response: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    """Deterministically saves durable user facts without storing secrets or ordinary chatter."""
    from app.brain.memory_os import memory_os

    saved: List[Dict[str, str]] = []
    skipped_reason = ""
    candidates = extract_memory_candidates(user_message, assistant_response)
    if not candidates:
        skipped_reason = "No durable user fact or preference detected."

    for candidate in candidates:
        if has_sensitive_secret(candidate):
            skipped_reason = "Sensitive secret-like content was skipped."
            continue
        content = f"User {sender}: {candidate}"
        memory_id = memory_os.add_semantic_memory(content)
        saved.append({"id": memory_id, "content": content})

    return {
        "saved_count": len(saved),
        "saved": saved,
        "skipped_reason": skipped_reason,
    }
