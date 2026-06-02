from typing import Dict, Any
from app.brain.thinking_palace import thinking_palace

class IntentService:
    """
    Decoupled service handling intent routing, greeting checks,
    and fast-classifier LLM calls to map queries.
    """
    @staticmethod
    def classify(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the fast-intent classification engine (LLM or heuristic fallback)."""
        return thinking_palace._classify_intent(message, context)

    @staticmethod
    def is_trivial(message: str) -> bool:
        """Stage 0 instant check to filter greetings/acks under 24 chars."""
        return thinking_palace._is_trivial(message)

intent_service = IntentService()
