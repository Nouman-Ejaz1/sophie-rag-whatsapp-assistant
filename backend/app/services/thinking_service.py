from typing import Dict, Any
from app.brain.thinking_palace import thinking_palace

class ThinkingService:
    """
    Decoupled service encapsulating the Thinking Palace reasoning flow.
    Runs intent classification, deep recursive planning loops, validation,
    and returns formatted WhatsApp outputs.
    """
    @staticmethod
    def process(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Routes message through the 3-stage Thinking Palace engine."""
        return thinking_palace.think(message, context)

thinking_service = ThinkingService()
