from typing import Dict, Any
from app.brain.preference_engine import preference_engine

class LearningService:
    """
    Decoupled service handling user style feedback, signal extraction,
    and structured user profile updates from interactive message triggers.
    """
    @staticmethod
    def extract_signals(sender: str, message: str) -> None:
        """Parses active message string and logs conciseness, topic, language cues to SQLite."""
        preference_engine.analyze_message(sender, message)

    @staticmethod
    def get_style_hints(sender: str) -> Dict[str, Any]:
        """Gathers aggregated behavioral feedback for modifying prompt instructions."""
        return preference_engine.get_style_hints(sender)

learning_service = LearningService()
