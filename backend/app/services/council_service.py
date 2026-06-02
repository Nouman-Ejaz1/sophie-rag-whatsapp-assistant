from typing import Dict, Any
from app.brain.thinking_palace import thinking_palace

class CouncilService:
    """
    Decoupled service encapsulating the multi-turn agent review council loops,
    draft validation, and plan execution feedback loops.
    """
    @staticmethod
    def execute_council(user_message: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the main sub-agent validation loop on intent plans."""
        return thinking_palace._deep_think(user_message, context, intent)

council_service = CouncilService()
