from typing import List, Dict, Any, Optional
from app.brain.memory_os import memory_os
from app.brain.doc_brain import doc_brain

class MemoryService:
    """
    Decoupled service for orchestrating 6-layer memory interactions.
    Provides standard wrappers around MemoryOS, Pinecone, and SQLite.
    """
    @staticmethod
    def retrieve_context(query: str, sender: str, doc_limit: int = 5, mem_limit: int = 5) -> Dict[str, Any]:
        """Retrieves combined Vector Doc chunks and decay-calculated memories."""
        docs = doc_brain.query_knowledge(query, limit=doc_limit)
        memories = memory_os.retrieve_memories(query, limit=mem_limit)
        return {
            "docs": docs,
            "memories": memories
        }

    @staticmethod
    def save_episodic(content: str) -> str:
        """Saves episodic conversation summary."""
        return memory_os.add_episodic_memory(content)

    @staticmethod
    def save_semantic(content: str) -> str:
        """Saves durable semantic facts."""
        return memory_os.add_semantic_memory(content)

    @staticmethod
    def save_associative(entity_a: str, relation: str, entity_b: str, strength: float = 1.0) -> str:
        """Links entities semantic associations."""
        return memory_os.add_associative_memory(entity_a, relation, entity_b, strength)

    @staticmethod
    def get_procedural(task_type: str) -> Optional[List[str]]:
        """Fetches permanent procedural checklists."""
        return memory_os.get_procedural_memory(task_type)

    @staticmethod
    def save_procedural(task_type: str, steps: List[str]) -> bool:
        """Saves/Updates permanent procedural checklists."""
        return memory_os.save_procedural_memory(task_type, steps)

memory_service = MemoryService()
