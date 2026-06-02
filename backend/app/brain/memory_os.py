import uuid
import datetime
import time
from typing import List, Dict, Any, Optional

from app.config import GEMINI_API_KEY, PINECONE_INDEX_NAME
from app.database import db
from app.brain.doc_brain import doc_brain

class WorkingMemory:
    """
    Per-session RAM buffer. Cleared after each conversation ends.
    Holds: current task state, entities mentioned, tool results so far,
    emotion/tone detected in user's last message.
    """
    def __init__(self):
        self._buffer: Dict[str, Any] = {}
        self._entity_mentions: List[dict] = []
        self._task_stack: List[dict] = []  # Stack for nested tasks

    def push_task(self, task: dict):
        self._task_stack.append(task)

    def pop_task(self) -> Optional[dict]:
        return self._task_stack.pop() if self._task_stack else None

    def note_entity(self, entity: str, entity_type: str):
        self._entity_mentions.append({"entity": entity, "type": entity_type, "ts": time.time()})

    def set(self, key: str, value: Any):
        self._buffer[key] = value

    def get(self, key: str, default=None) -> Any:
        return self._buffer.get(key, default)

    def clear(self):
        self._buffer.clear()
        self._entity_mentions.clear()
        self._task_stack.clear()

    def snapshot(self) -> dict:
        return {
            "buffer": dict(self._buffer),
            "entities": list(self._entity_mentions[-10:]),
            "task_depth": len(self._task_stack)
        }


class MemoryOS:
    def __init__(self):
        self.index = doc_brain.index
        # Default half-lives
        self.episodic_half_life_days = 7.0   # Conversations decay faster
        self.semantic_half_life_days = 30.0  # Facts remain fresh longer
        self.working_memory = WorkingMemory()

    def get_decay_score(self, created_at_str: str, base_score: float, half_life_days: float) -> float:
        """Calculates freshness score using half-life decay formula."""
        try:
            created_at = datetime.datetime.fromisoformat(created_at_str)
            now = datetime.datetime.utcnow()
            elapsed_seconds = (now - created_at).total_seconds()
            elapsed_days = elapsed_seconds / 86400.0
            
            # Freshness = base_score * (0.5 ** (days_elapsed / half_life))
            freshness = base_score * (0.5 ** (elapsed_days / half_life_days))
            return max(0.0, round(freshness, 2))
        except Exception as e:
            print(f"Error calculating decay score: {e}")
            return base_score

    def add_episodic_memory(self, content: str) -> str:
        """Saves a conversation session memory to Pinecone ('episodic' namespace) and SQLite."""
        mem_id = f"ep_{uuid.uuid4().hex[:12]}"
        embedding = doc_brain.get_embedding(content, is_query=False)
        
        # Save to Pinecone
        try:
            if self.index:
                self.index.upsert(
                    vectors=[{
                        "id": mem_id,
                        "values": embedding,
                        "metadata": {
                            "type": "episodic",
                            "content": content,
                            "timestamp": datetime.datetime.utcnow().isoformat()
                        }
                    }],
                    namespace="episodic"
                )
        except Exception as e:
            print(f"Error saving episodic vector: {e}")
            
        # Save to local SQLite decay tracker
        db.save_memory_record(
            memory_id=mem_id,
            mem_type="episodic",
            content=content,
            base_score=100.0,
            half_life_days=self.episodic_half_life_days
        )
        
        db.log_event(
            source="MemoryOS",
            message=f"Logged new episodic memory: {content[:45]}...",
            status="info",
            meta_dict={"memory_id": mem_id, "type": "episodic"}
        )
        return mem_id

    def add_semantic_memory(self, content: str) -> str:
        """Saves generalized facts/statements to Pinecone ('semantic' namespace) and SQLite."""
        mem_id = f"sem_{uuid.uuid4().hex[:12]}"
        embedding = doc_brain.get_embedding(content, is_query=False)
        
        # Save to Pinecone
        try:
            if self.index:
                self.index.upsert(
                    vectors=[{
                        "id": mem_id,
                        "values": embedding,
                        "metadata": {
                            "type": "semantic",
                            "content": content,
                            "timestamp": datetime.datetime.utcnow().isoformat()
                        }
                    }],
                    namespace="semantic"
                )
        except Exception as e:
            print(f"Error saving semantic vector: {e}")
            
        # Save to local SQLite decay tracker
        db.save_memory_record(
            memory_id=mem_id,
            mem_type="semantic",
            content=content,
            base_score=100.0,
            half_life_days=self.semantic_half_life_days
        )
        
        db.log_event(
            source="MemoryOS",
            message=f"Logged new semantic fact: {content[:45]}...",
            status="info",
            meta_dict={"memory_id": mem_id, "type": "semantic"}
        )
        return mem_id

    def add_associative_memory(self, entity_a: str, relation: str, entity_b: str, strength: float = 1.0) -> str:
        """Links concepts together in memory."""
        mem_id = f"asc_{uuid.uuid4().hex[:12]}"
        content = f"{entity_a} {relation} {entity_b}"
        embedding = doc_brain.get_embedding(content, is_query=False)
        
        # Store in Pinecone
        try:
            if self.index:
                self.index.upsert(
                    vectors=[{
                        "id": mem_id,
                        "values": embedding,
                        "metadata": {
                            "type": "associative",
                            "entity_a": entity_a,
                            "relation": relation,
                            "entity_b": entity_b,
                            "strength": strength,
                            "timestamp": datetime.datetime.utcnow().isoformat()
                        }
                    }],
                    namespace="associative"
                )
        except Exception as e:
            print(f"Error saving associative vector: {e}")
            
        # Save to SQLite table
        db.save_assoc_memory(
            id=mem_id,
            entity_a=entity_a,
            relation=relation,
            entity_b=entity_b,
            strength=strength
        )
        return mem_id

    def check_promotion_candidates(self) -> int:
        """Promotes active episodic memories that have been accessed multiple times to semantic status."""
        promoted_count = 0
        try:
            records = db.get_all_memory_records()
            for r in records:
                # Simple heuristic: if episodic and freshness is high and we decide to consolidate/promote
                # Normally daemon handles episodic consolidation, but this provides the trigger interface.
                pass
        except Exception as e:
            print(f"Error in check_promotion_candidates: {e}")
        return promoted_count

    def save_procedural_memory(self, task_type: str, steps: List[str]) -> bool:
        """Saves exact steps/workflows for handling recurring automation tasks."""
        success = db.save_procedural_memory(task_type, steps)
        if success:
            db.log_event(
                source="MemoryOS",
                message=f"Updated procedural steps for task: '{task_type}'",
                status="info",
                meta_dict={"task_type": task_type, "steps_count": len(steps)}
            )
        return success

    def get_procedural_memory(self, task_type: str) -> Optional[List[str]]:
        """Retrieves exact procedural workflow steps from SQLite."""
        return db.get_procedural_memory(task_type)

    def _retrieve_memories_local(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        records = db.get_all_memory_records()
        keywords = [kw.strip().lower() for kw in query.split() if len(kw.strip()) > 2]
        if not keywords:
            keywords = [query.lower()]

        local_matches = []
        for r in records:
            match_count = 0
            content_lower = r["content"].lower()
            for kw in keywords:
                if kw in content_lower:
                    match_count += 1

            if match_count > 0 or query.lower() in content_lower:
                freshness = self.get_decay_score(
                    created_at_str=r["created_at"],
                    base_score=r["base_score"],
                    half_life_days=r["half_life_days"]
                )
                status = "stale" if freshness < 40.0 else "active"

                local_matches.append({
                    "id": r["id"],
                    "type": r["type"],
                    "content": r["content"],
                    "freshness": freshness,
                    "status": status,
                    "score": 0.5 + (match_count * 0.1)
                })

        local_matches.sort(key=lambda x: x["score"], reverse=True)
        return local_matches[:limit]

    def retrieve_memories(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Vector searches episodic and semantic namespaces, recalculates freshness decay, and updates statuses."""
        if not GEMINI_API_KEY:
            print("[MemoryOS] GEMINI_API_KEY missing. Using local SQLite memory keyword search.")
            return self._retrieve_memories_local(query, limit)

        query_vector = doc_brain.get_embedding(query, is_query=True)
        if not any(query_vector):
            print("[MemoryOS] Embedding returned a zero vector. Using local SQLite memory keyword search.")
            return self._retrieve_memories_local(query, limit)

        results = []
        
        for ns in ["episodic", "semantic"]:
            try:
                response = self.index.query(
                    namespace=ns,
                    vector=query_vector,
                    top_k=limit,
                    include_metadata=True
                )
                
                for match in response.get("matches", []):
                    mem_id = match.get("id")
                    score = match.get("score", 0.0)
                    
                    # Fetch from SQLite decay record to calculate freshness based on time
                    decay_matches = [m for m in db.get_all_memory_records() if m["id"] == mem_id]
                    if decay_matches:
                        decay_rec = decay_matches[0]
                        freshness = self.get_decay_score(
                            created_at_str=decay_rec["created_at"],
                            base_score=decay_rec["base_score"],
                            half_life_days=decay_rec["half_life_days"]
                        )
                        
                        # Apply staleness threshold
                        status = "stale" if freshness < 40.0 else "active"
                        if status != decay_rec["status"]:
                            db.update_memory_status(mem_id, status)
                            
                        # Touch memory to show interaction
                        db.touch_memory(mem_id)
                        
                        results.append({
                            "id": mem_id,
                            "type": ns,
                            "content": decay_rec["content"],
                            "freshness": freshness,
                            "status": status,
                            "score": score
                        })
            except Exception as e:
                print(f"Pinecone query error in memory namespace '{ns}': {e}")
                
        # Fallback to local SQLite keyword matching if no vector memories are retrieved
        if not results:
            print("[MemoryOS] Pinecone memory search returned no results. Invoking local SQLite memory fallback search...")
            results = self._retrieve_memories_local(query, limit)
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_all_decay_states(self) -> List[Dict[str, Any]]:
        """Returns lists of all episodic/semantic memories from SQLite with computed freshness scores for UI view."""
        records = db.get_all_memory_records()
        enriched = []
        for r in records:
            freshness = self.get_decay_score(
                created_at_str=r["created_at"],
                base_score=r["base_score"],
                half_life_days=r["half_life_days"]
            )
            status = "stale" if freshness < 40.0 else "active"
            if status != r["status"]:
                db.update_memory_status(r["id"], status)
                
            enriched.append({
                "id": r["id"],
                "type": r["type"],
                "content": r["content"],
                "base_score": r["base_score"],
                "created_at": r["created_at"],
                "half_life_days": r["half_life_days"],
                "freshness": freshness,
                "status": status
            })
        return enriched

    def trigger_artificial_decay(self, elapsed_days: float) -> List[Dict[str, Any]]:
        """Simulates time elapsed to decay all base scores. Highly useful for demonstrating UI states."""
        records = db.get_all_memory_records()
        for r in records:
            # Shift the created_at timestamp backward in sqlite
            created = datetime.datetime.fromisoformat(r["created_at"])
            shifted = created - datetime.timedelta(days=elapsed_days)
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memory_decay_records SET created_at = ? WHERE id = ?",
                    (shifted.isoformat(), r["id"])
                )
                conn.commit()
        return self.get_all_decay_states()

# Global memory instance
memory_os = MemoryOS()
