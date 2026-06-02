import time
import threading
import datetime
from typing import List, Dict, Any

from app.database import db
from app.brain.memory_os import memory_os
from app.brain.preference_engine import preference_engine

class MemoryDaemon:
    """
    JARVIS Memory Consolidation and Maintenance Background Daemon.
    Performs:
    1. Decay Sweeps: Computes current freshness scores, updates status to 'stale', 
       and archives memories with freshness < 5.0.
    2. Episodic Consolidation: Groups episodic memories, compresses them into semantic facts via LLM.
    3. User Profile Synthesis: Compiles preference signals into structured JSON profiles.
    """
    def __init__(self, interval_seconds: float = 3600.0):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None

    def start(self):
        """Starts the background daemon thread."""
        if self._thread and self._thread.is_alive():
            print("[MemoryDaemon] Daemon is already running.")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MemoryDaemonThread")
        self._thread.start()
        print(f"[MemoryDaemon] Daemon started with interval of {self.interval_seconds}s.")
        db.log_event(source="MemoryDaemon", message="Memory Daemon started background thread", status="success")

    def stop(self):
        """Stops the background daemon thread."""
        if not self._thread or not self._thread.is_alive():
            return
        
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        print("[MemoryDaemon] Daemon stopped.")
        db.log_event(source="MemoryDaemon", message="Memory Daemon stopped background thread", status="info")

    def trigger_maintenance_sweep(self):
        """Manually triggers a full maintenance cycle (decay, consolidation, profile synthesis)."""
        print("[MemoryDaemon] Initiating manual maintenance sweep...")
        try:
            self._run_decay_sweep()
            self._run_episodic_consolidation()
            self._run_profile_synthesis()
            print("[MemoryDaemon] Manual maintenance sweep completed successfully.")
        except Exception as e:
            print(f"[MemoryDaemon] Error during manual sweep: {e}")

    def _run_loop(self):
        """Main daemon loop."""
        # Initial sleep to let the system warm up
        time.sleep(10.0)
        
        while not self._stop_event.is_set():
            try:
                self._run_decay_sweep()
                self._run_episodic_consolidation()
                self._run_profile_synthesis()
            except Exception as e:
                print(f"[MemoryDaemon] Error in background execution loop: {e}")
                db.log_event(source="MemoryDaemon", message=f"Daemon execution error: {str(e)}", status="error")
            
            # Wait for interval or stop event
            self._stop_event.wait(self.interval_seconds)

    def _run_decay_sweep(self):
        """
        Scans all memory records, recalculates freshness scores,
        updates status to stale if freshness < 40, and archives if freshness < 5.0.
        """
        records = db.get_all_memory_records()
        decayed_count = 0
        archived_count = 0

        for r in records:
            freshness = memory_os.get_decay_score(
                created_at_str=r["created_at"],
                base_score=r["base_score"],
                half_life_days=r["half_life_days"]
            )
            
            if freshness < 5.0:
                # Archive extremely stale memory
                db.archive_memory(
                    memory_id=r["id"],
                    memory_type=r["type"],
                    content=r["content"],
                    created_at=r["created_at"]
                )
                db.delete_memory_record(r["id"])
                
                # Delete vector from Pinecone if configured
                try:
                    if memory_os.index:
                        memory_os.index.delete(ids=[r["id"]], namespace=r["type"])
                except Exception as ve:
                    print(f"[MemoryDaemon] Vector deletion error for {r['id']}: {ve}")
                
                archived_count += 1
            elif freshness < 40.0 and r["status"] == "active":
                db.update_memory_status(r["id"], "stale")
                decayed_count += 1

        if decayed_count > 0 or archived_count > 0:
            msg = f"Decay sweep complete. Stale transition: {decayed_count}, Archived: {archived_count}."
            print(f"[MemoryDaemon] {msg}")
            db.log_event(
                source="MemoryDaemon",
                message=msg,
                status="info",
                meta_dict={"stale_transition": decayed_count, "archived": archived_count}
            )

    def _run_episodic_consolidation(self):
        """
        Gathers raw episodic memory logs, uses LLM (via OpenRouter) to consolidate them
        into generalizable semantic facts, and promotes them.
        """
        # Find active episodic memories ripe for consolidation (e.g. at least 3)
        records = db.get_all_memory_records()
        episodic_recs = [r for r in records if r["type"] == "episodic" and r["status"] == "active"]
        
        if len(episodic_recs) < 3:
            return  # Not enough memories to consolidate

        print(f"[MemoryDaemon] Consolidating {len(episodic_recs)} episodic memories...")
        
        # Compile logs content
        logs_text = "\n".join([f"- [{r['created_at']}] {r['content']}" for r in episodic_recs])
        
        system_prompt = (
            "You are JARVIS's Memory Consolidation Engine.\n"
            "Analyze the following list of raw episodic memories (conversations, queries, actions) and extract "
            "any long-term semantic facts, rules, or preferences about the user. Ignore conversational filler or temp tasks.\n"
            "Format the output strictly as standalone bullet points (one fact per line). Do not write intro/outro text, headers, or explanations."
        )
        
        try:
            # Import openrouter_chat dynamically to avoid circular dependencies
            from app.brain.orchestrator import openrouter_chat
            
            response = openrouter_chat(
                system_prompt=system_prompt,
                user_prompt="Consolidate these episodic logs:\n" + logs_text,
                user_message="memory_consolidation_sweep",
                requires_tools=False
            )
            
            facts = [line.strip("- *• ").strip() for line in response.strip().split("\n") if line.strip()]
            
            consolidated_count = 0
            for fact in facts:
                if len(fact) > 10:
                    memory_os.add_semantic_memory(fact)
                    consolidated_count += 1
            
            # Archive these episodic memories so they don't get consolidated again
            for r in episodic_recs:
                db.archive_memory(r["id"], r["type"], r["content"], r["created_at"])
                db.delete_memory_record(r["id"])
                try:
                    if memory_os.index:
                        memory_os.index.delete(ids=[r["id"]], namespace=r["type"])
                except Exception as ve:
                    print(f"[MemoryDaemon] Vector deletion error: {ve}")

            msg = f"Consolidated {len(episodic_recs)} episodic logs into {consolidated_count} semantic facts."
            print(f"[MemoryDaemon] {msg}")
            db.log_event(
                source="MemoryDaemon",
                message=msg,
                status="success",
                meta_dict={"episodic_count": len(episodic_recs), "semantic_created": consolidated_count}
            )

        except Exception as e:
            print(f"[MemoryDaemon] Episodic consolidation failed: {e}")
            db.log_event(source="MemoryDaemon", message=f"Episodic consolidation failed: {str(e)}", status="error")

    def _run_profile_synthesis(self):
        """
        Iterates over unique senders with preference signals, compiles style hints, 
        and updates their structured user profiles.
        """
        try:
            # Retrieve unique senders from preference_signals table
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT sender FROM preference_signals")
                senders = [row["sender"] for row in cursor.fetchall()]

            if not senders:
                return

            profile_updates = 0
            for sender in senders:
                hints = preference_engine.get_style_hints(sender)
                if hints:
                    db.save_user_profile(sender, hints)
                    profile_updates += 1

            if profile_updates > 0:
                print(f"[MemoryDaemon] User profiles updated for {profile_updates} users.")
                db.log_event(
                    source="MemoryDaemon",
                    message=f"Compiled structured profiles for {profile_updates} users",
                    status="success"
                )

        except Exception as e:
            print(f"[MemoryDaemon] Profile synthesis failed: {e}")
            db.log_event(source="MemoryDaemon", message=f"Profile synthesis failed: {str(e)}", status="error")

# Instantiate a global MemoryDaemon
memory_daemon = MemoryDaemon()
