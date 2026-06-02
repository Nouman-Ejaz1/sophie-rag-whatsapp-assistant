import sqlite3
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_FILE = Path(__file__).resolve().parent.parent / "local_data" / "sentinel_ai.db"

class Database:
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        # Automatically create the parent directory if it does not exist (gitignored local storage folder)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Procedural Memory Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedural_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT UNIQUE,
                workflow_steps TEXT,
                updated_at TEXT
            )
            """)

            # Ephemeral Logs Table (System wake-ups & alerts)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source TEXT,
                message TEXT,
                metadata TEXT,
                status TEXT
            )
            """)

            # Memory Decay Reference Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_decay_records (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                base_score REAL,
                created_at TEXT,
                half_life_days REAL,
                last_accessed_at TEXT,
                status TEXT
            )
            """)

            # Local Document Chunks Table (for local RAG fallback and premium Book library)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                chunk_index INTEGER,
                title TEXT,
                source TEXT,
                source_type TEXT,
                text TEXT,
                timestamp TEXT
            )
            """)

            # Repeating Tasks Table (for automated background research scheduler)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS repeating_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT UNIQUE,
                query TEXT,
                interval_hours REAL,
                last_run TEXT,
                target_number TEXT,
                active INTEGER DEFAULT 1
            )
            """)

            # Calendar Events Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                date_time TEXT
            )
            """)

            # One-time WhatsApp reminders
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                title TEXT,
                message TEXT,
                due_at TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                sent_at TEXT
            )
            """)

            # Nutrition tracking logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS nutrition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                food_name TEXT,
                quantity TEXT,
                calories REAL,
                protein_g REAL,
                carbs_g REAL,
                fat_g REAL,
                logged_at TEXT
            )
            """)

            # WhatsApp Chat History Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                role TEXT,
                message TEXT,
                timestamp TEXT
            )
            """)

            # Custom Sub-Agents Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_agents (
                name TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT
            )
            """)

            # Human Approval Queue (for destructive actions)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_approvals (
                id TEXT PRIMARY KEY,
                sender TEXT,
                action_type TEXT,
                command_line TEXT,
                target_path TEXT,
                created_at TEXT,
                expires_at TEXT,
                status TEXT,
                result TEXT
            )
            """)

            # User Profiles Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                sender TEXT PRIMARY KEY,
                profile_json TEXT,
                updated_at TEXT
            )
            """)

            # Preference Signals Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                signal_type TEXT,
                value TEXT,
                weight REAL,
                created_at TEXT
            )
            """)

            # Memory Archive Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_archive (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                created_at TEXT,
                archived_at TEXT
            )
            """)

            # Associative Memories Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS assoc_memories (
                id TEXT PRIMARY KEY,
                entity_a TEXT,
                relation TEXT,
                entity_b TEXT,
                strength REAL,
                created_at TEXT
            )
            """)

            # Request Logs/Metrics Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                timestamp TEXT,
                message_length INTEGER,
                response_time REAL,
                success INTEGER,
                error_message TEXT
            )
            """)

            # Smart Triggers Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS smart_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                name TEXT,
                metric TEXT,
                threshold_value REAL,
                threshold_direction TEXT,
                threshold_pct REAL,
                last_known_value REAL,
                watch_query TEXT,
                watch_keywords TEXT,
                schedule_cron TEXT,
                interval_hours REAL,
                message_template TEXT,
                last_run TEXT,
                last_triggered TEXT,
                fire_count INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                created_at TEXT,
                expires_at TEXT
            )
            """)

            conn.commit()

    # --- Procedural Memory Helpers ---
    def save_procedural_memory(self, task_type: str, steps: List[str]) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO procedural_memories (task_type, workflow_steps, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_type) DO UPDATE SET
                    workflow_steps=excluded.workflow_steps,
                    updated_at=excluded.updated_at
                """, (task_type, json.dumps(steps), now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_procedural_memory: {e}")
            return False

    def get_procedural_memory(self, task_type: str) -> Optional[List[str]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT workflow_steps FROM procedural_memories WHERE task_type = ?", 
                    (task_type,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row["workflow_steps"])
                return None
        except Exception as e:
            print(f"Database error in get_procedural_memory: {e}")
            return None

    # --- System Logs / Alerts Helpers ---
    def log_event(self, source: str, message: str, status: str = "info", meta_dict: Dict[str, Any] = None) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                meta_str = json.dumps(meta_dict or {})
                cursor.execute("""
                INSERT INTO system_logs (timestamp, source, message, metadata, status)
                VALUES (?, ?, ?, ?, ?)
                """, (now, source, message, meta_str, status))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Database error in log_event: {e}")
            return -1

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM system_logs ORDER BY id DESC LIMIT ?", 
                    (limit,)
                )
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "source": r["source"],
                        "message": r["message"],
                        "metadata": json.loads(r["metadata"]),
                        "status": r["status"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_logs: {e}")
            return []

    # --- Memory Decay Scoring Helpers ---
    def save_memory_record(self, memory_id: str, mem_type: str, content: str, base_score: float, half_life_days: float) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT OR REPLACE INTO memory_decay_records 
                (id, type, content, base_score, created_at, half_life_days, last_accessed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (memory_id, mem_type, content, base_score, now, half_life_days, now, "active"))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_memory_record: {e}")
            return False

    def get_all_memory_records(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memory_decay_records")
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "type": r["type"],
                        "content": r["content"],
                        "base_score": r["base_score"],
                        "created_at": r["created_at"],
                        "half_life_days": r["half_life_days"],
                        "last_accessed_at": r["last_accessed_at"],
                        "status": r["status"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_all_memory_records: {e}")
            return []

    def update_memory_status(self, memory_id: str, status: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memory_decay_records SET status = ? WHERE id = ?",
                    (status, memory_id)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in update_memory_status: {e}")
            return False

    def touch_memory(self, memory_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute(
                    "UPDATE memory_decay_records SET last_accessed_at = ? WHERE id = ?",
                    (now, memory_id)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in touch_memory: {e}")
            return False

    def delete_memory_record(self, memory_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory_decay_records WHERE id = ?", (memory_id,))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in delete_memory_record: {e}")
            return False


    # --- Local Knowledge / RAG Fallback Helpers ---
    def save_knowledge_chunk(self, doc_id: str, chunk_index: int, title: str, source: str, source_type: str, text: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO knowledge_chunks (doc_id, chunk_index, title, source, source_type, text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, chunk_index, title, source, source_type, text, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_knowledge_chunk: {e}")
            return False

    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT doc_id, title, source, source_type, COUNT(id) as chunk_count, MIN(timestamp) as created_at
                FROM knowledge_chunks
                GROUP BY title
                ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "doc_id": r["doc_id"],
                        "title": r["title"],
                        "source": r["source"],
                        "source_type": r["source_type"],
                        "chunk_count": r["chunk_count"],
                        "created_at": r["created_at"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_all_documents: {e}")
            return []

    def get_document_chunks(self, title: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT * FROM knowledge_chunks
                WHERE title = ?
                ORDER BY chunk_index ASC
                """, (title,))
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "doc_id": r["doc_id"],
                        "chunk_index": r["chunk_index"],
                        "title": r["title"],
                        "source": r["source"],
                        "source_type": r["source_type"],
                        "text": r["text"],
                        "timestamp": r["timestamp"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_document_chunks: {e}")
            return []

    def search_knowledge_chunks(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """Performs simple local text search fallback for RAG when Pinecone is missing/empty."""
        try:
            # Split the query into keywords
            keywords = [kw.strip() for kw in query.lower().split() if len(kw.strip()) > 2]
            if not keywords:
                keywords = [query.lower()]
                
            # Build basic OR query structure for keywords
            conditions = []
            params = []
            for kw in keywords:
                conditions.append("(LOWER(text) LIKE ? OR LOWER(title) LIKE ?)")
                params.extend([f"%{kw}%", f"%{kw}%"])
                
            where_clause = " OR ".join(conditions)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                SELECT *, 
                       (CASE WHEN LOWER(title) LIKE ? THEN 3.0 ELSE 0.0 END + 
                        CASE WHEN LOWER(text) LIKE ? THEN 1.0 ELSE 0.0 END) as relevance_score
                FROM knowledge_chunks
                WHERE {where_clause}
                ORDER BY relevance_score DESC, id ASC
                LIMIT ?
                """, [f"%{query.lower()}%", f"%{query.lower()}%"] + params + [limit])
                
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    # Synthesize a confidence score (local search matches start at 75% and decay slightly)
                    score = r["relevance_score"] if "relevance_score" in r.keys() else 1.0
                    confidence = min(95.0, 75.0 + score * 5.0)
                    
                    results.append({
                        "doc_id": r["doc_id"],
                        "title": r["title"],
                        "source": r["source"],
                        "source_type": r["source_type"],
                        "text": r["text"],
                        "chunk_index": r["chunk_index"],
                        "confidence": round(confidence, 1),
                        "score": float(score)
                    })
                return results
        except Exception as e:
            print(f"Database error in search_knowledge_chunks fallback: {e}")
            return []

    # --- Repeating Background Tasks Helpers ---
    def save_repeating_task(self, task_name: str, query: str, interval_hours: float, target_number: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO repeating_tasks (task_name, query, interval_hours, target_number)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_name) DO UPDATE SET
                    query=excluded.query,
                    interval_hours=excluded.interval_hours,
                    target_number=excluded.target_number,
                    active=1
                """, (task_name, query, interval_hours, target_number))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_repeating_task: {e}")
            return False

    def get_active_repeating_tasks(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM repeating_tasks WHERE active = 1")
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "task_name": r["task_name"],
                        "query": r["query"],
                        "interval_hours": r["interval_hours"],
                        "last_run": r["last_run"],
                        "target_number": r["target_number"],
                        "active": r["active"]
                      })
                return results
        except Exception as e:
            print(f"Database error in get_active_repeating_tasks: {e}")
            return []

    def update_task_last_run(self, task_id: int, last_run_time: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE repeating_tasks SET last_run = ? WHERE id = ?",
                    (last_run_time, task_id)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in update_task_last_run: {e}")
            return False

    # --- Smart Trigger Helpers ---
    def save_smart_trigger(self, sender: str, trigger_type: str, name: str, **kwargs) -> int:
        """Save a smart trigger. Returns the new trigger ID."""
        import datetime
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check for existing trigger with same name and sender, deactivate it or replace
                cursor.execute("""
                INSERT INTO smart_triggers 
                (sender, trigger_type, name, metric, threshold_value, threshold_direction,
                 threshold_pct, last_known_value, watch_query, watch_keywords,
                 schedule_cron, interval_hours, message_template, last_run, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)
                """, (
                    sender, trigger_type, name,
                    kwargs.get("metric"), kwargs.get("threshold_value"), kwargs.get("threshold_direction"),
                    kwargs.get("threshold_pct"), kwargs.get("last_known_value"),
                    kwargs.get("watch_query"), json.dumps(kwargs.get("watch_keywords", [])) if kwargs.get("watch_keywords") is not None else None,
                    kwargs.get("schedule_cron"), kwargs.get("interval_hours"),
                    kwargs.get("message_template"),
                    datetime.datetime.utcnow().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Database error in save_smart_trigger: {e}")
            return -1

    def get_active_smart_triggers(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM smart_triggers WHERE active = 1")
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "sender": r["sender"],
                        "trigger_type": r["trigger_type"],
                        "name": r["name"],
                        "metric": r["metric"],
                        "threshold_value": r["threshold_value"],
                        "threshold_direction": r["threshold_direction"],
                        "threshold_pct": r["threshold_pct"],
                        "last_known_value": r["last_known_value"],
                        "watch_query": r["watch_query"],
                        "watch_keywords": r["watch_keywords"],
                        "schedule_cron": r["schedule_cron"],
                        "interval_hours": r["interval_hours"],
                        "message_template": r["message_template"],
                        "last_run": r["last_run"],
                        "last_triggered": r["last_triggered"],
                        "fire_count": r["fire_count"],
                        "active": r["active"],
                        "created_at": r["created_at"],
                        "expires_at": r["expires_at"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_active_smart_triggers: {e}")
            return []

    def update_trigger_state(self, trigger_id: int, last_run: str = None,
                              last_triggered: str = None, last_known_value: float = None,
                              fire_count_increment: int = 0) -> bool:
        try:
            with self._get_connection() as conn:
                updates = []
                params = []
                if last_run is not None:
                    updates.append("last_run = ?")
                    params.append(last_run)
                if last_triggered is not None:
                    updates.append("last_triggered = ?")
                    params.append(last_triggered)
                if last_known_value is not None:
                    updates.append("last_known_value = ?")
                    params.append(last_known_value)
                if fire_count_increment > 0:
                    updates.append("fire_count = fire_count + ?")
                    params.append(fire_count_increment)
                if updates:
                    params.append(trigger_id)
                    conn.execute(f"UPDATE smart_triggers SET {', '.join(updates)} WHERE id = ?", params)
                    conn.commit()
                return True
        except Exception as e:
            print(f"Database error in update_trigger_state: {e}")
            return False

    def pause_trigger(self, trigger_id_or_name: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if str(trigger_id_or_name).isdigit():
                    cursor.execute("UPDATE smart_triggers SET active = 0 WHERE id = ?", (int(trigger_id_or_name),))
                else:
                    cursor.execute("UPDATE smart_triggers SET active = 0 WHERE name = ?", (trigger_id_or_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Database error in pause_trigger: {e}")
            return False

    def resume_trigger(self, trigger_id_or_name: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if str(trigger_id_or_name).isdigit():
                    cursor.execute("UPDATE smart_triggers SET active = 1 WHERE id = ?", (int(trigger_id_or_name),))
                else:
                    cursor.execute("UPDATE smart_triggers SET active = 1 WHERE name = ?", (trigger_id_or_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Database error in resume_trigger: {e}")
            return False

    def delete_trigger_by_id_or_name(self, trigger_id_or_name: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if str(trigger_id_or_name).isdigit():
                    cursor.execute("DELETE FROM smart_triggers WHERE id = ?", (int(trigger_id_or_name),))
                else:
                    cursor.execute("DELETE FROM smart_triggers WHERE name = ?", (trigger_id_or_name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Database error in delete_trigger_by_id_or_name: {e}")
            return False

    def list_smart_triggers(self, sender: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM smart_triggers WHERE sender = ?", (sender,))
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "sender": r["sender"],
                        "trigger_type": r["trigger_type"],
                        "name": r["name"],
                        "metric": r["metric"],
                        "threshold_value": r["threshold_value"],
                        "threshold_direction": r["threshold_direction"],
                        "threshold_pct": r["threshold_pct"],
                        "last_known_value": r["last_known_value"],
                        "watch_query": r["watch_query"],
                        "watch_keywords": r["watch_keywords"],
                        "schedule_cron": r["schedule_cron"],
                        "interval_hours": r["interval_hours"],
                        "message_template": r["message_template"],
                        "last_run": r["last_run"],
                        "last_triggered": r["last_triggered"],
                        "fire_count": r["fire_count"],
                        "active": r["active"],
                        "created_at": r["created_at"],
                        "expires_at": r["expires_at"]
                    })
                return results
        except Exception as e:
            print(f"Database error in list_smart_triggers: {e}")
            return []

    # --- Calendar Event Helpers ---
    def save_calendar_event(self, title: str, description: str, date_time: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO calendar_events (title, description, date_time) VALUES (?, ?, ?)",
                    (title, description, date_time)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_calendar_event: {e}")
            return False

    def get_calendar_events(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM calendar_events ORDER BY date_time ASC")
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "id": r["id"],
                        "title": r["title"],
                        "description": r["description"],
                        "date_time": r["date_time"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_calendar_events: {e}")
            return []

    # --- One-Time Reminder Helpers ---
    def save_reminder(self, sender: str, title: str, message: str, due_at: str) -> bool:
        try:
            import datetime
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO reminders (sender, title, message, due_at, status, created_at)
                    VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (sender, title, message, due_at, datetime.datetime.utcnow().isoformat())
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_reminder: {e}")
            return False

    def get_due_reminders(self, now_iso: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at ASC",
                    (now_iso,)
                )
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"Database error in get_due_reminders: {e}")
            return []

    def list_reminders(self, sender: str = "") -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if sender:
                    cursor.execute(
                        "SELECT * FROM reminders WHERE sender = ? ORDER BY due_at ASC",
                        (sender,)
                    )
                else:
                    cursor.execute("SELECT * FROM reminders ORDER BY due_at ASC")
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"Database error in list_reminders: {e}")
            return []

    def mark_reminder_sent(self, reminder_id: int, sent_at: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE reminders SET status = 'sent', sent_at = ? WHERE id = ?",
                    (sent_at, reminder_id)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in mark_reminder_sent: {e}")
            return False

    # --- Nutrition Helpers ---
    def save_nutrition_log(
        self,
        sender: str,
        food_name: str,
        quantity: str,
        calories: float,
        protein_g: float = 0.0,
        carbs_g: float = 0.0,
        fat_g: float = 0.0,
        logged_at: str = "",
    ) -> bool:
        try:
            import datetime
            timestamp = logged_at or datetime.datetime.utcnow().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO nutrition_logs
                    (sender, food_name, quantity, calories, protein_g, carbs_g, fat_g, logged_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sender, food_name, quantity, calories, protein_g, carbs_g, fat_g, timestamp)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_nutrition_log: {e}")
            return False

    def get_nutrition_logs(self, sender: str, date_prefix: str = "") -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if date_prefix:
                    cursor.execute(
                        """
                        SELECT * FROM nutrition_logs
                        WHERE sender = ? AND logged_at LIKE ?
                        ORDER BY logged_at ASC
                        """,
                        (sender, f"{date_prefix}%")
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM nutrition_logs WHERE sender = ? ORDER BY logged_at DESC LIMIT 50",
                        (sender,)
                    )
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"Database error in get_nutrition_logs: {e}")
            return []

    # --- WhatsApp Chat History Helpers ---
    def save_chat_message(self, sender: str, role: str, message: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO chat_history (sender, role, message, timestamp)
                VALUES (?, ?, ?, ?)
                """, (sender, role, message, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_chat_message: {e}")
            return False

    def get_chat_history(self, sender: str, limit: int = 10) -> List[Dict[str, str]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT role, message FROM chat_history 
                WHERE sender = ? 
                ORDER BY id DESC LIMIT ?
                """, (sender, limit))
                rows = cursor.fetchall()
                # Return in chronological order
                results = []
                for r in reversed(rows):
                    results.append({
                        "role": r["role"],
                        "content": r["message"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_chat_history: {e}")
            return []

    # --- Custom Sub-Agents Helpers ---
    def save_custom_agent(self, name: str, system_prompt: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO custom_agents (name, system_prompt, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    system_prompt=excluded.system_prompt,
                    created_at=excluded.created_at
                """, (name, system_prompt, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_custom_agent: {e}")
            return False

    def get_all_custom_agents(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM custom_agents ORDER BY name ASC")
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "name": r["name"],
                        "system_prompt": r["system_prompt"],
                        "created_at": r["created_at"]
                    })
                return results
        except Exception as e:
            print(f"Database error in get_all_custom_agents: {e}")
            return []

    def delete_custom_agent(self, name: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM custom_agents WHERE name = ?", (name,))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in delete_custom_agent: {e}")
            return False

    # --- Human Approval Helpers ---
    def create_pending_approval(
        self,
        approval_id: str,
        sender: str,
        action_type: str,
        command_line: str,
        target_path: str,
        expires_at: str
    ) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO pending_approvals
                (id, sender, action_type, command_line, target_path, created_at, expires_at, status, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    approval_id,
                    sender,
                    action_type,
                    command_line,
                    target_path,
                    now,
                    expires_at,
                    "pending",
                    ""
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in create_pending_approval: {e}")
            return False

    def get_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "sender": row["sender"],
                    "action_type": row["action_type"],
                    "command_line": row["command_line"],
                    "target_path": row["target_path"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "status": row["status"],
                    "result": row["result"]
                }
        except Exception as e:
            print(f"Database error in get_pending_approval: {e}")
            return None

    def update_pending_approval(self, approval_id: str, status: str, result: str = "") -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE pending_approvals SET status = ?, result = ? WHERE id = ?",
                    (status, result, approval_id)
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in update_pending_approval: {e}")
            return False

    # --- User Profile Helper Methods ---
    def save_user_profile(self, sender: str, profile: dict) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO user_profiles (sender, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(sender) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at
                """, (sender, json.dumps(profile), now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_user_profile: {e}")
            return False

    def get_user_profile(self, sender: str) -> Optional[dict]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT profile_json FROM user_profiles WHERE sender = ?", (sender,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row["profile_json"])
                return None
        except Exception as e:
            print(f"Database error in get_user_profile: {e}")
            return None

    # --- Preference Signal Helper Methods ---
    def log_preference_signal(self, sender: str, signal_type: str, value: str, weight: float = 1.0) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO preference_signals (sender, signal_type, value, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (sender, signal_type, value, weight, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in log_preference_signal: {e}")
            return False

    def get_preference_signals(self, sender: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT * FROM preference_signals WHERE sender = ? ORDER BY created_at DESC LIMIT ?
                """, (sender, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Database error in get_preference_signals: {e}")
            return []

    # --- Memory Archive Helper Methods ---
    def archive_memory(self, memory_id: str, memory_type: str, content: str, created_at: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO memory_archive (id, type, content, created_at, archived_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """, (memory_id, memory_type, content, created_at, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in archive_memory: {e}")
            return False

    # --- Associative Memories Helper Methods ---
    def save_assoc_memory(self, id: str, entity_a: str, relation: str, entity_b: str, strength: float = 1.0) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO assoc_memories (id, entity_a, relation, entity_b, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    strength=excluded.strength,
                    created_at=excluded.created_at
                """, (id, entity_a, relation, entity_b, strength, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in save_assoc_memory: {e}")
            return False

    # --- Request Metrics Helper Methods ---
    def log_request_metrics(self, sender: str, message_length: int, response_time: float, success: bool, error_message: str = "") -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                INSERT INTO request_logs (sender, timestamp, message_length, response_time, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (sender, now, message_length, response_time, 1 if success else 0, error_message))
                conn.commit()
                return True
        except Exception as e:
            print(f"Database error in log_request_metrics: {e}")
            return False

# Instantiate a global database instance
db = Database()
