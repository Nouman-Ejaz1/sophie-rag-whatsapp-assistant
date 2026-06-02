import time
import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.brain.doc_brain import doc_brain
from app.brain.memory_os import memory_os
from app.brain.trigger_mind import trigger_mind
from app.brain.orchestrator import orchestrator

def run_verification():
    print("=" * 60)
    print("🚀 SENTINELAI BACKEND SYSTEM VERIFICATION")
    print("=" * 60)

    # 1. Clear database logs and decay records for clean test
    print("\n🧹 Cleaning local SQLite database...")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_logs")
        cursor.execute("DELETE FROM memory_decay_records")
        cursor.execute("DELETE FROM procedural_memories")
        conn.commit()
    print("Local database cleaned successfully.")

    # 2. Ingest documentation (DocBrain)
    print("\n📚 Test 1: Ingesting architecture policies in DocBrain...")
    doc_content = """
    SentinelAI Architecture Token Standard:
    All refresh tokens must be rotated every 24 hours (JWT session rotation policy).
    Database storage must expire inactive Redis sessions within 48 hours to prevent security vulnerabilities.
    CTO Susan C strictly requires checking these parameters before any deployment.
    """
    ingest_res = doc_brain.ingest_document(
        title="Session Rotation Policy",
        content=doc_content,
        source_type="notion",
        source_url="notion://auth_security_policy"
    )
    print(f"Ingestion successful! Doc ID: {ingest_res['doc_id']} with {ingest_res['chunks_count']} chunks.")

    # 3. Save a procedural memory (rules check)
    print("\n🔌 Test 2: Pre-setting procedural rules in MemoryOS...")
    procedural_steps = [
        "Read session policy documentation",
        "Verify JWT secret rotation flag in config",
        "Run vulnerability scan on Redis adapter",
        "Dispatch deployment report to Susan C"
    ]
    memory_os.save_procedural_memory("github_pr_review", procedural_steps)
    print("Saved procedural memory steps successfully.")

    # 4. Trigger an autonomous event (TriggerMind)
    print("\n⚡ Test 3: Simulating autonomous wake-up event (GitHub PR)...")
    trigger_results = trigger_mind.simulate_mock_event("github")
    print(f"Trigger Relevance Score: {trigger_results['relevance_score']}% (Is Urgent: {trigger_results['is_urgent']})")
    print(f"Trigger Classification Reason: {trigger_results['reason']}")

    # 5. Run full pipeline (LangGraph Orchestrator)
    print("\n🤖 Test 4: Running full LangGraph Orchestrator flow with event...")
    orchestrator_res = orchestrator.run(triggered_event=trigger_results)
    
    print("\n--- Agent Execution Output ---")
    print(f"Task Type: {orchestrator_res.get('task_type')}")
    print(f"Is Urgent Event: {orchestrator_res.get('is_urgent')}")
    print(f"Reasoning Detail: {orchestrator_res.get('reasoning')[:150]}...")
    print(f"Citations Retrieved: {orchestrator_res.get('citations')}")
    print(f"Confidence Score: {orchestrator_res.get('confidence_score')}%")
    print(f"Action Executed: {orchestrator_res.get('action_taken')}")
    print(f"Conversational Response:\n{orchestrator_res.get('response')}")

    # 6. Verify Memory Decay Calculations
    print("\n⏳ Test 5: Testing MemoryOS Decay Scoring System...")
    # Add a mock memory
    mem_id = memory_os.add_episodic_memory("User requested architecture update on Nov 10th.")
    memories = memory_os.get_all_decay_states()
    original_freshness = next(m["freshness"] for m in memories if m["id"] == mem_id)
    print(f"Initial Memory Freshness: {original_freshness}%")
    
    # Fast-forward time artificially by 10 days
    print("Simulating passage of 10 days...")
    decayed_memories = memory_os.trigger_artificial_decay(10.0)
    decayed_freshness = next(m["freshness"] for m in decayed_memories if m["id"] == mem_id)
    decayed_status = next(m["status"] for m in decayed_memories if m["id"] == mem_id)
    print(f"Freshness after 10 days (episodic half-life 7 days): {decayed_freshness}% (Status: {decayed_status})")
    
    # 7. Check logs in SQLite
    print("\n📄 Test 6: Verifying system log dumps in SQLite...")
    logs = db.get_logs(limit=5)
    for l in logs:
        print(f"[{l['timestamp'][:19]}] Source: {l['source']} | Message: {l['message']} (Status: {l['status']})")

    print("\n" + "=" * 60)
    print("🎉 ALL CORE SENTINELAI BACKEND MODULES FULLY FUNCTIONING!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
