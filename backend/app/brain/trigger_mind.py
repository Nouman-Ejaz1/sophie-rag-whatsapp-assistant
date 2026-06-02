import json
import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from app.config import GEMINI_API_KEY
from app.database import db

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class TriggerMind:
    def __init__(self):
        # We use gemini-flash-lite-latest for speed and efficiency
        self.model = genai.GenerativeModel("models/gemini-flash-lite-latest")

    def classify_event(self, source: str, title: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Uses gemini-2.5-flash to assess whether an incoming real-world event 
        is urgent, calculate its relevance score (0-100), and decide on actions.
        """
        metadata = metadata or {}
        now = datetime.datetime.utcnow().isoformat()
        
        prompt = f"""
You are the TriggerMind classification nervous system of SentinelAI, an autonomous agent.
Analyze the following incoming event from {source} and classify its relevance and urgency.

Event Title: {title}
Event Description: {description}
Event Metadata: {json.dumps(metadata)}

Return a strict JSON object with the following fields:
1. "is_urgent": boolean (true if this demands immediate wake-up and action from the agent)
2. "relevance_score": number (0 to 100 representing how important this is to the user/organization)
3. "reason": string (brief explanation of your evaluation)
4. "action_type": string ("chat_alert", "email_dispatch", "team_notification", "ignore")
5. "suggested_task_type": string ("github_pr_review", "slack_keyword_alert", "news_briefing", "meeting_prep", "unknown")

Be precise. Events that are generic should have low scores. Events mentioning security, PR merges, calendar meetings with VIPs, or keyword alerts should have high urgency/scores.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            classification = json.loads(response.text.strip())
            
            # Add timestamps and base info
            classification["timestamp"] = now
            classification["source"] = source
            classification["title"] = title
            classification["description"] = description
            
            # Log into database for the UI feed
            status = "alert" if classification.get("is_urgent", False) else "info"
            db.log_event(
                source=source,
                message=f"Event triggered: {title}",
                status=status,
                meta_dict=classification
            )
            
            return classification
            
        except Exception as e:
            print(f"Failed to classify event using Gemini: {e}")
            # Safe fallback classification
            fallback = {
                "is_urgent": False,
                "relevance_score": 50.0,
                "reason": f"Fallback classification due to processing error: {e}",
                "action_type": "chat_alert",
                "suggested_task_type": "unknown",
                "timestamp": now,
                "source": source,
                "title": title,
                "description": description
            }
            db.log_event(
                source=source,
                message=f"Event triggered (Fallback): {title}",
                status="info",
                meta_dict=fallback
            )
            return fallback

    def simulate_mock_event(self, source: str) -> Dict[str, Any]:
        """Generates pre-defined, high-fidelity mock events representing typical corporate developer workflows."""
        events = {
            "github": {
                "title": "PR #402 Merged: Implement JWT Session Rotation",
                "description": "Developer 'alex_dev' merged code into 'main'. Changes touch 'auth.py' and modify database schemas to rotate refresh tokens every 24 hours.",
                "metadata": {"repo": "sentinel-core", "branch": "main", "author": "alex_dev"}
            },
            "slack": {
                "title": "Slack Keyword Mention: 'security_vulnerability' in #general",
                "description": "User '@jane_sec' posted: 'We need to double-check our token rotators. There might be a security_vulnerability in the redis token backend if we fail to expire logs.'",
                "metadata": {"channel": "general", "user": "jane_sec", "channel_id": "C992831"}
            },
            "calendar": {
                "title": "Google Calendar: Technical Architecture Alignment Meeting",
                "description": "Upcoming meeting in 15 minutes with CTO 'susan_c' regarding 'Authentication and Storage Architecture for next quarter.'",
                "metadata": {"attendees": ["susan_c", "alex_dev", "user_admin"], "duration_minutes": 30}
            },
            "news": {
                "title": "TechNews Alert: Pinecone launches high-speed Serverless namespaces",
                "description": "Pinecone announces multi-region AWS support for instant namespace retrieval. Developers report latency dropping by 40% when segmenting embeddings.",
                "metadata": {"category": "tech_announcement", "source_url": "https://technews.com/pinecone-namespaces"}
            }
        }
        
        target = events.get(source.lower())
        if not target:
            raise ValueError(f"Unknown mock event source: {source}")
            
        return self.classify_event(
            source=source,
            title=target["title"],
            description=target["description"],
            metadata=target["metadata"]
        )

# Global trigger mind instance
trigger_mind = TriggerMind()
