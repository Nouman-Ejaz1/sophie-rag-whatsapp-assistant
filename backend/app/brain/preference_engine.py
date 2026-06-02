import datetime
from typing import Dict, Any, List
from app.database import db

class PreferenceEngine:
    """
    Analyzes user messages for style, language, length, and topic preferences.
    Saves behavioral signals to SQLite and compiles actionable style hints for the agent.
    """
    def __init__(self):
        # Common Roman Urdu stopwords/keywords to detect Urdu preference
        self.roman_urdu_keywords = {
            "kia", "kya", "hai", "aur", "ko", "haan", "acha", "jee", "ji", "shukriya", 
            "thik", "theek", "bhai", "yaar", "کریں", "کیا", "ہے", "اور", "شکریہ", "کیسے"
        }

    def analyze_message(self, sender: str, message: str) -> None:
        """
        Runs heuristics on the user message and logs preference signals to SQLite.
        """
        if not message or not sender:
            return

        msg_lower = message.lower()
        msg_len = len(message)

        # 1. Length Signal (Conciseness preference)
        if msg_len < 20:
            db.log_preference_signal(sender, "length_preference", "concise", weight=0.5)
        elif msg_len > 300:
            db.log_preference_signal(sender, "length_preference", "detailed", weight=0.8)

        # 2. Language Signal (Urdu / Roman Urdu detection)
        words = set(msg_lower.split())
        urdu_word_count = len(words.intersection(self.roman_urdu_keywords))
        
        # Also check for actual Arabic/Urdu script characters
        has_arabic_script = any('\u0600' <= char <= '\u06FF' for char in message)
        
        if urdu_word_count >= 2 or has_arabic_script:
            db.log_preference_signal(sender, "language_preference", "Urdu", weight=1.0)
        else:
            db.log_preference_signal(sender, "language_preference", "English", weight=0.2)

        # 3. Topic Interest Signals
        topics = {
            "finance": ["stock", "price", "market", "finance", "crypto", "bitcoin", "rate", "usd", "pkr"],
            "weather": ["weather", "rain", "temperature", "temp", "forecast", "sun", "cloud"],
            "code": ["code", "python", "javascript", "error", "bug", "run", "execute", "api", "compile"],
            "nutrition": ["calorie", "diet", "protein", "carbs", "food", "eat", "weight", "fat", "meal"]
        }

        for topic, keywords in topics.items():
            for kw in keywords:
                if kw in msg_lower:
                    db.log_preference_signal(sender, "topic_interest", topic, weight=0.6)
                    break

        # 4. Interactive Question Followup Signal
        if message.endswith("?"):
            db.log_preference_signal(sender, "engagement_style", "questioning", weight=0.4)

    def get_style_hints(self, sender: str) -> Dict[str, Any]:
        """
        Aggregates recent signals to compile current style hints/preferences for the user.
        """
        signals = db.get_preference_signals(sender, limit=50)
        if not signals:
            return {
                "preferred_language": "English",
                "conciseness": "balanced",
                "top_interests": [],
                "engagement_style": "default"
            }

        # Aggregate counts and weights
        lang_scores = {"English": 0.0, "Urdu": 0.0}
        length_scores = {"concise": 0.0, "detailed": 0.0, "balanced": 0.0}
        topic_scores = {}
        style_scores = {"default": 0.0, "questioning": 0.0}

        for sig in signals:
            sig_type = sig["signal_type"]
            val = sig["value"]
            w = sig["weight"]

            if sig_type == "language_preference":
                lang_scores[val] = lang_scores.get(val, 0.0) + w
            elif sig_type == "length_preference":
                length_scores[val] = length_scores.get(val, 0.0) + w
            elif sig_type == "topic_interest":
                topic_scores[val] = topic_scores.get(val, 0.0) + w
            elif sig_type == "engagement_style":
                style_scores[val] = style_scores.get(val, 0.0) + w

        # Determine dominant preferences
        pref_lang = max(lang_scores, key=lang_scores.get) if any(lang_scores.values()) else "English"
        
        # Handle conciseness
        concise_val = length_scores.get("concise", 0.0)
        detailed_val = length_scores.get("detailed", 0.0)
        if concise_val > detailed_val + 1.0:
            pref_length = "concise"
        elif detailed_val > concise_val + 1.0:
            pref_length = "detailed"
        else:
            pref_length = "balanced"

        # Sort topics by interest weight
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        top_interests = [t[0] for t in sorted_topics[:3]]

        # Determine dominant engagement style
        pref_style = max(style_scores, key=style_scores.get) if any(style_scores.values()) else "default"

        return {
            "preferred_language": pref_lang,
            "conciseness": pref_length,
            "top_interests": top_interests,
            "engagement_style": pref_style
        }

preference_engine = PreferenceEngine()
