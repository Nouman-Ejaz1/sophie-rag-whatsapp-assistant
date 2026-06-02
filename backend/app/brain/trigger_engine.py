"""
Sophie Smart Trigger Engine.
Runs inside the scheduler loop. Evaluates triggers without burning LLM calls
for every check — uses lightweight Python logic and only calls LLM to compose
the final WhatsApp alert message when required, or uses direct formatting.
"""
import json
import re
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import requests

from app.database import db


class TriggerEvaluator:
    """
    Evaluates whether a trigger condition has been met.
    NO LLM calls here — pure Python logic.
    """

    def should_check_now(self, trigger: dict) -> bool:
        """Returns True if enough time has passed to check this trigger again."""
        trigger_type = trigger.get("trigger_type")
        last_run = trigger.get("last_run")
        
        # How often to check each type
        check_intervals = {
            "threshold": 0.5,    # Every 30 minutes
            "keyword": 1.0,      # Every 1 hour
            "recurring": float(trigger.get("interval_hours") or 24),
            "one_time": 0.05,    # Every 3 minutes (high precision for reminders)
        }
        
        interval_hours = check_intervals.get(trigger_type, 1.0)
        
        if not last_run:
            return True
        
        try:
            last_dt = datetime.fromisoformat(last_run)
            elapsed_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
            return elapsed_hours >= interval_hours
        except Exception:
            return True

    def evaluate_threshold(self, trigger: dict) -> Tuple[bool, Optional[str]]:
        """
        Check if a numeric threshold has been crossed.
        Returns (triggered, message) or (False, None).
        """
        metric = trigger.get("metric", "")
        threshold = trigger.get("threshold_value")
        direction = trigger.get("threshold_direction", "above")
        last_known = trigger.get("last_known_value")
        
        # Fetch current value
        current_value = self._fetch_metric_value(metric)
        if current_value is None:
            return False, None
        
        # Update last known value in DB (always)
        db.update_trigger_state(trigger["id"], last_known_value=current_value)
        
        # Check threshold condition
        triggered = False
        if direction == "above" and current_value >= threshold:
            # Only alert if it crossed (wasn't already above)
            if last_known is None or last_known < threshold:
                triggered = True
        elif direction == "below" and current_value <= threshold:
            if last_known is None or last_known > threshold:
                triggered = True
        elif direction == "change_by_percent":
            pct = trigger.get("threshold_pct", 5.0)
            if last_known and abs((current_value - last_known) / last_known * 100) >= pct:
                triggered = True
        
        if triggered:
            # Format the alert message
            name = trigger.get("name", metric)
            direction_label = "above" if direction == "above" else "below" if direction == "below" else f"changed by {trigger.get('threshold_pct', 5.0)}%"
            msg = (
                f"*🔔 Trigger Alert: {name}*\n"
                f"Current value: *{current_value:,.2f}*\n"
                f"Your threshold: {direction_label} {threshold:,.2f}\n"
                f"Previous value: {last_known:,.2f if last_known is not None else 'N/A'}"
            )
            return True, msg
        
        return False, None

    def _fetch_metric_value(self, metric: str) -> Optional[float]:
        """
        Fetches a numeric metric value without LLM.
        Supports: usd_pkr, btc_usd, petrol_pkr, eth_usd, and custom.
        """
        metric = (metric or "").lower().strip()
        try:
            # Currency Rates
            if metric in {"usd_pkr", "eur_pkr", "gbp_pkr", "usd_eur"}:
                base, quote = metric.upper().split("_")
                # Direct lookup
                resp = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    rates = data.get("rates", {})
                    if quote in rates:
                        return float(rates[quote])
            
            # Crypto Rates
            elif metric in {"btc_usd", "eth_usd", "bnb_usd"}:
                coin_map = {"btc_usd": "bitcoin", "eth_usd": "ethereum", "bnb_usd": "binancecoin"}
                coin_id = coin_map.get(metric, "bitcoin")
                resp = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": coin_id, "vs_currencies": "usd"},
                    timeout=8
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return float(data.get(coin_id, {}).get("usd", 0))
            
            # Petrol PKR
            elif metric == "petrol_pkr":
                from app.brain.research_engine import research_query
                result = research_query("petrol rate in pakistan today", mode="latest")
                claims = result.get("claims", [])
                for claim in claims:
                    val_str = str(claim.get("value", "")).lower()
                    # Find numbers like Rs. 270 or 272.50
                    match = re.search(r"rs\.?\s*([\d.]+)", val_str)
                    if match:
                        return float(match.group(1))
                    match = re.search(r"([\d.]+)\s*rupees", val_str)
                    if match:
                        return float(match.group(1))
                    match = re.search(r"\b(\d{3}(?:\.\d+)?)\b", val_str)
                    if match:
                        return float(match.group(1))
            
            # Fallback custom metric using search
            else:
                from app.brain.research_engine import research_query
                result = research_query(f"current price of {metric}", mode="latest")
                claims = result.get("claims", [])
                for claim in claims:
                    val_str = str(claim.get("value", ""))
                    match = re.search(r"[\d.,]+", val_str)
                    if match:
                        return float(match.group().replace(",", ""))
        
        except Exception as e:
            print(f"[TriggerEngine] _fetch_metric_value failed for {metric}: {e}")
        
        return None

    def evaluate_keyword_monitor(self, trigger: dict) -> Tuple[bool, Optional[str]]:
        """
        Checks if new content matching watched keywords has appeared.
        Uses search + keyword matching — no LLM.
        """
        query = trigger.get("watch_query", "")
        keywords_json = trigger.get("watch_keywords")
        last_triggered = trigger.get("last_triggered")
        
        try:
            if isinstance(keywords_json, str) and keywords_json:
                keywords = json.loads(keywords_json)
            elif isinstance(keywords_json, list):
                keywords = keywords_json
            else:
                keywords = []
        except Exception:
            keywords = []
        
        if not query:
            return False, None
        
        try:
            from app.brain.research_engine import research_query
            result = research_query(query, mode="latest")
            
            if not result.get("answerable") or result.get("trust_score", 0) < 40:
                return False, None
            
            freshness = result.get("freshness", "")
            claims = result.get("claims", [])
            
            # Check if the result is actually new since last trigger
            if last_triggered and freshness:
                try:
                    last_dt = datetime.fromisoformat(last_triggered)
                    result_dt = datetime.strptime(freshness, "%Y-%m-%d")
                    if result_dt.date() <= last_dt.date():
                        return False, None  # Nothing new since last alert
                except Exception:
                    pass
            
            # Check if keywords match
            if keywords:
                combined_text = " ".join([
                    str(c.get("value", "")) + " " + " ".join(c.get("details", []))
                    for c in claims
                ]).lower()
                if not any(kw.lower() in combined_text for kw in keywords):
                    return False, None
            
            # Build alert message from claims
            name = trigger.get("name", query)
            top_claim = claims[0] if claims else {}
            headline = str(top_claim.get("value", "New information found"))[:200]
            sources = top_claim.get("source_titles", [])
            source_text = f"\n_Sources: {', '.join(sources[:2])}_" if sources else ""
            
            msg = (
                f"*📡 Monitor Alert: {name}*\n\n"
                f"{headline}{source_text}\n\n"
                f"_Query: {query}_"
            )
            return True, msg
        
        except Exception as e:
            print(f"[TriggerEngine] evaluate_keyword_monitor failed: {e}")
            return False, None

    def evaluate_recurring(self, trigger: dict) -> Tuple[bool, Optional[str]]:
        """
        For recurring digest tasks — checks if interval has passed.
        Returns True when it's time to run the query and send a summary.
        """
        interval_hours = float(trigger.get("interval_hours") or 24)
        last_run = trigger.get("last_run")
        
        if not last_run:
            return True, None
        
        try:
            elapsed = (datetime.utcnow() - datetime.fromisoformat(last_run)).total_seconds() / 3600
            return elapsed >= interval_hours, None
        except Exception:
            return True, None


class TriggerEngine:
    """
    Main engine — called by the scheduler loop in main.py.
    Replaces/extends the repeating_tasks logic.
    """
    
    def __init__(self):
        self.evaluator = TriggerEvaluator()
    
    def run_cycle(self):
        """
        Called periodically by the scheduler.
        Checks all active triggers and fires WhatsApp messages when conditions are met.
        """
        now_iso = datetime.utcnow().isoformat()
        triggers = db.get_active_smart_triggers()
        
        for trigger in triggers:
            try:
                if not self.evaluator.should_check_now(trigger):
                    continue
                
                triggered = False
                message = None
                trigger_type = trigger.get("trigger_type")
                
                if trigger_type == "threshold":
                    triggered, message = self.evaluator.evaluate_threshold(trigger)
                elif trigger_type == "keyword":
                    triggered, message = self.evaluator.evaluate_keyword_monitor(trigger)
                elif trigger_type == "recurring":
                    triggered, _ = self.evaluator.evaluate_recurring(trigger)
                    if triggered:
                        message = self._compose_recurring_summary(trigger)
                
                # Update last_run regardless (so we don't re-check too soon)
                db.update_trigger_state(trigger["id"], last_run=now_iso)
                
                if triggered and message:
                    self._send_whatsapp_alert(
                        to=trigger["sender"],
                        message=message,
                        trigger_id=trigger["id"]
                    )
                    db.update_trigger_state(
                        trigger["id"],
                        last_triggered=now_iso,
                        fire_count_increment=1
                    )
                    print(f"[TriggerEngine] Fired trigger #{trigger['id']} ({trigger.get('name')}) -> {trigger['sender']}")
            
            except Exception as e:
                print(f"[TriggerEngine] Error evaluating trigger #{trigger.get('id')}: {e}")
    
    def _compose_recurring_summary(self, trigger: dict) -> str:
        """
        For recurring digests: run the search query and compose a summary.
        """
        query = trigger.get("watch_query") or trigger.get("name", "news update")
        
        try:
            from app.brain.research_engine import research_query
            result = research_query(query, mode="latest")
            claims = result.get("claims", [])
            
            if not claims:
                return f"*📰 Scheduled Update: {trigger.get('name', query)}*\n\nNo new information found for this search."
            
            # Compose WhatsApp summary WITHOUT LLM (fast, no API call)
            name = trigger.get("name", query)
            lines = [f"*📰 Scheduled: {name}*\n"]
            
            for i, claim in enumerate(claims[:3], 1):
                value = str(claim.get("value", ""))[:150]
                sources = claim.get("source_titles", [])
                source_tag = f" _{sources[0]}_" if sources else ""
                lines.append(f"{i}. {value}{source_tag}")
            
            lines.append(f"\n_Updated: {result.get('freshness', 'today')}_")
            return "\n".join(lines)
        
        except Exception as e:
            return f"*Scheduled Update: {trigger.get('name')}*\nCould not fetch update: {e}"
    
    def _send_whatsapp_alert(self, to: str, message: str, trigger_id: int):
        """Send WhatsApp message via the local gateway."""
        try:
            resp = requests.post(
                "http://127.0.0.1:3001/send",
                json={"to": to, "message": message},
                timeout=10
            )
            if resp.status_code != 200:
                print(f"[TriggerEngine] Gateway returned {resp.status_code} for trigger #{trigger_id}")
        except Exception as e:
            print(f"[TriggerEngine] Could not send alert for trigger #{trigger_id}: {e}")


trigger_engine = TriggerEngine()
