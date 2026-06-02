import json
import re
import time
import requests
from typing import Dict, Any, List, Optional

from app.database import db
from app.brain.memory_os import memory_os
from app.brain.response_cache import response_cache
from app.brain.system_prompt import build_stage1_planner_prompt

def update_whatsapp_status(sender: str, text: str):
    """Posts progress updates to http://localhost:3001/status."""
    if not sender or sender == "unknown_number":
        return
    try:
        url = "http://127.0.0.1:3001/status"
        payload = {"sender": sender, "text": text}
        response = requests.post(url, json=payload, timeout=2.0)
        response.raise_for_status()
    except Exception as e:
        print(f"[ThinkingPalace] Failed to update WhatsApp status for {sender}: {e}")

class LoopState:
    """Tracks executed tools in a single session to prevent infinite tool loops."""
    def __init__(self):
        self.executed_calls = {}

    def is_duplicate(self, action: str, payload: dict) -> bool:
        """Backward compatibility for tests."""
        status = self.check_duplicate_status(action, payload)
        if status == "ok":
            self.record_execution(action, payload, "test_execution")
            return False
        return True

    def record_execution(self, action: str, payload: dict, response: str):
        """Records a successful tool execution to remember its response."""
        payload_str = json.dumps(payload, sort_keys=True)
        call_hash = f"{action}:{payload_str}"
        if call_hash not in self.executed_calls:
            self.executed_calls[call_hash] = {"count": 1, "response": response}
        else:
            self.executed_calls[call_hash]["count"] += 1
            self.executed_calls[call_hash]["response"] = response

    def check_duplicate_status(self, action: str, payload: dict) -> str:
        """
        Checks if the call is a duplicate.
        Returns:
            "ok": Not a duplicate (first execution).
            "warn": First duplicate (second execution). We should inject a warning response.
            "abort": Second duplicate (third execution). We must abort.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        call_hash = f"{action}:{payload_str}"
        if call_hash not in self.executed_calls:
            return "ok"
        
        info = self.executed_calls[call_hash]
        if info["count"] == 1:
            return "warn"
        else:
            return "abort"


class ThinkingPalace:
    """
    JARVIS Mind Engine.
    Implements a 3-stage reasoning engine:
    - Stage 0: Instant greeting short-circuit filter.
    - Stage 1: Fast intent classification (cheap OpenRouter model).
    - Stage 2: Recursive Deep thinking council loop with sub-agent validation.
    - Stage 3: Executor with loop safety (LoopState) & strict iteration limits.
    """
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def _is_trivial(self, message: str) -> bool:
        """Stage 0 Instant Filter: Detects if the message is a lightweight greeting/ack under 24 chars."""
        q = re.sub(r"[^a-z0-9\s]", "", (message or "").lower()).strip()
        if not q:
            return True
        greetings = {
            "hi", "hello", "hey", "yo", "aoa", "salam", "assalamualaikum",
            "assalam o alaikum", "ok", "okay", "thanks", "thank you", "kesi ho",
            "kaisi ho", "how are you"
        }
        return len(q) <= 24 and (q in greetings or q.startswith(("hi ", "hello ", "hey ")))

    def think(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point. Runs the reasoning pipeline."""
        sender = context.get("sender") or "unknown_number"
        
        # Stage 0: Instant Filter
        if self._is_trivial(message):
            print(f"[ThinkingPalace] Stage 0 short-circuit hit for sender={sender}.")
            return {
                "route": "instant_greetings",
                "reasoning": "Instant greeting handled via short circuit under 100ms.",
                "response": "<ouput><user_output>Hi! I'm JARVIS. How can I help you today?</user_output><tools_call>None</tools_call></ouput>",
                "confidence_score": 100.0,
                "citations": []
            }

        # Stage 1: Intent classification using fast model candidates
        update_whatsapp_status(sender, "🤖 *Sophie is analyzing your request...*")
        intent_info = self._classify_intent(message, context)
        print(f"[ThinkingPalace] Stage 1 Intent parsed: {intent_info}")

        # Stage 2 & 3: Deep thinking loop
        return self._deep_think(message, context, intent_info)

    def _classify_intent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Intent engine. Small LLM call or fallback regex parser."""
        from app.brain.tools import openrouter_chat
        
        system_prompt = (
            "You are the JARVIS Intent Classifier.\n"
            "Analyze the user message and context, then output a JSON object describing the intent and requirements.\n"
            "Output JSON strictly with these fields:\n"
            "{\n"
            "  \"intent\": \"string value\",\n"
            "  \"complexity\": \"low\" | \"medium\" | \"high\",\n"
            "  \"needs_tools\": true | false,\n"
            "  \"needs_memory\": true | false,\n"
            "  \"risk_level\": \"low\" | \"medium\" | \"high\",\n"
            "  \"confidence\": 0-100,\n"
            "  \"suggested_plan\": [\"step 1\", \"step 2\"]\n"
            "}\n"
            "Do not include markdown wrappers, HTML, or explanations outside the JSON."
        )
        
        user_prompt = f"User Message: {message}\nContext: {str(context.get('retrieved_memories', []))}"
        
        try:
            raw_response = openrouter_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                user_message="intent_classification",
                requires_tools=False
            )
            # Find JSON boundaries
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_response[start:end])
                return data
        except Exception as e:
            print(f"[ThinkingPalace] Stage 1 intent LLM call failed: {e}. Falling back to heuristics.")

        # Fallback to local heuristic intents
        msg_lower = message.lower()
        needs_tools = any(kw in msg_lower for kw in ["weather", "search", "exchange", "usd", "pkr", "rate", "command", "run"])
        complexity = "medium" if needs_tools else "low"
        return {
            "intent": "general_chat",
            "complexity": complexity,
            "needs_tools": needs_tools,
            "needs_memory": True,
            "risk_level": "low",
            "confidence": 80.0,
            "suggested_plan": ["Run deep-thought loop to resolve query"]
        }

    def _deep_think(self, message: str, context: Dict[str, Any], intent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2 & 3: Deep thinking recursive tool-council loop with LoopState safety."""
        from app.brain.tools import (
            openrouter_chat,
            sophie_tool,
            validate_tool_call,
            validation_result_to_response,
            evaluate_tool_evidence,
            capability_index_summary,
            select_relevant_manuals,
            _fetch_requested_manuals,
            _parse_council_step,
            _council_system_prompt,
            _council_user_prompt,
            _wrap_whatsapp_response,
            _answer_from_missing_info
        )

        sender = context.get("sender") or "unknown_number"
        retrieved_docs = context.get("retrieved_docs") or []
        retrieved_memories = context.get("retrieved_memories") or []

        # Form context packet for OpenRouter
        docs_context = ""
        for idx, doc in enumerate(retrieved_docs):
            docs_context += f"\n[Doc {idx+1}] Source: {doc['source']} ({doc['source_type']})\nTitle: {doc['title']}\nContent: {doc['text']}\nConfidence Score: {doc['confidence']}%\n"
            
        memories_context = ""
        for idx, mem in enumerate(retrieved_memories):
            memories_context += f"\n[Memory {idx+1}] ID: {mem['id']} ({mem['type']})\nContent: {mem['content']}\nFreshness Score: {mem['freshness']}%\nStatus: {mem['status']}\n"
            
        context_packet = f"DOCUMENTS CONTEXT:\n{docs_context}\n\nMEMORIES CONTEXT:\n{memories_context}"

        current_date_text = time.strftime("%B %d, %Y")
        current_year_text = time.strftime("%Y")

        gateway_index = capability_index_summary()
        manuals_text = select_relevant_manuals(message, limit=8)
        tool_results: List[Dict[str, Any]] = []
        system_prompt = _council_system_prompt(current_date_text, current_year_text)

        loop_state = LoopState()

        for round_index in range(1, self.max_depth + 1):
            update_whatsapp_status(sender, f"🧠 *Sophie is reasoning (Round {round_index}/{self.max_depth})...*")
            user_prompt = _council_user_prompt(
                user_message=message,
                context_packet=context_packet,
                brain_decision=intent_info,
                gateway_index=gateway_index,
                manuals_text=manuals_text,
                tool_results=tool_results,
                round_index=round_index,
            )

            raw_step = openrouter_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                user_message=message,
                requires_tools=False
            )

            step = _parse_council_step(raw_step, round_index)

            fetched_manuals = _fetch_requested_manuals(list(step.get("manual_requests") or []))
            if fetched_manuals:
                manuals_text = f"{manuals_text}\n\n{fetched_manuals}".strip()

            tool_calls = list(step.get("tool_calls") or [])
            if tool_calls:
                idempotent_actions = {
                    "search.latest", "research_latest", "wikipedia", "document_search",
                    "time.current", "chat.history", "memory.recall", "tool_list",
                    "get_tool_manual", "tools.discover", "tools.manual"
                }
                for call in tool_calls[:4]:
                    action = str(call.get("action") or "").strip()
                    payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
                    reason = str(call.get("reason") or "")

                    # First, validate the tool call to get normalized action and payload
                    validation = validate_tool_call(action, json.dumps(payload, ensure_ascii=True))
                    
                    if validation.ok:
                        norm_action = validation.action
                        norm_payload = validation.normalized_payload
                    else:
                        norm_action = action
                        norm_payload = payload

                    # Loop Safety Check
                    status = loop_state.check_duplicate_status(norm_action, norm_payload)
                    
                    # If duplicate call to an idempotent tool, return cached response
                    if norm_action in idempotent_actions:
                        if status != "ok":
                            payload_str = json.dumps(norm_payload, sort_keys=True)
                            call_hash = f"{norm_action}:{payload_str}"
                            prev_info = loop_state.executed_calls.get(call_hash)
                            if prev_info:
                                cached_response = prev_info["response"]
                                print(f"[ThinkingPalace] Loop Safety: Smart duplicate bypass for idempotent tool={norm_action}. Returning cached response.")
                                loop_state.record_execution(norm_action, norm_payload, cached_response)
                                tool_results.append({
                                    "action": norm_action,
                                    "payload": norm_payload,
                                    "reason": reason,
                                    "response": cached_response,
                                    "quality": "cached",
                                })
                                continue

                    if status == "abort":
                        print(f"[ThinkingPalace] Loop Safety triggered (abort): Duplicate call detected! action={norm_action}")
                        return _wrap_whatsapp_response(
                            "I detected a loop in executing tool actions. Aborting to safeguard your system.",
                            confidence=40.0,
                            reasoning=f"Loop state triggered on duplicate call of action '{norm_action}'."
                        )
                    elif status == "warn":
                        print(f"[ThinkingPalace] Loop Safety triggered (warning): Duplicate call detected! action={norm_action}")
                        # Record warning execution to increment run count so next time it aborts
                        loop_state.record_execution(norm_action, norm_payload, "duplicate_warning_issued")
                        
                        # Fetch the previous response to inject into warning
                        payload_str = json.dumps(norm_payload, sort_keys=True)
                        call_hash = f"{norm_action}:{payload_str}"
                        prev_info = loop_state.executed_calls.get(call_hash)
                        prev_res = prev_info["response"] if prev_info else "No response found."
                        
                        warning_text = (
                            f"Warning: You already executed this exact tool '{norm_action}' with these identical parameters. "
                            f"Previous response was:\n{prev_res}\n\n"
                            "To prevent infinite loops, you cannot run the exact same call again. "
                            "Please refine your search query, use different parameters, or choose a different tool."
                        )
                        tool_results.append({
                            "action": norm_action,
                            "payload": norm_payload,
                            "reason": reason,
                            "response": warning_text,
                            "quality": "duplicate_warning",
                        })
                        continue

                    if not validation.ok:
                        validation_error = validation_result_to_response(validation)
                        # Record invalid execution to prevent loop state bypasses
                        loop_state.record_execution(norm_action, norm_payload, validation_error)
                        tool_results.append({
                            "action": validation.action,
                            "payload": payload,
                            "reason": reason,
                            "validation_error": validation_error,
                            "quality": "invalid",
                        })
                        continue

                    # Update WhatsApp status about executing the tool
                    if norm_action in ["search.latest", "research_latest"]:
                        query = norm_payload.get("query") or norm_payload.get("original_query") or "latest updates"
                        status_text = f"🔍 *Sophie is searching:* \"{query}\"..."
                    elif norm_action == "wikipedia":
                        query = norm_payload.get("query") or "wikipedia articles"
                        status_text = f"🔍 *Sophie is searching Wikipedia for:* \"{query}\"..."
                    elif norm_action == "document_search":
                        query = norm_payload.get("query") or "knowledge base"
                        status_text = f"🔍 *Sophie is searching knowledge base for:* \"{query}\"..."
                    else:
                        status_text = f"⚙️ *Sophie is preparing tool:* {norm_action}..."
                    update_whatsapp_status(sender, status_text)

                    response = sophie_tool(validation.action, json.dumps(validation.normalized_payload, ensure_ascii=True))
                    update_whatsapp_status(sender, "⚙️ *Sophie is analyzing results...*")
                    quality = evaluate_tool_evidence(validation.action, response)

                    # Record successful execution for loop tracking
                    loop_state.record_execution(validation.action, validation.normalized_payload, response)

                    tool_results.append({
                        "action": validation.action,
                        "payload": validation.normalized_payload,
                        "reason": reason,
                        "response": response,
                        "quality": quality.get("quality", "unknown"),
                    })
                continue

            missing_info = list(step.get("missing_info") or [])
            if missing_info:
                return _wrap_whatsapp_response(
                    _answer_from_missing_info(missing_info),
                    confidence=75.0,
                    reasoning="Council identified required missing information before tool execution.",
                )

            if step.get("final_ready"):
                answer = str(step.get("answer_draft") or "").strip()
                if not answer:
                    answer = "I do not have enough verified information to answer that yet."
                return _wrap_whatsapp_response(
                    answer,
                    confidence=88.0,
                    reasoning="Council produced final answer from available context/tool evidence.",
                )

        return _wrap_whatsapp_response(
            "I could not complete that within the iteration budget.",
            confidence=50.0,
            reasoning=f"Reached maximum recursion depth of {self.max_depth} rounds."
        )

# Global thinking palace instance
thinking_palace = ThinkingPalace()
