import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from app.brain.tool_gateway import (
    validate_tool_call,
    validation_result_to_response,
    capability_index_summary,
    select_relevant_manuals
)
from app.brain.response_cache import response_cache

CACHEABLE_ACTIONS = {"weather.forecast", "finance.exchange_rate", "search.benchmarks"}

class ToolService:
    """
    Decoupled service handling dynamic tool loading, parameter validation,
    5-minute cache wrapper lookups, and multi-thread parallel tool execution.
    """
    @staticmethod
    def get_summary() -> str:
        """Returns categories and brief descriptions to shrink the starting LLM context packet."""
        return capability_index_summary()

    @staticmethod
    def get_relevant_manuals(user_message: str, limit: int = 8) -> str:
        """Fetches markdown docs for category items matching user's keywords."""
        return select_relevant_manuals(user_message, limit=limit)

    @staticmethod
    def sophie_tool_cached(action: str, payload_json: str = "{}") -> str:
        """Invokes native gateway handler checking the response cache for deterministic queries."""
        payload_json = payload_json or "{}"
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}

        if action in CACHEABLE_ACTIONS:
            cache_key = response_cache.make_key(action, payload)
            cached = response_cache.get(cache_key)
            if cached:
                print(f"[ToolService] Cache Hit! action={action}")
                return cached

        # Call underlying tools layer sophie_tool dynamically
        from app.brain.tools import sophie_tool
        result = sophie_tool(action, payload_json)

        if action in CACHEABLE_ACTIONS and "error" not in result.lower() and "failed" not in result.lower():
            cache_key = response_cache.make_key(action, payload)
            response_cache.set(cache_key, result)

        return result

    @staticmethod
    def execute_parallel(tool_calls: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Spawns concurrent worker threads executing independent tools in parallel."""
        results = []
        if not tool_calls:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for call in tool_calls:
                action = call.get("action")
                payload = call.get("payload") or {}
                payload_json = json.dumps(payload)
                futures[executor.submit(ToolService.sophie_tool_cached, action, payload_json)] = call

            for future, call in futures.items():
                action = call.get("action")
                payload = call.get("payload") or {}
                try:
                    res = future.result(timeout=30.0)
                except Exception as e:
                    res = f"Parallel worker error: {str(e)}"
                results.append({
                    "action": action,
                    "payload": payload,
                    "response": res
                })
        return results

tool_service = ToolService()
