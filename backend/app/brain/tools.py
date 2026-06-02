import json
import re
from typing import Annotated, Dict, Any, List, Optional, TypedDict
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_FREE_MODEL_POOL,
    OPENROUTER_MAX_MODEL_ATTEMPTS,
    OPENROUTER_MODEL,
    OPENROUTER_TOOL_MODEL,
    SOPHIE_TOOL_MODE,
)
from app.database import db
from app.brain.doc_brain import doc_brain
from app.brain.memory_os import memory_os
from app.brain.document_ingest import (
    ingest_text_document as ingest_text_document_impl,
    ingest_local_file as ingest_local_file_impl,
    ingest_url_document as ingest_url_document_impl,
    list_ingested_documents as list_ingested_documents_impl,
)
from app.brain.safety import create_pending_approval, is_destructive_command, run_shell_command
from app.brain.research_engine import extract_currency_pair, format_research_result, is_exchange_rate_query, research_query
from app.brain.tool_gateway import (
    canonical_action_name,
    capability_index,
    capability_index_summary,
    dispatch_tool_action,
    discover_tools,
    format_context_packet,
    manual_for,
    select_relevant_manuals,
    validate_tool_call,
    validation_result_to_response,
)
from app.brain.memory_curator import review_and_save_memories
from app.brain.time_tools import current_time_reply, is_current_time_query
from app.brain.system_prompt import build_stage1_planner_prompt, build_stage2_synthesizer_prompt

# Configure Gemini when available. OpenRouter mode uses LangChain instead.
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Global tracker for executed tools in a single orchestrator run
ACTIVE_RUN_TOOLS = []
ACTIVE_RUN_SENDER = "unknown_sender"

def reset_active_run_tools(sender: str = "unknown_sender"):
    global ACTIVE_RUN_TOOLS, ACTIVE_RUN_SENDER
    ACTIVE_RUN_TOOLS = []
    ACTIVE_RUN_SENDER = sender

def record_tool_execution(tool_name: str, args: dict, response: str):
    global ACTIVE_RUN_TOOLS
    # Check if this tool execution was already recorded to avoid duplicates
    for t in ACTIVE_RUN_TOOLS:
        if t["tool"] == tool_name and t["arguments"] == args:
            return
    ACTIVE_RUN_TOOLS.append({
        "tool": tool_name,
        "arguments": args,
        "response": response
    })

def load_tool_manuals() -> str:
    """Loads all markdown manuals from backend/app/brain/tool_manuals/ to dynamically inject into the Planner model."""
    import os
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        manuals_dir = os.path.join(current_dir, "tool_manuals")
        if not os.path.exists(manuals_dir):
            return "No tool manuals found directory."

        manuals_content = []
        for filename in sorted(os.listdir(manuals_dir)):
            if filename.endswith(".md"):
                filepath = os.path.join(manuals_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    manuals_content.append(f.read().strip())

        return "\n\n---\n\n".join(manuals_content)
    except Exception as e:
        return f"Error loading tool manuals: {str(e)}"

def openrouter_chat(system_prompt: str, user_prompt: str, user_message: str = "", requires_tools: bool = False) -> str:
    """Configured LLM Router. Delegates chat calls to the dynamic LLM adapter."""
    from app.brain.llm_adapter import get_llm_adapter
    adapter = get_llm_adapter()
    return adapter.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        user_message=user_message,
        requires_tools=requires_tools
    )

def _raw_openrouter_chat(system_prompt: str, user_prompt: str, user_message: str = "", requires_tools: bool = False) -> str:
    """Runs a single OpenRouter chat completion through LangChain/OpenAI-compatible API."""
    from langchain_core.messages import HumanMessage, SystemMessage

    last_error = None
    for model_id in get_openrouter_model_candidates(user_message, requires_tools=requires_tools):
        try:
            log_to_sophie_brain("OPENROUTER_MODEL_ATTEMPT", f"model={model_id} mode={openrouter_model_mode()} path=chat")
            llm = build_openrouter_llm(model_id)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            log_to_sophie_brain("OPENROUTER_MODEL_SUCCESS", f"model={model_id} path=chat")
            return str(response.content or "").strip()
        except Exception as exc:
            last_error = exc
            log_to_sophie_brain("OPENROUTER_MODEL_FAILURE", f"model={model_id} path=chat error={exc}")
            if openrouter_model_mode() == "fixed" or not is_openrouter_retryable_error(exc):
                break
    raise RuntimeError(f"OpenRouter chat failed for all configured model candidates: {last_error}")

def openrouter_model_mode() -> str:
    return "free_auto" if not OPENROUTER_MODEL or OPENROUTER_MODEL.lower() == "none" else "fixed"

def get_openrouter_model_candidates(user_message: str = "", requires_tools: bool = True) -> List[str]:
    """Returns a fixed model or a task-aware free fallback pool."""
    if openrouter_model_mode() == "fixed":
        return [OPENROUTER_MODEL]

    pool = list(dict.fromkeys(OPENROUTER_FREE_MODEL_POOL or ["openrouter/free"]))
    q = (user_message or "").lower()
    preferred: List[str] = []
    if requires_tools and OPENROUTER_TOOL_MODEL:
        preferred.append(OPENROUTER_TOOL_MODEL)
    if any(term in q for term in ["code", "coding", "program", "script", "bug", "command", "powershell", "python", "javascript"]):
        preferred.extend([
            "qwen/qwen3-coder:free",
            "poolside/laguna-m.1:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ])
    elif requires_tools or openrouter_query_needs_tool(user_message):
        preferred.extend([
            "nvidia/nemotron-3-super-120b-a12b:free",
            "deepseek/deepseek-v4-flash:free",
            "qwen/qwen3-coder:free",
        ])
    preferred.extend(pool)
    preferred.append("openrouter/free")
    candidates = [
        model
        for model in dict.fromkeys(preferred)
        if model in pool or model == "openrouter/free" or (OPENROUTER_TOOL_MODEL and model == OPENROUTER_TOOL_MODEL)
    ]
    return candidates[:max(1, OPENROUTER_MAX_MODEL_ATTEMPTS)]

def is_openrouter_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in [
        "429",
        "rate limit",
        "rate-limit",
        "temporarily",
        "provider returned error",
        "no endpoints found",
        "unavailable",
        "overloaded",
        "timeout",
        "timed out",
        "tool use",
        "tool support",
    ])

def build_openrouter_llm(model_id: str):
    """Builds the LangChain OpenRouter chat model."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_id,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        timeout=45,
        max_retries=1,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Sophie WhatsApp Assistant",
        },
    )

class OpenRouterToolGraphState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    iterations: int

def build_sophie_langchain_tools():
    """Wraps Sophie's native tools as LangChain tools for OpenRouter tool calling."""
    from langchain_core.tools import StructuredTool

    if SOPHIE_TOOL_MODE == "gateway":
        return [StructuredTool.from_function(sophie_tool)]

    native_tools = [
        create_folder,
        create_desktop_folder,
        get_weather,
        search_web,
        google_search,
        research_latest,
        research_benchmarks,
        get_exchange_rate,
        create_repeating_task,
        add_calendar_event,
        list_calendar_events,
        view_local_file,
        write_local_file,
        ingest_text_document,
        ingest_local_file,
        ingest_url_document,
        list_ingested_documents,
        create_agent,
        list_agents,
        delete_agent,
        call_agent,
        execute_command,
    ]
    return [StructuredTool.from_function(tool_func) for tool_func in native_tools]

def run_openrouter_tool_graph(system_prompt: str, user_prompt: str, user_message: str = "") -> str:
    """Runs an OpenRouter-only LangGraph loop: model -> tools -> model until done."""
    from langchain_core.messages import HumanMessage, SystemMessage

    tools = build_sophie_langchain_tools()
    tool_node = ToolNode(tools)

    last_error = None
    for model_id in get_openrouter_model_candidates(user_message or user_prompt, requires_tools=True):
        try:
            log_to_sophie_brain("OPENROUTER_MODEL_ATTEMPT", f"model={model_id} mode={openrouter_model_mode()} path=tool_graph")
            llm_with_tools = build_openrouter_llm(model_id).bind_tools(tools)

            def agent_node(state: OpenRouterToolGraphState) -> Dict[str, Any]:
                log_to_sophie_brain(
                    "LANGGRAPH_AGENT_NODE",
                    f"iteration={int(state.get('iterations', 0)) + 1} model={model_id}"
                )
                response = llm_with_tools.invoke(state["messages"])
                return {
                    "messages": [response],
                    "iterations": int(state.get("iterations", 0)) + 1,
                }

            def tool_node_with_log(state: OpenRouterToolGraphState) -> Dict[str, Any]:
                messages = state.get("messages", [])
                last_message = messages[-1] if messages else None
                tool_calls = getattr(last_message, "tool_calls", None) or []
                names = [
                    call.get("name", "unknown_tool") if isinstance(call, dict) else str(call)
                    for call in tool_calls
                ]
                log_to_sophie_brain(
                    "LANGGRAPH_TOOL_NODE",
                    f"executing {len(tool_calls)} call(s): {', '.join(names) if names else 'none'}"
                )
                return tool_node.invoke(state)

            def should_continue(state: OpenRouterToolGraphState) -> str:
                messages = state.get("messages", [])
                if not messages:
                    log_to_sophie_brain("LANGGRAPH_ROUTE", "no messages -> end")
                    return "end"
                last_message = messages[-1]
                tool_calls = getattr(last_message, "tool_calls", None) or []
                if tool_calls and int(state.get("iterations", 0)) < 6:
                    log_to_sophie_brain("LANGGRAPH_ROUTE", f"tool_calls={len(tool_calls)} -> tools")
                    return "tools"
                log_to_sophie_brain("LANGGRAPH_ROUTE", "no tool calls -> end")
                return "end"

            workflow = StateGraph(OpenRouterToolGraphState)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", tool_node_with_log)
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
            workflow.add_edge("tools", "agent")
            graph = workflow.compile()

            result = graph.invoke({
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                "iterations": 0,
            })
            messages = result.get("messages", [])
            if not messages:
                raise RuntimeError("OpenRouter tool graph returned no messages.")
            log_to_sophie_brain("OPENROUTER_MODEL_SUCCESS", f"model={model_id} path=tool_graph")
            return str(getattr(messages[-1], "content", "") or "").strip()
        except Exception as exc:
            last_error = exc
            log_to_sophie_brain("OPENROUTER_MODEL_FAILURE", f"model={model_id} path=tool_graph error={exc}")
            if openrouter_model_mode() == "fixed" or not is_openrouter_retryable_error(exc):
                break
    raise RuntimeError(f"OpenRouter tool graph failed for all configured model candidates: {last_error}")

def maybe_prefetch_openrouter_tools(message: str) -> str:
    """Small deterministic tool pass for OpenRouter mode, which has no Gemini auto tool loop."""
    query = (message or "").strip()
    q = query.lower()
    if not query:
        return "No live tools were called."

    try:
        if any(term in q for term in ["benchmark", "leaderboard", "top model", "top 3", "top 5", "lmarena", "swe-bench", "arc-agi"]):
            return research_benchmarks(query)
        if any(term in q for term in ["today", "latest", "recent", "newest", "current", "news", "release", "announced", "schedule", "price"]):
            return research_latest(query)
    except Exception as exc:
        return f"OpenRouter prefetch tool failed: {exc}"

    return "No live tools were called."


def is_agent_request(message: str) -> bool:
    q = (message or "").lower()
    return any(term in q for term in ["agent", "sub agent", "sub-agent", "create agent", "call agent"])


def is_simple_current_time_query(message: str) -> bool:
    return is_current_time_query(message) and not is_agent_request(message)


def _contains_any(text: str, terms: List[str]) -> bool:
    q = (text or "").lower()
    return any(term in q for term in terms)


def _extract_after_pattern(message: str, patterns: List[str]) -> str:
    text = (message or "").strip()
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = re.sub(r"[?.!,]+$", "", match.group(1).strip())
            if value:
                return value
    return ""


def _tool_action(action: str, payload: Dict[str, Any], reason: str, required: bool = True) -> Dict[str, Any]:
    return {
        "action": action,
        "payload": payload,
        "reason": reason,
        "required": required,
    }


def _base_answer_policy(mode: str = "normal", allow_llm_tool_expansion: bool = False) -> Dict[str, Any]:
    return {
        "mode": mode,
        "allow_llm_tool_expansion": allow_llm_tool_expansion,
        "must_use_verified_tools": mode in {"evidence", "action"},
        "no_fake_tool_claims": True,
        "compact_whatsapp": True,
    }


def _currency_tool_payload(message: str) -> Optional[Dict[str, str]]:
    base, quote = extract_currency_pair(message)
    if not base or not quote:
        return None
    return {"base": base, "quote": quote, "date": "today"}


def _ps_single_quote(value: str) -> str:
    return "'" + (value or "").replace("'", "''") + "'"


def _extract_desktop_folder_name(message: str) -> str:
    text = re.sub(r"\s+", " ", (message or "").strip())
    patterns = [
        r"\bfolder\s+(?:name|named|called)?\s*['\"]?(.+?)['\"]?\s+(?:on|in|at)\s+(?:the\s+)?desktop\b",
        r"\b(?:is there|check|find|see if|whether)\s+(?:a\s+)?folder\s+(?:name|named|called)?\s*['\"]?(.+?)['\"]?\s+(?:on|in|at)\s+(?:the\s+)?desktop\b",
        r"\bdesktop\s+folder\s+(?:name|named|called)?\s*['\"]?(.+?)['\"]?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            name = re.sub(r"[?.!,]+$", "", match.group(1).strip(" '\""))
            if name:
                return name
    return ""


def _desktop_folder_check_command(folder_name: str) -> str:
    quoted_name = _ps_single_quote(folder_name)
    return (
        "$desktop = [Environment]::GetFolderPath('Desktop'); "
        f"$target = Join-Path $desktop {quoted_name}; "
        "if (Test-Path -LiteralPath $target -PathType Container) { "
        "\"FOUND folder: $target\" "
        "} else { "
        "\"NOT_FOUND folder: $target\" "
        "}"
    )


LIVE_EVENT_TERMS = [
    "shooting", "shot", "gunfire", "attack", "attacked", "stabbing", "killed",
    "injured", "wounded", "dead", "death", "arrested", "suspect", "police",
    "incident", "breaking", "explosion", "blast", "fire", "crash", "earthquake",
    "flood", "storm", "protest", "riot", "war", "mosque", "church", "school",
    "airport", "what happened", "updates on",
]

LIVE_INFO_TERMS = [
    "today", "latest", "recent", "newest", "current", "now", "news", "release",
    "released", "announced", "announcement", "schedule", "price", "public figure",
    "president", "prime minister", "ceo", "stock", "market", "peace talk",
    "peace talks", "delegation", "envoy", "official", "representative",
]

PERSON_LOOKUP_TERMS = [
    "who", "name of", "the name", "which guy", "what guy", "guy", "person",
    "official", "representative", "envoy", "delegate", "delegation", "sent",
    "send", "travel", "trip",
]

BENCHMARK_TERMS = [
    "benchmark", "leaderboard", "ranking", "score", "top model", "top 3", "top 5",
    "lmarena", "lmsys", "swe-bench", "arc-agi", "artificial analysis",
]

WEATHER_TERMS = ["weather", "forecast", "rain", "temperature", "humidity"]
FILE_TERMS = [
    "file", "folder", "directory", "document", "pdf", "desktop",
    "read file", "write file", "create file", "open file", "show file",
]
COMMAND_TERMS = ["run command", "powershell", "terminal", "cmd", "execute command", "shell"]
MEMORY_TERMS = ["remember", "memory", "preference", "what do you know about me"]
HISTORY_TERMS = ["what did i say", "what did i ask", "earlier", "before", "previous", "my first message", "same thing", "that"]
SCHEDULE_TERMS = ["remind", "reminder", "schedule", "calendar", "every day", "tomorrow"]

PUBLIC_AFFAIRS_CONTEXT_TERMS = [
    "america", "united states", "u.s.", "islamabad", "pakistan", "white house",
]
PUBLIC_AFFAIRS_ACTION_TERMS = [
    "peace talk", "peace talks", "talks", "delegation", "envoy", "official",
    "representative", "diplomat", "sent", "send", "travel", "trip", "headed",
    "negotiation", "ceasefire",
]


def is_public_affairs_person_lookup(message: str) -> bool:
    q = (message or "").lower()
    has_us_word = bool(re.search(r"\bus\b", q))
    has_context = has_us_word or _contains_any(q, PUBLIC_AFFAIRS_CONTEXT_TERMS)
    return (
        _contains_any(q, PERSON_LOOKUP_TERMS)
        and has_context
        and _contains_any(q, PUBLIC_AFFAIRS_ACTION_TERMS)
    )


def build_brain_decision(message: str, docs_count: int = 0, memories_count: int = 0) -> Dict[str, Any]:
    """Deterministic brain contract: Python decides when evidence/tools are required."""
    query = (message or "").strip()
    q = query.lower()
    tool_plan: List[Dict[str, Any]] = []
    missing_info: List[str] = []
    intent = "general_chat"
    route = "plain_chat"
    needs_live_evidence = False
    needs_memory = False
    needs_history = False
    risk_level = "safe"
    answer_policy = _base_answer_policy()

    if is_lightweight_chat_message(query):
        intent = "lightweight_chat"
        route = "lightweight_chat"
    elif is_simple_current_time_query(query):
        intent = "current_time"
        route = "deterministic_time"
        needs_live_evidence = True
        answer_policy = _base_answer_policy("evidence")
        tool_plan.append(_tool_action("time.current", {"query": query}, "Current time must come from the clock tool."))
    elif is_exchange_rate_query(query) or _contains_any(query, ["usd", "pkr", "dollar", "rupee", "forex", "exchange rate", "currency"]):
        intent = "finance_exchange_rate"
        route = "evidence_tool_plan"
        needs_live_evidence = True
        answer_policy = _base_answer_policy("evidence")
        payload = _currency_tool_payload(query)
        if payload:
            tool_plan.append(_tool_action("finance.exchange_rate", payload, "Exchange-rate answers require live finance evidence."))
        else:
            missing_info.append("currency_pair")
    elif _contains_any(query, WEATHER_TERMS):
        intent = "weather"
        route = "evidence_tool_plan"
        needs_live_evidence = True
        answer_policy = _base_answer_policy("evidence")
        location = _extract_after_pattern(query, [r"\bweather(?: forecast)?\s+(?:in|for|at)\s+(.+)$", r"\bforecast\s+(?:in|for|at)\s+(.+)$"])
        if location:
            tool_plan.append(_tool_action("weather.forecast", {"location": location, "days": 1}, "Weather must come from the live weather tool."))
        else:
            missing_info.append("weather_location")
    elif _contains_any(query, BENCHMARK_TERMS):
        intent = "benchmark_research"
        route = "evidence_tool_plan"
        needs_live_evidence = True
        answer_policy = _base_answer_policy("evidence")
        tool_plan.append(_tool_action("search.benchmarks", {"query": query}, "Benchmark/ranking answers require sourced benchmark research."))
    elif (
        _contains_any(query, LIVE_INFO_TERMS)
        or _contains_any(query, LIVE_EVENT_TERMS)
        or is_public_affairs_person_lookup(query)
        or re.search(r"\b(?:what happened|tell me about|updates? on)\b.+\bin\b.+", q)
    ):
        intent = "current_research"
        route = "evidence_tool_plan"
        needs_live_evidence = True
        answer_policy = _base_answer_policy("evidence")
        tool_plan.append({
            **_tool_action("search.latest", {"query": query}, "Current events and live facts require fresh sourced search."),
            "fallback_action": "search.web",
            "fallback_payload": {"query": query},
            "fallback_reason": "Use broad web search if latest research has weak/no claims.",
        })
    elif _contains_any(query, HISTORY_TERMS):
        intent = "chat_history"
        route = "context_chat"
        needs_history = True
        answer_policy = _base_answer_policy("context")
        tool_plan.append(_tool_action("chat.history", {"limit": 7}, "Follow-up/history questions need recent chat history."))
    elif _contains_any(query, MEMORY_TERMS):
        intent = "memory"
        route = "context_chat"
        needs_memory = True
        answer_policy = _base_answer_policy("context")
        if "remember" in q and len(query) > 12:
            tool_plan.append(_tool_action("memory.save", {"content": query, "type": "semantic"}, "Durable user facts can be saved as memory."))
        else:
            tool_plan.append(_tool_action("memory.recall", {"query": query, "limit": 5}, "Memory questions need long-term memory lookup."))
    elif _contains_any(query, SCHEDULE_TERMS):
        intent = "schedule"
        route = "action_tool_plan"
        risk_level = "safe_action"
        answer_policy = _base_answer_policy("action", allow_llm_tool_expansion=True)
        if "remind" in q or "reminder" in q:
            missing_info.append("due_at_iso_datetime")
        else:
            tool_plan.append(_tool_action("schedule.list", {}, "Calendar/schedule lookup can be read safely."))
    elif _contains_any(query, COMMAND_TERMS) or is_agent_request(query):
        intent = "agent_or_system"
        route = "action_tool_plan"
        risk_level = "approval_if_destructive"
        answer_policy = _base_answer_policy("action", allow_llm_tool_expansion=True)
        if is_agent_request(query):
            tool_plan.append(_tool_action("agents.list", {}, "Discover available advisory agents before using agent actions.", required=False))
        else:
            missing_info.append("exact_command_line")
    elif _contains_any(query, FILE_TERMS):
        intent = "workspace_or_document"
        route = "action_tool_plan"
        risk_level = "safe_action"
        answer_policy = _base_answer_policy("action", allow_llm_tool_expansion=False)
        desktop_folder_name = _extract_desktop_folder_name(query)
        if desktop_folder_name:
            tool_plan.append(_tool_action(
                "system.command",
                {"command_line": _desktop_folder_check_command(desktop_folder_name)},
                "Checking whether a named folder exists on the Windows Desktop is a safe read-only command.",
            ))
        elif any(term in q for term in ["read file", "open file", "show file", "view file"]):
            missing_info.append("file_path")
        elif any(term in q for term in ["create folder", "create a folder", "new folder", "make folder", "make a folder"]):
            missing_info.append("folder_name_or_path")
        else:
            answer_policy = _base_answer_policy("context", allow_llm_tool_expansion=True)
    elif docs_count or memories_count:
        intent = "context_chat"
        route = "context_chat"
        needs_memory = bool(memories_count)
        answer_policy = _base_answer_policy("context")

    needs_tool = route in {"evidence_tool_plan", "action_tool_plan", "deterministic_time"} or bool(tool_plan)
    decision = {
        "intent": intent,
        "route": route,
        "needs_tool": needs_tool,
        "needs_live_evidence": needs_live_evidence,
        "needs_memory": needs_memory,
        "needs_history": needs_history,
        "risk_level": risk_level,
        "tool_plan": tool_plan,
        "answer_policy": answer_policy,
        "missing_info": missing_info,
        "docs_count": docs_count,
        "memories_count": memories_count,
        "manuals_hint": select_relevant_manuals(query, limit=4),
    }
    decision["sophie"] = f"Intent={intent}; route={route}; tools={len(tool_plan)}; risk={risk_level}."
    decision["explorer"] = (
        f"Context candidates: docs={docs_count}, memories={memories_count}; "
        f"history_needed={needs_history}; memory_needed={needs_memory}."
    )
    decision["evaluator"] = (
        "Answer only from verified tool/context evidence when needs_tool or needs_live_evidence is true."
    )
    return decision


def openrouter_intent_needs_tools(message: str) -> bool:
    return bool(build_brain_decision(message).get("needs_tool"))

def openrouter_query_needs_tool(message: str) -> bool:
    """Detects cases where OpenRouter must not answer from memory alone."""
    return openrouter_intent_needs_tools(message)

def summarize_intent_for_logs(message: str) -> str:
    return str(build_brain_decision(message).get("intent") or "general_chat")


def determine_thinkbox_route(message: str) -> str:
    route = str(build_brain_decision(message).get("route") or "plain_chat")
    legacy_routes = {
        "deterministic_time": "deterministic_time_tool",
        "evidence_tool_plan": "langgraph_tool_graph",
        "action_tool_plan": "langgraph_tool_graph",
        "context_chat": "openrouter_plain_chat",
        "plain_chat": "openrouter_plain_chat",
    }
    return legacy_routes.get(route, route)


def build_thinkbox_payload(message: str, docs_count: int = 0, memories_count: int = 0) -> Dict[str, Any]:
    return build_brain_decision(message, docs_count=docs_count, memories_count=memories_count)


def log_thinkbox_payload(payload: Dict[str, Any]) -> None:
    compact = {
        "intent": payload.get("intent"),
        "route": payload.get("route"),
        "needs_tool": payload.get("needs_tool"),
        "needs_live_evidence": payload.get("needs_live_evidence"),
        "risk_level": payload.get("risk_level"),
        "tool_plan": payload.get("tool_plan"),
        "missing_info": payload.get("missing_info"),
    }
    log_to_sophie_brain("BRAIN_DECISION", json.dumps(compact, ensure_ascii=True)[:900])
    log_to_sophie_brain(
        "THINKBOX_OPEN",
        f"intent={payload.get('intent')} route={payload.get('route')} needs_tool={payload.get('needs_tool')}"
    )
    log_to_sophie_brain("THINKBOX_SOPHIE", str(payload.get("sophie", "")))
    log_to_sophie_brain("THINKBOX_EXPLORER", str(payload.get("explorer", "")))
    log_to_sophie_brain("THINKBOX_EVALUATOR", str(payload.get("evaluator", "")))
    log_to_sophie_brain(
        "THINKBOX_DECISION",
        f"route={payload.get('route')} manuals_hint={str(payload.get('manuals_hint', ''))[:220]}"
    )


def log_council_snapshot(message: str, route: str, needs_tool: bool) -> None:
    payload = build_thinkbox_payload(message)
    payload["route"] = route
    payload["needs_tool"] = needs_tool
    log_thinkbox_payload(payload)

def is_lightweight_chat_message(message: str) -> bool:
    """Short greetings/acks should not burn the full RAG + tool-calling graph."""
    q = re.sub(r"[^a-z0-9\s]", "", (message or "").lower()).strip()
    if not q:
        return True
    greetings = {
        "hi", "hello", "hey", "yo", "aoa", "salam", "assalamualaikum",
        "assalam o alaikum", "ok", "okay", "thanks", "thank you", "kesi ho",
        "kaisi ho", "how are you"
    }
    return len(q) <= 24 and (q in greetings or q.startswith(("hi ", "hello ", "hey ")))

def gateway_handlers() -> Dict[str, Any]:
    return {
        "search_web": search_web,
        "google_search": google_search,
        "research_latest": research_latest,
        "research_benchmarks": research_benchmarks,
        "get_exchange_rate": get_exchange_rate,
        "create_repeating_task": create_repeating_task,
        "add_calendar_event": add_calendar_event,
        "list_calendar_events": list_calendar_events,
        "view_local_file": view_local_file,
        "write_local_file": write_local_file,
        "ingest_text_document": ingest_text_document,
        "ingest_local_file": ingest_local_file,
        "ingest_url_document": ingest_url_document,
        "list_ingested_documents": list_ingested_documents,
        "create_agent": create_agent,
        "list_agents": list_agents,
        "delete_agent": delete_agent,
        "call_agent": call_agent,
        "execute_command": execute_command,
    }

def sophie_tool(action: str, payload_json: str = "{}") -> str:
    """Sophie gateway tool. Call one action by name with a JSON payload.

    Args:
        action: Gateway action such as 'tool_list', 'get_tool_manual', 'tools.discover', 'tools.manual', 'search.latest',
            'finance.exchange_rate', 'weather.forecast', 'time.current', 'chat.history', 'memory.recall',
            'memory.save', 'schedule.reminder', 'documents.ingest_url', 'system.command', or 'tools.call'.
        payload_json: JSON object string containing the action parameters.
    """
    payload_json = payload_json or "{}"
    log_to_sophie_brain("TOOL_CALL", f"sophie_tool(action='{action}', payload='{payload_json[:160]}')")
    validation = validate_tool_call(action, payload_json)
    log_to_sophie_brain(
        "TOOL_VALIDATION",
        f"action={validation.action} ok={validation.ok} missing={validation.missing_fields} errors={validation.errors}"
    )
    if not validation.ok:
        res = validation_result_to_response(validation)
        log_to_sophie_brain("TOOL_RESPONSE", f"sophie_tool validation failed: {res[:300]}...")
        return res
    try:
        res = dispatch_tool_action(
            action=validation.action,
            payload_json=json.dumps(validation.normalized_payload, ensure_ascii=True),
            context={"sender": ACTIVE_RUN_SENDER},
            handlers=gateway_handlers(),
        )
    except Exception as exc:
        res = f"sophie_tool failed for action '{validation.action}': {exc}"
    log_to_sophie_brain("TOOL_RESPONSE", f"sophie_tool returned: {res[:300]}...")
    record_tool_execution("sophie_tool", {"action": validation.action, "payload_json": json.dumps(validation.normalized_payload, ensure_ascii=True)}, res)
    return res


def _parse_structured_research_result(text: str) -> Optional[Dict[str, Any]]:
    if "STRUCTURED_RESEARCH_RESULT" not in (text or ""):
        return None
    json_start = text.find("{")
    if json_start < 0:
        return None
    try:
        return json.loads(text[json_start:])
    except Exception:
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(text[json_start:])
            return data if isinstance(data, dict) else None
        except Exception:
            return None


def evaluate_tool_evidence(action: str, response: str) -> Dict[str, Any]:
    response_text = response or ""
    lower = response_text.lower()
    if not response_text.strip():
        return {"quality": "failed", "reason": "empty response"}

    structured = _parse_structured_research_result(response_text)
    if structured:
        claims = structured.get("claims") or []
        citations = structured.get("citations") or []
        answerable = bool(structured.get("answerable"))
        if answerable and claims:
            return {"quality": "strong", "reason": "structured claims found", "claims": len(claims), "citations": len(citations)}
        if citations:
            return {"quality": "partial", "reason": "citations found but no high-confidence claims", "claims": len(claims), "citations": len(citations)}
        return {"quality": "weak", "reason": "structured research had no claims/citations", "claims": 0, "citations": 0}

    if any(marker in lower for marker in [" failed", "failed:", "error:", "cannot be empty", "no live search results"]):
        return {"quality": "failed", "reason": response_text[:180]}
    if action.startswith("search.") and any(marker in lower for marker in ["no result", "no relevant", "did not find"]):
        return {"quality": "weak", "reason": response_text[:180]}
    return {"quality": "strong", "reason": "non-empty tool result"}


def format_verified_live_data() -> str:
    if not ACTIVE_RUN_TOOLS:
        return "No live tools were called."
    lines = ["--- VERIFIED LIVE TOOL EXECUTION RESULTS ---"]
    for t in ACTIVE_RUN_TOOLS:
        lines.append(f"Tool Called: {t['tool']}")
        lines.append(f"Arguments: {json.dumps(t['arguments'], ensure_ascii=True)}")
        lines.append("Response:")
        lines.append(str(t.get("response", ""))[:6000])
        lines.append("---------------------------------------")
    return "\n".join(lines)


def _structured_results_from_active_tools() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for tool_run in ACTIVE_RUN_TOOLS:
        parsed = _parse_structured_research_result(str(tool_run.get("response", "")))
        if parsed:
            results.append(parsed)
    results.sort(key=lambda item: int(item.get("trust_score") or 0), reverse=True)
    return results


def _short_text(value: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", clean_xml_from_text(value or "")).strip()
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def build_deterministic_evidence_response(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fast synthesis for structured evidence, avoiding slow model calls for clear tool results."""
    if str((decision or {}).get("route")) != "evidence_tool_plan":
        return None
    results = _structured_results_from_active_tools()
    if not results:
        return None

    best = results[0]
    claims = list(best.get("claims") or [])
    citations = list(best.get("citations") or [])
    trust = int(best.get("trust_score") or 0)
    if not claims or not best.get("answerable"):
        return None

    claim_type = str(claims[0].get("claim_type") or "claim")
    if claim_type == "news_report":
        lines = ["I found current reports on this:"]
        for claim in claims[:3]:
            value = _short_text(str(claim.get("value") or ""), 260)
            if value:
                lines.append(f"- {value}")
        source_names = []
        for citation in citations[:4]:
            title = _short_text(str(citation.get("title") or citation.get("domain") or ""), 90)
            if title and title not in source_names:
                source_names.append(title)
        if source_names:
            lines.append(f"Sources: {', '.join(source_names[:3])}.")
        lines.append(f"Confidence: {trust}%.")
    elif claim_type == "person_lookup":
        top = claims[0]
        name = _short_text(str(top.get("value") or ""), 160)
        lines = [f"The name I found is: {name}."]
        details = top.get("details") or []
        if details:
            lines.append(_short_text(str(details[0]), 260))
        source_titles = top.get("source_titles") or []
        if source_titles:
            lines.append(f"Source: {_short_text(str(source_titles[0]), 120)}.")
        lines.append(f"Confidence: {trust}%.")
    elif claim_type == "exchange_rate":
        claim = claims[0]
        lines = [f"{claim.get('value')}"]
        details = claim.get("details") or []
        if details:
            lines.append(_short_text(str(details[0]), 220))
        lines.append(f"Confidence: {trust}%.")
    else:
        lines = ["I found verified source-backed results:"]
        for claim in claims[:3]:
            value = _short_text(str(claim.get("value") or ""), 240)
            if value:
                lines.append(f"- {value}")
        lines.append(f"Confidence: {trust}%.")

    response_text = f"<ouput><user_output>{chr(10).join(lines)}</user_output><tools_call>{_actual_tools_summary()}</tools_call></ouput>"
    return {
        "reasoning": "Structured tool evidence was strong enough for deterministic synthesis.",
        "response": response_text,
        "confidence_score": float(max(60, min(100, trust))),
        "citations": citations,
    }


def build_deterministic_action_response(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str((decision or {}).get("route")) != "action_tool_plan" or not ACTIVE_RUN_TOOLS:
        return None
    last = ACTIVE_RUN_TOOLS[-1]
    response = str(last.get("response") or "")
    args = last.get("arguments") or {}
    command_text = ""
    if last.get("tool") == "sophie_tool":
        try:
            payload = json.loads(str(args.get("payload_json") or "{}"))
            command_text = str(payload.get("command_line") or payload.get("command") or payload.get("cmd") or "")
        except Exception:
            command_text = str(args)

    user_output = ""
    if "FOUND folder:" in response:
        found = response.split("FOUND folder:", 1)[1].strip().splitlines()[0].strip()
        user_output = f"Yes, I found that folder on the Desktop:\n{found}"
    elif "NOT_FOUND folder:" in response:
        missing = response.split("NOT_FOUND folder:", 1)[1].strip().splitlines()[0].strip()
        user_output = f"No, I did not find that folder on the Desktop.\nChecked path:\n{missing}"
    elif last.get("tool") in {"execute_command", "sophie_tool"} and command_text:
        user_output = _short_text(response, 900)

    if not user_output:
        return None
    response_text = f"<ouput><user_output>{user_output}</user_output><tools_call>{_actual_tools_summary()}</tools_call></ouput>"
    return {
        "reasoning": "Safe action tool result was clear enough for deterministic synthesis.",
        "response": response_text,
        "confidence_score": 100.0 if "FOUND folder:" in response or "NOT_FOUND folder:" in response else 85.0,
        "citations": [],
    }


def build_deterministic_tool_response(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return build_deterministic_evidence_response(decision) or build_deterministic_action_response(decision)


class ToolCallRequest(TypedDict, total=False):
    action: str
    payload: Dict[str, Any]
    reason: str


class CouncilStep(TypedDict, total=False):
    round: int
    intent_summary: str
    manual_requests: List[Any]
    tool_calls: List[ToolCallRequest]
    missing_info: List[str]
    final_ready: bool
    answer_draft: str


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return {}
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text[start:])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_tool_calls(value: Any) -> List[ToolCallRequest]:
    calls: List[ToolCallRequest] = []
    for item in _coerce_list(value):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or item.get("tool") or "").strip()
        if not action:
            continue
        payload = item.get("payload")
        if payload is None:
            payload = item.get("payload_json")
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                payload = parsed if isinstance(parsed, dict) else {}
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        calls.append({
            "action": action,
            "payload": payload,
            "reason": str(item.get("reason") or item.get("why") or ""),
        })
    return calls


def _parse_council_step(raw_text: str, round_index: int) -> CouncilStep:
    data = _extract_json_object(raw_text)
    return {
        "round": round_index,
        "intent_summary": str(data.get("intent_summary") or data.get("intent") or "general"),
        "manual_requests": _coerce_list(data.get("manual_requests") or data.get("manuals") or []),
        "tool_calls": _normalize_tool_calls(data.get("tool_calls") or data.get("tools") or []),
        "missing_info": [str(item) for item in _coerce_list(data.get("missing_info") or []) if str(item).strip()],
        "final_ready": bool(data.get("final_ready")),
        "answer_draft": str(data.get("answer_draft") or data.get("answer") or ""),
    }


def _fetch_requested_manuals(manual_requests: List[Any]) -> str:
    chunks: List[str] = []
    for request in manual_requests:
        if isinstance(request, str):
            action = canonical_action_name(request)
            text = manual_for(actions=[action])
            chunks.append(text)
            log_to_sophie_brain("TOOL_SCHEMA_FETCH", f"actions={[action]}")
            continue
        if not isinstance(request, dict):
            continue
        category = str(request.get("category") or "").strip()
        actions = request.get("actions") or request.get("action") or []
        if isinstance(actions, str):
            actions = [actions]
        actions = [canonical_action_name(str(action)) for action in actions if str(action).strip()]
        if actions:
            text = manual_for(actions=actions)
            chunks.append(text)
            log_to_sophie_brain("TOOL_SCHEMA_FETCH", f"actions={actions}")
        elif category:
            text = discover_tools(category)
            chunks.append(text)
            log_to_sophie_brain("TOOL_SCHEMA_FETCH", f"category={category}")
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _format_council_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    if not tool_results:
        return "No council tools have run yet."
    lines: List[str] = []
    for idx, item in enumerate(tool_results[-8:], start=1):
        lines.append(f"<tool_result index=\"{idx}\" action=\"{item.get('action', '')}\" quality=\"{item.get('quality', 'unknown')}\">")
        if item.get("reason"):
            lines.append(f"reason: {item.get('reason')}")
        if item.get("validation_error"):
            lines.append(str(item.get("validation_error"))[:1800])
        else:
            lines.append(str(item.get("response", ""))[:5000])
        lines.append("</tool_result>")
    return "\n".join(lines)


def _answer_from_missing_info(missing_info: List[str]) -> str:
    clean_items = [item.strip() for item in missing_info if item.strip()]
    if not clean_items:
        return "I need one more detail before I can do that."
    if len(clean_items) == 1:
        return f"I need one more detail: {clean_items[0]}."
    return "I need these details first: " + ", ".join(clean_items[:4]) + "."


def _wrap_whatsapp_response(user_output: str, confidence: float = 80.0, citations: Optional[List[Dict[str, Any]]] = None, reasoning: str = "Council synthesis completed.") -> Dict[str, Any]:
    response_text = f"<ouput><user_output>{user_output.strip()}</user_output><tools_call>{_actual_tools_summary()}</tools_call></ouput>"
    return {
        "reasoning": reasoning,
        "response": enforce_honest_tool_report(response_text),
        "confidence_score": confidence,
        "citations": citations or [],
    }


def _council_system_prompt(current_date_text: str, current_year_text: str) -> str:
    return f"""
You are Sophie's private Tool Council for a WhatsApp assistant.
Today's date is {current_date_text}; current year is {current_year_text}.

Your job is to decide tools dynamically from the user's message and the tool manuals.
You do not reveal private council notes to the user.

Council roles:
- Sophie: understand the user's real goal.
- Explorer: check whether chat history, memory, documents, manuals, live evidence, or filesystem evidence are needed.
- Tool Clerk: fetch manuals when action schemas are uncertain and create exact JSON payloads.
- Evaluator: check safety, evidence quality, missing inputs, and whether the answer actually follows from tool results.

Rules:
- Return strict JSON only.
- Use tools.discover when the category/action is unclear.
- Use tools.manual before unfamiliar or schema-sensitive calls.
- Call tools only by returning tool_calls with exact action and payload.
- Do not invent payload fields. Use the manual schema and accepted aliases.
- If required input is missing, set missing_info and final_ready=true with a short answer_draft asking for it.
- Safe Auto: read/search/list/check/current-info tools may run; harmless creates may run when path/content is clear; destructive shell/file/agent actions require approval.
- Never output private reasoning, raw logs, secrets, or environment values.
- Never claim a tool ran unless it appears in the tool results.
- Final answers must use only executed tool results, relevant context, or clear uncertainty.

Strict JSON shape:
{{
  "intent_summary": "short public-safe operational summary",
  "manual_requests": [{{"actions": ["tool.name"]}}, {{"category": "search"}}],
  "tool_calls": [{{"action": "tool.name", "payload": {{}}, "reason": "why this exact tool is needed"}}],
  "missing_info": [],
  "final_ready": false,
  "answer_draft": ""
}}
"""


def _council_user_prompt(
    user_message: str,
    context_packet: str,
    brain_decision: Dict[str, Any],
    gateway_index: str,
    manuals_text: str,
    tool_results: List[Dict[str, Any]],
    round_index: int,
) -> str:
    return f"""
<council_input round="{round_index}">
<user_message>
{user_message}
</user_message>

<context_packet>
{context_packet}
</context_packet>

<brain_contract advisory_only="true">
{json.dumps(brain_decision, ensure_ascii=True, indent=2)}
</brain_contract>

<gateway_capability_index>
{gateway_index}
</gateway_capability_index>

<known_tool_manuals>
{manuals_text or "No manuals fetched yet. Request manuals if needed."}
</known_tool_manuals>

<council_tool_results>
{_format_council_tool_results(tool_results)}
</council_tool_results>
</council_input>

Decide the next council step. If tool results are enough, set final_ready=true and write answer_draft. If tools are needed, set final_ready=false and include tool_calls. If schemas are unclear, request manuals.
"""


def run_tool_council_loop(
    user_message: str,
    context_packet: str,
    brain_decision: Dict[str, Any],
    current_date_text: str,
    current_year_text: str,
    max_rounds: int = 4,
) -> Optional[Dict[str, Any]]:
    """Model-planned, Python-validated tool loop used for non-lightweight messages."""
    if not user_message.strip():
        return None

    gateway_index = capability_index_summary()
    manuals_text = select_relevant_manuals(user_message, limit=8)
    tool_results: List[Dict[str, Any]] = []
    system_prompt = _council_system_prompt(current_date_text, current_year_text)

    for round_index in range(1, max_rounds + 1):
        user_prompt = _council_user_prompt(
            user_message=user_message,
            context_packet=context_packet,
            brain_decision=brain_decision,
            gateway_index=gateway_index,
            manuals_text=manuals_text,
            tool_results=tool_results,
            round_index=round_index,
        )
        raw_step = openrouter_chat(system_prompt, user_prompt, user_message=user_message, requires_tools=False)
        step = _parse_council_step(raw_step, round_index)
        log_to_sophie_brain(
            "COUNCIL_PLAN",
            json.dumps({
                "round": round_index,
                "intent_summary": step.get("intent_summary"),
                "manual_requests": step.get("manual_requests"),
                "tool_calls": step.get("tool_calls"),
                "missing_info": step.get("missing_info"),
                "final_ready": step.get("final_ready"),
            }, ensure_ascii=True)[:1600]
        )

        fetched_manuals = _fetch_requested_manuals(list(step.get("manual_requests") or []))
        if fetched_manuals:
            manuals_text = f"{manuals_text}\n\n{fetched_manuals}".strip()

        tool_calls = list(step.get("tool_calls") or [])
        if tool_calls:
            for call in tool_calls[:4]:
                action = str(call.get("action") or "").strip()
                payload = call.get("payload") if isinstance(call.get("payload"), dict) else {}
                reason = str(call.get("reason") or "")
                validation = validate_tool_call(action, json.dumps(payload, ensure_ascii=True))
                log_to_sophie_brain(
                    "TOOL_VALIDATION",
                    f"council_round={round_index} action={validation.action} ok={validation.ok} missing={validation.missing_fields} errors={validation.errors}"
                )
                if not validation.ok:
                    validation_error = validation_result_to_response(validation)
                    tool_results.append({
                        "action": validation.action,
                        "payload": payload,
                        "reason": reason,
                        "validation_error": validation_error,
                        "quality": "invalid",
                    })
                    continue
                response = sophie_tool(validation.action, json.dumps(validation.normalized_payload, ensure_ascii=True))
                quality = evaluate_tool_evidence(validation.action, response)
                log_to_sophie_brain(
                    "EVIDENCE_QUALITY",
                    f"action={validation.action} quality={quality.get('quality')} reason={quality.get('reason')}"
                )
                log_to_sophie_brain(
                    "TOOL_EXECUTION",
                    f"action={validation.action} payload={json.dumps(validation.normalized_payload, ensure_ascii=True)[:300]}"
                )
                tool_results.append({
                    "action": validation.action,
                    "payload": validation.normalized_payload,
                    "reason": reason,
                    "response": response,
                    "quality": quality.get("quality", "unknown"),
                })
            deterministic_response = build_deterministic_tool_response(brain_decision)
            if deterministic_response and any(item.get("quality") == "strong" for item in tool_results):
                log_to_sophie_brain("COUNCIL_EVALUATION", "Strong verified tool result available; deterministic synthesis accepted.")
                return deterministic_response
            continue

        missing_info = list(step.get("missing_info") or [])
        if missing_info:
            log_to_sophie_brain("COUNCIL_EVALUATION", f"missing_info={missing_info}")
            return _wrap_whatsapp_response(
                _answer_from_missing_info(missing_info),
                confidence=75.0,
                reasoning="Council identified required missing information before tool execution.",
            )

        if step.get("final_ready"):
            answer = str(step.get("answer_draft") or "").strip()
            if not answer:
                answer = "I do not have enough verified information to answer that yet."
            log_to_sophie_brain("COUNCIL_EVALUATION", f"final_ready=true tools={_actual_tools_summary()}")
            return _wrap_whatsapp_response(
                answer,
                confidence=88.0 if ACTIVE_RUN_TOOLS else 70.0,
                reasoning="Council produced final answer from available context/tool evidence.",
            )

        if fetched_manuals:
            continue

        log_to_sophie_brain("COUNCIL_EVALUATION", "Council produced no tool calls, missing info, or final answer.")
        break

    deterministic_response = build_deterministic_tool_response(brain_decision)
    if deterministic_response:
        log_to_sophie_brain("COUNCIL_EVALUATION", "Max rounds reached; deterministic synthesis from executed evidence.")
        return deterministic_response

    if tool_results:
        synthesis_prompt = f"""
Use the verified tool results below to answer the user. Return strict JSON with reasoning, response, confidence_score, citations.
User: {user_message}
Tool results:
{_format_council_tool_results(tool_results)}
Remember: final response must be <ouput><user_output>...</user_output><tools_call>...</tools_call></ouput> and tools_call must list only actual executed tools.
"""
        raw_synthesis = openrouter_chat(
            "You are Sophie. Synthesize only from verified tool results. Do not reveal private council notes.",
            synthesis_prompt,
            user_message=user_message,
            requires_tools=False,
        )
        parsed = parse_strict_json_response(raw_synthesis)
        response_text = enforce_honest_tool_report(parsed["response"])
        log_to_sophie_brain("COUNCIL_EVALUATION", "Max rounds reached; model synthesis from tool results.")
        return {
            "reasoning": parsed["reasoning"],
            "response": response_text,
            "confidence_score": parsed["confidence_score"],
            "citations": parsed["citations"],
        }

    return _wrap_whatsapp_response(
        "I could not verify that with the available tools yet.",
        confidence=40.0,
        reasoning="Council ended without validated tools or sufficient context.",
    )


def execute_brain_tool_plan(decision: Dict[str, Any], user_message: str = "") -> str:
    """Execute the deterministic BrainDecision tool plan before model synthesis."""
    tool_plan = list((decision or {}).get("tool_plan") or [])
    if not tool_plan:
        return "No deterministic tool plan was needed."

    log_to_sophie_brain("TOOL_PLAN", json.dumps(tool_plan, ensure_ascii=True)[:1200])
    for item in tool_plan:
        action = str(item.get("action") or "").strip()
        if not action:
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        try:
            response = sophie_tool(action, json.dumps(payload, ensure_ascii=True))
        except Exception as exc:
            response = f"Deterministic tool execution failed for {action}: {exc}"
        quality = evaluate_tool_evidence(action, response)
        log_to_sophie_brain(
            "EVIDENCE_QUALITY",
            f"action={action} quality={quality.get('quality')} reason={quality.get('reason')}"
        )
        log_to_sophie_brain(
            "TOOL_EXECUTION",
            f"action={action} payload={json.dumps(payload, ensure_ascii=True)[:300]}"
        )

        fallback_action = str(item.get("fallback_action") or "").strip()
        if fallback_action and quality.get("quality") in {"failed", "weak", "partial"}:
            fallback_payload = item.get("fallback_payload") if isinstance(item.get("fallback_payload"), dict) else {"query": user_message}
            log_to_sophie_brain(
                "TOOL_PLAN",
                f"fallback action={fallback_action} reason={item.get('fallback_reason', 'weak primary evidence')}"
            )
            try:
                fallback_response = sophie_tool(fallback_action, json.dumps(fallback_payload, ensure_ascii=True))
            except Exception as exc:
                fallback_response = f"Fallback tool execution failed for {fallback_action}: {exc}"
            fallback_quality = evaluate_tool_evidence(fallback_action, fallback_response)
            log_to_sophie_brain(
                "EVIDENCE_QUALITY",
                f"action={fallback_action} quality={fallback_quality.get('quality')} reason={fallback_quality.get('reason')}"
            )

    return format_verified_live_data()

def execute_command(command_line: str) -> str:
    """Executes a terminal/shell command on the local Windows workspace host securely using PowerShell and returns stdout/stderr.
    
    Args:
        command_line: The exact CLI command line string to run (e.g. 'dir', 'git status', 'python --version').
    """
    log_to_sophie_brain("TOOL_CALL", f"execute_command(command_line='{command_line}')")
    if is_destructive_command(command_line):
        approval = create_pending_approval(
            sender=ACTIVE_RUN_SENDER,
            action_type="delete_command",
            command_line=command_line
        )
        res = approval["message"]
        log_to_sophie_brain("TOOL_RESPONSE", f"execute_command approval required: {res}")
        record_tool_execution("execute_command", {"command_line": command_line}, res)
        return res

    res = run_shell_command(command_line)
    log_to_sophie_brain("TOOL_RESPONSE", f"execute_command returned: {res[:200]}...")
    record_tool_execution("execute_command", {"command_line": command_line}, res)
    return res

def log_to_sophie_brain(category: str, message: str):
    import os
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{category}] {message}\n"
    try:
        print(f"🧠 [{category}] {message}")
    except UnicodeEncodeError:
        try:
            print(f"[BRAIN] [{category}] {message}")
        except Exception:
            pass
    except Exception:
        pass
    try:
        os.makedirs("local_data", exist_ok=True)
        with open("local_data/sophie_brain.log", "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        try:
            print(f"[ERROR] Failed to write to sophie_brain.log: {e}")
        except Exception:
            pass

# --- NATIVE SYSTEM TOOLS FOR SOPHIE ---

def create_folder(folder_name: str) -> str:
    """Creates a new folder directory in the local workspace.
    
    Args:
        folder_name: The name or relative path of the folder to create.
    """
    log_to_sophie_brain("TOOL_CALL", f"create_folder(folder_name='{folder_name}')")
    import os
    try:
        # Get workspace directory (parent of backend folder)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        target_path = os.path.join(base_dir, folder_name)
        os.makedirs(target_path, exist_ok=True)
        res = f"Successfully created folder directory at '{folder_name}'."
        log_to_sophie_brain("TOOL_RESPONSE", f"create_folder returned: {res}")
        record_tool_execution("create_folder", {"folder_name": folder_name}, res)
        return res
    except Exception as e:
        res = f"Failed to create folder '{folder_name}': {str(e)}"
        log_to_sophie_brain("TOOL_RESPONSE", f"create_folder returned: {res}")
        record_tool_execution("create_folder", {"folder_name": folder_name}, res)
        return res

def create_desktop_folder(folder_name: str) -> str:
    """Creates a new folder directory directly on the user's Windows Desktop.
    
    Args:
        folder_name: The name of the folder to create on the Desktop.
    """
    log_to_sophie_brain("TOOL_CALL", f"create_desktop_folder(folder_name='{folder_name}')")
    import os
    try:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        target_path = os.path.join(desktop_path, folder_name)
        os.makedirs(target_path, exist_ok=True)
        res = f"Successfully created folder '{folder_name}' on the Desktop."
        log_to_sophie_brain("TOOL_RESPONSE", f"create_desktop_folder returned: {res}")
        record_tool_execution("create_desktop_folder", {"folder_name": folder_name}, res)
        return res
    except Exception as e:
        res = f"Failed to create Desktop folder '{folder_name}': {str(e)}"
        log_to_sophie_brain("TOOL_RESPONSE", f"create_desktop_folder returned: {res}")
        record_tool_execution("create_desktop_folder", {"folder_name": folder_name}, res)
        return res

def get_weather(location: str) -> str:
    """Retrieves the current live weather, temperature, and condition forecast for any location.
    
    Args:
        location: City/Location name (e.g. 'Paris', 'New York', 'London').
    """
    log_to_sophie_brain("TOOL_CALL", f"get_weather(location='{location}')")
    import requests
    import urllib.parse
    try:
        api_key = "e6b90bde4e554ef5878113900261805"
        encoded_loc = urllib.parse.quote(location)
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={encoded_loc}&aqi=no"
        res = requests.get(url, timeout=5).json()
        if "error" in res:
            err_msg = res["error"].get("message", "Unknown API error")
            res_str = f"Could not find or retrieve weather for location '{location}': {err_msg}"
            log_to_sophie_brain("TOOL_RESPONSE", f"get_weather returned: {res_str}")
            record_tool_execution("get_weather", {"location": location}, res_str)
            return res_str
        
        location_data = res["location"]
        current_data = res["current"]
        
        name = location_data.get("name", location)
        region = location_data.get("region", "")
        country = location_data.get("country", "")
        temp_c = current_data.get("temp_c", "unknown")
        condition = current_data.get("condition", {}).get("text", "unknown")
        humidity = current_data.get("humidity", "unknown")
        wind_kph = current_data.get("wind_kph", "unknown")
        
        loc_str = f"{name}"
        if region:
            loc_str += f", {region}"
        if country:
            loc_str += f", {country}"
            
        res_str = f"Live weather forecast for {loc_str}: The temperature is {temp_c}°C ({condition}), with a humidity of {humidity}%, and wind speeds of {wind_kph} km/h."
        log_to_sophie_brain("TOOL_RESPONSE", f"get_weather returned: {res_str}")
        record_tool_execution("get_weather", {"location": location}, res_str)
        return res_str
    except Exception as e:
        res_str = f"Failed to fetch live weather forecast for '{location}': {str(e)}"
        log_to_sophie_brain("TOOL_RESPONSE", f"get_weather returned: {res_str}")
        record_tool_execution("get_weather", {"location": location}, res_str)
        return res_str

def _clean_search_text(value: str) -> str:
    import html as html_lib
    import re
    text = html_lib.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "how", "in", "is", "it",
    "latest", "major", "me", "new", "news", "of", "on", "or", "release", "releases",
    "tell", "the", "their", "today", "to", "was", "what", "when", "where", "with"
}

GENERIC_SEARCH_DOMAINS = {
    "google.com", "www.google.com", "bing.com", "www.bing.com", "imdb.com",
    "www.imdb.com", "key-test.ru", "en.key-test.ru"
}

def _current_search_year() -> int:
    from datetime import datetime
    return datetime.now().year

def _has_latest_search_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "today", "latest", "recent", "newest", "current", "now", "release",
        "released", "announcement", "announced"
    ])

def _normalize_search_query(query: str) -> str:
    import re

    text = _clean_search_text(query)
    replacements = {
        "goolge": "google",
        "googel": "google",
        "gooogle": "google",
        "googl ": "google ",
        "recet": "recent",
        "recnet": "recent",
        "relese": "release",
        "modle": "model",
        "whats": "what is",
    }
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.I)
    if _has_latest_search_intent(text):
        historical_intent = any(term in text.lower() for term in [
            "in 2024", "in 2025", "during 2024", "during 2025",
            "from 2024", "from 2025", "between 2024", "between 2025",
            "timeline", "history", "older", "previous"
        ])
        if not historical_intent:
            current_year = _current_search_year()
            text = re.sub(
                r"\b20\d{2}\b",
                lambda match: match.group(0) if int(match.group(0)) >= current_year else " ",
                text
            )
            text = re.sub(r"\s+", " ", text).strip()
    return text.strip()

def _query_terms(query: str) -> List[str]:
    import re
    raw_terms = re.findall(r"[a-z0-9][a-z0-9.+#/-]*", (query or "").lower())
    terms = []
    for term in raw_terms:
        clean = term.strip("-_/")
        if len(clean) < 2 or clean in SEARCH_STOPWORDS:
            continue
        terms.append(clean)
    return terms

def _search_result_score(query: str, item: Dict[str, str]) -> float:
    import urllib.parse
    title = _clean_search_text(item.get("title", "")).lower()
    snippet = _clean_search_text(item.get("snippet", "")).lower()
    url = (item.get("url") or "").lower()
    provider = (item.get("provider") or "").lower()
    haystack = f"{title} {snippet} {url}"
    terms = _query_terms(query)
    score = 0.0
    for term in terms:
        if term in title:
            score += 4.0
        if term in snippet:
            score += 2.0
        if term in url:
            score += 1.0
    if provider in {"google news", "bing news"}:
        score += 4.0
    elif provider in {"google custom search", "duckduckgo"}:
        score += 2.0

    query_lower = (query or "").lower()
    if ("google i/o" in query_lower or "google io" in query_lower) and ("google i/o" in haystack or "google io" in haystack):
        score += 12.0
    if "may 19" in query_lower and ("may 19" in haystack or "may 19," in haystack):
        score += 10.0
    if "gemini" in query_lower and "gemini" in haystack:
        score += 5.0
    if "android" in query_lower and "android" in haystack:
        score += 4.0

    is_latest_claude_query = (
        _has_latest_search_intent(query_lower)
        and ("claude" in query_lower or "anthropic" in query_lower)
    )
    if is_latest_claude_query:
        if "claude opus 4.7" in haystack or "opus 4.7" in haystack:
            score += 30.0
        if "anthropic.com" in url:
            score += 12.0
        if "april 16, 2026" in haystack or "apr 16, 2026" in haystack:
            score += 8.0
        if any(old_name in haystack for old_name in ["claude 3.5", "claude 3.7", "haiku 4.5", "sonnet 4.5", "opus 4.5"]):
            score -= 14.0

    company_aliases = {
        "openai": ["openai", "gpt"],
        "anthropic": ["anthropic", "claude"],
        "claude": ["anthropic", "claude"],
        "meta": ["meta", "llama"],
        "llama": ["meta", "llama"],
        "microsoft": ["microsoft", "copilot", "azure"],
        "nvidia": ["nvidia", "blackwell", "dgx"],
        "xai": ["xai", "grok"],
        "grok": ["xai", "grok"],
        "mistral": ["mistral"],
        "deepseek": ["deepseek"],
        "google": ["google", "gemini", "deepmind"],
    }
    for entity, aliases in company_aliases.items():
        if entity in query_lower:
            if any(alias in haystack for alias in aliases):
                score += 8.0
            else:
                score -= 18.0

    if any(word in query_lower for word in ["release", "released", "announcement", "announced", "latest", "recent"]):
        if any(word in haystack for word in ["release", "released", "announcement", "announced", "launch", "unveil", "introduced", "debut"]):
            score += 4.0
        if any(word in haystack for word in ["guide", "selection guide", "how to choose", "plugin", "plugins", "best free", "buyers guide"]):
            score -= 8.0
    try:
        domain = urllib.parse.urlparse(item.get("url", "")).netloc.lower().removeprefix("www.")
        if domain in GENERIC_SEARCH_DOMAINS:
            score -= 20.0
    except Exception:
        pass
    if title in {"google", "latest", "major"}:
        score -= 10.0
    if "ai" in (query or "").lower() and not any(word in haystack for word in ["ai", "artificial intelligence", "gemini", "openai", "claude", "llama", "model"]):
        score -= 8.0
    return score

def _rank_search_results(query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(results, key=lambda item: _search_result_score(query, item), reverse=True)

def _has_relevant_search_result(query: str, results: List[Dict[str, str]], threshold: float = 1.0) -> bool:
    if not results:
        return False
    ranked = _rank_search_results(query, results)
    return _search_result_score(query, ranked[0]) >= threshold

def _dedupe_search_results(results: List[Dict[str, str]], limit: int = 5) -> List[Dict[str, str]]:
    deduped = []
    seen_urls = set()
    seen_titles = set()
    for item in results:
        title = _clean_search_text(item.get("title", ""))
        url = (item.get("url") or "").strip()
        snippet = _clean_search_text(item.get("snippet", ""))
        provider = item.get("provider", "Web")
        if not title and not url:
            continue
        title_key = title.lower()
        url_key = url.lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append({
            "title": title or url,
            "url": url or "No URL returned",
            "snippet": snippet or "No summary snippet returned.",
            "provider": provider
        })
        if len(deduped) >= limit:
            break
    return deduped

def _format_search_results(
    query: str,
    results: List[Dict[str, str]],
    label: str,
    notes: Optional[List[str]] = None,
    ranking_query: Optional[str] = None,
) -> str:
    score_query = ranking_query or query
    ranked = _rank_search_results(score_query, results)
    relevant_ranked = [item for item in ranked if _search_result_score(score_query, item) >= 1.0]
    if relevant_ranked:
        ranked = relevant_ranked
    results = _dedupe_search_results(ranked, limit=5)
    if not results:
        note_text = f"\nSearch notes: {'; '.join(notes)}" if notes else ""
        return f"No live search results were found for '{query}'.{note_text}"

    lines = [f"Live {label} results for: {query}"]
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"Result {idx}: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Summary: {item['snippet']}\n"
            f"Provider: {item['provider']}"
        )
    if notes:
        lines.append(f"Search notes: {'; '.join(notes)}")
    return "\n\n".join(lines)

def _looks_current_or_news_query(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "today", "latest", "news", "release", "released", "announced", "announcement",
        "2026", "google i/o", "google io", "i/o", "keynote", "model"
    ])

def _expand_search_query(query: str) -> str:
    q = _normalize_search_query(query)
    lower = q.lower()
    additions = []
    current_year = str(_current_search_year())
    if _looks_current_or_news_query(lower) and current_year not in lower:
        additions.append(current_year)
    is_google_io = "google i/o" in lower or "google io" in lower or "i/o" in lower
    asks_when = any(word in lower for word in ["when", "date", "schedule", "time"])
    asks_topics = any(word in lower for word in ["topic", "topics", "expect", "announcement", "announcements", "ai"])
    if is_google_io:
        additions.append("May 19 2026")
        if asks_topics or not asks_when:
            additions.extend(["Gemini", "Android"])
    if "google" in lower and ("recent" in lower or "latest" in lower or "ai" in lower):
        additions.extend(["Gemini", "Google AI", "Google Blog"])

    company_terms = {
        "openai": ["GPT"],
        "anthropic": ["Claude"],
        "claude": ["Anthropic"],
        "meta": ["Llama"],
        "llama": ["Meta"],
        "microsoft": ["Copilot", "Azure AI"],
        "nvidia": ["Blackwell", "DGX"],
        "xai": ["Grok"],
        "grok": ["xAI"],
        "mistral": ["Mistral AI"],
        "deepseek": ["DeepSeek AI"],
    }
    matched_company = False
    for company, company_additions in company_terms.items():
        if company in lower:
            matched_company = True
            additions.extend(company_additions)
    if _has_latest_search_intent(lower) and ("claude" in lower or "anthropic" in lower):
        additions.extend(["Claude Opus 4.7", "Anthropic official", "April 2026"])
    if ("ai" in lower or "artificial intelligence" in lower or "model" in lower) and not matched_company:
        additions.extend(["AI model"])
    for addition in additions:
        if addition.lower() not in lower:
            q += f" {addition}"
    return q

def _decode_duckduckgo_url(url: str) -> str:
    import urllib.parse
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/l/"):
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("uddg"):
            return urllib.parse.unquote(params["uddg"][0])
    return url

def _duckduckgo_html_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    import urllib.parse
    from html.parser import HTMLParser

    class DuckDuckGoParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.current_result = None
            self.capture = None
            self.capture_tag = None
            self.buffer = []
            self.snippet_target = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            classes = attrs.get("class", "")
            if tag == "a" and "result__a" in classes:
                self.current_result = {
                    "title": "",
                    "url": _decode_duckduckgo_url(attrs.get("href", "")),
                    "snippet": "",
                    "provider": "DuckDuckGo"
                }
                self.capture = "title"
                self.capture_tag = tag
                self.buffer = []
            elif "result__snippet" in classes:
                self.snippet_target = self.current_result or (self.results[-1] if self.results else None)
                if self.snippet_target is not None:
                    self.capture = "snippet"
                    self.capture_tag = tag
                    self.buffer = []

        def handle_data(self, data):
            if self.capture:
                self.buffer.append(data)

        def handle_endtag(self, tag):
            if not self.capture or tag != self.capture_tag:
                return
            text = _clean_search_text(" ".join(self.buffer))
            if self.capture == "title" and self.current_result:
                self.current_result["title"] = text
                self.results.append(self.current_result)
                self.current_result = None
            elif self.capture == "snippet" and self.snippet_target:
                self.snippet_target["snippet"] = text
            self.capture = None
            self.capture_tag = None
            self.buffer = []
            self.snippet_target = None

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    response = requests.get(url, params={"q": query}, headers=headers, timeout=12)
    response.raise_for_status()
    parser = DuckDuckGoParser()
    parser.feed(response.text)
    return _dedupe_search_results(parser.results, limit=limit)

def _duckduckgo_instant_answer_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests

    response = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        },
        timeout=12
    )
    response.raise_for_status()
    data = response.json()
    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or data.get("Redirect") or "No URL returned",
            "snippet": data.get("AbstractText", ""),
            "provider": "DuckDuckGo Instant Answer"
        })

    def add_related(items):
        for item in items or []:
            if len(results) >= limit:
                return
            if "Topics" in item:
                add_related(item.get("Topics"))
                continue
            text = item.get("Text", "")
            url = item.get("FirstURL", "")
            if text or url:
                title = text.split(" - ", 1)[0] if text else url
                results.append({
                    "title": title,
                    "url": url or "No URL returned",
                    "snippet": text,
                    "provider": "DuckDuckGo Instant Answer"
                })

    add_related(data.get("RelatedTopics"))
    return _dedupe_search_results(results, limit=limit)

def _google_custom_search_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import os
    import requests

    api_key = (
        os.getenv("GOOGLE_CSE_API_KEY")
        or os.getenv("GOOGLE_SEARCH_API_KEY")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    )
    engine_id = (
        os.getenv("GOOGLE_CSE_ID")
        or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CX")
    )
    if not api_key or not engine_id:
        return []

    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": min(limit, 10)
        },
        timeout=12
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("items", [])[:limit]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "provider": "Google Custom Search"
        })
    return _dedupe_search_results(results, limit=limit)

def _google_html_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    import urllib.parse
    from html.parser import HTMLParser

    class GoogleParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.anchor_url = None
            self.capture_title = False
            self.buffer = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "a":
                href = attrs.get("href", "")
                parsed_url = ""
                if href.startswith("/url?"):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    parsed_url = params.get("q", [""])[0]
                elif href.startswith("http") and "google." not in urllib.parse.urlparse(href).netloc:
                    parsed_url = href
                self.anchor_url = parsed_url or None
            elif tag == "h3" and self.anchor_url:
                self.capture_title = True
                self.buffer = []

        def handle_data(self, data):
            if self.capture_title:
                self.buffer.append(data)

        def handle_endtag(self, tag):
            if tag == "h3" and self.capture_title:
                title = _clean_search_text(" ".join(self.buffer))
                if title and self.anchor_url:
                    self.results.append({
                        "title": title,
                        "url": self.anchor_url,
                        "snippet": "Google web result. No summary snippet returned by the public HTML endpoint.",
                        "provider": "Google Search"
                    })
                self.capture_title = False
                self.buffer = []
            elif tag == "a":
                self.anchor_url = None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    response = requests.get(
        "https://www.google.com/search",
        params={"q": query, "num": limit, "hl": "en", "safe": "active", "pws": "0"},
        headers=headers,
        timeout=12
    )
    response.raise_for_status()
    parser = GoogleParser()
    parser.feed(response.text)
    return _dedupe_search_results(parser.results, limit=limit)

def _google_news_rss_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    import xml.etree.ElementTree as ET

    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        description = item.findtext("description", default="")
        results.append({
            "title": title,
            "url": link,
            "snippet": _clean_search_text(description),
            "provider": "Google News"
        })
    return _dedupe_search_results(results, limit=limit)

def _decode_bing_url(url: str) -> str:
    import base64
    import urllib.parse

    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    if "url" in params:
        return urllib.parse.unquote(params["url"][0])
    if "bing.com" not in parsed.netloc or not parsed.path.startswith("/ck/a"):
        return url
    encoded = params.get("u", [""])[0]
    if encoded.startswith("a1"):
        encoded = encoded[2:]
    if not encoded:
        return url
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:
        return url

def _bing_news_rss_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    import xml.etree.ElementTree as ET

    response = requests.get(
        "https://www.bing.com/news/search",
        params={"format": "rss", "q": query, "count": limit},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall(".//item")[:limit]:
        results.append({
            "title": item.findtext("title", default=""),
            "url": _decode_bing_url(item.findtext("link", default="")),
            "snippet": item.findtext("description", default=""),
            "provider": "Bing News"
        })
    return _dedupe_search_results(_rank_search_results(query, results), limit=limit)

def _bing_rss_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    import xml.etree.ElementTree as ET

    response = requests.get(
        "https://www.bing.com/search",
        params={"format": "rss", "q": query, "count": limit},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall(".//item")[:limit]:
        results.append({
            "title": item.findtext("title", default=""),
            "url": item.findtext("link", default=""),
            "snippet": item.findtext("description", default=""),
            "provider": "Bing"
        })
    return _dedupe_search_results(results, limit=limit)

def _bing_html_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    import requests
    from html.parser import HTMLParser

    class BingParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.current = None
            self.depth = 0
            self.capture = None
            self.capture_tag = None
            self.buffer = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            classes = attrs.get("class", "")
            if self.current is None and tag == "li" and "b_algo" in classes:
                self.current = {
                    "title": "",
                    "url": "",
                    "snippet": "",
                    "provider": "Bing"
                }
                self.depth = 1
                return
            if self.current is None:
                return

            self.depth += 1
            if tag == "a" and not self.current["url"]:
                href = attrs.get("href", "")
                if href.startswith("http"):
                    self.current["url"] = _decode_bing_url(href)
                    self.capture = "title"
                    self.capture_tag = tag
                    self.buffer = []
            elif tag == "p" and not self.current["snippet"]:
                self.capture = "snippet"
                self.capture_tag = tag
                self.buffer = []

        def handle_data(self, data):
            if self.capture:
                self.buffer.append(data)

        def handle_endtag(self, tag):
            if self.current is None:
                return
            if self.capture and tag == self.capture_tag:
                text = _clean_search_text(" ".join(self.buffer))
                if self.capture == "title":
                    self.current["title"] = text
                elif self.capture == "snippet":
                    self.current["snippet"] = text
                self.capture = None
                self.capture_tag = None
                self.buffer = []

            self.depth -= 1
            if tag == "li" and self.depth <= 0:
                if self.current.get("title") or self.current.get("url"):
                    self.results.append(self.current)
                self.current = None
                self.depth = 0

    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": limit},
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        },
        timeout=12
    )
    response.raise_for_status()
    parser = BingParser()
    parser.feed(response.text)
    return _dedupe_search_results(parser.results, limit=limit)

def search_web(query: str) -> str:
    """Searches the live internet using news-first relevance ranking, then web fallbacks.

    Args:
        query: Specific search query (e.g. 'Gemini model release news', 'crude oil prices today').
    """
    log_to_sophie_brain("TOOL_CALL", f"search_web(query='{query}')")
    query = (query or "").strip()
    if not query:
        res_str = "Search failed: query cannot be empty."
        log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str}")
        record_tool_execution("search_web", {"query": query}, res_str)
        return res_str

    try:
        research = research_query(query, mode="auto")
        res_str = format_research_result(research)
        log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned structured research: {res_str[:300]}...")
        record_tool_execution("search_web", {"query": query}, res_str)
        return res_str
    except Exception as e:
        log_to_sophie_brain("TOOL_RESPONSE", f"search_web structured research failed, using legacy fallback: {e}")

    original_query = query
    query = _expand_search_query(query)
    notes = []
    if query != original_query:
        notes.append(f"Expanded query from '{original_query}' to improve relevance")

    if _looks_current_or_news_query(query):
        news_results = []
        for label, search_func in [
            ("Google News RSS", _google_news_rss_results),
            ("Bing News RSS", _bing_news_rss_results),
        ]:
            try:
                results = search_func(query)
                if results:
                    news_results.extend(results)
                else:
                    notes.append(f"{label} returned no results")
            except Exception as e:
                notes.append(f"{label} failed: {str(e)}")
        if _has_relevant_search_result(query, news_results):
            res_str = _format_search_results(original_query, news_results, "news search", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
            record_tool_execution("search_web", {"query": original_query}, res_str)
            return res_str
        notes.append("News search returned no relevant results")

    try:
        results = _bing_rss_results(query)
        if _has_relevant_search_result(query, results):
            res_str = _format_search_results(original_query, results, "Bing web search", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
            record_tool_execution("search_web", {"query": original_query}, res_str)
            return res_str
        notes.append("Bing RSS returned no relevant parsed results")
    except Exception as e:
        notes.append(f"Bing RSS failed: {str(e)}")

    try:
        results = _bing_html_results(query)
        if _has_relevant_search_result(query, results):
            notes.append("Bing RSS failed, so Bing HTML search was used")
            res_str = _format_search_results(original_query, results, "Bing web search", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
            record_tool_execution("search_web", {"query": original_query}, res_str)
            return res_str
        notes.append("Bing HTML returned no relevant parsed results")
    except Exception as e:
        notes.append(f"Bing HTML failed: {str(e)}")

    try:
        results = _duckduckgo_html_results(query)
        if _has_relevant_search_result(query, results):
            notes.append("Bing providers failed, so DuckDuckGo HTML search was used")
            res_str = _format_search_results(original_query, results, "DuckDuckGo web search", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
            record_tool_execution("search_web", {"query": original_query}, res_str)
            return res_str
        notes.append("DuckDuckGo HTML returned no relevant parsed results")
    except Exception as e:
        notes.append(f"DuckDuckGo HTML failed: {str(e)}")

    try:
        results = _duckduckgo_instant_answer_results(query)
        if _has_relevant_search_result(query, results):
            notes.append("Bing and DuckDuckGo HTML failed, so DuckDuckGo Instant Answer was used")
            res_str = _format_search_results(original_query, results, "DuckDuckGo instant answer", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
            record_tool_execution("search_web", {"query": original_query}, res_str)
            return res_str
        notes.append("DuckDuckGo Instant Answer returned no results")
    except Exception as e:
        notes.append(f"DuckDuckGo Instant Answer failed: {str(e)}")

    google_fallback = google_search(original_query)
    if google_fallback.startswith("No live search results") or google_fallback.startswith("Google search failed"):
        res_str = f"Bing and DuckDuckGo search failed, and Google fallback did not find results.\n{google_fallback}\nSearch notes: {'; '.join(notes)}"
    else:
        res_str = f"Bing and DuckDuckGo search failed or returned no results, so Google fallback was used.\n\n{google_fallback}\n\nSearch notes: {'; '.join(notes)}"
    log_to_sophie_brain("TOOL_RESPONSE", f"search_web returned: {res_str[:300]}...")
    record_tool_execution("search_web", {"query": original_query}, res_str)
    return res_str

def google_search(query: str) -> str:
    """Searches Google/news-backed live results with relevance filtering and fallbacks.

    Args:
        query: Specific search query (e.g. 'Gemini model release news', 'crude oil prices today').
    """
    log_to_sophie_brain("TOOL_CALL", f"google_search(query='{query}')")
    query = (query or "").strip()
    if not query:
        res_str = "Google search failed: query cannot be empty."
        log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str}")
        record_tool_execution("google_search", {"query": query}, res_str)
        return res_str

    try:
        research = research_query(query, mode="auto")
        res_str = format_research_result(research)
        log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned structured research: {res_str[:300]}...")
        record_tool_execution("google_search", {"query": query}, res_str)
        return res_str
    except Exception as e:
        log_to_sophie_brain("TOOL_RESPONSE", f"google_search structured research failed, using legacy fallback: {e}")

    original_query = query
    query = _expand_search_query(query)
    notes = []
    if query != original_query:
        notes.append(f"Expanded query from '{original_query}' to improve relevance")

    if _looks_current_or_news_query(query):
        news_results = []
        for label, search_func in [
            ("Google News RSS", _google_news_rss_results),
            ("Bing News RSS", _bing_news_rss_results),
        ]:
            try:
                results = search_func(query)
                if results:
                    news_results.extend(results)
                else:
                    notes.append(f"{label} returned no results")
            except Exception as e:
                notes.append(f"{label} failed: {str(e)}")
        if _has_relevant_search_result(query, news_results):
            res_str = _format_search_results(original_query, news_results, "Google/Bing News", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str[:300]}...")
            record_tool_execution("google_search", {"query": original_query}, res_str)
            return res_str
        notes.append("News search returned no relevant results")

    search_attempts = []
    search_attempts.extend([
        ("Google Custom Search", _google_custom_search_results),
        ("Google public web search", _google_html_results),
        ("Bing fallback", _bing_rss_results),
    ])

    for label, search_func in search_attempts:
        try:
            results = search_func(query)
            if _has_relevant_search_result(query, results):
                res_str = _format_search_results(original_query, results, label, notes, ranking_query=query)
                log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str[:300]}...")
                record_tool_execution("google_search", {"query": original_query}, res_str)
                return res_str
            notes.append(f"{label} returned no relevant results")
        except Exception as e:
            notes.append(f"{label} failed: {str(e)}")

    try:
        results = _bing_html_results(query)
        if _has_relevant_search_result(query, results):
            notes.append("Google providers failed, so Bing fallback was used")
            res_str = _format_search_results(original_query, results, "Bing fallback", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str[:300]}...")
            record_tool_execution("google_search", {"query": original_query}, res_str)
            return res_str
        notes.append("Bing fallback returned no relevant parsed results")
    except Exception as e:
        notes.append(f"Bing fallback failed: {str(e)}")

    try:
        results = _duckduckgo_html_results(query)
        if _has_relevant_search_result(query, results):
            notes.append("Google and Bing providers failed, so DuckDuckGo fallback was used")
            res_str = _format_search_results(original_query, results, "DuckDuckGo fallback", notes, ranking_query=query)
            log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str[:300]}...")
            record_tool_execution("google_search", {"query": original_query}, res_str)
            return res_str
        notes.append("DuckDuckGo fallback returned no relevant results")
    except Exception as e:
        notes.append(f"DuckDuckGo fallback failed: {str(e)}")

    res_str = f"No live search results were found for '{original_query}'. Search notes: {'; '.join(notes)}"
    log_to_sophie_brain("TOOL_RESPONSE", f"google_search returned: {res_str}")
    record_tool_execution("google_search", {"query": original_query}, res_str)
    return res_str

def research_latest(query: str) -> str:
    """Runs trusted-source current research for latest/recent facts and model releases.

    Args:
        query: Specific latest/current question to verify across trusted sources.
    """
    log_to_sophie_brain("TOOL_CALL", f"research_latest(query='{query}')")
    query = (query or "").strip()
    if not query:
        res_str = "Research failed: query cannot be empty."
        log_to_sophie_brain("TOOL_RESPONSE", f"research_latest returned: {res_str}")
        record_tool_execution("research_latest", {"query": query}, res_str)
        return res_str
    try:
        research = research_query(query, mode="latest")
        res_str = format_research_result(research)
    except Exception as e:
        res_str = f"Research failed for '{query}': {e}"
    log_to_sophie_brain("TOOL_RESPONSE", f"research_latest returned: {res_str[:300]}...")
    record_tool_execution("research_latest", {"query": query}, res_str)
    return res_str

def research_benchmarks(query: str) -> str:
    """Runs trusted-source benchmark research across official leaderboards and aggregators.

    Args:
        query: Benchmark/stat/ranking question to verify with source trust scoring.
    """
    log_to_sophie_brain("TOOL_CALL", f"research_benchmarks(query='{query}')")
    query = (query or "").strip()
    if not query:
        res_str = "Benchmark research failed: query cannot be empty."
        log_to_sophie_brain("TOOL_RESPONSE", f"research_benchmarks returned: {res_str}")
        record_tool_execution("research_benchmarks", {"query": query}, res_str)
        return res_str
    try:
        research = research_query(query, mode="benchmarks")
        res_str = format_research_result(research)
    except Exception as e:
        res_str = f"Benchmark research failed for '{query}': {e}"
    log_to_sophie_brain("TOOL_RESPONSE", f"research_benchmarks returned: {res_str[:300]}...")
    record_tool_execution("research_benchmarks", {"query": query}, res_str)
    return res_str

def get_exchange_rate(base: str, quote: str, date: str = "today") -> str:
    """Gets a current exchange rate through Sophie's free-first finance pipeline.

    Args:
        base: Base currency code or name (e.g. 'USD', 'dollar', 'EUR').
        quote: Quote currency code or name (e.g. 'PKR', 'pakistani rupee').
        date: Requested date, usually 'today'. Historical dates are best-effort only.
    """
    base = (base or "").strip().upper()
    quote = (quote or "").strip().upper()
    date = (date or "today").strip()
    log_to_sophie_brain("TOOL_CALL", f"get_exchange_rate(base='{base}', quote='{quote}', date='{date}')")
    if not base or not quote:
        res_str = "Exchange-rate lookup failed: base and quote currencies are required."
        log_to_sophie_brain("TOOL_RESPONSE", f"get_exchange_rate returned: {res_str}")
        record_tool_execution("get_exchange_rate", {"base": base, "quote": quote, "date": date}, res_str)
        return res_str
    try:
        research = research_query(f"{base} to {quote} exchange rate {date}", mode="exchange_rate")
        res_str = format_research_result(research)
    except Exception as e:
        res_str = f"Exchange-rate lookup failed for {base}/{quote}: {e}"
    log_to_sophie_brain("TOOL_RESPONSE", f"get_exchange_rate returned: {res_str[:300]}...")
    record_tool_execution("get_exchange_rate", {"base": base, "quote": quote, "date": date}, res_str)
    return res_str

def create_repeating_task(task_name: str, query: str, interval_hours: float, target_number: str) -> str:
    """Saves and schedules a recurring task to research a topic periodically and send summary updates directly to the user.
    
    Args:
        task_name: Unique, descriptive name of the repeating task (e.g. 'gemini_news_hour', 'usd_to_pkr_daily').
        query: Specific topic query to search and summary-alert (e.g. 'Gemini models updates', 'Pakistan dollar price exchange rate').
        interval_hours: Task execution frequency in hours (e.g. 1.0 for hourly alerts, 24.0 for daily alerts).
        target_number: Contact WhatsApp number to push alerts to (e.g. '447123456789' or '12345').
    """
    from app.database import db
    success = db.save_repeating_task(task_name, query, interval_hours, target_number)
    if success:
        res = f"Successfully registered background scheduled task '{task_name}' to search for '{query}' every {interval_hours} hours."
    else:
        res = f"Failed to record background scheduled task '{task_name}' in the database."
    record_tool_execution("create_repeating_task", {
        "task_name": task_name,
        "query": query,
        "interval_hours": interval_hours,
        "target_number": target_number
    }, res)
    return res

def add_calendar_event(title: str, description: str, date_time: str) -> str:
    """Adds a scheduled event or appointment to the user's database calendar.
    
    Args:
        title: Title of the calendar event (e.g. 'Dentist appointment', 'Meeting with team').
        description: Additional details or notes about the event.
        date_time: ISO-8601 formatted date and time string (e.g. '2026-05-20T14:30:00').
    """
    from app.database import db
    success = db.save_calendar_event(title, description, date_time)
    if success:
        res = f"Successfully added calendar event '{title}' scheduled for {date_time}."
    else:
        res = f"Failed to record calendar event '{title}' in database."
    record_tool_execution("add_calendar_event", {
        "title": title,
        "description": description,
        "date_time": date_time
    }, res)
    return res

def list_calendar_events() -> str:
    """Retrieves all scheduled calendar events and appointments."""
    from app.database import db
    events = db.get_calendar_events()
    if not events:
        res = "Your calendar is currently clear. No events scheduled!"
    else:
        lines = []
        for e in events:
            lines.append(f"- Event: {e['title']} | Date: {e['date_time']} | Notes: {e['description']}")
        res = "Current Calendar Events:\n" + "\n".join(lines)
    record_tool_execution("list_calendar_events", {}, res)
    return res

def view_local_file(file_path: str) -> str:
    """Reads and displays the content of a local text file within the workspace.
    
    Args:
        file_path: The name or relative path of the file to view (e.g. 'backend/package.json').
    """
    import os
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        target_path = os.path.join(base_dir, file_path)
        if not os.path.exists(target_path):
            res = f"Error: File '{file_path}' does not exist."
            record_tool_execution("view_local_file", {"file_path": file_path}, res)
            return res
        if os.path.isdir(target_path):
            res = f"Error: '{file_path}' is a directory. Use list directories instead."
            record_tool_execution("view_local_file", {"file_path": file_path}, res)
            return res
            
        with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(5000) # limit to first 5000 chars
            res = f"--- Content of {file_path} (first 5000 chars) ---\n{content}"
            record_tool_execution("view_local_file", {"file_path": file_path}, res)
            return res
    except Exception as e:
        res = f"Failed to read file '{file_path}': {str(e)}"
        record_tool_execution("view_local_file", {"file_path": file_path}, res)
        return res

def write_local_file(file_path: str, content: str) -> str:
    """Creates a new text file or overwrites an existing one with new content in the workspace.
    
    Args:
        file_path: The name or relative path of the file to write to.
        content: The text content to write.
    """
    import os
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        target_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        res = f"Successfully wrote file contents to '{file_path}'."
        record_tool_execution("write_local_file", {"file_path": file_path, "content": content}, res)
        return res
    except Exception as e:
        res = f"Failed to write file '{file_path}': {str(e)}"
        record_tool_execution("write_local_file", {"file_path": file_path, "content": content}, res)
        return res

def ingest_text_document(title: str, content: str, source_type: str = "url", source_url: str = "") -> str:
    """Ingests pasted text into Sophie's vector database and local document library.

    Args:
        title: Document title.
        content: Text content to chunk and embed.
        source_type: Namespace/source type, such as url, pdf, notion, gmail, whatsapp.
        source_url: Optional source URL or provenance string.
    """
    log_to_sophie_brain("TOOL_CALL", f"ingest_text_document(title='{title}', source_type='{source_type}')")
    res = ingest_text_document_impl(title, content, source_type, source_url or None)
    log_to_sophie_brain("TOOL_RESPONSE", f"ingest_text_document returned: {res[:300]}...")
    record_tool_execution("ingest_text_document", {
        "title": title,
        "source_type": source_type,
        "source_url": source_url
    }, res)
    return res

def ingest_local_file(file_path: str, title: str = "", source_type: str = "pdf") -> str:
    """Reads a local text/PDF/DOCX file, extracts text, and ingests it into vector memory.

    Args:
        file_path: Workspace-relative or absolute local path.
        title: Optional document title.
        source_type: Namespace/source type, usually pdf or whatsapp.
    """
    log_to_sophie_brain("TOOL_CALL", f"ingest_local_file(file_path='{file_path}', title='{title}')")
    res = ingest_local_file_impl(file_path, title or None, source_type)
    log_to_sophie_brain("TOOL_RESPONSE", f"ingest_local_file returned: {res[:300]}...")
    record_tool_execution("ingest_local_file", {
        "file_path": file_path,
        "title": title,
        "source_type": source_type
    }, res)
    return res

def ingest_url_document(url: str, title: str = "", source_type: str = "pdf") -> str:
    """Downloads a URL/PDF, extracts text, and ingests it into vector memory.

    Args:
        url: HTTP/HTTPS URL to download and ingest.
        title: Optional document title.
        source_type: Namespace/source type, usually pdf or url.
    """
    log_to_sophie_brain("TOOL_CALL", f"ingest_url_document(url='{url}', title='{title}')")
    res = ingest_url_document_impl(url, title or None, source_type)
    log_to_sophie_brain("TOOL_RESPONSE", f"ingest_url_document returned: {res[:300]}...")
    record_tool_execution("ingest_url_document", {
        "url": url,
        "title": title,
        "source_type": source_type
    }, res)
    return res

def list_ingested_documents() -> str:
    """Lists documents already saved into Sophie's local library/vector memory."""
    log_to_sophie_brain("TOOL_CALL", "list_ingested_documents()")
    res = list_ingested_documents_impl()
    log_to_sophie_brain("TOOL_RESPONSE", f"list_ingested_documents returned: {res[:300]}...")
    record_tool_execution("list_ingested_documents", {}, res)
    return res

# --- CUSTOM SUB-AGENT LIFECYCLE MANAGEMENT TOOLS ---

def create_agent(agent_name: str, system_prompt: str) -> str:
    """Creates and registers a custom sub-agent with a persistent role and system prompt instruction.
    
    Args:
        agent_name: Unique, descriptive name of the agent (e.g. 'Coder', 'Researcher', 'Poet').
        system_prompt: Detailed prompt instruction specifying the agent's behavior, style, and rules.
    """
    from app.database import db
    success = db.save_custom_agent(agent_name, system_prompt)
    if success:
        res = f"Successfully created and registered sub-agent '{agent_name}'."
    else:
        res = f"Failed to register sub-agent '{agent_name}' in the database."
    record_tool_execution("create_agent", {"agent_name": agent_name, "system_prompt": system_prompt}, res)
    return res

def list_agents() -> str:
    """Retrieves and lists all currently registered custom sub-agents."""
    from app.database import db
    agents = db.get_all_custom_agents()
    if not agents:
        res = "No custom sub-agents are currently registered. You can create one using the create_agent tool!"
    else:
        lines = []
        for idx, a in enumerate(agents):
            lines.append(f"{idx+1}. Agent: '{a['name']}' | Created At: {a['created_at']}\nPrompt: {a['system_prompt'][:120]}...")
        res = "Registered Custom Sub-Agents:\n" + "\n".join(lines)
    record_tool_execution("list_agents", {}, res)
    return res

def delete_agent(agent_name: str) -> str:
    """Deletes and removes a registered custom sub-agent by its name.
    
    Args:
        agent_name: The name of the sub-agent to delete.
    """
    approval = create_pending_approval(
        sender=ACTIVE_RUN_SENDER,
        action_type="delete_agent",
        command_line=f"delete_agent {agent_name}",
        target_path=agent_name,
    )
    res = approval["message"]
    record_tool_execution("delete_agent", {"agent_name": agent_name}, res)
    return res

def call_agent(agent_name: str, task: str) -> str:
    """Delegates a specific task or query to a registered custom sub-agent and retrieves their deep report.
    
    Args:
        agent_name: The name of the sub-agent to call (must be one of the registered sub-agents).
        task: The task description or query you want this specific agent to perform or answer.
    """
    log_to_sophie_brain("TOOL_CALL", f"call_agent(agent_name='{agent_name}', task='{task}')")
    from app.database import db
    
    agents = db.get_all_custom_agents()
    agent_data = None
    for a in agents:
        if a["name"].lower() == agent_name.lower():
            agent_data = a
            break
            
    if not agent_data:
        registered_names = [a["name"] for a in agents]
        res_str = f"Error: Custom sub-agent '{agent_name}' is not registered. Registered agents: {registered_names}"
        log_to_sophie_brain("TOOL_RESPONSE", f"call_agent returned: {res_str}")
        record_tool_execution("call_agent", {"agent_name": agent_name, "task": task}, res_str)
        return res_str
        
    try:
        agent_prompt = f"""
        You are a dedicated custom sub-agent named '{agent_data['name']}'.
        Your specific instructions and persona rules:
        {agent_data['system_prompt']}
        
        You are performing a task delegated to you by Sophie (the primary orchestrator).
        
        TASK: {task}
        
        Please execute the task thoroughly and provide a structured, detailed, and clear report of your findings/work.
        """
        if LLM_PROVIDER == "openrouter":
            report = openrouter_chat(agent_prompt, task, user_message=task, requires_tools=False)
        else:
            import google.generativeai as genai
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
            sub_model = genai.GenerativeModel(GEMINI_MODEL)
            response = sub_model.generate_content(agent_prompt)
            report = response.text

        res_str = f"--- Sub-Agent '{agent_data['name']}' Report ---\n{report}"
        log_to_sophie_brain("TOOL_RESPONSE", f"call_agent successfully completed. Report size: {len(res_str)} chars")
        record_tool_execution("call_agent", {"agent_name": agent_name, "task": task}, res_str)
        return res_str
    except Exception as e:
        res_str = f"Failed to execute sub-agent '{agent_name}': {str(e)}"
        log_to_sophie_brain("TOOL_RESPONSE", f"call_agent returned error: {res_str}")
        record_tool_execution("call_agent", {"agent_name": agent_name, "task": task}, res_str)
        return res_str

def clean_xml_from_text(text: str) -> str:
    """Strips XML tags like <ouput>, <thinking>, <user_output>, <tools_call> from text content to prevent context contamination."""
    import re
    if not text:
        return ""
    # 1. Try to extract user_output tag content
    match = re.search(r"<user_output>([\s\S]*?)</user_output>", text, re.I)
    if match:
        return match.group(1).strip()
    # 2. Otherwise, clean all other tags completely
    cleaned = text
    cleaned = re.sub(r"<thinking>[\s\S]*?</thinking>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<tools_call>[\s\S]*?</tools_call>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<ouput>([\s\S]*?)</ouput>", r"\1", cleaned, flags=re.I)
    cleaned = re.sub(r"</?(ouput|user_output|tools_call|thinking)>", "", cleaned, flags=re.I)
    return cleaned.strip()


def parse_strict_json_response(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        data = json.loads(text)
        response_text = data.get("response", "")
        if not response_text:
            response_text = f"<ouput><user_output>{clean_xml_from_text(text)}</user_output><tools_call>None</tools_call></ouput>"
        return {
            "reasoning": data.get("reasoning", "Model response generated."),
            "response": response_text,
            "confidence_score": float(data.get("confidence_score", 85.0)),
            "citations": data.get("citations", []),
            "raw_text": text,
        }
    except Exception:
        safe_text = clean_xml_from_text(text)
        return {
            "reasoning": "Model returned plain text instead of strict JSON.",
            "response": f"<ouput><user_output>{safe_text}</user_output><tools_call>None</tools_call></ouput>",
            "confidence_score": 70.0,
            "citations": [],
            "raw_text": text,
        }


def _actual_tools_summary() -> str:
    if not ACTIVE_RUN_TOOLS:
        return "None"
    summaries = []
    for tool_run in ACTIVE_RUN_TOOLS:
        tool_name = tool_run.get("tool", "unknown_tool")
        args = tool_run.get("arguments") or {}
        if tool_name == "sophie_tool":
            action = args.get("action", "unknown_action")
            summaries.append(f"sophie_tool(action={action})")
        else:
            summaries.append(f"{tool_name}({json.dumps(args, ensure_ascii=True)})")
    return "; ".join(summaries)


def _replace_user_output(response_text: str, user_output: str) -> str:
    if re.search(r"<user_output>[\s\S]*?</user_output>", response_text, re.I):
        return re.sub(
            r"<user_output>[\s\S]*?</user_output>",
            f"<user_output>{user_output}</user_output>",
            response_text,
            flags=re.I,
        )
    return f"<ouput><user_output>{user_output}</user_output><tools_call>{_actual_tools_summary()}</tools_call></ouput>"


def response_claims_unexecuted_tools(response_text: str) -> bool:
    if not response_text or ACTIVE_RUN_TOOLS:
        return False
    tool_match = re.search(r"<tools_call>([\s\S]*?)</tools_call>", response_text, re.I)
    claimed_tools = (tool_match.group(1).strip() if tool_match else "")
    if claimed_tools and claimed_tools.lower() not in {"none", "n/a", "no", "no tools"}:
        return True
    user_output = clean_xml_from_text(response_text)
    return bool(re.search(r"\b(checked|searched|looked up|verified|latest|current|today|now|real-time|live)\b", user_output, re.I))


def maybe_recover_missing_tools(decision: Dict[str, Any], user_message: str = "") -> str:
    if ACTIVE_RUN_TOOLS:
        return ""
    if not (decision or {}).get("needs_tool"):
        return ""
    if not (decision or {}).get("tool_plan"):
        return ""
    log_to_sophie_brain("HONESTY_RECOVERY", "Model implied tool use before any runtime tool execution; running deterministic BrainDecision plan.")
    return execute_brain_tool_plan(decision, user_message=user_message)


def enforce_honest_tool_report(response_text: str) -> str:
    """Ensure Sophie cannot claim tool calls that did not actually run."""
    if not response_text:
        return response_text

    actual_summary = _actual_tools_summary()
    tool_match = re.search(r"<tools_call>([\s\S]*?)</tools_call>", response_text, re.I)
    claimed_tools = (tool_match.group(1).strip() if tool_match else "")
    claimed_real_tool = claimed_tools and claimed_tools.lower() not in {"none", "n/a", "no", "no tools"}

    if claimed_real_tool and actual_summary == "None":
        log_to_sophie_brain(
            "HONESTY_GUARD",
            f"Model claimed tools without execution. claimed={claimed_tools[:160]}"
        )
        user_match = re.search(r"<user_output>([\s\S]*?)</user_output>", response_text, re.I)
        user_output = user_match.group(1).strip() if user_match else clean_xml_from_text(response_text)
        risky_live_claim = re.search(
            r"\b(checked|searched|looked up|verified|latest|current|today|now|real-time|live)\b",
            user_output,
            re.I,
        )
        if risky_live_claim:
            user_output = (
                "I need to be honest: I did not actually get verified tool data for that, "
                "so I should not present it as confirmed. Please ask again and I will use the right tool if available."
            )
            response_text = _replace_user_output(response_text, user_output)

    if tool_match:
        response_text = re.sub(
            r"<tools_call>[\s\S]*?</tools_call>",
            f"<tools_call>{actual_summary}</tools_call>",
            response_text,
            flags=re.I,
        )
    else:
        response_text = f"{response_text}<tools_call>{actual_summary}</tools_call>"
    return response_text

# Define State Structure
class AgentGraphState(TypedDict):
    triggered_event: Optional[Dict[str, Any]]
    messages: List[Dict[str, str]]
    retrieved_docs: List[Dict[str, Any]]
    retrieved_memories: List[Dict[str, Any]]
    thinkbox: Dict[str, Any]
    is_urgent: bool
    reasoning: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    action_taken: Optional[str]
    response: str
    task_type: str
    sender: Optional[str]

