from __future__ import annotations

import json
import re
from html import escape
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from app.database import db
from app.brain.memory_curator import has_sensitive_secret, review_and_save_memories
from app.brain.time_tools import current_time_reply, get_current_time_data, get_current_time_results
from app.brain.tool_decorators import with_logging, with_retry, with_timeout
from app.brain.response_cache import response_cache


WEATHER_API_KEY = "e6b90bde4e554ef5878113900261805"


@dataclass
class ToolFieldSpec:
    name: str
    description: str = ""
    field_type: str = "string"
    aliases: List[str] = field(default_factory=list)
    default: Any = None


@dataclass
class ToolSpec:
    action: str
    category: str
    description: str
    required_fields: List[ToolFieldSpec] = field(default_factory=list)
    optional_fields: List[ToolFieldSpec] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    safety_level: str = "safe_read"
    verification_rule: str = ""
    response_contract: str = "Returns text or JSON describing the result."


@dataclass
class ToolCallRequest:
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ToolValidationResult:
    ok: bool
    action: str
    original_action: str
    normalized_payload: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    accepted_aliases: Dict[str, List[str]] = field(default_factory=dict)
    example: Dict[str, Any] = field(default_factory=dict)
    manual: str = ""
    safety_level: str = "unknown"


@dataclass
class ToolExecutionResult:
    action: str
    payload: Dict[str, Any]
    validation: ToolValidationResult
    response: str
    evidence_quality: str = "unknown"


def _field(name: str, aliases: Optional[List[str]] = None, description: str = "", field_type: str = "string", default: Any = None) -> ToolFieldSpec:
    return ToolFieldSpec(name=name, aliases=aliases or [], description=description, field_type=field_type, default=default)


def _spec(
    action: str,
    category: str,
    description: str,
    required: Optional[List[ToolFieldSpec]] = None,
    optional: Optional[List[ToolFieldSpec]] = None,
    aliases: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    safety_level: str = "safe_read",
    verification_rule: str = "",
    response_contract: str = "Returns text or JSON describing the result.",
) -> ToolSpec:
    return ToolSpec(
        action=action,
        category=category,
        description=description,
        required_fields=required or [],
        optional_fields=optional or [],
        aliases=aliases or [],
        examples=examples or [],
        safety_level=safety_level,
        verification_rule=verification_rule,
        response_contract=response_contract,
    )


TOOL_SPECS: Dict[str, ToolSpec] = {}


def _register_specs(*specs: ToolSpec) -> None:
    for spec in specs:
        TOOL_SPECS[spec.action] = spec


_register_specs(
    _spec("tools.discover", "tools", "List available Sophie gateway actions.", optional=[_field("category")], aliases=["tools.list", "tool_list", "get_tool_list"], examples=[{"category": "search"}], response_contract="Plain-text grouped action list or category manuals."),
    _spec("tools.manual", "tools", "Fetch exact manuals and schemas for actions or a category.", optional=[_field("actions", field_type="list"), _field("category")], aliases=["tools.get_manual", "tool_manual", "get_tool_manual"], examples=[{"actions": ["weather.forecast"]}, {"category": "search"}], response_contract="Readable schemas generated from ToolSpec."),
    _spec("tools.call", "tools", "Nested helper that calls another action after validation.", required=[_field("action")], optional=[_field("payload", field_type="object"), _field("payload_json")], aliases=["call_tool"], examples=[{"action": "weather.forecast", "payload": {"location": "Lahore", "days": 1}}], response_contract="Response from the nested action."),
    _spec("context.lookup", "context", "Build a context packet from exact user input, recent chat, RAG documents, and memories.", required=[_field("query", ["q", "text"])], optional=[_field("recent_limit", field_type="integer", default=7), _field("doc_limit", field_type="integer", default=4), _field("memory_limit", field_type="integer", default=5)], examples=[{"query": "what did I ask earlier", "recent_limit": 7}], response_contract="Tagged context packet; candidate data must be judged for relevance."),
    _spec("search.web", "search", "General web search for broad facts that need internet evidence.", required=[_field("query", ["q", "search", "text"])], examples=[{"query": "US envoys Islamabad peace talks"}], response_contract="STRUCTURED_RESEARCH_RESULT or summarized search evidence."),
    _spec("search.latest", "search", "Trusted latest/current research for today/latest/recent/news/releases/current public facts.", required=[_field("query", ["q", "search", "text"])], examples=[{"query": "latest shooting mosque San Diego"}], response_contract="STRUCTURED_RESEARCH_RESULT with answerable, claims, citations, warnings, trust_score."),
    _spec("search.benchmarks", "search", "Benchmark/leaderboard research for rankings and scores.", required=[_field("query", ["q", "search", "text"])], examples=[{"query": "latest LMArena top model benchmark"}], response_contract="STRUCTURED_RESEARCH_RESULT with benchmark claims/citations."),
    _spec("finance.exchange_rate", "finance", "Currency exchange rate.", required=[_field("base", ["from", "source_currency"]), _field("quote", ["to", "target_currency"])], optional=[_field("date", default="today")], examples=[{"base": "USD", "quote": "PKR", "date": "today"}], response_contract="JSON/text containing exchange-rate claim and source/date."),
    _spec("weather.forecast", "weather", "Weather forecast.", required=[_field("location", ["city", "place", "query", "q"])], optional=[_field("days", field_type="integer", default=1)], examples=[{"location": "London", "days": 1}], response_contract="Weather JSON/text with current conditions/forecast."),
    _spec("time.current", "time", "Current local time by place, multiple places, or UTC offset.", optional=[_field("location", ["timezone", "query", "q", "place"])], examples=[{"location": "Pakistan"}, {"query": "time in America and China"}], response_contract="JSON with reply and time results."),
    _spec("wiki.lookup", "knowledge", "Wikipedia summary lookup.", required=[_field("query", ["q", "topic", "text"])], optional=[_field("language", ["lang"], default="en")], examples=[{"query": "Isaac Newton", "language": "en"}], response_contract="Short encyclopedia summary or not-found message."),
    _spec("history.check", "knowledge", "Historical fact lookup using Wikipedia summary/search.", required=[_field("query", ["q", "topic", "text"])], optional=[_field("language", ["lang"], default="en")], examples=[{"query": "Battle of Plassey", "language": "en"}], response_contract="Historical summary or not-found message."),
    _spec("nutrition.lookup_food", "nutrition", "Nutrition estimate lookup.", required=[_field("food")], examples=[{"food": "banana"}], response_contract="JSON nutrition estimate."),
    _spec("nutrition.log_food", "nutrition", "Save eaten food.", required=[_field("food", ["food_name"])], optional=[_field("quantity"), _field("calories", field_type="number"), _field("protein_g", field_type="number"), _field("carbs_g", field_type="number"), _field("fat_g", field_type="number")], examples=[{"food": "banana", "quantity": "1 medium", "calories": 105}], safety_level="safe_write", verification_rule="Return save confirmation.", response_contract="Save confirmation or failure."),
    _spec("nutrition.daily_summary", "nutrition", "Summarize nutrition logs.", optional=[_field("date")], examples=[{"date": "2026-05-19"}], response_contract="JSON totals and items."),
    _spec("schedule.reminder", "schedule", "Create one-time WhatsApp reminder.", required=[_field("message", ["text", "body"]), _field("due_at", ["datetime", "date_time", "time", "when"])], optional=[_field("title")], examples=[{"title": "Call Ali", "message": "Call Ali", "due_at": "2026-05-19T17:00:00"}], safety_level="safe_write", verification_rule="Return scheduled confirmation or missing datetime error.", response_contract="Reminder confirmation or validation error."),
    _spec("schedule.recurring_task", "schedule", "Create recurring research alert.", required=[_field("task_name"), _field("query")], optional=[_field("interval_hours", field_type="number", default=24), _field("target_number")], examples=[{"task_name": "ai_news", "query": "AI news", "interval_hours": 24}], safety_level="safe_write", response_contract="Recurring task confirmation."),
    _spec("schedule.list", "schedule", "List reminders, calendar events, and recurring tasks.", examples=[{}], response_contract="JSON schedule listing."),
    _spec("memory.recall", "memory", "Recall long-term memory.", required=[_field("query", ["q", "text"])], optional=[_field("limit", field_type="integer", default=5)], examples=[{"query": "user preference", "limit": 5}], response_contract="JSON list of memories."),
    _spec("memory.save", "memory", "Save durable non-secret memory.", required=[_field("content")], optional=[_field("type", default="semantic")], examples=[{"content": "User prefers concise replies", "type": "semantic"}], safety_level="safe_write", verification_rule="Refuse secrets; return memory id on save.", response_contract="Memory id or refusal."),
    _spec("memory.review_chat", "memory", "Review recent chat and save durable facts.", required=[_field("user_message")], optional=[_field("assistant_response")], examples=[{"user_message": "remember I like short replies", "assistant_response": "Saved."}], safety_level="safe_write", response_contract="JSON memory review result."),
    _spec("chat.history", "memory", "Fetch recent chat messages for continuity.", optional=[_field("limit", field_type="integer", default=7)], examples=[{"limit": 7}], response_contract="JSON recent chat messages."),
    _spec("documents.ingest_text", "documents", "Save text into RAG.", required=[_field("content", ["text", "body"])], optional=[_field("title", default="Untitled text"), _field("source_type", default="whatsapp"), _field("source_url")], examples=[{"title": "Manual", "content": "tool instructions", "source_type": "whatsapp"}], safety_level="safe_write", verification_rule="Verify with documents.list after ingestion when user asks to save/ingest.", response_contract="Ingestion confirmation."),
    _spec("documents.ingest_file", "documents", "Save local file into RAG.", required=[_field("file_path", ["path", "filename", "file"])], optional=[_field("title"), _field("source_type", default="pdf")], examples=[{"file_path": "C:/Users/pak7/Desktop/file.pdf", "title": "Report"}], safety_level="safe_write", verification_rule="Verify with documents.list after ingestion.", response_contract="Ingestion confirmation."),
    _spec("documents.ingest_url", "documents", "Download/extract URL/PDF into RAG.", required=[_field("url", ["link", "href"])], optional=[_field("title"), _field("source_type", default="pdf")], examples=[{"url": "https://example.com/file.pdf", "title": "Manual"}], safety_level="safe_write", verification_rule="Verify with documents.list after ingestion.", response_contract="Ingestion confirmation."),
    _spec("documents.list", "documents", "List ingested documents.", examples=[{}], response_contract="Plain text or JSON document list."),
    _spec("files.read", "files", "Read workspace/local file.", required=[_field("file_path", ["path", "filename", "file"])], examples=[{"file_path": "backend/app/main.py"}]),
    _spec("files.write", "files", "Write workspace file.", required=[_field("file_path", ["path", "filename", "file"]), _field("content", ["text", "body"])], examples=[{"file_path": "notes.txt", "content": "hello"}], safety_level="safe_write", verification_rule="Verify after writing by reading/checking content.", response_contract="Write confirmation or error."),
    _spec("system.command", "system", "Run PowerShell command.", required=[_field("command_line", ["command", "cmd", "powershell"])], examples=[{"command_line": "Test-Path -LiteralPath $HOME"}], safety_level="approval_if_destructive", verification_rule="Destructive commands must create approval and stop; empty commands are blocked.", response_contract="Command stdout/stderr/status or approval request."),
    _spec("agents.create", "agents", "Create or update custom advisory sub-agent.", required=[_field("agent_name", ["name", "agent"]), _field("system_prompt", ["prompt", "instructions"])], examples=[{"agent_name": "Researcher", "system_prompt": "You research carefully."}], safety_level="safe_write", response_contract="Agent save confirmation."),
    _spec("agents.list", "agents", "List custom advisory sub-agents.", examples=[{}], response_contract="JSON/list of agents."),
    _spec("agents.call", "agents", "Call a custom advisory sub-agent.", required=[_field("agent_name", ["name", "agent"]), _field("task", ["query", "message", "prompt"])], examples=[{"agent_name": "Researcher", "task": "Find risks in this plan."}], response_contract="Agent response JSON/text."),
    _spec("agents.delete", "agents", "Delete a custom sub-agent.", required=[_field("agent_name", ["name", "agent"])], examples=[{"agent_name": "Researcher"}], safety_level="approval_required", response_contract="Approval request or delete result."),
    _spec("workspace.run_python", "workspace", "Execute Python code inside the isolated ./jarvis_workspace sandbox.", required=[_field("code", ["script", "python_code"])], examples=[{"code": "print('Hello from sandbox!')"}], safety_level="safe_read", response_contract="Subprocess stdout/stderr/exit_code."),
    _spec("workspace.write_file", "workspace", "Write content to a file inside the isolated ./jarvis_workspace sandbox.", required=[_field("file_path", ["path", "filename", "file"]), _field("content", ["text", "body"])], examples=[{"file_path": "script.py", "content": "print('hello')"}], safety_level="safe_write", response_contract="File write status."),
    _spec("workspace.read_file", "workspace", "Read the content of a file inside the isolated ./jarvis_workspace sandbox.", required=[_field("file_path", ["path", "filename", "file"])], examples=[{"file_path": "output.txt"}], response_contract="File content or error message."),
    _spec("workspace.list_files", "workspace", "List all files recursively inside the isolated ./jarvis_workspace sandbox.", examples=[{}], response_contract="JSON array of files inside workspace."),
    _spec("trust.add_site", "knowledge", "Add a website to your personal trusted sources list for future searches. Example: 'I trust pakwheels.com for fuel prices'.", required=[_field("domain", ["url", "website", "site"])], optional=[_field("trust_score", field_type="integer", default=70), _field("topics", field_type="list"), _field("note")], examples=[{"domain": "pakwheels.com", "trust_score": 80, "topics": ["petrol", "fuel"], "note": "Trusted for Pakistan fuel prices"}], safety_level="safe_write", response_contract="Confirmation that site was added."),
    _spec("trust.remove_site", "knowledge", "Remove a website from your trusted list.", required=[_field("domain", ["url", "site"])], safety_level="safe_write", response_contract="Confirmation of removal."),
    _spec("trust.list_sites", "knowledge", "Show all websites you have personally added as trusted.", examples=[{}], response_contract="List of user-trusted domains with topics and trust scores."),
    _spec("trigger.create_threshold", "triggers", "Create a price/value alert. JARVIS will check automatically and message you when the value crosses your threshold. No AI needed for each check — runs silently in background.", required=[_field("name", ["title", "label"]), _field("metric", ["watch", "track"]), _field("threshold_value", ["value", "price", "level"], field_type="number"), _field("direction", ["when"], default="above")], optional=[_field("threshold_pct", field_type="number")], examples=[{"name": "Dollar Alert", "metric": "usd_pkr", "threshold_value": 290, "direction": "above"}], safety_level="safe_write", response_contract="Trigger created confirmation with ID."),
    _spec("trigger.create_keyword_monitor", "triggers", "Create a keyword/news monitor. JARVIS will watch for news matching your keywords and alert you automatically when something new appears.", required=[_field("name", ["title", "label"]), _field("query", ["search", "watch_for"])], optional=[_field("keywords", field_type="list"), _field("check_interval_hours", field_type="number", default=4)], examples=[{"name": "Gemini News", "query": "Google Gemini new model release", "keywords": ["gemini", "release", "launch"]}], safety_level="safe_write", response_contract="Monitor created confirmation with ID."),
    _spec("trigger.create_recurring", "triggers", "Create a scheduled recurring update. JARVIS will run a search on a schedule and send you a digest automatically.", required=[_field("name", ["title", "label"]), _field("query", ["search", "topic"]), _field("schedule", ["interval", "when"])], examples=[{"name": "Morning News", "query": "Pakistan news today", "schedule": "daily_9am"}], safety_level="safe_write", response_contract="Recurring task created with schedule confirmation."),
    _spec("trigger.list", "triggers", "List all your active triggers, monitors, and schedules.", examples=[{}], response_contract="List of all triggers with type, name, status, last run."),
    _spec("trigger.pause", "triggers", "Pause a trigger by ID or name so it stops checking.", required=[_field("trigger_id_or_name", ["id", "name", "trigger"])], safety_level="safe_write", response_contract="Pause confirmation."),
    _spec("trigger.resume", "triggers", "Resume a paused trigger.", required=[_field("trigger_id_or_name", ["id", "name", "trigger"])], safety_level="safe_write", response_contract="Resume confirmation."),
    _spec("trigger.delete", "triggers", "Delete a trigger permanently.", required=[_field("trigger_id_or_name", ["id", "name", "trigger"])], safety_level="approval_required", response_contract="Delete confirmation."),
)


ACTION_ALIAS_TO_CANONICAL = {
    alias: action
    for action, spec in TOOL_SPECS.items()
    for alias in spec.aliases
}


def canonical_action_name(action: str) -> str:
    action = (action or "").strip()
    return ACTION_ALIAS_TO_CANONICAL.get(action, action)


def _manual_text_from_spec(spec: ToolSpec) -> str:
    required = [field_spec.name for field_spec in spec.required_fields]
    optional = [field_spec.name for field_spec in spec.optional_fields]
    alias_parts = []
    for field_spec in spec.required_fields + spec.optional_fields:
        if field_spec.aliases:
            alias_parts.append(f"{field_spec.name}: {', '.join(field_spec.aliases)}")
    example = spec.examples[0] if spec.examples else {}
    return (
        f"{spec.action}: {spec.description}\n"
        f"  category: {spec.category}; safety: {spec.safety_level}\n"
        f"  required_fields: {required or []}\n"
        f"  optional_fields: {optional or []}\n"
        f"  accepted_aliases: {alias_parts or []}\n"
        f"  example_payload: {json.dumps(example, ensure_ascii=True)}\n"
        f"  verification_rule: {spec.verification_rule or 'Use the returned result honestly.'}\n"
        f"  response_contract: {spec.response_contract}"
    )


def tool_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    return {action: asdict(spec) for action, spec in TOOL_SPECS.items()}


ACTION_MANUALS: Dict[str, Dict[str, str]] = {
    "tools.discover": {
        "category": "tools",
        "manual": "List available Sophie gateway actions. Use when unsure which action exists.",
    },
    "tool_list": {
        "category": "tools",
        "manual": "Alias for tools.discover. Payload: {'category':'search'} optional. Use to list available tools/actions before choosing one.",
    },
    "tools.manual": {
        "category": "tools",
        "manual": "Fetch manuals for actions or categories. Payload: {'actions':['finance.exchange_rate']} or {'category':'search'}.",
    },
    "get_tool_manual": {
        "category": "tools",
        "manual": "Alias for tools.manual. Payload: {'actions':['weather.forecast']} or {'category':'weather'}. Use before calling a tool when arguments are unclear.",
    },
    "tools.call": {
        "category": "tools",
        "manual": "Optional nested call helper. Payload: {'action':'weather.forecast','payload':{'location':'Lahore','days':1}}. Prefer calling the target action directly with sophie_tool when possible.",
    },
    "context.lookup": {
        "category": "context",
        "manual": "Build a pre-context packet from the exact user input, recent chat, similar RAG documents, and long-term memories. Payload: {'query':'user message','recent_limit':7,'doc_limit':4,'memory_limit':5}. Treat returned context as candidate data that may be irrelevant.",
    },
    "search.web": {
        "category": "search",
        "manual": "General web search. Payload: {'query':'specific search query'}. Use for broad facts that need internet evidence.",
    },
    "search.latest": {
        "category": "search",
        "manual": "Trusted latest/current research. Payload: {'query':'latest current fact'}. Use for today/latest/recent/news/releases.",
    },
    "search.benchmarks": {
        "category": "search",
        "manual": "Benchmark/leaderboard research. Payload: {'query':'top model benchmark stats'}. Use for rankings and scores.",
    },
    "finance.exchange_rate": {
        "category": "finance",
        "manual": "Currency exchange rate. Payload: {'base':'USD','quote':'PKR','date':'today'}.",
    },
    "weather.forecast": {
        "category": "weather",
        "manual": "Weather forecast. Payload: {'location':'London','days':1}. Days may be 1 or 7.",
    },
    "time.current": {
        "category": "time",
        "manual": "Current local time by place, multiple places, or UTC offset. Payload: {'location':'Pakistan'} or {'query':'time in America and China'} or {'location':'UTC+05:00'}. Use for 'PK time', 'current time', and clock questions.",
    },
    "wiki.lookup": {
        "category": "knowledge",
        "manual": "Wikipedia summary lookup. Payload: {'query':'Isaac Newton','language':'en'}.",
    },
    "history.check": {
        "category": "knowledge",
        "manual": "Historical fact lookup using Wikipedia summary/search. Payload: {'query':'Battle of Plassey','language':'en'}.",
    },
    "nutrition.lookup_food": {
        "category": "nutrition",
        "manual": "Nutrition lookup. Payload: {'food':'banana'}. Uses local/free lookup and returns estimate.",
    },
    "nutrition.log_food": {
        "category": "nutrition",
        "manual": "Save eaten food. Payload: {'food':'banana','quantity':'1 medium','calories':105}.",
    },
    "nutrition.daily_summary": {
        "category": "nutrition",
        "manual": "Summarize today's nutrition logs. Payload: {'date':'YYYY-MM-DD'}; date optional.",
    },
    "schedule.reminder": {
        "category": "schedule",
        "manual": "One-time WhatsApp reminder. Payload: {'title':'Call Ali','message':'Call Ali','due_at':'2026-05-19T17:00:00'}.",
    },
    "schedule.recurring_task": {
        "category": "schedule",
        "manual": "Recurring research alert. Payload: {'task_name':'ai_news','query':'AI news','interval_hours':24}.",
    },
    "schedule.list": {
        "category": "schedule",
        "manual": "List reminders, calendar events, and recurring tasks. Payload: {}.",
    },
    "memory.recall": {
        "category": "memory",
        "manual": "Recall long-term memory. Payload: {'query':'user preference','limit':5}.",
    },
    "memory.save": {
        "category": "memory",
        "manual": "Save durable non-secret memory. Payload: {'content':'User prefers concise replies','type':'semantic'}.",
    },
    "memory.review_chat": {
        "category": "memory",
        "manual": "Review recent chat and save durable facts. Payload: {'user_message':'...','assistant_response':'...'}.",
    },
    "chat.history": {
        "category": "memory",
        "manual": "Fetch recent chat messages for continuity. Payload: {'limit':7}; allowed 5-10.",
    },
    "documents.ingest_text": {
        "category": "documents",
        "manual": "Save text into RAG. Payload: {'title':'Manual','content':'...','source_type':'whatsapp'}.",
    },
    "documents.ingest_file": {
        "category": "documents",
        "manual": "Save local file into RAG. Payload: {'file_path':'path','title':'optional','source_type':'pdf'}.",
    },
    "documents.ingest_url": {
        "category": "documents",
        "manual": "Download/extract URL/PDF into RAG. Payload: {'url':'https://...','title':'optional'}.",
    },
    "documents.list": {
        "category": "documents",
        "manual": "List ingested documents. Payload: {}.",
    },
    "files.read": {
        "category": "files",
        "manual": "Read workspace file. Payload: {'file_path':'backend/app/main.py'}.",
    },
    "files.write": {
        "category": "files",
        "manual": "Write workspace file. Payload: {'file_path':'notes.txt','content':'...'}; verify after writing.",
    },
    "system.command": {
        "category": "system",
        "manual": "Run PowerShell command. Payload: {'command_line':'dir'}; destructive commands require approval.",
    },
    "agents.create": {
        "category": "agents",
        "manual": "Create or update a custom advisory sub-agent. Payload: {'agent_name':'Researcher','system_prompt':'...'}; sub-agents are advisory only.",
    },
    "agents.list": {
        "category": "agents",
        "manual": "List custom advisory sub-agents. Payload: {}.",
    },
    "agents.call": {
        "category": "agents",
        "manual": "Call a custom sub-agent. Payload: {'agent_name':'Researcher','task':'...'}; advisory only.",
    },
    "agents.delete": {
        "category": "agents",
        "manual": "Delete a custom sub-agent. Payload: {'agent_name':'Researcher'}; requires approval.",
    },
}


ACTION_MANUALS = {
    action: {"category": spec.category, "manual": _manual_text_from_spec(spec)}
    for action, spec in TOOL_SPECS.items()
}


KEYWORD_CATEGORIES = [
    ("finance", ["usd", "pkr", "dollar", "rupee", "forex", "exchange rate", "currency"]),
    ("time", ["time now", "current time", "pk time", "pakistan time", "pkt", "clock", "time in", "time of", "time for", "dubai", "china", "america", "usa", "united states"]),
    ("weather", ["weather", "forecast", "rain", "temperature", "humidity"]),
    ("schedule", ["remind", "reminder", "schedule", "calendar", "every day", "tomorrow", "at "]),
    ("memory", ["remember", "what did i say", "before", "my first", "same thing", "that", "preference"]),
    ("context", ["what did i say", "my first", "same thing", "that", "before", "similar", "related"]),
    ("documents", ["pdf", "document", "file", "ingest", "save this", "read this url"]),
    ("nutrition", ["calorie", "calories", "protein", "carbs", "fat", "ate", "nutrition", "food"]),
    ("knowledge", ["wikipedia", "history", "who was", "when was", "historical"]),
    ("search", ["latest", "today", "news", "recent", "release", "benchmark", "leaderboard"]),
    ("files", ["read file", "write file", "open file"]),
    ("system", ["run command", "powershell", "terminal", "cmd"]),
    ("agents", ["agent", "sub agent", "sub-agent", "create agent", "call agent"]),
    ("workspace", ["workspace", "sandbox", "python script", "run code", "evaluate script", "run python"]),
]


def parse_payload(payload_json: str) -> Dict[str, Any]:
    if not payload_json:
        return {}
    if isinstance(payload_json, dict):
        return payload_json
    try:
        parsed = json.loads(payload_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _accepted_aliases_for_spec(spec: ToolSpec) -> Dict[str, List[str]]:
    return {
        field_spec.name: list(field_spec.aliases)
        for field_spec in spec.required_fields + spec.optional_fields
        if field_spec.aliases
    }


def _first_payload_value(payload: Dict[str, Any], field_spec: ToolFieldSpec) -> Any:
    keys = [field_spec.name] + list(field_spec.aliases)
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def validate_tool_call(action: str, payload_json: Any = "{}") -> ToolValidationResult:
    original_action = (action or "").strip()
    canonical_action = canonical_action_name(original_action)
    payload = parse_payload(payload_json)
    spec = TOOL_SPECS.get(canonical_action)
    if not spec:
        return ToolValidationResult(
            ok=False,
            action=canonical_action,
            original_action=original_action,
            normalized_payload=payload,
            errors=[f"Unknown gateway action '{original_action}'."],
            manual="Call tools.discover or tools.manual for the exact action.",
            accepted_aliases={},
            example={},
            safety_level="unknown",
        )

    normalized: Dict[str, Any] = {}
    missing: List[str] = []
    errors: List[str] = []
    for field_spec in spec.required_fields:
        value = _first_payload_value(payload, field_spec)
        if value in (None, ""):
            missing.append(field_spec.name)
        else:
            normalized[field_spec.name] = value

    for field_spec in spec.optional_fields:
        value = _first_payload_value(payload, field_spec)
        if value not in (None, ""):
            normalized[field_spec.name] = value
        elif field_spec.default is not None:
            normalized[field_spec.name] = field_spec.default

    if canonical_action == "system.command":
        command_line = str(normalized.get("command_line") or "").strip()
        if not command_line:
            if "command_line" not in missing:
                missing.append("command_line")
        elif re.fullmatch(r"(?:uname(?:\s+-s)?|echo\s+\$HOME|pwd|cd|dir|ls)\s*", command_line, flags=re.I):
            errors.append("system.command refused: command is too generic for the user's request. Build a command that directly answers the user's requested action or ask for missing details.")

    if canonical_action == "files.write" and not str(normalized.get("content") or "").strip():
        errors.append("files.write refused: content is required and cannot be empty.")

    example = spec.examples[0] if spec.examples else {}
    all_errors = [f"{field_name} is required." for field_name in missing] + errors
    return ToolValidationResult(
        ok=not missing and not errors,
        action=canonical_action,
        original_action=original_action,
        normalized_payload=normalized,
        missing_fields=missing,
        errors=all_errors,
        accepted_aliases=_accepted_aliases_for_spec(spec),
        example=example,
        manual=_manual_text_from_spec(spec),
        safety_level=spec.safety_level,
    )


def validation_result_to_response(result: ToolValidationResult) -> str:
    payload = {
        "ok": result.ok,
        "action": result.action,
        "original_action": result.original_action,
        "missing_fields": result.missing_fields,
        "errors": result.errors,
        "accepted_aliases": result.accepted_aliases,
        "example_payload": result.example,
        "manual": result.manual,
        "safety_level": result.safety_level,
    }
    return "TOOL_VALIDATION_ERROR\n" + json.dumps(payload, ensure_ascii=True, indent=2)


def capability_index() -> str:
    categories: Dict[str, List[str]] = {}
    for action, spec in TOOL_SPECS.items():
        categories.setdefault(spec.category, []).append(action)
    return "\n".join(
        f"- {category}: {', '.join(actions)}"
        for category, actions in sorted(categories.items())
    )


def capability_index_summary() -> str:
    """Returns a one-line category summary of tools to save context length."""
    categories = sorted(list(set(spec.category for spec in TOOL_SPECS.values())))
    return f"Available tool categories: {', '.join(categories)}. Use 'tools.discover' with a category name to list specific actions, or 'tools.manual' with specific action names to load full schemas."


def lazy_load_manual(actions: Optional[List[str]] = None, category: str = "") -> str:
    """Dynamically loads exact manuals and schemas only when explicitly requested by action name or category."""
    return manual_for(actions, category)


def select_relevant_manuals(user_message: str, limit: int = 6) -> str:
    q = (user_message or "").lower()
    selected_categories: List[str] = []
    for category, keywords in KEYWORD_CATEGORIES:
        if any(keyword in q for keyword in keywords):
            selected_categories.append(category)
    if not selected_categories:
        selected_categories = ["tools", "memory", "search"]
    selected_categories = list(dict.fromkeys(["tools"] + selected_categories))

    manuals = []
    for action, spec in TOOL_SPECS.items():
        if spec.category in selected_categories:
            manuals.append(_manual_text_from_spec(spec))
        if len(manuals) >= limit:
            break
    return "\n".join(manuals)


def discover_tools(category: str = "") -> str:
    if not category:
        return capability_index()
    actions = [
        _manual_text_from_spec(spec)
        for action, spec in TOOL_SPECS.items()
        if spec.category == category
    ]
    return "\n".join(actions) if actions else f"No tools found for category '{category}'."


def manual_for(actions: Optional[List[str]] = None, category: str = "") -> str:
    if category:
        return discover_tools(category)
    if isinstance(actions, str):
        actions = [actions]
    actions = actions or []
    lines = []
    for action in actions:
        canonical = canonical_action_name(str(action))
        spec = TOOL_SPECS.get(canonical)
        lines.append(_manual_text_from_spec(spec) if spec else f"{action}: unknown action")
    return "\n".join(lines) if lines else capability_index()


def _shorten_text(value: Any, max_chars: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _xml_text(value: Any, max_chars: int = 900) -> str:
    return escape(_shorten_text(value, max_chars), quote=False)


def _xml_attr(value: Any, max_chars: int = 120) -> str:
    return escape(_shorten_text(value, max_chars), quote=True)


def format_context_packet(
    user_input: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    docs: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format retrieved context as tagged candidate data, separate from the actual user input."""
    recent_messages = recent_messages or []
    docs = docs or []
    memories = memories or []

    lines = [
        '<context_packet relevance_note="candidate_context_may_be_irrelevant">',
        f"  <user_input>{_xml_text(user_input, 1800)}</user_input>",
        '  <recent_previous_messages relevance_note="chronological_chat_context">',
    ]
    if recent_messages:
        for idx, msg in enumerate(recent_messages[-10:], 1):
            role = _xml_attr(msg.get("role", "unknown"), 40)
            content = _xml_text(msg.get("content", msg.get("message", "")), 700)
            lines.append(f'    <message index="{idx}" role="{role}">{content}</message>')
    else:
        lines.append("    <none>No recent chat context was found.</none>")
    lines.append("  </recent_previous_messages>")

    lines.append('  <similar_database_results relevance_note="rag_candidates_verify_before_using">')
    if docs:
        for idx, doc in enumerate(docs[:6], 1):
            title = _xml_text(doc.get("title", "Untitled"), 160)
            source_type = _xml_attr(doc.get("source_type", ""), 80)
            source = _xml_text(doc.get("source", ""), 180)
            confidence = _xml_attr(doc.get("confidence", doc.get("score", "")), 40)
            text = _xml_text(doc.get("text", doc.get("content", "")), 1000)
            lines.extend([
                f'    <document index="{idx}" confidence="{confidence}" source_type="{source_type}">',
                f"      <title>{title}</title>",
                f"      <source>{source}</source>",
                f"      <text>{text}</text>",
                "    </document>",
            ])
    else:
        lines.append("    <none>No similar database documents were found.</none>")
    lines.append("  </similar_database_results>")

    lines.append('  <long_term_memory_results relevance_note="memory_candidates_do_not_override_user_input">')
    if memories:
        for idx, mem in enumerate(memories[:6], 1):
            mem_type = _xml_attr(mem.get("type", ""), 80)
            freshness = _xml_attr(mem.get("freshness", ""), 40)
            status = _xml_attr(mem.get("status", ""), 60)
            content = _xml_text(mem.get("content", ""), 900)
            lines.append(
                f'    <memory index="{idx}" type="{mem_type}" freshness="{freshness}" status="{status}">{content}</memory>'
            )
    else:
        lines.append("    <none>No related long-term memories were found.</none>")
    lines.append("  </long_term_memory_results>")
    lines.append("</context_packet>")
    return "\n".join(lines)


def build_context_packet(
    query: str,
    sender: str,
    recent_limit: int = 7,
    doc_limit: int = 4,
    memory_limit: int = 5,
) -> str:
    recent_limit = max(5, min(int(recent_limit or 7), 10))
    doc_limit = max(1, min(int(doc_limit or 4), 8))
    memory_limit = max(1, min(int(memory_limit or 5), 10))
    query = str(query or "").strip()

    recent_messages = db.get_chat_history(sender, limit=recent_limit)
    while (
        recent_messages
        and recent_messages[-1].get("role") == "user"
        and recent_messages[-1].get("content") == query
    ):
        recent_messages.pop()
    docs: List[Dict[str, Any]] = []
    memories: List[Dict[str, Any]] = []
    if query:
        try:
            from app.brain.doc_brain import doc_brain
            docs = doc_brain.query_knowledge(query, limit=doc_limit)
        except Exception as exc:
            docs = [{"title": "Doc lookup error", "source": "DocBrain", "source_type": "error", "text": str(exc), "confidence": 0}]
        try:
            from app.brain.memory_os import memory_os
            memories = memory_os.retrieve_memories(query, limit=memory_limit)
        except Exception as exc:
            memories = [{"type": "error", "content": str(exc), "freshness": 0, "status": "error"}]

    return format_context_packet(query, recent_messages=recent_messages, docs=docs, memories=memories)


@with_logging("wiki.lookup")
@with_retry(attempts=2)
@with_timeout(seconds=30)
def wikipedia_lookup(query: str, language: str = "en") -> str:
    language = re.sub(r"[^a-z-]", "", (language or "en").lower()) or "en"
    query = (query or "").strip()
    if not query:
        return "Wikipedia lookup failed: query is required."
    search = requests.get(
        f"https://{language}.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        },
        timeout=10,
        headers={"User-Agent": "SophieAssistant/1.0"},
    )
    search.raise_for_status()
    results = search.json().get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia result found for '{query}'."
    title = results[0]["title"]
    summary = requests.get(
        f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
        timeout=10,
        headers={"User-Agent": "SophieAssistant/1.0"},
    )
    summary.raise_for_status()
    data = summary.json()
    extract = data.get("extract") or "No summary extract returned."
    url = (data.get("content_urls") or {}).get("desktop", {}).get("page", "")
    return f"Wikipedia: {data.get('title', title)}\nSummary: {extract}\nURL: {url}"


LOCAL_NUTRITION = {
    "banana": {"calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4, "quantity": "1 medium"},
    "apple": {"calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3, "quantity": "1 medium"},
    "egg": {"calories": 78, "protein_g": 6.0, "carbs_g": 0.6, "fat_g": 5.0, "quantity": "1 large"},
    "rice": {"calories": 205, "protein_g": 4.3, "carbs_g": 45, "fat_g": 0.4, "quantity": "1 cup cooked"},
    "chicken breast": {"calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6, "quantity": "100g cooked"},
}


def nutrition_lookup(food: str) -> Dict[str, Any]:
    key = (food or "").strip().lower()
    if not key:
        return {"error": "food is required"}
    for name, values in LOCAL_NUTRITION.items():
        if name in key or key in name:
            return {"food": name, **values, "source": "local estimate"}
    try:
        res = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={"search_terms": food, "search_simple": 1, "action": "process", "json": 1, "page_size": 1},
            timeout=10,
            headers={"User-Agent": "SophieAssistant/1.0"},
        )
        res.raise_for_status()
        products = res.json().get("products", [])
        if products:
            product = products[0]
            nutrients = product.get("nutriments") or {}
            return {
                "food": product.get("product_name") or food,
                "quantity": "100g",
                "calories": float(nutrients.get("energy-kcal_100g") or 0),
                "protein_g": float(nutrients.get("proteins_100g") or 0),
                "carbs_g": float(nutrients.get("carbohydrates_100g") or 0),
                "fat_g": float(nutrients.get("fat_100g") or 0),
                "source": "Open Food Facts per 100g",
            }
    except Exception:
        pass
    return {"food": food, "error": "No nutrition estimate found."}


@with_logging("weather.forecast")
@with_retry(attempts=2)
@with_timeout(seconds=30)
def weather_forecast(location: str, days: int = 1) -> str:
    cache_key = response_cache.make_key("weather.forecast", {"location": location, "days": days})
    cached = response_cache.get(cache_key)
    if cached:
        print(f"[CacheHit] Returning cached weather forecast for key: {cache_key}")
        return cached

    days = 7 if int(days or 1) > 1 else 1
    res = requests.get(
        "http://api.weatherapi.com/v1/forecast.json",
        params={"key": WEATHER_API_KEY, "q": location, "days": days, "aqi": "no", "alerts": "no"},
        timeout=10,
    )
    data = res.json()
    if "error" in data:
        return f"Weather lookup failed for {location}: {data['error'].get('message', 'unknown error')}"
    loc = data.get("location", {})
    lines = [f"Weather forecast for {loc.get('name', location)}, {loc.get('country', '')}:"]
    for day in (data.get("forecast") or {}).get("forecastday", [])[:days]:
        info = day.get("day") or {}
        condition = (info.get("condition") or {}).get("text", "unknown")
        lines.append(
            f"- {day.get('date')}: {condition}, {info.get('mintemp_c')}C-{info.get('maxtemp_c')}C, rain chance {info.get('daily_chance_of_rain', 'unknown')}%"
        )
    out_res = "\n".join(lines)
    response_cache.set(cache_key, out_res)
    return out_res


def dispatch_tool_action(
    action: str,
    payload_json: str,
    context: Dict[str, Any],
    handlers: Dict[str, Callable[..., str]],
) -> str:
    validation = validate_tool_call(action, payload_json)
    if not validation.ok:
        return validation_result_to_response(validation)
    action = validation.action
    payload = dict(validation.normalized_payload)
    sender = context.get("sender") or "unknown_number"
    action = (action or "").strip()

    def first_value(*keys: str, default: Any = "") -> Any:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
        return default

    if action in {"tools.discover", "tools.list", "tool_list", "get_tool_list"}:
        return discover_tools(str(payload.get("category", "")))
    if action in {"tools.manual", "tools.get_manual", "tool_manual", "get_tool_manual"}:
        return manual_for(payload.get("actions") or [], str(payload.get("category", "")))
    if action in {"tools.call", "call_tool"}:
        nested_action = str(payload.get("action", "")).strip()
        if not nested_action:
            return "tools.call failed: payload.action is required."
        if nested_action in {"tools.call", "call_tool"}:
            return "tools.call failed: recursive tools.call is not allowed."
        nested_payload = payload.get("payload_json", payload.get("payload", {}))
        if isinstance(nested_payload, str):
            nested_payload_json = nested_payload
        else:
            nested_payload_json = json.dumps(nested_payload, ensure_ascii=True)
        return dispatch_tool_action(nested_action, nested_payload_json, context, handlers)

    if action == "search.web":
        return handlers["search_web"](str(first_value("query", "q", "search", "text")))
    if action == "search.latest":
        decorated = with_logging("search.latest")(with_retry(attempts=2)(with_timeout(seconds=30)(handlers["research_latest"])))
        return decorated(str(first_value("query", "q", "search", "text")))
    if action == "search.benchmarks":
        return handlers["research_benchmarks"](str(first_value("query", "q", "search", "text")))
    if action == "finance.exchange_rate":
        base = str(first_value("base", "from", "source_currency"))
        quote = str(first_value("quote", "to", "target_currency"))
        date = str(first_value("date", default="today"))
        cache_key = response_cache.make_key("finance.exchange_rate", {"base": base, "quote": quote, "date": date})
        cached = response_cache.get(cache_key)
        if cached:
            print(f"[CacheHit] Returning cached exchange rate for key: {cache_key}")
            return cached
        decorated = with_logging("finance.exchange_rate")(with_retry(attempts=2)(with_timeout(seconds=30)(handlers["get_exchange_rate"])))
        res = decorated(base, quote, date)
        if "Error" not in res and "failed" not in res:
            response_cache.set(cache_key, res)
        return res
    if action == "weather.forecast":
        return weather_forecast(str(first_value("location", "city", "place", "query", "q")), int(first_value("days", default=1) or 1))
    if action == "time.current":
        location = str(first_value("location", "timezone", "query", "q", "place"))
        results = get_current_time_results(location=location)
        data = get_current_time_data(location=location)
        return json.dumps(
            {"reply": current_time_reply(location=location), "results": results, **data},
            ensure_ascii=True,
            indent=2,
        )
    if action in {"wiki.lookup", "history.check"}:
        return wikipedia_lookup(str(first_value("query", "q", "topic", "text")), str(first_value("language", "lang", default="en")))

    if action == "nutrition.lookup_food":
        return json.dumps(nutrition_lookup(str(payload.get("food", ""))), ensure_ascii=True, indent=2)
    if action == "nutrition.log_food":
        food = str(payload.get("food", payload.get("food_name", "")))
        estimate = nutrition_lookup(food)
        calories = float(payload.get("calories", estimate.get("calories", 0)) or 0)
        protein = float(payload.get("protein_g", estimate.get("protein_g", 0)) or 0)
        carbs = float(payload.get("carbs_g", estimate.get("carbs_g", 0)) or 0)
        fat = float(payload.get("fat_g", estimate.get("fat_g", 0)) or 0)
        quantity = str(payload.get("quantity", estimate.get("quantity", "")))
        ok = db.save_nutrition_log(sender, food, quantity, calories, protein, carbs, fat)
        return f"Nutrition log saved: {quantity} {food}, {calories} calories." if ok else "Failed to save nutrition log."
    if action == "nutrition.daily_summary":
        date_prefix = str(payload.get("date") or datetime.utcnow().date().isoformat())
        logs = db.get_nutrition_logs(sender, date_prefix)
        if not logs:
            return f"No nutrition logs found for {date_prefix}."
        totals = {
            "calories": sum(float(item.get("calories") or 0) for item in logs),
            "protein_g": sum(float(item.get("protein_g") or 0) for item in logs),
            "carbs_g": sum(float(item.get("carbs_g") or 0) for item in logs),
            "fat_g": sum(float(item.get("fat_g") or 0) for item in logs),
        }
        return json.dumps({"date": date_prefix, "items": logs, "totals": totals}, ensure_ascii=True, indent=2)

    if action == "schedule.reminder":
        title = str(payload.get("title") or "Reminder")
        message = str(first_value("message", "text", "body", default=title))
        due_at = str(first_value("due_at", "datetime", "date_time", "time", "when"))
        if not due_at:
            return "Reminder failed: due_at ISO datetime is required."
        ok = db.save_reminder(sender, title, message, due_at)
        return f"Reminder scheduled for {due_at}: {message}" if ok else "Failed to save reminder."
    if action == "schedule.recurring_task":
        return handlers["create_repeating_task"](
            str(payload.get("task_name", "")),
            str(payload.get("query", "")),
            float(payload.get("interval_hours", 24)),
            str(payload.get("target_number") or sender),
        )
    if action == "schedule.list":
        reminders = db.list_reminders(sender)
        calendars = handlers["list_calendar_events"]()
        return json.dumps({"reminders": reminders, "calendar_events": calendars}, ensure_ascii=True, indent=2)

    if action == "chat.history":
        limit = max(5, min(int(payload.get("limit", 7) or 7), 10))
        return json.dumps(db.get_chat_history(sender, limit=limit), ensure_ascii=True, indent=2)
    if action == "context.lookup":
        return build_context_packet(
            query=str(first_value("query", "q", "text")),
            sender=sender,
            recent_limit=int(payload.get("recent_limit", 7) or 7),
            doc_limit=int(payload.get("doc_limit", 4) or 4),
            memory_limit=int(payload.get("memory_limit", 5) or 5),
        )
    if action == "memory.recall":
        from app.brain.memory_os import memory_os
        limit = max(1, min(int(payload.get("limit", 5) or 5), 10))
        memories = memory_os.retrieve_memories(str(first_value("query", "q", "text")), limit=limit)
        return json.dumps(memories, ensure_ascii=True, indent=2)
    if action == "memory.save":
        content = str(payload.get("content", "")).strip()
        if not content:
            return "Memory save failed: content is required."
        if has_sensitive_secret(content):
            return "Memory save refused: content looks like a secret or API key."
        from app.brain.memory_os import memory_os
        mem_type = str(payload.get("type", "semantic")).lower()
        memory_id = memory_os.add_episodic_memory(content) if mem_type == "episodic" else memory_os.add_semantic_memory(content)
        return f"Saved {mem_type} memory: {memory_id}"
    if action == "memory.review_chat":
        result = review_and_save_memories(
            sender=sender,
            user_message=str(payload.get("user_message", "")),
            assistant_response=str(payload.get("assistant_response", "")),
        )
        return json.dumps(result, ensure_ascii=True, indent=2)

    if action == "documents.ingest_text":
        return handlers["ingest_text_document"](
            str(payload.get("title", "Untitled text")),
            str(first_value("content", "text", "body")),
            str(payload.get("source_type", "whatsapp")),
            str(payload.get("source_url", "")),
        )
    if action == "documents.ingest_file":
        return handlers["ingest_local_file"](
            str(first_value("file_path", "path", "filename", "file")),
            str(payload.get("title", "")),
            str(payload.get("source_type", "pdf")),
        )
    if action == "documents.ingest_url":
        return handlers["ingest_url_document"](
            str(first_value("url", "link", "href")),
            str(payload.get("title", "")),
            str(payload.get("source_type", "pdf")),
        )
    if action == "documents.list":
        return handlers["list_ingested_documents"]()

    if action == "files.read":
        file_path = str(first_value("file_path", "path", "filename", "file"))
        if not file_path:
            return "files.read failed: file_path is required."
        return handlers["view_local_file"](file_path)
    if action == "files.write":
        file_path = str(first_value("file_path", "path", "filename", "file"))
        content = str(first_value("content", "text", "body"))
        if not file_path:
            return "files.write failed: file_path is required."
        return handlers["write_local_file"](file_path, content)
    if action == "system.command":
        command_line = str(first_value("command_line", "command", "cmd", "powershell"))
        if not command_line.strip():
            return "system.command failed: command_line is required. Accepted aliases: command, cmd, powershell."
        decorated = with_logging("system.command")(with_timeout(seconds=30)(handlers["execute_command"]))
        return decorated(command_line)
    if action == "agents.create":
        return handlers["create_agent"](
            str(first_value("agent_name", "name", "agent")),
            str(first_value("system_prompt", "prompt", "instructions")),
        )
    if action == "agents.list":
        return handlers["list_agents"]()
    if action == "agents.call":
        return handlers["call_agent"](str(first_value("agent_name", "name", "agent")), str(first_value("task", "query", "message", "prompt")))
    if action == "agents.delete":
        return handlers["delete_agent"](str(first_value("agent_name", "name", "agent")))

    if action == "workspace.run_python":
        from app.brain.workspace import workspace_sandbox
        code = str(first_value("code", "script", "python_code"))
        res = workspace_sandbox.run_python(code)
        return json.dumps(res, ensure_ascii=True, indent=2)
    if action == "workspace.write_file":
        from app.brain.workspace import workspace_sandbox
        file_path = str(first_value("file_path", "path", "filename", "file"))
        content = str(first_value("content", "text", "body"))
        return workspace_sandbox.write_file(file_path, content)
    if action == "workspace.read_file":
        from app.brain.workspace import workspace_sandbox
        file_path = str(first_value("file_path", "path", "filename", "file"))
        return workspace_sandbox.read_file(file_path)
    if action == "workspace.list_files":
        from app.brain.workspace import workspace_sandbox
        files = workspace_sandbox.list_files()
        return json.dumps({"files": files}, ensure_ascii=True, indent=2)

    if action == "trust.add_site":
        domain = str(first_value("domain", "url", "website", "site"))
        trust_score = int(payload.get("trust_score", 70) or 70)
        topics = payload.get("topics") or []
        note = str(payload.get("note") or "")
        from app.brain.trust_registry import trust_registry
        ok = trust_registry.add_user_site(domain, trust_score, topics, note)
        return f"Successfully added {domain} to trusted sites with trust score {trust_score}." if ok else f"Failed to add {domain} to trusted sites."
        
    if action == "trust.remove_site":
        domain = str(first_value("domain", "url", "site"))
        from app.brain.trust_registry import trust_registry
        ok = trust_registry.remove_user_site(domain)
        return f"Successfully removed {domain} from trusted sites." if ok else f"Domain {domain} was not found in user custom trusted sites."
        
    if action == "trust.list_sites":
        from app.brain.trust_registry import trust_registry
        sites = trust_registry.list_user_sites()
        return json.dumps({"trusted_sites": sites}, ensure_ascii=True, indent=2)

    if action == "trigger.create_threshold":
        name = str(first_value("name", "title", "label"))
        metric = str(first_value("metric", "watch", "track"))
        threshold_value = float(first_value("threshold_value", "value", "price", "level") or 0.0)
        direction = str(first_value("direction", "when", default="above"))
        threshold_pct = payload.get("threshold_pct")
        if threshold_pct is not None:
            threshold_pct = float(threshold_pct)
        trigger_id = db.save_smart_trigger(
            sender=sender,
            trigger_type="threshold",
            name=name,
            metric=metric,
            threshold_value=threshold_value,
            threshold_direction=direction,
            threshold_pct=threshold_pct
        )
        return f"Successfully created threshold alert trigger '{name}' (ID: {trigger_id}) to track {metric} when {direction} {threshold_value}." if trigger_id > 0 else "Failed to create threshold alert trigger."

    if action == "trigger.create_keyword_monitor":
        name = str(first_value("name", "title", "label"))
        query = str(first_value("query", "search", "watch_for"))
        keywords = payload.get("keywords") or []
        check_interval_hours = float(payload.get("check_interval_hours", 4) or 4)
        trigger_id = db.save_smart_trigger(
            sender=sender,
            trigger_type="keyword",
            name=name,
            watch_query=query,
            watch_keywords=keywords,
            interval_hours=check_interval_hours
        )
        return f"Successfully created news/keyword monitor trigger '{name}' (ID: {trigger_id}) for query '{query}' with keywords {keywords}." if trigger_id > 0 else "Failed to create keyword monitor trigger."

    if action == "trigger.create_recurring":
        name = str(first_value("name", "title", "label"))
        query = str(first_value("query", "search", "topic"))
        schedule = str(first_value("schedule", "interval", "when"))
        
        # Simple parser for schedule strings
        interval_hours = 24.0
        if schedule == "daily_9am":
            interval_hours = 24.0
        elif schedule == "weekly_monday":
            interval_hours = 168.0
        elif schedule.startswith("every_"):
            match = re.search(r"every_(\d+)h", schedule)
            if match:
                interval_hours = float(match.group(1))
            else:
                interval_hours = 1.0 # default to every 1h
        
        trigger_id = db.save_smart_trigger(
            sender=sender,
            trigger_type="recurring",
            name=name,
            watch_query=query,
            interval_hours=interval_hours,
            schedule_cron=schedule
        )
        return f"Successfully created recurring update trigger '{name}' (ID: {trigger_id}) scheduled for {schedule}." if trigger_id > 0 else "Failed to create recurring update trigger."

    if action == "trigger.list":
        triggers = db.list_smart_triggers(sender)
        return json.dumps({"active_triggers": triggers}, ensure_ascii=True, indent=2)

    if action == "trigger.pause":
        trigger_id_or_name = str(first_value("trigger_id_or_name", "id", "name", "trigger"))
        ok = db.pause_trigger(trigger_id_or_name)
        return f"Successfully paused trigger '{trigger_id_or_name}'." if ok else f"Trigger '{trigger_id_or_name}' not found or could not be paused."

    if action == "trigger.resume":
        trigger_id_or_name = str(first_value("trigger_id_or_name", "id", "name", "trigger"))
        ok = db.resume_trigger(trigger_id_or_name)
        return f"Successfully resumed trigger '{trigger_id_or_name}'." if ok else f"Trigger '{trigger_id_or_name}' not found or could not be resumed."

    if action == "trigger.delete":
        trigger_id_or_name = str(first_value("trigger_id_or_name", "id", "name", "trigger"))
        ok = db.delete_trigger_by_id_or_name(trigger_id_or_name)
        return f"Successfully deleted trigger '{trigger_id_or_name}'." if ok else f"Trigger '{trigger_id_or_name}' not found or could not be deleted."

    return f"Unknown gateway action '{action}'. Call tool_list/tools.discover first, then get_tool_manual/tools.manual for the exact action."
