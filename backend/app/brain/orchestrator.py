import json
from typing import Annotated, Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from app.config import (
    GEMINI_MODEL,
    LLM_PROVIDER,
)
from app.database import db
from app.brain.tools import (
    sophie_tool,
    openrouter_chat,
    record_tool_execution,
    load_tool_manuals,
    openrouter_model_mode,
    get_openrouter_model_candidates,
    is_openrouter_retryable_error,
    build_openrouter_llm,
    build_sophie_langchain_tools,
    run_openrouter_tool_graph,
    maybe_prefetch_openrouter_tools,
    is_agent_request,
    is_simple_current_time_query,
    _contains_any,
    _extract_after_pattern,
    _tool_action,
    _base_answer_policy,
    _currency_tool_payload,
    _ps_single_quote,
    _extract_desktop_folder_name,
    _desktop_folder_check_command,
    is_public_affairs_person_lookup,
    build_brain_decision,
    openrouter_intent_needs_tools,
    openrouter_query_needs_tool,
    summarize_intent_for_logs,
    determine_thinkbox_route,
    build_thinkbox_payload,
    log_thinkbox_payload,
    log_council_snapshot,
    is_lightweight_chat_message,
    gateway_handlers,
    _parse_structured_research_result,
    evaluate_tool_evidence,
    format_verified_live_data,
    _structured_results_from_active_tools,
    _short_text,
    build_deterministic_evidence_response,
    build_deterministic_action_response,
    build_deterministic_tool_response,
    _extract_json_object,
    _coerce_list,
    _normalize_tool_calls,
    _parse_council_step,
    _fetch_requested_manuals,
    _format_council_tool_results,
    _answer_from_missing_info,
    _wrap_whatsapp_response,
    _council_system_prompt,
    _council_user_prompt,
    run_tool_council_loop,
    execute_brain_tool_plan,
    execute_command,
    log_to_sophie_brain,
    create_folder,
    create_desktop_folder,
    get_weather,
    _clean_search_text,
    _current_search_year,
    _has_latest_search_intent,
    _normalize_search_query,
    _query_terms,
    _search_result_score,
    _rank_search_results,
    _has_relevant_search_result,
    _dedupe_search_results,
    _format_search_results,
    _looks_current_or_news_query,
    _expand_search_query,
    _decode_duckduckgo_url,
    _duckduckgo_html_results,
    _duckduckgo_instant_answer_results,
    _google_custom_search_results,
    _google_html_results,
    _google_news_rss_results,
    _decode_bing_url,
    _bing_news_rss_results,
    _bing_rss_results,
    _bing_html_results,
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
    clean_xml_from_text,
    parse_strict_json_response,
    _actual_tools_summary,
    _replace_user_output,
    response_claims_unexecuted_tools,
    maybe_recover_missing_tools,
    enforce_honest_tool_report,
)

# State Graph Definition
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

class Orchestrator:
    """
    Slim, high-performance service-coordinated Orchestrator.
    Maintains the LangGraph state machine but delegates the operational
    nodes to decoupled services.
    """
    def __init__(self):
        self.reasoning_model = None
        # Configuration matches the active backend framework
        if LLM_PROVIDER != "openrouter":
            import google.generativeai as genai
            self.reasoning_model = genai.GenerativeModel(
                GEMINI_MODEL,
                tools=[
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
            )

    def evaluate_trigger_node(self, state: AgentGraphState) -> AgentGraphState:
        """Evaluates urgency of event triggers and classifies task workflow type."""
        event = state.get("triggered_event")
        if event and not isinstance(event, dict):
            print(f"[WARNING] triggered_event is not a dict! Value: {event}")
            event = None
            
        is_urgent = False
        task_type = "general_chat"
        
        if event:
            is_urgent = event.get("is_urgent", False)
            task_type = event.get("suggested_task_type", "unknown")
            print(f"[Orchestrator Node] Evaluating Trigger: {event.get('title')}. Urgent={is_urgent}")
        
        return {
            **state,
            "is_urgent": is_urgent,
            "task_type": task_type,
            "triggered_event": event
        }

    def retrieve_context_node(self, state: AgentGraphState) -> AgentGraphState:
        """Retrieves semantic documents from DocBrain and Episodic/Semantic memory from MemoryOS."""
        query = ""
        event = state.get("triggered_event")
        if event:
            query = f"{event.get('title')} {event.get('description')}"
        elif state.get("messages"):
            query = state.get("messages")[-1]["content"]
            
        print(f"[Orchestrator Node] Querying knowledge base with: '{query[:40]}...'")
        
        if is_lightweight_chat_message(query) or is_simple_current_time_query(query):
            print("[Orchestrator Node] Direct deterministic query detected. Skipping RAG/memory prefetch.")
            return {
                **state,
                "retrieved_docs": [],
                "retrieved_memories": []
            }
        
        # Route retrieve_context_node to MemoryService
        from app.services.memory_service import memory_service
        sender = state.get("sender") or "unknown_number"
        ctx = memory_service.retrieve_context(query, sender, doc_limit=3, mem_limit=3)
        
        return {
            **state,
            "retrieved_docs": ctx.get("docs", []),
            "retrieved_memories": ctx.get("memories", [])
        }

    def thinkbox_node(self, state: AgentGraphState) -> AgentGraphState:
        """LangGraph council panel: structured operational thinking before reasoning/tools."""
        query = ""
        event = state.get("triggered_event")
        if event:
            query = f"{event.get('title')} {event.get('description')}"
        elif state.get("messages"):
            query = state.get("messages")[-1]["content"]

        # Route thinkbox_node to IntentService.classify
        from app.services.intent_service import intent_service
        sender = state.get("sender") or "unknown_number"
        
        # Prepare classification context
        context = {
            "sender": sender,
            "retrieved_docs": state.get("retrieved_docs", []),
            "retrieved_memories": state.get("retrieved_memories", [])
        }
        payload = intent_service.classify(query, context)
        
        print(
            f"[Orchestrator Node] Thinkbox council route={payload.get('route')} "
            f"intent={payload.get('intent')} needs_tools={payload.get('needs_tools')}",
            flush=True,
        )
        return {
            **state,
            "thinkbox": payload,
        }

    def reason_node(self, state: AgentGraphState) -> AgentGraphState:
        """Refactored Two-Stage Agent reasoning using ThinkingService (Thinking Palace)."""
        messages = state.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        sender = state.get("sender") or "unknown_number"
        
        # Prepare context for thinking palace reasoning loop
        context = {
            "sender": sender,
            "retrieved_docs": state.get("retrieved_docs", []),
            "retrieved_memories": state.get("retrieved_memories", []),
            "thinkbox": state.get("thinkbox", {}),
            "task_type": state.get("task_type"),
            "triggered_event": state.get("triggered_event"),
            "history": messages[:-1] if len(messages) > 1 else []
        }
        
        # Route reason_node to ThinkingService.process
        from app.services.thinking_service import thinking_service
        result = thinking_service.process(user_message, context)
        
        return {
            **state,
            **result
        }

    def execute_action_node(self, state: AgentGraphState) -> AgentGraphState:
        """Simulates action execution, checking procedural steps from memory or drafting new procedures."""
        task_type = state.get("task_type", "general_chat")
        event = state.get("triggered_event")
        action_taken = None
        
        if event and task_type != "general_chat":
            from app.services.memory_service import memory_service
            
            # Retrieve procedural workflow
            steps = memory_service.get_procedural(task_type)
            
            if steps:
                action_taken = f"Executed procedural steps for '{task_type}': " + " -> ".join(steps)
                print(f"[Orchestrator Node] Procedural steps retrieved: {steps}")
            else:
                # First time seeing this task. Draft a new procedure!
                drafted_steps = [
                    "Assess security relevance",
                    "Scan Notion architecture docs",
                    "Post webhook alert to Slack channel #security-ops",
                    "Draft email recap to CTO"
                ]
                memory_service.save_procedural(task_type, drafted_steps)
                action_taken = f"First-time task. Drafted new procedural steps: " + " -> ".join(drafted_steps)
                print(f"[Orchestrator Node] No procedure found. Drafted & saved new checklist for: '{task_type}'")
                
            # Log action as system log
            db.log_event(
                source="ActionLayer",
                message=f"Action fired: {action_taken[:60]}...",
                status="success",
                meta_dict={"task_type": task_type, "action": action_taken}
            )
            
            # Save the event log as a new episodic memory conversation
            memory_service.save_episodic(
                f"Triggered event '{event.get('title')}' led to action: '{action_taken}'"
            )
            
        return {
            **state,
            "action_taken": action_taken
        }

    def urgency_router(self, state: AgentGraphState) -> str:
        """Decides routing path based on trigger urgency."""
        return "action" if state.get("is_urgent", False) else "think"

    def _build_graph(self) -> StateGraph:
        """Compiles the StateGraph of JARVIS."""
        workflow = StateGraph(AgentGraphState)
        
        # Add Nodes
        workflow.add_node("evaluate_trigger", self.evaluate_trigger_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("thinkbox", self.thinkbox_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("execute_action", self.execute_action_node)
        
        # Define Entry point
        workflow.set_entry_point("evaluate_trigger")
        
        # Define conditional routing from evaluate trigger
        workflow.add_conditional_edges(
            "evaluate_trigger",
            self.urgency_router,
            {
                "action": "execute_action",
                "think": "retrieve_context"
            }
        )
        
        # Define straight edges
        workflow.add_edge("retrieve_context", "thinkbox")
        workflow.add_edge("thinkbox", "reason")
        workflow.add_edge("reason", END)
        workflow.add_edge("execute_action", END)
        
        return workflow.compile()

    def run(
        self,
        message: Optional[str] = None,
        triggered_event: Optional[Dict[str, Any]] = None,
        history: List[Dict[str, str]] = None,
        sender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the coordinated service graph state machine."""
        # Reset active tools tracker for the current request context
        from app.brain.tools import reset_active_run_tools
        reset_active_run_tools(sender or "unknown_sender")

        state = {
            "messages": history or [],
            "triggered_event": triggered_event,
            "retrieved_docs": [],
            "retrieved_memories": [],
            "thinkbox": {},
            "is_urgent": False,
            "reasoning": "",
            "citations": [],
            "confidence_score": 0.0,
            "action_taken": None,
            "response": "",
            "task_type": "general_chat",
            "sender": sender
        }
        
        if message:
            state["messages"].append({"role": "user", "content": message})
            
        graph = self._build_graph()
        result = graph.invoke(state)
        return result

# Global orchestrator instance
orchestrator = Orchestrator()
