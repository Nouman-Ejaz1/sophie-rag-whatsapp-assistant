import os
import sys
import unittest
from unittest.mock import patch


BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.brain import orchestrator as orch
from app.brain import research_engine
from app.brain import tool_gateway
from app.brain import memory_curator
from app.brain import tools


class BrainDecisionTests(unittest.TestCase):
    def test_lightweight_chat_needs_no_tools(self):
        decision = orch.build_brain_decision("Hi")
        self.assertEqual(decision["route"], "lightweight_chat")
        self.assertFalse(decision["needs_tool"])
        self.assertEqual(decision["tool_plan"], [])

    def test_current_event_shooting_routes_to_search(self):
        decision = orch.build_brain_decision("Tell me about the shooting at mosque in San Diego")
        self.assertEqual(decision["route"], "evidence_tool_plan")
        self.assertTrue(decision["needs_live_evidence"])
        self.assertEqual(decision["tool_plan"][0]["action"], "search.latest")
        self.assertEqual(decision["tool_plan"][0]["fallback_action"], "search.web")

    def test_location_news_routes_to_search(self):
        decision = orch.build_brain_decision("What happened in Lahore today?")
        self.assertEqual(decision["intent"], "current_research")
        self.assertEqual(decision["tool_plan"][0]["action"], "search.latest")

    def test_public_affairs_name_lookup_routes_to_search(self):
        decision = orch.build_brain_decision("tell me the name of the guy america send to islamabad for peace talk")
        self.assertEqual(decision["intent"], "current_research")
        self.assertEqual(decision["route"], "evidence_tool_plan")
        self.assertTrue(decision["needs_live_evidence"])
        self.assertEqual(decision["tool_plan"][0]["action"], "search.latest")

    def test_plain_country_question_does_not_force_live_search(self):
        decision = orch.build_brain_decision("Tell me about America")
        self.assertNotEqual(decision["route"], "evidence_tool_plan")
        self.assertFalse(decision["needs_tool"])

    def test_exchange_rate_routes_to_finance_tool(self):
        decision = orch.build_brain_decision("USD to PKR exchange rate")
        self.assertEqual(decision["intent"], "finance_exchange_rate")
        self.assertEqual(decision["tool_plan"][0]["action"], "finance.exchange_rate")
        self.assertEqual(decision["tool_plan"][0]["payload"]["base"], "USD")
        self.assertEqual(decision["tool_plan"][0]["payload"]["quote"], "PKR")

    def test_weather_routes_to_weather_tool(self):
        decision = orch.build_brain_decision("Weather in London")
        self.assertEqual(decision["intent"], "weather")
        self.assertEqual(decision["tool_plan"][0]["action"], "weather.forecast")
        self.assertEqual(decision["tool_plan"][0]["payload"]["location"], "London")

    def test_history_question_routes_to_chat_history(self):
        decision = orch.build_brain_decision("What did I ask earlier?")
        self.assertEqual(decision["intent"], "chat_history")
        self.assertEqual(decision["tool_plan"][0]["action"], "chat.history")

    def test_desktop_folder_request_is_action_plan(self):
        decision = orch.build_brain_decision("Create a folder on Desktop")
        self.assertEqual(decision["route"], "action_tool_plan")
        self.assertIn("folder_name_or_path", decision["missing_info"])

    def test_desktop_folder_existence_check_is_deterministic(self):
        decision = orch.build_brain_decision("tell me is there folder name my app in the desktop")
        self.assertEqual(decision["route"], "action_tool_plan")
        self.assertEqual(decision["tool_plan"][0]["action"], "system.command")
        self.assertIn("Test-Path", decision["tool_plan"][0]["payload"]["command_line"])
        self.assertIn("my app", decision["tool_plan"][0]["payload"]["command_line"])
        self.assertFalse(decision["answer_policy"]["allow_llm_tool_expansion"])


class HonestyRecoveryTests(unittest.TestCase):
    def setUp(self):
        orch.ACTIVE_RUN_TOOLS = []
        tools.ACTIVE_RUN_TOOLS = []

    def test_detects_model_claimed_tools_without_execution(self):
        response = "<ouput><user_output>I searched it.</user_output><tools_call>search.web</tools_call></ouput>"
        self.assertTrue(orch.response_claims_unexecuted_tools(response))

    def test_recovery_uses_brain_decision_tool_plan(self):
        decision = orch.build_brain_decision("Tell me about the shooting at mosque in San Diego")
        with patch("app.brain.tools.execute_brain_tool_plan", return_value="verified evidence") as execute_mock:
            recovered = orch.maybe_recover_missing_tools(decision, "Tell me about the shooting at mosque in San Diego")
        self.assertEqual(recovered, "verified evidence")
        execute_mock.assert_called_once()

    def test_weak_latest_result_falls_back_to_web_search(self):
        decision = orch.build_brain_decision("Tell me about the shooting at mosque in San Diego")
        weak = 'STRUCTURED_RESEARCH_RESULT\n{"answerable": false, "claims": [], "citations": [], "warnings": []}'
        strong = 'STRUCTURED_RESEARCH_RESULT\n{"answerable": true, "claims": [{"value": "confirmed"}], "citations": [{"title": "source"}], "warnings": []}'
        with patch("app.brain.tools.sophie_tool", side_effect=[weak, strong]) as tool_mock:
            orch.execute_brain_tool_plan(decision, "Tell me about the shooting at mosque in San Diego")
        self.assertEqual(tool_mock.call_count, 2)
        self.assertEqual(tool_mock.call_args_list[0].args[0], "search.latest")
        self.assertEqual(tool_mock.call_args_list[1].args[0], "search.web")

    def test_structured_claims_win_over_failure_words_in_text(self):
        response = (
            'STRUCTURED_RESEARCH_RESULT\n{"answerable": true, "trust_score": 90, '
            '"claims": [{"claim_type": "person_lookup", "value": "Witkoff and Kushner", '
            '"details": ["Trump calls off a trip after talks failed earlier."]}], '
            '"citations": [{"title": "source"}]}'
        )
        quality = orch.evaluate_tool_evidence("search.latest", response)
        self.assertEqual(quality["quality"], "strong")


class ResearchEngineGenericNewsTests(unittest.TestCase):
    def test_generic_news_sources_become_answerable_claims(self):
        source = research_engine.make_source(
            "Recap: Three killed, two suspects dead in shooting at San Diego mosque",
            "https://www.nbcsandiego.com/news/local/example",
            "Law enforcement officers responded Monday to an active shooter at a mosque in the Clairemont neighborhood of San Diego.",
            "Bing News",
        )
        claims = research_engine.build_generic_news_claims(
            "Tell me about the shooting at mosque in San Diego",
            [source],
            "latest",
        )
        self.assertTrue(claims)
        self.assertEqual(claims[0].claim_type, "news_report")
        self.assertGreaterEqual(claims[0].trust_score, 55)
        self.assertIn("reputable_press", claims[0].source_categories)

    def test_deterministic_evidence_response_uses_structured_claims(self):
        tools.ACTIVE_RUN_TOOLS = [{
            "tool": "research_latest",
            "arguments": {"query": "Tell me about the shooting at mosque in San Diego"},
            "response": (
                'STRUCTURED_RESEARCH_RESULT\n{"answerable": true, "trust_score": 81, '
                '"claims": [{"claim_type": "news_report", "value": "Three people killed in shooting at San Diego mosque.", '
                '"details": ["NBC San Diego report"], "source_titles": ["NBC San Diego"], "source_urls": ["https://example.com"], '
                '"source_categories": ["reputable_press"]}], '
                '"citations": [{"title": "NBC San Diego", "url": "https://example.com", "snippet": "Three killed"}]}'
            ),
        }]
        decision = orch.build_brain_decision("Tell me about the shooting at mosque in San Diego")
        response = orch.build_deterministic_evidence_response(decision)
        self.assertIsNotNone(response)
        self.assertIn("Three people killed", response["response"])
        self.assertGreaterEqual(response["confidence_score"], 80)


class ResearchEnginePersonLookupTests(unittest.TestCase):
    def test_surname_pair_source_becomes_person_lookup_claim(self):
        source = research_engine.make_source(
            "Trump calls off Witkoff, Kushner trip to Pakistan for Iran peace talks - The Washington Post",
            "https://www.washingtonpost.com/world/2026/05/18/example",
            "White House officials said the U.S. envoys were expected to travel to Islamabad for talks.",
            "Bing News",
        )
        claims = research_engine.build_person_lookup_claims(
            "tell me the name of the guy america send to islamabad for peace talk",
            [source],
            "latest",
        )
        self.assertTrue(claims)
        self.assertEqual(claims[0].claim_type, "person_lookup")
        self.assertEqual(claims[0].value, "Witkoff and Kushner")
        self.assertIn("reputable_press", claims[0].source_categories)

    def test_full_names_are_preserved_when_present(self):
        source = research_engine.make_source(
            "US envoys Steve Witkoff and Jared Kushner head to Pakistan for Iran talks",
            "https://www.reuters.com/world/example",
            "The delegation was scheduled to travel to Islamabad for peace talks.",
            "Bing News",
        )
        claims = research_engine.build_person_lookup_claims(
            "who did america send to islamabad for peace talks",
            [source],
            "latest",
        )
        self.assertTrue(claims)
        self.assertEqual(claims[0].value, "Steve Witkoff and Jared Kushner")

    def test_deterministic_evidence_response_uses_person_lookup_claim(self):
        tools.ACTIVE_RUN_TOOLS = [{
            "tool": "research_latest",
            "arguments": {"query": "tell me the name of the guy america send to islamabad for peace talk"},
            "response": (
                'STRUCTURED_RESEARCH_RESULT\n{"answerable": true, "trust_score": 95, '
                '"claims": [{"claim_type": "person_lookup", "value": "Witkoff and Kushner", '
                '"details": ["The Washington Post: Trump calls off Witkoff, Kushner trip to Pakistan for Iran peace talks"], '
                '"source_titles": ["The Washington Post"], "source_urls": ["https://example.com"], '
                '"source_categories": ["reputable_press"]}], '
                '"citations": [{"title": "The Washington Post", "url": "https://example.com"}]}'
            ),
        }]
        decision = orch.build_brain_decision("tell me the name of the guy america send to islamabad for peace talk")
        response = orch.build_deterministic_evidence_response(decision)
        self.assertIsNotNone(response)
        self.assertIn("Witkoff and Kushner", response["response"])
        self.assertGreaterEqual(response["confidence_score"], 90)


class GatewayPayloadAliasTests(unittest.TestCase):
    def test_every_dispatchable_action_has_tool_spec(self):
        dispatch_actions = {
            "search.web", "search.latest", "search.benchmarks", "finance.exchange_rate",
            "weather.forecast", "time.current", "wiki.lookup", "history.check",
            "nutrition.lookup_food", "nutrition.log_food", "nutrition.daily_summary",
            "schedule.reminder", "schedule.recurring_task", "schedule.list",
            "chat.history", "context.lookup", "memory.recall", "memory.save",
            "memory.review_chat", "documents.ingest_text", "documents.ingest_file",
            "documents.ingest_url", "documents.list", "files.read", "files.write",
            "system.command", "agents.create", "agents.list", "agents.call",
            "agents.delete", "tools.discover", "tools.manual", "tools.call",
        }
        self.assertTrue(dispatch_actions.issubset(set(tool_gateway.TOOL_SPECS)))

    def test_manuals_are_generated_from_tool_specs(self):
        manual = tool_gateway.manual_for(actions=["weather.forecast"])
        self.assertIn("required_fields", manual)
        self.assertIn("location", manual)
        self.assertIn("example_payload", manual)

    def test_payload_aliases_are_normalized_before_dispatch(self):
        result = tool_gateway.validate_tool_call("finance.exchange_rate", '{"from":"USD","to":"PKR"}')
        self.assertTrue(result.ok)
        self.assertEqual(result.normalized_payload["base"], "USD")
        self.assertEqual(result.normalized_payload["quote"], "PKR")

    def test_invalid_payload_returns_schema_feedback(self):
        result = tool_gateway.dispatch_tool_action(
            "weather.forecast",
            "{}",
            context={"sender": "test"},
            handlers={},
        )
        self.assertIn("TOOL_VALIDATION_ERROR", result)
        self.assertIn("location is required", result)
        self.assertIn("example_payload", result)

    def test_generic_system_command_is_blocked_by_validation(self):
        result = tool_gateway.validate_tool_call("system.command", '{"command":"uname -s"}')
        self.assertFalse(result.ok)
        self.assertIn("too generic", " ".join(result.errors))

    def test_system_command_accepts_command_alias(self):
        calls = []
        result = tool_gateway.dispatch_tool_action(
            "system.command",
            '{"command": "Write-Output ok"}',
            context={"sender": "test"},
            handlers={"execute_command": lambda command_line: calls.append(command_line) or "ok"},
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["Write-Output ok"])

    def test_system_command_rejects_empty_command(self):
        result = tool_gateway.dispatch_tool_action(
            "system.command",
            "{}",
            context={"sender": "test"},
            handlers={"execute_command": lambda command_line: "should not run"},
        )
        self.assertIn("command_line is required", result)


class ToolCouncilLoopTests(unittest.TestCase):
    def setUp(self):
        orch.ACTIVE_RUN_TOOLS = []
        tools.ACTIVE_RUN_TOOLS = []

    def _context(self, message):
        return tool_gateway.format_context_packet(user_input=message, recent_messages=[], docs=[], memories=[])

    def test_council_fetches_manual_calls_tool_and_synthesizes(self):
        message = "Weather in London"
        model_steps = [
            '{"intent_summary":"weather lookup","manual_requests":[{"actions":["weather.forecast"]}],"tool_calls":[{"action":"weather.forecast","payload":{"location":"London","days":1},"reason":"Need live weather"}],"missing_info":[],"final_ready":false,"answer_draft":""}',
            '{"intent_summary":"weather lookup","manual_requests":[],"tool_calls":[],"missing_info":[],"final_ready":true,"answer_draft":"London weather is mild from the verified tool result."}',
        ]

        def fake_tool(action, payload_json):
            orch.record_tool_execution("sophie_tool", {"action": action, "payload_json": payload_json}, "Weather result: mild")
            return "Weather result: mild"

        with patch("app.brain.tools.openrouter_chat", side_effect=model_steps), patch("app.brain.tools.sophie_tool", side_effect=fake_tool) as tool_mock:
            response = orch.run_tool_council_loop(
                message,
                self._context(message),
                orch.build_brain_decision(message),
                "May 19, 2026",
                "2026",
            )

        self.assertIsNotNone(response)
        tool_mock.assert_called_once()
        self.assertIn("London weather", response["response"])
        self.assertIn("sophie_tool(action=weather.forecast)", response["response"])

    def test_invalid_first_payload_gets_feedback_and_corrected_next_round(self):
        message = "Weather in London"
        model_steps = [
            '{"intent_summary":"weather lookup","manual_requests":[{"actions":["weather.forecast"]}],"tool_calls":[{"action":"weather.forecast","payload":{},"reason":"Need live weather"}],"missing_info":[],"final_ready":false,"answer_draft":""}',
            '{"intent_summary":"weather lookup","manual_requests":[],"tool_calls":[{"action":"weather.forecast","payload":{"location":"London"},"reason":"Corrected missing location"}],"missing_info":[],"final_ready":false,"answer_draft":""}',
            '{"intent_summary":"weather lookup","manual_requests":[],"tool_calls":[],"missing_info":[],"final_ready":true,"answer_draft":"I found the weather after correcting the tool payload."}',
        ]

        def fake_tool(action, payload_json):
            orch.record_tool_execution("sophie_tool", {"action": action, "payload_json": payload_json}, "Weather result: mild")
            return "Weather result: mild"

        with patch("app.brain.tools.openrouter_chat", side_effect=model_steps), patch("app.brain.tools.sophie_tool", side_effect=fake_tool) as tool_mock:
            response = orch.run_tool_council_loop(
                message,
                self._context(message),
                orch.build_brain_decision(message),
                "May 19, 2026",
                "2026",
            )

        self.assertIsNotNone(response)
        tool_mock.assert_called_once()
        self.assertIn("correcting the tool payload", response["response"])

    def test_vague_file_question_does_not_execute_generic_command(self):
        message = "check my folder"
        model_steps = [
            '{"intent_summary":"vague file check","manual_requests":[{"actions":["system.command"]}],"tool_calls":[{"action":"system.command","payload":{"command":"uname -s"},"reason":"Bad generic probe"}],"missing_info":[],"final_ready":false,"answer_draft":""}',
            '{"intent_summary":"vague file check","manual_requests":[],"tool_calls":[],"missing_info":["folder name or path"],"final_ready":true,"answer_draft":"I need the folder name or path."}',
        ]

        with patch("app.brain.tools.openrouter_chat", side_effect=model_steps), patch("app.brain.tools.sophie_tool") as tool_mock:
            response = orch.run_tool_council_loop(
                message,
                self._context(message),
                orch.build_brain_decision(message),
                "May 19, 2026",
                "2026",
            )

        self.assertIsNotNone(response)
        tool_mock.assert_not_called()
        self.assertIn("folder name or path", response["response"])
        self.assertIn("<tools_call>None</tools_call>", response["response"])

    def test_missing_required_input_asks_user(self):
        message = "Remind me"
        model_steps = [
            '{"intent_summary":"reminder","manual_requests":[{"actions":["schedule.reminder"]}],"tool_calls":[],"missing_info":["reminder time and message"],"final_ready":true,"answer_draft":"I need the reminder time and message."}',
        ]
        with patch("app.brain.tools.openrouter_chat", side_effect=model_steps):
            response = orch.run_tool_council_loop(
                message,
                self._context(message),
                orch.build_brain_decision(message),
                "May 19, 2026",
                "2026",
            )
        self.assertIn("reminder time and message", response["response"])


class MemoryCuratorNoiseTests(unittest.TestCase):
    def test_desktop_folder_question_is_not_saved_as_memory(self):
        candidates = memory_curator.extract_memory_candidates(
            "tell me is there folder name my app in the desktop"
        )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
