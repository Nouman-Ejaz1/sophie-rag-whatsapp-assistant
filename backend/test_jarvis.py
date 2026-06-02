import os
import sys
import time
import json
import unittest
import threading
from unittest.mock import patch, MagicMock

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.brain.thinking_palace import ThinkingPalace, LoopState, thinking_palace
from app.brain.response_cache import response_cache
from app.brain.workspace import workspace_sandbox, WorkspaceSandbox
from app.brain.watchdog import RequestWatchdog, WatchdogInterrupt
from app.services.tool_service import tool_service

class TestStage0GreetingShortCircuit(unittest.TestCase):
    def test_is_trivial_greetings(self):
        palace = ThinkingPalace()
        # True greetings
        self.assertTrue(palace._is_trivial("hi"))
        self.assertTrue(palace._is_trivial("hello"))
        self.assertTrue(palace._is_trivial("hey there"))
        self.assertTrue(palace._is_trivial("assalam o alaikum"))
        self.assertTrue(palace._is_trivial("how are you"))
        
        # False cases (long queries, search queries, commands)
        self.assertFalse(palace._is_trivial("what is the weather in London?"))
        self.assertFalse(palace._is_trivial("show me the exchange rate of USD to PKR"))
        self.assertFalse(palace._is_trivial("run python code to list files"))
        self.assertFalse(palace._is_trivial("a very long message that is definitely not a trivial greeting or acknowledgement"))

    def test_greeting_short_circuit_timing(self):
        palace = ThinkingPalace()
        context = {"sender": "test_sender"}
        
        start_time = time.perf_counter()
        result = palace.think("hi", context)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000.0
        
        # Verify it short-circuits to instant_greetings
        self.assertEqual(result["route"], "instant_greetings")
        self.assertIn("user_output", result["response"])
        self.assertIn("Hi! I'm JARVIS", result["response"])
        self.assertEqual(result["confidence_score"], 100.0)
        
        # Verify execution is extremely fast (well under 100ms)
        self.assertLess(duration_ms, 100.0)
        print(f"[Test] Trivial greeting short-circuit completed in {duration_ms:.2f}ms")

class TestResponseCacheAndToolService(unittest.TestCase):
    def setUp(self):
        response_cache.clear()

    def test_response_cache_expiry_and_key_normalization(self):
        # Key normalization: keys should match even with minor JSON syntax spacing differences
        key1 = response_cache.make_key("weather.forecast", {"location": "London", "days": 1})
        key2 = response_cache.make_key("weather.forecast", {"days": 1, "location": "London"})
        self.assertEqual(key1, key2)
        
        # Cache set and get
        response_cache.set(key1, "Sunny 22C", ttl=2)
        self.assertEqual(response_cache.get(key1), "Sunny 22C")
        
        # Test expiration
        time.sleep(2.5)
        self.assertIsNone(response_cache.get(key1))

    @patch("app.brain.tools.sophie_tool")
    def test_tool_service_cached_exchange_rate(self, mock_sophie_tool):
        mock_sophie_tool.return_value = "USD to PKR is 278.50"
        
        action = "finance.exchange_rate"
        payload = '{"base": "USD", "quote": "PKR"}'
        
        # Call 1: Cache Miss. Underlying tool should be invoked.
        res1 = tool_service.sophie_tool_cached(action, payload)
        self.assertEqual(res1, "USD to PKR is 278.50")
        mock_sophie_tool.assert_called_once_with(action, payload)
        
        mock_sophie_tool.reset_mock()
        
        # Call 2: Cache Hit. Underlying tool should NOT be called.
        res2 = tool_service.sophie_tool_cached(action, payload)
        self.assertEqual(res2, "USD to PKR is 278.50")
        mock_sophie_tool.assert_not_called()

class TestWorkspaceSandbox(unittest.TestCase):
    def setUp(self):
        # We can use a custom workspace name to avoid polluting jarvis_workspace
        self.sandbox = WorkspaceSandbox(workspace_name="test_sandbox_workspace")

    def tearDown(self):
        # Clean up files in our test sandbox
        import shutil
        if self.sandbox.workspace_dir.exists():
            try:
                shutil.rmtree(self.sandbox.workspace_dir)
            except Exception as e:
                print(f"[TestWorkspaceSandbox] Tear down clean-up failed: {e}")

    def test_file_read_write(self):
        write_res = self.sandbox.write_file("test.txt", "hello sandbox")
        self.assertIn("Successfully wrote", write_res)
        
        read_res = self.sandbox.read_file("test.txt")
        self.assertEqual(read_res, "hello sandbox")

    def test_path_traversal_prevention(self):
        # Try writing outside sandbox
        write_res = self.sandbox.write_file("../outside.txt", "malicious content")
        self.assertIn("Error writing file", write_res)
        self.assertIn("outside sandbox boundaries", write_res)
        
        # Try reading outside sandbox
        read_res = self.sandbox.read_file("../outside.txt")
        self.assertIn("Error reading file", read_res)
        self.assertIn("outside sandbox boundaries", read_res)

    def test_run_python_execution(self):
        code = "print('hello from code')"
        res = self.sandbox.run_python(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"].strip(), "hello from code")
        self.assertEqual(res["exit_code"], 0)

    def test_run_python_timeout(self):
        # Run code that takes too long
        code = "import time\ntime.sleep(2.0)"
        # Use small timeout
        res = self.sandbox.run_python(code, timeout_seconds=0.5)
        self.assertFalse(res["success"])
        self.assertIn("Execution timed out", res["stderr"])
        self.assertEqual(res["exit_code"], -1)

class TestLoopSafety(unittest.TestCase):
    def test_loop_state_duplicate_checking(self):
        ls = LoopState()
        action = "weather.forecast"
        payload = {"location": "London"}
        
        # First call is fine
        self.assertFalse(ls.is_duplicate(action, payload))
        # Second identical call is duplicate
        self.assertTrue(ls.is_duplicate(action, payload))
        # Same action but different payload is fine
        self.assertFalse(ls.is_duplicate(action, {"location": "Paris"}))

    @patch("app.brain.tools.openrouter_chat")
    def test_thinking_palace_loop_abort(self, mock_openrouter):
        # Simulating council loop returning identical tool call in two consecutive rounds
        round_1_response = json.dumps({
            "intent_summary": "weather lookup",
            "manual_requests": [],
            "tool_calls": [{"action": "weather.forecast", "payload": {"location": "London"}, "reason": "need weather"}],
            "missing_info": [],
            "final_ready": False,
            "answer_draft": ""
        })
        
        mock_openrouter.side_effect = [
            # Stage 1 Intent classification (low complexity, needs tools)
            json.dumps({
                "intent": "weather",
                "complexity": "low",
                "needs_tools": True,
                "needs_memory": False,
                "risk_level": "low",
                "confidence": 90,
                "suggested_plan": ["check weather"]
            }),
            # Round 1
            round_1_response,
            # Round 2 (same tool call -> triggers warning)
            round_1_response,
            # Round 3 (same tool call again -> triggers loop abort)
            round_1_response
        ]
        
        # Mock sophie_tool, validate_tool_call, etc. to make sure it handles them
        with patch("app.brain.tools.validate_tool_call") as mock_validate, \
             patch("app.brain.tools.sophie_tool") as mock_sophie, \
             patch("app.brain.tools.evaluate_tool_evidence") as mock_eval:
             
            # Setup validator mock
            val_mock = MagicMock()
            val_mock.ok = True
            val_mock.action = "weather.forecast"
            val_mock.normalized_payload = {"location": "London"}
            mock_validate.return_value = val_mock
            
            mock_sophie.return_value = "Mock Weather: Sunny 22C"
            mock_eval.return_value = {"quality": "strong"}
            
            palace = ThinkingPalace(max_depth=3)
            result = palace.think("weather in London", {"sender": "test_sender"})
            
            # Should abort with loop safety message
            self.assertIn("I detected a loop", result["response"])
            self.assertEqual(result["confidence_score"], 40.0)
            self.assertIn("Loop state triggered", result["reasoning"])

class TestWatchdogCircuitBreaker(unittest.TestCase):
    def test_watchdog_thread_exception_injection(self):
        # Setup short timeout watchdog
        test_watchdog = RequestWatchdog(timeout_seconds=0.2, check_interval=0.05)
        test_watchdog.start()
        
        interrupt_caught = threading.Event()
        thread_finished = threading.Event()
        
        def run_slow_operation():
            test_watchdog.register_request("test_request_timeout")
            try:
                # Busy wait to allow async exception injection
                start = time.time()
                while time.time() - start < 3.0:
                    time.sleep(0.001)
            except BaseException as e:
                print(f"[DEBUG TEST] Caught exception type={type(e)}, name={e.__class__.__name__}, module={e.__class__.__module__}", flush=True)
                interrupt_caught.set()
            finally:
                test_watchdog.unregister_request()
                thread_finished.set()
                
        # Start slow thread
        t = threading.Thread(target=run_slow_operation)
        t.start()
        
        # Wait for thread to finish
        t.join(timeout=4.0)
        
        # Stop watchdog
        test_watchdog.stop()
        
        # Assert exception was successfully injected and caught
        self.assertTrue(interrupt_caught.is_set())
        self.assertTrue(thread_finished.is_set())

class TestSmarterLoopSafetyAndStatus(unittest.TestCase):
    @patch("app.brain.thinking_palace.requests.post")
    @patch("app.brain.tools.openrouter_chat")
    def test_idempotent_duplicate_bypass_and_status(self, mock_openrouter, mock_post):
        # 1. Simulating intent classification (needs tools)
        intent_info = {
            "intent": "search",
            "complexity": "low",
            "needs_tools": True,
            "needs_memory": False,
            "risk_level": "low",
            "confidence": 92,
            "suggested_plan": ["Search latest price"]
        }
        
        # Simulating council loop that calls search.latest twice with identical query
        step_response = json.dumps({
            "intent_summary": "petrol search",
            "manual_requests": [],
            "tool_calls": [{"action": "search.latest", "payload": {"query": "petrol price"}, "reason": "search first"}],
            "missing_info": [],
            "final_ready": False,
            "answer_draft": ""
        })
        
        # Round 2: Same tool call (idempotent search.latest)
        step_response_round_2 = json.dumps({
            "intent_summary": "petrol search",
            "manual_requests": [],
            "tool_calls": [{"action": "search.latest", "payload": {"query": "petrol price"}, "reason": "search second"}],
            "missing_info": [],
            "final_ready": False,
            "answer_draft": ""
        })
        
        # Round 3: Yields final ready answer draft using previous response
        step_response_round_3 = json.dumps({
            "intent_summary": "petrol search",
            "manual_requests": [],
            "tool_calls": [],
            "missing_info": [],
            "final_ready": True,
            "answer_draft": "Petrol price is 280 PKR"
        })
        
        mock_openrouter.side_effect = [
            json.dumps(intent_info),
            step_response,
            step_response_round_2,
            step_response_round_3
        ]
        
        with patch("app.brain.tools.validate_tool_call") as mock_validate, \
             patch("app.brain.tools.sophie_tool") as mock_sophie, \
             patch("app.brain.tools.evaluate_tool_evidence") as mock_eval:
             
            val_mock = MagicMock()
            val_mock.ok = True
            val_mock.action = "search.latest"
            val_mock.normalized_payload = {"query": "petrol price"}
            mock_validate.return_value = val_mock
            
            mock_sophie.return_value = "Petrol is 280 PKR"
            mock_eval.return_value = {"quality": "strong"}
            
            palace = ThinkingPalace(max_depth=3)
            result = palace.think("petrol price today", {"sender": "12345"})
            
            # Should NOT abort! Should return final answer draft
            self.assertEqual(result["response"], "<ouput><user_output>Petrol price is 280 PKR</user_output><tools_call>None</tools_call></ouput>")
            
            # Verify sophie_tool was called exactly ONCE (since the second was bypassed via cache)
            mock_sophie.assert_called_once()
            
            # Verify requests.post was called to send status updates to port 3001
            self.assertTrue(mock_post.called)
            status_calls = [call[1]['json']['text'] for call in mock_post.call_args_list]
            self.assertTrue(any("Sophie is analyzing" in text for text in status_calls))
            self.assertTrue(any("Sophie is reasoning" in text for text in status_calls))
            self.assertTrue(any("Sophie is searching" in text for text in status_calls))

if __name__ == "__main__":
    unittest.main()
