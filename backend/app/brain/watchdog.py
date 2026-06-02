import time
import threading
import ctypes
import inspect
from typing import Dict, Any

class WatchdogInterrupt(BaseException):
    """Custom exception raised when a request is hard-killed by the watchdog circuit breaker."""
    pass

class RequestWatchdog:
    """
    Independent thread-isolated watchdog circuit-breaker.
    Tracks requests by thread ID and hard-kills/aborts those exceeding the timeout (120s).
    """
    def __init__(self, timeout_seconds: float = 360.0, check_interval: float = 2.0):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self._active_requests: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None

    def start(self):
        """Starts the watchdog monitoring thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="WatchdogMonitorThread")
            self._thread.start()
            print("[Watchdog] Started independent monitoring thread.")

    def stop(self):
        """Stops the watchdog monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[Watchdog] Stopped independent monitoring thread.")

    def register_request(self, request_id: str):
        """Registers the current running request and thread under the watchdog."""
        thread_id = threading.get_ident()
        with self._lock:
            self._active_requests[thread_id] = {
                "request_id": request_id,
                "start_time": time.time(),
                "thread": threading.current_thread()
            }
        print(f"[Watchdog] Registered request {request_id} on thread {thread_id}.")

    def unregister_request(self):
        """Unregisters the request running on the current thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._active_requests:
                req = self._active_requests.pop(thread_id)
                duration = time.time() - req["start_time"]
                print(f"[Watchdog] Unregistered request {req['request_id']} after {duration:.2f}s.")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            to_kill = []
            with self._lock:
                for thread_id, info in list(self._active_requests.items()):
                    elapsed = now - info["start_time"]
                    if elapsed > self.timeout_seconds:
                        to_kill.append((thread_id, info["request_id"]))
            
            for thread_id, req_id in to_kill:
                print(f"[Watchdog] WARNING: Request {req_id} on thread {thread_id} exceeded {self.timeout_seconds}s. Injecting WatchdogInterrupt!")
                self._inject_exception(thread_id, WatchdogInterrupt)
                # Remove from active requests
                with self._lock:
                    if thread_id in self._active_requests:
                        self._active_requests.pop(thread_id)
            
            self._stop_event.wait(self.check_interval)

    def _inject_exception(self, thread_id: int, exc_class):
        """Raises an exception in the context of the target thread (Windows/Linux compatible)."""
        if not inspect.isclass(exc_class):
            raise TypeError("Only exception classes can be injected.")
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(exc_class))
        if res == 0:
            print(f"[Watchdog] Failed to inject exception: Thread ID {thread_id} not found.")
        elif res > 1:
            # Revert since it affects other threads
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
            print(f"[Watchdog] PyThreadState_SetAsyncExc failed: cleanup triggered.")

# Global instance
watchdog = RequestWatchdog()
