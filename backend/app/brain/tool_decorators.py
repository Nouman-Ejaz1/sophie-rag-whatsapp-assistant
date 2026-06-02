"""
Robust tool execution wrappers: @with_timeout, @with_retry, and @with_logging.
Specially designed to work on Windows (using thread pool execution for timeouts).
"""

import time
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Any, Optional
from app.database import db

def with_timeout(seconds: int = 30):
    """
    Decorator to enforce a execution timeout.
    Uses ThreadPoolExecutor for cross-platform (Windows) safety.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Use ThreadPoolExecutor to run the function in a background thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FutureTimeoutError:
                    error_msg = f"Tool execution timed out after {seconds} seconds."
                    print(f"[TimeoutError] {error_msg}")
                    # Attempt to log the timeout
                    try:
                        db.log_event(
                            source="ToolDecorator",
                            message=f"Timeout: {func.__name__} exceeded {seconds}s",
                            status="error"
                        )
                    except Exception:
                        pass
                    return error_msg
        return wrapper
    return decorator

def with_retry(attempts: int = 2, delay: float = 1.0):
    """
    Decorator to automatically retry a function if it raises an exception.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[RetryWarning] Attempt {attempt}/{attempts} for {func.__name__} failed: {e}")
                    if attempt < attempts:
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
            
            # If all attempts failed, raise or return error string
            error_msg = f"Tool {func.__name__} failed after {attempts} attempts. Last error: {last_exception}"
            try:
                db.log_event(
                    source="ToolDecorator",
                    message=error_msg,
                    status="error"
                )
            except Exception:
                pass
            return f"Error: {error_msg}"
        return wrapper
    return decorator

def with_logging(tool_name: Optional[str] = None):
    """
    Decorator to log tool execution inputs, duration, and output/error to the database.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = tool_name or func.__name__
            start_time = time.time()
            
            # Format arguments for logging
            args_summary = f"args={args[:3]}" if args else ""
            kwargs_summary = f"kwargs={list(kwargs.keys())}" if kwargs else ""
            inputs = f"{args_summary} {kwargs_summary}".strip()
            
            print(f"[ToolCall] Running tool '{name}' with inputs: {inputs}")
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000  # Milliseconds
                
                # Log success
                try:
                    # Avoid recursive logging by not logging database log_event itself
                    if func.__name__ not in {"log_event", "save_memory_record"}:
                        db.log_event(
                            source="ToolExecution",
                            message=f"Tool '{name}' completed successfully in {duration:.1f}ms",
                            status="success",
                            meta_dict={
                                "tool": name,
                                "duration_ms": duration,
                                "result_preview": str(result)[:300]
                            }
                        )
                except Exception:
                    pass
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                error_msg = f"Tool '{name}' failed in {duration:.1f}ms: {e}"
                print(f"[ToolError] {error_msg}")
                
                try:
                    db.log_event(
                        source="ToolExecution",
                        message=error_msg,
                        status="error",
                        meta_dict={"tool": name, "duration_ms": duration, "error": str(e)}
                    )
                except Exception:
                    pass
                
                raise e
        return wrapper
    return decorator
