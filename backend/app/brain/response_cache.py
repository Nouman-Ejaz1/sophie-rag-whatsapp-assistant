"""
Thread-safe response caching layer.
Used to cache slow, rate-limited deterministic queries like exchange rates and weather.
"""

import time
import hashlib
import json
import threading
from typing import Optional, Dict, Any

class ResponseCache:
    """
    A thread-safe caching mechanism for slow/deterministic tool responses.
    Allows specifying customized TTLs per cached item.
    """
    def __init__(self, default_ttl: int = 300):  # Default 5 minutes TTL
        self._cache: Dict[str, dict] = {}
        self.default_ttl = default_ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        """Retrieves a cached value if it exists and has not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                if time.time() < entry["expires_at"]:
                    return entry["value"]
                else:
                    # Clean up expired entry
                    del self._cache[key]
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Stores a value in the cache with a designated TTL."""
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl_seconds
        
        with self._lock:
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at
            }

    def make_key(self, action: str, payload: Dict[str, Any]) -> str:
        """Creates a unique hash key based on the action and its arguments payload."""
        # Normalize and sort dictionary to ensure consistent hash representation
        try:
            payload_str = json.dumps(payload, sort_keys=True)
        except Exception:
            payload_str = str(payload)
            
        raw_key = f"{action}:{payload_str}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """Clears all elements in the cache."""
        with self._lock:
            self._cache.clear()

# Global instanced cache
response_cache = ResponseCache()
