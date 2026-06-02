"""
Universal LLM Adapter.
Any LLM connected here gets the full Sophie/JARVIS superpower stack:
- Tool registry
- Memory system
- Trigger engine
- Trust registry
- Research engine
- Workspace sandbox

Add a new LLM by implementing the BaseLLMAdapter interface.
"""
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
)

class BaseLLMAdapter(ABC):
    """
    Implement this to connect any LLM to the Sophie framework.
    """
    model_name: str = "unknown"
    provider: str = "unknown"
    
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str,
             history: List[Dict] = None, **kwargs) -> str:
        """
        Single chat call. Returns the model's text response.
        history: list of {"role": "user"|"assistant", "content": "..."}
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if this model can currently be called (not rate-limited, key valid, etc.)"""
        pass
    
    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Optional: return estimated USD cost for this call."""
        return 0.0


class OpenRouterAdapter(BaseLLMAdapter):
    """Wraps the existing OpenRouter implementation."""
    provider = "openrouter"
    
    def __init__(self, model_id: str = ""):
        self.model_name = model_id
        self._on_cooldown_until = 0.0
    
    def chat(self, system_prompt: str, user_prompt: str,
             history: List[Dict] = None, **kwargs) -> str:
        from app.brain.tools import _raw_openrouter_chat
        requires_tools = kwargs.get("requires_tools", False)
        user_message = kwargs.get("user_message", "")
        return _raw_openrouter_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            user_message=user_message or user_prompt,
            requires_tools=requires_tools
        )
    
    def is_available(self) -> bool:
        return time.time() > self._on_cooldown_until


class GeminiAdapter(BaseLLMAdapter):
    """Wraps the Gemini implementation using the official SDK."""
    provider = "gemini"
    
    def __init__(self, model_id: str = ""):
        self.model_name = model_id or GEMINI_MODEL
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"
    
    def chat(self, system_prompt: str, user_prompt: str,
             history: List[Dict] = None, **kwargs) -> str:
        import google.generativeai as genai
        from app.brain.tools import log_to_sophie_brain
        
        log_to_sophie_brain("GEMINI_MODEL_ATTEMPT", f"model={self.model_name} path=chat")
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Create a model instance with system instruction
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt
            )
            
            # Format history if provided
            contents = []
            if history:
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            contents.append({"role": "user", "parts": [{"text": user_prompt}]})
            
            response = model.generate_content(contents)
            log_to_sophie_brain("GEMINI_MODEL_SUCCESS", f"model={self.model_name} path=chat")
            return response.text.strip()
        except Exception as e:
            log_to_sophie_brain("GEMINI_MODEL_FAILURE", f"model={self.model_name} path=chat error={e}")
            raise e
    
    def is_available(self) -> bool:
        return bool(GEMINI_API_KEY)


def get_llm_adapter() -> BaseLLMAdapter:
    """Returns the configured LLM adapter based on env variables."""
    if LLM_PROVIDER == "gemini":
        return GeminiAdapter()
    else:
        # Default to OpenRouter with config-defined model
        from app.config import OPENROUTER_MODEL
        return OpenRouterAdapter(OPENROUTER_MODEL)
