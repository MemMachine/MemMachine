"""
Enhanced model_utils.py - Multi-provider model management
Handles OpenAI, Anthropic Claude, and Google Gemini models
Replaces and extends the original model_utils.py
"""
from typing import Dict, List, Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
import anthropic
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL INFORMATION CLASSES
# ============================================================================

class ModelInfo:
    """Model information container"""
    def __init__(
        self,
        id: str,
        name: str,
        provider: str,
        context_window: int = 128000,
        supports_streaming: bool = True,
        supports_vision: bool = False,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0
    ):
        self.id = id
        self.name = name
        self.provider = provider
        self.context_window = context_window
        self.supports_streaming = supports_streaming
        self.supports_vision = supports_vision
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "context_window": self.context_window,
            "supports_streaming": self.supports_streaming,
            "supports_vision": self.supports_vision,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
        }


# ============================================================================
# ORIGINAL OPENAI FUNCTIONS (Keep for backward compatibility)
# ============================================================================

def get_model_display_name(model_id: str) -> str:
    """
    Convert model ID to user-friendly display name.
    Works for OpenAI, Anthropic, and Google models.
    """
    model_name_map = {
        # OpenAI
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o mini",
        "gpt-4-turbo": "GPT-4 Turbo",
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o1-preview": "o1-preview",
        
        # Anthropic
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
        "claude-3-opus-20240229": "Claude 3 Opus",
        "claude-3-sonnet-20240229": "Claude 3 Sonnet",
        "claude-3-haiku-20240307": "Claude 3 Haiku",
        
        # Google
        "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
        "gemini-exp-1206": "Gemini Exp 1206",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
        "gemini-1.5-flash": "Gemini 1.5 Flash",
        "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8B",
    }
    
    return model_name_map.get(model_id, model_id)


def is_chat_model(model_id: str) -> bool:
    """Check if a model is suitable for chat completions"""
    chat_patterns = ["gpt", "claude", "gemini", "o1"]
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in chat_patterns)


def get_fallback_models() -> List[Dict[str, str]]:
    """Return safe fallback models if API call fails"""
    return [
        {"id": "gpt-4o-mini", "name": "GPT-4o mini"},
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
    ]


# ============================================================================
# MODEL CATALOGS
# ============================================================================

OPENAI_MODELS = [
    ModelInfo("gpt-4o", "GPT-4o", "openai", 128000, True, True, 2.5, 10.0),
    ModelInfo("gpt-4o-mini", "GPT-4o mini", "openai", 128000, True, True, 0.15, 0.60),
    ModelInfo("gpt-4-turbo", "GPT-4 Turbo", "openai", 128000, True, True, 10.0, 30.0),
    ModelInfo("gpt-4", "GPT-4", "openai", 8192, True, False, 30.0, 60.0),
    ModelInfo("gpt-3.5-turbo", "GPT-3.5 Turbo", "openai", 16385, True, False, 0.5, 1.5),
    ModelInfo("o1", "o1", "openai", 200000, False, False, 15.0, 60.0),
    ModelInfo("o1-mini", "o1-mini", "openai", 128000, False, False, 3.0, 12.0),
    ModelInfo("o1-preview", "o1-preview", "openai", 128000, False, False, 15.0, 60.0),
]

ANTHROPIC_MODELS = [
    ModelInfo("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "anthropic", 200000, True, True, 3.0, 15.0),
    ModelInfo("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "anthropic", 200000, True, False, 1.0, 5.0),
    ModelInfo("claude-3-opus-20240229", "Claude 3 Opus", "anthropic", 200000, True, True, 15.0, 75.0),
    ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet", "anthropic", 200000, True, True, 3.0, 15.0),
    ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku", "anthropic", 200000, True, True, 0.25, 1.25),
]

GOOGLE_MODELS = [
    ModelInfo("gemini-2.0-flash-exp", "Gemini 2.0 Flash", "google", 1000000, True, True, 0.0, 0.0),
    ModelInfo("gemini-exp-1206", "Gemini Exp 1206", "google", 2000000, True, True, 0.0, 0.0),
    ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro", "google", 2000000, True, True, 1.25, 5.0),
    ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash", "google", 1000000, True, True, 0.075, 0.30),
    ModelInfo("gemini-1.5-flash-8b", "Gemini 1.5 Flash 8B", "google", 1000000, True, True, 0.0375, 0.15),
]


# ============================================================================
# DYNAMIC MODEL DISCOVERY
# ============================================================================

def fetch_available_models(api_key: str) -> List[Dict[str, str]]:
    """
    Fetch OpenAI models available to the API key.
    Kept for backward compatibility with original model_utils.py
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        
        chat_models = []
        for model in response.data:
            if is_chat_model(model.id):
                chat_models.append({
                    "id": model.id,
                    "name": get_model_display_name(model.id),
                })
        
        # Sort by priority
        def model_priority(model):
            model_id = model["id"].lower()
            if "gpt-4o" in model_id and "mini" not in model_id:
                return 1
            elif "gpt-4o-mini" in model_id:
                return 2
            elif "gpt-4-turbo" in model_id:
                return 3
            elif "gpt-4" in model_id:
                return 4
            elif "o1" in model_id:
                return 5
            else:
                return 99
        
        chat_models.sort(key=model_priority)
        
        # Remove duplicates
        seen = set()
        unique = []
        for m in chat_models:
            if m["name"] not in seen:
                seen.add(m["name"])
                unique.append(m)
        
        return unique
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {e}")
        return get_fallback_models()


def discover_openai_models(api_key: str) -> List[Dict]:
    """Discover OpenAI models with full info"""
    if not api_key:
        return []
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        available_ids = {m.id for m in response.data}
        
        return [m.to_dict() for m in OPENAI_MODELS if m.id in available_ids]
    except Exception as e:
        logger.error(f"OpenAI discovery failed: {e}")
        return []


def discover_anthropic_models(api_key: str) -> List[Dict]:
    """Discover Anthropic Claude models"""
    if not api_key:
        return []
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        accessible = []
        
        for model_info in ANTHROPIC_MODELS:
            try:
                # Test with minimal request
                client.messages.create(
                    model=model_info.id,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                accessible.append(model_info.to_dict())
            except (anthropic.NotFoundError, anthropic.PermissionDeniedError):
                pass
            except Exception:
                # Other errors likely mean model is accessible
                accessible.append(model_info.to_dict())
        
        return accessible
    except Exception as e:
        logger.error(f"Anthropic discovery failed: {e}")
        return []


def discover_google_models(api_key: str) -> List[Dict]:
    """Discover Google Gemini models"""
    if not api_key:
        return []
    
    try:
        genai.configure(api_key=api_key)
        available_models = genai.list_models()
        
        available_ids = set()
        for model in available_models:
            if 'generateContent' in model.supported_generation_methods:
                model_id = model.name.replace('models/', '')
                available_ids.add(model_id)
        
        return [m.to_dict() for m in GOOGLE_MODELS if m.id in available_ids]
    except Exception as e:
        logger.error(f"Google discovery failed: {e}")
        return []


# ============================================================================
# UNIFIED AI CLIENT FOR CHAT COMPLETIONS
# ============================================================================

class UnifiedAIClient:
    """Unified interface for all AI providers"""
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None
    ):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.google_key = google_key
        
        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        self.openai_async = AsyncOpenAI(api_key=openai_key) if openai_key else None
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
        
        if google_key:
            genai.configure(api_key=google_key)
    
    async def chat_completion_stream(
        self,
        provider: str,
        model: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from any provider"""
        
        if provider == "openai":
            async for chunk in self._openai_stream(model, messages, temperature, max_tokens):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._anthropic_stream(model, messages, temperature, max_tokens):
                yield chunk
        elif provider == "google":
            async for chunk in self._google_stream(model, messages, temperature, max_tokens):
                yield chunk
    
    async def _openai_stream(self, model, messages, temperature, max_tokens):
        """OpenAI streaming"""
        if not self.openai_async:
            raise ValueError("OpenAI API key not configured")
        
        stream = await self.openai_async.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _anthropic_stream(self, model, messages, temperature, max_tokens):
        """Anthropic Claude streaming"""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        import asyncio
        
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        with self.anthropic_client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message if system_message else None,
            messages=claude_messages
        ) as stream:
            for text in stream.text_stream:
                yield text
                await asyncio.sleep(0)
    
    async def _google_stream(self, model, messages, temperature, max_tokens):
        """Google Gemini streaming"""
        if not self.google_key:
            raise ValueError("Google API key not configured")
        
        import asyncio
        
        gemini_model = genai.GenerativeModel(model)
        
        chat_history = []
        prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                prompt = f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                if not chat_history:
                    prompt += msg["content"]
                else:
                    chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})
        
        chat = gemini_model.start_chat(history=chat_history[:-1] if len(chat_history) > 1 else [])
        
        response = chat.send_message(
            prompt if not chat_history else chat_history[-1]["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                await asyncio.sleep(0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_provider_from_model(model_id: str) -> Optional[str]:
    """Detect provider from model ID"""
    model_lower = model_id.lower()
    
    if any(m.id == model_id for m in OPENAI_MODELS):
        return "openai"
    elif any(m.id == model_id for m in ANTHROPIC_MODELS):
        return "anthropic"
    elif any(m.id == model_id for m in GOOGLE_MODELS):
        return "google"
    
    # Fallback pattern matching
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    
    return None


def get_all_models() -> List[ModelInfo]:
    """Get all available models across all providers"""
    return OPENAI_MODELS + ANTHROPIC_MODELS + GOOGLE_MODELS