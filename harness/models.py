from abc import ABC, abstractmethod
from openai import OpenAI
# import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


# BASE Interface
class BaseLLMAdapter(ABC):
    """Every model adapter must implement this"""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human readable name for reports"""
        pass


# ─── Ollama (Local, Free) ─────────────────────────────────────────
class OllamaAdapter(BaseLLMAdapter):
    """Talks to local Ollama server. OpenAI-compatible API."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key='ollama' #use any key for Ollama
        )

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


# ─── Groq (Free API Tier) ─────────────────────────────────────────
class GroqAdapter(BaseLLMAdapter):
    """Groq cloud — fast inference, free tier, OpenAI-compatible."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    @property
    def model_id(self) -> str:
        return f"groq/{self.model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


# ─── Registry ─────────────────────────────────────────────────────
# YAML tasks reference providers by name — this maps names to adapters.
def get_adapter(provider_name: str) -> BaseLLMAdapter:
    registry = {
        "ollama":       OllamaAdapter(),
        "groq":         GroqAdapter(),
    }
    adapter = registry.get(provider_name)
    if adapter is None:
        raise ValueError(f"Unknown provider: '{provider_name}'")
    return adapter