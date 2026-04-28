from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from litellm import acompletion
from pydantic_ai import Agent

from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig


@runtime_checkable
class RecommendationClient(Protocol):
    """Common interface for plain-text recommendation generation."""

    async def generate(self, prompt: str) -> str:
        """Generate a plain-text recommendation response."""
        ...


@dataclass
class LiteLLMRecommendationClient:
    """LiteLLM implementation of the recommendation client."""

    model_name: str
    ollama_provider_config: OllamaProviderConfig
    temperature: float = 0.2
    max_tokens: int = 350

    async def generate(self, prompt: str) -> str:
        api_base = self.ollama_provider_config.OLLAMA_BASE_URL.rstrip("/")
        api_base = api_base.removesuffix("/v1")

        response = await acompletion(
            model=f"ollama_chat/{self.model_name}",
            api_base=api_base,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a League of Legends recommendation assistant. "
                        "Recommend champions or skins based on the user's preferences. "
                        "Give concise but useful reasoning. "
                        "When context is provided, ground your answer in that context. "
                        "Return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            think=False,
        )

        return self.extract_litellm_text(response)

    def extract_litellm_text(self, response) -> str:
        """Extract text from LiteLLM response, including reasoning-model outputs."""
        message = response["choices"][0]["message"]

        content = getattr(message, "content", None)
        if content and str(content).strip():
            return str(content).strip()

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content and str(reasoning_content).strip():
            return str(reasoning_content).strip()

        # dict fallback just in case
        if isinstance(message, dict):
            content = message.get("content")
            if content and str(content).strip():
                return str(content).strip()

            reasoning_content = message.get("reasoning_content")
            if reasoning_content and str(reasoning_content).strip():
                return str(reasoning_content).strip()

        return ""


@dataclass
class PydanticAIRecommendationClient:
    """PydanticAI implementation of the recommendation client."""

    agent: Agent[None, str]

    async def generate(self, prompt: str) -> str:
        result = await self.agent.run(prompt)
        return str(result.output).strip()
