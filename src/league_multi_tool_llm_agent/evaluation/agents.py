from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from league_multi_tool_llm_agent.graph.utils import build_ollama_agent_model
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig


class JudgeScore(BaseModel):
    """LLM-as-judge evaluation output."""

    relevance: int = Field(ge=1, le=5)
    explanation_quality: int = Field(ge=1, le=5)
    personalization: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    notes: str = Field(default="")


@dataclass
class EvalCaseResult:
    """Single evaluation row."""

    query: str
    condition: str
    model_name: str
    use_rag: bool
    answer: str
    retrieved_context: str
    latency_seconds: float
    relevance: int | None = None
    explanation_quality: int | None = None
    personalization: int | None = None
    groundedness: int | None = None
    judge_notes: str = ""


def build_recommendation_litellm_agent(
    model_name: str, ollama_provider_config: OllamaProviderConfig
) -> Agent[None, str]:
    """Build an LLM agent used to generate recommendations with LiteLLM."""
    model = build_ollama_agent_model(
        model_name=model_name,
        ollama_provider_config=ollama_provider_config,
    )
    # Use LiteLLM instead of PydanticAI Agent
    return Agent(
        model=model,
        output_type=str,  # Output is plain text, no shape constraints
        instructions=(
            "You are a League of Legends recommendation assistant. "
            "Recommend champions or skins based on the user's preferences. "
            "Give concise but useful reasoning. "
            "When context is provided, ground your answer in that context."
        ),
    )


# Agent Builders
def build_recommendation_agent(
    model_name: str, ollama_provider_config: OllamaProviderConfig
) -> Agent[None, str]:
    """Build an LLM agent used to generate recommendations."""
    model = build_ollama_agent_model(
        model_name=model_name, ollama_provider_config=ollama_provider_config
    )
    # model = OllamaModel(model_name=model_name)

    return Agent(
        model=model,
        output_type=str,
        instructions=(
            "You are a League of Legends recommendation assistant. "
            "Recommend champions or skins based on the user's preferences. "
            "Give concise but useful reasoning. "
            "When context is provided, ground your answer in that context."
        ),
    )


def build_judge_agent(
    model_name: str, ollama_provider_config: OllamaProviderConfig
) -> Agent[None, JudgeScore]:
    """Build an LLM-as-judge agent for scoring outputs."""
    # model = OllamaModel(model_name=model_name)
    model = build_ollama_agent_model(
        model_name=model_name, ollama_provider_config=ollama_provider_config
    )

    return Agent(
        model=model,
        output_type=JudgeScore,
        retries=3,
        instructions=(
            "You are evaluating a League of Legends recommendation system.\n"
            "Return ONLY valid JSON.\n"
            "Do not include markdown, explanations, or text outside the JSON.\n"
            "Score each response from 1 to 5 using these criteria:\n"
            "- relevance: matches the user's request\n"
            "- explanation_quality: reasoning is clear and useful\n"
            "- personalization: uses the user's preferences\n"
            "- groundedness: answer is supported by retrieved context when context is present\n"
            "Be fair but not overly generous."
        ),
    )
