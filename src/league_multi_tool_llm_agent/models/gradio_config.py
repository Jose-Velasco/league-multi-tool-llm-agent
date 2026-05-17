from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field
from pydantic_graph import Graph
from pydantic_settings import BaseSettings, SettingsConfigDict

from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
)


class GradioAppSettings(BaseSettings):
    """Settings for the Gradio demo app."""

    model_config = SettingsConfigDict(
        env_file=".env.gradio",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OLLAMA_BASE_URL: str = "http://localhost:11434/v1/"
    OLLAMA_BASE_URL: str = "http://ollama:11434/v1/"

    SMALL_MODEL: str = "qwen3.5:2b-q4_K_M"
    LARGE_MODEL: str = "gemma4:e4b-it-q4_K_M"
    # DEFAULT_MODEL: str = "qwen3.5:2b-q4_K_M"
    DEFAULT_MODEL: str = "gemma4:e4b-it-q4_K_M"

    GRADIO_SHARE: bool = True
    GRADIO_DEBUG: bool = True
    GRADIO_SERVER_NAME: str | None = None
    GRADIO_SERVER_PORT: int | None = None

    EXAMPLES: list[str] = Field(
        default_factory=lambda: [
            "Recommend champion: I like strong female leads and dark aesthetics.",
            "I prefer supportive roles and teamwork, who should I main?",
            "Recommend cute skins for a support champion.",
            "Give me beginner-friendly mid champions.",
            "Suggest space themed skins with robotic vibes.",
        ]
    )


@dataclass
class AgentBundle:
    """Model-specific PydanticAI agents."""

    parser_agent: Any
    fallback_agent: Any
    synthesis_agent: Any
    reflection_agent: Any
    revision_agent: Any


@dataclass
class AppServices:
    """Long-lived services reused across Gradio requests."""

    settings: GradioAppSettings
    graph: Graph[AssistantState, GraphDeps, FinalAnswer]
    base_deps: GraphDeps
