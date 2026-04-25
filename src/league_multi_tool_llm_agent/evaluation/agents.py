from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

from league_multi_tool_llm_agent.evaluation.dataclasses import JudgeScore


# Agent Builders
def build_recommendation_agent(model_name: str) -> Agent[None, str]:
    """Build an LLM agent used to generate recommendations."""
    model = OllamaModel(model_name=model_name)

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


def build_judge_agent(model_name: str) -> Agent[None, JudgeScore]:
    """Build an LLM-as-judge agent for scoring outputs."""
    model = OllamaModel(model_name=model_name)

    return Agent(
        model=model,
        output_type=JudgeScore,
        instructions=(
            "You are evaluating a League of Legends recommendation system.\n"
            "Score each response from 1 to 5 using these criteria:\n"
            "- relevance: matches the user's request\n"
            "- explanation_quality: reasoning is clear and useful\n"
            "- personalization: uses the user's preferences\n"
            "- groundedness: answer is supported by retrieved context when context is present\n"
            "Be fair but not overly generous."
        ),
    )
