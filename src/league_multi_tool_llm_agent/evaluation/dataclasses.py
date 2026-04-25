from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


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
