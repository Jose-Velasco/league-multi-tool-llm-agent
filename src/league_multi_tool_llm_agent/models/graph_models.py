from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from league_multi_tool_llm_agent.db.rag_service import RagService
from league_multi_tool_llm_agent.graph.prompt_cache import PromptCache
from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient
from league_multi_tool_llm_agent.models.rag_models import RagSearchResult

# class IntentType(StrEnum):
#     PROFILE_ANALYSIS = "profile_analysis"
#     MATCH_HISTORY_ANALYSIS = "match_history_analysis"
#     CHAMPION_META = "champion_meta"
#     CHAMPION_RECOMMENDATION = "champion_recommendation"
#     SKIN_SEARCH = "skin_search"
#     MATCHUP_GUIDE = "matchup_guide"
#     CACHED_RESPONSE = "cached_response"
#     ERROR = "error"


class IntentType(StrEnum):
    CHAMPION_RECOMMENDATION = "champion_recommendation"
    SKIN_SEARCH = "skin_search"
    OPGG_MCP = "opgg_mcp"
    CACHED_RESPONSE = "cached_response"
    ERROR = "error"


class UserQueryPosition(StrEnum):
    ALL = "all"
    NONE = "none"
    TOP = "top"
    MID = "mid"
    JUNGLE = "jungle"
    ADC = "adc"
    SUPPORT = "support"


class UserQuery(BaseModel):
    query: str
    riot_id: str | None = None
    region: str | None = None
    champion: str | None = None
    position: UserQueryPosition | None = None
    opponent_champion: str | None = None


class FinalAnswer(BaseModel):
    answer: str
    used_cache: bool = False
    intent: IntentType | None = None
    raw_context_blocks: list[str] = Field(default_factory=list)
    synthesis_node_metadata: SynthesizedAnswer | None = None
    reflection_node_metadata: ReflectionResult | None = None
    revision_answer_node_metadata: RevisedAnswer | None = None


@dataclass
class AssistantState:
    original_query: str = ""
    parsed_query: UserQuery | None = None

    parsed_intent: ParsedIntent | None = None

    cache_key: str | None = None
    cache_hit: bool = False
    cached_response: str | None = None

    # collected evidence / context
    profile_text: str | None = None
    match_history_text: str | None = None
    champion_meta_text: str | None = None
    recommendation_text: str | None = None
    matchup_text: str | None = None
    opgg_mpc_text: str | None = None

    # RAG data
    rag_text: str | None = None
    rag_docs: list[RagSearchResult] | None = None

    merged_context_blocks: list[str] = field(default_factory=list)
    draft_answer: str | None = None
    final_answer: str | None = None

    # end of pipeline metadata
    synthesis_node_metadata: SynthesizedAnswer | None = None
    reflection_node_metadata: ReflectionResult | None = None
    revision_answer_node_metadata: RevisedAnswer | None = None

    last_error: str | None = None
    used_fallback_tool_selection: bool = False
    allowed_tool_names: list[str] | None = None

    failed_tool_name: str | None = None

    chat_history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class GraphDeps:
    opgg_client: OPGGMCPClient
    fallback_agent: Any
    parser_agent: Any
    synthesis_agent: Agent[None, SynthesizedAnswer]
    reflection_agent: Agent[None, ReflectionResult]
    revision_agent: Agent[None, RevisedAnswer]
    # Replace these with your real classes later
    prompt_cache: PromptCache | None = None
    controller: object | None = None
    rag_service: RagService | None = None
    llm_service: object | None = None


class ParsedIntent(BaseModel):
    intent: IntentType
    role_preference: str | None = None
    aesthetic_preference: str | None = None
    personality_preference: str | None = None
    playstyle_preference: str | None = None
    difficulty_preference: str | None = None
    query_for_rag: str = Field(description="Search query to use for retrieval")


class SynthesizedAnswer(BaseModel):
    """Final answer generated for the user."""

    answer: str = Field(
        description="User-facing response. Should be concise, helpful, and grounded."
    )
    recommended_items: list[str] = Field(
        default_factory=list,
        description="Champions or skins recommended in the answer.",
    )
    # used_context: bool = Field(description="Whether retrieved/tool context was used.")
    used_context: bool = Field(
        default=False,
        description="Whether retrieved/tool context was used.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence that the answer satisfies the user request.",
    )
    # confidence: float = Field(
    #     ge=0.0,
    #     le=1.0,
    #     description="Confidence that the answer satisfies the user request.",
    # )


class ReflectionResult(BaseModel):
    """Quality check for a synthesized answer."""

    approved: bool = Field(
        description="Whether the answer is good enough to return to the user."
    )
    needs_revision: bool = Field(
        description="Whether the synthesis node should revise the answer."
    )
    relevance_score: int = Field(ge=1, le=5)
    groundedness_score: int = Field(ge=1, le=5)
    safety_score: int = Field(ge=1, le=5)
    issues: list[str] = Field(
        default_factory=list,
        description="Brief problems found in the answer.",
    )
    revision_instructions: str | None = Field(
        default=None,
        description="Specific instructions for improving the answer if revision is needed.",
    )


class RevisedAnswer(BaseModel):
    """Revised version of a synthesized answer."""

    revised_answer: str = Field(description="Improved user-facing response.")

    changes_made: list[str] = Field(
        default_factory=list,
        description="Short list of improvements applied.",
    )

    addressed_issues: bool = Field(
        description="Whether the reflection issues were addressed."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the revised answer is improved.",
    )
