from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from league_multi_tool_llm_agent.db.rag_service import RagService
from league_multi_tool_llm_agent.graph.prompt_cache import PromptCache
from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient
from league_multi_tool_llm_agent.models.rag_models import RagSearchResult


class IntentType(StrEnum):
    PROFILE_ANALYSIS = "profile_analysis"
    MATCH_HISTORY_ANALYSIS = "match_history_analysis"
    CHAMPION_META = "champion_meta"
    # RECOMMENDATION = "recommendation"
    CHAMPION_RECOMMENDATION = "champion_recommendation"
    SKIN_SEARCH = "skin_search"
    MATCHUP_GUIDE = "matchup_guide"
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

    rag_text: str | None = None
    rag_docs: list[RagSearchResult] | None = None

    merged_context_blocks: list[str] = field(default_factory=list)
    draft_answer: str | None = None
    final_answer: str | None = None

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
