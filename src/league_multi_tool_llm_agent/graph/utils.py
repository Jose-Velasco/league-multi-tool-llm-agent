from __future__ import annotations

from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.graph_models import IntentType, ParsedIntent


async def route_intent(query: str) -> IntentType:
    q = query.lower()

    if "build" in q or "counter" in q or "tier" in q or "meta" in q:
        return IntentType.CHAMPION_META

    if any(
        w in q
        for w in [
            "recommend",
            "suggest",
            "main",
            "what should i play",
            "who should i play",
        ]
    ):
        return IntentType.CHAMPION_RECOMMENDATION

    if "matchup" in q or "vs " in q:
        return IntentType.MATCHUP_GUIDE

    if "match history" in q or "recent games" in q or "how am i doing" in q:
        return IntentType.MATCH_HISTORY_ANALYSIS

    if "rank" in q or "profile" in q or "lp" in q:
        return IntentType.PROFILE_ANALYSIS

    if "skin" in q:
        return IntentType.SKIN_SEARCH

    return IntentType.CHAMPION_RECOMMENDATION


async def parse_intent_with_fallback(
    user_prompt: str,
    parser_agent,
) -> ParsedIntent:
    try:
        result = await parser_agent.run(user_prompt)
        parsed: ParsedIntent = result.output

        # Safety: if LLM gives bad/unknown intent => fallback
        if parsed.intent not in {
            IntentType.CHAMPION_RECOMMENDATION,
            IntentType.SKIN_SEARCH,
            IntentType.CHAMPION_META,
            IntentType.MATCHUP_GUIDE,
            IntentType.MATCH_HISTORY_ANALYSIS,
            IntentType.PROFILE_ANALYSIS,
        }:
            fallback_intent = await route_intent(user_prompt)
            parsed.intent = fallback_intent

        return parsed

    except Exception:
        fallback_intent = await route_intent(user_prompt)

        return ParsedIntent(
            intent=fallback_intent,
            query_for_rag=user_prompt,
        )


def build_ollama_agent_model(
    model_name: str = "gemma3:4b-it-qat",
    ollama_provider_config: OllamaProviderConfig | None = None,
) -> OllamaModel:
    """
    Build a PydanticAI OllamaModel to be used in a paydanticAI Agent.

    """
    if ollama_provider_config:
        model = OllamaModel(
            model_name,
            provider=OllamaProvider(
                base_url=ollama_provider_config.OLLAMA_BASE_URL,
                api_key=ollama_provider_config.OLLAMA_API_KEY,
            ),
        )
    else:
        model = OllamaModel(model_name)
    return model


# async def route_intent(query: str) -> IntentType:
#     q = query.lower()
#     if "build" in q or "counter" in q or "tier" in q or "meta" in q:
#         return IntentType("champion_meta")
#     if "recommend" in q or "beginner" in q or "what should i play" in q:
#         return IntentType("recommendation")
#     if "matchup" in q or "vs " in q:
#         return IntentType("matchup_guide")
#     if "match history" in q or "recent games" in q or "how am i doing" in q:
#         return IntentType("match_history_analysis")
#     if "rank" in q or "profile" in q or "lp" in q:
#         return IntentType("profile_analysis")
#     return IntentType("recommendation")


# async def parse_intent_with_fallback(user_prompt: str, parser_agent) -> ParsedIntent:
#     try:
#         result = await parser_agent.run(user_prompt)
#         return result.output
#     except Exception:
#         text = user_prompt.lower()

#         if any(
#             w in text for w in ["recommend", "suggest", "main", "who should i play"]
#         ):
#             return ParsedIntent(
#                 # intent="champion_recommendation",
#                 intent=IntentType.CHAMPION_RECOMMENDATION,
#                 query_for_rag=user_prompt,
#             )

#         if "skin" in text:
#             return ParsedIntent(
#                 # intent="skin_search",
#                 intent=IntentType.SKIN_SEARCH,
#                 query_for_rag=user_prompt,
#             )

#         return ParsedIntent(
#             intent=IntentType.ERROR,
#             query_for_rag=user_prompt,
#         )
