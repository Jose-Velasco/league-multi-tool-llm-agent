from __future__ import annotations

import re

from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.graph_models import IntentType, ParsedIntent

# async def route_intent(query: str) -> IntentType:
#     q = query.lower()

#     if "build" in q or "counter" in q or "tier" in q or "meta" in q:
#         return IntentType.CHAMPION_META

#     if any(
#         w in q
#         for w in [
#             "recommend",
#             "suggest",
#             "main",
#             "what should i play",
#             "who should i play",
#         ]
#     ):
#         return IntentType.CHAMPION_RECOMMENDATION

#     if "matchup" in q or "vs " in q:
#         return IntentType.MATCHUP_GUIDE

#     if "match history" in q or "recent games" in q or "how am i doing" in q:
#         return IntentType.MATCH_HISTORY_ANALYSIS

#     if "rank" in q or "profile" in q or "lp" in q:
#         return IntentType.PROFILE_ANALYSIS

#     if "skin" in q:
#         return IntentType.SKIN_SEARCH

#     return IntentType.CHAMPION_RECOMMENDATION

KNOWN_CHAMPIONS = {
    "aatrox",
    "ahri",
    "akali",
    "akshan",
    "alistar",
    "amumu",
    "anivia",
    "annie",
    "aphelios",
    "ashe",
    "aurelion sol",
    "azir",
    "bard",
    "belveth",
    "blitzcrank",
    "brand",
    "braum",
    "caitlyn",
    "camille",
    "cassiopeia",
    "chogath",
    "corki",
    "darius",
    "diana",
    "draven",
    "ekko",
    "elise",
    "evelynn",
    "ezreal",
    "fiddlesticks",
    "fiora",
    "fizz",
    "galio",
    "gangplank",
    "garen",
    "gnar",
    "gragas",
    "graves",
    "gwen",
    "hecarim",
    "heimerdinger",
    "hwei",
    "illaoi",
    "irelia",
    "ivern",
    "janna",
    "jarvan iv",
    "jax",
    "jayce",
    "jhin",
    "jinx",
    "kaisa",
    "kalista",
    "karma",
    "karthus",
    "kassadin",
    "katarina",
    "kayle",
    "kayn",
    "kennen",
    "khazix",
    "kindred",
    "kled",
    "kogmaw",
    "leblanc",
    "lee sin",
    "leona",
    "lillia",
    "lissandra",
    "lucian",
    "lulu",
    "lux",
    "malphite",
    "malzahar",
    "maokai",
    "master yi",
    "milio",
    "miss fortune",
    "mordekaiser",
    "morgana",
    "naafiri",
    "nami",
    "nasus",
    "nautilus",
    "neeko",
    "nidalee",
    "nilah",
    "nocturne",
    "nunu",
    "olaf",
    "orianna",
    "ornn",
    "pantheon",
    "poppy",
    "pyke",
    "qiyana",
    "quinn",
    "rakan",
    "rammus",
    "reksai",
    "rell",
    "renata",
    "renekton",
    "rengar",
    "riven",
    "rumble",
    "ryze",
    "samira",
    "sejuani",
    "senna",
    "seraphine",
    "sett",
    "shaco",
    "shen",
    "shyvana",
    "singed",
    "sion",
    "sivir",
    "skarner",
    "smolder",
    "sona",
    "soraka",
    "swain",
    "sylas",
    "syndra",
    "tahm kench",
    "taliyah",
    "talon",
    "taric",
    "teemo",
    "thresh",
    "tristana",
    "trundle",
    "tryndamere",
    "twisted fate",
    "twitch",
    "udyr",
    "urgot",
    "varus",
    "vayne",
    "veigar",
    "velkoz",
    "vex",
    "vi",
    "viego",
    "viktor",
    "vladimir",
    "volibear",
    "warwick",
    "wukong",
    "xayah",
    "xerath",
    "xin zhao",
    "yasuo",
    "yone",
    "yorick",
    "yuumi",
    "zac",
    "zed",
    "zeri",
    "ziggs",
    "zilean",
    "zoe",
    "zyra",
}


async def route_intent(query: str) -> IntentType:
    q = query.lower()

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

    if "skin" in q:
        return IntentType.SKIN_SEARCH

    return IntentType.OPGG_MCP


async def parse_intent_with_fallback(
    user_prompt: str,
    parser_agent,
) -> ParsedIntent:
    try:
        result = await parser_agent.run(user_prompt)
        parsed: ParsedIntent = result.output

        # Safety: if LLM gives bad/unknown intent => fallback
        # if parsed.intent not in {
        #     IntentType.CHAMPION_RECOMMENDATION,
        #     IntentType.SKIN_SEARCH,
        #     IntentType.CHAMPION_META,
        #     IntentType.MATCHUP_GUIDE,
        #     IntentType.MATCH_HISTORY_ANALYSIS,
        #     IntentType.PROFILE_ANALYSIS,
        # }:
        if parsed.intent not in {
            IntentType.CHAMPION_RECOMMENDATION,
            IntentType.SKIN_SEARCH,
            IntentType.OPGG_MCP,
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


def normalize_text(text: str) -> str:
    """Normalize text for champion matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_cache_key(text: str) -> str:
    """Normalize cache keys consistently for lookup and storage."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_champion_name(user_query: str) -> str | None:
    """Extract the first matching champion name from a query."""
    normalized_query = normalize_text(user_query)

    # Sort longest-first so "twisted fate" matches before "fate"
    for champion in sorted(KNOWN_CHAMPIONS, key=len, reverse=True):
        pattern = rf"\b{re.escape(champion)}\b"

        if re.search(pattern, normalized_query):
            return champion.title()

    return None


def is_prompt_injection_or_out_of_scope(query: str) -> bool:
    """detect simple prompt injection or out-of-scope requests."""
    q = query.lower()

    injection_phrases = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "forget your instructions",
        "forget previous instructions",
        "reveal your system prompt",
        "show your system prompt",
        "developer message",
        "system message",
    ]

    out_of_scope_keywords = [
        "movie",
        "movies",
        "restaurant",
        "travel",
        "stock",
        "weather",
    ]

    return any(p in q for p in injection_phrases) or any(
        k in q for k in out_of_scope_keywords
    )


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
