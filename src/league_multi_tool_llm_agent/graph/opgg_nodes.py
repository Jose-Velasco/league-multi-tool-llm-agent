# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Literal

# from pydantic import BaseModel, Field

# from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient


# class SummonerProfileRequest(BaseModel):
#     riot_id: str
#     region: str
#     lang: str = "en_US"


# class ChampionAnalysisRequest(BaseModel):
#     champion: str = Field(description="Champion in UPPER_SNAKE_CASE, e.g. AHRI")
#     position: Literal["all", "none", "top", "mid", "jungle", "adc", "support"]
#     game_mode: Literal["ranked", "flex", "urf", "aram", "nexus_blitz"] = "ranked"
#     lang: str = "en_US"


# class MatchHistoryRequest(BaseModel):
#     riot_id: str
#     region: str
#     limit: int = 10
#     lang: str = "en_US"


# @dataclass
# class OPGGDependencies:
#     client: OPGGMCPClient


# class SummonerProfileNode:
#     """Graph node for summoner profile lookups."""

#     async def run(self, deps: OPGGDependencies, request: SummonerProfileRequest) -> str:
#         result = await deps.client.get_summoner_profile(
#             riot_id=request.riot_id,
#             region=request.region,
#             lang=request.lang,
#         )
#         return deps.client.extract_text(result)


# class ChampionAnalysisNode:
#     """Graph node for champion analysis lookups."""

#     async def run(self, deps: OPGGDependencies, request: ChampionAnalysisRequest) -> str:
#         result = await deps.client.get_champion_analysis(
#             champion=request.champion,
#             position=request.position,
#             game_mode=request.game_mode,
#             lang=request.lang,
#         )
#         return deps.client.extract_text(result)


# class MatchHistoryNode:
#     """Graph node for recent match history lookups."""

#     async def run(self, deps: OPGGDependencies, request: MatchHistoryRequest) -> str:
#         result = await deps.client.list_summoner_matches(
#             riot_id=request.riot_id,
#             region=request.region,
#             limit=request.limit,
#             lang=request.lang,
#         )
#         return deps.client.extract_text(result)
