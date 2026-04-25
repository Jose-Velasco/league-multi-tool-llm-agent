from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .field_presets import FieldPresets
from .types import (
    LoLGameMode,
    LoLLanguage,
    LoLPosition,
    LoLRegion,
    OPGGMCPConfig,
    OPGGMCPError,
    ToolInfo,
    parse_riot_id,
)


class OPGGMCPClient:
    """Thin wrapper around the OP.GG MCP server for stable app-facing usage."""

    def __init__(self, config: OPGGMCPConfig | None = None) -> None:
        self.config = config or OPGGMCPConfig()
        self._tool_registry: dict[str, ToolInfo] = {}

    @property
    def tool_registry(self) -> dict[str, ToolInfo]:
        """Return cached tool metadata."""
        return self._tool_registry

    @asynccontextmanager
    async def _session(self):
        """Open a fresh MCP session."""
        async with streamablehttp_client(self.config.SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def refresh_tool_registry(self) -> dict[str, ToolInfo]:
        """Fetch and cache the current MCP tool registry."""
        async with self._session() as session:
            result = await session.list_tools()

        registry: dict[str, ToolInfo] = {}
        for tool in result.tools:
            registry[tool.name] = ToolInfo(
                name=tool.name,
                description=getattr(tool, "description", None),
                input_schema=getattr(tool, "inputSchema", None),
            )

        self._tool_registry = registry
        return registry

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a raw MCP tool."""
        async with self._session() as session:
            result = await session.call_tool(tool_name, arguments)

        if getattr(result, "isError", False):
            raise OPGGMCPError(f"Tool call failed: {tool_name} | result={result}")

        return result

    @staticmethod
    def extract_text(result: Any) -> str:
        """Extract concatenated text content from an MCP result."""
        contents = getattr(result, "content", None) or []
        texts: list[str] = []

        for item in contents:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)

        return "\n".join(texts).strip()

    @staticmethod
    def _fields(
        custom_fields: Iterable[str] | None, default_fields: list[str]
    ) -> list[str]:
        """Use caller-supplied fields if present, otherwise defaults."""
        return list(custom_fields) if custom_fields else default_fields

    @staticmethod
    def _resolve_riot_identity(
        *,
        riot_id: str | None,
        game_name: str | None,
        tag_line: str | None,
    ) -> tuple[str, str]:
        """Normalize either Riot ID or explicit game_name/tag_line inputs."""
        if riot_id:
            return parse_riot_id(riot_id)
        if game_name and tag_line:
            return game_name, tag_line
        raise ValueError("Provide either riot_id or both game_name and tag_line")

    async def get_summoner_profile(
        self,
        *,
        riot_id: str | None = None,
        game_name: str | None = None,
        tag_line: str | None = None,
        region: LoLRegion = "na",
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get summoner profile and rank information."""
        game_name, tag_line = self._resolve_riot_identity(
            riot_id=riot_id,
            game_name=game_name,
            tag_line=tag_line,
        )
        return await self.call_tool(
            "lol_get_summoner_profile",
            {
                "game_name": game_name,
                "tag_line": tag_line,
                "region": region,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.SUMMONER_PROFILE_BASIC,
                ),
            },
        )

    async def list_summoner_matches(
        self,
        *,
        riot_id: str | None = None,
        game_name: str | None = None,
        tag_line: str | None = None,
        region: LoLRegion = "na",
        limit: int = 5,
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """List recent matches for a summoner."""
        game_name, tag_line = self._resolve_riot_identity(
            riot_id=riot_id,
            game_name=game_name,
            tag_line=tag_line,
        )
        return await self.call_tool(
            "lol_list_summoner_matches",
            {
                "game_name": game_name,
                "tag_line": tag_line,
                "region": region,
                "lang": lang or self.config.DEFAULT_LANG,
                "limit": limit or self.config.DEFAULT_MATCH_LIMIT,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.SUMMONER_MATCHES_BASIC,
                ),
            },
        )

    async def get_match_detail(
        self,
        *,
        region: LoLRegion = "na",
        game_id: str,
        created_at: str,
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get full expanded detail for one LoL match."""
        return await self.call_tool(
            "lol_get_summoner_game_detail",
            {
                "region": region,
                "game_id": game_id,
                "created_at": created_at,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.MATCH_DETAIL_BASIC,
                ),
            },
        )

    async def get_champion_analysis(
        self,
        *,
        champion: str,
        position: LoLPosition = "all",
        game_mode: LoLGameMode = "ranked",
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get champion meta, runes, items, and counters."""
        return await self.call_tool(
            "lol_get_champion_analysis",
            {
                "champion": champion,
                "position": position,
                "game_mode": game_mode,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.CHAMPION_ANALYSIS_CORE,
                ),
            },
        )

    async def get_lane_matchup_guide(
        self,
        *,
        my_champion: str,
        opponent_champion: str,
        position: LoLPosition = "all",
        lang: LoLLanguage | None = None,
    ) -> Any:
        """Get a specific lane matchup guide."""
        return await self.call_tool(
            "lol_get_lane_matchup_guide",
            {
                "my_champion": my_champion,
                "opponent_champion": opponent_champion,
                "position": position,
                "lang": lang or self.config.DEFAULT_LANG,
            },
        )

    # async def search_champion_meta(self, *, query: str) -> Any:
    #     """Run the LoL champion RAG search tool."""
    #     return await self.call_tool("lol_search_champion_meta", {"query": query})

    async def get_champion_synergies(
        self,
        *,
        champion: str,
        my_position: LoLPosition,
        synergy_position: LoLPosition,
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get synergy recommendations for a champion and teammate role."""
        return await self.call_tool(
            "lol_get_champion_synergies",
            {
                "champion": champion,
                "my_position": my_position,
                "synergy_position": synergy_position,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.CHAMPION_SYNERGIES,
                ),
            },
        )

    async def list_champion_details(
        self,
        *,
        champions: list[str],
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get champion lore, abilities, and stats for up to 10 champions."""
        if not champions:
            raise ValueError("champions must not be empty")
        if len(champions) > 10:
            raise ValueError("lol_list_champion_details supports at most 10 champions")

        return await self.call_tool(
            "lol_list_champion_details",
            {
                "champions": champions,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.CHAMPION_DETAILS_BASIC,
                ),
            },
        )

    async def list_champion_leaderboard(
        self,
        *,
        champion: str,
        region: LoLRegion = "na",
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get top champion players for a region."""
        return await self.call_tool(
            "lol_list_champion_leaderboard",
            {
                "champion": champion,
                "region": region,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.CHAMPION_LEADERBOARD_BASIC,
                ),
            },
        )

    async def list_champions(
        self,
        *,
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """List all champions."""
        return await self.call_tool(
            "lol_list_champions",
            {
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.CHAMPION_LIST_BASIC,
                ),
            },
        )

    async def list_lane_meta_champions(
        self,
        *,
        position: LoLPosition = "all",
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get lane-by-lane champion tier data."""
        return await self.call_tool(
            "lol_list_lane_meta_champions",
            {
                "position": position,
                "lang": lang or self.config.DEFAULT_LANG,
                "desired_output_fields": self._fields(
                    desired_output_fields,
                    FieldPresets.LANE_META_BASIC,
                ),
            },
        )

    async def list_discounted_skins(
        self,
        *,
        champion: str | None = None,
        lang: LoLLanguage | None = None,
        desired_output_fields: Iterable[str] | None = None,
    ) -> Any:
        """Get discounted skins, optionally filtered to one champion."""
        args: dict[str, Any] = {
            "lang": lang or self.config.DEFAULT_LANG,
            "desired_output_fields": self._fields(
                desired_output_fields,
                FieldPresets.DISCOUNTED_SKINS_BASIC,
            ),
        }
        if champion:
            args["champion"] = champion
        return await self.call_tool("lol_list_discounted_skins", args)

    async def list_items(
        self,
        *,
        lang: LoLLanguage | None = None,
        map_name: str = "SUMMONERS_RIFT",
    ) -> Any:
        """List localized LoL items for a map."""
        return await self.call_tool(
            "lol_list_items",
            {
                "lang": lang or self.config.DEFAULT_LANG,
                "map": map_name,
            },
        )

    async def get_pro_player_riot_id(
        self,
        *,
        player_name: str,
        region: str = "na",
        return_suggestions: bool = False,
    ) -> Any:
        """Resolve a pro player alias to Riot ID and team metadata."""
        return await self.call_tool(
            "lol_get_pro_player_riot_id",
            {
                "player_name": player_name,
                "region": region,
                "return_suggestions": return_suggestions,
            },
        )

    async def esports_list_schedules(self) -> Any:
        """Get upcoming LoL esports schedules."""
        return await self.call_tool("lol_esports_list_schedules", {})

    async def esports_list_team_standings(self, *, short_name: str) -> Any:
        """Get current standings for a LoL esports league."""
        return await self.call_tool(
            "lol_esports_list_team_standings",
            {"short_name": short_name},
        )
