from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LoLRegion = Literal[
    "na", "kr", "euw", "eune", "jp", "br", "lan", "las", "oce", "tr", "ru"
]
LoLLanguage = str
LoLPosition = Literal["all", "none", "top", "mid", "jungle", "adc", "support"]
LoLGameMode = Literal["ranked", "flex", "urf", "aram", "nexus_blitz"]


class OPGGMCPError(RuntimeError):
    """Raised when an OP.GG MCP call fails."""


class OPGGMCPConfig(BaseSettings):
    """Configuration for the OP.GG MCP client."""

    model_config = SettingsConfigDict(
        env_prefix="OPGG_MCP_",
        env_file=".env.opgg",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVER_URL: str = "https://mcp-api.op.gg/mcp"
    DEFAULT_REGION: LoLRegion = "na"
    DEFAULT_LANG: LoLLanguage = "en_US"
    DEFAULT_POSITION: LoLPosition = "mid"
    DEFAULT_GAME_MODE: LoLGameMode = "ranked"
    DEFAULT_MATCH_LIMIT: int = 10
    DEFAULT_SYNERGY_POSITION: LoLPosition = "jungle"


@dataclass(slots=True)
class ToolInfo:
    """Minimal cached MCP tool metadata."""

    name: str
    description: str | None = None
    input_schema: dict | None = None


def parse_riot_id(riot_id: str) -> tuple[str, str]:
    """Split Riot ID in the form GameName#TAG into game_name and tag_line."""
    cleaned = riot_id.strip()

    if "#" not in cleaned:
        raise ValueError(f"Expected Riot ID in 'GameName#TAG' format, got: {riot_id!r}")

    game_name, tag_line = cleaned.split("#", 1)
    game_name = game_name.strip()
    tag_line = tag_line.strip()

    if not game_name or not tag_line:
        raise ValueError(f"Invalid Riot ID: {riot_id!r}")

    return game_name, tag_line
