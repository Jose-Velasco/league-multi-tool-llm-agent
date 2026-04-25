from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_settings import BaseSettings

LoLRegion = str
LoLLanguage = str
LoLPosition = Literal["all", "none", "top", "mid", "jungle", "adc", "support"]
LoLGameMode = Literal["ranked", "flex", "urf", "aram", "nexus_blitz"]


class OPGGMCPError(RuntimeError):
    """Raised when an OP.GG MCP call fails."""


# @dataclass(slots=True)
# class OPGGMCPConfig:
#     """Configuration for the OP.GG MCP client."""

#     server_url: str = "https://mcp-api.op.gg/mcp"
#     default_lang: LoLLanguage = "en_US"
#     default_match_limit: int = 10


class OPGGMCPConfig(BaseSettings):
    """Configuration for the OP.GG MCP client."""

    SERVER_URL: str = "https://mcp-api.op.gg/mcp"
    DEFAULT_LANG: LoLLanguage = "en_US"
    DEFAULT_MATCH_LIMIT: int = 10


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
