from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig


class MCPToolSelection(BaseModel):
    """Structured tool choice returned by the fallback LLM agent."""

    tool_name: str = Field(description="Exact MCP tool name to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected MCP tool.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the tool choice.",
    )


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class FallbackMCPAgentError(RuntimeError):
    """Raised when the fallback MCP agent cannot safely choose or run a tool."""


def _compact_tool_registry(tool_registry: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert the cached MCP registry into a compact prompt-friendly format.

    Expected tool objects come from your OPGGMCPClient.refresh_tool_registry().
    """

    compact_tools: list[dict[str, Any]] = []

    for name, tool in tool_registry.items():
        input_schema = getattr(tool, "input_schema", None) or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        compact_tools.append(
            {
                "name": name,
                "description": getattr(tool, "description", None),
                "required_args": required,
                "optional_args": [k for k in properties.keys() if k not in required],
            }
        )

    return compact_tools


def _format_chat_history(
    chat_history: list[ChatMessage] | list[dict[str, str]] | None,
    *,
    max_messages: int = 8,
) -> str:
    """Format recent chat history for the fallback prompt."""
    if not chat_history:
        return "(none)"

    trimmed = chat_history[-max_messages:]
    lines: list[str] = []

    for msg in trimmed:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = msg.role
            content = msg.content

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _validate_selection_against_registry(
    selection: MCPToolSelection,
    tool_registry: dict[str, Any],
) -> None:
    """Validate tool name and arguments against the cached registry."""
    if selection.tool_name not in tool_registry:
        raise FallbackMCPAgentError(
            f"Unknown MCP tool selected by fallback agent: {selection.tool_name}"
        )

    tool = tool_registry[selection.tool_name]
    input_schema = getattr(tool, "input_schema", None) or {}

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    allowed_args = set(properties.keys())
    provided_args = set(selection.arguments.keys())

    unknown_args = sorted(provided_args - allowed_args)
    if unknown_args:
        raise FallbackMCPAgentError(
            f"Tool '{selection.tool_name}' received unsupported args: {unknown_args}"
        )

    missing_required = sorted(arg for arg in required if arg not in selection.arguments)
    if missing_required:
        raise FallbackMCPAgentError(
            f"Tool '{selection.tool_name}' missing required args: {missing_required}"
        )


def build_fallback_agent(
    model_name: str = "gemma3:4b-it-qat",
    ollama_provider_config: OllamaProviderConfig | None = None,
) -> Agent[None, MCPToolSelection]:
    """
    Build a PydanticAI agent that returns validated MCPToolSelection output.

    Current PydanticAI uses output_type rather than result_type.
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

    instructions = (
        "You are selecting exactly one OP.GG MCP tool for a League of Legends assistant.\n"
        "Choose the single best tool for the user's request.\n"
        "Return only a valid structured MCPToolSelection output.\n"
        "Rules:\n"
        "- Use the exact tool name from the provided tool list.\n"
        "- Only include arguments that exist in the selected tool schema.\n"
        "- Include all required arguments if they can be inferred from the user query and chat history.\n"
        "- Do not invent tools.\n"
        "- Do not invent unsupported argument names.\n"
        "- Prefer the most direct tool for the request.\n"
        "- If the request is ambiguous, choose the safest reasonable tool and best-effort arguments.\n"
    )

    return Agent(
        model=model,
        output_type=MCPToolSelection,
        instructions=instructions,
    )


async def fallback_mcp_agent(
    *,
    user_query: str,
    chat_history: list[ChatMessage] | list[dict[str, str]] | None,
    tool_registry: dict[str, Any],
    mcp_client: Any,
    fallback_agent: Agent[None, MCPToolSelection],
    allowed_tool_names: list[str] | None = None,
) -> str:
    """
    LLM-guided fallback MCP tool selection.

    Args:
        user_query: Current user request.
        chat_history: Recent conversation history.
        tool_registry: Cached registry from OPGGMCPClient.refresh_tool_registry().
        mcp_client: Your OPGGMCPClient instance.
        fallback_agent: PydanticAI agent with output_type=MCPToolSelection.
        allowed_tool_names: Optional whitelist for this node/branch.

    Returns:
        Extracted text from the selected MCP tool call.
    """
    if not tool_registry:
        tool_registry = await mcp_client.refresh_tool_registry()

    candidate_registry = tool_registry
    if allowed_tool_names is not None:
        candidate_registry = {
            name: tool
            for name, tool in tool_registry.items()
            if name in allowed_tool_names
        }

    if not candidate_registry:
        raise FallbackMCPAgentError("No MCP tools available for fallback selection.")

    compact_tools = _compact_tool_registry(candidate_registry)
    history_text = _format_chat_history(chat_history)

    prompt = f"""
        User query:
        {user_query}

        Recent chat history:
        {history_text}

        Available MCP tools:
        {compact_tools}

        Select the single best tool and arguments.
        """.strip()

    selection: MCPToolSelection | None = None
    try:
        # run_result = await fallback_agent.run(prompt)
        # selection = run_result.output
        selection = (await fallback_agent.run(prompt)).output
        result = await mcp_client.call_tool(selection.tool_name, selection.arguments)
    except ValidationError as e:
        error_message = f"Fallback agent failed before tool execution: {e}"
        if selection:
            error_message += selection.model_dump_json()

        raise FallbackMCPAgentError(error_message) from e
    except Exception as e:
        error_message = f"Fallback agent failed before tool execution: {e}"
        if selection:
            error_message += selection.model_dump_json()

        raise FallbackMCPAgentError(error_message) from e

    _validate_selection_against_registry(selection, candidate_registry)

    try:
        result = await mcp_client.call_tool(selection.tool_name, selection.arguments)
        return mcp_client.extract_text(result)
    except Exception as e:
        raise FallbackMCPAgentError(
            f"Fallback MCP tool execution failed: tool={selection.tool_name}, {selection.arguments = }, {selection.reasoning = }, error={e}"
        ) from e
