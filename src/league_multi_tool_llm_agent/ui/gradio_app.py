from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr
from sqlalchemy import URL

# from league_multi_tool_llm_agent.llm.agents import build_response_agent
from league_multi_tool_llm_agent.db.rag_service import RagService, RagSettings
from league_multi_tool_llm_agent.graph.agents import build_parser_agent
from league_multi_tool_llm_agent.graph.capability_nodes import (
    BuildInitialAssistantStateNode,
    # league_assistant_graph,
)
from league_multi_tool_llm_agent.graph.catch_all_node import build_fallback_agent
from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    GraphDeps,
    UserQuery,
)


@dataclass
class AppServices:
    """Long-lived services reused across Gradio requests."""

    deps: GraphDeps


def history_to_messages(
    history: list[dict[str, str]] | list[list[str]],
) -> list[dict[str, str]]:
    """Convert Gradio history into simple role/content messages."""
    messages: list[dict[str, str]] = []

    for item in history or []:
        # Newer Gradio type="messages"
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append({"role": role, "content": content})
            continue

        # Older tuple/list style: [user, assistant]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, assistant_msg = item[0], item[1]
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": str(assistant_msg)})

    return messages


async def run_graph_for_message(
    *,
    message: str,
    history: list[Any],
    services: AppServices,
) -> str:
    """Run the Pydantic Graph for one Gradio chat turn."""
    state = AssistantState(
        original_query=message,
        chat_history=history_to_messages(history),
    )

    user_query = UserQuery(query=message)

    try:
        result = await league_assistant_graph.run(
            start_node=BuildInitialAssistantStateNode(user_input=user_query),
            state=state,
            deps=services.deps,
        )
        return result.output.answer

    except Exception:
        # Keep this user-facing, not a stack trace.
        return (
            "I ran into an issue while processing that request. "
            "Try rephrasing it or asking for a champion recommendation, skin search, or build suggestion."
        )


def build_services() -> AppServices:
    """Initialize reusable app dependencies once."""

    opgg_client = OPGGMCPClient()

    rag_settings = RagSettings()
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=rag_settings.db_user,
        password=rag_settings.db_password,
        host=rag_settings.db_host,
        port=rag_settings.db_port,
        database=rag_settings.db_name,
    )
    rag_service = RagService(db_url=db_url, settings=rag_settings)

    # parser_agent = build_parser_agent()
    # llm_agent = build_response_agent()

    local_llm_config = OllamaProviderConfig(OLLAMA_BASE_URL="http://ollama:11434/v1/")
    parser_agent = build_parser_agent(
        "gemma4:e2b-it-q4_K_M",
        ollama_provider_config=local_llm_config,
    )

    fallback_agent = build_fallback_agent(
        # "gemma4:e4b-it-q4_K_M", ollama_provider_config=local_llm_config
        "gemma4:e2b-it-q4_K_M",
        ollama_provider_config=local_llm_config,
    )

    deps = GraphDeps(
        opgg_client=opgg_client,
        parser_agent=parser_agent,
        # llm_service=llm_agent, # still work in progress place holder but not yet implemented
        rag_service=rag_service,
        fallback_agent=fallback_agent,
    )

    return AppServices(deps=deps)


services = build_services()


async def chat_fn(message: str, history: list[Any]) -> str:
    """Gradio chat callback."""
    return await run_graph_for_message(
        message=message,
        history=history,
        services=services,
    )


demo = gr.ChatInterface(
    fn=chat_fn,
    title="League Multi-Tool LLM Agent",
    description=(
        "Ask for champion recommendations, skins, builds, counters, or playstyle advice."
    ),
    examples=[
        "I like strong female leads and dark aesthetics, recommend a champion.",
        "I prefer supportive roles and teamwork, who should I main?",
        "Recommend cute skins for support champions.",
        "Give me beginner-friendly mid champions.",
    ],
    type="messages",
)

demo.launch(share=True)
