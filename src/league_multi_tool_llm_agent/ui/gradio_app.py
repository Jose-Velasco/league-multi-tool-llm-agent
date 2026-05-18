import logging
from functools import lru_cache
from pprint import pformat
from typing import Any

import gradio as gr
from pydantic import BaseModel
from pydantic_graph import Graph
from sqlalchemy import URL

from league_multi_tool_llm_agent.db.llm_utils import EmbeddingClient
from league_multi_tool_llm_agent.db.rag_service import RagService
from league_multi_tool_llm_agent.graph.agents import (
    build_parser_agent,
    build_reflection_agent,
    build_revision_agent,
    build_synthesis_agent,
)
from league_multi_tool_llm_agent.graph.capability_nodes import (
    BuildInitialAssistantStateNode,
    OPGG_MPC_Node,
    ParseAndRouteNode,
    PromptCacheCheckNode,
    RecommendationNode,
    ReturnCachedResponseNode,
    SkinSearchNode,
)
from league_multi_tool_llm_agent.graph.catch_all_node import build_fallback_agent
from league_multi_tool_llm_agent.graph.prompt_cache import InMemoryDictCache
from league_multi_tool_llm_agent.graph.prompting_techniques import (
    AggregationNode,
    ReflectionNode,
    RevisionNode,
    StorePromptCacheNode,
    SynthesisNode,
)
from league_multi_tool_llm_agent.graph.system_nodes import ErrorRecoveryNode
from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient
from league_multi_tool_llm_agent.integrations.opgg.types import OPGGMCPConfig
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.gradio_config import (
    AgentBundle,
    AppServices,
    GradioAppSettings,
)
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    UserQuery,
)
from league_multi_tool_llm_agent.models.rag_configs import RagSettings

logger = logging.getLogger(__name__)


def build_league_assistant_graph() -> Graph[AssistantState, GraphDeps, FinalAnswer]:
    """Build the assistant graph."""
    return Graph(
        nodes=(
            BuildInitialAssistantStateNode,
            PromptCacheCheckNode,
            ReturnCachedResponseNode,
            ParseAndRouteNode,
            RecommendationNode,
            SkinSearchNode,
            OPGG_MPC_Node,
            SynthesisNode,
            StorePromptCacheNode,
            ErrorRecoveryNode,
            AggregationNode,
            ReflectionNode,
            RevisionNode,
        ),
        state_type=AssistantState,
        run_end_type=FinalAnswer,
    )


def history_to_messages(history: list[Any]) -> list[dict[str, str]]:
    """Convert Gradio history into role/content messages."""
    messages: list[dict[str, str]] = []

    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            if content:
                messages.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, assistant_msg = item[0], item[1]
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": str(assistant_msg)})

    return messages


def build_rag_service() -> RagService:
    """Build PostgreSQL + pgvector RAG service."""
    rag_settings = RagSettings()

    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=rag_settings.db_user,
        password=rag_settings.db_password,
        host=rag_settings.db_host,
        port=rag_settings.db_port,
        database=rag_settings.db_name,
    )

    return RagService(
        db_url=db_url,
        settings=rag_settings,
        embedder=EmbeddingClient(),
    )


@lru_cache(maxsize=8)
def build_agent_bundle(
    model_name: str,
    ollama_base_url: str,
) -> AgentBundle:
    """Build and cache model-specific agents."""
    provider_config = OllamaProviderConfig(
        OLLAMA_BASE_URL=ollama_base_url,
    )

    return AgentBundle(
        parser_agent=build_parser_agent(
            model_name,
            ollama_provider_config=provider_config,
        ),
        fallback_agent=build_fallback_agent(
            model_name,
            ollama_provider_config=provider_config,
        ),
        synthesis_agent=build_synthesis_agent(
            model_name,
            ollama_provider_config=provider_config,
        ),
        reflection_agent=build_reflection_agent(
            model_name,
            ollama_provider_config=provider_config,
        ),
        revision_agent=build_revision_agent(
            model_name,
            ollama_provider_config=provider_config,
        ),
    )


def build_graph_deps_for_model(
    services: AppServices,
    model_name: str,
) -> GraphDeps:
    """Build GraphDeps for the selected model while reusing shared services."""
    bundle = build_agent_bundle(
        model_name=model_name,
        ollama_base_url=services.settings.OLLAMA_BASE_URL,
    )

    return GraphDeps(
        opgg_client=services.base_deps.opgg_client,
        prompt_cache=services.base_deps.prompt_cache,
        fallback_agent=bundle.fallback_agent,
        parser_agent=bundle.parser_agent,
        synthesis_agent=bundle.synthesis_agent,
        reflection_agent=bundle.reflection_agent,
        revision_agent=bundle.revision_agent,
        controller=None,
        rag_service=services.base_deps.rag_service,
        llm_service=None,
    )


def build_services() -> AppServices:
    """Initialize long-lived app services."""
    settings = GradioAppSettings()

    default_bundle = build_agent_bundle(
        model_name=settings.DEFAULT_MODEL,
        ollama_base_url=settings.OLLAMA_BASE_URL,
    )

    base_deps = GraphDeps(
        opgg_client=OPGGMCPClient(config=OPGGMCPConfig()),
        prompt_cache=InMemoryDictCache(),
        fallback_agent=default_bundle.fallback_agent,
        parser_agent=default_bundle.parser_agent,
        synthesis_agent=default_bundle.synthesis_agent,
        reflection_agent=default_bundle.reflection_agent,
        revision_agent=default_bundle.revision_agent,
        controller=None,
        rag_service=build_rag_service(),
        llm_service=None,
    )

    return AppServices(
        settings=settings,
        graph=build_league_assistant_graph(),
        base_deps=base_deps,
    )


def pretty_format_metadata(obj: object) -> str:
    """format metadata objects for gradio"""
    if obj is None or not isinstance(obj, BaseModel):
        return "None"

    # Pydantic model
    # if isinstance(obj, BaseModel):
    data = obj.model_dump()

    # # Dataclass
    # elif hasattr(obj, "__dict__"):
    #     data = vars(obj)

    # else:
    #     return pformat(obj, indent=2, width=100)

    lines: list[str] = []

    for key, value in data.items():
        lines.append(f"\n{key}:")

        if isinstance(value, (dict, list)):
            formatted = pformat(value, indent=2, width=100)
            lines.append(f"\n{formatted}\n")
        else:
            lines.append(f"{value}")

    return "\n".join(lines)


def format_debug_metadata(
    final_answer: FinalAnswer,
    model_name: str,
) -> str:
    """Format graph metadata for the Gradio debug accordion."""
    lines = [
        "## Debug Metadata",
        f"- **Model:** `{model_name}`",
        f"- **Used cache:** `{getattr(final_answer, 'used_cache', None)}`",
        f"- **Intent:** `{getattr(final_answer, 'intent', None)}`",
    ]

    raw_context_blocks = getattr(final_answer, "raw_context_blocks", None)
    synthesis_meta = getattr(final_answer, "synthesis_node_metadata", None)
    reflection_meta = getattr(final_answer, "reflection_node_metadata", None)
    revision_meta = getattr(final_answer, "revision_answer_node_metadata", None)

    if raw_context_blocks is not None:
        lines.extend(
            [
                "",
                "#### Context Blocks",
                f"```text\n{raw_context_blocks}\n```",
            ]
        )

    if synthesis_meta is not None:
        lines.extend(
            [
                "",
                "#### Synthesis",
                pretty_format_metadata(synthesis_meta),
            ]
        )

    if reflection_meta is not None:
        lines.extend(
            [
                "",
                "#### Reflection",
                pretty_format_metadata(reflection_meta),
            ]
        )

    if revision_meta is not None:
        lines.extend(
            [
                "",
                "#### Revision",
                pretty_format_metadata(revision_meta),
            ]
        )

    return "\n".join(lines)


async def run_graph_for_message_with_debug(
    message: str,
    history: list[Any],
    services: AppServices,
    model_name: str,
) -> tuple[str, str]:
    """Run the assistant graph and return answer plus debug markdown."""
    state = AssistantState(
        original_query=message,
        chat_history=history_to_messages(history),
        allowed_tool_names=[
            "lol_get_summoner_profile",
            "lol_list_summoner_matches",
            "lol_get_champion_analysis",
            "lol_get_lane_matchup_guide",
            "lol_list_lane_meta_champions",
            "lol_list_champion_details",
        ],
    )

    deps = build_graph_deps_for_model(
        services=services,
        model_name=model_name,
    )

    try:
        result = await services.graph.run(
            start_node=BuildInitialAssistantStateNode(
                user_input=UserQuery(query=message),
            ),
            state=state,
            deps=deps,
        )

        final_answer = result.output

        answer = final_answer.answer or (
            "I generated an empty response. Try asking with a clearer role, "
            "playstyle, champion, or skin preference."
        )

        debug_text = format_debug_metadata(
            final_answer=final_answer,
            model_name=model_name,
        )

        return answer, debug_text

    except Exception as e:
        logger.exception("Gradio graph request failed for message=%r", message)

        return (
            "I ran into an issue while processing that request. "
            "Try rephrasing it or asking for a champion recommendation, "
            "skin search, or playstyle suggestion.",
            f"### Error\n\n`{type(e).__name__}: {e}`",
        )


services = build_services()


def build_demo() -> gr.Blocks:
    """Build the Gradio UI."""
    settings = services.settings

    model_choices = [
        # settings.SMALL_MODEL,
        settings.LARGE_MODEL,
    ]

    with gr.Blocks(
        title="League Multi-Tool LLM Agent",
        css="""
        #debug-accordion button {
            color: #00e5ff !important;
            font-weight: bold !important;
            font-size: 18px !important;
        }

        #debug-accordion:hover {
            box-shadow: 0 0 10px rgba(255, 152, 0, 0.5);
        }
        """,
    ) as demo:
        gr.Markdown(
            """
            # League Multi-Tool LLM Agent

            Ask for champion recommendations, skin recommendations, builds,
            counters, or playstyle advice.
            """
        )

        # with gr.Row():
        # model_choice = gr.Dropdown(
        #     choices=model_choices,
        #     value=settings.DEFAULT_MODEL,
        #     label="Model",
        #     info="Toggle between smaller and larger local LLMs.",
        # )

        chatbot = gr.Chatbot(
            label="League Assistant",
            # type="messages",
            height=500,
        )

        message_box = gr.Textbox(
            label="Your message",
            placeholder="Ask for a champion or skin recommendation...",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(
            examples=settings.EXAMPLES,
            inputs=message_box,
        )

        with gr.Accordion("Debug Metadata", open=False, elem_id="debug-accordion"):
            debug_output = gr.Markdown("No debug info yet.")

        async def submit_message(
            message: str,
            history: list[dict[str, str]],
            # history: list[list[str]],
            # model_name: str,
        ) -> tuple[str, list[dict[str, str]], str]:
            # ) -> tuple[str, list[list[str]], str]:
            """Handle one chat submission and update debug panel."""
            if not message.strip():
                return "", history or [], "No message submitted."

            history = history or []
            # history.append({"role": "user", "content": message})
            history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            answer, debug_text = await run_graph_for_message_with_debug(
                message=message,
                history=history,
                services=services,
                # model_name=model_name,
                model_name=settings.DEFAULT_MODEL,
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # history.append({"role": "assistant", "content": answer})
            # history.append([message, answer])
            return "", history, debug_text

        # model_choice = settings.DEFAULT_MODEL

        submit_btn.click(
            fn=submit_message,
            inputs=[
                message_box,
                chatbot,
                # model_choice,
            ],
            outputs=[
                message_box,
                chatbot,
                debug_output,
            ],
        )

        message_box.submit(
            fn=submit_message,
            inputs=[
                message_box,
                chatbot,
                # model_choice,
            ],
            outputs=[
                message_box,
                chatbot,
                debug_output,
            ],
        )

        clear_btn.click(
            fn=lambda: ([], "No debug info yet."),
            inputs=None,
            outputs=[chatbot, debug_output],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    demo.launch(
        share=services.settings.GRADIO_SHARE,
        debug=services.settings.GRADIO_DEBUG,
        server_name=services.settings.GRADIO_SERVER_NAME,
        server_port=services.settings.GRADIO_SERVER_PORT,
    )
# OLLAMA_BASE_URL=http://localhost:11434/v1/ uv run src/league_multi_tool_llm_agent/ui/gradio_app.py
