from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_graph import BaseNode, End, GraphRunContext

from league_multi_tool_llm_agent.graph.catch_all_node import fallback_mcp_agent
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    IntentType,
)


class FriendlyErrorResponse(BaseModel):
    message: str = Field(description="Short helpful message for the user.")


def build_error_message_agent(
    model_name: str = "gemma4:e4b-it-q4_K_M",
    ollama_provider_config: OllamaProviderConfig | None = None,
) -> Agent[None, FriendlyErrorResponse]:
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
    # model = OllamaModel(model_name=model_name)

    return Agent(
        model=model,
        output_type=FriendlyErrorResponse,
        retries=2,
        instructions=(
            "You convert internal tool errors into short, helpful, non-technical user messages. "
            "Do not expose stack traces. "
            "If the error suggests missing identity info, ask for the user's Riot ID in GameName#TAG format. "
            "If the error suggests a missing summoner, ask them to verify spelling and tag."
        ),
    )


async def build_user_friendly_error_message(
    *,
    llm: Agent[None, FriendlyErrorResponse],
    user_query: str,
    failed_tool_name: str,
    raw_error: str,
) -> str:
    prompt = f"""
    User query:
    {user_query}

    Failed tool:
    {failed_tool_name}

    Raw internal error:
    {raw_error}

    Write a short, friendly response to the user that helps them fix the issue.
    """.strip()

    result = await llm.run(prompt)
    return result.output.message


# async def cache_lookup(cache: PromptCache | None, key: str) -> str | None:
#     return cache.get(key) if cache else None


@dataclass
class ErrorRecoveryNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> End[FinalAnswer]:
        print("In ErrorRecoveryNode")
        error_text = (ctx.state.last_error or "").lower()
        tool_name = ctx.state.failed_tool_name or ""

        print(f"{error_text = }")
        print(f"{tool_name = }")

        if ctx.state.parsed_query is None:
            return End(
                FinalAnswer(
                    answer=("No query provided."),
                    intent=IntentType(IntentType.ERROR),
                )
            )

        # # 1. Riot ID format issue
        # if "expected riot id in 'gamename#tag' format" in error_text:
        #     return End(
        #         FinalAnswer(
        #             answer=(
        #                 "I need your Riot ID in the format `GameName#TAG` to look up your profile. "
        #                 "For example: `Pobelter#NA1`."
        #             ),
        #             intent=IntentType(IntentType.ERROR),
        #         )
        #     )

        # # 2. Summoner not found
        # if (
        #     "summoner not found" in error_text
        #     and tool_name == "lol_get_summoner_profile"
        # ):
        #     return End(
        #         FinalAnswer(
        #             answer=(
        #                 "I couldn't find that summoner profile. "
        #                 "Please send your full Riot ID in the format `GameName#TAG` "
        #                 "and your region if needed."
        #             ),
        #             intent=IntentType(IntentType.ERROR),
        #         )
        #     )

        # # 3. Generic tool-specific clarification
        # if tool_name == "lol_get_summoner_profile":
        #     return End(
        #         FinalAnswer(
        #             answer=(
        #                 "I ran into a problem looking up that League profile. "
        #                 "Please provide your full Riot ID like `GameName#TAG`."
        #             ),
        #             intent=IntentType(IntentType.ERROR),
        #         )
        #     )

        try:
            ctx.state.profile_text = await fallback_mcp_agent(
                user_query=ctx.state.parsed_query.query,
                chat_history=ctx.state.chat_history,
                tool_registry=ctx.deps.opgg_client.tool_registry,
                mcp_client=ctx.deps.opgg_client,
                fallback_agent=ctx.deps.fallback_agent,
                allowed_tool_names=ctx.state.allowed_tool_names,
            )
            ctx.state.used_fallback_tool_selection = True
            return End(
                FinalAnswer(
                    answer=ctx.state.profile_text, intent=IntentType(IntentType.ERROR)
                )
            )
        except Exception as e:
            ctx.state.last_error = (
                ctx.state.last_error + str(e) if ctx.state.last_error else ""
            )
            # 4. Unknown error -> let an LLM turn it into a friendly reply
            try:
                friendly = await build_user_friendly_error_message(
                    llm=ctx.deps.fallback_agent,
                    user_query=ctx.state.original_query,
                    failed_tool_name=tool_name,
                    raw_error=ctx.state.last_error,
                )
                return End(
                    FinalAnswer(answer=friendly, intent=IntentType(IntentType.ERROR))
                )
            except Exception as e:
                ctx.state.last_error = (
                    ctx.state.last_error + str(e) if ctx.state.last_error else ""
                )
                return End(
                    FinalAnswer(
                        answer=(
                            "I ran into an issue while processing that request. "
                            "Please try again with a bit more detail."
                        ),
                        intent=IntentType(IntentType.ERROR),
                    )
                )
