from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from league_multi_tool_llm_agent.graph.catch_all_node import fallback_mcp_agent
from league_multi_tool_llm_agent.graph.prompting_techniques import (
    AggregationNode,
    StorePromptCacheNode,
)
from league_multi_tool_llm_agent.graph.system_nodes import (
    ErrorRecoveryNode,
)
from league_multi_tool_llm_agent.graph.utils import (
    is_prompt_injection_or_out_of_scope,
    normalize_cache_key,
    parse_intent_with_fallback,
)
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    IntentType,
    ParsedIntent,
    UserQuery,
)


@dataclass
class BuildInitialAssistantStateNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    user_input: UserQuery

    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> PromptCacheCheckNode:
        ctx.state.original_query = self.user_input.query
        ctx.state.parsed_query = self.user_input
        ctx.state.cache_key = self.user_input.query.strip().lower()
        return PromptCacheCheckNode()


@dataclass
class PromptCacheCheckNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> ParseAndRouteNode | ReturnCachedResponseNode:

        if ctx.deps.prompt_cache:
            cache_key = normalize_cache_key(ctx.state.cache_key or "")
            cache_response = ctx.deps.prompt_cache.get(cache_key)
            if cache_response is not None:
                ctx.state.cache_hit = True
                ctx.state.cached_response = cache_response
                ctx.state.parsed_intent = ParsedIntent(
                    intent=IntentType("cached_response"),
                    query_for_rag="cached_response",
                )
                return ReturnCachedResponseNode()

        return ParseAndRouteNode()


@dataclass
class ParseAndRouteNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> SkinSearchNode | RecommendationNode | OPGG_MPC_Node | StorePromptCacheNode:
        assert ctx.state.parsed_query is not None

        parsed_intent = await parse_intent_with_fallback(
            user_prompt=ctx.state.parsed_query.query,
            parser_agent=ctx.deps.parser_agent,
        )

        if is_prompt_injection_or_out_of_scope(ctx.state.original_query):
            ctx.state.parsed_intent = ParsedIntent(
                intent=IntentType.ERROR,
                query_for_rag=ctx.state.original_query,
            )
            ctx.state.final_answer = (
                "I can only help with League of Legends questions, such as champion "
                "recommendations, skin searches, matchups, builds, or player/profile analysis."
            )
            return StorePromptCacheNode()

        ctx.state.parsed_intent = parsed_intent

        if parsed_intent.intent == IntentType.SKIN_SEARCH:
            return SkinSearchNode()

        if parsed_intent.intent == IntentType.CHAMPION_RECOMMENDATION:
            return RecommendationNode()

        # if parsed_intent.intent == IntentType.CHAMPION_RECOMMENDATION:
        #     return OPGG_MPC_Node()

        # print("### ParseAndRouteNode falling back to RecommendationNode ###")

        return RecommendationNode()


@dataclass
class ReturnCachedResponseNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> End[FinalAnswer]:
        return End(
            FinalAnswer(
                answer=ctx.state.cached_response or "",
                used_cache=True,
                intent=ctx.state.parsed_intent.intent
                if ctx.state.parsed_intent
                else IntentType.CACHED_RESPONSE,
                raw_context_blocks=[],
            )
        )


@dataclass
class RecommendationNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> AggregationNode | ErrorRecoveryNode:
        # print("### Starting RecommendationNode ###")

        if ctx.deps.rag_service is None or ctx.state.parsed_intent is None:
            return ErrorRecoveryNode()

        query = ctx.state.parsed_intent.query_for_rag

        retrieved_docs = await ctx.deps.rag_service.search(
            query=query,
            doc_type="champion_profile",
            limit=6,
        )
        ctx.state.rag_docs = retrieved_docs

        ctx.state.rag_text = "\n\n".join(doc.content for doc in retrieved_docs)

        return AggregationNode()


@dataclass
class SkinSearchNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> AggregationNode | ErrorRecoveryNode:
        # print("### Starting SkinSearchNode ###")

        if ctx.deps.rag_service is None or ctx.state.parsed_intent is None:
            return ErrorRecoveryNode()

        query = ctx.state.parsed_intent.query_for_rag

        # RAG search on skin documents
        retrieved_docs = await ctx.deps.rag_service.search(
            query=query,
            doc_type="champion_skin",
            limit=6,
        )

        # save for downstream synthesis
        ctx.state.rag_docs = retrieved_docs

        # also flatten text for LLM
        ctx.state.rag_text = "\n\n".join(doc.content for doc in retrieved_docs)

        return AggregationNode()


@dataclass
class OPGG_MPC_Node(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> AggregationNode | RecommendationNode:
        # print("### Starting OPGG_MPC_Node ###")

        try:
            opgg_mpc_text = await fallback_mcp_agent(
                user_query=ctx.state.original_query,
                chat_history=ctx.state.chat_history,
                tool_registry=ctx.deps.opgg_client.tool_registry,
                mcp_client=ctx.deps.opgg_client,
                fallback_agent=ctx.deps.fallback_agent,
                allowed_tool_names=ctx.state.allowed_tool_names,
            )
            ctx.state.used_fallback_tool_selection = True
            ctx.state.opgg_mpc_text = opgg_mpc_text
        except Exception:
            # print(e)
            return RecommendationNode()

        return AggregationNode()
