from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from league_multi_tool_llm_agent.graph.prompting_techniques import AggregationNode
from league_multi_tool_llm_agent.graph.system_nodes import (
    ErrorRecoveryNode,
)
from league_multi_tool_llm_agent.graph.utils import parse_intent_with_fallback
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
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        ctx.state.original_query = self.user_input.query
        ctx.state.parsed_query = self.user_input
        ctx.state.cache_key = self.user_input.query.strip().lower()
        return PromptCacheCheckNode()


@dataclass
class PromptCacheCheckNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        # cached = await cache_lookup(ctx.deps.prompt_cache, ctx.state.cache_key or "")
        cached = (
            ctx.deps.prompt_cache.get(ctx.state.cache_key or "")
            if ctx.deps.prompt_cache
            else None
        )

        if cached is not None:
            ctx.state.cache_hit = True
            ctx.state.cached_response = cached
            ctx.state.parsed_intent = ParsedIntent(
                intent=IntentType("cached_response"), query_for_rag="cached_response"
            )
            return ReturnCachedResponseNode()
        return ParseAndRouteNode()


@dataclass
class ParseAndRouteNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        assert ctx.state.parsed_query is not None

        parsed_intent = await parse_intent_with_fallback(
            user_prompt=ctx.state.parsed_query.query,
            parser_agent=ctx.deps.parser_agent,
        )

        ctx.state.parsed_intent = parsed_intent

        if ctx.state.parsed_intent == IntentType.PROFILE_ANALYSIS:
            return ProfileAnalysisNode()

        if ctx.state.parsed_intent == IntentType.MATCH_HISTORY_ANALYSIS:
            return MatchHistoryAnalysisNode()

        if ctx.state.parsed_intent == IntentType.CHAMPION_META:
            return ChampionMetaNode()

        if ctx.state.parsed_intent == IntentType.MATCHUP_GUIDE:
            return MatchGuideNode()

        if ctx.state.parsed_intent == IntentType.SKIN_SEARCH:
            return SkinSearchNode()

        if ctx.state.parsed_intent == IntentType.CHAMPION_RECOMMENDATION:
            return RecommendationNode()

        return RecommendationNode()


# @dataclass
# class ParseAndRouteNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
#     async def run(
#         self, ctx: GraphRunContext[AssistantState, GraphDeps]
#     ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
#         assert ctx.state.parsed_query is not None
#         ctx.state.intent = await route_intent(ctx.state.parsed_query.query)

#         if ctx.state.intent == "profile_analysis":
#             return ProfileAnalysisNode()
#         if ctx.state.intent == "match_history_analysis":
#             return MatchHistoryAnalysisNode()
#         if ctx.state.intent == "champion_meta":
#             return ChampionMetaNode()
#         if ctx.state.intent == "matchup_guide":
#             return MatchGuideNode()
#         return RecommendationNode()


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
class ProfileAnalysisNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        q = ctx.state.parsed_query
        assert q is not None

        # if q.riot_id is None:
        #     return End(
        #         FinalAnswer(
        #             answer="Please provide your Riot ID (for example, Pobelter#NA1) and region so I can analyze your profile."
        #         )
        #     )

        try:
            result = await ctx.deps.opgg_client.get_summoner_profile(
                riot_id=q.riot_id,
                # region=q.region,
            )
            ctx.state.profile_text = ctx.deps.opgg_client.extract_text(result)

            return AggregationNode()
        except Exception as e:
            # ctx.state.last_error = str(e)

            # ctx.state.profile_text = await fallback_mcp_agent(
            #     user_query=q.query,
            #     chat_history=ctx.state.chat_history,
            #     tool_registry=ctx.deps.opgg_client.tool_registry,
            #     mcp_client=ctx.deps.opgg_client,
            #     fallback_agent=ctx.deps.fallback_agent,
            # allowed_tool_names=[
            #     "lol_get_summoner_profile",
            #     "lol_list_summoner_matches",
            #     "lol_get_pro_player_riot_id",
            # ],
            # )
            # ctx.state.used_fallback_tool_selection = True
            ctx.state.allowed_tool_names = [
                "lol_get_summoner_profile",
                "lol_list_summoner_matches",
                "lol_get_pro_player_riot_id",
            ]
            ctx.state.last_error = str(e)
            ctx.state.failed_tool_name = "lol_get_summoner_profile"
            return ErrorRecoveryNode()


@dataclass
class MatchHistoryAnalysisNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        q = ctx.state.parsed_query
        assert q is not None

        if q.riot_id is None:
            return End(
                FinalAnswer(
                    answer="Please provide your Riot ID (for example, Pobelter#NA1) and region so I can analyze your profile."
                )
            )

        result = await ctx.deps.opgg_client.list_summoner_matches(
            riot_id=q.riot_id,
            region=q.region if q.region else "na",
            limit=10,
        )
        ctx.state.match_history_text = ctx.deps.opgg_client.extract_text(result)

        return AggregationNode()


@dataclass
class ChampionMetaNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        q = ctx.state.parsed_query
        assert q is not None

        # if not (q.champion and q.position):
        if q.champion is None:
            return End(
                FinalAnswer(
                    # answer="Please provide a Champion (for example, Ahri) and region so I can analyze your profile."
                    answer="Please provide a Champion (for example, Ahri) so I can analyze your profile."
                )
            )

        result = await ctx.deps.opgg_client.get_champion_analysis(
            champion=q.champion,
            position=q.position.value if q.position else "none",
        )
        ctx.state.champion_meta_text = ctx.deps.opgg_client.extract_text(result)

        return AggregationNode()


@dataclass
class RecommendationNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(self, ctx: GraphRunContext[AssistantState, GraphDeps]):
        if ctx.deps.rag_service is None or ctx.state.parsed_intent is None:
            return ErrorRecoveryNode()

        query = ctx.state.parsed_intent.query_for_rag

        retrieved_docs = await ctx.deps.rag_service.search(
            query=query,
            doc_type="champion_profile",
            limit=5,
        )
        ctx.state.rag_docs = retrieved_docs

        ctx.state.rag_text = "\n\n".join(doc.content for doc in retrieved_docs)

        return AggregationNode()


@dataclass
class SkinSearchNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        if ctx.deps.rag_service is None or ctx.state.parsed_intent is None:
            return ErrorRecoveryNode()

        query = ctx.state.parsed_intent.query_for_rag

        # RAG search on skin documents
        retrieved_docs = await ctx.deps.rag_service.search(
            query=query,
            doc_type="champion_skin",
            limit=5,
        )

        # Save for downstream synthesis
        ctx.state.rag_docs = retrieved_docs

        # Optional: also flatten text for LLM
        ctx.state.rag_text = "\n\n".join(doc.content for doc in retrieved_docs)

        return AggregationNode()


# @dataclass
# class RecommendationNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
#     async def run(
#         self, ctx: GraphRunContext[AssistantState, GraphDeps]
#     ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
#         q = ctx.state.parsed_query
#         assert q is not None

#         blocks: list[str] = []

#         if q.riot_id:
#             profile = await ctx.deps.opgg_client.get_summoner_profile(
#                 riot_id=q.riot_id,
#                 region=q.region if q.region else "na",
#             )
#             blocks.append(ctx.deps.opgg_client.extract_text(profile))

#             matches = await ctx.deps.opgg_client.list_summoner_matches(
#                 riot_id=q.riot_id,
#                 region=q.region if q.region else "na",
#                 limit=10,
#             )
#             blocks.append(ctx.deps.opgg_client.extract_text(matches))

#         if q.champion and q.position:
#             analysis = await ctx.deps.opgg_client.get_champion_analysis(
#                 champion=q.champion,
#                 position=q.position.value,
#             )
#             blocks.append(ctx.deps.opgg_client.extract_text(analysis))

#             synergies = await ctx.deps.opgg_client.get_champion_synergies(
#                 champion=q.champion,
#                 my_position=q.position.value,
#                 synergy_position="all",
#             )
#             blocks.append(ctx.deps.opgg_client.extract_text(synergies))

#         # Placeholder for pgvector / RAG
#         if ctx.deps.rag_service is not None:
#             ctx.state.rag_text = "RAG results placeholder"
#             blocks.append(ctx.state.rag_text)

#         ctx.state.recommendation_text = "\n\n".join(blocks)
#         return AggregationNode()


@dataclass
class MatchGuideNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer] | End[FinalAnswer]:
        q = ctx.state.parsed_query
        assert q is not None

        if not (q.champion and q.opponent_champion and q.position):
            return End(
                FinalAnswer(
                    # answer="Please provide a Champion (for example, Ahri) and region so I can analyze your profile."
                    answer="Please provide a champion (for example, Ahri), opponent champion, and position so I can analyze your profile."
                )
            )

        result = await ctx.deps.opgg_client.get_lane_matchup_guide(
            my_champion=q.champion,
            opponent_champion=q.opponent_champion,
            position=q.position.value,
        )
        ctx.state.matchup_text = ctx.deps.opgg_client.extract_text(result)

        return AggregationNode()
