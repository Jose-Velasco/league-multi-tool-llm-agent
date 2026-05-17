from dataclasses import asdict, dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    IntentType,
)


# TODO: implement
async def llm_synthesize(state: AssistantState) -> str:
    joined = "\n\n".join(state.merged_context_blocks)
    return f"Answer based on context:\n\n{joined}"


# TODO: implement
async def llm_reflect(answer: str) -> str:
    return answer


@dataclass
class StorePromptCacheNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> End[FinalAnswer]:
        final = ctx.state.final_answer or ""
        # await cache_store(ctx.deps.prompt_cache, ctx.state.cache_key or "", final)
        if ctx.deps.prompt_cache:
            ctx.deps.prompt_cache.insert(ctx.state.cache_key or "", final)

        return End(
            FinalAnswer(
                answer=final,
                used_cache=False,
                intent=ctx.state.parsed_intent.intent
                if ctx.state.parsed_intent
                else IntentType.ERROR,
                raw_context_blocks=ctx.state.merged_context_blocks,
                synthesis_node_metadata=ctx.state.synthesis_node_metadata,
                reflection_node_metadata=ctx.state.reflection_node_metadata,
                revision_answer_node_metadata=ctx.state.revision_answer_node_metadata,
            )
        )


@dataclass
class RevisionNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> StorePromptCacheNode:
        print("### Starting RevisionNode ###")

        # ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        # ctx.state.final_answer = await llm_reflect(ctx.state.draft_answer or "")
        prompt = f"""original_query:
        {ctx.state.original_query}
        
        ReflectionResult metadata:
        {ctx.state.reflection_node_metadata.model_dump_json() if ctx.state.reflection_node_metadata else "ReflectionResult metadata not provided."}

        pipeline context:
        {asdict(ctx.state)}
        """
        result = await ctx.deps.revision_agent.run(prompt)

        ctx.state.revision_answer_node_metadata = result.output
        ctx.state.final_answer = result.output.revised_answer

        return StorePromptCacheNode()


@dataclass
class ReflectionNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> StorePromptCacheNode | RevisionNode:
        print("### Starting ReflectionNode ###")

        # ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        # ctx.state.final_answer = await llm_reflect(ctx.state.draft_answer or "")
        result = await ctx.deps.reflection_agent.run(ctx.state.draft_answer or "")
        ctx.state.reflection_node_metadata = result.output

        if result.output.needs_revision:
            return RevisionNode()
        else:
            ctx.state.final_answer = ctx.state.draft_answer

        return StorePromptCacheNode()


@dataclass
class SynthesisNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> ReflectionNode:
        print("### Starting SynthesisNode ###")
        # ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        # ctx.state.draft_answer = await llm_synthesize(ctx.state)
        joined = "\n\n".join(ctx.state.merged_context_blocks)
        prompt = f""" original_query:
        {ctx.state.original_query}

        parsed_intent context:
        {ctx.state.parsed_intent.model_dump_json() if ctx.state.parsed_intent else "No parsed_intent context provided"}

        merged_context_blocks:
        {joined}
        """
        # prompt = f""" pipeline context:
        # {asdict(ctx.state)}

        # merged_context_blocks:
        # {joined}
        # """
        result = await ctx.deps.synthesis_agent.run(prompt)
        ctx.state.draft_answer = result.output.answer
        ctx.state.synthesis_node_metadata = result.output

        return ReflectionNode()


@dataclass
class AggregationNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> SynthesisNode:
        # ) -> BaseNode[AssistantState, GraphDeps, FinalAnswer]:
        blocks: list[str] = []

        if ctx.state.profile_text:
            blocks.append(f"[PROFILE]\n{ctx.state.profile_text}")
        if ctx.state.match_history_text:
            blocks.append(f"[MATCH_HISTORY]\n{ctx.state.match_history_text}")
        if ctx.state.champion_meta_text:
            blocks.append(f"[CHAMPION_META]\n{ctx.state.champion_meta_text}")
        if ctx.state.recommendation_text:
            blocks.append(f"[RECOMMENDATION_INPUTS]\n{ctx.state.recommendation_text}")
        if ctx.state.matchup_text:
            blocks.append(f"[MATCHUP]\n{ctx.state.matchup_text}")
        if ctx.state.rag_text:
            blocks.append(f"[RAG]\n{ctx.state.rag_text}")

        ctx.state.merged_context_blocks = blocks
        return SynthesisNode()
