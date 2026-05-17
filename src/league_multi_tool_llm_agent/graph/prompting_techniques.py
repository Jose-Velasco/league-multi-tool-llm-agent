import re
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from league_multi_tool_llm_agent.graph.prompt_cache import PromptCache
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    IntentType,
    ReflectionResult,
    RevisedAnswer,
)


def handle_prompt_cache(
    prompt_cache: PromptCache,
    final_response: str,
    cache_key: str,
    reflection_node_metadata: ReflectionResult | None,
    revision_answer_node_metadata: RevisedAnswer | None,
) -> None:
    cache_key_normalized = cache_key.lower()
    cache_key_normalized = re.sub(r"[^\w\s]", "", cache_key_normalized)
    if reflection_node_metadata and reflection_node_metadata.approved:
        prompt_cache.insert(cache_key_normalized, final_response)
    elif reflection_node_metadata and reflection_node_metadata.needs_revision:
        if (
            revision_answer_node_metadata
            and revision_answer_node_metadata.addressed_issues
        ):
            prompt_cache.insert(cache_key_normalized, final_response)


@dataclass
class StorePromptCacheNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> End[FinalAnswer]:
        final = ctx.state.final_answer or ""
        if ctx.deps.prompt_cache and ctx.state.cache_key:
            handle_prompt_cache(
                prompt_cache=ctx.deps.prompt_cache,
                final_response=final,
                cache_key=ctx.state.cache_key,
                reflection_node_metadata=ctx.state.reflection_node_metadata,
                revision_answer_node_metadata=ctx.state.revision_answer_node_metadata,
            )

            # ctx.deps.prompt_cache.insert(ctx.state.cache_key or "", final)

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
        # prompt = f"""Original User Query:
        # {ctx.state.original_query}

        # Original Draft Answer:
        # {ctx.state.draft_answer}

        # parsed_intent context:
        # {ctx.state.parsed_intent.model_dump_json() if ctx.state.parsed_intent else "No parsed_intent context provided"}

        # Reflection Feedback:
        # {ctx.state.reflection_node_metadata.model_dump_json() if ctx.state.reflection_node_metadata else "ReflectionResult metadata not provided."}

        # pipeline context:
        # {asdict(ctx.state)}
        # """
        prompt = f"""Original User Query:
        {ctx.state.original_query}

        Original Draft Answer:
        {ctx.state.draft_answer}

        parsed_intent context:
        {ctx.state.parsed_intent.model_dump_json() if ctx.state.parsed_intent else "No parsed_intent context provided"}
        
        Reflection Feedback:
        {ctx.state.reflection_node_metadata.model_dump_json() if ctx.state.reflection_node_metadata else "ReflectionResult metadata not provided."}

        Revise the answer to address the issues.
        """
        try:
            response_result = await ctx.deps.revision_agent.run(prompt)
            result = response_result.output
            # ctx.state.revision_answer_node_metadata = result.output
            # ctx.state.final_answer = result.output.revised_answer
        except Exception:
            print("Revision failed.")

            fall_back_answer = "I ran into an issue while revising my response for that request. Try rephrasing it or asking for a champion recommendation, skin search, or playstyle suggestion."
            if ctx.state.synthesis_node_metadata:
                fall_back_answer = ctx.state.synthesis_node_metadata.answer

            # fallback to original synthesis answer
            result = RevisedAnswer(
                revised_answer=fall_back_answer,
                changes_made=[
                    "Revision failed; returning original synthesized answer."
                ],
                addressed_issues=False,
                confidence=0.0,
            )

        ctx.state.revision_answer_node_metadata = result
        ctx.state.final_answer = result.revised_answer

        return StorePromptCacheNode()


@dataclass
class ReflectionNode(BaseNode[AssistantState, GraphDeps, FinalAnswer]):
    async def run(
        self, ctx: GraphRunContext[AssistantState, GraphDeps]
    ) -> StorePromptCacheNode | RevisionNode:
        print("### Starting ReflectionNode ###")

        joined = "\n\n".join(ctx.state.merged_context_blocks)
        prompt = f"""
        original_query:
        {ctx.state.original_query}

        merged_context_blocks:
        {joined}

        Draft Answer:
        {ctx.state.draft_answer}

        Evaluate the answer.
        """

        # ctx.state.final_answer = await llm_reflect(ctx.state.draft_answer or "")
        # result = await ctx.deps.reflection_agent.run(ctx.state.draft_answer or "")
        result = await ctx.deps.reflection_agent.run(prompt)
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
        if ctx.state.opgg_mpc_text:
            blocks.append(f"[OP.GG MCP Results]\n{ctx.state.opgg_mpc_text}")

        ctx.state.merged_context_blocks = blocks
        return SynthesisNode()
