from __future__ import annotations

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
from league_multi_tool_llm_agent.models.graph_models import (
    AssistantState,
    FinalAnswer,
    GraphDeps,
    UserQuery,
)
from league_multi_tool_llm_agent.models.rag_configs import RagSettings


async def main() -> None:
    league_assistant_graph = Graph(
        nodes=(
            BuildInitialAssistantStateNode,
            PromptCacheCheckNode,
            ReturnCachedResponseNode,
            ParseAndRouteNode,
            # ProfileAnalysisNode,
            # MatchHistoryAnalysisNode,
            # ChampionMetaNode,
            RecommendationNode,
            SkinSearchNode,
            # MatchGuideNode,
            AggregationNode,
            SynthesisNode,
            ReflectionNode,
            StorePromptCacheNode,
            ErrorRecoveryNode,
            RevisionNode,
        ),
        state_type=AssistantState,
        run_end_type=FinalAnswer,
    )

    local_llm_config = OllamaProviderConfig(OLLAMA_BASE_URL="http://ollama:11434/v1/")

    model = "qwen3.5:2b-q4_K_M"
    parser_agent = build_parser_agent(
        model,
        ollama_provider_config=local_llm_config,
    )
    fallback_agent = build_fallback_agent(
        model,
        ollama_provider_config=local_llm_config,
    )

    synthesis_agent = build_synthesis_agent(
        model,
        ollama_provider_config=local_llm_config,
    )

    reflection_agent = build_reflection_agent(
        model,
        ollama_provider_config=local_llm_config,
    )

    revision_agent = build_revision_agent(
        model,
        ollama_provider_config=local_llm_config,
    )

    rag_settings = RagSettings()
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=rag_settings.db_user,
        password=rag_settings.db_password,
        host=rag_settings.db_host,
        port=rag_settings.db_port,
        database=rag_settings.db_name,
    )
    rag_query_embedder = EmbeddingClient()
    rag_service = RagService(
        db_url=db_url, settings=rag_settings, embedder=rag_query_embedder
    )

    deps = GraphDeps(
        opgg_client=OPGGMCPClient(config=OPGGMCPConfig()),
        prompt_cache=InMemoryDictCache(),
        fallback_agent=fallback_agent,
        parser_agent=parser_agent,
        synthesis_agent=synthesis_agent,
        reflection_agent=reflection_agent,
        revision_agent=revision_agent,
        controller=None,
        rag_service=rag_service,
        llm_service=None,
    )

    # print(
    #     league_assistant_graph.mermaid_code(start_node=BuildInitialAssistantStateNode)
    # )

    # state0 = AssistantState()

    # result0 = await league_assistant_graph.run(
    #     start_node=BuildInitialAssistantStateNode(
    #         user_input=UserQuery(
    #             query="Give me Ahri mid build tips",
    #             champion="AHRI",
    #             position=UserQueryPosition("mid"),
    #         )
    #     ),
    #     state=state0,
    #     deps=deps,
    # )
    # print("\n### Result0: ###\n")
    # result_final_answer0 = result0.output
    # print(f"{result_final_answer0.answer =}")
    # print(f"{result_final_answer0.used_cache =}")
    # print(f"{result_final_answer0.intent =}")
    # print()
    # print(f"{result_final_answer0.raw_context_blocks =}")

    # state1 = AssistantState()

    # result1 = await league_assistant_graph.run(
    #     start_node=BuildInitialAssistantStateNode(
    #         user_input=UserQuery(
    #             query="Give me Ahri mid build tips",
    #             champion="AHRI",
    #             position=UserQueryPosition("mid"),
    #         )
    #     ),
    #     state=state1,
    #     deps=deps,
    # )
    # print("\n#### Result1: ####\n")
    # result_final_answer1 = result1.output
    # print(f"{result_final_answer1.answer =}")
    # print(f"{result_final_answer1.used_cache =}")
    # print(f"{result_final_answer1.intent =}")
    # print()
    # print(f"{result_final_answer1.raw_context_blocks =}")

    # state2 = AssistantState()

    # result2 = await league_assistant_graph.run(
    #     start_node=BuildInitialAssistantStateNode(
    #         user_input=UserQuery(
    #             query="Look up my ranked profile spypig",
    #             # riot_id="spypig",
    #             # champion="AHRI",
    #             # position=UserQueryPosition("mid"),
    #         )
    #     ),
    #     state=state2,
    #     deps=deps,
    # )
    # print("\n#### Result2: ####\n")
    # result_final_answer2 = result2.output
    # print(f"{result_final_answer2.answer =}")
    # print(f"{result_final_answer2.used_cache =}")
    # print(f"{result_final_answer2.intent =}")
    # print()
    # print(f"{result_final_answer2.raw_context_blocks =}")

    # state3 = AssistantState()

    # result3 = await league_assistant_graph.run(
    #     start_node=BuildInitialAssistantStateNode(
    #         user_input=UserQuery(
    #             query="Look up my ranked profile Spypig#2963",
    #             # riot_id="spypig",
    #             # champion="AHRI",
    #             # position=UserQueryPosition("mid"),
    #         )
    #     ),
    #     state=state3,
    #     deps=deps,
    # )
    # print("\n#### Result3: ####\n")
    # result_final_answer3 = result3.output
    # print(f"{result_final_answer3.answer =}")
    # print(f"{result_final_answer3.used_cache =}")
    # print(f"{result_final_answer3.intent =}")
    # print()
    # print(f"{result_final_answer3.raw_context_blocks =}")

    state4 = AssistantState()

    query4 = "Recommend champion: I like strong female leads and dark aesthetics"
    result4 = await league_assistant_graph.run(
        start_node=BuildInitialAssistantStateNode(
            user_input=UserQuery(
                query=query4,
            )
        ),
        state=state4,
        deps=deps,
    )
    print("\n#### Result4: ####\n")
    result_final_answer4 = result4.output
    print(f"{query4 = }")
    print(f"{result_final_answer4.answer =}")
    print(f"{result_final_answer4.used_cache =}")
    print(f"{result_final_answer4.intent =}")
    print()
    # print(f"{result_final_answer4.raw_context_blocks =}")
    print(f"{result_final_answer4.synthesis_node_metadata =}")
    print(f"{result_final_answer4.reflection_node_metadata =}")
    print(f"{result_final_answer4.revision_answer_node_metadata =}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
