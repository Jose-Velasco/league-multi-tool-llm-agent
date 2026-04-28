import asyncio

from league_multi_tool_llm_agent.evaluation.run_evaluation import generate_no_rag_answer
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.protocols.agent import (
    LiteLLMRecommendationClient,
)


async def test():
    model_name = "qwen3.5:2b-q4_K_M"
    local_llm_config = OllamaProviderConfig(OLLAMA_BASE_URL="http://ollama:11434/v1")

    client = LiteLLMRecommendationClient(
        model_name=model_name, ollama_provider_config=local_llm_config
    )

    result = await generate_no_rag_answer(
        query="Champions that fit a strategic playstyle", client=client
    )
    # r = await acompletion(
    #     model="ollama_chat/qwen3.5:2b-q4_K_M",
    #     api_base="http://ollama:11434",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Recommend one League champion in one sentence.",
    #         }
    #     ],
    # )
    # print(r["choices"][0]["message"]["content"])
    print(result)


async def main() -> None:
    await test()


if __name__ == "__main__":
    asyncio.run(main())


# from __future__ import annotations

# import asyncio

# from sqlalchemy import URL

# from league_multi_tool_llm_agent.db.llm_utils import EmbeddingClient
# from league_multi_tool_llm_agent.evaluation.configs import EvalSettings
# from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
# from league_multi_tool_llm_agent.models.rag_configs import EmbeddingSettings


# # Example main
# async def main() -> None:
#     from league_multi_tool_llm_agent.db.rag_service import RagService, RagSettings

#     print("before using embed client")
#     embedder = EmbeddingClient(
#         EmbeddingSettings(EMBEDDING_API_BASE="http://localhost:11434")
#     )

#     embedding = await embedder.embed("Test query!!! using clinet")

#     print("embed client cmpleted!!!!!!!!!!!!!!!###")

#     settings = EvalSettings()
#     local_llm_config = OllamaProviderConfig()
#     rag_settings = RagSettings(
#         EMBEDDING_API_BASE="http://localhost:11434", db_host="localhost"
#     )

#     db_url = URL.create(
#         drivername="postgresql+psycopg2",
#         username=rag_settings.db_user,
#         password=rag_settings.db_password,
#         host=rag_settings.db_host,
#         port=rag_settings.db_port,
#         database=rag_settings.db_name,
#     )
#     rag_service = RagService(db_url=db_url, settings=rag_settings)

#     test_rag_res = await rag_service.search(
#         query="Test query!!!",
#         doc_type="champion_profile",
#         limit=5,
#     )
#     print(test_rag_res)


# if __name__ == "__main__":
#     asyncio.run(main())
