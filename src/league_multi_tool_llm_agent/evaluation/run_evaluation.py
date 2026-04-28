from __future__ import annotations

import asyncio
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from sqlalchemy import URL
from tqdm.asyncio import tqdm_asyncio

from league_multi_tool_llm_agent.db.llm_utils import EmbeddingClient
from league_multi_tool_llm_agent.evaluation.agents import (
    EvalCaseResult,
    JudgeScore,
    build_judge_agent,
)
from league_multi_tool_llm_agent.evaluation.configs import EvalSettings
from league_multi_tool_llm_agent.evaluation.utils import (
    build_recommendation_client,
    checkpoint_results,
    save_eval_results,
)
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.rag_configs import EmbeddingSettings
from league_multi_tool_llm_agent.protocols.agent import RecommendationClient

LOG_DIR = Path("data/eval_results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "evaluation.log"

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "Good champions for climbing ranked solo queue.",
    "Beginner-friendly supports.",
    "Champions with strong late game scaling.",
    "I like strong female leads and dark aesthetics, recommend a champion.",
    "I prefer supportive roles and teamwork, who should I main?",
    "Aggressive junglers with high damage.",
    "Cute skins for support champions.",
    "Champions similar to Ahri.",
    "Skins with futuristic or sci-fi themes.",
    "Easy beginner champions for new players.",
    "Tanky champions that can engage fights.",
    "High mobility assassins.",
    "Dark themed skins with magical vibes.",
    "Fun off-meta champions.",
    "Champions that fit a strategic playstyle.",
]


async def generate_no_rag_answer(
    *,
    query: str,
    client: RecommendationClient,
) -> tuple[str, str]:
    """Generate a recommendation without retrieval context."""
    prompt = f"""User request:
    {query}

    Recommend 2-3 League of Legends champions or skins that fit the request.
    Explain why each recommendation fits.
    Keep the answer concise.
    Return plain text only.
    """

    answer = await client.generate(prompt)

    return answer, ""


# async def generate_no_rag_answer(
#     *,
#     query: str,
#     agent: Agent[None, str],
# ) -> tuple[str, str]:
#     """Generate a recommendation without retrieval context."""
#     prompt = f"""
#     User request:
#     {query}

#     Recommend 2-3 League of Legends champions or skins that fit the request.
#     Explain why each recommendation fits.
#     Return plain text only.
#     """

#     result = await agent.run(prompt)
#     return result.output, ""


async def generate_rag_answer(
    *,
    query: str,
    client: RecommendationClient,
    rag_service,
    top_k: int,
) -> tuple[str, str]:
    """Retrieve context and generate a grounded recommendation."""
    doc_type = "champion_skin" if "skin" in query.lower() else "champion_profile"

    retrieved_docs = await rag_service.search(
        query=query,
        doc_type=doc_type,
        limit=top_k,
    )

    retrieved_context = "\n\n".join(
        f"[{i + 1}] {doc.content}" for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"""User request:
    {query}

    Retrieved context:
    {retrieved_context}

    Recommend 2-3 League of Legends champions or skins that fit the request.
    Use the retrieved context when possible.
    Explain why each recommendation fits.
    Return plain text only.
    """

    answer = await client.generate(prompt)
    return answer, retrieved_context


# async def generate_rag_answer(
#     *,
#     query: str,
#     agent: Agent[None, str],
#     rag_service: Any,
#     top_k: int,
# ) -> tuple[str, str]:
#     """Retrieve context and generate a RAG-grounded recommendation."""
#     doc_type = "champion_skin" if "skin" in query.lower() else "champion_profile"

#     retrieved_docs = await rag_service.search(
#         query=query,
#         doc_type=doc_type,
#         limit=top_k,
#     )

#     retrieved_context = "\n\n".join(
#         f"[{i + 1}] {doc.content}" for i, doc in enumerate(retrieved_docs)
#     )

#     prompt = f"""
#     User request:
#     {query}

#     Retrieved context:
#     {retrieved_context}

#     Recommend 2-3 League of Legends champions or skins that fit the request.
#     Use the retrieved context when possible.
#     Explain why each recommendation fits.
#     Return plain text only.
#     """

#     result = await agent.run(prompt)
#     return result.output, retrieved_context


async def judge_answer(
    *,
    query: str,
    answer: str,
    retrieved_context: str,
    judge_agent: Agent[None, JudgeScore],
) -> JudgeScore:
    """Score one generated answer with an LLM judge."""
    prompt = f"""
    User request:
    {query}

    Retrieved context:
    {retrieved_context or "(no retrieved context)"}

    System response:
    {answer}

    Evaluate the system response.
    """

    result = await judge_agent.run(prompt)
    return result.output


async def run_single_eval_case(
    *,
    query: str,
    condition: str,
    model_name: str,
    use_rag: bool,
    rag_service: Any | None,
    # rec_agent: Agent[None, str],
    rec_agent: RecommendationClient,
    judge_agent: Agent[None, JudgeScore],
    settings: EvalSettings,
    semaphore: asyncio.Semaphore,
) -> EvalCaseResult:
    """Run one query for one ablation condition with bounded concurrency."""
    async with semaphore:
        start = time.perf_counter()

        try:
            if use_rag:
                if rag_service is None:
                    raise ValueError("rag_service is required when use_rag=True")

                answer, context = await generate_rag_answer(
                    query=query,
                    client=rec_agent,
                    rag_service=rag_service,
                    top_k=settings.EVAL_RAG_TOP_K,
                )
            else:
                answer, context = await generate_no_rag_answer(
                    query=query,
                    client=rec_agent,
                )
        except Exception as e:
            latency = time.perf_counter() - start
            error_type = type(e).__name__
            error_msg = str(e)
            tb = traceback.format_exc(limit=5)

            logger.exception(
                "Generation failed | condition=%s | model=%s | use_rag=%s | query=%r",
                condition,
                model_name,
                use_rag,
                query,
            )

            return EvalCaseResult(
                query=query,
                condition=condition,
                model_name=model_name,
                use_rag=use_rag,
                answer="ERROR",
                retrieved_context="",
                latency_seconds=latency,
                relevance=None,
                explanation_quality=None,
                personalization=None,
                groundedness=None,
                judge_notes=(
                    f"Generation failed: {error_type}: {error_msg}\nTraceback:\n{tb}"
                ),
                # judge_notes=f"Generation failed: {e}",
            )

        latency = time.perf_counter() - start
        logger.info(
            "recommend agent answer | condition=%s | model=%s | use_rag=%s | query=%r | answer=%r | context_preview=%r",
            condition,
            model_name,
            use_rag,
            query,
            answer[:1000] if answer else "",
            context[:1000] if context else "",
        )

        try:
            score = await judge_answer(
                query=query,
                answer=answer,
                retrieved_context=context,
                judge_agent=judge_agent,
            )
            judge_error = ""
        except Exception as e:
            score = None
            judge_error = str(e)
            error_type = type(e).__name__
            error_msg = str(e)
            tb = traceback.format_exc(limit=5)

            logger.exception(
                "Judge failed | condition=%s | model=%s | use_rag=%s | query=%r | answer=%r | context_preview=%r",
                condition,
                model_name,
                use_rag,
                query,
                answer[:1000] if answer else "",
                context[:1000] if context else "",
            )

            score = None
            judge_error = f"{error_type}: {error_msg}\nTraceback:\n{tb}"

        return EvalCaseResult(
            query=query,
            condition=condition,
            model_name=model_name,
            use_rag=use_rag,
            answer=answer,
            retrieved_context=context,
            latency_seconds=latency,
            relevance=score.relevance if score else None,
            explanation_quality=score.explanation_quality if score else None,
            personalization=score.personalization if score else None,
            groundedness=score.groundedness if score else None,
            judge_notes=score.notes if score else f"Judge failed: {judge_error}",
        )


async def run_eval_condition(
    *,
    queries: list[str],
    condition: str,
    model_name: str,
    use_rag: bool,
    rag_service: Any | None,
    judge_agent: Agent[None, JudgeScore],
    settings: EvalSettings,
    ollama_provider_config: OllamaProviderConfig,
) -> list[EvalCaseResult]:
    """Run one ablation condition across all test queries."""
    # rec_agent = build_recommendation_agent(
    #     model_name,
    #     ollama_provider_config=ollama_provider_config,
    # )

    rec_client = build_recommendation_client(
        model_name=model_name,
        ollama_provider_config=ollama_provider_config,
        settings=settings,
    )

    semaphore = asyncio.Semaphore(settings.EVAL_MAX_CONCURRENCY)

    tasks = [
        run_single_eval_case(
            query=query,
            condition=condition,
            model_name=model_name,
            use_rag=use_rag,
            rag_service=rag_service,
            rec_agent=rec_client,
            judge_agent=judge_agent,
            settings=settings,
            semaphore=semaphore,
        )
        for query in queries
    ]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=condition,
    )

    for r in results:
        print(
            f"[{r.condition}] {r.query[:50]}... "
            f"latency={r.latency_seconds:.2f}s "
            f"rel={r.relevance}"
        )

    return list(results)


async def run_full_evaluation(
    *,
    rag_service: Any,
    settings: EvalSettings | None = None,
    ollama_provider_config: OllamaProviderConfig,
) -> list[EvalCaseResult]:
    """Run RAG/no-RAG and small/large model ablations with checkpointing."""
    cfg = settings or EvalSettings()
    queries = TEST_QUERIES[: cfg.EVAL_MAX_TEST_QUERIES]

    judge_agent = build_judge_agent(
        cfg.EVAL_JUDGE_MODEL,
        ollama_provider_config=ollama_provider_config,
    )

    conditions = [
        {
            "condition": "small_no_rag",
            "model_name": cfg.EVAL_SMALL_MODEL,
            "use_rag": False,
        },
        {"condition": "small_rag", "model_name": cfg.EVAL_SMALL_MODEL, "use_rag": True},
        {"condition": "large_rag", "model_name": cfg.EVAL_LARGE_MODEL, "use_rag": True},
        {
            "condition": "large_no_rag",
            "model_name": cfg.EVAL_LARGE_MODEL,
            "use_rag": False,
        },
    ]

    all_results: list[EvalCaseResult] = []

    for c in conditions:
        try:
            condition_results = await run_eval_condition(
                queries=queries,
                condition=c["condition"],
                model_name=c["model_name"],
                use_rag=c["use_rag"],
                rag_service=rag_service,
                judge_agent=judge_agent,
                settings=cfg,
                ollama_provider_config=ollama_provider_config,
            )
            all_results.extend(condition_results)

            checkpoint_results(
                results=all_results,
                output_dir=cfg.EVAL_OUTPUT_DIR,
                label="latest_checkpoint",
            )

        except Exception as e:
            checkpoint_results(
                results=all_results,
                output_dir=cfg.EVAL_OUTPUT_DIR,
                label="crash_checkpoint",
            )
            raise RuntimeError(
                f"Evaluation crashed during condition={c['condition']}. "
                f"Saved {len(all_results)} completed results."
            ) from e

    return all_results


# async def run_full_evaluation(
#     *,
#     rag_service: Any,
#     settings: EvalSettings | None = None,
#     ollama_provider_config: OllamaProviderConfig,
# ) -> list[EvalCaseResult]:
#     """Run RAG/no-RAG and small/large model ablations."""
#     cfg = settings or EvalSettings()
#     queries = TEST_QUERIES[: cfg.EVAL_MAX_TEST_QUERIES]

#     judge_agent = build_judge_agent(
#         cfg.EVAL_JUDGE_MODEL,
#         ollama_provider_config=ollama_provider_config,
#     )

#     conditions = [
#         {
#             "condition": "small_no_rag",
#             "model_name": cfg.EVAL_SMALL_MODEL,
#             "use_rag": False,
#         },
#         {
#             "condition": "small_rag",
#             "model_name": cfg.EVAL_SMALL_MODEL,
#             "use_rag": True,
#         },
#         {
#             "condition": "large_no_rag",
#             "model_name": cfg.EVAL_LARGE_MODEL,
#             "use_rag": False,
#         },
#         {
#             "condition": "large_rag",
#             "model_name": cfg.EVAL_LARGE_MODEL,
#             "use_rag": True,
#         },
#     ]

#     all_results: list[EvalCaseResult] = []

#     for c in conditions:
#         condition_results = await run_eval_condition(
#             queries=queries,
#             condition=c["condition"],
#             model_name=c["model_name"],
#             use_rag=c["use_rag"],
#             rag_service=rag_service,
#             judge_agent=judge_agent,
#             settings=cfg,
#             ollama_provider_config=ollama_provider_config,
#         )
#         all_results.extend(condition_results)

#     return all_results


async def main() -> None:
    """Run evaluation in a properly configured Colab/local environment."""
    import os

    from league_multi_tool_llm_agent.db.rag_service import RagService, RagSettings

    os.environ["OLLAMA_NUM_PARALLEL"] = "2"
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"

    settings = EvalSettings()
    local_llm_config = OllamaProviderConfig()

    rag_settings = RagSettings(db_host="localhost")
    embed_client_config = EmbeddingSettings(EMBEDDING_API_BASE="http://localhost:11434")
    embed_client = EmbeddingClient(embed_client_config)

    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=rag_settings.db_user,
        password=rag_settings.db_password,
        host=rag_settings.db_host,
        port=rag_settings.db_port,
        database=rag_settings.db_name,
    )

    rag_service = RagService(
        db_url=db_url,
        settings=rag_settings,
        embedder=embed_client,
    )

    results = await run_full_evaluation(
        rag_service=rag_service,
        settings=settings,
        ollama_provider_config=local_llm_config,
    )

    save_eval_results(
        results=results,
        output_dir=settings.EVAL_OUTPUT_DIR,
    )


if __name__ == "__main__":
    asyncio.run(main())


# from __future__ import annotations

# import asyncio
# import time
# from typing import Any

# from pydantic_ai import Agent
# from sqlalchemy import URL
# from tqdm.asyncio import tqdm_asyncio
# from tqdm.auto import tqdm

# from league_multi_tool_llm_agent.db.llm_utils import EmbeddingClient
# from league_multi_tool_llm_agent.evaluation.agents import (
#     EvalCaseResult,
#     JudgeScore,
#     build_judge_agent,
#     build_recommendation_agent,
# )
# from league_multi_tool_llm_agent.evaluation.configs import EvalSettings
# from league_multi_tool_llm_agent.evaluation.utils import save_eval_results
# from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
# from league_multi_tool_llm_agent.models.rag_configs import EmbeddingSettings

# # Test Set

# TEST_QUERIES = [
#     "I like strong female leads and dark aesthetics, recommend a champion.",
#     "Easy beginner champions for new players.",
#     "I prefer supportive roles and teamwork, who should I main?",
#     "Aggressive junglers with high damage.",
#     "Cute skins for support champions.",
#     "Tanky champions that can engage fights.",
#     "High mobility assassins.",
#     "Champions similar to Ahri.",
#     "Dark themed skins with magical vibes.",
#     "Good champions for climbing ranked solo queue.",
#     "Beginner-friendly supports.",
#     "Champions with strong late game scaling.",
#     "Fun off-meta champions.",
#     "Champions that fit a strategic playstyle.",
#     "Skins with futuristic or sci-fi themes.",
# ]


# # Recommendation Generation
# async def generate_no_rag_answer(
#     *,
#     query: str,
#     agent: Agent[None, str],
# ) -> tuple[str, str]:
#     """Generate a recommendation without retrieval context."""
#     prompt = f"""
#     User request:
#     {query}

#     Recommend 2-3 League of Legends champions or skins that fit the request.
#     Explain why each recommendation fits.
#     """.strip()

#     result = await agent.run(prompt)
#     return result.output, ""


# async def generate_rag_answer(
#     *,
#     query: str,
#     agent: Agent[None, str],
#     rag_service: Any,
#     top_k: int,
# ) -> tuple[str, str]:
#     """Retrieve context and generate a RAG-grounded recommendation."""
#     # TODO: adjust doc_type logic later.
#     doc_type = "champion_skin" if "skin" in query.lower() else "champion_profile"

#     retrieved_docs = await rag_service.search(
#         query=query,
#         doc_type=doc_type,
#         limit=top_k,
#     )

#     retrieved_context = "\n\n".join(
#         f"[{i + 1}] {doc.content}" for i, doc in enumerate(retrieved_docs)
#     )

#     prompt = f"""
#     User request:
#     {query}

#     Retrieved context:
#     {retrieved_context}

#     Recommend 2-3 League of Legends champions or skins that fit the request.
#     Use the retrieved context when possible.
#     Explain why each recommendation fits.
#     """.strip()

#     result = await agent.run(prompt)
#     return result.output, retrieved_context


# # Judging
# async def judge_answer(
#     *,
#     query: str,
#     answer: str,
#     retrieved_context: str,
#     judge_agent: Agent[None, JudgeScore],
# ) -> JudgeScore:
#     """Score one generated answer with an LLM judge."""
#     prompt = f"""
#     User request:
#     {query}

#     Retrieved context:
#     {retrieved_context or "(no retrieved context)"}

#     System response:
#     {answer}

#     Evaluate the system response.
#     """.strip()

#     result = await judge_agent.run(prompt)
#     return result.output


# # Evaluation Runner
# async def run_eval_condition(
#     *,
#     queries: list[str],
#     condition: str,
#     model_name: str,
#     use_rag: bool,
#     rag_service: Any | None,
#     judge_agent: Agent[None, JudgeScore],
#     settings: EvalSettings,
#     ollama_provider_config: OllamaProviderConfig,
#     progress_bar: tqdm_asyncio,
# ) -> list[EvalCaseResult]:
#     """Run one ablation condition across all test queries."""
#     rec_agent = build_recommendation_agent(
#         model_name, ollama_provider_config=ollama_provider_config
#     )
#     results: list[EvalCaseResult] = []

#     for query in queries:
#         start = time.perf_counter()

#         if use_rag:
#             if rag_service is None:
#                 raise ValueError("rag_service is required when use_rag=True")

#             answer, context = await generate_rag_answer(
#                 query=query,
#                 agent=rec_agent,
#                 rag_service=rag_service,
#                 top_k=settings.EVAL_RAG_TOP_K,
#             )
#         else:
#             answer, context = await generate_no_rag_answer(
#                 query=query,
#                 agent=rec_agent,
#             )

#         latency = time.perf_counter() - start
#         progress_bar.set_postfix(
#             {"latency": f"{latency:.2f}s", "rel": results[-1].relevance}
#         )

#         try:
#             score = await judge_answer(
#                 query=query,
#                 answer=answer,
#                 retrieved_context=context,
#                 judge_agent=judge_agent,
#             )
#         except Exception as e:
#             score = None
#             judge_error = str(e)
#         else:
#             judge_error = ""

#         results.append(
#             EvalCaseResult(
#                 query=query,
#                 condition=condition,
#                 model_name=model_name,
#                 use_rag=use_rag,
#                 answer=answer,
#                 retrieved_context=context,
#                 latency_seconds=latency,
#                 relevance=score.relevance if score else None,
#                 explanation_quality=score.explanation_quality if score else None,
#                 personalization=score.personalization if score else None,
#                 groundedness=score.groundedness if score else None,
#                 judge_notes=score.notes if score else f"Judge failed: {judge_error}",
#             )
#         )

#         progress_bar.update(1)
#         print(
#             f"[{condition}] {query[:50]}... "
#             f"latency={latency:.2f}s "
#             f"rel={results[-1].relevance}"
#         )

#     return results


# async def run_full_evaluation(
#     *,
#     rag_service: Any,
#     settings: EvalSettings | None = None,
#     ollama_provider_config: OllamaProviderConfig,
# ) -> list[EvalCaseResult]:
#     """Run RAG/no-RAG and small/large model ablations."""
#     cfg = settings or EvalSettings()
#     queries = TEST_QUERIES[: cfg.EVAL_MAX_TEST_QUERIES]

#     judge_agent = build_judge_agent(
#         cfg.EVAL_JUDGE_MODEL, ollama_provider_config=ollama_provider_config
#     )

#     conditions = [
#         {
#             "condition": "small_no_rag",
#             "model_name": cfg.EVAL_SMALL_MODEL,
#             "use_rag": False,
#         },
#         {
#             "condition": "small_rag",
#             "model_name": cfg.EVAL_SMALL_MODEL,
#             "use_rag": True,
#         },
#         {
#             "condition": "large_no_rag",
#             "model_name": cfg.EVAL_LARGE_MODEL,
#             "use_rag": False,
#         },
#         {
#             "condition": "large_rag",
#             "model_name": cfg.EVAL_LARGE_MODEL,
#             "use_rag": True,
#         },
#     ]

#     all_results: list[EvalCaseResult] = []

#     total_steps = len(queries) * len(conditions)
#     pbar: tqdm_asyncio = tqdm(total=total_steps, desc="Full Evaluation")

#     for c in conditions:
#         condition_results = await run_eval_condition(
#             queries=queries,
#             condition=c["condition"],
#             model_name=c["model_name"],
#             use_rag=c["use_rag"],
#             rag_service=rag_service,
#             judge_agent=judge_agent,
#             settings=cfg,
#             ollama_provider_config=ollama_provider_config,
#             progress_bar=pbar,
#         )
#         all_results.extend(condition_results)

#     return all_results


# # setup to run on google colab (in properly set up colab env)
# async def main() -> None:
#     from league_multi_tool_llm_agent.db.rag_service import RagService, RagSettings

#     settings = EvalSettings()
#     local_llm_config = OllamaProviderConfig()
#     rag_settings = RagSettings(db_host="localhost")
#     embed_client_config = EmbeddingSettings(EMBEDDING_API_BASE="http://localhost:11434")
#     embed_client = EmbeddingClient(embed_client_config)

#     db_url = URL.create(
#         drivername="postgresql+psycopg2",
#         username=rag_settings.db_user,
#         password=rag_settings.db_password,
#         host=rag_settings.db_host,
#         port=rag_settings.db_port,
#         database=rag_settings.db_name,
#     )
#     rag_service = RagService(
#         db_url=db_url, settings=rag_settings, embedder=embed_client
#     )

#     results = await run_full_evaluation(
#         rag_service=rag_service,
#         settings=settings,
#         ollama_provider_config=local_llm_config,
#     )

#     save_eval_results(
#         results=results,
#         output_dir=settings.EVAL_OUTPUT_DIR,
#     )


# if __name__ == "__main__":
#     asyncio.run(main())
