from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import Agent
from sqlalchemy import URL

from league_multi_tool_llm_agent.evaluation.agents import (
    EvalCaseResult,
    JudgeScore,
    build_judge_agent,
    build_recommendation_agent,
)
from league_multi_tool_llm_agent.evaluation.configs import EvalSettings
from league_multi_tool_llm_agent.evaluation.utils import save_eval_results
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig

# Test Set

TEST_QUERIES = [
    "I like strong female leads and dark aesthetics, recommend a champion.",
    "Easy beginner champions for new players.",
    "I prefer supportive roles and teamwork, who should I main?",
    "Aggressive junglers with high damage.",
    "Cute skins for support champions.",
    "Tanky champions that can engage fights.",
    "High mobility assassins.",
    "Champions similar to Ahri.",
    "Dark themed skins with magical vibes.",
    "Good champions for climbing ranked solo queue.",
    "Beginner-friendly supports.",
    "Champions with strong late game scaling.",
    "Fun off-meta champions.",
    "Champions that fit a strategic playstyle.",
    "Skins with futuristic or sci-fi themes.",
]


# Recommendation Generation
async def generate_no_rag_answer(
    *,
    query: str,
    agent: Agent[None, str],
) -> tuple[str, str]:
    """Generate a recommendation without retrieval context."""
    prompt = f"""
    User request:
    {query}

    Recommend 2-3 League of Legends champions or skins that fit the request.
    Explain why each recommendation fits.
    """.strip()

    result = await agent.run(prompt)
    return result.output, ""


async def generate_rag_answer(
    *,
    query: str,
    agent: Agent[None, str],
    rag_service: Any,
    top_k: int,
) -> tuple[str, str]:
    """Retrieve context and generate a RAG-grounded recommendation."""
    # TODO: adjust doc_type logic later.
    doc_type = "champion_skin" if "skin" in query.lower() else "champion_profile"

    retrieved_docs = await rag_service.search(
        query=query,
        doc_type=doc_type,
        limit=top_k,
    )

    retrieved_context = "\n\n".join(
        f"[{i + 1}] {doc.content}" for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"""
    User request:
    {query}

    Retrieved context:
    {retrieved_context}

    Recommend 2-3 League of Legends champions or skins that fit the request.
    Use the retrieved context when possible.
    Explain why each recommendation fits.
    """.strip()

    result = await agent.run(prompt)
    return result.output, retrieved_context


# Judging
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
    """.strip()

    result = await judge_agent.run(prompt)
    return result.output


# Evaluation Runner
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
    rec_agent = build_recommendation_agent(
        model_name, ollama_provider_config=ollama_provider_config
    )
    results: list[EvalCaseResult] = []

    for query in queries:
        start = time.perf_counter()

        if use_rag:
            if rag_service is None:
                raise ValueError("rag_service is required when use_rag=True")

            answer, context = await generate_rag_answer(
                query=query,
                agent=rec_agent,
                rag_service=rag_service,
                top_k=settings.EVAL_RAG_TOP_K,
            )
        else:
            answer, context = await generate_no_rag_answer(
                query=query,
                agent=rec_agent,
            )

        latency = time.perf_counter() - start

        try:
            score = await judge_answer(
                query=query,
                answer=answer,
                retrieved_context=context,
                judge_agent=judge_agent,
            )
        except Exception as e:
            score = None
            judge_error = str(e)
        else:
            judge_error = ""

        results.append(
            EvalCaseResult(
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
        )

        print(
            f"[{condition}] {query[:50]}... "
            f"latency={latency:.2f}s "
            f"rel={results[-1].relevance}"
        )

    return results


async def run_full_evaluation(
    *,
    rag_service: Any,
    settings: EvalSettings | None = None,
    ollama_provider_config: OllamaProviderConfig,
) -> list[EvalCaseResult]:
    """Run RAG/no-RAG and small/large model ablations."""
    cfg = settings or EvalSettings()
    queries = TEST_QUERIES[: cfg.EVAL_MAX_TEST_QUERIES]

    judge_agent = build_judge_agent(
        cfg.EVAL_JUDGE_MODEL, ollama_provider_config=ollama_provider_config
    )

    conditions = [
        {
            "condition": "small_no_rag",
            "model_name": cfg.EVAL_SMALL_MODEL,
            "use_rag": False,
        },
        {
            "condition": "small_rag",
            "model_name": cfg.EVAL_SMALL_MODEL,
            "use_rag": True,
        },
        {
            "condition": "large_no_rag",
            "model_name": cfg.EVAL_LARGE_MODEL,
            "use_rag": False,
        },
        {
            "condition": "large_rag",
            "model_name": cfg.EVAL_LARGE_MODEL,
            "use_rag": True,
        },
    ]

    all_results: list[EvalCaseResult] = []

    for c in conditions:
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

    return all_results


# Example main
async def main() -> None:
    from league_multi_tool_llm_agent.db.rag_service import RagService, RagSettings

    settings = EvalSettings()
    local_llm_config = OllamaProviderConfig()
    rag_settings = RagSettings(
        EMBEDDING_API_BASE="http://localhost:11434", db_host="localhost"
    )

    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=rag_settings.db_user,
        password=rag_settings.db_password,
        host=rag_settings.db_host,
        port=rag_settings.db_port,
        database=rag_settings.db_name,
    )
    rag_service = RagService(db_url=db_url, settings=rag_settings)

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
