from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """Runtime settings for recommendation evaluation."""

    model_config = SettingsConfigDict(env_file=".env.eval", env_file_encoding="utf-8")

    EVAL_OUTPUT_DIR: str = "data/eval_results"

    # Models for ablation
    # EVAL_SMALL_MODEL: str = "qwen3-vl:2b"
    # EVAL_LARGE_MODEL: str = "gemma4:e4b-it-q4_K_M"
    # EVAL_JUDGE_MODEL: str = "gemma4:e4b-it-q4_K_M"
    # EVAL_SMALL_MODEL: str = "gemma4:e4b"
    # EVAL_LARGE_MODEL: str = "qwen3.5:9b"
    # EVAL_JUDGE_MODEL: str = "qwen3.5:9b"
    # "qwen3:8b-q4_K_M"
    # EVAL_LARGE_MODEL: str = "qwen3:8b-q4_K_M"
    # EVAL_JUDGE_MODEL: str = "qwen3:8b-q4_K_M"

    # EVAL_SMALL_MODEL: str = "qwen3.5:2b"
    EVAL_SMALL_MODEL: str = "qwen3.5:2b-q4_K_M"
    EVAL_LARGE_MODEL: str = "gemma4:e4b-it-q4_K_M"
    EVAL_JUDGE_MODEL: str = "gemma4:e4b-it-q4_K_M"

    # Retrieval
    EVAL_RAG_TOP_K: int = 5

    # Generation
    EVAL_TEMPERATURE: float = 0.2
    EVAL_MAX_TEST_QUERIES: int = 20
    EVAL_MAX_CONCURRENCY: int = 3
