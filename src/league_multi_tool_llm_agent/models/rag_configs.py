from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    # Vector store
    db_host: str = Field(
        default="postgres",
        description="Database connection host. Note in docker-compose its the name of the db service",
    )
    db_port: int = Field(
        default=5432,
        description="Database connection port",
    )
    db_name: str = Field(
        default="league",
        description="database name",
    )
    db_user: str = Field(
        default="league",
        description="database user",
    )
    db_password: str = Field(
        default="league",
        description="database password",
    )

    model_config = SettingsConfigDict(
        env_prefix="PGVECTOR_DEV_",
        env_file=".env.db",
        env_file_encoding="utf-8",
    )


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.embedding", env_file_encoding="utf-8"
    )

    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "ollama/qwen3-embedding:0.6b"
    EMBEDDING_API_BASE: str = "http://ollama:11434"


class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.visionllm", env_file_encoding="utf-8"
    )

    VISION_MODEL: str = "ollama_chat/gemma3:4b-it-qat"
    VISION_API_BASE: str = "http://ollama:11434"
    VISION_MAX_CONCURRENCY: int = 2
    VISION_MAX_TOKENS: int = 220
    VISION_TEMPERATURE: float = 0.2


class RagSettings(DatabaseConfig):
    model_config = SettingsConfigDict(env_file=".env.rag", env_file_encoding="utf-8")

    EMBEDDING_MODEL: str = "ollama/qwen3-embedding:0.6"
    EMBEDDING_API_BASE: str = "http://ollama:11434"
    RAG_TOP_K: int = 5
