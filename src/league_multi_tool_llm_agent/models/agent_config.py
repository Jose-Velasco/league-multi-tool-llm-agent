from pydantic_settings import BaseSettings


class OllamaProviderConfig(BaseSettings):
    """Configuration for OllamaProvider using paydanticAI."""

    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_API_KEY: str = "your-api-key"
