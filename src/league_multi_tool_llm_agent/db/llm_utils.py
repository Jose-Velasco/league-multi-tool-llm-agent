from litellm import aembedding
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.rag_configs import EmbeddingSettings


class EmbeddingClient:
    def __init__(self, settings: EmbeddingSettings | None = None):
        self.settings = settings or EmbeddingSettings()

    async def embed(self, text: str) -> list[float]:
        response = await aembedding(
            model=self.settings.EMBEDDING_MODEL,
            input=[text],
            api_base=self.settings.EMBEDDING_API_BASE,
        )
        return response["data"][0]["embedding"]


class SkinDescriptionOutput(BaseModel):
    description: str = Field(
        description="Detailed visual description of the skin for semantic retrieval"
    )


def build_skin_description_agent(
    model_name: str, ollama_config: OllamaProviderConfig | None
):
    if ollama_config:
        model = OllamaModel(
            model_name,
            provider=OllamaProvider(
                base_url=ollama_config.OLLAMA_BASE_URL,
                api_key=ollama_config.OLLAMA_API_KEY,
            ),
        )
    else:
        model = OllamaModel(model_name)

    return Agent(
        model=model,
        output_type=SkinDescriptionOutput,
        instructions=(
            "You MUST return valid JSON with this exact schema:\n"
            '{ "description": string }\n\n'
            "Do not return plain text. Do not add extra fields.\n"
            "You generate detailed visual descriptions of League of Legends skins "
            "for semantic search and similarity retrieval.\n\n"
            "Focus on:\n"
            "- colors\n"
            "- outfit / armor\n"
            "- theme (fantasy, cyber, spirit, dark, cute, elegant, etc.)\n"
            "- visual motifs (magic, animals, celestial, tech, etc.)\n"
            "- overall aesthetic and mood\n\n"
            "Rules:\n"
            "- Do NOT say 'this image shows'\n"
            "- Do NOT mention UI, camera, or framing\n"
            "- Do NOT hallucinate lore\n"
            "- Keep it 80-150 words\n"
            "- Make it useful for retrieval similarity"
        ),
    )
