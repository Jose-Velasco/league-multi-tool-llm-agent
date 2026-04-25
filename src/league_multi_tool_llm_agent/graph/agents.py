from pydantic_ai import Agent

from league_multi_tool_llm_agent.graph.utils import build_ollama_agent_model
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.graph_models import ParsedIntent


def build_parser_agent(
    model_name: str = "gemma3:4b-it-qat",
    ollama_provider_config: OllamaProviderConfig | None = None,
) -> Agent[None, ParsedIntent]:
    model = build_ollama_agent_model(
        model_name=model_name, ollama_provider_config=ollama_provider_config
    )

    return Agent(
        model=model,
        output_type=ParsedIntent,
        instructions=(
            "You parse League of Legends assistant user requests into structured intent.\n\n"
            "Allowed intents:\n"
            "- champion_recommendation: user wants champion suggestions, mains, role picks, or personality/playstyle-based recommendations.\n"
            "- skin_search: user asks about skins, skin aesthetics, visual themes, or cosmetic recommendations.\n"
            "- unknown: request is not clearly about champion recommendation or skins.\n\n"
            "Extract preferences when present:\n"
            "- role_preference: top, jungle, mid, adc/bottom, support, teamwork, carry, etc.\n"
            "- aesthetic_preference: dark, cute, elegant, futuristic, spirit, monster, celestial, etc.\n"
            "- personality_preference: strong female lead, calm, aggressive, strategic, chaotic, protective, etc.\n"
            "- playstyle_preference: supportive, aggressive, beginner-friendly, mobile, tanky, ranged, burst, utility, etc.\n"
            "- difficulty_preference: easy, beginner, hard, mechanical, simple, etc.\n\n"
            "query_for_rag should be a concise search query combining the user's strongest preferences. "
            "Do not include irrelevant filler words."
        ),
    )
