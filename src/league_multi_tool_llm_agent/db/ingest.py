import argparse
import asyncio
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlmodel import Session, SQLModel, create_engine
from tqdm import tqdm

from league_multi_tool_llm_agent.db.llm_utils import (
    EmbeddingClient,
)
from league_multi_tool_llm_agent.db.rag_db_model import RagDocument, create_vector_index
from league_multi_tool_llm_agent.db.utils import (
    build_rag_docs,
    # enrich_all_skin_descriptions,
    load_jsonl,
)
from league_multi_tool_llm_agent.models.agent_config import OllamaProviderConfig
from league_multi_tool_llm_agent.models.rag_configs import (
    DatabaseConfig,
    VisionSettings,
)


def create_db(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    create_vector_index(engine)
    SQLModel.metadata.create_all(engine)


def parse_args():
    parser = argparse.ArgumentParser(description="Configure data paths")

    parser.add_argument(
        "--root-data-dir",
        type=Path,
        default=Path("../../../data/"),
        help="Root data directory",
    )

    parser.add_argument(
        "--skin-imgs-dir",
        type=Path,
        default=None,
        help="Directory for skin images (defaults to <root-data-dir>/images)",
    )

    parser.add_argument(
        "--local-champion-data",
        type=Path,
        default=None,
        help="Path to local champion JSONL file",
    )

    return parser.parse_args()


async def ingest_jsonl_to_pgvector(
    jsonl_path: str,
    database_config: DatabaseConfig,
    ollama_config: OllamaProviderConfig,
    image_analyzer_model_name: str,
    skin_imgs_dir: Path,
    vision_settings: VisionSettings,
) -> None:
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=database_config.db_user,
        password=database_config.db_password,
        host=database_config.db_host,
        port=database_config.db_port,
        database=database_config.db_name,
    )

    engine = create_engine(db_url)
    create_db(engine)

    champions = load_jsonl(jsonl_path)

    # skin_description_agent = build_skin_description_agent(
    #     image_analyzer_model_name, ollama_config
    # )

    # await enrich_all_skin_descriptions(
    #     champions=champions,
    #     vision_settings=vision_settings,
    #     root_skin_imgs_dir=skin_imgs_dir,
    # )

    rag_docs = await build_rag_docs(
        champions,
        skin_imgs_dir=skin_imgs_dir,
        # champions, skin_description_agent, skin_imgs_dir=skin_imgs_dir
    )

    # embedder = OllamaEmbeddingClient(model="qwen3-embedding:0.6b")
    embedder = EmbeddingClient()

    rows: list[RagDocument] = []
    print("Generating Embeddings")
    with tqdm(total=len(rag_docs), desc="Generating Embeddings") as pbar:
        for doc in rag_docs:
            pbar.set_postfix({"champion": doc["champion_name"]})

            embedding = await embedder.embed(doc["content"])
            rows.append(
                RagDocument(
                    doc_type=doc["doc_type"],
                    champion_name=doc["champion_name"],
                    skin_name=doc["skin_name"],
                    main_role=doc["main_role"],
                    difficulty=doc["difficulty"],
                    source_url=doc["source_url"],
                    content=doc["content"],
                    meta_json=doc["meta_json"],
                    embedding=embedding,
                )
            )
            pbar.update(1)

    with Session(engine) as session:
        session.add_all(rows)
        session.commit()


if __name__ == "__main__":
    args = parse_args()

    ROOT_DATA_DIR = args.root_data_dir

    skin_imgs_dir = (
        args.skin_imgs_dir
        if args.skin_imgs_dir is not None
        else ROOT_DATA_DIR / "images"
    )

    local_champion_data = (
        args.local_champion_data
        if args.local_champion_data is not None
        else ROOT_DATA_DIR / "enriched_champion_2026-04-14T03-12-56+00-00.jsonl"
        # else ROOT_DATA_DIR / "champion_2026-04-14T03-12-56+00-00.jsonl"
    )

    print("ROOT_DATA_DIR:", ROOT_DATA_DIR)
    print("skin_imgs_dir:", skin_imgs_dir)
    print("local_champion_data:", local_champion_data)

    # ROOT_DATA_DIR = Path("../../../data/")
    # skin_imgs_dir = Path(ROOT_DATA_DIR / "images")
    # local_champion_data = ROOT_DATA_DIR / "champion_2026-04-14T03-12-56+00-00.jsonl"

    database_config = DatabaseConfig()
    ollama_config = OllamaProviderConfig(OLLAMA_BASE_URL="http://ollama:11434/v1/")
    image_analyzer_model_name = "gemma3:4b-it-qat"

    vision_settings = VisionSettings()

    asyncio.run(
        ingest_jsonl_to_pgvector(
            str(local_champion_data),
            database_config,
            ollama_config,
            image_analyzer_model_name,
            skin_imgs_dir=skin_imgs_dir,
            vision_settings=vision_settings,
        )
    )
