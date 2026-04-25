from __future__ import annotations

from dataclasses import dataclass

from litellm import aembedding
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlmodel import Session, create_engine

from league_multi_tool_llm_agent.models.rag_configs import RagSettings
from league_multi_tool_llm_agent.models.rag_models import RagSearchResult


@dataclass
class RagService:
    db_url: str | URL
    settings: RagSettings

    def __post_init__(self) -> None:
        self.engine = create_engine(self.db_url, echo=False)

    async def embed_query(self, query: str) -> list[float]:
        response = await aembedding(
            model=self.settings.EMBEDDING_MODEL,
            input=[query],
            api_base=self.settings.EMBEDDING_API_BASE,
        )
        return response["data"][0]["embedding"]

    async def search(
        self,
        *,
        query: str,
        doc_type: str | None = None,
        limit: int | None = None,
    ) -> list[RagSearchResult]:
        query_embedding = await self.embed_query(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        top_k = limit or self.settings.RAG_TOP_K

        if doc_type:
            sql = text("""
                SELECT
                    id,
                    doc_type,
                    champion_name,
                    skin_name,
                    main_role,
                    difficulty,
                    source_url,
                    content,
                    meta_json,
                    embedding <=> CAST(:embedding AS vector) AS distance
                FROM rag_documents
                WHERE doc_type = :doc_type
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """)
            params = {
                "embedding": embedding_str,
                "doc_type": doc_type,
                "limit": top_k,
            }
        else:
            sql = text("""
                SELECT
                    id,
                    doc_type,
                    champion_name,
                    skin_name,
                    main_role,
                    difficulty,
                    source_url,
                    content,
                    meta_json,
                    embedding <=> CAST(:embedding AS vector) AS distance
                FROM rag_documents
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """)
            params = {
                "embedding": embedding_str,
                "limit": top_k,
            }

        with Session(self.engine) as session:
            rows = session.connection().execute(sql, params).mappings().all()

        return [RagSearchResult.model_validate(dict(row)) for row in rows]
