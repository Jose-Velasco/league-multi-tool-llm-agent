from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class RagDocument(SQLModel, table=True):
    __tablename__ = "rag_documents"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    doc_type: str = Field(index=True)
    champion_name: str | None = Field(default=None, index=True)
    skin_name: str | None = Field(default=None, index=True)
    main_role: str | None = Field(default=None, index=True)
    difficulty: int | None = Field(default=None)
    source_url: str | None = Field(default=None)
    content: str = Field(sa_column=Column(Text, nullable=False))
    meta_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )
    embedding: list[float] = Field(sa_column=Column(Vector(1024), nullable=False))

    __table_args__ = (
        Index(
            "ix_rag_documents_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


def create_vector_index(engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS ix_rag_documents_embedding
            ON rag_documents
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        )
        conn.execute(text("ANALYZE rag_documents;"))
