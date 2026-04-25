from sqlalchemy import text
from sqlmodel import Session


def search_similar(engine, query_embedding: list[float], limit: int = 5):
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with Session(engine) as session:
        result = session.connection().execute(
            text("""
                SELECT id, doc_type, champion_name, skin_name, content, meta_json,
                       embedding <=> CAST(:embedding AS vector) AS distance
                FROM rag_documents
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """),
            {
                "embedding": embedding_str,
                "limit": limit,
            },
        )
        return result.fetchall()


# def search_similar(engine, query_embedding: list[float], limit: int = 5):
#     with Session(engine) as session:
#         sql = text("""
#             SELECT id, doc_type, champion_name, skin_name, content, metadata,
#                    embedding <=> CAST(:embedding AS vector) AS distance
#             FROM rag_documents
#             ORDER BY embedding <=> CAST(:embedding AS vector)
#             LIMIT :limit
#         """)
#         result = session.exec(
#             sql,
#             params={
#                 "embedding": str(query_embedding),
#                 "limit": limit,
#             },
#         )
#         return result.all()
