CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id BIGSERIAL PRIMARY KEY,
    doc_type TEXT NOT NULL,
    champion_name TEXT,
    skin_name TEXT,
    main_role TEXT,
    difficulty INT,
    source_url TEXT,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS rag_documents_doc_type_idx
    ON rag_documents (doc_type);

CREATE INDEX IF NOT EXISTS rag_documents_champion_name_idx
    ON rag_documents (champion_name);

CREATE INDEX IF NOT EXISTS rag_documents_embedding_idx
    ON rag_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);