#!/usr/bin/env bash
set -e

echo "Enabling pgvector extension"
psql \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Restoring RAG database dump"

pg_restore \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  /docker-entrypoint-initdb.d/rag_documents.dump

psql \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  -c "CREATE INDEX IF NOT EXISTS ix_rag_documents_embedding ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"

echo "Verifying restore"
psql \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  -c "SELECT COUNT(*) FROM rag_documents;"