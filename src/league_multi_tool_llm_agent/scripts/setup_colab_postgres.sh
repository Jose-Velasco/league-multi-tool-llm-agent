# might not longer be needed
#!/usr/bin/env bash
set -e

DATABASE_BACKUP=$1

# Install PostgreSQL + pgvector
apt-get update -qq
# apt-get install -y postgresql postgresql-contrib
apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
apt-get install -y git build-essential postgresql-server-dev-14

cd /content
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd /content/pgvector
make
make install

# Start PostgreSQL and create user, password, and database
service postgresql start

sudo -u postgres psql -c "CREATE USER league WITH PASSWORD 'league';" || true
sudo -u postgres psql -c "CREATE DATABASE league OWNER league;" || true

# will need to install posgres vector in colab and set up db and user first before vector extension create
sudo -u postgres psql -d league -c "CREATE EXTENSION IF NOT EXISTS vector;"

# restore from backup with already generated embeddings
pg_restore -d "postgresql://league:league@localhost:5432/league"  --clean --if-exists "$DATABASE_BACKUP"

# verify db restore
psql "postgresql://league:league@localhost:5432/league" -c "SELECT COUNT(*) FROM rag_documents;"

# I forgot to add pgvector index before export so adding it now
psql "postgresql://league:league@localhost:5432/league" -c " CREATE INDEX IF NOT EXISTS ix_rag_documents_embedding ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"

echo "Install postgres & pgvetor script complete"