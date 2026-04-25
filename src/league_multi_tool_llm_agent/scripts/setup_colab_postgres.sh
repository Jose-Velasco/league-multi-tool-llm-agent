# might not longer be needed
#!/usr/bin/env bash
set -e

# apt-get update -qq
# apt-get install -y postgresql postgresql-contrib

service postgresql start

sudo -u postgres psql -c "CREATE USER league WITH PASSWORD 'league';" || true
sudo -u postgres psql -c "CREATE DATABASE league OWNER league;" || true

# will need to install posgres vector in colab and set up db and user first before vector extension create
# sudo -u postgres psql -d league -c "CREATE EXTENSION IF NOT EXISTS vector;"