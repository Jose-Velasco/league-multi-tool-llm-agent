#!/usr/bin/env bash
set -e

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_HOST=127.0.0.1:11434

sudo apt-get update -qq

sudo apt-get install lshw

sudo apt-get install zstd

sudo apt install -y pciutils

curl -fsSL https://ollama.com/install.sh | sh

# sleep 8

# Start Ollama server in background
ollama serve > /tmp/ollama.log 2>&1 &

# Wait until server is ready
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama..."
  sleep 2
done

echo "Ollama is running."

# ollama serve &
# # ollama serve --think=false
# sleep 8

echo "Downloading Ollama qwen3.5:2b-q4_K_M..."
ollama pull qwen3.5:2b-q4_K_M

echo "Downloading gemma4:e4b-it-q4_K_M..."
ollama pull gemma4:e4b-it-q4_K_M

echo "Downloading qwen3-embedding:0.6b..."
ollama pull qwen3-embedding:0.6b

echo "Install Ollama script complete!!!!"
ollama list