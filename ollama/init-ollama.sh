#!/usr/bin/env bash
set -e

# start ollama server in background
ollama serve &

# wait until API is up
until curl -sSf http://localhost:11434/api/tags > /dev/null; do
  echo "Waiting for ollama..."
  sleep 2
done

# ollama pull gemma3:270m
# ollama pull gemma3:4b-it-qat
# ollama pull qwen3-embedding:0.6b
if ! ollama list | grep -q qwen3-embedding:0.6b; then
  ollama pull qwen3-embedding:0.6b
fi

if ! ollama list | grep -q gemma3:4b-it-qat; then
  ollama pull gemma3:4b-it-qat
fi

# keep server in foreground (wait for background ollama)
wait